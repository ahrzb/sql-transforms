//! Parser for the IR text format — the inverse of [`print`]. Hand-rolled
//! lexer + recursive descent; no dependencies. The parser owns *syntax* only:
//! it interns value names to dense, definition-ordered ids (the canonical
//! form) and rejects malformed text, redefinition of a value name, use of an
//! unknown value/label/column, and unknown opcodes. Everything semantic
//! (types, SSA scoping, CFG shape, store completeness) is the verifier's job.
//!
//! [`print`]: super::print

use std::collections::HashMap;

use super::{
    Block, BlockId, BinOp, Col, ColTy, CmpPred, Inst, Lit, Program, RoundMode, StaticTy, Term,
    Ty, Value,
};

#[derive(Debug)]
pub struct ParseError {
    pub line: u32,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

pub fn parse(text: &str) -> Result<Program, ParseError> {
    let tokens = lex(text)?;
    Parser {
        toks: tokens,
        pos: 0,
        values: HashMap::new(),
        next_value: 0,
    }
    .program()
}

// ---------------------------------------------------------------- lexer --

#[derive(Clone, PartialEq, Debug)]
enum Tok {
    Ident(String),
    Val(String),    // %name
    Num(String),    // integer or float text, optional leading '-'
    Str(String),    // unescaped content
    At,             // @
    Comma,
    Colon,
    Dot,
    Eq,
    Question,
    Arrow,          // ->
    LParen,
    RParen,
    LBrace,
    RBrace,
    Lt,
    Gt,
    Eof,
}

impl Tok {
    fn show(&self) -> String {
        match self {
            Tok::Ident(s) => format!("'{s}'"),
            Tok::Val(s) => format!("'%{s}'"),
            Tok::Num(s) => format!("number '{s}'"),
            Tok::Str(_) => "string literal".to_string(),
            Tok::At => "'@'".to_string(),
            Tok::Comma => "','".to_string(),
            Tok::Colon => "':'".to_string(),
            Tok::Dot => "'.'".to_string(),
            Tok::Eq => "'='".to_string(),
            Tok::Question => "'?'".to_string(),
            Tok::Arrow => "'->'".to_string(),
            Tok::LParen => "'('".to_string(),
            Tok::RParen => "')'".to_string(),
            Tok::LBrace => "'{'".to_string(),
            Tok::RBrace => "'}'".to_string(),
            Tok::Lt => "'<'".to_string(),
            Tok::Gt => "'>'".to_string(),
            Tok::Eof => "end of input".to_string(),
        }
    }
}

fn lex(text: &str) -> Result<Vec<(Tok, u32)>, ParseError> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    let mut line: u32 = 1;
    while let Some(&c) = chars.peek() {
        match c {
            '\n' => {
                line += 1;
                chars.next();
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            '#' => {
                // comment to end of line
                for c in chars.by_ref() {
                    if c == '\n' {
                        line += 1;
                        break;
                    }
                }
            }
            '@' => {
                chars.next();
                out.push((Tok::At, line));
            }
            ',' => {
                chars.next();
                out.push((Tok::Comma, line));
            }
            ':' => {
                chars.next();
                out.push((Tok::Colon, line));
            }
            '.' => {
                chars.next();
                out.push((Tok::Dot, line));
            }
            '=' => {
                chars.next();
                out.push((Tok::Eq, line));
            }
            '?' => {
                chars.next();
                out.push((Tok::Question, line));
            }
            '(' => {
                chars.next();
                out.push((Tok::LParen, line));
            }
            ')' => {
                chars.next();
                out.push((Tok::RParen, line));
            }
            '{' => {
                chars.next();
                out.push((Tok::LBrace, line));
            }
            '}' => {
                chars.next();
                out.push((Tok::RBrace, line));
            }
            '<' => {
                chars.next();
                out.push((Tok::Lt, line));
            }
            '>' => {
                chars.next();
                out.push((Tok::Gt, line));
            }
            '%' => {
                chars.next();
                let name = take_ident(&mut chars);
                if name.is_empty() {
                    return Err(ParseError {
                        line,
                        msg: "expected a value name after '%'".to_string(),
                    });
                }
                out.push((Tok::Val(name), line));
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                loop {
                    match chars.next() {
                        None => {
                            return Err(ParseError {
                                line,
                                msg: "unterminated string literal".to_string(),
                            })
                        }
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some('"') => s.push('"'),
                            Some('\\') => s.push('\\'),
                            Some('n') => s.push('\n'),
                            Some('r') => s.push('\r'),
                            Some('t') => s.push('\t'),
                            Some('u') => {
                                if chars.next() != Some('{') {
                                    return Err(ParseError {
                                        line,
                                        msg: "expected '{' after \\u".to_string(),
                                    });
                                }
                                let mut hex = String::new();
                                loop {
                                    match chars.next() {
                                        Some('}') => break,
                                        Some(h) if h.is_ascii_hexdigit() => hex.push(h),
                                        _ => {
                                            return Err(ParseError {
                                                line,
                                                msg: "bad \\u{...} escape".to_string(),
                                            })
                                        }
                                    }
                                }
                                let cp = u32::from_str_radix(&hex, 16).ok();
                                match cp.and_then(char::from_u32) {
                                    Some(c) => s.push(c),
                                    None => {
                                        return Err(ParseError {
                                            line,
                                            msg: format!("bad codepoint in \\u{{{hex}}}"),
                                        })
                                    }
                                }
                            }
                            other => {
                                return Err(ParseError {
                                    line,
                                    msg: format!("unknown escape: \\{}", show_esc(other)),
                                })
                            }
                        },
                        Some('\n') => {
                            return Err(ParseError {
                                line,
                                msg: "newline in string literal (use \\n)".to_string(),
                            })
                        }
                        Some(c) => s.push(c),
                    }
                }
                out.push((Tok::Str(s), line));
            }
            '-' => {
                chars.next();
                match chars.peek() {
                    Some('>') => {
                        chars.next();
                        out.push((Tok::Arrow, line));
                    }
                    Some(c) if c.is_ascii_digit() => {
                        let num = take_number(&mut chars);
                        out.push((Tok::Num(format!("-{num}")), line));
                    }
                    Some('i') => {
                        let word = take_ident(&mut chars);
                        if word == "inf" {
                            out.push((Tok::Num("-inf".to_string()), line));
                        } else {
                            return Err(ParseError {
                                line,
                                msg: format!("unexpected '-{word}'"),
                            });
                        }
                    }
                    _ => {
                        return Err(ParseError {
                            line,
                            msg: "unexpected '-'".to_string(),
                        })
                    }
                }
            }
            c if c.is_ascii_digit() => {
                let num = take_number(&mut chars);
                out.push((Tok::Num(num), line));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let word = take_ident(&mut chars);
                out.push((Tok::Ident(word), line));
            }
            other => {
                return Err(ParseError {
                    line,
                    msg: format!("unexpected character '{other}'"),
                })
            }
        }
    }
    out.push((Tok::Eof, line));
    Ok(out)
}

fn show_esc(c: Option<char>) -> String {
    match c {
        Some(c) => c.to_string(),
        None => "<eof>".to_string(),
    }
}

fn take_ident(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c);
            chars.next();
        } else {
            break;
        }
    }
    s
}

/// Number text: digits, optional fraction, optional exponent. The '.' is
/// consumed only when followed by a digit, so `1.` never eats a field access
/// dot (which cannot follow a number anyway in this grammar).
fn take_number(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            s.push(c);
            chars.next();
        } else {
            break;
        }
    }
    if let Some('.') = chars.peek() {
        let mut ahead = chars.clone();
        ahead.next();
        if matches!(ahead.peek(), Some(d) if d.is_ascii_digit()) {
            s.push('.');
            chars.next();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    s.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
        }
    }
    if matches!(chars.peek(), Some('e') | Some('E')) {
        let mut ahead = chars.clone();
        ahead.next();
        let sign = matches!(ahead.peek(), Some('+') | Some('-'));
        if sign {
            ahead.next();
        }
        if matches!(ahead.peek(), Some(d) if d.is_ascii_digit()) {
            s.push(chars.next().unwrap());
            if sign {
                s.push(chars.next().unwrap());
            }
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    s.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
        }
    }
    s
}

// --------------------------------------------------------------- parser --

struct Parser {
    toks: Vec<(Tok, u32)>,
    pos: usize,
    values: HashMap<String, Value>,
    next_value: u32,
}

/// A block body before label resolution: terminator targets are still names.
struct RawBlock {
    label: String,
    params: Vec<(Value, Ty)>,
    insts: Vec<Inst>,
    term: RawTerm,
    line: u32,
}

enum RawTerm {
    Jump(String, Vec<Value>),
    Brif(Value, (String, Vec<Value>), (String, Vec<Value>)),
    Emit,
    Skip,
    Trap(String),
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].0
    }

    fn line(&self) -> u32 {
        self.toks[self.pos].1
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].0.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            line: self.line(),
            msg: msg.into(),
        }
    }

    fn expect(&mut self, tok: Tok) -> Result<(), ParseError> {
        if *self.peek() == tok {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected {}, found {}", tok.show(), self.peek().show())))
        }
    }

    fn ident(&mut self, what: &str) -> Result<String, ParseError> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(self.err(format!("expected {what}, found {}", other.show()))),
        }
    }

    /// Keyword = identifier with a required spelling.
    fn keyword(&mut self, kw: &str) -> Result<(), ParseError> {
        match self.peek() {
            Tok::Ident(s) if s == kw => {
                self.bump();
                Ok(())
            }
            other => Err(self.err(format!("expected '{kw}', found {}", other.show()))),
        }
    }

    fn define_value(&mut self, name: String) -> Result<Value, ParseError> {
        if self.values.contains_key(&name) {
            return Err(self.err(format!("value '%{name}' defined twice")));
        }
        let v = Value(self.next_value);
        self.next_value += 1;
        self.values.insert(name, v);
        Ok(v)
    }

    fn use_value(&mut self) -> Result<Value, ParseError> {
        match self.bump() {
            Tok::Val(name) => self
                .values
                .get(&name)
                .copied()
                .ok_or_else(|| self.err(format!("use of undefined value '%{name}'"))),
            other => Err(self.err(format!("expected a %value, found {}", other.show()))),
        }
    }

    fn program(mut self) -> Result<Program, ParseError> {
        let mut statics = Vec::new();
        while matches!(self.peek(), Tok::Ident(s) if s == "static") {
            self.bump();
            self.expect(Tok::At)?;
            let idx = self.int_literal("static index")?;
            if idx != statics.len() as i64 {
                return Err(self.err(format!(
                    "static ids must be dense and in order: expected @{}, found @{idx}",
                    statics.len()
                )));
            }
            self.expect(Tok::Colon)?;
            statics.push(self.static_ty()?);
        }

        self.keyword("fn")?;
        let name = self.ident("function name")?;
        self.expect(Tok::LParen)?;
        self.keyword("in")?;
        self.expect(Tok::Colon)?;
        let in_cols = self.batch()?;
        self.expect(Tok::Comma)?;
        self.keyword("out")?;
        self.expect(Tok::Colon)?;
        let out_cols = self.batch()?;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;

        let mut raw_blocks: Vec<RawBlock> = Vec::new();
        while *self.peek() != Tok::RBrace {
            let block = self.block(&statics, &in_cols, &out_cols)?;
            raw_blocks.push(block);
        }
        self.expect(Tok::RBrace)?;
        if *self.peek() != Tok::Eof {
            return Err(self.err(format!("trailing input: {}", self.peek().show())));
        }
        if raw_blocks.is_empty() {
            return Err(self.err("a function needs at least one block"));
        }

        // Resolve labels.
        let mut label_ids: HashMap<String, BlockId> = HashMap::new();
        for (i, rb) in raw_blocks.iter().enumerate() {
            if label_ids.insert(rb.label.clone(), BlockId(i as u32)).is_some() {
                return Err(ParseError {
                    line: rb.line,
                    msg: format!("block label '{}' defined twice", rb.label),
                });
            }
        }
        let resolve = |label: &str, line: u32| -> Result<BlockId, ParseError> {
            label_ids.get(label).copied().ok_or(ParseError {
                line,
                msg: format!("jump to unknown block '{label}'"),
            })
        };
        let mut blocks = Vec::with_capacity(raw_blocks.len());
        for rb in raw_blocks {
            let term = match rb.term {
                RawTerm::Jump(label, args) => Term::Jump {
                    to: resolve(&label, rb.line)?,
                    args,
                },
                RawTerm::Brif(cond, (tl, ta), (el, ea)) => Term::Brif {
                    cond,
                    then_to: resolve(&tl, rb.line)?,
                    then_args: ta,
                    else_to: resolve(&el, rb.line)?,
                    else_args: ea,
                },
                RawTerm::Emit => Term::Emit,
                RawTerm::Skip => Term::Skip,
                RawTerm::Trap(msg) => Term::Trap { msg },
            };
            blocks.push(Block {
                params: rb.params,
                insts: rb.insts,
                term,
            });
        }

        Ok(Program {
            statics,
            name,
            in_cols,
            out_cols,
            blocks,
        })
    }

    fn int_literal(&mut self, what: &str) -> Result<i64, ParseError> {
        match self.bump() {
            Tok::Num(s) => s
                .parse::<i64>()
                .map_err(|_| self.err(format!("bad {what}: '{s}'"))),
            other => Err(self.err(format!("expected {what}, found {}", other.show()))),
        }
    }

    fn static_ty(&mut self) -> Result<StaticTy, ParseError> {
        let kind = self.ident("'scalar' or 'map'")?;
        match kind.as_str() {
            "scalar" => {
                self.expect(Tok::Lt)?;
                let ct = self.col_ty()?;
                self.expect(Tok::Gt)?;
                Ok(StaticTy::Scalar(ct))
            }
            "map" => {
                self.expect(Tok::LParen)?;
                let keys = self.ty_list()?;
                self.expect(Tok::RParen)?;
                self.expect(Tok::Arrow)?;
                self.expect(Tok::LParen)?;
                let values = self.ty_list()?;
                self.expect(Tok::RParen)?;
                Ok(StaticTy::Map { keys, values })
            }
            other => Err(self.err(format!("expected 'scalar' or 'map', found '{other}'"))),
        }
    }

    fn ty_list(&mut self) -> Result<Vec<Ty>, ParseError> {
        let mut list = vec![self.ty()?];
        while *self.peek() == Tok::Comma {
            self.bump();
            list.push(self.ty()?);
        }
        Ok(list)
    }

    fn ty(&mut self) -> Result<Ty, ParseError> {
        let name = self.ident("a type")?;
        match name.as_str() {
            "i1" => Ok(Ty::I1),
            "i64" => Ok(Ty::I64),
            "f64" => Ok(Ty::F64),
            "str" => Ok(Ty::Str),
            other => Err(self.err(format!("unknown type '{other}'"))),
        }
    }

    fn col_ty(&mut self) -> Result<ColTy, ParseError> {
        let ty = self.ty()?;
        let nullable = if *self.peek() == Tok::Question {
            self.bump();
            true
        } else {
            false
        };
        Ok(ColTy { ty, nullable })
    }

    fn batch(&mut self) -> Result<Vec<Col>, ParseError> {
        self.keyword("batch")?;
        self.expect(Tok::LBrace)?;
        let mut cols = Vec::new();
        if *self.peek() != Tok::RBrace {
            loop {
                let name = match self.bump() {
                    Tok::Ident(s) => s,
                    Tok::Str(s) => s,
                    other => {
                        return Err(
                            self.err(format!("expected a column name, found {}", other.show()))
                        )
                    }
                };
                if cols.iter().any(|c: &Col| c.name == name) {
                    return Err(self.err(format!("column '{name}' declared twice")));
                }
                self.expect(Tok::Colon)?;
                let ty = self.col_ty()?;
                cols.push(Col { name, ty });
                if *self.peek() == Tok::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(cols)
    }

    fn block(
        &mut self,
        statics: &[StaticTy],
        in_cols: &[Col],
        out_cols: &[Col],
    ) -> Result<RawBlock, ParseError> {
        let line = self.line();
        let label = self.ident("a block label")?;
        let mut params = Vec::new();
        if *self.peek() == Tok::LParen {
            self.bump();
            loop {
                let name = match self.bump() {
                    Tok::Val(n) => n,
                    other => {
                        return Err(
                            self.err(format!("expected a %param, found {}", other.show()))
                        )
                    }
                };
                self.expect(Tok::Colon)?;
                let ty = self.ty()?;
                let v = self.define_value(name)?;
                params.push((v, ty));
                if *self.peek() == Tok::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
        }
        self.expect(Tok::Colon)?;

        let mut insts = Vec::new();
        let term = loop {
            if let Some(term) = self.try_terminator()? {
                break term;
            }
            insts.push(self.inst(statics, in_cols, out_cols)?);
        };
        Ok(RawBlock {
            label,
            params,
            insts,
            term,
            line,
        })
    }

    fn try_terminator(&mut self) -> Result<Option<RawTerm>, ParseError> {
        let kw = match self.peek() {
            Tok::Ident(s) => s.clone(),
            _ => return Ok(None),
        };
        match kw.as_str() {
            "emit" => {
                self.bump();
                Ok(Some(RawTerm::Emit))
            }
            "skip" => {
                self.bump();
                Ok(Some(RawTerm::Skip))
            }
            "trap" => {
                self.bump();
                match self.bump() {
                    Tok::Str(msg) => Ok(Some(RawTerm::Trap(msg))),
                    other => Err(self.err(format!(
                        "expected a string message after 'trap', found {}",
                        other.show()
                    ))),
                }
            }
            "jump" => {
                self.bump();
                let (label, args) = self.target()?;
                Ok(Some(RawTerm::Jump(label, args)))
            }
            "brif" => {
                self.bump();
                let cond = self.use_value()?;
                self.expect(Tok::Comma)?;
                let then_t = self.target()?;
                self.expect(Tok::Comma)?;
                let else_t = self.target()?;
                Ok(Some(RawTerm::Brif(cond, then_t, else_t)))
            }
            _ => Ok(None),
        }
    }

    fn target(&mut self) -> Result<(String, Vec<Value>), ParseError> {
        let label = self.ident("a block label")?;
        let mut args = Vec::new();
        if *self.peek() == Tok::LParen {
            self.bump();
            loop {
                args.push(self.use_value()?);
                if *self.peek() == Tok::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
        }
        Ok((label, args))
    }

    /// One instruction line: `dsts = opcode operands`.
    fn inst(
        &mut self,
        statics: &[StaticTy],
        in_cols: &[Col],
        out_cols: &[Col],
    ) -> Result<Inst, ParseError> {
        // store/store.opt have no dsts and start with an Ident.
        if matches!(self.peek(), Tok::Ident(s) if s == "store") {
            self.bump();
            let opt = self.dot_suffix()?;
            let col = self.col_ref("out", out_cols)?;
            self.expect(Tok::Comma)?;
            return match opt.as_deref() {
                None => {
                    let val = self.use_value()?;
                    Ok(Inst::Store { col, val })
                }
                Some("opt") => {
                    let flag = self.use_value()?;
                    self.expect(Tok::Comma)?;
                    let val = self.use_value()?;
                    Ok(Inst::StoreOpt { col, flag, val })
                }
                Some(other) => Err(self.err(format!("unknown opcode 'store.{other}'"))),
            };
        }

        // Everything else: one or more dsts, '=', opcode.
        let mut dst_names = Vec::new();
        loop {
            match self.bump() {
                Tok::Val(n) => dst_names.push(n),
                other => {
                    return Err(self.err(format!(
                        "expected an instruction or terminator, found {}",
                        other.show()
                    )))
                }
            }
            if *self.peek() == Tok::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(Tok::Eq)?;
        let head = self.ident("an opcode")?;
        let suffix = self.dot_suffix()?;
        let opcode = match &suffix {
            Some(sfx) => format!("{head}.{sfx}"),
            None => head.clone(),
        };

        let want_dsts = |n: usize, this: &Parser| -> Result<(), ParseError> {
            if dst_names.len() != n {
                Err(this.err(format!(
                    "'{opcode}' defines {n} value(s), found {}",
                    dst_names.len()
                )))
            } else {
                Ok(())
            }
        };

        // Helper closures cannot borrow self mutably twice; do defs inline.
        macro_rules! def {
            ($i:expr) => {
                self.define_value(dst_names[$i].clone())?
            };
        }

        let inst = match opcode.as_str() {
            "const.i1" | "const.i64" | "const.f64" | "const.str" => {
                want_dsts(1, self)?;
                let lit = match opcode.as_str() {
                    "const.i1" => match self.ident("'true' or 'false'")?.as_str() {
                        "true" => Lit::I1(true),
                        "false" => Lit::I1(false),
                        other => {
                            return Err(
                                self.err(format!("expected 'true' or 'false', found '{other}'"))
                            )
                        }
                    },
                    "const.i64" => Lit::I64(self.int_literal("integer literal")?),
                    "const.f64" => Lit::F64(self.f64_literal()?),
                    _ => match self.bump() {
                        Tok::Str(s) => Lit::Str(s),
                        other => {
                            return Err(
                                self.err(format!("expected a string, found {}", other.show()))
                            )
                        }
                    },
                };
                Inst::Const { dst: def!(0), lit }
            }
            "iadd" | "isub" | "imul" | "idiv" | "irem" | "fadd" | "fsub" | "fmul" | "fdiv"
            | "and" | "or" | "xor" => {
                want_dsts(1, self)?;
                let op = match opcode.as_str() {
                    "iadd" => BinOp::Iadd,
                    "isub" => BinOp::Isub,
                    "imul" => BinOp::Imul,
                    "idiv" => BinOp::Idiv,
                    "irem" => BinOp::Irem,
                    "fadd" => BinOp::Fadd,
                    "fsub" => BinOp::Fsub,
                    "fmul" => BinOp::Fmul,
                    "fdiv" => BinOp::Fdiv,
                    "and" => BinOp::And,
                    "or" => BinOp::Or,
                    _ => BinOp::Xor,
                };
                let a = self.use_value()?;
                self.expect(Tok::Comma)?;
                let b = self.use_value()?;
                Inst::Bin { op, dst: def!(0), a, b }
            }
            _ if head == "icmp" || head == "fcmp" || head == "scmp" => {
                want_dsts(1, self)?;
                let ty = match head.as_str() {
                    "icmp" => Ty::I64,
                    "fcmp" => Ty::F64,
                    _ => Ty::Str,
                };
                let pred = match suffix.as_deref() {
                    Some("eq") => CmpPred::Eq,
                    Some("ne") => CmpPred::Ne,
                    Some("lt") => CmpPred::Lt,
                    Some("le") => CmpPred::Le,
                    Some("gt") => CmpPred::Gt,
                    Some("ge") => CmpPred::Ge,
                    _ => return Err(self.err(format!("unknown opcode '{opcode}'"))),
                };
                let a = self.use_value()?;
                self.expect(Tok::Comma)?;
                let b = self.use_value()?;
                Inst::Cmp { pred, ty, dst: def!(0), a, b }
            }
            "not" => {
                want_dsts(1, self)?;
                let a = self.use_value()?;
                Inst::Not { dst: def!(0), a }
            }
            "select" => {
                want_dsts(1, self)?;
                let cond = self.use_value()?;
                self.expect(Tok::Comma)?;
                let a = self.use_value()?;
                self.expect(Tok::Comma)?;
                let b = self.use_value()?;
                Inst::Select { dst: def!(0), cond, a, b }
            }
            "itof" => {
                want_dsts(1, self)?;
                let a = self.use_value()?;
                Inst::Itof { dst: def!(0), a }
            }
            "ftoi.trunc" | "ftoi.round" => {
                want_dsts(1, self)?;
                let mode = if opcode.ends_with("trunc") {
                    RoundMode::Trunc
                } else {
                    RoundMode::Round
                };
                let a = self.use_value()?;
                Inst::Ftoi { mode, dst: def!(0), a }
            }
            "itos" => {
                want_dsts(1, self)?;
                let a = self.use_value()?;
                Inst::Itos { dst: def!(0), a }
            }
            "ftos" => {
                want_dsts(1, self)?;
                let a = self.use_value()?;
                Inst::Ftos { dst: def!(0), a }
            }
            "stoi.opt" | "stof.opt" => {
                want_dsts(2, self)?;
                let a = self.use_value()?;
                let flag = def!(0);
                let dst = def!(1);
                if opcode.starts_with("stoi") {
                    Inst::StoiOpt { flag, dst, a }
                } else {
                    Inst::StofOpt { flag, dst, a }
                }
            }
            "sconcat" => {
                want_dsts(1, self)?;
                let a = self.use_value()?;
                self.expect(Tok::Comma)?;
                let b = self.use_value()?;
                Inst::Sconcat { dst: def!(0), a, b }
            }
            "load" => {
                want_dsts(1, self)?;
                let col = self.col_ref("in", in_cols)?;
                Inst::Load { dst: def!(0), col }
            }
            "load.opt" => {
                want_dsts(2, self)?;
                let col = self.col_ref("in", in_cols)?;
                Inst::LoadOpt { flag: def!(0), dst: def!(1), col }
            }
            "probe" => {
                if dst_names.is_empty() {
                    return Err(self.err("'probe' defines at least a hit flag"));
                }
                let static_id = self.static_ref(statics)?;
                let mut keys = Vec::new();
                while *self.peek() == Tok::Comma {
                    self.bump();
                    keys.push(self.use_value()?);
                }
                let hit = def!(0);
                let mut dsts = Vec::with_capacity(dst_names.len() - 1);
                for i in 1..dst_names.len() {
                    dsts.push(def!(i));
                }
                Inst::Probe { static_id, hit, dsts, keys }
            }
            "sload" => {
                want_dsts(1, self)?;
                let static_id = self.static_ref(statics)?;
                Inst::Sload { static_id, dst: def!(0) }
            }
            "sload.opt" => {
                want_dsts(2, self)?;
                let static_id = self.static_ref(statics)?;
                Inst::SloadOpt { static_id, flag: def!(0), dst: def!(1) }
            }
            other => return Err(self.err(format!("unknown opcode '{other}'"))),
        };
        Ok(inst)
    }

    fn dot_suffix(&mut self) -> Result<Option<String>, ParseError> {
        if *self.peek() == Tok::Dot {
            self.bump();
            Ok(Some(self.ident("an opcode suffix")?))
        } else {
            Ok(None)
        }
    }

    /// `in.NAME` / `out.NAME` (NAME bare or quoted) -> column index.
    fn col_ref(&mut self, side: &str, cols: &[Col]) -> Result<u32, ParseError> {
        self.keyword(side)?;
        self.expect(Tok::Dot)?;
        let name = match self.bump() {
            Tok::Ident(s) => s,
            Tok::Str(s) => s,
            other => {
                return Err(self.err(format!("expected a column name, found {}", other.show())))
            }
        };
        cols.iter()
            .position(|c| c.name == name)
            .map(|i| i as u32)
            .ok_or_else(|| self.err(format!("unknown column '{side}.{name}'")))
    }

    fn static_ref(&mut self, statics: &[StaticTy]) -> Result<u32, ParseError> {
        self.expect(Tok::At)?;
        let idx = self.int_literal("static index")?;
        if idx < 0 || idx as usize >= statics.len() {
            return Err(self.err(format!("unknown static '@{idx}'")));
        }
        Ok(idx as u32)
    }

    fn f64_literal(&mut self) -> Result<f64, ParseError> {
        match self.bump() {
            Tok::Num(s) => {
                if s == "-inf" {
                    Ok(f64::NEG_INFINITY)
                } else {
                    s.parse::<f64>()
                        .map_err(|_| self.err(format!("bad float literal '{s}'")))
                }
            }
            Tok::Ident(s) if s == "inf" => Ok(f64::INFINITY),
            Tok::Ident(s) if s == "nan" => Ok(f64::NAN),
            other => Err(self.err(format!("expected a float literal, found {}", other.show()))),
        }
    }
}
