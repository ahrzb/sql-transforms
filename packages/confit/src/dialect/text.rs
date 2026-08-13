//! Canonical plan text — mandatory properties 2 and 3 of the ir/ recipe.
//!
//! One deterministic spelling per plan ([`print`]), parsed back by
//! [`parse`]; the round-trip law is `parse(print(p)) == p` on the PLAN
//! (the parser is whitespace-insensitive; the printer is the canon).
//! Bindings and types are visible: fixtures show exactly what the frontend
//! bound, and a drifted printer fails the round-trip test instead of
//! shipping.
//!
//! Grammar (s-expressions; heads are bare atoms, every name/type/lexeme is
//! a quoted string, ordinals are bare integers):
//!
//! ```text
//! rel  := (scan "table")
//!       | (filter rel expr)
//!       | (project rel (item "name" expr)+)
//!       | (join inner|left|right|full rel rel expr)
//!       | (join cross rel rel)
//! expr := (col N "name" "ty")
//!       | (lit "ty" "lexeme")
//!       | (bin OP expr expr)            OP: add sub mul fdiv idiv rem
//!       |                                   concat and or eq neq lt lte gt gte
//!       | (un neg|not expr)
//!       | (cast strict|try "ty" expr)
//!       | (case (when expr expr)+ (else expr)?)
//!       | (isnull expr) | (isnotnull expr)
//!       | (isdistinct expr expr) | (isnotdistinct expr expr)
//! ```

use super::plan::{BinOp, Expr, JoinKind, Rel, ScalarFn, UnOp};
use super::ty::DTy;
use super::DialectError;

// --- printing ---------------------------------------------------------------

pub fn print(rel: &Rel) -> String {
    let mut out = String::new();
    print_rel(rel, 0, &mut out);
    out
}

fn indent(depth: usize, out: &mut String) {
    out.push('\n');
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn print_rel(rel: &Rel, depth: usize, out: &mut String) {
    match rel {
        Rel::Scan { table } => {
            out.push_str("(scan ");
            push_quoted(table, out);
            out.push(')');
        }
        Rel::Filter { input, pred } => {
            out.push_str("(filter");
            indent(depth + 1, out);
            print_rel(input, depth + 1, out);
            indent(depth + 1, out);
            print_expr(pred, out);
            out.push(')');
        }
        Rel::Project { input, items } => {
            out.push_str("(project");
            indent(depth + 1, out);
            print_rel(input, depth + 1, out);
            for (name, e) in items {
                indent(depth + 1, out);
                out.push_str("(item ");
                push_quoted(name, out);
                out.push(' ');
                print_expr(e, out);
                out.push(')');
            }
            out.push(')');
        }
        Rel::Join {
            left,
            right,
            kind,
            on,
        } => {
            out.push_str(&format!("(join {}", kind.name()));
            indent(depth + 1, out);
            print_rel(left, depth + 1, out);
            indent(depth + 1, out);
            print_rel(right, depth + 1, out);
            if let Some(pred) = on {
                indent(depth + 1, out);
                print_expr(pred, out);
            }
            out.push(')');
        }
    }
}

fn print_expr(e: &Expr, out: &mut String) {
    match e {
        Expr::Col { ordinal, name, ty } => {
            out.push_str(&format!("(col {ordinal} "));
            push_quoted(name, out);
            out.push(' ');
            push_quoted(&ty.name(), out);
            out.push(')');
        }
        Expr::Lit { lexeme, ty } => {
            out.push_str("(lit ");
            push_quoted(&ty.name(), out);
            out.push(' ');
            push_quoted(lexeme, out);
            out.push(')');
        }
        Expr::Bin { op, l, r } => {
            out.push_str(&format!("(bin {} ", op.name()));
            print_expr(l, out);
            out.push(' ');
            print_expr(r, out);
            out.push(')');
        }
        Expr::Un { op, e } => {
            let n = match op {
                UnOp::Neg => "neg",
                UnOp::Not => "not",
            };
            out.push_str(&format!("(un {n} "));
            print_expr(e, out);
            out.push(')');
        }
        Expr::Cast { strict, e, target } => {
            out.push_str(&format!(
                "(cast {} ",
                if *strict { "strict" } else { "try" }
            ));
            push_quoted(&target.name(), out);
            out.push(' ');
            print_expr(e, out);
            out.push(')');
        }
        Expr::Case { whens, else_ } => {
            out.push_str("(case");
            for (c, v) in whens {
                out.push_str(" (when ");
                print_expr(c, out);
                out.push(' ');
                print_expr(v, out);
                out.push(')');
            }
            if let Some(el) = else_ {
                out.push_str(" (else ");
                print_expr(el, out);
                out.push(')');
            }
            out.push(')');
        }
        Expr::IsNull { negated, e } => {
            out.push_str(if *negated { "(isnotnull " } else { "(isnull " });
            print_expr(e, out);
            out.push(')');
        }
        Expr::IsDistinct { negated, l, r } => {
            out.push_str(if *negated {
                "(isnotdistinct "
            } else {
                "(isdistinct "
            });
            print_expr(l, out);
            out.push(' ');
            print_expr(r, out);
            out.push(')');
        }
        Expr::Call { func, args } => {
            out.push_str(&format!("(call {}", func.name()));
            for a in args {
                out.push(' ');
                print_expr(a, out);
            }
            out.push(')');
        }
    }
}

fn push_quoted(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
}

// --- parsing ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Open,
    Close,
    Atom(String),
    Quoted(String),
}

fn lex(s: &str) -> Result<Vec<Tok>, DialectError> {
    let mut toks = Vec::new();
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '(' => toks.push(Tok::Open),
            ')' => toks.push(Tok::Close),
            '"' => {
                let mut v = String::new();
                loop {
                    match chars.next() {
                        Some((_, '"')) => break,
                        Some((_, '\\')) => match chars.next() {
                            Some((_, e @ ('"' | '\\'))) => v.push(e),
                            other => {
                                return Err(DialectError::Text(format!(
                                    "bad escape in string at byte {i}: {other:?}"
                                )));
                            }
                        },
                        Some((_, c)) => v.push(c),
                        None => {
                            return Err(DialectError::Text(format!(
                                "unterminated string starting at byte {i}"
                            )));
                        }
                    }
                }
                toks.push(Tok::Quoted(v));
            }
            c if c.is_whitespace() => {}
            _ => {
                let mut a = String::from(c);
                while let Some(&(_, n)) = chars.peek() {
                    if n.is_whitespace() || n == '(' || n == ')' || n == '"' {
                        break;
                    }
                    a.push(n);
                    chars.next();
                }
                toks.push(Tok::Atom(a));
            }
        }
    }
    Ok(toks)
}

struct P {
    toks: Vec<Tok>,
    pos: usize,
}

impl P {
    fn err(&self, m: impl Into<String>) -> DialectError {
        DialectError::Text(format!("{} at token {}", m.into(), self.pos))
    }

    fn next(&mut self) -> Result<Tok, DialectError> {
        let t = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| self.err("unexpected end"))?;
        self.pos += 1;
        Ok(t)
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn expect_open(&mut self) -> Result<(), DialectError> {
        match self.next()? {
            Tok::Open => Ok(()),
            t => Err(self.err(format!("expected ( got {t:?}"))),
        }
    }

    fn expect_close(&mut self) -> Result<(), DialectError> {
        match self.next()? {
            Tok::Close => Ok(()),
            t => Err(self.err(format!("expected ) got {t:?}"))),
        }
    }

    fn head(&mut self) -> Result<String, DialectError> {
        match self.next()? {
            Tok::Atom(a) => Ok(a),
            t => Err(self.err(format!("expected head atom got {t:?}"))),
        }
    }

    fn quoted(&mut self) -> Result<String, DialectError> {
        match self.next()? {
            Tok::Quoted(s) => Ok(s),
            t => Err(self.err(format!("expected quoted string got {t:?}"))),
        }
    }

    fn ty(&mut self) -> Result<DTy, DialectError> {
        let s = self.quoted()?;
        DTy::parse(&s).ok_or_else(|| self.err(format!("unknown type: {s}")))
    }

    fn rel(&mut self) -> Result<Rel, DialectError> {
        self.expect_open()?;
        let head = self.head()?;
        let rel = match head.as_str() {
            "scan" => Rel::Scan {
                table: self.quoted()?,
            },
            "filter" => {
                let input = Box::new(self.rel()?);
                let pred = self.expr()?;
                Rel::Filter { input, pred }
            }
            "project" => {
                let input = Box::new(self.rel()?);
                let mut items = Vec::new();
                while matches!(self.peek(), Some(Tok::Open)) {
                    self.expect_open()?;
                    let h = self.head()?;
                    if h != "item" {
                        return Err(self.err(format!("expected item got {h}")));
                    }
                    let name = self.quoted()?;
                    let e = self.expr()?;
                    self.expect_close()?;
                    items.push((name, e));
                }
                Rel::Project { input, items }
            }
            "join" => {
                let kname = self.head()?;
                let kind = JoinKind::parse(&kname)
                    .ok_or_else(|| self.err(format!("unknown join kind: {kname}")))?;
                let left = Box::new(self.rel()?);
                let right = Box::new(self.rel()?);
                let on = if kind == JoinKind::Cross {
                    None
                } else {
                    Some(self.expr()?)
                };
                Rel::Join {
                    left,
                    right,
                    kind,
                    on,
                }
            }
            h => return Err(self.err(format!("unknown rel head: {h}"))),
        };
        self.expect_close()?;
        Ok(rel)
    }

    fn expr(&mut self) -> Result<Expr, DialectError> {
        self.expect_open()?;
        let head = self.head()?;
        let e = match head.as_str() {
            "col" => {
                let ordinal: usize = match self.next()? {
                    Tok::Atom(a) => a
                        .parse()
                        .map_err(|_| self.err(format!("bad ordinal: {a}")))?,
                    t => return Err(self.err(format!("expected ordinal got {t:?}"))),
                };
                let name = self.quoted()?;
                let ty = self.ty()?;
                Expr::Col { ordinal, name, ty }
            }
            "lit" => {
                let ty = self.ty()?;
                let lexeme = self.quoted()?;
                Expr::Lit { lexeme, ty }
            }
            "bin" => {
                let opname = self.head()?;
                let op = BinOp::parse(&opname)
                    .ok_or_else(|| self.err(format!("unknown bin op: {opname}")))?;
                Expr::Bin {
                    op,
                    l: Box::new(self.expr()?),
                    r: Box::new(self.expr()?),
                }
            }
            "un" => {
                let opname = self.head()?;
                let op = match opname.as_str() {
                    "neg" => UnOp::Neg,
                    "not" => UnOp::Not,
                    _ => return Err(self.err(format!("unknown un op: {opname}"))),
                };
                Expr::Un {
                    op,
                    e: Box::new(self.expr()?),
                }
            }
            "cast" => {
                let strict = match self.head()?.as_str() {
                    "strict" => true,
                    "try" => false,
                    k => return Err(self.err(format!("expected strict|try got {k}"))),
                };
                let target = self.ty()?;
                Expr::Cast {
                    strict,
                    e: Box::new(self.expr()?),
                    target,
                }
            }
            "case" => {
                let mut whens = Vec::new();
                let mut else_ = None;
                while matches!(self.peek(), Some(Tok::Open)) {
                    self.expect_open()?;
                    match self.head()?.as_str() {
                        "when" => {
                            let c = self.expr()?;
                            let v = self.expr()?;
                            whens.push((c, v));
                        }
                        "else" => {
                            else_ = Some(Box::new(self.expr()?));
                        }
                        k => return Err(self.err(format!("expected when|else got {k}"))),
                    }
                    self.expect_close()?;
                }
                if whens.is_empty() {
                    return Err(self.err("case with no when arms"));
                }
                Expr::Case { whens, else_ }
            }
            "isnull" => Expr::IsNull {
                negated: false,
                e: Box::new(self.expr()?),
            },
            "isnotnull" => Expr::IsNull {
                negated: true,
                e: Box::new(self.expr()?),
            },
            "isdistinct" => Expr::IsDistinct {
                negated: false,
                l: Box::new(self.expr()?),
                r: Box::new(self.expr()?),
            },
            "isnotdistinct" => Expr::IsDistinct {
                negated: true,
                l: Box::new(self.expr()?),
                r: Box::new(self.expr()?),
            },
            "call" => {
                let fname = self.head()?;
                let func = ScalarFn::parse(&fname)
                    .ok_or_else(|| self.err(format!("unknown function: {fname}")))?;
                let mut args = Vec::new();
                while matches!(self.peek(), Some(Tok::Open)) {
                    args.push(self.expr()?);
                }
                Expr::Call { func, args }
            }
            h => return Err(self.err(format!("unknown expr head: {h}"))),
        };
        self.expect_close()?;
        Ok(e)
    }
}

pub fn parse(s: &str) -> Result<Rel, DialectError> {
    let mut p = P {
        toks: lex(s)?,
        pos: 0,
    };
    let rel = p.rel()?;
    if p.pos != p.toks.len() {
        return Err(p.err("trailing tokens after plan"));
    }
    Ok(rel)
}
