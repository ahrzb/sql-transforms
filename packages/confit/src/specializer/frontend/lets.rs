//! A SQL function's lets ([`super::macros::SqlMacro::lets`]), bound once.
//!
//! Macro expansion gives each call's lets numbers of their own and parses
//! each once, beside the query (`__cf_let(i)` reads entry `i`). The meaning
//! is DuckDB's: the let spelled out at every read. So a read binds the let
//! where it stands, as the spelled-out text would, and the binder keeps
//! what it bound, folded as the projection is, for the next read in the
//! same scope:
//!
//! - a value that cannot trap ([`can_trap`], whose allowlist also keeps out
//!   every call with an effect), is not closed and has at least
//!   [`MIN_LET_SIZE`] nodes becomes an entry of the level's lets, which a
//!   read in the projection reads through [`SKind::Let`]. Where and how
//!   often such a value is evaluated is invisible, so `share.rs` computes
//!   it once per row, as it computes any repeated value, in the DAG of the
//!   projection: the decisions are the ones it makes for the text spelled
//!   out, at the size of the lets. A read anywhere else (a WHERE, a JOIN
//!   ON, a `shape='many'` query) answers the value itself, so every check
//!   made there sees what the text spelled out gives;
//! - a smaller one, or a closed one, is copied to every read;
//! - one that can trap binds again at every read, where it stands: each
//!   read keeps its own place, its own guards and its own traps.
//!
//! Spelled out, a body can grow exponentially (a recurrence whose every
//! step reads the step before twice doubles per step), and expansion
//! refuses a query past [`MAX_TOKENS`] tokens. What still spells a let out
//! here counts against one more budget of that size for the whole query
//! ([`spend`]): binding a let again (its tokens) and a value read in place
//! (its nodes). An unaliased item spells out nothing: it is named after its
//! text as written, which reads the call (`sc(x)`), not its lets.
//!
//! A read past DuckDB's depth limit refuses as the spelled-out text does:
//! each kept value carries the depth its binding reached.
//!
//! A read is its text for the checks that type a value by its spelling,
//! too: DuckDB types `tinyint + (48)` TINYINT, as it types `tinyint + 48`
//! ([`through`]). And a kept value is folded only as far as its spelling
//! allows: a VARCHAR constant that is not a string literal stays a VARCHAR
//! ([`fold_kept`]).

use std::cell::OnceCell;
use std::collections::HashSet;
use std::rc::Rc;

use sqlparser::tokenizer::Token;

use super::macros::{LET_MARKER, MAX_TOKENS};
use super::*;

/// One let of the query being bound.
pub(super) struct LetText {
    /// The SQL function it belongs to.
    pub(super) name: String,
    pub(super) ast: SqlExpr,
    /// Its tokens: what binding it once costs.
    tokens: usize,
    /// Its tokens with the lets it reads spelled out (saturating).
    spelled: usize,
    /// The identifiers in its text and in the lets it reads: the lateral
    /// aliases among them are what its binding can depend on ([`LetKey`]).
    words: Rc<HashSet<String>>,
    /// Its SQL text, the lets it reads spelled out (`None` past the cap).
    shown: OnceCell<Option<Rc<str>>>,
    /// What the checks of its spelling answer, each once.
    spelling: Spelling,
}

/// What the checks of a spelling (`typing::ast_int_literal` and its kin)
/// answer for a let's text, each computed once: a recurrence reads the let
/// before it twice, so walking the text again at every read would take
/// exponential time.
#[derive(Default)]
pub(super) struct Spelling {
    pub(super) int_literal: OnceCell<Option<i128>>,
    pub(super) signed_number: OnceCell<Option<(u32, String)>>,
    pub(super) decimal_literal: OnceCell<bool>,
    pub(super) decimal_typed: OnceCell<bool>,
    pub(super) i32_fold: OnceCell<I32Fold>,
}

/// The lets of a query, each `(function name, tokens, parsed text)` in
/// order: each reads only the ones before it.
pub(super) fn table(lets: Vec<(String, Vec<Token>, SqlExpr)>) -> Vec<LetText> {
    let mut out: Vec<LetText> = Vec::with_capacity(lets.len());
    for (name, tokens, ast) in lets {
        let mut words = HashSet::new();
        let mut reads = Vec::new();
        let mut it = tokens.iter().filter(|t| !matches!(t, Token::Whitespace(_)));
        while let Some(t) = it.next() {
            let Token::Word(w) = t else { continue };
            if w.quote_style.is_none() && w.value.eq_ignore_ascii_case(LET_MARKER) {
                // `__cf_let(i)`: what let `i` names, this one names too.
                if let (Some(Token::LParen), Some(Token::Number(n, _))) = (it.next(), it.next()) {
                    reads.extend(n.parse::<usize>().ok().filter(|&i| i < out.len()));
                }
                continue;
            }
            words.insert(w.value.to_ascii_lowercase());
        }
        // Spelled out, each read's four tokens give way to the let's own.
        let spelled = reads.iter().fold(
            tokens.len().saturating_sub(4 * reads.len()),
            |n, &i| n.saturating_add(out[i].spelled),
        );
        reads.sort_unstable();
        reads.dedup();
        for i in reads {
            words.extend(out[i].words.iter().cloned());
        }
        out.push(LetText {
            name,
            ast,
            tokens: tokens.len(),
            spelled,
            words: Rc::new(words),
            shown: OnceCell::new(),
            spelling: Spelling::default(),
        });
    }
    out
}

impl LetText {
    fn shown(&self, lets: &[LetText]) -> Option<Rc<str>> {
        self.shown
            .get_or_init(|| {
                stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || {
                    spell_out(&self.ast.to_string(), lets).map(Rc::from)
                })
            })
            .clone()
    }
}

/// `text` with every let read replaced by the let's SQL text, spelled out
/// in turn: `None` past the cap.
fn spell_out(text: &str, lets: &[LetText]) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let mut out = String::new();
    let mut at = 0;
    while let Some(k) = lower[at..].find(LET_MARKER) {
        let start = at + k;
        let after = start + LET_MARKER.len();
        let read = text[after..].strip_prefix('(').and_then(|inner| {
            let digits = inner.chars().take_while(char::is_ascii_digit).count();
            let id: usize = inner[..digits].parse().ok()?;
            inner[digits..]
                .starts_with(')')
                .then(|| (lets.get(id), after + 1 + digits + 1))
        });
        let Some((Some(l), end)) = read else {
            out.push_str(&text[at..after]);
            at = after;
            continue;
        };
        out.push_str(&text[at..start]);
        out.push_str(&l.shown(lets)?);
        if out.len() > MAX_TOKENS {
            return None;
        }
        at = end;
    }
    out.push_str(&text[at..]);
    (out.len() <= MAX_TOKENS).then_some(out)
}

/// A message with every let read in it spelled out, as it reads for the
/// text spelled out: `None` when it reads none, or one past the cap.
fn spell_out_text(text: &str) -> Option<String> {
    if !text.to_ascii_lowercase().contains(LET_MARKER) {
        return None;
    }
    let lets = LETS.with(|c| c.borrow().clone());
    spell_out(text, &lets)
}

/// A refusal reads a let as its text, as the refusal of the text spelled
/// out does.
pub(super) fn spell_out_error(e: PrepareError) -> PrepareError {
    let spell = |m: String| spell_out_text(&m).unwrap_or(m);
    match e {
        PrepareError::Parse(m) => PrepareError::Parse(spell(m)),
        PrepareError::Unsupported(m) => PrepareError::Unsupported(spell(m)),
        PrepareError::Bind(m) => PrepareError::Bind(spell(m)),
        PrepareError::Internal(m) => PrepareError::Internal(spell(m)),
    }
}

thread_local! {
    static LETS: std::cell::RefCell<Rc<Vec<LetText>>> =
        std::cell::RefCell::new(Rc::new(Vec::new()));
    /// What the query being bound has spelled out again so far.
    static SPENT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The let table of the query being bound, installed for the duration of
/// [`super::frontend`] and removed on drop (an error included), with a
/// budget of its own.
pub(super) struct Installed(Rc<Vec<LetText>>, usize);

impl Installed {
    pub(super) fn new(lets: Vec<LetText>) -> Self {
        let prev = LETS.with(|c| c.replace(Rc::new(lets)));
        Installed(prev, SPENT.with(|s| s.replace(0)))
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.0);
        LETS.with(|c| *c.borrow_mut() = prev);
        SPENT.with(|s| s.set(self.1));
    }
}

/// Count `n` more tokens (or nodes) of a let spelled out again, for the
/// whole query: `false` once the total passes [`MAX_TOKENS`]. Each read
/// alone may stay under the cap while many together do not, as many reads
/// in the text spelled out pass it together.
fn spend(n: usize) -> bool {
    SPENT.with(|s| {
        let total = s.get().saturating_add(n);
        s.set(total);
        total <= MAX_TOKENS
    })
}

/// The let a `__cf_let(i)` marker reads: its number and the table.
pub(super) fn marker_let(e: &SqlExpr) -> Option<(usize, Rc<Vec<LetText>>)> {
    let SqlExpr::Function(f) = e else {
        return None;
    };
    if !f.name.to_string().eq_ignore_ascii_case(LET_MARKER) {
        return None;
    }
    let sqlparser::ast::FunctionArguments::List(list) = &f.args else {
        return None;
    };
    let [sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
        SqlExpr::Value(v),
    ))] = list.args.as_slice()
    else {
        return None;
    };
    let SqlValue::Number(n, _) = &v.value else {
        return None;
    };
    let id: usize = n.parse().ok()?;
    let lets = LETS.with(|c| c.borrow().clone());
    (id < lets.len()).then_some((id, lets))
}

/// A check of a spelling (an integer or decimal literal, see
/// `typing::ast_int_literal`) of the text a let read stands for, as DuckDB
/// reads it spelled out: `check` of the let's text, kept in its `cell`.
/// `None` when `e` is not a let read.
pub(super) fn through<T: Clone>(
    e: &SqlExpr,
    cell: fn(&Spelling) -> &OnceCell<T>,
    check: fn(&SqlExpr) -> T,
) -> Option<T> {
    let (id, lets) = marker_let(e)?;
    let text = &lets[id];
    let answer = cell(&text.spelling).get_or_init(|| {
        stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || check(&text.ast))
    });
    Some(answer.clone())
}

/// A let's value folded as the projection is, keeping the one mark a read's
/// context takes from the bound value: it reads as a string literal only
/// when its text is one. Folded, `upper('5')` is the constant `'5'`, but
/// to DuckDB it is a VARCHAR, which casts to a number in fewer places than
/// a literal does.
fn fold_kept(e: SExpr) -> SExpr {
    let literal = matches!(e.kind, SKind::Lit(Lit::Str(_)));
    let e = fold(e);
    if literal || !matches!(e.kind, SKind::Lit(Lit::Str(_))) {
        return e;
    }
    // The node a string literal cast to VARCHAR stays (`Binder::cast`).
    let nullable = e.nullable;
    SExpr {
        kind: SKind::Cast {
            inner: Box::new(e),
            trying: false,
        },
        ty: Ty::Str,
        nullable,
    }
}

/// The fewest nodes a let read through [`SKind::Let`] has; a smaller value
/// is cheaper to copy to each read (`share.rs` holds the same line).
const MIN_LET_SIZE: usize = 6;

/// What a let binds to depends on the scope it binds in, as a call's
/// siblings do (`calls::ScopeKey`): the let, the joins in scope, the
/// lateral aliases its text can name, and the ON-key contexts. The guard
/// depth is not part of it: it decides only whether a constant that traps
/// refuses, and a value that cannot trap holds none.
pub(super) type LetKey = (usize, usize, usize, bool, usize);

/// A let's binding in one scope.
#[derive(Clone)]
pub(super) enum LetVal {
    /// Every read answers this (`None`: a bare NULL, typed where it is
    /// read), reaching `height` levels below the read.
    Kept { e: Option<SExpr>, height: u32 },
    /// It can trap: every read binds it again where it stands.
    Rebind,
}

fn size_at_least(e: &mut SExpr, n: usize) -> bool {
    fn count(e: &mut SExpr, left: &mut usize) {
        if *left == 0 {
            return;
        }
        *left -= 1;
        for c in e.children_mut() {
            count(c, left);
        }
    }
    let mut left = n;
    count(e, &mut left);
    left == 0
}

impl Binder<'_> {
    fn let_key(&self, id: usize, text: &LetText) -> LetKey {
        let aliases = self
            .bound_aliases
            .borrow()
            .iter()
            .filter(|(a, _)| text.words.contains(&a.to_ascii_lowercase()))
            .count();
        (
            id,
            self.joins.len(),
            aliases,
            self.classify_keys.get(),
            self.beside.borrow().len(),
        )
    }

    /// A read of let `id`.
    pub(super) fn let_read(&self, id: usize, text: &LetText) -> Result<Option<SExpr>, PrepareError> {
        // The lets a let reads are read through `SKind::Let` wherever it
        // stands, so its value stays the size of its text; a read outside
        // the projection replaces them all, once, here.
        let outer = self.let_reads.replace(true);
        let read = self.let_value(id, text);
        self.let_reads.set(outer);
        match read? {
            (Some(mut e), copy) if !outer => {
                let cost = if copy { Cost::Whole } else { Cost::Reads };
                inline(&mut e, &self.lets.borrow(), &mut self.let_inlined.borrow_mut(), cost)?;
                Ok(Some(e))
            }
            (read, _) => Ok(read),
        }
    }

    /// The let's value at this read, and whether it is a copy of an earlier
    /// read's (a value bound here stands where its text does).
    fn let_value(&self, id: usize, text: &LetText) -> Result<(Option<SExpr>, bool), PrepareError> {
        let key = self.let_key(id, text);
        let kept = self.let_vals.borrow().get(&key).cloned();
        match kept {
            Some(LetVal::Kept { e, height }) => {
                expr::check_depth(height)?;
                return Ok((e, true));
            }
            Some(LetVal::Rebind) => {
                self.spell_again(text)?;
                return Ok((self.expr_or_null(&text.ast)?.map(fold_kept), false));
            }
            None => {}
        }
        let (bound, height) = expr::measure_depth(|| self.expr_or_null(&text.ast));
        let Some(mut e) = bound?.map(fold_kept) else {
            self.let_vals
                .borrow_mut()
                .insert(key, LetVal::Kept { e: None, height });
            return Ok((None, false));
        };
        if can_trap(&e) {
            self.let_vals.borrow_mut().insert(key, LetVal::Rebind);
            return Ok((Some(e), false));
        }
        if !bind_foldable(&e) && size_at_least(&mut e, MIN_LET_SIZE) {
            let mut lets = self.lets.borrow_mut();
            let (ty, nullable) = (e.ty, e.nullable);
            lets.push(e);
            e = SExpr {
                kind: SKind::Let((lets.len() - 1) as u32),
                ty,
                nullable,
            };
        }
        self.let_vals.borrow_mut().insert(
            key,
            LetVal::Kept {
                e: Some(e.clone()),
                height,
            },
        );
        Ok((Some(e), false))
    }

    /// Count a let's text bound once more toward the query's budget.
    pub(super) fn spell_again(&self, text: &LetText) -> Result<(), PrepareError> {
        if !spend(text.tokens) {
            return Err(unsup(format!(
                "sql function '{}' expands past {MAX_TOKENS} tokens",
                text.name
            )));
        }
        Ok(())
    }
}

/// Let values with their own reads replaced, and their sizes so, per entry
/// of a level's lets.
#[derive(Default)]
pub(super) struct Inlined {
    full: Vec<Option<SExpr>>,
    size: Vec<Option<usize>>,
}

/// What [`inline`] counts toward the budget.
pub(super) enum Cost {
    /// All of `e`: a copy of a value that stood elsewhere.
    Whole,
    /// What the let reads in `e` add: `e` itself stands where its text does.
    Reads,
}

/// `e` with every let read replaced by its value, for a clause that does
/// not read lets. Refuses once the values read in place, with the rest the
/// query spelled out again, pass [`MAX_TOKENS`] nodes, as the text spelled
/// out would have passed that many tokens.
pub(super) fn inline(
    e: &mut SExpr,
    lets: &[SExpr],
    memo: &mut Inlined,
    cost: Cost,
) -> Result<(), PrepareError> {
    if memo.full.len() < lets.len() {
        memo.full.resize(lets.len(), None);
        memo.size.resize(lets.len(), None);
    }
    let n = match cost {
        Cost::Whole => size(e, lets, memo),
        Cost::Reads => added(e, lets, memo),
    };
    if !spend(n) {
        return Err(unsup(format!(
            "a sql function's value read outside the projection spells out past \
             {MAX_TOKENS} nodes"
        )));
    }
    replace(e, lets, memo);
    Ok(())
}

/// Nodes in `e` with every let read replaced (saturating).
fn size(e: &mut SExpr, lets: &[SExpr], memo: &mut Inlined) -> usize {
    stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || {
        if let SKind::Let(n) = e.kind {
            let n = n as usize;
            if let Some(s) = memo.size[n] {
                return s;
            }
            let s = size(&mut lets[n].clone(), lets, memo);
            memo.size[n] = Some(s);
            return s;
        }
        let mut total = 1usize;
        for c in e.children_mut() {
            total = total.saturating_add(size(c, lets, memo));
        }
        total
    })
}

/// Nodes the let reads in `e` bring when replaced (saturating).
fn added(e: &mut SExpr, lets: &[SExpr], memo: &mut Inlined) -> usize {
    stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || {
        if let SKind::Let(_) = e.kind {
            return size(e, lets, memo);
        }
        let mut total = 0usize;
        for c in e.children_mut() {
            total = total.saturating_add(added(c, lets, memo));
        }
        total
    })
}

fn replace(e: &mut SExpr, lets: &[SExpr], memo: &mut Inlined) {
    stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || {
        if let SKind::Let(n) = e.kind {
            let n = n as usize;
            if memo.full[n].is_none() {
                let mut v = lets[n].clone();
                replace(&mut v, lets, memo);
                memo.full[n] = Some(v);
            }
            *e = memo.full[n].clone().expect("inlined above");
            return;
        }
        for c in e.children_mut() {
            replace(c, lets, memo);
        }
    })
}

/// Field reads and other syntax over a let read see the let's text: the
/// read stands for it, and binds it again (see [`Binder::spell_again`]).
pub(super) fn spelled(e: &SqlExpr) -> Option<(SqlExpr, Rc<Vec<LetText>>, usize)> {
    let mut base = e;
    while let SqlExpr::Nested(i) = base {
        base = i;
    }
    let (id, lets) = marker_let(base)?;
    Some((SqlExpr::Nested(Box::new(lets[id].ast.clone())), lets, id))
}
