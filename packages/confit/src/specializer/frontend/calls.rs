//! Field reads over SQL function calls (`f(x).a`), bound once per call.
//!
//! Macro expansion turns a call read by field into
//! `(__cf_call(id, args...)).a` and parses the call's expansion once (see
//! [`super::macros::Expanded`]). Here a read binds the field it reads, and
//! the fields beside it bind once per call and scope: DuckDB builds every
//! field of a struct_pack, so a read keeps the siblings' traps, as their
//! trap skeletons (`plan::trap_skeleton`), each distinct one once.

use std::rc::Rc;

use super::*;

thread_local! {
    static CALLS: std::cell::RefCell<Rc<Vec<(String, SqlExpr)>>> =
        std::cell::RefCell::new(Rc::new(Vec::new()));
}

/// The call table of the query being bound, installed for the duration of
/// [`super::frontend`] and removed on drop (an error included).
pub(super) struct Installed(Rc<Vec<(String, SqlExpr)>>);

impl Installed {
    pub(super) fn new(mut calls: Vec<(String, SqlExpr)>) -> Self {
        split_case_arms(&mut calls);
        let prev = CALLS.with(|c| c.replace(Rc::new(calls)));
        Installed(prev)
    }
}

/// A call whose expansion is a CASE over struct_packs (a `null_when` wraps
/// the body as `CASE WHEN c THEN NULL ELSE struct_pack(...) END`): each
/// struct_pack arm becomes a call of its own, `__cf_call(arm)`, so a field
/// read over the CASE reads each arm through [`Binder::call_field`] and its
/// sibling cache instead of re-binding every field per read.
fn split_case_arms(calls: &mut Vec<(String, SqlExpr)>) {
    for id in 0..calls.len() {
        let next = calls.len();
        let mut body = &mut calls[id].1;
        while let SqlExpr::Nested(i) = body {
            body = i;
        }
        let SqlExpr::Case {
            conditions,
            else_result,
            ..
        } = body
        else {
            continue;
        };
        let mut arms: Vec<&mut SqlExpr> = conditions.iter_mut().map(|w| &mut w.result).collect();
        if let Some(e) = else_result {
            arms.push(e);
        }
        let mut split = Vec::new();
        for arm in arms {
            let mut inner: &SqlExpr = arm;
            while let SqlExpr::Nested(i) = inner {
                inner = i;
            }
            let SqlExpr::Function(f) = inner else {
                continue;
            };
            if !(f.name.0.len() == 1 && f.name.to_string().eq_ignore_ascii_case("struct_pack")) {
                continue;
            }
            let sub = next + split.len();
            let Ok(marker) = Parser::new(&GenericDialect {})
                .try_with_sql(&format!("{}({sub})", macros::CALL_MARKER))
                .and_then(|mut p| p.parse_expr())
            else {
                continue;
            };
            split.push(std::mem::replace(arm, marker));
        }
        let name = calls[id].0.clone();
        calls.extend(split.into_iter().map(|a| (name.clone(), a)));
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.0);
        CALLS.with(|c| *c.borrow_mut() = prev);
    }
}

/// The call a `__cf_call(id, ...)` marker stands for: its function's name and
/// its parsed expansion.
pub(super) fn marker_call(e: &SqlExpr) -> Option<(usize, Rc<Vec<(String, SqlExpr)>>)> {
    let SqlExpr::Function(f) = e else {
        return None;
    };
    if !f.name.to_string().eq_ignore_ascii_case(macros::CALL_MARKER) {
        return None;
    }
    let sqlparser::ast::FunctionArguments::List(list) = &f.args else {
        return None;
    };
    let Some(sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
        SqlExpr::Value(v),
    ))) = list.args.first()
    else {
        return None;
    };
    let SqlValue::Number(n, _) = &v.value else {
        return None;
    };
    let id: usize = n.parse().ok()?;
    let calls = CALLS.with(|c| c.borrow().clone());
    (id < calls.len()).then_some((id, calls))
}

/// The marker's arguments as the call spelled them.
pub(super) fn marker_args(f: &sqlparser::ast::Function) -> Vec<SqlExpr> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
    let FunctionArguments::List(list) = &f.args else {
        return Vec::new();
    };
    list.args
        .iter()
        .skip(1)
        .filter_map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e.clone()),
            _ => None,
        })
        .collect()
}

/// Whether the marker has arguments ([`marker_args`] is not empty).
fn has_args(f: &sqlparser::ast::Function) -> bool {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
    let FunctionArguments::List(list) = &f.args else {
        return false;
    };
    list.args
        .iter()
        .skip(1)
        .any(|a| matches!(a, FunctionArg::Unnamed(FunctionArgExpr::Expr(_))))
}

/// The marker `base` (with its arguments) standing for call `id` instead.
fn with_id(base: &SqlExpr, id: usize) -> SqlExpr {
    let mut out = base.clone();
    if let SqlExpr::Function(f) = &mut out {
        if let sqlparser::ast::FunctionArguments::List(list) = &mut f.args {
            if let Some(sqlparser::ast::FunctionArg::Unnamed(
                sqlparser::ast::FunctionArgExpr::Expr(SqlExpr::Value(v)),
            )) = list.args.first_mut()
            {
                v.value = SqlValue::Number(id.to_string(), false);
            }
        }
    }
    out
}

/// What a read's siblings bind to depends on the scope it binds in; the
/// cache is keyed by this as well as the call.
pub(super) type ScopeKey = (usize, usize, usize, u32, bool, usize);

/// One call's siblings in one scope: each distinct trap skeleton, at the
/// first field that has it.
pub(super) type Siblings = Rc<Vec<(usize, SExpr)>>;

impl Binder<'_> {
    fn scope_key(&self, id: usize, marker: &SqlExpr) -> ScopeKey {
        // A lateral alias bound since the last read changes what the call
        // binds to only when the call's ARGUMENTS name it (the body names
        // only its parameters); counting every alias would miss the cache
        // on each projection item (n reads, n^2 binds). The words come from
        // the marker, which carries the arguments, not from the expansion,
        // whose struct field names are often the output aliases
        // (`f(x)."f0" AS "f0"`).
        (
            id,
            self.joins.len(),
            self.aliases_named(id, marker),
            self.in_guarded.get(),
            self.classify_keys.get(),
            self.beside.borrow().len(),
        )
    }

    /// How many of the aliases bound so far the arguments of call `id`
    /// name. Aliases are only added, so the count goes on from where the
    /// last read of the call left it: counting from the first alias at each
    /// read would take n^2 steps over n items that each read the call.
    fn aliases_named(&self, id: usize, marker: &SqlExpr) -> usize {
        let bound = self.bound_aliases.borrow();
        let words = match marker {
            SqlExpr::Function(f) if has_args(f) => self.arg_words(id, f),
            _ => Rc::default(),
        };
        // A marker without arguments (a split CASE arm bound bare) counts
        // every alias.
        if words.is_empty() {
            return bound.len();
        }
        let mut counts = self.call_aliases.borrow_mut();
        let (from, n) = counts.entry(id).or_default();
        *n += bound[*from..]
            .iter()
            .filter(|(a, _)| words.contains(&a.to_ascii_lowercase()))
            .count();
        *from = bound.len();
        *n
    }

    /// The identifier words of the arguments of call `id`, as `f` spells
    /// them, taken once.
    fn arg_words(
        &self,
        id: usize,
        f: &sqlparser::ast::Function,
    ) -> Rc<std::collections::HashSet<String>> {
        self.call_words
            .borrow_mut()
            .entry(id)
            .or_insert_with(|| {
                Rc::new(
                    marker_args(f)
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" ")
                        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .filter(|w| !w.is_empty())
                        .map(str::to_ascii_lowercase)
                        .collect(),
                )
            })
            .clone()
    }

    /// A field read whose root is a call marker: `Ok(None)` when `e` is not
    /// one. Over a call whose expansion is a plain `struct_pack`, binds the
    /// read field and the cached sibling skeletons; any other expansion (a
    /// CASE, say) reads as the expansion itself would.
    pub(super) fn call_field(&self, e: &SqlExpr) -> Result<Option<Option<SExpr>>, PrepareError> {
        let SqlExpr::CompoundFieldAccess { root, access_chain } = e else {
            return Ok(None);
        };
        // A field read over a let read reads the let's text.
        if let Some((text, lets, id)) = lets::spelled(root) {
            self.spell_again(&lets[id])?;
            let read = SqlExpr::CompoundFieldAccess {
                root: Box::new(text),
                access_chain: access_chain.clone(),
            };
            return self.expr_or_null(&read).map(Some);
        }
        let Some((AccessExpr::Dot(SqlExpr::Identifier(field)), rest)) =
            access_chain.split_first()
        else {
            return Ok(None);
        };
        let mut base: &SqlExpr = root;
        while let SqlExpr::Nested(i) = base {
            base = i;
        }
        let Some((id, calls)) = marker_call(base) else {
            return Ok(None);
        };
        let mut body = &calls[id].1;
        while let SqlExpr::Nested(i) = body {
            body = i;
        }
        if let SqlExpr::Case { .. } = body {
            // A read of a struct DuckDB folds to NULL is a bare NULL.
            let key = self.scope_key(id, base);
            let cached = self.call_null.borrow().get(&key).copied();
            let folds = cached.unwrap_or_else(|| {
                let folds = self.struct_folds_to_null(body);
                self.call_null.borrow_mut().insert(key, folds);
                folds
            });
            if folds {
                return Ok(Some(None));
            }
            // Its struct_pack arms were split into calls of their own
            // (`split_case_arms`): each arm reads as that call, spelled with
            // this call's arguments, so it shares their scope key.
            let mut case = body.clone();
            if let SqlExpr::Case {
                conditions,
                else_result,
                ..
            } = &mut case
            {
                let arms = conditions
                    .iter_mut()
                    .map(|w| &mut w.result)
                    .chain(else_result.iter_mut().map(|e| &mut **e));
                for arm in arms {
                    if let Some((sub, _)) = marker_call(arm) {
                        *arm = with_id(base, sub);
                    }
                }
            }
            return self
                .expr_or_null(&structs::case_field(&case, access_chain))
                .map(Some);
        }
        let SqlExpr::Function(f) = body else {
            // Not a struct_pack: read it as written.
            let read = SqlExpr::CompoundFieldAccess {
                root: Box::new(SqlExpr::Nested(Box::new(body.clone()))),
                access_chain: access_chain.clone(),
            };
            return self.expr_or_null(&read).map(Some);
        };
        let Some((values, pick)) = self.struct_pack_values(f, &field.value)? else {
            let read = SqlExpr::CompoundFieldAccess {
                root: Box::new(SqlExpr::Nested(Box::new(body.clone()))),
                access_chain: access_chain.clone(),
            };
            return self.expr_or_null(&read).map(Some);
        };
        let picked = if rest.is_empty() {
            values[pick].clone()
        } else {
            SqlExpr::CompoundFieldAccess {
                root: Box::new(SqlExpr::Nested(Box::new(values[pick].clone()))),
                access_chain: rest.to_vec(),
            }
        };
        if values.len() == 1 {
            return self.expr_or_null(&picked).map(Some);
        }
        let key = self.scope_key(id, base);
        let cached = self.call_siblings.borrow().get(&key).cloned();
        let siblings: Siblings = match cached {
            Some(s) => s,
            None => {
                let mut reps: Vec<(usize, SExpr)> = Vec::new();
                for (i, v) in values.iter().enumerate() {
                    if let Some(s) = self.expr_or_null(v)?.as_ref().and_then(trap_skeleton) {
                        if !reps.iter().any(|(_, r)| *r == s) {
                            reps.push((i, s));
                        }
                    }
                }
                let s = Rc::new(reps);
                self.call_siblings.borrow_mut().insert(key, s.clone());
                s
            }
        };
        let answer = self.expr_or_null(&picked)?;
        let mut items = Vec::with_capacity(siblings.len() + 1);
        let mut at = None;
        for (i, s) in siblings.iter() {
            if *i == pick {
                // The read field traps where its skeleton would.
                continue;
            }
            if *i > pick && at.is_none() {
                at = Some(items.len());
                items.push(answer.clone().unwrap_or_else(|| null_of(Ty::I32)));
            }
            items.push(s.clone());
        }
        let at = match at {
            Some(a) => a,
            None => {
                items.push(answer.clone().unwrap_or_else(|| null_of(Ty::I32)));
                items.len() - 1
            }
        };
        if items.len() == 1 {
            return Ok(Some(answer));
        }
        if answer.is_none() {
            return Err(unsup(
                "a bare-NULL struct field read beside a field that can trap",
            ));
        }
        let v = &items[at];
        let (ty, nullable) = (v.ty, v.nullable);
        Ok(Some(Some(SExpr {
            kind: SKind::Seq { items, pick: at },
            ty,
            nullable,
        })))
    }
}
