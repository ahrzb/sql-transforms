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
    pub(super) fn new(calls: Vec<(String, SqlExpr)>) -> Self {
        let prev = CALLS.with(|c| c.replace(Rc::new(calls)));
        Installed(prev)
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

/// The marker's arguments as the call spelled them, for its name.
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

/// What a read's siblings bind to depends on the scope it binds in; the
/// cache is keyed by this as well as the call.
pub(super) type ScopeKey = (usize, usize, usize, u32, bool, usize);

/// One call's siblings in one scope: each distinct trap skeleton, at the
/// first field that has it.
pub(super) type Siblings = Rc<Vec<(usize, SExpr)>>;

impl Binder<'_> {
    fn scope_key(&self, id: usize) -> ScopeKey {
        (
            id,
            self.joins.len(),
            self.bound_aliases.borrow().len(),
            self.in_guarded.get(),
            self.classify_keys.get(),
            self.beside.borrow().len(),
        )
    }

    /// A field read whose root is a call marker: `Ok(None)` when `e` is not
    /// one. Over a call whose expansion is a plain `struct_pack`, binds the
    /// read field and the cached sibling skeletons; any other expansion (a
    /// CASE, say) reads as the expansion itself would.
    pub(super) fn call_field(&self, e: &SqlExpr) -> Result<Option<Option<SExpr>>, PrepareError> {
        let SqlExpr::CompoundFieldAccess { root, access_chain } = e else {
            return Ok(None);
        };
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
        let key = self.scope_key(id);
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
