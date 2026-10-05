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
    let [id] = marker_numbers(e, macros::CALL_MARKER, 1)?[..] else {
        return None;
    };
    let calls = CALLS.with(|c| c.borrow().clone());
    (id < calls.len()).then_some((id, calls))
}

/// The internal marker a field read over a call's guarded body reads its
/// `struct_pack` arms through: `__cf_call_pack(id, arm)`, arm `k` being the
/// k-th WHEN's result and `conditions.len()` the ELSE. Never written by a
/// user: `__cf_call` is a reserved prefix.
const PACK_MARKER: &str = "__cf_call_pack";

fn pack_marker(id: usize, arm: usize) -> SqlExpr {
    let num = |n: usize| {
        sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
            SqlExpr::Value(SqlValue::Number(n.to_string(), false).into()),
        ))
    };
    SqlExpr::Function(sqlparser::ast::Function {
        name: sqlparser::ast::ObjectName::from(vec![sqlparser::ast::Ident::new(PACK_MARKER)]),
        uses_odbc_syntax: false,
        parameters: sqlparser::ast::FunctionArguments::None,
        args: sqlparser::ast::FunctionArguments::List(sqlparser::ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![num(id), num(arm)],
            clauses: Vec::new(),
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: Vec::new(),
    })
}

/// The leading integer arguments of a `name(n1, .., nk, ...)` marker.
fn marker_numbers(e: &SqlExpr, name: &str, k: usize) -> Option<Vec<usize>> {
    let SqlExpr::Function(f) = e else {
        return None;
    };
    if !f.name.to_string().eq_ignore_ascii_case(name) {
        return None;
    }
    let sqlparser::ast::FunctionArguments::List(list) = &f.args else {
        return None;
    };
    let mut out = Vec::with_capacity(k);
    for a in list.args.iter().take(k) {
        let sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
            SqlExpr::Value(v),
        )) = a
        else {
            return None;
        };
        let SqlValue::Number(n, _) = &v.value else {
            return None;
        };
        out.push(n.parse().ok()?);
    }
    (out.len() == k).then_some(out)
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
/// `(call, arm, ...scope)`: arm `usize::MAX` is a body that is itself the
/// `struct_pack`.
pub(super) type ScopeKey = (usize, usize, usize, usize, u32, bool, usize);

/// One call's siblings in one scope: each distinct trap skeleton, at the
/// first field that has it.
pub(super) type Siblings = Rc<Vec<(usize, SExpr)>>;

impl Binder<'_> {
    fn scope_key(&self, id: usize, arm: usize) -> ScopeKey {
        (
            id,
            arm,
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
        let calls = CALLS.with(|c| c.borrow().clone());
        // A read through the pack marker: arm `arm` of call `id`'s CASE.
        if let Some([id, arm]) = marker_numbers(base, PACK_MARKER, 2).as_deref() {
            let (id, arm) = (*id, *arm);
            let Some(SqlExpr::Case {
                conditions,
                else_result,
                ..
            }) = calls.get(id).map(|c| unnest(&c.1))
            else {
                return Err(PrepareError::Internal("a pack marker off its call".into()));
            };
            let result = match conditions.get(arm) {
                Some(w) => &w.result,
                None => else_result.as_deref().ok_or_else(|| {
                    PrepareError::Internal("a pack marker past its CASE".into())
                })?,
            };
            let SqlExpr::Function(f) = unnest(result) else {
                return Err(PrepareError::Internal("a pack marker off a struct_pack".into()));
            };
            if let Some(read) = self.pack_field(id, arm, f, field, rest)? {
                return Ok(Some(read));
            }
            let read = SqlExpr::CompoundFieldAccess {
                root: Box::new(SqlExpr::Nested(Box::new(result.clone()))),
                access_chain: access_chain.clone(),
            };
            return self.expr_or_null(&read).map(Some);
        }
        let Some((id, calls)) = marker_call(base) else {
            return Ok(None);
        };
        let body = unnest(&calls[id].1);
        match body {
            SqlExpr::Function(f) => {
                if let Some(read) = self.pack_field(id, usize::MAX, f, field, rest)? {
                    return Ok(Some(read));
                }
            }
            // A guarded body (`CASE WHEN id IS NULL THEN NULL ELSE
            // struct_pack(..) END`, a call with `null_when`): the read moves
            // into each arm, as `case_field` moves it, and a `struct_pack`
            // arm reads through the marker, so its siblings bind once per
            // call too. Nothing of the body is copied but its conditions.
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                case_token,
                end_token,
            } => {
                let arm_read = |k: usize, r: &SqlExpr| -> SqlExpr {
                    match unnest(r) {
                        SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => r.clone(),
                        SqlExpr::Function(f) if is_struct_pack(f) => {
                            SqlExpr::CompoundFieldAccess {
                                root: Box::new(SqlExpr::Nested(Box::new(pack_marker(id, k)))),
                                access_chain: access_chain.clone(),
                            }
                        }
                        _ => SqlExpr::CompoundFieldAccess {
                            root: Box::new(SqlExpr::Nested(Box::new(r.clone()))),
                            access_chain: access_chain.clone(),
                        },
                    }
                };
                let read = SqlExpr::Case {
                    operand: operand.clone(),
                    conditions: conditions
                        .iter()
                        .enumerate()
                        .map(|(k, w)| sqlparser::ast::CaseWhen {
                            condition: w.condition.clone(),
                            result: arm_read(k, &w.result),
                        })
                        .collect(),
                    else_result: else_result
                        .as_deref()
                        .map(|e| Box::new(arm_read(conditions.len(), e))),
                    case_token: case_token.clone(),
                    end_token: end_token.clone(),
                };
                return self.expr_or_null(&read).map(Some);
            }
            _ => {}
        }
        // Any other body: read it as written.
        let read = SqlExpr::CompoundFieldAccess {
            root: Box::new(SqlExpr::Nested(Box::new(body.clone()))),
            access_chain: access_chain.clone(),
        };
        self.expr_or_null(&read).map(Some)
    }

    /// Field `field` (then `rest`) of the `struct_pack` `f`, arm `arm` of
    /// call `id`: the read field and the call's cached sibling skeletons.
    /// `Ok(None)` when `f` is not a plain `struct_pack`.
    fn pack_field(
        &self,
        id: usize,
        arm: usize,
        f: &sqlparser::ast::Function,
        field: &sqlparser::ast::Ident,
        rest: &[AccessExpr],
    ) -> Result<Option<Option<SExpr>>, PrepareError> {
        let Some((values, pick)) = self.struct_pack_values(f, &field.value)? else {
            return Ok(None);
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
        // Siblings that read no lateral alias bind alike whichever aliases
        // are bound, so they are kept under the alias-free key and reused by
        // every later item; without that, n reads of n lanes bound n^2.
        let key = self.scope_key(id, arm);
        let free_key = (key.0, key.1, key.2, usize::MAX, key.4, key.5, key.6);
        let cached = {
            let cache = self.call_siblings.borrow();
            match cache.get(&free_key) {
                Some(s) => Some(s.clone()),
                None => cache.get(&key).cloned().inspect(|_| {
                    // Reused, it still read an alias: a call around this one
                    // must not take the alias-free key either.
                    self.alias_reads.set(self.alias_reads.get() + 1);
                }),
            }
        };
        let siblings: Siblings = match cached {
            Some(s) => s,
            None => {
                let reads = self.alias_reads.get();
                let mut reps: Vec<(usize, SExpr)> = Vec::new();
                for (i, v) in values.iter().enumerate() {
                    if let Some(s) = self.expr_or_null(v)?.as_ref().and_then(trap_skeleton) {
                        if !reps.iter().any(|(_, r)| *r == s) {
                            reps.push((i, s));
                        }
                    }
                }
                let s = Rc::new(reps);
                let key = if self.alias_reads.get() == reads { free_key } else { key };
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

fn unnest(mut e: &SqlExpr) -> &SqlExpr {
    while let SqlExpr::Nested(i) = e {
        e = i;
    }
    e
}

fn is_struct_pack(f: &sqlparser::ast::Function) -> bool {
    f.name.to_string().eq_ignore_ascii_case("struct_pack")
}
