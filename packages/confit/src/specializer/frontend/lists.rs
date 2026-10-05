//! List literals, `[e1, ..., ek]` and `list_value(e1, ..., ek)`, in the two
//! places a row-local list is served: read by a constant index, and
//! projected whole at the top level (the same wide-lane boundary as an
//! extern's unnamed list output).
//!
//! DuckDB builds every element (`[a, a * 9223372036854775807][1]` traps,
//! measured on 1.5.5), indexes from 1, counts a negative index from the end,
//! and answers NULL past either end or at 0. Elements must bind to one type:
//! DuckDB unifies mixed element types, which is refused here rather than
//! approximated.

use super::*;

/// The elements of a list literal, or `None` when `e` is not one.
pub(super) fn list_literal(e: &SqlExpr) -> Option<Vec<&SqlExpr>> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
    let mut e = e;
    while let SqlExpr::Nested(i) = e {
        e = i;
    }
    match e {
        SqlExpr::Array(a) if !a.named => Some(a.elem.iter().collect()),
        SqlExpr::Function(f) if f.name.to_string().eq_ignore_ascii_case("list_value") => {
            let FunctionArguments::List(list) = &f.args else {
                return None;
            };
            if list.duplicate_treatment.is_some()
                || !list.clauses.is_empty()
                || f.over.is_some()
                || f.filter.is_some()
            {
                return None;
            }
            list.args
                .iter()
                .map(|a| match a {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(x)) => Some(x),
                    _ => None,
                })
                .collect()
        }
        _ => None,
    }
}

impl Binder<'_> {
    /// Every element bound, all of one type; a bare NULL element takes it.
    pub(super) fn list_elements(&self, elems: &[&SqlExpr]) -> Result<Vec<SExpr>, PrepareError> {
        let bound: Vec<Option<SExpr>> = elems
            .iter()
            .map(|x| self.expr_or_null(x))
            .collect::<Result<_, _>>()?;
        let Some(ty) = bound.iter().flatten().map(|x| x.ty).next() else {
            return Err(unsup("a list of only NULLs"));
        };
        if let Some(other) = bound.iter().flatten().find(|x| x.ty != ty) {
            return Err(unsup(format!(
                "list elements of different types ({} and {}): DuckDB unifies them; \
                 cast the elements to one type",
                duck_ty_name(ty),
                duck_ty_name(other.ty)
            )));
        }
        Ok(bound
            .into_iter()
            .map(|x| x.unwrap_or_else(|| null_of(ty)))
            .collect())
    }

    /// `list[index]` over a list literal: the element the index names, with
    /// every element that can trap still evaluated, in order.
    pub(super) fn list_element(
        &self,
        elems: &[&SqlExpr],
        index: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let Some(i) = ast_int_literal(index).and_then(|v| i64::try_from(v).ok()) else {
            return Err(unsup("a list index that is not an integer constant"));
        };
        let bound = self.list_elements(elems)?;
        let k = bound.len() as i64;
        let pos = match i {
            1.. if i <= k => Some((i - 1) as usize),
            ..=-1 if i >= -k => Some((k + i) as usize),
            _ => None,
        };
        let ty = bound[0].ty;
        let mut items = Vec::with_capacity(bound.len() + 1);
        let mut pick = None;
        for (j, x) in bound.into_iter().enumerate() {
            if Some(j) == pos {
                pick = Some(items.len());
                items.push(x);
            } else if can_trap(&x) {
                items.push(x);
            }
        }
        let pick = pick.unwrap_or_else(|| {
            items.push(null_of(ty));
            items.len() - 1
        });
        if items.len() == 1 {
            return Ok(items.pop().expect("one item"));
        }
        let (ty, nullable) = (items[pick].ty, items[pick].nullable);
        Ok(SExpr {
            kind: SKind::Seq { items, pick },
            ty,
            nullable,
        })
    }
}
