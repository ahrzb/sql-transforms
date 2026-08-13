//! The plan verifier — mandatory property 1 of the ir/ recipe.
//!
//! A plan that verifies against a catalog is one every printer may trust:
//! every Col ordinal resolves, its carried name and type match the catalog
//! (case-insensitive name match, DuckDB identifier semantics), every
//! expression's type derives, every Filter predicate is BOOLEAN, every
//! relation has a schema. Frontends verify before returning; fixtures
//! verify after text-parsing — an unverifiable plan is [`DialectError::
//! Internal`] out of a frontend and a fixture bug out of text.

use super::plan::{Catalog, Expr, Rel};
use super::ty::DTy;
use super::DialectError;

pub fn verify(rel: &Rel, cat: &Catalog) -> Result<(), DialectError> {
    walk(rel, cat)?;
    Ok(())
}

/// Verify the node and return its input schema for expression checking.
fn walk(rel: &Rel, cat: &Catalog) -> Result<Vec<(String, DTy)>, DialectError> {
    match rel {
        Rel::Scan { .. } => rel.schema(cat),
        Rel::Filter { input, pred } => {
            let in_schema = walk(input, cat)?;
            check_expr(pred, &in_schema)?;
            if pred.ty()? != DTy::Bool {
                return Err(DialectError::Internal(format!(
                    "filter predicate is {}, not bool",
                    pred.ty()?.name()
                )));
            }
            Ok(in_schema)
        }
        Rel::Project { input, items } => {
            let in_schema = walk(input, cat)?;
            if items.is_empty() {
                return Err(DialectError::Internal("empty projection".into()));
            }
            for (_, e) in items {
                check_expr(e, &in_schema)?;
            }
            rel.schema(cat)
        }
    }
}

fn check_expr(e: &Expr, input: &[(String, DTy)]) -> Result<(), DialectError> {
    // Every carried type must re-derive — this is the whole check for
    // interior nodes, since they carry nothing.
    e.ty()?;
    each_col(e, &mut |ordinal, name, ty| {
        let Some((bound_name, bound_ty)) = input.get(ordinal) else {
            return Err(DialectError::Internal(format!(
                "col ordinal {ordinal} out of range for input arity {}",
                input.len()
            )));
        };
        if !bound_name.eq_ignore_ascii_case(name) {
            return Err(DialectError::Internal(format!(
                "col {ordinal} carries name {name}, input says {bound_name}"
            )));
        }
        if bound_ty != ty {
            return Err(DialectError::Internal(format!(
                "col {ordinal} ({name}) carries type {}, input says {}",
                ty.name(),
                bound_ty.name()
            )));
        }
        Ok(())
    })
}

fn each_col(
    e: &Expr,
    f: &mut impl FnMut(usize, &str, &DTy) -> Result<(), DialectError>,
) -> Result<(), DialectError> {
    match e {
        Expr::Col { ordinal, name, ty } => f(*ordinal, name, ty),
        Expr::Lit { .. } => Ok(()),
        Expr::Bin { l, r, .. } | Expr::IsDistinct { l, r, .. } => {
            each_col(l, f)?;
            each_col(r, f)
        }
        Expr::Un { e, .. } | Expr::Cast { e, .. } | Expr::IsNull { e, .. } => each_col(e, f),
        Expr::Case { whens, else_ } => {
            for (c, v) in whens {
                each_col(c, f)?;
                each_col(v, f)?;
            }
            if let Some(el) = else_ {
                each_col(el, f)?;
            }
            Ok(())
        }
    }
}
