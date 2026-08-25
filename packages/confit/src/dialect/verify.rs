//! The plan verifier — mandatory property 1 of the ir/ recipe.
//!
//! A plan that verifies against a catalog is one every printer may trust:
//! every Col ordinal resolves, its carried name and type match the catalog
//! (case-insensitive name match, DuckDB identifier semantics), every
//! expression's type derives, every Filter predicate is BOOLEAN, every
//! relation has a schema. Frontends verify before returning; fixtures
//! verify after text-parsing — an unverifiable plan is [`DialectError::
//! Internal`] out of a frontend and a fixture bug out of text.

use super::plan::{Catalog, Expr, JoinKind, Rel};
use super::ty::DTy;
use super::DialectError;

/// Check `rel` against `cat`; `Ok(())` is the guarantee every printer relies
/// on. A structural fault is [`DialectError::Internal`], an unknown table is
/// [`DialectError::Bind`], and a type combination the lattice has not pinned
/// surfaces as [`DialectError::Unsupported`] out of derivation.
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
        Rel::Join {
            left,
            right,
            kind,
            on,
        } => {
            let mut combined = walk(left, cat)?;
            combined.extend(walk(right, cat)?);
            match (kind, on) {
                (JoinKind::Cross, Some(_)) => {
                    return Err(DialectError::Internal("CROSS join carries an ON".into()));
                }
                (JoinKind::Cross, None) => {}
                (_, None) => {
                    return Err(DialectError::Internal(format!(
                        "{} join without an ON",
                        kind.name()
                    )));
                }
                (_, Some(pred)) => {
                    check_expr(pred, &combined)?;
                    if pred.ty()? != DTy::Bool {
                        return Err(DialectError::Internal(format!(
                            "join ON is {}, not bool",
                            pred.ty()?.name()
                        )));
                    }
                }
            }
            Ok(combined)
        }
    }
}

fn check_expr(e: &Expr, input: &[(String, DTy)]) -> Result<(), DialectError> {
    // Every carried type must re-derive — this is the whole check for
    // interior nodes, since they carry nothing.
    e.ty()?;
    check_lexemes_and_types(e)?;
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
        Expr::Call { args, .. } => {
            for a in args {
                each_col(a, f)?;
            }
            Ok(())
        }
    }
}

/// Printers splice numeric/bool LEXEMES into SQL verbatim (string lexemes
/// are escaped, so they are safe as data). A lexeme that is not a plain
/// number is therefore arbitrary SQL smuggled through a "verified" plan —
/// refuse it here, not in every printer. Carried types must also be
/// well-formed (Dec bounds, struct field names) or no text form can
/// round-trip them.
fn check_lexemes_and_types(e: &Expr) -> Result<(), DialectError> {
    fn numeric_lexeme(s: &str) -> bool {
        // digits [. digits] [e|E [+|-] digits] — the exact shape the
        // frontend's number_type admits. No sign: negation is Un{Neg}.
        let mut rest = s;
        let digits = |r: &mut &str| {
            let n = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            *r = &r[n..];
            n > 0
        };
        if !digits(&mut rest) {
            return false;
        }
        if let Some(r) = rest.strip_prefix('.') {
            rest = r;
            if !digits(&mut rest) {
                return false;
            }
        }
        if let Some(r) = rest.strip_prefix(['e', 'E']) {
            rest = r.strip_prefix(['+', '-']).unwrap_or(r);
            if !digits(&mut rest) {
                return false;
            }
        }
        rest.is_empty()
    }
    match e {
        Expr::Lit { lexeme, ty } => {
            let ok = match ty {
                DTy::Bool => lexeme == "true" || lexeme == "false",
                DTy::Str => true, // escaped at print time
                t if t.is_numeric() => numeric_lexeme(lexeme),
                // Other types print as quoted-and-escaped strings + cast.
                _ => true,
            };
            if !ok {
                return Err(DialectError::Internal(format!(
                    "lexeme {lexeme:?} is not a bare {} literal",
                    ty.name()
                )));
            }
            if !ty.is_well_formed() {
                return Err(DialectError::Internal(format!(
                    "malformed literal type {}",
                    ty.name()
                )));
            }
            Ok(())
        }
        Expr::Col { ty, .. } => {
            if !ty.is_well_formed() {
                return Err(DialectError::Internal(format!(
                    "malformed column type {}",
                    ty.name()
                )));
            }
            Ok(())
        }
        Expr::Cast { e, target, .. } => {
            if !target.is_well_formed() {
                return Err(DialectError::Internal(format!(
                    "malformed CAST target {}",
                    target.name()
                )));
            }
            check_lexemes_and_types(e)
        }
        Expr::Bin { l, r, .. } | Expr::IsDistinct { l, r, .. } => {
            check_lexemes_and_types(l)?;
            check_lexemes_and_types(r)
        }
        Expr::Un { e, .. } | Expr::IsNull { e, .. } => check_lexemes_and_types(e),
        Expr::Case { whens, else_ } => {
            for (c, v) in whens {
                check_lexemes_and_types(c)?;
                check_lexemes_and_types(v)?;
            }
            if let Some(el) = else_ {
                check_lexemes_and_types(el)?;
            }
            Ok(())
        }
        Expr::Call { args, .. } => {
            for a in args {
                check_lexemes_and_types(a)?;
            }
            Ok(())
        }
    }
}
