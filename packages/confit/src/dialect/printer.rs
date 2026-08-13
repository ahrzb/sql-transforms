//! The dialect-independent half of printing: query shapes and column
//! references. Each dialect supplies identifier quoting and expression
//! spelling through [`ExprPrinter`]; the SELECT/FROM/WHERE skeleton, ribbon
//! flattening, join FROM rendering, and the duplicate-name refusal are the
//! same decision in every dialect and live here once.
//!
//! Column references are ordinal-addressed: every expression prints against
//! a per-ordinal table of pre-rendered SQL refs ([`ColRef`]). Name-addressed
//! sources (a subquery boundary, a bare table) mark duplicate names
//! unresolvable — referencing one refuses, same words as before. Join sides
//! get deterministic aliases (`__cf_jN`), so cross-side duplicates are
//! simply unambiguous (2026-08-13-dialect-join-node-design.md, approach A).

use super::plan::{Catalog, Expr, JoinKind, Rel};
use super::ty::DTy;
use super::{unsup, DialectError};

/// One input column as the current dialect spells it. `sql: None` means the
/// column cannot be addressed unambiguously here (duplicate name at a
/// name-addressed boundary) — referencing it refuses by name.
pub(crate) struct ColRef {
    pub name: String,
    pub sql: Option<String>,
}

pub(crate) trait ExprPrinter {
    fn quote_ident(&self, name: &str) -> String;
    fn expr(&self, e: &Expr, input: &[ColRef]) -> Result<String, DialectError>;
}

/// Refs for a name-addressed source: the bound spelling, quoted; duplicate
/// names (case-insensitive, DuckDB identifier semantics) are unresolvable.
pub(crate) fn name_refs<P: ExprPrinter + ?Sized>(
    p: &P,
    schema: &[(String, DTy)],
) -> Vec<ColRef> {
    schema
        .iter()
        .map(|(n, _)| {
            let dups = schema
                .iter()
                .filter(|(m, _)| m.eq_ignore_ascii_case(n))
                .count();
            ColRef {
                name: n.clone(),
                sql: (dups == 1).then(|| p.quote_ident(n)),
            }
        })
        .collect()
}

/// Refs for an aliased join side: `alias.col`. Duplicates WITHIN the side
/// (a dup-name Project used as a join input) stay unresolvable — the alias
/// cannot disambiguate those.
fn qualified_refs<P: ExprPrinter + ?Sized>(
    p: &P,
    alias: &str,
    schema: &[(String, DTy)],
) -> Vec<ColRef> {
    schema
        .iter()
        .map(|(n, _)| {
            let dups = schema
                .iter()
                .filter(|(m, _)| m.eq_ignore_ascii_case(n))
                .count();
            ColRef {
                name: n.clone(),
                sql: (dups == 1)
                    .then(|| format!("{}.{}", p.quote_ident(alias), p.quote_ident(n))),
            }
        })
        .collect()
}

/// A column reference prints as its pre-rendered ref, or refuses.
pub(crate) fn col_ref(
    ordinal: usize,
    name: &str,
    input: &[ColRef],
) -> Result<String, DialectError> {
    let Some(r) = input.get(ordinal) else {
        return Err(DialectError::Internal(format!(
            "col ordinal {ordinal} out of range for printer input arity {}",
            input.len()
        )));
    };
    r.sql.clone().ok_or_else(|| {
        unsup(format!(
            "printing a column reference over duplicate upstream names: {name}"
        ))
    })
}

pub(crate) fn refuse_dup_names(schema: &[(String, DTy)]) -> Result<(), DialectError> {
    for (i, (n, _)) in schema.iter().enumerate() {
        if schema[..i].iter().any(|(m, _)| m.eq_ignore_ascii_case(n)) {
            return Err(unsup(format!(
                "printing a pass-through over duplicate upstream names: {n}"
            )));
        }
    }
    Ok(())
}

fn join_kw(kind: JoinKind) -> &'static str {
    match kind {
        JoinKind::Inner => "INNER JOIN",
        JoinKind::Left => "LEFT JOIN",
        JoinKind::Right => "RIGHT JOIN",
        JoinKind::Full => "FULL JOIN",
        JoinKind::Cross => "CROSS JOIN",
    }
}

/// Render a join tree as a FROM clause: `x AS __cf_j0 INNER JOIN y AS
/// __cf_j1 ON ...`. Left-nested joins flatten into one chain; every ON is
/// printed over the refs of everything joined so far (DuckDB scope
/// semantics, which the frontend also binds). Returns the FROM text and the
/// combined refs/schema.
fn join_from<P: ExprPrinter + ?Sized>(
    p: &P,
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
    next_alias: &mut usize,
) -> Result<(String, Vec<ColRef>, Vec<(String, DTy)>), DialectError> {
    match rel {
        Rel::Join {
            left,
            right,
            kind,
            on,
        } => {
            let (l_sql, mut refs, mut schema) = join_from(p, left, cat, depth, next_alias)?;
            let (r_sql, r_refs, r_schema) = leaf_from(p, right, cat, depth, next_alias)?;
            refs.extend(r_refs);
            schema.extend(r_schema);
            let on_sql = match on {
                Some(pred) => format!(" ON {}", p.expr(pred, &refs)?),
                None => String::new(),
            };
            Ok((
                format!("{l_sql} {} {r_sql}{on_sql}", join_kw(*kind)),
                refs,
                schema,
            ))
        }
        _ => leaf_from(p, rel, cat, depth, next_alias),
    }
}

/// One aliased join side: a base table directly, anything else as a
/// subquery. (The frontend only produces Scan sides; text fixtures may
/// nest — the subquery path covers them, right-nested joins included.)
fn leaf_from<P: ExprPrinter + ?Sized>(
    p: &P,
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
    next_alias: &mut usize,
) -> Result<(String, Vec<ColRef>, Vec<(String, DTy)>), DialectError> {
    let alias = format!("__cf_j{}", *next_alias);
    *next_alias += 1;
    let (item, schema) = match rel {
        Rel::Scan { table } => (
            format!("{} AS {}", p.quote_ident(table), p.quote_ident(&alias)),
            rel.schema(cat)?,
        ),
        other => {
            let (inner, schema) = query(p, other, cat, depth + 1)?;
            (
                format!("({inner}) AS {}", p.quote_ident(&alias)),
                schema,
            )
        }
    };
    let refs = qualified_refs(p, &alias, &schema);
    Ok((item, refs, schema))
}

/// Print a relation as a full query. Ribbon shapes — project over
/// filter?(scan | join-tree) and filter(scan) — flatten to one SELECT (the
/// frontend∘printer fixpoint depends on this in the DuckDB dialect);
/// everything else nests.
pub(crate) fn query<P: ExprPrinter + ?Sized>(
    p: &P,
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
) -> Result<(String, Vec<(String, DTy)>), DialectError> {
    match rel {
        Rel::Project { input, items } => {
            if let Some((from, refs, where_)) = as_ribbon(p, input, cat, depth)? {
                let mut outs = Vec::new();
                for (name, e) in items {
                    outs.push(format!(
                        "{} AS {}",
                        p.expr(e, &refs)?,
                        p.quote_ident(name)
                    ));
                }
                return Ok((
                    format!("SELECT {} FROM {from}{where_}", outs.join(", ")),
                    rel.schema(cat)?,
                ));
            }
            let (inner, in_schema) = query(p, input, cat, depth + 1)?;
            let refs = name_refs(p, &in_schema);
            let mut outs = Vec::new();
            for (name, e) in items {
                outs.push(format!(
                    "{} AS {}",
                    p.expr(e, &refs)?,
                    p.quote_ident(name)
                ));
            }
            Ok((
                format!(
                    "SELECT {} FROM ({inner}) AS {}",
                    outs.join(", "),
                    p.quote_ident(&format!("__cf_q{depth}"))
                ),
                rel.schema(cat)?,
            ))
        }
        Rel::Filter { .. } | Rel::Join { .. } => {
            if let Some((from, refs, where_)) = as_ribbon(p, rel, cat, depth)? {
                // A pass-through SELECT list: every input column, re-aliased
                // to its bound name (duplicates are legal as OUTPUT names;
                // an unresolvable ref refuses in col_ref's words).
                let schema = rel.schema(cat)?;
                let mut outs = Vec::new();
                for (i, (n, _)) in schema.iter().enumerate() {
                    let r = col_ref(i, n, &refs)?;
                    outs.push(format!("{r} AS {}", p.quote_ident(n)));
                }
                return Ok((
                    format!("SELECT {} FROM {from}{where_}", outs.join(", ")),
                    schema,
                ));
            }
            let Rel::Filter { input, pred } = rel else {
                return Err(DialectError::Internal(
                    "non-ribbon join fell through".into(),
                ));
            };
            let (inner, in_schema) = query(p, input, cat, depth + 1)?;
            refuse_dup_names(&in_schema)?;
            let refs = name_refs(p, &in_schema);
            let pred = p.expr(pred, &refs)?;
            let cols: Vec<String> = in_schema.iter().map(|(n, _)| p.quote_ident(n)).collect();
            Ok((
                format!(
                    "SELECT {} FROM ({inner}) AS {} WHERE {pred}",
                    cols.join(", "),
                    p.quote_ident(&format!("__cf_q{depth}"))
                ),
                rel.schema(cat)?,
            ))
        }
        Rel::Scan { table } => {
            let schema = rel.schema(cat)?;
            let cols: Vec<String> = schema.iter().map(|(n, _)| p.quote_ident(n)).collect();
            Ok((
                format!("SELECT {} FROM {}", cols.join(", "), p.quote_ident(table)),
                schema,
            ))
        }
    }
}

/// Flatten a source shape into (FROM text, refs, WHERE text): scan,
/// filter(scan), join-tree, filter(join-tree). Anything else — None, the
/// caller nests a subquery.
#[allow(clippy::type_complexity)]
fn as_ribbon<P: ExprPrinter + ?Sized>(
    p: &P,
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
) -> Result<Option<(String, Vec<ColRef>, String)>, DialectError> {
    let (source, pred) = match rel {
        Rel::Filter { input, pred } => (input.as_ref(), Some(pred)),
        other => (other, None),
    };
    let (from, refs) = match source {
        Rel::Scan { table } => {
            let schema = source.schema(cat)?;
            (p.quote_ident(table), name_refs(p, &schema))
        }
        Rel::Join { .. } => {
            let mut next_alias = 0usize;
            let (from, refs, _) = join_from(p, source, cat, depth, &mut next_alias)?;
            (from, refs)
        }
        _ => return Ok(None),
    };
    let where_ = match pred {
        Some(w) => format!(" WHERE {}", p.expr(w, &refs)?),
        None => String::new(),
    };
    Ok(Some((from, refs, where_)))
}
