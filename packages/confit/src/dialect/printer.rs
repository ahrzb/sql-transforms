//! The dialect-independent half of printing: query shapes and column
//! references. Each dialect supplies identifier quoting and expression
//! spelling through [`ExprPrinter`]; the SELECT/FROM/WHERE skeleton,
//! ribbon flattening, and the duplicate-upstream-name refusal are the
//! same decision in every dialect and live here once.

use super::plan::{Catalog, Expr, Rel};
use super::ty::DTy;
use super::{unsup, DialectError};

pub(crate) trait ExprPrinter {
    fn quote_ident(&self, name: &str) -> String;
    fn expr(&self, e: &Expr, input: &[(String, DTy)]) -> Result<String, DialectError>;
}

/// Any name-addressed pass-through of a schema (a Filter's SELECT list)
/// silently rebinds duplicates to the FIRST occurrence downstream — refuse
/// duplicate names wherever a printed list re-reads columns by name.
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

/// A column reference prints as the input schema's bound spelling — but a
/// name-addressed subquery boundary cannot express duplicates, so those
/// refuse identically everywhere.
pub(crate) fn col_ref<P: ExprPrinter>(
    p: &P,
    ordinal: usize,
    name: &str,
    input: &[(String, DTy)],
) -> Result<String, DialectError> {
    let dups = input
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case(name))
        .count();
    if dups != 1 {
        return Err(unsup(format!(
            "printing a column reference over duplicate upstream names: {name}"
        )));
    }
    let (bound_name, _) = &input[ordinal];
    Ok(p.quote_ident(bound_name))
}

/// Print a relation as a full query. Ribbon shapes — project(filter?(scan))
/// and filter(scan) — flatten to one SELECT (the frontend∘printer fixpoint
/// depends on this in the DuckDB dialect); everything else nests.
pub(crate) fn query<P: ExprPrinter>(
    p: &P,
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
) -> Result<(String, Vec<(String, DTy)>), DialectError> {
    match rel {
        Rel::Project { input, items } => {
            if let Some((table, pred)) = as_ribbon(input) {
                let in_schema = Rel::Scan {
                    table: table.clone(),
                }
                .schema(cat)?;
                let mut outs = Vec::new();
                for (name, e) in items {
                    outs.push(format!(
                        "{} AS {}",
                        p.expr(e, &in_schema)?,
                        p.quote_ident(name)
                    ));
                }
                let where_ = match pred {
                    Some(w) => format!(" WHERE {}", p.expr(w, &in_schema)?),
                    None => String::new(),
                };
                return Ok((
                    format!(
                        "SELECT {} FROM {}{where_}",
                        outs.join(", "),
                        p.quote_ident(&table)
                    ),
                    rel.schema(cat)?,
                ));
            }
            let (inner, in_schema) = query(p, input, cat, depth + 1)?;
            let mut outs = Vec::new();
            for (name, e) in items {
                outs.push(format!(
                    "{} AS {}",
                    p.expr(e, &in_schema)?,
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
        Rel::Filter { input, pred } => {
            if let Rel::Scan { table } = input.as_ref() {
                let in_schema = input.schema(cat)?;
                refuse_dup_names(&in_schema)?;
                let cols: Vec<String> = in_schema.iter().map(|(n, _)| p.quote_ident(n)).collect();
                return Ok((
                    format!(
                        "SELECT {} FROM {} WHERE {}",
                        cols.join(", "),
                        p.quote_ident(table),
                        p.expr(pred, &in_schema)?
                    ),
                    rel.schema(cat)?,
                ));
            }
            let (inner, in_schema) = query(p, input, cat, depth + 1)?;
            refuse_dup_names(&in_schema)?;
            let pred = p.expr(pred, &in_schema)?;
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

fn as_ribbon(input: &Rel) -> Option<(String, Option<&Expr>)> {
    match input {
        Rel::Scan { table } => Some((table.clone(), None)),
        Rel::Filter { input, pred } => match input.as_ref() {
            Rel::Scan { table } => Some((table.clone(), Some(pred))),
            _ => None,
        },
        _ => None,
    }
}
