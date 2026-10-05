//! DuckDB's name for an unaliased projection item: the parsed expression
//! printed the way DuckDB prints it (`ParsedExpression::ToString` after its
//! transformer), measured on 1.5.5. A bare column is named after its last
//! segment; anything else after its printed form: operators parenthesized,
//! `true` as `CAST('t' AS BOOLEAN)`, a double literal by its shortest
//! value, CASE as `CASE  WHEN (c) THEN (r) ELSE e END`, keyword-named
//! functions and identifiers quoted, and so on. `None` is a form not
//! modeled here; the caller keeps the SQL text for it.

use super::*;
use sqlparser::ast::{
    BinaryOperator as B, CastKind, DataType, FunctionArg, FunctionArgExpr, FunctionArguments,
};

/// DuckDB's keywords (`duckdb_keywords()`, every category): an identifier
/// or function name spelled as one prints quoted.
const KEYWORDS: &[&str] = &[
    "abort", "absolute", "access", "action", "add", "admin", "after", "aggregate", "all", "also",
    "alter", "always", "analyse", "analyze", "and", "anti", "any", "array", "as", "asc", "asof",
    "assertion", "assignment", "asymmetric", "at", "attach", "attribute", "authorization",
    "backward", "before", "begin", "between", "bigint", "binary", "bit", "boolean", "both", "by",
    "cache", "call", "called", "cascade", "cascaded", "case", "cast", "catalog", "centuries",
    "century", "chain", "char", "character", "characteristics", "check", "checkpoint", "class",
    "close", "cluster", "coalesce", "collate", "collation", "column", "columns", "comment",
    "comments", "commit", "committed", "compression", "concurrently", "configuration", "conflict",
    "connection", "constraint", "constraints", "content", "continue", "conversion", "copy", "cost",
    "create", "cross", "csv", "cube", "current", "cursor", "cycle", "data", "database", "day",
    "days", "deallocate", "dec", "decade", "decades", "decimal", "declare", "default", "defaults",
    "deferrable", "deferred", "definer", "delete", "delimiter", "delimiters", "depends", "desc",
    "describe", "detach", "dictionary", "disable", "discard", "distinct", "do", "document",
    "domain", "double", "drop", "each", "else", "enable", "encoding", "encrypted", "end", "enum",
    "error", "escape", "event", "except", "exclude", "excluding", "exclusive", "execute", "exists",
    "explain", "export", "export_state", "extension", "extensions", "external", "extract", "false",
    "family", "fetch", "filter", "first", "float", "following", "for", "force", "foreign",
    "forward", "freeze", "from", "full", "function", "functions", "generated", "glob", "global",
    "grant", "granted", "group", "grouping", "grouping_id", "groups", "handler", "having",
    "header", "hold", "hour", "hours", "identity", "if", "ignore", "ilike", "immediate",
    "immutable", "implicit", "import", "in", "include", "including", "increment", "index",
    "indexes", "inherit", "inherits", "initially", "inline", "inner", "inout", "input",
    "insensitive", "insert", "install", "instead", "int", "integer", "intersect", "interval",
    "into", "invoker", "is", "isnull", "isolation", "join", "json", "key", "label", "lambda",
    "language", "large", "last", "lateral", "leading", "leakproof", "left", "level", "like",
    "limit", "listen", "load", "local", "location", "lock", "locked", "logged", "macro", "map",
    "mapping", "match", "matched", "materialized", "maxvalue", "merge", "method", "microsecond",
    "microseconds", "millennia", "millennium", "millisecond", "milliseconds", "minute", "minutes",
    "minvalue", "mode", "month", "months", "move", "name", "names", "national", "natural", "nchar",
    "new", "next", "no", "none", "not", "nothing", "notify", "notnull", "nowait", "null", "nullif",
    "nulls", "numeric", "object", "of", "off", "offset", "oids", "old", "on", "only", "operator",
    "option", "options", "or", "order", "ordinality", "others", "out", "outer", "over", "overlaps",
    "overlay", "overriding", "owned", "owner", "parallel", "parser", "partial", "partition",
    "partitioned", "passing", "password", "percent", "persistent", "pivot", "pivot_longer",
    "pivot_wider", "placing", "plans", "policy", "position", "positional", "pragma", "preceding",
    "precision", "prepare", "prepared", "preserve", "primary", "prior", "privileges", "procedural",
    "procedure", "program", "publication", "qualify", "quarter", "quarters", "quote", "range",
    "read", "real", "reassign", "recheck", "recursive", "ref", "references", "referencing",
    "refresh", "reindex", "relative", "release", "rename", "repeatable", "replace", "replica",
    "reset", "respect", "restart", "restrict", "returning", "returns", "revoke", "right", "role",
    "rollback", "rollup", "row", "rows", "rule", "sample", "savepoint", "schema", "schemas",
    "scope", "scroll", "search", "second", "seconds", "secret", "security", "select", "semi",
    "sequence", "sequences", "serializable", "server", "session", "set", "setof", "sets", "share",
    "show", "similar", "simple", "skip", "smallint", "snapshot", "some", "sorted", "source", "sql",
    "stable", "standalone", "start", "statement", "statistics", "stdin", "stdout", "storage",
    "stored", "strict", "strip", "struct", "subscription", "substring", "summarize", "symmetric",
    "sysid", "system", "table", "tables", "tablesample", "tablespace", "target", "temp",
    "template", "temporary", "text", "then", "ties", "time", "timestamp", "to", "trailing",
    "transaction", "transform", "treat", "trigger", "trim", "true", "truncate", "trusted",
    "try_cast", "type", "types", "unbounded", "uncommitted", "unencrypted", "union", "unique",
    "unknown", "unlisten", "unlogged", "unpack", "unpivot", "until", "update", "use", "user",
    "using", "vacuum", "valid", "validate", "validator", "value", "values", "varchar", "variable",
    "variadic", "varying", "verbose", "version", "view", "views", "virtual", "volatile", "week",
    "weeks", "when", "where", "whitespace", "window", "with", "within", "without", "work",
    "wrapper", "write", "xml", "xmlattributes", "xmlconcat", "xmlelement", "xmlexists",
    "xmlforest", "xmlnamespaces", "xmlparse", "xmlpi", "xmlroot", "xmlserialize", "xmltable",
    "year", "years", "yes", "zone",];

fn is_keyword(s: &str) -> bool {
    KEYWORDS.binary_search(&s.to_ascii_lowercase().as_str()).is_ok()
}

/// An identifier as DuckDB writes it: bare when it is a plain ASCII word
/// that is not a keyword, double-quoted otherwise.
fn ident(s: &str) -> String {
    let plain = s
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain && !is_keyword(s) {
        s.to_string()
    } else {
        format!("\"{}\"", s.replace('"', "\"\""))
    }
}

fn quote_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// A double as Python's `repr` writes it, which is DuckDB's rendering:
/// the shortest round-trip digits, fixed notation for exponents -4..16 with
/// at least one decimal, scientific beyond with a two-digit exponent.
pub(super) fn double_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sci = format!("{v:e}");
    let (mant, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let body = if exp >= 0 {
            let e = exp as usize;
            let int: String = if digits.len() > e + 1 {
                digits[..=e].to_string()
            } else {
                format!("{digits:0<width$}", width = e + 1)
            };
            let frac = if digits.len() > e + 1 { &digits[e + 1..] } else { "0" };
            format!("{int}.{frac}")
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let (d0, rest) = digits.split_at(1);
        let m = if rest.is_empty() { d0.to_string() } else { format!("{d0}.{rest}") };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// A numeric literal, possibly under unary minus (DuckDB folds the sign
/// into the constant), as its printed value.
fn number(e: &SqlExpr) -> Option<String> {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(t, _) => Some(if t.contains(['e', 'E']) {
                double_repr(t.parse().ok()?)
            } else {
                t.clone()
            }),
            _ => None,
        },
        SqlExpr::Nested(i) => number(i),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let n = number(expr)?;
            let double = is_double(expr);
            Some(match n.strip_prefix('-') {
                Some(pos) => pos.to_string(),
                // An integer or DECIMAL zero has no sign to flip; a double
                // has -0.0.
                None if !double && n.chars().all(|c| c == '0' || c == '.') => n,
                None => format!("-{n}"),
            })
        }
        _ => None,
    }
}

/// Whether a numeric literal (under any minus signs) is a DOUBLE: written
/// with an exponent.
fn is_double(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Value(v) => matches!(&v.value, SqlValue::Number(t, _) if t.contains(['e', 'E'])),
        SqlExpr::Nested(i) | SqlExpr::UnaryOp { expr: i, .. } => is_double(i),
        _ => false,
    }
}

/// A CAST target as DuckDB prints it: the types its grammar resolves
/// bare, the rest as a quoted name it binds later.
fn type_name(t: &DataType) -> Option<String> {
    let s = t.to_string().to_ascii_uppercase();
    let (base, args) = match s.find('(') {
        Some(i) => (s[..i].trim().to_string(), Some(s[i..].replace(' ', "").replace(',', ", "))),
        None => (s.trim().to_string(), None),
    };
    let args = args.unwrap_or_default();
    Some(match base.as_str() {
        "VARCHAR" | "TEXT" | "STRING" | "CHARACTER VARYING" | "BPCHAR" | "CHAR" => {
            format!("VARCHAR{args}")
        }
        "INTEGER" | "INT" | "INT4" | "SIGNED" => "INTEGER".into(),
        "BIGINT" | "INT8" | "LONG" => "BIGINT".into(),
        "SMALLINT" | "INT2" | "SHORT" => "SMALLINT".into(),
        "BOOLEAN" | "BOOL" | "LOGICAL" => "BOOLEAN".into(),
        "FLOAT" | "REAL" | "FLOAT4" if args.is_empty() => "FLOAT".into(),
        "DECIMAL" | "NUMERIC" => format!("DECIMAL{args}"),
        "DOUBLE" | "DOUBLE PRECISION" | "FLOAT8" => "\"DOUBLE\"".into(),
        "TINYINT" | "INT1" => "\"TINYINT\"".into(),
        "HUGEINT" | "UHUGEINT" | "UBIGINT" | "UINTEGER" | "USMALLINT" | "UTINYINT" | "DATE"
            if args.is_empty() =>
        {
            format!("\"{base}\"")
        }
        _ => return None,
    })
}

fn cmp_op(op: &B) -> Option<&'static str> {
    Some(match op {
        B::Eq => "=",
        B::NotEq => "!=",
        B::Lt => "<",
        B::LtEq => "<=",
        B::Gt => ">",
        B::GtEq => ">=",
        _ => return None,
    })
}

fn negate_cmp(op: &B) -> Option<&'static str> {
    Some(match op {
        B::Eq => "!=",
        B::NotEq => "=",
        B::Lt => ">=",
        B::LtEq => ">",
        B::Gt => "<=",
        B::GtEq => "<",
        _ => return None,
    })
}

/// `e` as the comparison DuckDB's transformer makes of it: a comparison,
/// or NOT over one (bottom-up: `NOT (NOT (a = b))` is `a = b` again).
fn as_cmp(e: &SqlExpr) -> Option<(&SqlExpr, B, &SqlExpr)> {
    match unwrap(e) {
        SqlExpr::BinaryOp { left, op, right } if cmp_op(op).is_some() => {
            Some((left, op.clone(), right))
        }
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => {
            let (l, op, r) = as_cmp(expr)?;
            let flipped = match op {
                B::Eq => B::NotEq,
                B::NotEq => B::Eq,
                B::Lt => B::GtEq,
                B::GtEq => B::Lt,
                B::LtEq => B::Gt,
                B::Gt => B::LtEq,
                _ => return None,
            };
            Some((l, flipped, r))
        }
        _ => None,
    }
}

fn bin_op(op: &B) -> Option<&'static str> {
    if let Some(c) = cmp_op(op) {
        return Some(c);
    }
    Some(match op {
        B::Plus => "+",
        B::Minus => "-",
        B::Multiply => "*",
        B::Divide => "/",
        B::Modulo => "%",
        B::StringConcat => "||",
        B::DuckIntegerDivide => "//",
        B::BitwiseAnd => "&",
        B::BitwiseOr => "|",
        B::PGBitwiseShiftLeft => "<<",
        B::PGBitwiseShiftRight => ">>",
        _ => return None,
    })
}

fn unwrap(e: &SqlExpr) -> &SqlExpr {
    let mut e = e;
    while let SqlExpr::Nested(i) = e {
        e = i;
    }
    e
}

/// The operands of a chain of one conjunction, nested parentheses and all.
fn conj<'e>(e: &'e SqlExpr, want: &B, out: &mut Vec<&'e SqlExpr>) {
    match unwrap(e) {
        SqlExpr::BinaryOp { left, op, right } if op == want => {
            conj(left, want, out);
            conj(right, want, out);
        }
        other => out.push(other),
    }
}

fn list(items: &[SqlExpr]) -> Option<String> {
    Some(items.iter().map(print).collect::<Option<Vec<_>>>()?.join(", "))
}

fn args_of(f: &sqlparser::ast::Function) -> Option<String> {
    let FunctionArguments::List(l) = &f.args else {
        return match f.args {
            FunctionArguments::None => Some(String::new()),
            _ => None,
        };
    };
    if l.duplicate_treatment.is_some() || !l.clauses.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(l.args.len());
    for a in &l.args {
        out.push(match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(x)) => print(x)?,
            FunctionArg::Named {
                name,
                arg: FunctionArgExpr::Expr(x),
                ..
            } => format!("{} := {}", ident(&name.value), print(x)?),
            FunctionArg::ExprNamed {
                name: SqlExpr::Identifier(name),
                arg: FunctionArgExpr::Expr(x),
                ..
            } => format!("{} := {}", ident(&name.value), print(x)?),
            _ => return None,
        });
    }
    Some(out.join(", "))
}

fn function(f: &sqlparser::ast::Function) -> Option<String> {
    if f.over.is_some() || f.filter.is_some() || !f.within_group.is_empty() {
        return None;
    }
    if f.name.0.len() != 1 {
        return None;
    }
    let name = f.name.to_string();
    let lower = name.trim_matches('"').to_ascii_lowercase();
    // A SQL function call read by field prints as the call, as DuckDB names
    // a macro call (`(pair(a)).lo`).
    if lower == super::macros::CALL_MARKER {
        let (id, calls) = super::calls::marker_call(&SqlExpr::Function(f.clone()))?;
        let args = super::calls::marker_args(f)
            .iter()
            .map(print)
            .collect::<Option<Vec<_>>>()?;
        return Some(format!(
            "{}({})",
            ident(&calls[id].0.to_ascii_lowercase()),
            args.join(", ")
        ));
    }
    let args = args_of(f)?;
    Some(match lower.as_str() {
        "coalesce" | "ifnull" => format!("COALESCE({args})"),
        "if" => {
            let FunctionArguments::List(l) = &f.args else { return None };
            let [FunctionArg::Unnamed(FunctionArgExpr::Expr(c)), FunctionArg::Unnamed(FunctionArgExpr::Expr(t)), FunctionArg::Unnamed(FunctionArgExpr::Expr(e))] =
                l.args.as_slice()
            else {
                return None;
            };
            format!("CASE  WHEN ({}) THEN ({}) ELSE {} END", print(c)?, print(t)?, print(e)?)
        }
        // The rewrite marker for GLOB is an identity on its operand.
        "__glob_pat" => args,
        _ => format!("{}({args})", ident(&lower)),
    })
}

/// The printed form of `e` inside an expression.
fn print(e: &SqlExpr) -> Option<String> {
    if let Some(n) = number(e) {
        return Some(n);
    }
    Some(match e {
        SqlExpr::Nested(i) => print(i)?,
        SqlExpr::Identifier(i) => ident(&i.value),
        SqlExpr::CompoundIdentifier(parts) => {
            parts.iter().map(|p| ident(&p.value)).collect::<Vec<_>>().join(".")
        }
        SqlExpr::Value(v) => match &v.value {
            SqlValue::SingleQuotedString(s) => quote_str(s),
            SqlValue::Boolean(true) => "CAST('t' AS BOOLEAN)".into(),
            SqlValue::Boolean(false) => "CAST('f' AS BOOLEAN)".into(),
            SqlValue::Null => "NULL".into(),
            _ => return None,
        },
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Minus => format!("-({})", print(expr)?),
            UnaryOperator::Plus => format!("+({})", print(expr)?),
            UnaryOperator::BitwiseNot => format!("~({})", print(expr)?),
            UnaryOperator::Not => match unwrap(expr) {
                inner if as_cmp(inner).is_some() => {
                    let (l, op, r) = as_cmp(inner).expect("checked");
                    format!("({} {} {})", print(l)?, negate_cmp(&op)?, print(r)?)
                }
                SqlExpr::InList {
                    expr,
                    list: l,
                    negated: false,
                } => format!("({} NOT IN ({}))", print(expr)?, list(l)?),
                SqlExpr::IsDistinctFrom(a, b) => {
                    format!("({} IS NOT DISTINCT FROM {})", print(a)?, print(b)?)
                }
                SqlExpr::IsNotDistinctFrom(a, b) => {
                    format!("({} IS DISTINCT FROM {})", print(a)?, print(b)?)
                }
                inner => format!("(NOT {})", print(inner)?),
            },
            _ => return None,
        },
        SqlExpr::BinaryOp { op: op @ (B::And | B::Or), .. } => {
            let mut parts = Vec::new();
            conj(e, op, &mut parts);
            let word = if *op == B::And { " AND " } else { " OR " };
            format!(
                "({})",
                parts.into_iter().map(print).collect::<Option<Vec<_>>>()?.join(word)
            )
        }
        SqlExpr::BinaryOp { left, op, right } => {
            format!("({} {} {})", print(left)?, bin_op(op)?, print(right)?)
        }
        SqlExpr::IsNull(x) => format!("({} IS NULL)", print(x)?),
        SqlExpr::IsNotNull(x) => format!("({} IS NOT NULL)", print(x)?),
        SqlExpr::IsDistinctFrom(a, b) => format!("({} IS DISTINCT FROM {})", print(a)?, print(b)?),
        SqlExpr::IsNotDistinctFrom(a, b) => {
            format!("({} IS NOT DISTINCT FROM {})", print(a)?, print(b)?)
        }
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let b = format!("({} BETWEEN {} AND {})", print(expr)?, print(low)?, print(high)?);
            if *negated {
                format!("(NOT {b})")
            } else {
                b
            }
        }
        SqlExpr::InList {
            expr,
            list: l,
            negated,
        } => format!(
            "({} {}IN ({}))",
            print(expr)?,
            if *negated { "NOT " } else { "" },
            list(l)?
        ),
        SqlExpr::Like {
            negated,
            any: false,
            expr,
            pattern,
            escape_char: None,
        } => {
            // GLOB arrives as LIKE over the rewrite marker.
            if let SqlExpr::Function(f) = unwrap(pattern) {
                if f.name.to_string().eq_ignore_ascii_case("__glob_pat") && !negated {
                    return Some(format!("({} ~~~ {})", print(expr)?, function(f)?));
                }
            }
            let op = if *negated { "!~~" } else { "~~" };
            format!("({} {op} {})", print(expr)?, print(pattern)?)
        }
        SqlExpr::ILike {
            negated,
            any: false,
            expr,
            pattern,
            escape_char: None,
        } => {
            let op = if *negated { "!~~*" } else { "~~*" };
            format!("({} {op} {})", print(expr)?, print(pattern)?)
        }
        SqlExpr::SimilarTo {
            negated: false,
            expr,
            pattern,
            escape_char: None,
        } => format!("regexp_full_match({}, {})", print(expr)?, print(pattern)?),
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format: None,
            ..
        } => {
            let word = match kind {
                CastKind::Cast | CastKind::DoubleColon => "CAST",
                CastKind::TryCast => "TRY_CAST",
                _ => return None,
            };
            format!("{word}({} AS {})", print(expr)?, type_name(data_type)?)
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            let mut out = String::from("CASE ");
            for w in conditions {
                let cond = match operand {
                    Some(op) => format!("({} = {})", print(op)?, print(&w.condition)?),
                    None => print(&w.condition)?,
                };
                out += &format!(" WHEN ({cond}) THEN ({})", print(&w.result)?);
            }
            let tail = match else_result {
                Some(x) => print(x)?,
                None => "NULL".into(),
            };
            out += &format!(" ELSE {tail} END");
            out
        }
        SqlExpr::Function(f) => function(f)?,
        SqlExpr::Ceil {
            expr,
            field: sqlparser::ast::CeilFloorKind::DateTimeField(sqlparser::ast::DateTimeField::NoDateTime),
        } => format!("ceil({})", print(expr)?),
        SqlExpr::Floor {
            expr,
            field: sqlparser::ast::CeilFloorKind::DateTimeField(sqlparser::ast::DateTimeField::NoDateTime),
        } => format!("floor({})", print(expr)?),
        SqlExpr::CompoundFieldAccess { root, access_chain } => {
            // A dotted run on a column stays one column reference; a field
            // read off anything else is struct_extract, printed
            // `(base).field`; a subscript appends `[i]`.
            let (mut cur, mut colref) = match root.as_ref() {
                SqlExpr::Identifier(i) => (ident(&i.value), true),
                SqlExpr::CompoundIdentifier(p) => {
                    (p.iter().map(|x| ident(&x.value)).collect::<Vec<_>>().join("."), true)
                }
                SqlExpr::Nested(x) => (print(x)?, false),
                other => (print(other)?, false),
            };
            for acc in access_chain {
                match acc {
                    sqlparser::ast::AccessExpr::Dot(SqlExpr::Identifier(d)) => {
                        cur = if colref {
                            format!("{cur}.{}", ident(&d.value))
                        } else {
                            format!("({cur}).{}", ident(&d.value))
                        };
                    }
                    sqlparser::ast::AccessExpr::Subscript(
                        sqlparser::ast::Subscript::Index { index },
                    ) => {
                        cur = format!("{cur}[{}]", print(index)?);
                        colref = false;
                    }
                    _ => return None,
                }
            }
            cur
        }
        SqlExpr::Array(a) if !a.named => format!("main.list_value({})", list(&a.elem)?),
        SqlExpr::Trim {
            trim_where: None,
            trim_what,
            expr,
            trim_characters: None,
        } => match trim_what {
            None => format!("main.\"trim\"({})", print(expr)?),
            Some(w) => format!("main.\"trim\"({}, {})", print(expr)?, print(w)?),
        },
        SqlExpr::Substring {
            expr,
            substring_from: Some(from),
            substring_for,
            shorthand: false,
            ..
        } => match substring_for {
            Some(n) => format!("main.\"substring\"({}, {}, {})", print(expr)?, print(from)?, print(n)?),
            None => format!("main.\"substring\"({}, {})", print(expr)?, print(from)?),
        },
        SqlExpr::Position { expr, r#in } => {
            format!("main.\"position\"({}, {})", print(r#in)?, print(expr)?)
        }
        _ => return None,
    })
}

/// DuckDB's output name for an unaliased item that is not a bare column
/// (the caller names those), or `None` for a form not modeled here.
pub(super) fn duck_name(e: &SqlExpr) -> Option<String> {
    print(e)
}
