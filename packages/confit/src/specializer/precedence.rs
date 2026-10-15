//! The sqlparser dialect confit parses with: `GenericDialect`, with DuckDB's
//! operator precedence.
//!
//! sqlparser 0.62 ranks several operators unlike DuckDB's grammar, so a query
//! without parentheses could parse to another tree and serve another answer:
//! `a IS DISTINCT FROM b AND c` parsed as `a IS DISTINCT FROM (b AND c)`,
//! `s || 'b' ^@ 'a'` as `s || ('b' ^@ 'a')`, `x = s LIKE p` as
//! `(x = s) LIKE p`, `NOT EXISTS (q) IS NULL` as `(NOT EXISTS (q)) IS NULL`.
//! This dialect ranks them as DuckDB does, lowest first:
//!
//! 1. `OR`, then `AND`, then a prefix `NOT`;
//! 2. `IS` (`IS NULL`, `IS TRUE`, `IS [NOT] DISTINCT FROM`);
//! 3. the comparisons `= <> < <= > >=`;
//! 4. `LIKE`, `ILIKE`, `SIMILAR TO`, `GLOB`, `IN` and `BETWEEN`;
//! 5. the other operators, left-associative: `|| ^@ ~ !~ & | << >>`;
//! 6. `+ -`, then `* / // %`, then `^`, then `AT TIME ZONE`, then a prefix
//!    sign, then `::` and `[]`.
//!
//! The levels 2 to 4 do not chain in DuckDB: `a = b < c`, `a LIKE b IN (c)`
//! and `a IS DISTINCT FROM b IS NULL` are syntax errors there, as are a
//! prefix `NOT` right after an operator of level 5 (`a || NOT b`) and as the
//! lower bound of `BETWEEN`. This dialect refuses them too.
//!
//! The tests hold this to DuckDB's own parses, pinned in
//! `docs/specs/pins-wave5/operator-precedence.json`
//! (`scripts/pin_precedence.py` records them).

use std::any::TypeId;

use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator};
use sqlparser::dialect::{Dialect, GenericDialect, Precedence};
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

const OR: u8 = 5;
const AND: u8 = 10;
const NOT: u8 = 15;
const IS: u8 = 17;
const CMP: u8 = 20;
const LIKE: u8 = 22;
const OP: u8 = 25;
const ADD: u8 = 30;
const MUL: u8 = 40;
const POW: u8 = 42;
const AT: u8 = 44;
const SIGN: u8 = 46;
const CAST: u8 = 50;
const PERIOD: u8 = 100;

/// `GenericDialect` with DuckDB's operator precedence (see the module).
#[derive(Debug, Default, Clone, Copy)]
pub struct DuckDbPrecedence;

/// The methods `GenericDialect` overrides in sqlparser 0.62, passed through.
/// Check this list when sqlparser changes version.
macro_rules! generic {
    ($($f:ident),* $(,)?) => {
        $(fn $f(&self) -> bool { GenericDialect.$f() })*
    };
}

impl Dialect for DuckDbPrecedence {
    fn dialect(&self) -> TypeId {
        // sqlparser checks for `GenericDialect` by this id.
        TypeId::of::<GenericDialect>()
    }

    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        GenericDialect.is_delimited_identifier_start(ch)
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        GenericDialect.is_identifier_start(ch)
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        GenericDialect.is_identifier_part(ch)
    }

    generic!(
        supports_unicode_string_literal,
        supports_partition_by_after_order_by,
        supports_array_join_syntax,
        supports_group_by_expr,
        supports_group_by_with_modifier,
        supports_left_associative_joins_without_parens,
        supports_connect_by,
        supports_match_recognize,
        supports_pipe_operator,
        supports_start_transaction_modifier,
        supports_window_function_null_treatment_arg,
        supports_dictionary_syntax,
        supports_window_clause_named_window_reference,
        supports_parenthesized_set_variables,
        supports_select_wildcard_except,
        support_map_literal_syntax,
        allow_extract_custom,
        allow_extract_single_quotes,
        supports_extract_comma_syntax,
        supports_create_view_comment_syntax,
        supports_parens_around_table_factor,
        supports_values_as_table_factor,
        supports_create_index_with_clause,
        supports_explain_with_utility_options,
        supports_limit_comma,
        supports_update_order_by,
        supports_from_first_select,
        supports_projection_trailing_commas,
        supports_asc_desc_in_column_definition,
        supports_try_convert,
        supports_bitwise_shift_operators,
        supports_comment_on,
        supports_load_extension,
        supports_named_fn_args_with_assignment_operator,
        supports_struct_literal,
        supports_empty_projections,
        supports_nested_comments,
        supports_multiline_comment_hints,
        supports_user_host_grantee,
        supports_string_escape_constant,
        supports_array_typedef_with_brackets,
        supports_match_against,
        supports_set_names,
        supports_comma_separated_set_assignments,
        supports_filter_during_aggregation,
        supports_select_wildcard_exclude,
        supports_data_type_signed_suffix,
        supports_interval_options,
        supports_quote_delimited_string,
        supports_select_wildcard_replace,
        supports_select_wildcard_ilike,
        supports_select_wildcard_rename,
        supports_optimize_table,
        supports_install,
        supports_detach,
        supports_prewhere,
        supports_with_fill,
        supports_limit_by,
        supports_interpolate,
        supports_settings,
        supports_select_format,
        supports_comment_optimizer_hint,
        supports_constraint_keyword_without_name,
        supports_key_column_option,
        supports_comma_separated_trim,
        supports_cte_without_as,
        supports_select_item_multi_column_alias,
        supports_xml_expressions,
    );

    fn prec_value(&self, prec: Precedence) -> u8 {
        match prec {
            Precedence::Period => PERIOD,
            Precedence::DoubleColon => CAST,
            Precedence::AtTz => AT,
            Precedence::Caret => POW,
            Precedence::MulDivModOp => MUL,
            Precedence::PlusMinus => ADD,
            Precedence::Xor
            | Precedence::Ampersand
            | Precedence::Pipe
            | Precedence::Colon
            | Precedence::PgOther => OP,
            Precedence::Between | Precedence::Like => LIKE,
            Precedence::Eq => CMP,
            Precedence::Is => IS,
            Precedence::UnaryNot => NOT,
            Precedence::And => AND,
            Precedence::Or => OR,
        }
    }

    fn get_next_precedence(&self, parser: &Parser) -> Option<Result<u8, ParserError>> {
        // sqlparser ranks these with the comparisons (`~`), with `*` (`||`),
        // with `::` (`^@`) or on levels of their own (`& | << >>`).
        match parser.peek_token_ref().token {
            Token::StringConcat
            | Token::CaretAt
            | Token::Tilde
            | Token::TildeAsterisk
            | Token::ExclamationMarkTilde
            | Token::ExclamationMarkTildeAsterisk
            | Token::DoubleTilde
            | Token::DoubleTildeAsterisk
            | Token::ExclamationMarkDoubleTilde
            | Token::ExclamationMarkDoubleTildeAsterisk
            | Token::Ampersand
            | Token::Pipe
            | Token::ShiftLeft
            | Token::ShiftRight => Some(Ok(OP)),
            _ => None,
        }
    }

    fn parse_prefix(&self, parser: &mut Parser) -> Option<Result<Expr, ParserError>> {
        // sqlparser parses `NOT EXISTS (...)` as one operand, so
        // `NOT EXISTS (...) IS NULL` would be `(NOT EXISTS (...)) IS NULL`;
        // DuckDB's NOT takes `EXISTS (...) IS NULL`, as any other NOT does.
        if keyword(parser, 0, Keyword::NOT) && keyword(parser, 1, Keyword::EXISTS) {
            parser.advance_token();
            return Some(parser.parse_subexpr(NOT).map(|e| Expr::UnaryOp {
                op: UnaryOperator::Not,
                expr: Box::new(e),
            }));
        }
        // sqlparser takes the operand of a sign up to `*` and ranks `^@` with
        // `::`, so `-a ^@ b` would be `-(a ^@ b)`; DuckDB binds the sign
        // tighter.
        let op = match parser.peek_token_ref().token {
            Token::Minus => UnaryOperator::Minus,
            Token::Plus => UnaryOperator::Plus,
            _ => return None,
        };
        parser.advance_token();
        Some(parser.parse_subexpr(SIGN).map(|e| Expr::UnaryOp {
            op,
            expr: Box::new(e),
        }))
    }

    fn parse_infix(
        &self,
        parser: &mut Parser,
        expr: &Expr,
        precedence: u8,
    ) -> Option<Result<Expr, ParserError>> {
        if let Err(e) = no_chain(parser, expr, precedence) {
            return Some(Err(e));
        }
        // sqlparser takes a whole expression after IS [NOT] DISTINCT FROM,
        // a following AND or OR included.
        let negated = if parser.parse_keywords(&[Keyword::IS, Keyword::DISTINCT, Keyword::FROM]) {
            false
        } else if parser.parse_keywords(&[
            Keyword::IS,
            Keyword::NOT,
            Keyword::DISTINCT,
            Keyword::FROM,
        ]) {
            true
        } else {
            return None;
        };
        let right = match parser.parse_subexpr(precedence) {
            Ok(r) => Box::new(r),
            Err(e) => return Some(Err(e)),
        };
        let left = Box::new(expr.clone());
        Some(Ok(if negated {
            Expr::IsNotDistinctFrom(left, right)
        } else {
            Expr::IsDistinctFrom(left, right)
        }))
    }
}

/// Refuses the operator after `left` where DuckDB's grammar has no parse: a
/// second operator of level 2, 3 or 4 right after one (`a = b < c`), a
/// prefix `NOT` right after an operator of level 5 (`a || NOT b`) or as the
/// lower bound of `BETWEEN`.
fn no_chain(parser: &Parser, left: &Expr, precedence: u8) -> Result<(), ParserError> {
    use BinaryOperator as B;
    let what = match left {
        Expr::BinaryOp {
            op: B::Eq | B::NotEq | B::Lt | B::LtEq | B::Gt | B::GtEq | B::Spaceship,
            ..
        } if precedence == CMP => "does not chain comparisons",
        Expr::Like { .. }
        | Expr::ILike { .. }
        | Expr::SimilarTo { .. }
        | Expr::RLike { .. }
        | Expr::Between { .. }
            if precedence == LIKE =>
        {
            "does not chain LIKE, SIMILAR TO, GLOB, IN and BETWEEN"
        }
        Expr::IsDistinctFrom(..) | Expr::IsNotDistinctFrom(..) if precedence == IS => {
            "takes no IS test right after IS [NOT] DISTINCT FROM"
        }
        _ if precedence == OP && keyword(parser, 1, Keyword::NOT) => {
            "takes no NOT right after this operator"
        }
        _ if precedence == LIKE
            && (keyword(parser, 0, Keyword::BETWEEN) && keyword(parser, 1, Keyword::NOT)
                || keyword(parser, 0, Keyword::NOT)
                    && keyword(parser, 1, Keyword::BETWEEN)
                    && keyword(parser, 2, Keyword::NOT)) =>
        {
            "takes no NOT as the lower bound of BETWEEN"
        }
        _ => return Ok(()),
    };
    Err(ParserError::ParserError(format!(
        "syntax error at or near \"{}\": DuckDB {what}; add parentheses",
        parser.peek_token_ref().token
    )))
}

/// True when the token `n` places ahead is the keyword `k`.
fn keyword(parser: &Parser, n: usize, k: Keyword) -> bool {
    matches!(&parser.peek_nth_token_ref(n).token, Token::Word(w) if w.keyword == k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::ast::{DataType, SelectItem, SetExpr, Statement};

    /// The pin: `(form, DuckDB's tree)` for each form DuckDB parses, and the
    /// forms it refuses, one row per line.
    fn pin() -> (Vec<(&'static str, &'static str)>, Vec<&'static str>) {
        let text = include_str!("../../docs/specs/pins-wave5/operator-precedence.json");
        let (mut trees, mut rejected, mut section) = (Vec::new(), Vec::new(), "");
        for line in text.lines() {
            if let Some(key) = line
                .strip_prefix("  \"")
                .and_then(|l| l.strip_suffix("\": ["))
            {
                section = key;
                continue;
            }
            let Some(row) = line.trim().trim_end_matches(',').strip_prefix("[\"") else {
                continue;
            };
            let (form, rest) = row.split_once("\", \"").expect("a row of two strings");
            match section {
                "trees" => trees.push((form, rest.strip_suffix("\"]").expect("a row's end"))),
                "rejected" => rejected.push(form),
                other => panic!("a row outside trees and rejected: {other}"),
            }
        }
        (trees, rejected)
    }

    /// The one projection of `SELECT <form>`.
    fn parse(form: &str) -> Result<Expr, ParserError> {
        let statements = Parser::parse_sql(&DuckDbPrecedence, &format!("SELECT {form}"))?;
        let [Statement::Query(q)] = statements.as_slice() else {
            panic!("one query: {form}");
        };
        let SetExpr::Select(s) = q.body.as_ref() else {
            panic!("a SELECT: {form}");
        };
        let [SelectItem::UnnamedExpr(e)] = s.projection.as_slice() else {
            panic!("one item: {form}");
        };
        Ok(e.clone())
    }

    fn bare(e: &Expr) -> &Expr {
        match e {
            Expr::Nested(x) => bare(x),
            _ => e,
        }
    }

    fn comparison(op: &BinaryOperator) -> Option<(&'static str, &'static str)> {
        use BinaryOperator as B;
        Some(match op {
            B::Eq => ("=", "!="),
            B::NotEq => ("!=", "="),
            B::Lt => ("<", ">="),
            B::LtEq => ("<=", ">"),
            B::Gt => (">", "<="),
            B::GtEq => (">=", "<"),
            _ => return None,
        })
    }

    fn conjuncts(e: &Expr, op: &BinaryOperator, out: &mut Vec<String>) {
        match bare(e) {
            Expr::BinaryOp { left, op: o, right } if o == op => {
                conjuncts(left, op, out);
                conjuncts(right, op, out);
            }
            x => out.push(tree(x)),
        }
    }

    fn is_bool(x: &Expr, value: &str, distinct: bool) -> String {
        let d = if distinct { "" } else { "NOT " };
        format!("(CAST({} AS BOOLEAN) IS {d}DISTINCT FROM {value})", tree(x))
    }

    fn regex(e: &Expr, p: &Expr, negated: bool) -> String {
        let m = format!("regexp_full_match({}, {})", tree(e), tree(p));
        if negated {
            format!("(NOT {m})")
        } else {
            m
        }
    }

    fn list(l: &[Expr]) -> String {
        l.iter().map(tree).collect::<Vec<_>>().join(", ")
    }

    /// `NOT e` where DuckDB's parser folds the NOT into `e`.
    fn folded_not(e: &Expr) -> Option<String> {
        Some(match bare(e) {
            Expr::BinaryOp { left, op, right } => {
                format!("({} {} {})", tree(left), comparison(op)?.1, tree(right))
            }
            Expr::IsDistinctFrom(a, b) => format!("({} IS NOT DISTINCT FROM {})", tree(a), tree(b)),
            Expr::IsNotDistinctFrom(a, b) => format!("({} IS DISTINCT FROM {})", tree(a), tree(b)),
            Expr::IsTrue(x) => is_bool(x, "true", true),
            Expr::IsNotTrue(x) => is_bool(x, "true", false),
            Expr::IsFalse(x) => is_bool(x, "false", true),
            Expr::IsNotFalse(x) => is_bool(x, "false", false),
            Expr::InList {
                expr,
                list: l,
                negated: false,
            } => format!("({} NOT IN ({}))", tree(expr), list(l)),
            _ => return None,
        })
    }

    /// `e` as DuckDB's `json_deserialize_sql` prints its parse: every
    /// operator in parentheses, under DuckDB's own names.
    fn tree(e: &Expr) -> String {
        use BinaryOperator as B;
        match bare(e) {
            Expr::Identifier(i) => i.value.clone(),
            Expr::BinaryOp {
                op: op @ (B::And | B::Or),
                ..
            } => {
                let mut v = Vec::new();
                conjuncts(e, op, &mut v);
                format!("({})", v.join(&format!(" {op} ")))
            }
            Expr::BinaryOp {
                left,
                op: B::PGRegexMatch,
                right,
            } => regex(left, right, false),
            Expr::BinaryOp {
                left,
                op: B::PGRegexNotMatch,
                right,
            } => regex(left, right, true),
            Expr::BinaryOp { left, op, right } => {
                let op = comparison(op).map_or(op.to_string(), |c| c.0.to_string());
                format!("({} {op} {})", tree(left), tree(right))
            }
            Expr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
            } => folded_not(expr).unwrap_or_else(|| format!("(NOT {})", tree(expr))),
            Expr::UnaryOp { op, expr } => format!("{op}({})", tree(expr)),
            Expr::IsNull(x) => format!("({} IS NULL)", tree(x)),
            Expr::IsNotNull(x) => format!("({} IS NOT NULL)", tree(x)),
            Expr::IsTrue(x) => is_bool(x, "true", false),
            Expr::IsNotTrue(x) => is_bool(x, "true", true),
            Expr::IsFalse(x) => is_bool(x, "false", false),
            Expr::IsNotFalse(x) => is_bool(x, "false", true),
            Expr::IsDistinctFrom(a, b) => format!("({} IS DISTINCT FROM {})", tree(a), tree(b)),
            Expr::IsNotDistinctFrom(a, b) => {
                format!("({} IS NOT DISTINCT FROM {})", tree(a), tree(b))
            }
            Expr::Like {
                negated,
                expr,
                pattern,
                ..
            } => {
                let op = if *negated { "!~~" } else { "~~" };
                format!("({} {op} {})", tree(expr), tree(pattern))
            }
            Expr::ILike {
                negated,
                expr,
                pattern,
                ..
            } => {
                let op = if *negated { "!~~*" } else { "~~*" };
                format!("({} {op} {})", tree(expr), tree(pattern))
            }
            Expr::SimilarTo {
                negated,
                expr,
                pattern,
                ..
            } => regex(expr, pattern, *negated),
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let b = format!("({} BETWEEN {} AND {})", tree(expr), tree(low), tree(high));
                if *negated {
                    format!("(NOT {b})")
                } else {
                    b
                }
            }
            Expr::InList {
                expr,
                list: l,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                format!("({} {not}IN ({}))", tree(expr), list(l))
            }
            Expr::Cast {
                expr,
                data_type: DataType::Int(None),
                ..
            } => format!("CAST({} AS INTEGER)", tree(expr)),
            // The parser writes `NOT EXISTS` as NOT over EXISTS.
            Expr::Exists {
                subquery,
                negated: false,
            } => format!("EXISTS({subquery})"),
            Expr::InSubquery {
                expr,
                subquery,
                negated,
            } => {
                let any = format!("({} = ANY({subquery}))", tree(expr));
                if *negated {
                    format!("(NOT {any})")
                } else {
                    any
                }
            }
            other => format!("<unprinted: {other}>"),
        }
    }

    #[test]
    fn parses_each_pinned_form_to_duckdbs_tree() {
        let (trees, _) = pin();
        assert!(trees.len() > 900, "the pin holds {} trees", trees.len());
        let wrong: Vec<String> = trees
            .iter()
            .filter_map(|(form, want)| {
                let got = parse(form)
                    .map(|e| tree(&e))
                    .unwrap_or_else(|e| e.to_string());
                (got != *want).then(|| format!("{form}\n  DuckDB: {want}\n  confit: {got}"))
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "{} of {} differ:\n{}",
            wrong.len(),
            trees.len(),
            wrong.join("\n")
        );
    }

    #[test]
    fn refuses_each_pinned_form_duckdb_refuses() {
        let (_, rejected) = pin();
        assert!(
            rejected.len() > 50,
            "the pin holds {} refusals",
            rejected.len()
        );
        for form in rejected {
            match parse(form) {
                Err(e) => assert!(e.to_string().contains("add parentheses"), "{form}: {e}"),
                Ok(e) => panic!("{form} parses to {}", tree(&e)),
            }
        }
    }

    #[test]
    fn parses_statements_as_generic_dialect_does() {
        // Forms that need one of GenericDialect's settings; the precedence
        // changes nothing here.
        for sql in [
            "SELECT * EXCLUDE (a) FROM t",
            "SELECT * REPLACE (a + 1 AS a) FROM t",
            "SELECT * ILIKE 'a%' FROM t",
            "SELECT * RENAME (a AS b) FROM t",
            "SELECT {'x': 1, 'y': a} AS s FROM t",
            "SELECT MAP {'k': 1} FROM t",
            "SELECT count(*) FILTER (WHERE a > 1) FROM t",
            "SELECT f(x := 1), TRY_CAST(a AS INT), a::INT[] FROM t",
            "SELECT a, FROM t",
            "SELECT s ^@ 'a', a // 2, a << 1 FROM t AS x(a, s) WHERE a IN (1, 2)",
        ] {
            let ours = Parser::parse_sql(&DuckDbPrecedence, sql).map_err(|e| e.to_string());
            let theirs = Parser::parse_sql(&GenericDialect, sql).map_err(|e| e.to_string());
            assert!(ours.is_ok(), "{sql}: {ours:?}");
            assert_eq!(format!("{ours:?}"), format!("{theirs:?}"), "{sql}");
        }
    }
}
