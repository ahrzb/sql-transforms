//! Token-level pre-rewrites applied before sqlparser sees the query.
//!
//! sqlparser 0.62 cannot represent some DuckDB surface forms; worse, one of
//! them parses SILENTLY WRONG: `SELECT k: expr` becomes a Snowflake
//! JsonAccess (`k:expr` as a JSON path) under every dialect, so no parse
//! error will ever fire (pins-wave5/sqlparser-spike.json). We fix these on
//! the token stream: `ident COLON` at a select-item start becomes
//! `expr AS ident` — the exact desugaring DuckDB documents for its
//! prefix-alias syntax. The binder's JsonAccess rejection stays as the
//! backstop for anything this pass misses.

use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Whitespace, Word};

/// Rewrite `SELECT k: expr, ...` into `SELECT expr AS k, ...`.
///
/// Trigger: a (possibly quoted) word followed by a single `:` at the START
/// of a select item (right after SELECT / DISTINCT / ALL or a top-level
/// comma of that select's projection list). `::` is a distinct token
/// (DoubleColon) and slice colons live behind `[`, so neither can trigger.
/// The alias is re-emitted as ` AS k` at the end of that item (next
/// top-level comma, a clause keyword such as FROM, a closing paren of the
/// enclosing subquery, or end of input).
pub fn rewrite_colon_aliases(tokens: Vec<Token>) -> Vec<Token> {
    // One projection-list context per (possibly nested) SELECT: the paren
    // depth at which its items sit.
    struct Ctx {
        paren: i32,
    }
    let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut selects: Vec<Ctx> = Vec::new();
    let mut pending: Vec<Option<Word>> = Vec::new(); // parallel to `selects`
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut item_start = false;

    // Keywords that terminate a projection list at its own depth.
    const LIST_END: &[Keyword] = &[
        Keyword::FROM,
        Keyword::WHERE,
        Keyword::GROUP,
        Keyword::HAVING,
        Keyword::ORDER,
        Keyword::LIMIT,
        Keyword::OFFSET,
        Keyword::UNION,
        Keyword::EXCEPT,
        Keyword::INTERSECT,
        Keyword::WINDOW,
        Keyword::QUALIFY,
    ];

    let flush = |out: &mut Vec<Token>, pending: &mut Vec<Option<Word>>| {
        if let Some(slot) = pending.last_mut() {
            if let Some(alias) = slot.take() {
                out.push(Token::Whitespace(Whitespace::Space));
                out.push(Token::make_keyword("AS"));
                out.push(Token::Whitespace(Whitespace::Space));
                out.push(Token::Word(alias));
            }
        }
    };

    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if matches!(t, Token::Whitespace(_)) {
            out.push(t.clone());
            i += 1;
            continue;
        }
        let at_list_depth = selects
            .last()
            .is_some_and(|c| c.paren == paren && bracket == 0);
        match t {
            Token::Word(w) if w.quote_style.is_none() && w.keyword == Keyword::SELECT => {
                out.push(t.clone());
                selects.push(Ctx { paren });
                pending.push(None);
                item_start = true;
            }
            Token::Word(w)
                if w.quote_style.is_none()
                    && matches!(w.keyword, Keyword::DISTINCT | Keyword::ALL)
                    && item_start =>
            {
                out.push(t.clone());
            }
            Token::Word(w) if at_list_depth && LIST_END.contains(&w.keyword) => {
                flush(&mut out, &mut pending);
                selects.pop();
                pending.pop();
                out.push(t.clone());
                item_start = false;
            }
            Token::Comma if at_list_depth => {
                flush(&mut out, &mut pending);
                out.push(t.clone());
                item_start = true;
            }
            Token::Word(w) if item_start => {
                // Lookahead past whitespace for a single `:`.
                let mut j = i + 1;
                while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                    j += 1;
                }
                if matches!(tokens.get(j), Some(Token::Colon)) {
                    if let Some(slot) = pending.last_mut() {
                        *slot = Some(w.clone());
                    }
                    i = j + 1; // drop `ident` and `:`, keep the expression
                    item_start = false;
                    continue;
                }
                out.push(t.clone());
                item_start = false;
            }
            Token::LParen => {
                paren += 1;
                out.push(t.clone());
                item_start = false;
            }
            Token::RParen => {
                if selects.last().is_some_and(|c| paren == c.paren) {
                    flush(&mut out, &mut pending);
                    selects.pop();
                    pending.pop();
                }
                paren -= 1;
                out.push(t.clone());
                item_start = false;
            }
            Token::LBracket => {
                bracket += 1;
                out.push(t.clone());
                item_start = false;
            }
            Token::RBracket => {
                bracket -= 1;
                out.push(t.clone());
                item_start = false;
            }
            _ => {
                out.push(t.clone());
                item_start = false;
            }
        }
        i += 1;
    }
    while pending.last().is_some() {
        flush(&mut out, &mut pending);
        pending.pop();
        selects.pop();
    }
    out
}
