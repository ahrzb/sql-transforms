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

/// True when `t` can END a value expression — i.e. a following bare word
/// like GLOB sits in infix position.
fn ends_value(t: Option<&Token>) -> bool {
    match t {
        Some(Token::Word(w)) => {
            w.quote_style.is_some()
                || matches!(
                    w.keyword,
                    Keyword::NoKeyword
                        | Keyword::NULL
                        | Keyword::TRUE
                        | Keyword::FALSE
                        | Keyword::END
                )
        }
        Some(Token::Number(..))
        | Some(Token::SingleQuotedString(_))
        | Some(Token::RParen)
        | Some(Token::RBracket) => true,
        _ => false,
    }
}

/// Rewrite star name filters — `* [EXCLUDE (...)] {LIKE | NOT LIKE | GLOB |
/// NOT ILIKE} '<pat>'` — into the one form sqlparser CAN parse, `* ILIKE`,
/// encoding the real operator as a `\u{1}<code>:` prefix inside the pattern
/// string (decoded by the binder; a plain `* ILIKE` has no marker). Runs
/// BEFORE the infix-GLOB rewrite so star-GLOB is consumed here. `* SIMILAR
/// TO` stays a parse error (regexp semantics — wave B).
pub fn rewrite_star_filters(tokens: Vec<Token>) -> Vec<Token> {
    let star_position = |out: &[Token]| {
        match out.iter().rev().find(|t| !matches!(t, Token::Whitespace(_))) {
            Some(Token::Word(w)) => w.quote_style.is_none() && w.keyword == Keyword::SELECT,
            Some(Token::Comma) | Some(Token::Period) => true,
            _ => false,
        }
    };
    let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if !(matches!(tokens[i], Token::Mul) && star_position(&out)) {
            out.push(tokens[i].clone());
            i += 1;
            continue;
        }
        out.push(tokens[i].clone()); // the star
        i += 1;
        // Buffer whitespace and an optional EXCLUDE group (parenthesized or
        // a single bare identifier): DuckDB writes the filter AFTER EXCLUDE
        // but sqlparser only parses ILIKE BEFORE it, so on a filter match
        // the synthesized ILIKE is emitted ahead of the buffered group.
        let mut buf: Vec<Token> = Vec::new();
        while matches!(tokens.get(i), Some(Token::Whitespace(_))) {
            buf.push(tokens[i].clone());
            i += 1;
        }
        if matches!(&tokens.get(i), Some(Token::Word(w)) if w.value.eq_ignore_ascii_case("exclude"))
        {
            buf.push(tokens[i].clone());
            i += 1;
            while matches!(tokens.get(i), Some(Token::Whitespace(_))) {
                buf.push(tokens[i].clone());
                i += 1;
            }
            if matches!(tokens.get(i), Some(Token::LParen)) {
                let mut depth = 0i32;
                while let Some(tok) = tokens.get(i) {
                    match tok {
                        Token::LParen => depth += 1,
                        Token::RParen => depth -= 1,
                        _ => {}
                    }
                    buf.push(tok.clone());
                    i += 1;
                    if depth == 0 {
                        break;
                    }
                }
            } else if matches!(tokens.get(i), Some(Token::Word(_))) {
                buf.push(tokens[i].clone());
                i += 1;
            }
            while matches!(tokens.get(i), Some(Token::Whitespace(_))) {
                buf.push(tokens[i].clone());
                i += 1;
            }
        }
        // Match [NOT] LIKE / NOT ILIKE / GLOB followed by a string literal.
        let mut j = i;
        let mut negated = false;
        if matches!(&tokens.get(j), Some(Token::Word(w)) if w.keyword == Keyword::NOT) {
            negated = true;
            j += 1;
            while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                j += 1;
            }
        }
        let op = match &tokens.get(j) {
            Some(Token::Word(w)) if w.keyword == Keyword::LIKE => {
                Some(if negated { "NL" } else { "L" })
            }
            Some(Token::Word(w)) if w.keyword == Keyword::ILIKE && negated => Some("NI"),
            Some(Token::Word(w)) if w.value.eq_ignore_ascii_case("glob") && !negated => {
                Some("G")
            }
            // `* [NOT] SIMILAR TO 're'` (wave-B): consume the TO here.
            Some(Token::Word(w)) if w.keyword == Keyword::SIMILAR => {
                j += 1;
                while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                    j += 1;
                }
                match tokens.get(j) {
                    Some(Token::Word(w2)) if w2.keyword == Keyword::TO => {
                        Some(if negated { "NS" } else { "S" })
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(code) = op else {
            out.extend(buf);
            continue;
        };
        j += 1;
        while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
            j += 1;
        }
        let Some(Token::SingleQuotedString(pat)) = tokens.get(j) else {
            out.extend(buf);
            continue;
        };
        // sqlparser parses ILIKE and EXCLUDE as mutually exclusive, so the
        // buffered EXCLUDE entries ride INSIDE the marker (`\u{2}`-split);
        // quoted identifiers bail to a parse error (clean) rather than a
        // lossy encoding.
        let mut exc = String::new();
        let mut ok = true;
        for t in &buf {
            match t {
                Token::Word(w) if w.value.eq_ignore_ascii_case("exclude") => {}
                Token::Word(w) if w.quote_style.is_none() => exc.push_str(&w.value),
                Token::Period => exc.push('.'),
                Token::Comma => exc.push(','),
                Token::LParen | Token::RParen | Token::Whitespace(_) => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            out.extend(buf);
            continue;
        }
        out.push(Token::Whitespace(Whitespace::Space));
        out.push(Token::make_keyword("ILIKE"));
        out.push(Token::Whitespace(Whitespace::Space));
        out.push(Token::SingleQuotedString(format!(
            "\u{1}{code}:{exc}\u{2}{pat}"
        )));
        i = j + 1;
    }
    out
}

/// Rewrite infix `expr GLOB pat` into `expr LIKE __glob_pat(pat)` — sqlparser
/// cannot parse GLOB (it eats it as an implicit alias), and GLOB is NOT
/// expressible as a LIKE pattern (wave-5 pins), so the marker routes the
/// binder to the dedicated byte matcher. The wrap covers the next PRIMARY
/// (literal / identifier chain / parenthesized group); it is an identity
/// marker, so a pattern continuing past the primary (`'a' || x`) still
/// computes correctly once the binder unwraps it in place. `NOT GLOB` is a
/// DuckDB parse error and is deliberately not rewritten (prev token NOT is
/// not a value end), so it stays a parse error here too.
pub fn rewrite_glob(tokens: Vec<Token>) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let is_glob = matches!(&tokens[i], Token::Word(w)
            if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("glob"));
        let prev = out.iter().rev().find(|t| !matches!(t, Token::Whitespace(_)));
        if !(is_glob && ends_value(prev)) {
            out.push(tokens[i].clone());
            i += 1;
            continue;
        }
        out.push(Token::make_keyword("LIKE"));
        out.push(Token::Whitespace(Whitespace::Space));
        out.push(Token::make_word("__glob_pat", None));
        out.push(Token::LParen);
        i += 1;
        while matches!(tokens.get(i), Some(Token::Whitespace(_))) {
            i += 1;
        }
        // Copy one primary: a balanced group or a single token, then
        // directly-attached `.word` / `(...)` / `[...]` chains.
        let mut depth = 0i32;
        while let Some(tok) = tokens.get(i) {
            match tok {
                Token::LParen | Token::LBracket => {
                    depth += 1;
                    out.push(tok.clone());
                    i += 1;
                }
                Token::RParen | Token::RBracket => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    out.push(tok.clone());
                    i += 1;
                    if depth == 0 && !matches!(tokens.get(i), Some(Token::Period)) {
                        break;
                    }
                }
                _ if depth > 0 => {
                    out.push(tok.clone());
                    i += 1;
                }
                Token::Period => {
                    out.push(tok.clone());
                    i += 1;
                }
                Token::Whitespace(_) => break,
                _ => {
                    out.push(tok.clone());
                    i += 1;
                    if !matches!(
                        tokens.get(i),
                        Some(Token::Period) | Some(Token::LParen) | Some(Token::LBracket)
                    ) {
                        break;
                    }
                }
            }
        }
        out.push(Token::RParen);
    }
    out
}

/// Rewrite the paren-less star REPLACE — `* REPLACE expr AS name` — into
/// the parenthesized form sqlparser accepts. DuckDB's paren-less form
/// consumes exactly ONE item: a following comma starts a NEW select item
/// (measured — `* REPLACE i+100 AS i, j+1 AS j` yields three columns
/// i,j,j; pins-waveA/columns-replace.json), which terminating the wrap at
/// the top-level `AS name` reproduces exactly. sqlparser already parses
/// paren-less EXCLUDE/RENAME singles. A `3 * replace(...)` multiplication
/// is untouched: function calls have `(` right after the word.
pub fn rewrite_parenless_replace(tokens: Vec<Token>) -> Vec<Token> {
    const STOP: &[Keyword] = &[
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
    let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let is_replace = matches!(&tokens[i], Token::Word(w)
            if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("replace"));
        // The `*` before REPLACE must be a WILDCARD (after SELECT / a
        // comma / a qualifying period), not multiplication.
        let prev_is_wildcard_star = {
            let mut it = out.iter().rev().filter(|t| !matches!(t, Token::Whitespace(_)));
            matches!(it.next(), Some(Token::Mul))
                && match it.next() {
                    Some(Token::Word(w)) => {
                        w.quote_style.is_none() && w.keyword == Keyword::SELECT
                    }
                    Some(Token::Comma) | Some(Token::Period) => true,
                    _ => false,
                }
        };
        if !(is_replace && prev_is_wildcard_star) {
            out.push(tokens[i].clone());
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
            j += 1;
        }
        if matches!(tokens.get(j), Some(Token::LParen)) {
            // Already parenthesized.
            out.push(tokens[i].clone());
            i += 1;
            continue;
        }
        // Buffer exactly `expr AS name`; bail on any other shape so the
        // original stream (and its parse error) survives unchanged.
        let mut buf: Vec<Token> = Vec::new();
        let mut depth = 0i32;
        let mut ok = false;
        while let Some(tok) = tokens.get(j) {
            match tok {
                Token::LParen | Token::LBracket => {
                    depth += 1;
                    buf.push(tok.clone());
                    j += 1;
                }
                Token::RParen | Token::RBracket => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    buf.push(tok.clone());
                    j += 1;
                }
                Token::Comma if depth == 0 => break,
                Token::Word(w)
                    if depth == 0 && w.quote_style.is_none() && w.keyword == Keyword::AS =>
                {
                    if !buf.iter().any(|t| !matches!(t, Token::Whitespace(_))) {
                        break; // `REPLACE AS x` — no expression, bail
                    }
                    buf.push(tok.clone());
                    j += 1;
                    while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                        buf.push(tokens[j].clone());
                        j += 1;
                    }
                    if let Some(name @ Token::Word(_)) = tokens.get(j) {
                        buf.push(name.clone());
                        j += 1;
                        ok = true;
                    }
                    break;
                }
                Token::Word(w)
                    if depth == 0 && w.quote_style.is_none() && STOP.contains(&w.keyword) =>
                {
                    break
                }
                _ => {
                    buf.push(tok.clone());
                    j += 1;
                }
            }
        }
        if !ok {
            out.push(tokens[i].clone());
            i += 1;
            continue;
        }
        out.push(tokens[i].clone()); // REPLACE
        out.push(Token::Whitespace(Whitespace::Space));
        out.push(Token::LParen);
        out.extend(buf);
        out.push(Token::RParen);
        i = j;
    }
    out
}

/// Rewrite DuckDB's FROM-position prefix alias `FROM x : T` into
/// `FROM T AS x` — measured IDENTICAL to the AS form in every probed
/// behavior: shadowing, duplicate-alias laxity, binder errors
/// (pins-waveA/from-colon-alias.json). Fires when a (possibly quoted)
/// single identifier followed by `:` sits at a table-ref position (right
/// after FROM, JOIN, or a FROM-list comma). Only ident-chain right sides
/// (`T`, `s.T`) are rewritten; a right side continuing with `(` is a
/// table function / subquery — left alone, so it stays the same clean
/// parse error as before (unsupported relation kinds regardless).
pub fn rewrite_from_colon_aliases(tokens: Vec<Token>) -> Vec<Token> {
    const FROM_END: &[Keyword] = &[
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
    let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut fromctx: Vec<i32> = Vec::new(); // paren depths of active FROM lists
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut table_pos = false;

    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if matches!(t, Token::Whitespace(_)) {
            out.push(t.clone());
            i += 1;
            continue;
        }
        let in_ctx = fromctx.last().is_some_and(|d| *d == paren && bracket == 0);
        match t {
            Token::Word(w) if w.quote_style.is_none() && w.keyword == Keyword::FROM => {
                out.push(t.clone());
                fromctx.push(paren);
                table_pos = true;
            }
            Token::Word(w)
                if in_ctx && w.quote_style.is_none() && FROM_END.contains(&w.keyword) =>
            {
                fromctx.pop();
                out.push(t.clone());
                table_pos = false;
            }
            Token::Word(w) if in_ctx && w.quote_style.is_none() && w.keyword == Keyword::JOIN => {
                out.push(t.clone());
                table_pos = true;
            }
            Token::Comma if in_ctx => {
                out.push(t.clone());
                table_pos = true;
            }
            Token::Word(alias) if table_pos => {
                // Lookahead for `alias : ident(.ident)*`.
                let mut j = i + 1;
                while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                    j += 1;
                }
                if matches!(tokens.get(j), Some(Token::Colon)) {
                    j += 1;
                    while matches!(tokens.get(j), Some(Token::Whitespace(_))) {
                        j += 1;
                    }
                    let mut chain: Vec<Token> = Vec::new();
                    while let Some(Token::Word(_)) = tokens.get(j) {
                        chain.push(tokens[j].clone());
                        j += 1;
                        if matches!(tokens.get(j), Some(Token::Period)) {
                            chain.push(tokens[j].clone());
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    let chain_ok = matches!(chain.last(), Some(Token::Word(_)))
                        && !matches!(tokens.get(j), Some(Token::LParen));
                    if chain_ok {
                        out.extend(chain);
                        out.push(Token::Whitespace(Whitespace::Space));
                        out.push(Token::make_keyword("AS"));
                        out.push(Token::Whitespace(Whitespace::Space));
                        out.push(Token::Word(alias.clone()));
                        table_pos = false;
                        i = j;
                        continue;
                    }
                }
                out.push(t.clone());
                table_pos = false;
            }
            Token::LParen => {
                paren += 1;
                out.push(t.clone());
                table_pos = false;
            }
            Token::RParen => {
                paren -= 1;
                while fromctx.last().is_some_and(|d| *d > paren) {
                    fromctx.pop();
                }
                out.push(t.clone());
                table_pos = false;
            }
            Token::LBracket => {
                bracket += 1;
                out.push(t.clone());
                table_pos = false;
            }
            Token::RBracket => {
                bracket -= 1;
                out.push(t.clone());
                table_pos = false;
            }
            _ => {
                out.push(t.clone());
                table_pos = false;
            }
        }
        i += 1;
    }
    out
}

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
