//! SQL functions (`confit.SqlFunction`): a call is replaced by its body.
//!
//! DuckDB registers a SQL function as a macro (`CREATE MACRO f(x) AS body`)
//! and binds a call by substituting each argument expression for its
//! parameter in the body. This pass does the same substitution on the token
//! stream, before the query is parsed: `f(a + 1)` becomes `(body)` with every
//! parameter reference spelled `((a + 1))`. Both engines then bind the same
//! expression, so the function's meaning cannot drift between them.
//!
//! Expanding tokens rather than the parsed tree keeps a body's struct field
//! names (`(e)."p"`, `struct_pack("p" := e)`) from being mistaken for
//! parameter references, which an expression walk would visit as plain
//! identifiers. The rule for a parameter reference is the binder's for a
//! column: a word not after a `.`, not a named argument's name, not a call.

use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Tokenizer};

use super::refusal::unsup;
use super::PrepareError;

/// One SQL function: its name, parameter names in call order, and body.
#[derive(Clone, Debug)]
pub struct SqlMacro {
    pub name: String,
    pub params: Vec<String>,
    pub body: String,
}

/// Expansions per query before the pass refuses: a body that calls itself,
/// directly or through another function, never stops expanding.
const MAX_EXPANSIONS: usize = 2_000;
/// Tokens after expansion: a chain of functions each using its parameter
/// twice doubles per level, so the bound is checked before each splice.
const MAX_TOKENS: usize = 4_000_000;

fn is_ws(t: &Token) -> bool {
    matches!(t, Token::Whitespace(_))
}

fn prev_solid(toks: &[Token], i: usize) -> Option<&Token> {
    toks[..i].iter().rev().find(|t| !is_ws(t))
}

fn next_solid(toks: &[Token], i: usize) -> Option<(usize, &Token)> {
    toks.iter()
        .enumerate()
        .skip(i + 1)
        .find(|(_, t)| !is_ws(t))
}

/// A word that names something in expression position: not a field after
/// `.`, not a named argument's name (`p := e`, `p => e`).
fn is_reference(toks: &[Token], i: usize) -> bool {
    !matches!(prev_solid(toks, i), Some(Token::Period))
        && !matches!(
            next_solid(toks, i).map(|(_, t)| t),
            Some(Token::Assignment | Token::RArrow)
        )
}

/// The first call of a declared function: `(name position, macro index)`.
fn find_call(toks: &[Token], from: usize, macros: &[SqlMacro]) -> Option<(usize, usize)> {
    for (i, t) in toks.iter().enumerate().skip(from) {
        let Token::Word(w) = t else { continue };
        let Some(m) = macros
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(&w.value))
        else {
            continue;
        };
        if is_reference(toks, i) && matches!(next_solid(toks, i), Some((_, Token::LParen))) {
            return Some((i, m));
        }
    }
    None
}

/// The arguments of the call whose `(` is at `open`, split at top-level
/// commas, and the index just past its `)`.
fn split_args(toks: &[Token], open: usize) -> Result<(Vec<Vec<Token>>, usize), PrepareError> {
    let mut args: Vec<Vec<Token>> = vec![Vec::new()];
    let mut depth = 0usize;
    for (i, t) in toks.iter().enumerate().skip(open) {
        match t {
            Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => {
                depth -= 1;
                if depth == 0 {
                    if args.len() == 1 && args[0].iter().all(is_ws) {
                        args.clear();
                    } else if args.iter().any(|a| a.iter().all(is_ws)) {
                        return Err(PrepareError::Parse("an empty function argument".into()));
                    }
                    return Ok((args, i + 1));
                }
            }
            Token::Comma if depth == 1 => {
                args.push(Vec::new());
                continue;
            }
            _ => {}
        }
        if i > open {
            args.last_mut().expect("one argument at least").push(t.clone());
        }
    }
    Err(PrepareError::Parse("unbalanced parentheses in a function call".into()))
}

fn tokenize(sql: &str) -> Result<Vec<Token>, PrepareError> {
    Tokenizer::new(&GenericDialect {}, sql)
        .tokenize()
        .map_err(|e| PrepareError::Parse(e.to_string()))
}

/// The marker a struct field read over a SQL function call becomes:
/// `(__cf_call(id, arg1, ..., argn)).field`, `id` indexing [`Expanded::calls`].
/// The arguments stay in place, so whatever walks the query sees them; the
/// body, with the arguments substituted, is expanded and parsed once per
/// distinct call (see [`Expanded`]). The name is reserved.
pub const CALL_MARKER: &str = "__cf_call";

/// One distinct SQL function call read by field: the function's name (an
/// unaliased read is named after the call, as DuckDB names it) and its
/// expansion, `(body)` with the arguments substituted.
#[derive(Clone, Debug)]
pub struct Call {
    pub name: String,
    pub tokens: Vec<Token>,
}

/// The expanded query, and the calls it reads fields of. A call read by
/// several fields (`f(x).a, f(x).b, ...`) is expanded once here rather than
/// once per read, which made a function with n fields read n times cost n^2
/// tokens.
#[derive(Debug)]
pub struct Expanded {
    pub tokens: Vec<Token>,
    pub calls: Vec<Call>,
}

struct State<'m> {
    macros: &'m [SqlMacro],
    bodies: Vec<Vec<Token>>,
    /// Per macro, its ASCII-lowercased parameter names to positions.
    params: Vec<std::collections::HashMap<String, usize>>,
    rounds: usize,
    tokens: usize,
    calls: Vec<Call>,
    keys: Vec<(usize, String)>,
}

/// Replace every call of a declared SQL function by its body, innermost
/// last: an argument's own calls, and calls in a body, expand in later
/// rounds. A call read by field becomes a [`CALL_MARKER`].
pub fn expand(tokens: Vec<Token>, macros: &[SqlMacro]) -> Result<Expanded, PrepareError> {
    if macros.is_empty() {
        return Ok(Expanded {
            tokens,
            calls: Vec::new(),
        });
    }
    let bodies = macros
        .iter()
        .map(|m| {
            // The markers the token rewrites reserve are as invalid in a
            // body as in the query it lands in.
            let lower = m.body.to_ascii_lowercase();
            if m.body.contains('\u{1}')
                || lower.contains("__glob_pat")
                || lower.contains(super::structs::SEQ_MARKER)
                || lower.contains(CALL_MARKER)
            {
                return Err(unsup(format!(
                    "sql function '{}': a reserved marker in its body",
                    m.name
                )));
            }
            tokenize(&m.body).map_err(|e| {
                PrepareError::Bind(format!("sql function '{}': its body: {e}", m.name))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let params = macros
        .iter()
        .map(|m| {
            let mut map = std::collections::HashMap::new();
            for (i, p) in m.params.iter().enumerate() {
                // The first of two names equal but for case wins, as the
                // linear search did.
                map.entry(p.to_ascii_lowercase()).or_insert(i);
            }
            map
        })
        .collect();
    let mut st = State {
        macros,
        bodies,
        params,
        rounds: 0,
        tokens: tokens.len(),
        calls: Vec::new(),
        keys: Vec::new(),
    };
    let tokens = expand_in(tokens, &mut st)?;
    // A call's expansion may hold calls of its own, read by field or not.
    let mut i = 0;
    while i < st.calls.len() {
        let toks = std::mem::take(&mut st.calls[i].tokens);
        st.calls[i].tokens = expand_in(toks, &mut st)?;
        i += 1;
    }
    Ok(Expanded {
        tokens,
        calls: st.calls,
    })
}

/// Is the call ending just before `end` read by field (`f(x).name`)?
fn read_by_field(toks: &[Token], end: usize) -> bool {
    let next = toks[end..].iter().position(|t| !is_ws(t)).map(|i| end + i);
    let Some(dot) = next.filter(|&i| matches!(toks[i], Token::Period)) else {
        return false;
    };
    matches!(next_solid(toks, dot), Some((_, Token::Word(_))))
}

/// `(body)` with every parameter reference spelled `(argument)`. `params`
/// maps an ASCII-lowercased parameter name to its position.
fn substitute(
    body: &[Token],
    params: &std::collections::HashMap<String, usize>,
    args: &[Vec<Token>],
) -> Vec<Token> {
    let mut rep = Vec::with_capacity(body.len() + 2);
    rep.push(Token::LParen);
    for (j, t) in body.iter().enumerate() {
        let param = match t {
            // A quoted name, or a bare one that is not a keyword: a
            // bare `end` is CASE's, never a parameter.
            Token::Word(w)
                if (w.quote_style == Some('"')
                    || (w.quote_style.is_none() && w.keyword == Keyword::NoKeyword))
                    && is_reference(body, j) =>
            {
                params
                    .get(&w.value.to_ascii_lowercase())
                    .copied()
                    .filter(|_| !matches!(next_solid(body, j), Some((_, Token::LParen))))
            }
            _ => None,
        };
        match param {
            Some(p) => {
                rep.push(Token::LParen);
                rep.extend(args[p].iter().cloned());
                rep.push(Token::RParen);
            }
            None => rep.push(t.clone()),
        }
    }
    rep.push(Token::RParen);
    rep
}

fn expand_in(tokens: Vec<Token>, st: &mut State<'_>) -> Result<Vec<Token>, PrepareError> {
    let macros = st.macros;
    let mut toks = tokens;
    // Everything before the last expansion is call-free, so the next search
    // starts there.
    let mut from = 0usize;
    while let Some((at, m)) = find_call(&toks, from, macros) {
        from = at;
        st.rounds += 1;
        if st.rounds > MAX_EXPANSIONS {
            return Err(unsup(format!(
                "sql function '{}' expands more than {MAX_EXPANSIONS} times (a body \
                 that calls itself?)",
                macros[m].name
            )));
        }
        let open = next_solid(&toks, at).expect("found before a paren").0;
        let (args, end) = split_args(&toks, open)?;
        let mac = &macros[m];
        if args.len() != mac.params.len() {
            return Err(PrepareError::Bind(format!(
                "sql function '{}' takes {} arguments, got {}",
                mac.name,
                mac.params.len(),
                args.len()
            )));
        }
        let trim = |a: &[Token]| -> Vec<Token> {
            let lo = a.iter().position(|t| !is_ws(t)).unwrap_or(0);
            let hi = a.iter().rposition(|t| !is_ws(t)).map_or(0, |i| i + 1);
            a[lo..hi].to_vec()
        };
        let args: Vec<Vec<Token>> = args.iter().map(|a| trim(a)).collect();
        if read_by_field(&toks, end) {
            // One expansion per distinct call, keyed by its spelling: a
            // call read again reuses it without substituting the body.
            let key: String = args
                .iter()
                .map(|a| a.iter().map(|t| t.to_string()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\u{0}");
            let id = match st.keys.iter().position(|k| *k == (m, key.clone())) {
                Some(id) => id,
                None => {
                    let rep = substitute(&st.bodies[m], &st.params[m], &args);
                    st.tokens += rep.len();
                    st.keys.push((m, key));
                    st.calls.push(Call {
                        name: mac.name.clone(),
                        tokens: rep,
                    });
                    st.calls.len() - 1
                }
            };
            let mut marker = vec![
                Token::LParen,
                Token::make_word(CALL_MARKER, None),
                Token::LParen,
                Token::Number(id.to_string(), false),
            ];
            for a in &args {
                marker.push(Token::Comma);
                marker.extend(a.iter().cloned());
            }
            marker.push(Token::RParen);
            marker.push(Token::RParen);
            toks.splice(at..end, marker);
        } else {
            let rep = substitute(&st.bodies[m], &st.params[m], &args);
            st.tokens = st.tokens - (end - at) + rep.len();
            toks.splice(at..end, rep);
        }
        if st.tokens > MAX_TOKENS {
            return Err(unsup(format!(
                "sql function '{}' expands past {MAX_TOKENS} tokens",
                mac.name
            )));
        }
    }
    Ok(toks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(toks: &[Token]) -> String {
        toks.iter().map(|t| t.to_string()).collect()
    }

    fn run(sql: &str, macros: &[SqlMacro]) -> Result<String, PrepareError> {
        expand(tokenize(sql).unwrap(), macros).map(|t| text(&t.tokens))
    }

    fn mac(name: &str, params: &[&str], body: &str) -> SqlMacro {
        SqlMacro {
            name: name.into(),
            params: params.iter().map(|p| p.to_string()).collect(),
            body: body.into(),
        }
    }

    #[test]
    fn a_call_becomes_its_body_with_arguments_in_parentheses() {
        let m = [mac("f", &["x", "y"], "\"x\" * 2 + y")];
        assert_eq!(
            run("SELECT F(a + 1, g(b, c)) AS o FROM t", &m).unwrap(),
            "SELECT ((a + 1) * 2 + (g(b, c))) AS o FROM t"
        );
    }

    #[test]
    fn field_names_and_named_arguments_are_not_parameters() {
        let m = [mac("f", &["p"], "struct_pack(\"p\" := p, q => p).p + (p).\"p\"")];
        assert_eq!(
            run("SELECT f(a) FROM t", &m).unwrap(),
            "SELECT (struct_pack(\"p\" := (a), q => (a)).p + ((a)).\"p\") FROM t"
        );
    }

    #[test]
    fn nested_calls_expand_in_arguments_and_bodies() {
        let m = [mac("f", &["x"], "x + 1"), mac("g", &["y"], "f(y) * f(y)")];
        assert_eq!(
            run("SELECT g(f(a)) FROM t", &m).unwrap(),
            "SELECT ((((((a) + 1))) + 1) * (((((a) + 1))) + 1)) FROM t"
        );
    }

    #[test]
    fn a_keyword_is_not_a_parameter_unless_quoted() {
        let m = [mac("f", &["end"], "CASE WHEN \"end\" > 0 THEN 1 END")];
        assert_eq!(
            run("SELECT f(a) FROM t", &m).unwrap(),
            "SELECT (CASE WHEN (a) > 0 THEN 1 END) FROM t"
        );
    }

    #[test]
    fn a_qualified_name_or_a_column_is_not_a_call() {
        let m = [mac("f", &["x"], "x")];
        assert_eq!(run("SELECT s.f(a), f FROM t", &m).unwrap(), "SELECT s.f(a), f FROM t");
    }

    #[test]
    fn a_string_or_quoted_text_is_not_expanded() {
        let m = [mac("f", &["x"], "x")];
        assert_eq!(run("SELECT 'f(a)' FROM t", &m).unwrap(), "SELECT 'f(a)' FROM t");
    }

    #[test]
    fn the_wrong_argument_count_refuses() {
        let m = [mac("f", &["x"], "x")];
        assert!(matches!(run("SELECT f(a, b) FROM t", &m), Err(PrepareError::Bind(_))));
        assert!(matches!(run("SELECT f() FROM t", &m), Err(PrepareError::Bind(_))));
        assert!(matches!(run("SELECT f(a,) FROM t", &m), Err(PrepareError::Parse(_))));
    }

    #[test]
    fn a_call_read_by_field_expands_once() {
        let m = [mac("f", &["x"], "struct_pack(p := x, q := x + 1)")];
        let e = expand(tokenize("SELECT f(a).p, f(a).q, f(b).p FROM t").unwrap(), &m).unwrap();
        assert_eq!(
            text(&e.tokens),
            "SELECT (__cf_call(0,a)).p, (__cf_call(0,a)).q, (__cf_call(1,b)).p FROM t"
        );
        assert_eq!(e.calls.len(), 2);
        assert_eq!(text(&e.calls[1].tokens), "(struct_pack(p := (b), q := (b) + 1))");
        // Not a field read: inline, as before.
        let e = expand(tokenize("SELECT f(a) FROM t").unwrap(), &m).unwrap();
        assert!(e.calls.is_empty());
    }

    #[test]
    fn a_recursive_body_refuses() {
        let m = [mac("f", &["x"], "f(x)")];
        assert!(matches!(run("SELECT f(a) FROM t", &m), Err(PrepareError::Unsupported(_))));
        let m = [mac("f", &["x"], "x + x + x + x")];
        let deep = (0..12).fold("a".to_string(), |acc, _| format!("f({acc})"));
        assert!(run(&format!("SELECT {deep} FROM t"), &m).is_err());
    }
}
