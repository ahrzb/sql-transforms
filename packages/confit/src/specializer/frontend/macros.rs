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
//!
//! A body that reads one subexpression in several places can declare it
//! once, as a let ([`SqlMacro::lets`]): the body and the later lets read it
//! as `__cf_let(i)`. Spelled out, such a body grows with every read (a
//! recurrence whose every step reads the previous one twice doubles per
//! step); declared, each call's lets expand once, beside the query
//! ([`Expanded::lets`]), and the binder binds each once (`lets.rs`). The
//! meaning is the body with every let spelled out, which is what DuckDB
//! runs (`confit.SqlFunction` renders both forms from one expression).

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
    /// The subexpressions the body reads more than once, in order: each
    /// over the parameters and the lets before it, read as `__cf_let(i)`
    /// (see the module doc). Empty for a body spelled out.
    pub lets: Vec<String>,
}

/// Expansions per query before the pass refuses: a body that calls itself,
/// directly or through another function, never stops expanding.
const MAX_EXPANSIONS: usize = 2_000;
/// Tokens after expansion: a chain of functions each using its parameter
/// twice doubles per level, so the bound is checked before each splice.
pub(super) const MAX_TOKENS: usize = 4_000_000;

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

/// A let read: `__cf_let(i)`, in a function's body and lets for its own
/// let `i`, and in the expanded query for entry `i` of [`Expanded::lets`].
/// The name is reserved.
pub const LET_MARKER: &str = "__cf_let";

/// One let of one call: its function's name and its expansion, `(let)`
/// with the call's arguments substituted.
#[derive(Clone, Debug)]
pub struct Let {
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
    /// Every call's lets, numbered as the query reads them.
    pub lets: Vec<Let>,
}

/// A body or a let, tokenized once: its tokens, and the positions of the
/// numbers in its `__cf_let(i)` reads, which a call renumbers.
struct Text {
    tokens: Vec<Token>,
    reads: Vec<(usize, usize)>,
}

struct State<'m> {
    macros: &'m [SqlMacro],
    bodies: Vec<Text>,
    lets: Vec<Vec<Text>>,
    rounds: usize,
    tokens: usize,
    calls: Vec<Call>,
    keys: Vec<(usize, String)>,
    expanded_lets: Vec<Let>,
}

/// `text` tokenized, with its let reads: `__cf_let(i)` with `i` below
/// `bound`. Any other use of the reserved name refuses.
fn text(m: &SqlMacro, what: &str, src: &str, bound: usize) -> Result<Text, PrepareError> {
    let tokens = tokenize(src)
        .map_err(|e| PrepareError::Bind(format!("sql function '{}': its {what}: {e}", m.name)))?;
    let mut reads = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let Token::Word(w) = t else { continue };
        if !w.value.eq_ignore_ascii_case(LET_MARKER) {
            continue;
        }
        let read = (|| {
            let (open, _) = next_solid(&tokens, i).filter(|(_, t)| matches!(t, Token::LParen))?;
            let (num, n) = next_solid(&tokens, open)?;
            let Token::Number(n, false) = n else { return None };
            let n: usize = n.parse().ok().filter(|&n| n < bound)?;
            next_solid(&tokens, num).filter(|(_, t)| matches!(t, Token::RParen))?;
            Some((num, n))
        })();
        match read {
            Some(r) => reads.push(r),
            None => {
                return Err(unsup(format!(
                    "sql function '{}': its {what} reads {LET_MARKER} other than one of \
                     the lets before it",
                    m.name
                )))
            }
        }
    }
    Ok(Text { tokens, reads })
}

/// Replace every call of a declared SQL function by its body, innermost
/// last: an argument's own calls, and calls in a body, expand in later
/// rounds. A call read by field becomes a [`CALL_MARKER`].
pub fn expand(tokens: Vec<Token>, macros: &[SqlMacro]) -> Result<Expanded, PrepareError> {
    if macros.is_empty() {
        return Ok(Expanded {
            tokens,
            calls: Vec::new(),
            lets: Vec::new(),
        });
    }
    // The markers the token rewrites reserve are as invalid in a body as
    // in the query it lands in; a let read is the only marker a body has.
    let reserved = |m: &SqlMacro, src: &str| {
        let lower = src.to_ascii_lowercase();
        if src.contains('\u{1}')
            || lower.contains("__glob_pat")
            || lower.contains(super::structs::SEQ_MARKER)
            || lower.contains(CALL_MARKER)
        {
            return Err(unsup(format!(
                "sql function '{}': a reserved marker in its body",
                m.name
            )));
        }
        Ok(())
    };
    let mut bodies = Vec::with_capacity(macros.len());
    let mut lets = Vec::with_capacity(macros.len());
    for m in macros {
        reserved(m, &m.body)?;
        bodies.push(text(m, "body", &m.body, m.lets.len())?);
        let mut texts = Vec::with_capacity(m.lets.len());
        for (i, l) in m.lets.iter().enumerate() {
            reserved(m, l)?;
            texts.push(text(m, "let", l, i)?);
        }
        lets.push(texts);
    }
    let mut st = State {
        macros,
        bodies,
        lets,
        rounds: 0,
        tokens: 0,
        calls: Vec::new(),
        keys: Vec::new(),
        expanded_lets: Vec::new(),
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
        lets: st.expanded_lets,
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

/// `(body)` with every parameter reference spelled `(argument)`, and its
/// own let `i` read as the query's let `first + i`. A let is not wrapped:
/// its text stands at each read as it is.
fn substitute(
    text: &Text,
    mac: &SqlMacro,
    args: &[Vec<Token>],
    first: usize,
    wrap: bool,
) -> Vec<Token> {
    let body = &text.tokens;
    let mut rep = Vec::with_capacity(body.len() + 2);
    if wrap {
        rep.push(Token::LParen);
    }
    let mut reads = text.reads.iter().peekable();
    for (j, t) in body.iter().enumerate() {
        if let Some(&&(at, i)) = reads.peek() {
            if at == j {
                reads.next();
                rep.push(Token::Number((first + i).to_string(), false));
                continue;
            }
        }
        let param = match t {
            // A quoted name, or a bare one that is not a keyword: a bare
            // `end` is CASE's, never a parameter.
            Token::Word(w)
                if (w.quote_style == Some('"')
                    || (w.quote_style.is_none() && w.keyword == Keyword::NoKeyword))
                    && is_reference(body, j) =>
            {
                mac.params
                    .iter()
                    .position(|p| p.eq_ignore_ascii_case(&w.value))
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
    if wrap {
        rep.push(Token::RParen);
    }
    rep
}

/// One pass over `tokens`, building the output as it goes: a call's
/// arguments and its body expand recursively before they are appended, so
/// no splice ever shifts the rest of a growing token vector (a query reading
/// n fields of one wide function spliced n times into an O(n) vector).
fn expand_in(tokens: Vec<Token>, st: &mut State<'_>) -> Result<Vec<Token>, PrepareError> {
    stacker::maybe_grow(super::RED_ZONE, super::STACK_SEGMENT, || expand_here(&tokens, st))
}

fn expand_here(toks: &[Token], st: &mut State<'_>) -> Result<Vec<Token>, PrepareError> {
    let macros = st.macros;
    let mut out: Vec<Token> = Vec::with_capacity(toks.len());
    let mut i = 0usize;
    while let Some((at, m)) = find_call(toks, i, macros) {
        out.extend_from_slice(&toks[i..at]);
        let open = next_solid(toks, at).expect("found before a paren").0;
        let (args, end) = split_args(toks, open)?;
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
        let by_field = read_by_field(toks, end);
        // A call read by field and already expanded is just its marker:
        // building its expansion again, per read, was quadratic.
        let key: String = args
            .iter()
            .map(|a| a.iter().map(|t| t.to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\u{0}");
        let seen = if by_field {
            st.keys.iter().position(|k| k.0 == m && k.1 == key)
        } else {
            None
        };
        // Only an expansion counts toward the cap, which stops a body that
        // calls itself; a further read of an expanded call is a marker.
        let rep = if seen.is_some() {
            Vec::new()
        } else {
            st.rounds += 1;
            if st.rounds > MAX_EXPANSIONS {
                return Err(unsup(format!(
                    "sql function '{}' expands more than {MAX_EXPANSIONS} times (a body \
                     that calls itself?)",
                    macros[m].name
                )));
            }
            // This call's lets take the next numbers; each expands once,
            // its own calls included, beside the query.
            let first = st.expanded_lets.len();
            let reps: Vec<Vec<Token>> = st.lets[m]
                .iter()
                .map(|l| substitute(l, mac, &args, first, false))
                .collect();
            st.expanded_lets.extend(reps.iter().map(|_| Let {
                name: mac.name.clone(),
                tokens: Vec::new(),
            }));
            for (i, r) in reps.into_iter().enumerate() {
                let toks = expand_in(r, st)?;
                st.tokens += toks.len();
                st.expanded_lets[first + i].tokens = toks;
            }
            substitute(&st.bodies[m], mac, &args, first, true)
        };
        if by_field {
            let id = match seen {
                Some(id) => id,
                None => {
                    st.tokens += rep.len();
                    st.keys.push((m, key));
                    st.calls.push(Call {
                        name: mac.name.clone(),
                        tokens: rep,
                    });
                    st.calls.len() - 1
                }
            };
            out.extend([
                Token::LParen,
                Token::make_word(CALL_MARKER, None),
                Token::LParen,
                Token::Number(id.to_string(), false),
            ]);
            for a in args {
                out.push(Token::Comma);
                out.extend(expand_in(a, st)?);
            }
            out.push(Token::RParen);
            out.push(Token::RParen);
        } else {
            out.extend(expand_in(rep, st)?);
        }
        if out.len() + st.tokens > MAX_TOKENS {
            return Err(unsup(format!(
                "sql function '{}' expands past {MAX_TOKENS} tokens",
                mac.name
            )));
        }
        i = end;
    }
    out.extend_from_slice(&toks[i..]);
    Ok(out)
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
            lets: Vec::new(),
        }
    }

    fn mac_lets(name: &str, params: &[&str], body: &str, lets: &[&str]) -> SqlMacro {
        SqlMacro {
            lets: lets.iter().map(|l| l.to_string()).collect(),
            ..mac(name, params, body)
        }
    }

    #[test]
    fn a_let_expands_once_beside_the_query() {
        let m = [mac_lets("f", &["x"], "__cf_let(0) * __cf_let(0)", &["x + 1"])];
        let e = expand(tokenize("SELECT f(a) FROM t").unwrap(), &m).unwrap();
        assert_eq!(text(&e.tokens), "SELECT (__cf_let(0) * __cf_let(0)) FROM t");
        assert_eq!(e.lets.len(), 1);
        assert_eq!(e.lets[0].name, "f");
        assert_eq!(text(&e.lets[0].tokens), "(a) + 1");
    }

    #[test]
    fn each_call_numbers_its_lets_after_the_last() {
        let m = [mac_lets(
            "f",
            &["x"],
            "__cf_let(1) + __cf_let(0)",
            &["x * 2", "__cf_let(0) - x"],
        )];
        let e = expand(tokenize("SELECT f(a), f(b) FROM t").unwrap(), &m).unwrap();
        assert_eq!(
            text(&e.tokens),
            "SELECT (__cf_let(1) + __cf_let(0)), (__cf_let(3) + __cf_let(2)) FROM t"
        );
        let lets: Vec<String> = e.lets.iter().map(|l| text(&l.tokens)).collect();
        assert_eq!(lets, ["(a) * 2", "__cf_let(0) - (a)", "(b) * 2", "__cf_let(2) - (b)"]);
    }

    #[test]
    fn a_call_inside_a_let_expands_with_its_own_lets() {
        let m = [
            mac_lets("g", &["y"], "__cf_let(0) / __cf_let(0)", &["y + 1"]),
            mac_lets("f", &["x"], "__cf_let(0) * __cf_let(0)", &["g(x)"]),
        ];
        let e = expand(tokenize("SELECT f(a) FROM t").unwrap(), &m).unwrap();
        assert_eq!(text(&e.tokens), "SELECT (__cf_let(0) * __cf_let(0)) FROM t");
        let lets: Vec<String> = e.lets.iter().map(|l| text(&l.tokens)).collect();
        assert_eq!(lets, ["(__cf_let(1) / __cf_let(1))", "((a)) + 1"]);
    }

    #[test]
    fn a_let_read_must_name_an_earlier_let() {
        for (body, lets) in [
            ("__cf_let(1)", vec!["x"]),
            ("__cf_let(0)", vec!["__cf_let(0)"]),
            ("__cf_let", vec![]),
            ("__cf_let(a)", vec!["x"]),
        ] {
            let m = [mac_lets("f", &["x"], body, &lets)];
            assert!(
                matches!(run("SELECT f(a) FROM t", &m), Err(PrepareError::Unsupported(_))),
                "{body} {lets:?}"
            );
        }
    }

    #[test]
    fn a_doubling_chain_of_lets_stays_linear() {
        // Each step reads the one before twice: spelled out, 2^40 reads.
        let lets: Vec<String> = std::iter::once("x".to_string())
            .chain((1..40).map(|i| format!("__cf_let({}) + __cf_let({})", i - 1, i - 1)))
            .collect();
        let lets: Vec<&str> = lets.iter().map(String::as_str).collect();
        let m = [mac_lets("f", &["x"], "__cf_let(39)", &lets)];
        let e = expand(tokenize("SELECT f(a) FROM t").unwrap(), &m).unwrap();
        assert_eq!(e.lets.len(), 40);
        assert!(e.lets.iter().map(|l| l.tokens.len()).sum::<usize>() < 40 * 16);
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
    fn reads_of_one_call_do_not_count_as_expansions() {
        // 3000 reads of one call: one expansion, under the cap.
        let m = [mac("f", &["x"], "struct_pack(p := x)")];
        let reads: Vec<String> = (0..3000).map(|i| format!("f(a).p AS o{i}")).collect();
        let e = expand(
            tokenize(&format!("SELECT {} FROM t", reads.join(", "))).unwrap(),
            &m,
        )
        .unwrap();
        assert_eq!(e.calls.len(), 1);
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
