//! DuckDB-RE2 -> rust-regex translation (wave-B pins,
//! pins-waveB/re2-vs-rust-regex.json). Raw pass-through is WRONG for any
//! pattern with a Perl class: RE2's \d \w \s \b are ASCII while rust-regex
//! defaults Unicode. The measured fix is this bind-time rewrite (verified
//! byte-for-byte on the differential battery) plus a reject list for
//! constructs that differ irreconcilably. Keep default Unicode mode —
//! `unicode(false)` was measured to BREAK (?i) folding parity.

use super::frontend::PrepareError;

fn unsup(what: impl Into<String>) -> PrepareError {
    PrepareError::Unsupported(what.into())
}

/// Parsed DuckDB regex options string (wave-B pins: parsed left-to-right,
/// whitespace skipped, LAST conflicting letter wins; `m`/`n`/`p` are
/// functional no-ops — multiline anchors exist only via inline (?m)).
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReOptions {
    pub case_insensitive: bool,
    pub dotall: bool,
    pub literal: bool,
    pub global: bool,
}

/// `allow_g`: only regexp_replace accepts 'g' (pinned error otherwise).
pub fn parse_options(opts: &str, allow_g: bool) -> Result<ReOptions, PrepareError> {
    let mut o = ReOptions::default();
    for c in opts.chars() {
        match c {
            'c' => o.case_insensitive = false,
            'i' => o.case_insensitive = true,
            'l' => o.literal = true,
            's' => o.dotall = true,
            // Measured no-ops in 1.5.5 (do NOT wire to multiline/never_nl).
            'm' | 'n' | 'p' => o.dotall = false,
            'g' if allow_g => o.global = true,
            'g' => {
                return Err(PrepareError::Bind(
                    "Option 'g' (global replace) is only valid for regexp_replace".into(),
                ))
            }
            w if w.is_whitespace() => {}
            other => {
                return Err(PrepareError::Bind(format!(
                    "Unrecognized Regex option {other}"
                )))
            }
        }
    }
    Ok(o)
}

/// Strict RE2 repetition bounds at `p[i..]` (`p.as_bytes()[i] == b'{'`):
/// `{m}` / `{m,}` / `{m,n}`, ASCII digits only, no whitespace. Returns
/// (index after `}`, largest bound named, product bound — `u64::MAX` for an
/// unbounded `{m,}`).
fn strict_bounds(p: &str, i: usize) -> Option<(usize, u64, u64)> {
    let rest = p.get(i + 1..)?;
    let body = &rest[..rest.find('}')?];
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    let mut parts = body.splitn(2, ',');
    let lo: u64 = parts.next().filter(|s| digits(s))?.parse().ok()?;
    let (max, prod) = match parts.next() {
        None => (lo, lo),
        Some("") => (lo, u64::MAX),
        Some(hi) if digits(hi) => {
            let hi: u64 = hi.parse().ok()?;
            (lo.max(hi), hi)
        }
        Some(_) => return None,
    };
    Some((i + 1 + body.len() + 1, max, prod))
}

/// True if the pattern is nothing but text anchors (`^ $ \A \z`), `\b`, and
/// flag-group / empty `(?:)` noise, with at least two anchors. DuckDB's ROW
/// path literal-optimizes these into string equality ('' only) while its own
/// CONSTANT fold matches normally (fuzzer-measured: '$\z' over a table is
/// FALSE for 'hello' but TRUE as a constant) — unservable either way.
fn anchors_only_multi(p: &str) -> bool {
    let b = p.as_bytes();
    let mut i = 0;
    let mut anchors = 0;
    while i < b.len() {
        match b[i] {
            b'^' | b'$' => {
                anchors += 1;
                i += 1;
            }
            b'\\' => match b.get(i + 1) {
                Some(b'A') | Some(b'z') => {
                    anchors += 1;
                    i += 2;
                }
                Some(b'b') => i += 2,
                _ => return false,
            },
            b'(' if b.get(i + 1) == Some(&b'?') => {
                i += 2;
                if b.get(i) == Some(&b':') {
                    i += 1; // only the EMPTY (?:) counts as noise
                } else {
                    while matches!(
                        b.get(i),
                        Some(b'i' | b'm' | b's' | b'u' | b'U' | b'x' | b'-')
                    ) {
                        i += 1;
                    }
                }
                if b.get(i) != Some(&b')') {
                    return false;
                }
                i += 1;
            }
            _ => return false,
        }
    }
    anchors >= 2
}

/// Rewrite a DuckDB/RE2 pattern into a rust-regex pattern with identical
/// semantics, or reject constructs on the measured divergence list (wave-B
/// pins + the TASK-54 standing fuzzer's findings).
pub fn translate_pattern(p: &str) -> Result<String, PrepareError> {
    if anchors_only_multi(p) {
        return Err(unsup(
            "anchor-only regex pattern (DuckDB's row path diverges from its own \
             constant fold)",
        ));
    }
    let b = p.as_bytes();
    let mut out = String::with_capacity(p.len() + 8);
    let mut i = 0;
    let mut in_class = false;
    // Quantifier state (outside classes): 0 = none, 1 = after a quantifier,
    // 2 = after quantifier + lazy '?'. RE2 rejects ALL further stacking
    // ('a?*', 'a{2}+', 'a*{2}', 'a???') while rust silently reinterprets —
    // wrong-answer risk, reject (fuzzer-measured, not just '*'/'+' pairs).
    let mut quant: u8 = 0;
    // Nested counted repetition: RE2 caps the PRODUCT of nested {m,n} bounds
    // at 1000 ("invalid repetition size") while rust serves — track the max
    // counted bound per open group (plus whether the group captures) and
    // reject past the cap.
    let mut groups: Vec<(u64, bool)> = Vec::new();
    let mut counted: u64 = 0;
    // Character-class member tracking for range-endpoint rejects.
    let mut class_members = 0usize;
    let mut rangey_dash = false; // last member was a bare '-' with a left operand
    // RE2 program-size budget (fuzzer 2026-07-28 seed 20260728: DuckDB
    // errors "pattern too large" on '(\p{L}){1,500}' while rust serves).
    // One-sided over-estimate in "range units": \p/\P weigh 800 (above any
    // property's real range count), classes their member count, literals a
    // flat 4; a counted repetition whose weight product clears 100_000 is
    // rejected — always BEFORE DuckDB's real budget (measured error floor
    // ~216k units), so we can refuse but never serve where DuckDB errors.
    let mut weights: Vec<u64> = Vec::new(); // per open group
    let mut cur_weight: u64 = 0;
    let mut last_weight: u64 = 4;
    let mut class_p: u64 = 0; // \p/\P weight inside the open class
    while i < b.len() {
        let c = b[i];
        match c {
            b'\\' => {
                let Some(&n) = b.get(i + 1) else {
                    // Trailing backslash: both engines reject; pass through
                    // so the compile error surfaces uniformly.
                    out.push('\\');
                    i += 1;
                    continue;
                };
                let perl = matches!(n, b'd' | b'w' | b's');
                if in_class && perl {
                    // '[a-\d]': RE2 rejects a Perl-class range endpoint; the
                    // expansion could compile in rust — reject both sides.
                    if rangey_dash {
                        return Err(unsup(
                            "character class range with a Perl class endpoint",
                        ));
                    }
                    if b.get(i + 2) == Some(&b'-') && b.get(i + 3) != Some(&b']') {
                        return Err(unsup(
                            "character class range with a Perl class endpoint",
                        ));
                    }
                }
                match n {
                    // Perl classes: RE2 is ASCII, rust is Unicode (measured
                    // rewrites; negated forms must be Unicode-mode classes,
                    // and (?-u:..) cannot appear inside [...]).
                    b'd' => out.push_str(if in_class { "0-9" } else { "(?-u:\\d)" }),
                    b'w' => out.push_str(if in_class {
                        "0-9A-Za-z_"
                    } else {
                        "[0-9A-Za-z_]"
                    }),
                    b's' => out.push_str(if in_class {
                        "\\t\\n\\x0c\\r "
                    } else {
                        "[\\t\\n\\x0c\\r ]"
                    }),
                    b'D' | b'W' | b'S' if in_class => {
                        return Err(unsup(
                            "negated Perl class inside a character class (RE2/rust-regex \
                             semantics differ; not translatable in place)",
                        ))
                    }
                    b'D' => out.push_str("[^0-9]"),
                    b'W' => out.push_str("[^0-9A-Za-z_]"),
                    b'S' => out.push_str("[^\\t\\n\\x0c\\r ]"),
                    b'b' if !in_class => out.push_str("(?-u:\\b)"),
                    // \B: DuckDB itself traps at runtime on non-ASCII (RE2's
                    // ASCII \B matches inside multibyte chars) — unservable.
                    b'B' => return Err(unsup("\\B in a regex (unservable RE2 byte semantics)")),
                    // \1-\9 outside a class: RE2 rejects them as backrefs,
                    // but octal mode makes rust read octal escapes — serve
                    // nothing (fuzzer-measured). In-class both are octal.
                    b'1'..=b'9' if !in_class => {
                        return Err(unsup(
                            "backreference-style \\N escape in a regex (RE2 rejects)",
                        ))
                    }
                    // \uXXXX: rust-only extension, DuckDB rejects.
                    b'u' | b'U' => {
                        return Err(unsup("\\u escape in a regex (not RE2 syntax)"))
                    }
                    // \Q...\E literal quoting: RE2-only, rust rejects.
                    b'Q' => return Err(unsup("\\Q...\\E literal quoting in a regex")),
                    other => {
                        out.push('\\');
                        out.push(other as char);
                    }
                }
                let w = if matches!(n, b'p' | b'P') { 800 } else { 8 };
                if in_class {
                    class_p += w;
                } else {
                    last_weight = w;
                    cur_weight = cur_weight.saturating_add(w);
                }
                i += 2;
                quant = 0;
                class_members += 1;
                rangey_dash = false;
            }
            b'[' if !in_class => {
                in_class = true;
                out.push('[');
                i += 1;
                // Consume the class-leading forms so a literal ']' first
                // member ('[]a]' — legal, same meaning in both) survives.
                if b.get(i) == Some(&b'^') {
                    out.push('^');
                    i += 1;
                }
                class_members = 0;
                if b.get(i) == Some(&b']') {
                    out.push(']');
                    i += 1;
                    class_members = 1;
                    // '[]-X]': a range with the literal ']' endpoint to RE2
                    // (backwards ones error), a literal '-' to rust — reject
                    // unless the '-' is the trailing literal ('[]-]').
                    if b.get(i) == Some(&b'-') && b.get(i + 1) != Some(&b']') {
                        return Err(unsup(
                            "character class range starting at a literal ']'",
                        ));
                    }
                }
                rangey_dash = false;
            }
            b'[' if in_class => {
                // POSIX element ('[:alpha:]', '[:^digit:]'): same ASCII
                // semantics in both engines — consume atomically so the
                // class tracker stays in sync. Any other '[' in a class is
                // literal to RE2 but a NESTED CLASS to rust — reject.
                if p[i..].starts_with("[:") {
                    if let Some(end) = p[i..].find(":]") {
                        out.push_str(&p[i..i + end + 2]);
                        i += end + 2;
                        class_members += 1;
                        rangey_dash = false;
                        continue;
                    }
                }
                return Err(unsup(
                    "'[' inside a character class (rust-regex nested-class semantics)",
                ));
            }
            b']' if in_class => {
                in_class = false;
                out.push(']');
                i += 1;
                quant = 0;
                last_weight = 2 + 2 * class_members as u64 + class_p;
                cur_weight = cur_weight.saturating_add(last_weight);
                class_p = 0;
            }
            b'&' | b'~' | b'-' if in_class && b.get(i + 1) == Some(&c) => {
                // '--' / '&&' / '~~' in a class: literals-or-ranges to RE2,
                // set operations to rust — silent wrong answers (measured).
                return Err(unsup(format!(
                    "'{0}{0}' inside a character class (rust-regex set-operation syntax)",
                    c as char
                )));
            }
            b'-' if in_class => {
                rangey_dash = class_members >= 1;
                class_members += 1;
                out.push('-');
                i += 1;
            }
            b'$' if !in_class => {
                // A '$' anchor in NON-final position (not before '|', ')',
                // or the end): DuckDB's row path literal-optimizes a
                // leading '$'+literal into a PREFIX match ('$hello' matches
                // 'hello world'!) while its own constant fold matches
                // normally — self-inconsistent, unservable (fuzzer-measured
                // 2026-07-28, seed 20260728 case 4275).
                let benign = match b.get(i + 1) {
                    None | Some(b'|') | Some(b')') | Some(b'$') => true,
                    // Another anchor right after keeps both paths agreeing
                    // (a '$$h' chain still rejects at ITS last '$').
                    Some(b'\\') => matches!(b.get(i + 2), Some(b'A' | b'z' | b'b' | b'B')),
                    _ => false,
                };
                if !benign {
                    return Err(unsup(
                        "regex '$' anchor in non-final position (DuckDB's row \
                         path diverges from its own constant fold)",
                    ));
                }
                out.push('$');
                i += 1;
                quant = 0;
            }
            b'(' if !in_class => {
                // (?<name>...) angle-bracket groups: DuckDB rejects them.
                if p[i..].starts_with("(?<") && !p[i..].starts_with("(?<=") {
                    return Err(unsup("(?<name>...) capture group (not RE2 syntax)"));
                }
                let capturing =
                    b.get(i + 1) != Some(&b'?') || p[i..].starts_with("(?P<");
                groups.push((counted, capturing));
                counted = 0;
                weights.push(cur_weight);
                cur_weight = 0;
                out.push('(');
                i += 1;
                quant = 0;
            }
            b')' if !in_class => {
                let inner = counted;
                let capturing;
                (counted, capturing) = groups.pop().unwrap_or((0, false));
                if let Some((_, max, prod)) = strict_bounds(p, i + 1)
                    .filter(|_| b.get(i + 1) == Some(&b'{'))
                {
                    // '(x){0}': RE2 keeps the group in its count while rust
                    // ERASES it, shifting every later group number and the
                    // rewrite's MaxSubmatch pre-check (fuzzer-measured).
                    if capturing && max == 0 {
                        return Err(unsup(
                            "capture group under a {0} repetition (rust-regex \
                             erases the group)",
                        ));
                    }
                    if inner > 0 {
                        if inner.saturating_mul(prod) > 1000 {
                            return Err(unsup(
                                "nested counted repetition over RE2's size cap of 1000",
                            ));
                        }
                        counted = counted.max(inner.saturating_mul(prod));
                    }
                } else {
                    counted = counted.max(inner);
                }
                // The group is one atom for a following {m,n}'s budget.
                last_weight = cur_weight.max(1);
                cur_weight = weights.pop().unwrap_or(0).saturating_add(last_weight);
                out.push(')');
                i += 1;
                quant = 0;
            }
            b'{' if !in_class => {
                if let Some((end, max, prod)) = strict_bounds(p, i) {
                    if quant > 0 {
                        return Err(unsup("stacked regex quantifiers (RE2 rejects them)"));
                    }
                    // Repetition bound > 1000 is a DuckDB error; rust allows
                    // larger — reject past the pinned cap.
                    if max > 1000 {
                        return Err(unsup(format!(
                            "regex repetition bound over 1000 ({})",
                            &p[i..end]
                        )));
                    }
                    counted = counted.max(prod.min(1001));
                    if last_weight.saturating_mul(max.max(1)) > 100_000 {
                        return Err(unsup(
                            "counted repetition over RE2's program-size budget \
                             (pattern too large in DuckDB)",
                        ));
                    }
                    cur_weight =
                        cur_weight.saturating_add(last_weight.saturating_mul(max.max(1)));
                    out.push_str(&p[i..end]);
                    i = end;
                    quant = 1;
                } else {
                    // Not strict bounds. Whitespace-padded digits ('{1, 3}')
                    // are LITERAL to RE2 but a repetition to rust — reject;
                    // anything else is literal '{' in both.
                    let body = match p[i + 1..].find('}') {
                        Some(e) => &p[i + 1..i + 1 + e],
                        None => "",
                    };
                    let stripped: String =
                        body.chars().filter(|c| !c.is_whitespace()).collect();
                    if stripped != body
                        && strict_bounds(&format!("{{{stripped}}}"), 0).is_some()
                    {
                        return Err(unsup(format!(
                            "whitespace inside repetition bounds ({{{body}}}) \
                             (literal to RE2, a repetition to rust-regex)"
                        )));
                    }
                    out.push('{');
                    i += 1;
                    quant = 0;
                }
            }
            b'*' | b'+' | b'?' if !in_class => {
                if c == b'?' && quant == 1 {
                    quant = 2; // lazy modifier — the one legal follower
                } else if quant > 0 {
                    return Err(unsup("stacked regex quantifiers (RE2 rejects them)"));
                } else {
                    quant = 1;
                }
                out.push(c as char);
                i += 1;
            }
            _ => {
                // Preserve the raw byte run (UTF-8 safe: we only split at
                // ASCII metacharacters above).
                let start = i;
                i += 1;
                while i < b.len() && !p.is_char_boundary(i) {
                    i += 1;
                }
                out.push_str(&p[start..i]);
                quant = 0;
                class_members += 1;
                rangey_dash = false;
                if !in_class {
                    last_weight = 4;
                    cur_weight = cur_weight.saturating_add(4);
                }
            }
        }
    }
    // Duplicate group names: DuckDB's RE2 accepts them, rust rejects —
    // rust's own compile error would misclassify, so pre-reject.
    let mut names: Vec<&str> = Vec::new();
    let mut j = 0;
    while let Some(pos) = p[j..].find("(?P<") {
        let start = j + pos + 4;
        if let Some(end) = p[start..].find('>') {
            let name = &p[start..start + end];
            if names.contains(&name) {
                return Err(unsup(format!(
                    "duplicate regex capture group name '{name}' (rust-regex rejects)"
                )));
            }
            names.push(name);
            j = start + end;
        } else {
            break;
        }
    }
    Ok(out)
}

/// Translate a DuckDB replacement template (`\N` backrefs, literal `$`)
/// into a rust-regex template (`$N` refs, literal `\`), resolving the
/// pinned invalid-rewrite quirks at bind time.
pub enum Rewrite {
    /// A valid rust template.
    Template(String),
    /// Out-of-range backref (or non-global bad escape): the WHOLE replace
    /// is a silent no-op — bind the subject expression unchanged.
    Identity,
    /// Global bad escape: each match is CONSUMED and only the template
    /// prefix before the bad escape is emitted (measured RE2 quirk).
    ConsumeWithPrefix(String),
}

pub fn translate_rewrite(r: &str, group_count: usize, global: bool) -> Rewrite {
    let b = r.as_bytes();
    // RE2's MaxSubmatch pre-check scans the WHOLE template before any
    // per-match work (fuzzer-measured): an out-of-range \N anywhere is a
    // full no-op even when a bad escape precedes it in the template.
    let mut j = 0;
    while j + 1 < b.len() {
        if b[j] == b'\\' {
            let d = b[j + 1];
            if d.is_ascii_digit() && (d - b'0') as usize > group_count {
                return Rewrite::Identity;
            }
            j += 2;
        } else {
            j += 1;
        }
    }
    let mut out = String::with_capacity(r.len() + 4);
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'$' => {
                out.push_str("$$");
                i += 1;
            }
            b'\\' => match b.get(i + 1) {
                Some(&d) if d.is_ascii_digit() => {
                    let n = (d - b'0') as usize;
                    if n > group_count {
                        // MaxSubmatch pre-check: full no-op, global or not.
                        return Rewrite::Identity;
                    }
                    out.push_str(&format!("${{{n}}}"));
                    i += 2;
                }
                Some(b'\\') => {
                    out.push('\\');
                    i += 2;
                }
                _ => {
                    // Bad escape (incl. trailing lone backslash).
                    return if global {
                        Rewrite::ConsumeWithPrefix(out)
                    } else {
                        Rewrite::Identity
                    };
                }
            },
            _ => {
                let start = i;
                i += 1;
                while i < b.len() && !r.is_char_boundary(i) {
                    i += 1;
                }
                out.push_str(&r[start..i]);
            }
        }
    }
    Rewrite::Template(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perl_classes_rewrite_in_and_out_of_classes() {
        assert_eq!(translate_pattern(r"\d+").unwrap(), r"(?-u:\d)+");
        assert_eq!(translate_pattern(r"[\d]").unwrap(), "[0-9]");
        assert_eq!(translate_pattern(r"[a\w]").unwrap(), "[a0-9A-Za-z_]");
        assert_eq!(translate_pattern(r"\s").unwrap(), "[\\t\\n\\x0c\\r ]");
        assert_eq!(translate_pattern(r"\S").unwrap(), "[^\\t\\n\\x0c\\r ]");
        assert_eq!(translate_pattern(r"a\bb").unwrap(), r"a(?-u:\b)b");
        // ']' literal-if-first survives; '\D' inside a class rejects.
        assert_eq!(translate_pattern("[]a]").unwrap(), "[]a]");
        assert!(translate_pattern(r"[a\D]").is_err());
    }

    #[test]
    fn reject_list() {
        for p in [r"a\B", "(?<n>a)", "a{1001}", "a*+", r"\u0041", r"\Qab\E"] {
            assert!(translate_pattern(p).is_err(), "{p} should reject");
        }
        assert!(translate_pattern("(?P<n>a)(?P<n>b)").is_err());
        // Lookbehind (?<=..) is NOT the angle-group reject — it passes
        // through and fails to compile in both engines.
        assert!(translate_pattern("(?<=a)b").is_ok());
        assert!(translate_pattern("a{1000}").is_ok());
    }

    #[test]
    fn fuzzer_reject_list() {
        // TASK-54 standing-fuzzer findings (spec addendum pins).
        for p in [
            r"a\1",         // RE2 backref reject vs rust octal-mode escape
            "a?*",          // stacked quantifiers beyond the */+ pairs
            "a{2}+",
            "a*{2}",
            "a???",
            "a{1, 3}",      // spaced bounds: literal to RE2, repetition to rust
            "(a{600}b){2}", // nested counted product over RE2's 1000 cap
            "(a{2,}){3}",   // unbounded inner counts as over-cap
            "[a--b]",       // rust class set operations vs RE2 literals/ranges
            "[--0]",
            "[a&&b]",
            "[a~~b]",
            "[a[b]]",       // nested class in rust, literal '[' in RE2
            r"[a-\d]",      // Perl-class range endpoint (RE2 rejects)
            r"[\d-a]",
        ] {
            assert!(translate_pattern(p).is_err(), "{p} should reject");
        }
        // Still-fine neighbors of the rejects.
        assert_eq!(translate_pattern("a{2}?b").unwrap(), "a{2}?b");
        assert_eq!(translate_pattern("(a{2})b{3}").unwrap(), "(a{2})b{3}");
        assert_eq!(translate_pattern("(a{20}){50}").unwrap(), "(a{20}){50}");
        assert_eq!(translate_pattern("a{abc}").unwrap(), "a{abc}");
        assert_eq!(translate_pattern("[a-f-]").unwrap(), "[a-f-]");
        assert_eq!(translate_pattern(r"[-\d]").unwrap(), "[-0-9]");
        assert_eq!(translate_pattern(r"[\d-]").unwrap(), "[0-9-]");
        assert_eq!(translate_pattern(r"[\1]").unwrap(), r"[\1]"); // in-class octal
        // POSIX elements pass through atomically — the tracker no longer
        // desyncs, so a following Perl class still rewrites in-class.
        assert_eq!(translate_pattern(r"[[:alpha:]\d]").unwrap(), "[[:alpha:]0-9]");
        assert_eq!(translate_pattern("[[:^digit:]x]").unwrap(), "[[:^digit:]x]");
        // Leading-']' ranges; the trailing-'-' literal form stays fine.
        assert!(translate_pattern("[]-Z]").is_err());
        assert!(translate_pattern("[^]-0-7]").is_err());
        assert_eq!(translate_pattern("[]-]").unwrap(), "[]-]");
        // Anchor-only multi-anchor patterns (DuckDB row/const divergence);
        // a consuming atom or a capture group rescues them.
        for p in [r"$\z", "^^", r"\A\A", r"(?i)$\z", "^(?:)^", r"^\b$"] {
            assert!(translate_pattern(p).is_err(), "{p} should reject");
        }
        for p in [r"d$\z", "^", r"\z", r"($\z)", r"x|$\z", "^()^"] {
            assert!(translate_pattern(p).is_ok(), "{p} should serve");
        }
        // '(x){0}' erases the capture group in rust, shifting group numbers.
        assert!(translate_pattern("(a){0}").is_err());
        assert!(translate_pattern("(?P<n>a){0,0}").is_err());
        assert!(translate_pattern("(?:a){0}").is_ok());
        assert!(translate_pattern("(a){0,1}").is_ok());
        assert!(translate_pattern("a{0}").is_ok());
    }

    #[test]
    fn options_alphabet() {
        let o = parse_options("gi", true).unwrap();
        assert!(o.global && o.case_insensitive);
        // Last conflicting letter wins.
        assert!(!parse_options("ic", true).unwrap().case_insensitive);
        assert!(parse_options("ns", false).unwrap().dotall);
        assert!(!parse_options("sn", false).unwrap().dotall);
        assert!(parse_options("g", false).is_err());
        assert!(parse_options("q", false).is_err());
        assert!(parse_options(" i ", false).unwrap().case_insensitive);
    }

    #[test]
    fn rewrite_translation_and_quirks() {
        assert!(matches!(
            translate_rewrite(r"[\1]", 1, false),
            Rewrite::Template(t) if t == "[${1}]"
        ));
        assert!(matches!(
            translate_rewrite("a$b", 0, false),
            Rewrite::Template(t) if t == "a$$b"
        ));
        // Out-of-range backref: full no-op both modes — even AFTER a bad
        // escape (MaxSubmatch pre-scans the whole template).
        assert!(matches!(translate_rewrite(r"\2", 1, true), Rewrite::Identity));
        assert!(matches!(translate_rewrite(r"\x\2", 0, true), Rewrite::Identity));
        assert!(matches!(
            translate_rewrite(r"\\2", 0, true),
            Rewrite::Template(t) if t == r"\2"
        ));
        // Bad escape: no-op non-global, consume-with-prefix global.
        assert!(matches!(translate_rewrite(r"A\xB", 0, false), Rewrite::Identity));
        assert!(matches!(
            translate_rewrite(r"A\xB", 0, true),
            Rewrite::ConsumeWithPrefix(p) if p == "A"
        ));
        // \10 = \1 then literal '0'; \0 = whole match.
        assert!(matches!(
            translate_rewrite(r"\10", 1, false),
            Rewrite::Template(t) if t == "${1}0"
        ));
        assert!(matches!(
            translate_rewrite(r"\0", 0, false),
            Rewrite::Template(t) if t == "${0}"
        ));
    }
}
