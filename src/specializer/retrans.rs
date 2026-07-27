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

/// Rewrite a DuckDB/RE2 pattern into a rust-regex pattern with identical
/// semantics, or reject constructs on the measured divergence list.
pub fn translate_pattern(p: &str) -> Result<String, PrepareError> {
    let b = p.as_bytes();
    let mut out = String::with_capacity(p.len() + 8);
    let mut i = 0;
    let mut in_class = false;
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
                i += 2;
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
                if b.get(i) == Some(&b']') {
                    out.push(']');
                    i += 1;
                }
            }
            b']' if in_class => {
                in_class = false;
                out.push(']');
                i += 1;
            }
            b'(' if !in_class => {
                // (?<name>...) angle-bracket groups: DuckDB rejects them.
                if p[i..].starts_with("(?<") && !p[i..].starts_with("(?<=") {
                    return Err(unsup("(?<name>...) capture group (not RE2 syntax)"));
                }
                out.push('(');
                i += 1;
            }
            b'{' if !in_class => {
                // Repetition bound > 1000 is a DuckDB error; rust allows
                // larger — reject past the pinned cap.
                let close = b[i..].iter().position(|&x| x == b'}');
                if let Some(off) = close {
                    let body = &p[i + 1..i + off];
                    let too_big = body
                        .split(',')
                        .filter_map(|s| s.trim().parse::<u64>().ok())
                        .any(|n| n > 1000);
                    if too_big {
                        return Err(unsup(format!(
                            "regex repetition bound over 1000 ({{{body}}})"
                        )));
                    }
                }
                out.push('{');
                i += 1;
            }
            b'*' | b'+' if !in_class => {
                out.push(c as char);
                i += 1;
                // Stacked quantifiers (a*+, a++): DuckDB errors while rust
                // silently reinterprets — a wrong-answer risk, reject.
                if matches!(b.get(i), Some(b'*') | Some(b'+')) {
                    return Err(unsup("stacked regex quantifiers (RE2 rejects them)"));
                }
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
        // Out-of-range backref: full no-op both modes.
        assert!(matches!(translate_rewrite(r"\2", 1, true), Rewrite::Identity));
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
