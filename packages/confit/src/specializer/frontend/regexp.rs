//! Regular-expression functions and their option strings.

use super::*;

impl Binder<'_> {
    /// `~` / `!~` / SIMILAR TO: FULL match on the raw pattern (measured —
    /// `~` is regexp_full_match in DuckDB, NOT the Postgres search).
    pub(super) fn regex_full_predicate(
        &self,
        name: &str,
        subject: &SqlExpr,
        pattern: &SqlExpr,
        negated: bool,
    ) -> Result<SExpr, PrepareError> {
        let Some(bs) = self.expr_or_null(subject)? else {
            return Ok(null_of(Ty::I1));
        };
        let bs = str_only(name, bs)?;
        let Some(re) = self.regex_pattern(pattern, super::super::retrans::ReOptions::default(), true)?
        else {
            return Ok(null_of(Ty::I1));
        };
        let nullable = bs.nullable;
        let m = SExpr {
            kind: SKind::ReMatch {
                re,
                a: Box::new(bs),
            },
            ty: Ty::I1,
            nullable,
        };
        Ok(if negated {
            SExpr {
                kind: SKind::Not(Box::new(m)),
                ty: Ty::I1,
                nullable,
            }
        } else {
            m
        })
    }

    /// Bind a regex options argument: absent -> defaults; otherwise a
    /// non-NULL constant string ("must not be NULL" / "must be a constant"
    /// are the pinned texts).
    pub(super) fn regex_options(
        &self,
        opts: Option<&SqlExpr>,
        allow_g: bool,
    ) -> Result<super::super::retrans::ReOptions, PrepareError> {
        match self.regex_options_raw(opts, allow_g)? {
            Some(o) => Ok(o),
            None => Err(PrepareError::Bind(
                "Regex options field must not be NULL".into(),
            )),
        }
    }

    /// regexp_replace's variant: a NULL options argument makes the whole
    /// RESULT NULL (pinned asymmetry) — None here means "return NULL".
    pub(super) fn regex_options_nullable(
        &self,
        opts: Option<&SqlExpr>,
    ) -> Result<Option<super::super::retrans::ReOptions>, PrepareError> {
        self.regex_options_raw(opts, true)
    }

    pub(super) fn regex_options_raw(
        &self,
        opts: Option<&SqlExpr>,
        allow_g: bool,
    ) -> Result<Option<super::super::retrans::ReOptions>, PrepareError> {
        let Some(o) = opts else {
            return Ok(Some(super::super::retrans::ReOptions::default()));
        };
        match self.expr_or_null(o)? {
            None => Ok(None),
            // CAST(NULL AS VARCHAR) options behave exactly like bare NULL
            // options (measured: same "must not be NULL" error class /
            // regexp_replace NULL result).
            Some(b) if matches!(b.kind, SKind::NullOf) && b.ty == Ty::Str => Ok(None),
            Some(b) => match b.kind {
                SKind::Lit(Lit::Str(s)) => {
                    super::super::retrans::parse_options(&s, allow_g).map(Some)
                }
                _ => Err(PrepareError::Bind(
                    "Regex options field must be a constant".into(),
                )),
            },
        }
    }

    /// Bind a constant regex pattern into the program regex table:
    /// translate (retrans), optionally full-match anchor, and COMPILE NOW
    /// so invalid patterns error at prepare (pinned bind-time eagerness).
    /// `Ok(None)` = the pattern was a NULL literal (result is NULL).
    pub(super) fn regex_pattern(
        &self,
        p: &SqlExpr,
        o: super::super::retrans::ReOptions,
        full: bool,
    ) -> Result<Option<u32>, PrepareError> {
        Ok(self.regex_pattern_counted(p, o, full)?.map(|(re, _)| re))
    }

    pub(super) fn regex_pattern_counted(
        &self,
        p: &SqlExpr,
        o: super::super::retrans::ReOptions,
        full: bool,
    ) -> Result<Option<(u32, usize)>, PrepareError> {
        let Some(bp) = self.expr_or_null(p)? else {
            return Ok(None);
        };
        if matches!(bp.kind, SKind::NullOf) && bp.ty == Ty::Str {
            // CAST(NULL AS VARCHAR) pattern — measured NULL result, same
            // as a bare NULL literal (pins-waveA/regex-null-pattern.json).
            return Ok(None);
        }
        let SKind::Lit(Lit::Str(raw)) = bp.kind else {
            if bp.ty != Ty::Str {
                return Err(PrepareError::Bind(format!(
                    "no function matches a regex with a {} pattern",
                    bp.ty.name()
                )));
            }
            // Column patterns compile per row in DuckDB; the engine model
            // is prepare-time compilation only.
            return Err(unsup("non-constant regex pattern (patterns compile at prepare)"));
        };
        let translated = if o.literal {
            regex::escape(&raw)
        } else {
            super::super::retrans::translate_pattern(&raw)?
        };
        let pattern = if full {
            format!("\\A(?:{translated})\\z")
        } else {
            translated
        };
        let rx = regex::RegexBuilder::new(&pattern)
            .case_insensitive(o.case_insensitive)
            .dot_matches_new_line(o.dotall)
            .octal(true)
            .build()
            .map_err(|e| PrepareError::Bind(format!("Invalid Input Error: {e}")))?;
        let group_count = rx.captures_len() - 1;
        let spec = super::super::ir::ReSpec {
            pattern,
            ci: o.case_insensitive,
            dotall: o.dotall,
            rewrite: None,
        };
        let mut v = self.regexes.borrow_mut();
        // Reuse identical rewrite-less entries (star filters + repeated
        // predicates); replace ops mutate `rewrite` after, so only share
        // entries that still have none.
        if let Some(i) = v.iter().position(|r| *r == spec) {
            return Ok(Some((i as u32, group_count)));
        }
        v.push(spec);
        Ok(Some(((v.len() - 1) as u32, group_count)))
    }
}
