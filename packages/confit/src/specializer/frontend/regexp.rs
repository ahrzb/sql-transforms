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
        let bs = self.expr_or_null(subject)?.map(|b| str_only(name, b)).transpose()?;
        // The pattern compiles at bind whatever the subject is (measured:
        // `NULL ~ 'a('` is DuckDB's bind error).
        let re = self.regex_pattern(pattern, super::super::retrans::ReOptions::default(), true)?;
        // A NULL subject keeps the call a node (see `typed_null`).
        let Some(bs) = bs else {
            return self.typed_null(subject, Ty::I1);
        };
        let Some(re) = re else {
            return self.regex_null_pattern(subject, bs, Ty::I1);
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

    /// A regexp call whose pattern is a NULL constant. DuckDB's regexp
    /// functions handle a NULL themselves, so its binder keeps the call: it
    /// reads its subject for each row and answers NULL (measured, 1.5.5:
    /// `CAST((s ~ NULL) AS BIGINT) + (a + 9223372036854775807)` traps on
    /// the overflow, and so do the same over `regexp_matches(s, NULL)` and
    /// `regexp_extract(s, NULL)`). A call whose text reads no column is
    /// foldable there, and an operator with the default NULL handling over
    /// it folds to NULL, unless its subject traps.
    pub(super) fn regex_null_pattern(
        &self,
        subject: &SqlExpr,
        bs: SExpr,
        ty: Ty,
    ) -> Result<SExpr, PrepareError> {
        if DuckNulls::new(self).closed(subject)? {
            return Ok(match self.eval_closed(&bs) {
                Some(_) => null_of(ty),
                // A subject that traps runs for each row there.
                None => super::expr::null_reading(bs, ty),
            });
        }
        // Our binder drops a name whose value cannot matter (`coalesce('x',
        // s)` binds as 'x'), but DuckDB's call still reads it for each row.
        if bind_foldable(&bs) {
            return Err(unsup(
                "a regexp call over a NULL pattern whose subject names a column \
                 that its value does not read",
            ));
        }
        Ok(super::expr::null_reading(bs, ty))
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
        // Options whose text reads a column are no constant there, also
        // where our binder drops the column (a NULL column, `coalesce('i',
        // s)`).
        if !DuckNulls::new(self).closed(o)? {
            return Err(PrepareError::Bind(
                "Regex options field must be a constant".into(),
            ));
        }
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
        let bp = self.expr_or_null(p)?;
        self.regex_compile(p, bp, o, full)
    }

    /// [`Self::regex_pattern_counted`] over pattern `p` already bound as
    /// `bp`.
    pub(super) fn regex_compile(
        &self,
        p: &SqlExpr,
        bp: Option<SExpr>,
        o: super::super::retrans::ReOptions,
        full: bool,
    ) -> Result<Option<(u32, usize)>, PrepareError> {
        let Some(bp) = bp else {
            // A NULL that reads a column is a pattern DuckDB compiles per
            // row, as any column is.
            if !DuckNulls::new(self).closed(p)? {
                return Err(unsup("non-constant regex pattern (patterns compile at prepare)"));
            }
            return Ok(None);
        };
        if bp.ty == Ty::Str
            && (matches!(bp.kind, SKind::NullOf)
                || (folds_to_null(&bp) && DuckNulls::new(self).closed(p)?))
        {
            // CAST(NULL AS VARCHAR) pattern — measured NULL result, same
            // as a bare NULL literal (pins-waveA/regex-null-pattern.json);
            // so is a closed value that evaluates to NULL (`CASE WHEN FALSE
            // THEN 'a' END`, measured).
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
