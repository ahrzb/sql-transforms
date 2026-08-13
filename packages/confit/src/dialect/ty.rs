//! The plan's type lattice: DuckDB's types, unconditionally representable
//! (design D2). Printers buy landing zones per dialect by reachability;
//! nothing here is about any target engine.
//!
//! Two text forms live here:
//! * the SHORT name (`i32`, `dec(18,3)`, `list<str>`) — the canonical plan
//!   text's vocabulary, round-tripped by [`DTy::parse`];
//! * the DuckDB name (`INTEGER`, `DECIMAL(18,3)`) — what catalogs built
//!   from `DESCRIBE` speak, ingested by [`DTy::from_duckdb`] and emitted
//!   back by [`DTy::duckdb_name`] (CAST targets in the DuckDB printer).
//!
//! Types not yet ingestible (MAP, UNION, ENUM, BIT) refuse by name at the
//! boundary — the design's rule that representable grows with a verifier,
//! both text forms, and a printer row, never ad hoc.

use super::{unsup, DialectError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DTy {
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    /// DuckDB DECIMAL(p, s): p in 1..=38, s <= p.
    Dec(u8, u8),
    Str,
    Blob,
    Date,
    Time,
    /// TIME WITH TIME ZONE — representable; note the pinned lossy Arrow
    /// export (pins-dialect/arrow-export.json: offset dropped).
    TimeTz,
    /// TIMESTAMP_S / TIMESTAMP_MS / TIMESTAMP (µs, wall-clock) / TIMESTAMP_NS.
    TsS,
    TsMs,
    TsUs,
    TsNs,
    /// TIMESTAMP WITH TIME ZONE — the instant type.
    TsTz,
    Interval,
    Uuid,
    List(Box<DTy>),
    Struct(Vec<(String, DTy)>),
}

impl DTy {
    /// A type every text form can round-trip: Dec params in range, struct
    /// field names free of the type grammar's delimiters. verify() refuses
    /// plans carrying anything else - API-built plans get the same bounds
    /// as text-built ones.
    pub fn is_well_formed(&self) -> bool {
        match self {
            DTy::Dec(p, s) => *p != 0 && *p <= 38 && s <= p,
            DTy::List(e) => e.is_well_formed(),
            DTy::Struct(fs) => fs.iter().all(|(n, t)| {
                !n.is_empty()
                    && !n.contains([':', ',', '<', '>'])
                    && !n.contains(char::is_whitespace)
                    && t.is_well_formed()
            }),
            _ => true,
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DTy::I8
                | DTy::I16
                | DTy::I32
                | DTy::I64
                | DTy::I128
                | DTy::U8
                | DTy::U16
                | DTy::U32
                | DTy::U64
                | DTy::U128
        )
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || matches!(self, DTy::F32 | DTy::F64 | DTy::Dec(..))
    }

    /// Short name, the canonical plan text vocabulary.
    pub fn name(&self) -> String {
        match self {
            DTy::Bool => "bool".into(),
            DTy::I8 => "i8".into(),
            DTy::I16 => "i16".into(),
            DTy::I32 => "i32".into(),
            DTy::I64 => "i64".into(),
            DTy::I128 => "i128".into(),
            DTy::U8 => "u8".into(),
            DTy::U16 => "u16".into(),
            DTy::U32 => "u32".into(),
            DTy::U64 => "u64".into(),
            DTy::U128 => "u128".into(),
            DTy::F32 => "f32".into(),
            DTy::F64 => "f64".into(),
            DTy::Dec(p, s) => format!("dec({p},{s})"),
            DTy::Str => "str".into(),
            DTy::Blob => "blob".into(),
            DTy::Date => "date".into(),
            DTy::Time => "time".into(),
            DTy::TimeTz => "timetz".into(),
            DTy::TsS => "ts_s".into(),
            DTy::TsMs => "ts_ms".into(),
            DTy::TsUs => "ts".into(),
            DTy::TsNs => "ts_ns".into(),
            DTy::TsTz => "tstz".into(),
            DTy::Interval => "interval".into(),
            DTy::Uuid => "uuid".into(),
            DTy::List(e) => format!("list<{}>", e.name()),
            DTy::Struct(fs) => {
                let inner: Vec<String> = fs
                    .iter()
                    .map(|(n, t)| format!("{n}:{}", t.name()))
                    .collect();
                format!("struct<{}>", inner.join(","))
            }
        }
    }

    /// Parse a short name. Total inverse of [`DTy::name`] on its image.
    pub fn parse(s: &str) -> Option<DTy> {
        let flat = match s {
            "bool" => Some(DTy::Bool),
            "i8" => Some(DTy::I8),
            "i16" => Some(DTy::I16),
            "i32" => Some(DTy::I32),
            "i64" => Some(DTy::I64),
            "i128" => Some(DTy::I128),
            "u8" => Some(DTy::U8),
            "u16" => Some(DTy::U16),
            "u32" => Some(DTy::U32),
            "u64" => Some(DTy::U64),
            "u128" => Some(DTy::U128),
            "f32" => Some(DTy::F32),
            "f64" => Some(DTy::F64),
            "str" => Some(DTy::Str),
            "blob" => Some(DTy::Blob),
            "date" => Some(DTy::Date),
            "time" => Some(DTy::Time),
            "timetz" => Some(DTy::TimeTz),
            "ts_s" => Some(DTy::TsS),
            "ts_ms" => Some(DTy::TsMs),
            "ts" => Some(DTy::TsUs),
            "ts_ns" => Some(DTy::TsNs),
            "tstz" => Some(DTy::TsTz),
            "interval" => Some(DTy::Interval),
            "uuid" => Some(DTy::Uuid),
            _ => None,
        };
        if flat.is_some() {
            return flat;
        }
        if let Some(rest) = s.strip_prefix("dec(").and_then(|r| r.strip_suffix(')')) {
            let (p, sc) = rest.split_once(',')?;
            let (p, sc): (u8, u8) = (p.trim().parse().ok()?, sc.trim().parse().ok()?);
            // Same bounds as from_duckdb - an invalid Dec must not exist,
            // whichever door it comes through (review: u8 underflow in the
            // BigQuery NUMERIC/BIGNUMERIC split for s > p).
            if p == 0 || p > 38 || sc > p {
                return None;
            }
            return Some(DTy::Dec(p, sc));
        }
        if let Some(rest) = s.strip_prefix("list<").and_then(|r| r.strip_suffix('>')) {
            return Some(DTy::List(Box::new(DTy::parse(rest)?)));
        }
        if let Some(rest) = s.strip_prefix("struct<").and_then(|r| r.strip_suffix('>')) {
            let mut fields = Vec::new();
            for part in split_top_level(rest) {
                let (n, t) = part.split_once(':')?;
                let n = n.trim();
                // Names carrying the grammar's own delimiters cannot
                // round-trip; reject here so print∘parse stays identity.
                if n.is_empty()
                    || n.contains([':', ',', '<', '>'])
                    || n.contains(char::is_whitespace)
                {
                    return None;
                }
                fields.push((n.to_string(), DTy::parse(t.trim())?));
            }
            return Some(DTy::Struct(fields));
        }
        None
    }

    /// Ingest a DuckDB type name (`DESCRIBE` spelling). Named refusal for
    /// types not yet carried.
    pub fn from_duckdb(s: &str) -> Result<DTy, DialectError> {
        let t = s.trim();
        let upper = t.to_ascii_uppercase();
        let flat = match upper.as_str() {
            "BOOLEAN" => Some(DTy::Bool),
            "TINYINT" => Some(DTy::I8),
            "SMALLINT" => Some(DTy::I16),
            "INTEGER" => Some(DTy::I32),
            "BIGINT" => Some(DTy::I64),
            "HUGEINT" => Some(DTy::I128),
            "UTINYINT" => Some(DTy::U8),
            "USMALLINT" => Some(DTy::U16),
            "UINTEGER" => Some(DTy::U32),
            "UBIGINT" => Some(DTy::U64),
            "UHUGEINT" => Some(DTy::U128),
            "FLOAT" | "REAL" => Some(DTy::F32),
            "DOUBLE" => Some(DTy::F64),
            "VARCHAR" | "TEXT" | "STRING" => Some(DTy::Str),
            "BLOB" | "BYTEA" => Some(DTy::Blob),
            "DATE" => Some(DTy::Date),
            "TIME" => Some(DTy::Time),
            "TIME WITH TIME ZONE" | "TIMETZ" => Some(DTy::TimeTz),
            "TIMESTAMP_S" => Some(DTy::TsS),
            "TIMESTAMP_MS" => Some(DTy::TsMs),
            "TIMESTAMP" | "DATETIME" => Some(DTy::TsUs),
            "TIMESTAMP_NS" => Some(DTy::TsNs),
            "TIMESTAMP WITH TIME ZONE" | "TIMESTAMPTZ" => Some(DTy::TsTz),
            "INTERVAL" => Some(DTy::Interval),
            "UUID" => Some(DTy::Uuid),
            _ => None,
        };
        if let Some(t) = flat {
            return Ok(t);
        }
        if let Some(rest) = upper
            .strip_prefix("DECIMAL(")
            .and_then(|r| r.strip_suffix(')'))
        {
            let Some((p, sc)) = rest.split_once(',') else {
                return Err(DialectError::Bind(format!("malformed DECIMAL type: {t}")));
            };
            let (Ok(p), Ok(sc)) = (p.trim().parse::<u8>(), sc.trim().parse::<u8>()) else {
                return Err(DialectError::Bind(format!("malformed DECIMAL type: {t}")));
            };
            if p == 0 || p > 38 || sc > p {
                return Err(DialectError::Bind(format!("DECIMAL out of range: {t}")));
            }
            return Ok(DTy::Dec(p, sc));
        }
        if let Some(rest) = t.strip_suffix("[]") {
            return Ok(DTy::List(Box::new(DTy::from_duckdb(rest)?)));
        }
        // STRUCT(a INTEGER, ...) — grown when a frontend needs it; MAP/UNION/
        // ENUM/BIT likewise. Named, so corpus accounting sees the class.
        Err(unsup(format!("catalog type not carried yet: {t}")))
    }

    /// The DuckDB spelling — CAST targets in the DuckDB printer.
    pub fn duckdb_name(&self) -> Result<String, DialectError> {
        Ok(match self {
            DTy::Bool => "BOOLEAN".into(),
            DTy::I8 => "TINYINT".into(),
            DTy::I16 => "SMALLINT".into(),
            DTy::I32 => "INTEGER".into(),
            DTy::I64 => "BIGINT".into(),
            DTy::I128 => "HUGEINT".into(),
            DTy::U8 => "UTINYINT".into(),
            DTy::U16 => "USMALLINT".into(),
            DTy::U32 => "UINTEGER".into(),
            DTy::U64 => "UBIGINT".into(),
            DTy::U128 => "UHUGEINT".into(),
            DTy::F32 => "FLOAT".into(),
            DTy::F64 => "DOUBLE".into(),
            DTy::Dec(p, s) => format!("DECIMAL({p},{s})"),
            DTy::Str => "VARCHAR".into(),
            DTy::Blob => "BLOB".into(),
            DTy::Date => "DATE".into(),
            DTy::Time => "TIME".into(),
            DTy::TimeTz => "TIMETZ".into(),
            DTy::TsS => "TIMESTAMP_S".into(),
            DTy::TsMs => "TIMESTAMP_MS".into(),
            DTy::TsUs => "TIMESTAMP".into(),
            DTy::TsNs => "TIMESTAMP_NS".into(),
            DTy::TsTz => "TIMESTAMPTZ".into(),
            DTy::Interval => "INTERVAL".into(),
            DTy::Uuid => "UUID".into(),
            DTy::List(e) => format!("{}[]", e.duckdb_name()?),
            DTy::Struct(_) => {
                return Err(unsup("printing STRUCT type names".to_string()));
            }
        })
    }
}

/// Split on commas not nested inside `<...>` or `(...)`.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}
