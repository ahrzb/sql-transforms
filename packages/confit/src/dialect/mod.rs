//! The dialect logical plan — the hub of Query ⇄ Plan conversion
//! (docs/superpowers/specs/2026-08-13-dialect-logical-plan-design.md).
//!
//! A bound, typed, order-free relational plan whose semantics are DuckDB's,
//! as measured (design decision D1; pins in specs/pins-dialect/). Dialect
//! frontends parse SQL into it; dialect printers emit SQL out of it, forcing
//! the plan's explicit semantics in the target's syntax. This module is the
//! plan CORE: nodes, the type lattice, type derivation, the verifier, and
//! the canonical plan text. Frontends and printers are siblings, per phase.
//!
//! The module carries the `specializer/ir` recipe's three mandatory
//! properties:
//! 1. **airtight verifier** ([`verify`]) — explicitness, determinism and
//!    type-derivability are checked structurally, never by review;
//! 2. **canonical text** ([`text`]) — a plan prints to one spelling with
//!    bindings and types visible, for fixtures and review;
//! 3. **round-trip** — `parse(print(p)) == p`, so fixtures are executable.
//!
//! Error discipline is the three-outcome contract, same words as the
//! specializer frontend:
//! * [`DialectError::Unsupported`] — real SQL / a real plan shape we don't
//!   do YET; the message names the construct. Corpus gates count these as
//!   clean.
//! * [`DialectError::Bind`] — wrong against this catalog (unknown column,
//!   type mismatch). Never used for missing features.
//! * [`DialectError::Internal`] — an unverifiable plan escaped a
//!   constructor; always a bug in this module or a frontend.
//!
//! [`DialectError::Text`] sits outside that contract: only hand-written or
//! printer-drifted canonical plan text can raise it.
//!
//! v0 deliberate coarseness, each a named refusal, none a silent guess:
//! decimal arithmetic result scales are not derived (lattice-spec phase 5
//! owns the measurements), f32 arithmetic is not derived, expression
//! nullability is not tracked (columns carry it; expressions will when
//! confit lowers from this plan).

pub mod bigquery;
pub mod duckdb;
pub(crate) mod duckdb_keywords;
pub mod plan;
pub(crate) mod printer;
pub mod py;
pub mod spark;
pub mod text;
pub mod ty;
pub mod verify;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialectError {
    /// Real SQL / plan shape, not covered yet — names the construct.
    Unsupported(String),
    /// Wrong against this catalog or type system.
    Bind(String),
    /// Canonical plan text that does not parse (fixture typo, drifted
    /// printer) — names the token and position.
    Text(String),
    /// An unverifiable plan reached the verifier — a bug here, never data.
    Internal(String),
}

impl std::fmt::Display for DialectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialectError::Unsupported(m) => write!(f, "unsupported: {m}"),
            DialectError::Bind(m) => write!(f, "bind error: {m}"),
            DialectError::Text(m) => write!(f, "plan text error: {m}"),
            DialectError::Internal(m) => write!(f, "internal dialect bug: {m}"),
        }
    }
}

/// Shorthand for [`DialectError::Unsupported`]. `what` NAMES the construct
/// that is not lowered — corpus accounting reads these messages to count a
/// refusal class, so "expression" is a bad name and "SEMI join" a good one.
pub fn unsup(what: impl Into<String>) -> DialectError {
    DialectError::Unsupported(what.into())
}
