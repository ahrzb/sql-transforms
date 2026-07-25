//! The SQL specializer: a partial evaluator that turns (fixed SQL + static
//! tables) into a specialized native function `f : Rows -> Rows`, prepared
//! once and invoked millions of times with a small dynamic input relation.
//!
//! Design: docs/superpowers/specs/2026-07-25-sql-specializer-design.md.
//! Build order (backlog milestone m-7): the imperative IR below (M-ir), then
//! the closure-compiled interpreter oracle (M-interp), the frontend + BTA +
//! lowering (M-lower), the Cranelift backend (M-cranelift), and the generated
//! Python-boundary marshaller (M-boundary).

pub mod exec;
pub mod ir;
