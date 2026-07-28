//! The verifier — the airtight boundary. Everything upstream (lowering) and
//! downstream (backends) is allowed to assume a verified program; nothing may
//! execute or compile an unverified one.
//!
//! Rules enforced (numbers referenced from tests):
//!  1. Structure: at least one block; entry (b0) has no params and is never a
//!     branch target; batch column names unique per side; the function name
//!     is an identifier and map statics have >= 1 key and >= 1 value column
//!     (a verified program must print to parseable canonical text).
//!  2. SSA: every value defined exactly once function-wide; every use sees a
//!     definition earlier in the same block or a param of the same block
//!     (strict block-param form — cross-block uses are illegal, which is what
//!     lets the verifier skip dominance analysis entirely).
//!  3. Types: every operand matches its instruction's signature; `.opt` forms
//!     are mandatory for nullable columns/statics and illegal on non-nullable
//!     ones (the null lane can be neither skipped nor invented).
//!  4. Statics: every `@N` resolves; probe/sload match the static's kind,
//!     arity, and types.
//!  5. CFG: all blocks reachable from entry; branch args (cycles are legal
//!     since stage-B multiplicity loops; see the back-edge notes below);
//!     match target params in count and type.
//!  6. Stores: no path stores a column twice, whatever its terminator
//!     (including `trap` — a double store is always a lowering bug); paths
//!     to `emit` store every column exactly once; paths to `skip` store
//!     nothing; store states must agree at joins.

use std::collections::HashMap;

use super::{Block, Col, Inst, Program, StaticTy, Term, Ty, Value};

#[derive(Debug, PartialEq, Eq)]
pub struct VerifyError {
    /// Block index, when the error is inside a block.
    pub block: Option<usize>,
    /// Instruction index within the block; `None` for param/terminator errors.
    pub inst: Option<usize>,
    pub msg: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.block, self.inst) {
            (Some(b), Some(i)) => write!(f, "b{b}[{i}]: {}", self.msg),
            (Some(b), None) => write!(f, "b{b}: {}", self.msg),
            _ => write!(f, "{}", self.msg),
        }
    }
}

pub fn verify(p: &Program) -> Result<(), Vec<VerifyError>> {
    let mut errs = Vec::new();

    check_structure(p, &mut errs);
    // Function-wide definition table: value -> (type, defining block),
    // discovered at definition sites. Collected fully before use-checking so
    // an early error doesn't cascade into spurious "undefined value" noise.
    let mut def_types: HashMap<u32, (Ty, usize)> = HashMap::new();
    collect_defs(p, &mut def_types, &mut errs);
    for (bi, b) in p.blocks.iter().enumerate() {
        check_block(p, bi, b, &def_types, &mut errs);
    }
    check_cfg_and_stores(p, &mut errs);

    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

fn err(errs: &mut Vec<VerifyError>, block: Option<usize>, inst: Option<usize>, msg: String) {
    errs.push(VerifyError { block, inst, msg });
}

fn check_structure(p: &Program, errs: &mut Vec<VerifyError>) {
    // A verified program must print to parseable canonical text, so text-only
    // constraints (identifier function name, non-empty map signatures — the
    // grammar cannot express `map() -> ()`) are verifier rules too.
    if !super::print::is_ident(&p.name) {
        err(
            errs,
            None,
            None,
            format!("function name '{}' must be an identifier", p.name),
        );
    }
    // Wave-4: maps may have EMPTY keys (cross join to a table whose
    // single-entry-ness the duplicate-key check enforces at compile) and
    // EMPTY values (all-key/semi joins — the probe carries only the hit).
    // A map that is empty on BOTH axes carries no information at all.
    for (i, st) in p.statics.iter().enumerate() {
        if let StaticTy::Map { keys, values } = st {
            if keys.is_empty() && values.is_empty() {
                err(
                    errs,
                    None,
                    None,
                    format!("@{i}: map static with neither keys nor values"),
                );
            }
        }
    }
    if p.blocks.is_empty() {
        err(errs, None, None, "function has no blocks".to_string());
        return;
    }
    if !p.blocks[0].params.is_empty() {
        err(
            errs,
            Some(0),
            None,
            "entry block cannot have params".to_string(),
        );
    }
    for (side, cols) in [("in", &p.in_cols), ("out", &p.out_cols)] {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        for c in cols.iter() {
            if seen.insert(c.name.as_str(), ()).is_some() {
                err(
                    errs,
                    None,
                    None,
                    format!("duplicate {side} column '{}'", c.name),
                );
            }
        }
    }
}

/// First pass: register every definition (params + inst dsts) with its type,
/// flagging double definitions. A definition's type is derivable from the
/// instruction plus the column/static tables alone — except `select`, whose
/// result type equals its operands'; it is registered per its `a` operand
/// during the per-block pass (see `check_block`), and as a placeholder here
/// only to occupy the SSA slot.
fn collect_defs(
    p: &Program,
    def_types: &mut HashMap<u32, (Ty, usize)>,
    errs: &mut Vec<VerifyError>,
) {
    for (bi, b) in p.blocks.iter().enumerate() {
        for (v, ty) in &b.params {
            if def_types.insert(v.0, (*ty, bi)).is_some() {
                err(
                    errs,
                    Some(bi),
                    None,
                    format!("%v{} defined more than once", v.0),
                );
            }
        }
        for (ii, inst) in b.insts.iter().enumerate() {
            for (dst, ty) in dst_types(p, inst) {
                if def_types.insert(dst.0, (ty, bi)).is_some() {
                    err(
                        errs,
                        Some(bi),
                        Some(ii),
                        format!("%v{} defined more than once", dst.0),
                    );
                }
            }
        }
    }
}

/// Result types of an instruction's definitions, best-effort when the
/// program is malformed (bad indices fall back so verification continues and
/// the real error is reported at the site check).
fn dst_types(p: &Program, inst: &Inst) -> Vec<(Value, Ty)> {
    let in_col = |c: u32| p.in_cols.get(c as usize).map(|c| c.ty.ty);
    let scalar_ty = |id: u32| match p.statics.get(id as usize) {
        Some(StaticTy::Scalar(ct)) => ct.ty,
        _ => Ty::I1,
    };
    match inst {
        Inst::Const { dst, lit } => vec![(*dst, lit.ty())],
        Inst::Bin { op, dst, .. } => vec![(*dst, op.sig().1)],
        Inst::Cmp { dst, .. } | Inst::Not { dst, .. } => vec![(*dst, Ty::I1)],
        // Placeholder; corrected per-block (see collect_defs docs).
        Inst::Select { dst, .. } => vec![(*dst, Ty::I1)],
        Inst::Itof { dst, .. } => vec![(*dst, Ty::F64)],
        Inst::Ftoi { dst, .. } => vec![(*dst, Ty::I64)],
        Inst::Itos { dst, .. }
        | Inst::Ftos { dst, .. }
        | Inst::Sconcat { dst, .. }
        | Inst::Str1 { dst, .. }
        | Inst::Str3 { dst, .. }
        | Inst::Str2i { dst, .. }
        | Inst::Spad { dst, .. }
        | Inst::Sslice { dst, .. }
        | Inst::Strim { dst, .. }
        | Inst::Ssubstr { dst, .. } => {
            vec![(*dst, Ty::Str)]
        }
        Inst::Num1 { op, dst, .. } => vec![(*dst, op.sig())],
        Inst::Str2 { op, dst, .. } => vec![(*dst, op.result_ty())],
        Inst::SLen { dst, .. } | Inst::Sord { dst, .. } => vec![(*dst, Ty::I64)],
        Inst::Slike { dst, .. } | Inst::ReMatch { dst, .. } => vec![(*dst, Ty::I1)],
        Inst::ReExtract { dst, .. } | Inst::ReReplace { dst, .. } => vec![(*dst, Ty::Str)],
        Inst::Round2f { dst, .. } => vec![(*dst, Ty::F64)],
        Inst::Round2i { dst, .. } => vec![(*dst, Ty::I64)],
        Inst::StoiOpt { flag, dst, .. } => vec![(*flag, Ty::I1), (*dst, Ty::I64)],
        Inst::StofOpt { flag, dst, .. } => vec![(*flag, Ty::I1), (*dst, Ty::F64)],
        Inst::Load { dst, col } => vec![(*dst, in_col(*col).unwrap_or(Ty::I1))],
        Inst::LoadOpt { flag, dst, col } => {
            vec![(*flag, Ty::I1), (*dst, in_col(*col).unwrap_or(Ty::I1))]
        }
        Inst::Store { .. } | Inst::StoreOpt { .. } => vec![],
        Inst::Probe {
            static_id,
            hit,
            dsts,
            ..
        } => {
            let mut v = vec![(*hit, Ty::I1)];
            if let Some(StaticTy::Map { values, .. }) = p.statics.get(*static_id as usize) {
                for (d, ty) in dsts.iter().zip(values.iter()) {
                    v.push((*d, *ty));
                }
            }
            // Dsts beyond the declared value columns keep no type; the site
            // check reports the arity mismatch.
            v
        }
        Inst::ProbeRange { start, end, .. } => vec![(*start, Ty::I64), (*end, Ty::I64)],
        Inst::ProbeRead {
            static_id, dsts, ..
        } => {
            let mut v = Vec::new();
            if let Some(StaticTy::MultiMap { values, .. }) = p.statics.get(*static_id as usize) {
                for (d, ty) in dsts.iter().zip(values.iter()) {
                    v.push((*d, *ty));
                }
            }
            v
        }
        Inst::Sload { static_id, dst } => vec![(*dst, scalar_ty(*static_id))],
        Inst::SloadOpt {
            static_id,
            flag,
            dst,
        } => {
            vec![(*flag, Ty::I1), (*dst, scalar_ty(*static_id))]
        }
    }
}

/// Look up a use in the block's scope, reporting scope violations with a
/// message that distinguishes "defined later in this block" (use before def)
/// from "defined in another block" (illegal crossing) from "never defined".
fn scope_ty(
    in_scope: &HashMap<u32, Ty>,
    def_types: &HashMap<u32, (Ty, usize)>,
    v: Value,
    what: &str,
    bi: usize,
    ii: Option<usize>,
    errs: &mut Vec<VerifyError>,
) -> Option<Ty> {
    match in_scope.get(&v.0) {
        Some(ty) => Some(*ty),
        None => {
            let msg = match def_types.get(&v.0) {
                Some((_, def_bi)) if *def_bi != bi => format!(
                    "{what} %v{} is not visible here: values cross blocks only as branch \
                     args to block params",
                    v.0
                ),
                _ => format!("{what} %v{} is used before any definition", v.0),
            };
            err(errs, Some(bi), ii, msg);
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn want(
    in_scope: &HashMap<u32, Ty>,
    def_types: &HashMap<u32, (Ty, usize)>,
    v: Value,
    ty: Ty,
    what: &str,
    bi: usize,
    ii: Option<usize>,
    errs: &mut Vec<VerifyError>,
) {
    if let Some(actual) = scope_ty(in_scope, def_types, v, what, bi, ii, errs) {
        if actual != ty {
            err(
                errs,
                Some(bi),
                ii,
                format!(
                    "{what} %v{} must be {}, got {}",
                    v.0,
                    ty.name(),
                    actual.name()
                ),
            );
        }
    }
}

/// Second pass, per block: scoping (uses see only same-block earlier defs or
/// own params) and per-instruction operand typing.
fn check_block(
    p: &Program,
    bi: usize,
    b: &Block,
    def_types: &HashMap<u32, (Ty, usize)>,
    errs: &mut Vec<VerifyError>,
) {
    // Values visible at the current point of this block.
    let mut in_scope: HashMap<u32, Ty> = HashMap::new();
    for (v, ty) in &b.params {
        in_scope.insert(v.0, *ty);
    }

    for (ii, inst) in b.insts.iter().enumerate() {
        let i = Some(ii);
        match inst {
            Inst::Const { .. } => {}
            Inst::Bin { op, a, b: rhs, .. } => {
                let (operand, _) = op.sig();
                want(&in_scope, def_types, *a, operand, "operand", bi, i, errs);
                want(&in_scope, def_types, *rhs, operand, "operand", bi, i, errs);
            }
            Inst::Cmp { ty, a, b: rhs, .. } => {
                if *ty == Ty::I1 {
                    err(
                        errs,
                        Some(bi),
                        i,
                        "cmp on i1 is not defined; use xor/not".into(),
                    );
                }
                want(&in_scope, def_types, *a, *ty, "operand", bi, i, errs);
                want(&in_scope, def_types, *rhs, *ty, "operand", bi, i, errs);
            }
            Inst::Not { a, .. } => want(&in_scope, def_types, *a, Ty::I1, "operand", bi, i, errs),
            Inst::Select {
                dst,
                cond,
                a,
                b: rhs,
            } => {
                want(
                    &in_scope,
                    def_types,
                    *cond,
                    Ty::I1,
                    "condition",
                    bi,
                    i,
                    errs,
                );
                let ta = scope_ty(&in_scope, def_types, *a, "operand", bi, i, errs);
                let tb = scope_ty(&in_scope, def_types, *rhs, "operand", bi, i, errs);
                if let (Some(ta), Some(tb)) = (ta, tb) {
                    if ta != tb {
                        err(
                            errs,
                            Some(bi),
                            i,
                            format!("select arms differ: {} vs {}", ta.name(), tb.name()),
                        );
                    }
                }
                // Correct the placeholder from collect_defs with the real type.
                if let Some(ta) = ta {
                    in_scope.insert(dst.0, ta);
                }
            }
            Inst::Itof { a, .. } => want(&in_scope, def_types, *a, Ty::I64, "operand", bi, i, errs),
            Inst::Ftoi { a, .. } => want(&in_scope, def_types, *a, Ty::F64, "operand", bi, i, errs),
            Inst::Itos { a, .. } => want(&in_scope, def_types, *a, Ty::I64, "operand", bi, i, errs),
            Inst::Ftos { a, .. } => want(&in_scope, def_types, *a, Ty::F64, "operand", bi, i, errs),
            Inst::StoiOpt { a, .. } | Inst::StofOpt { a, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs)
            }
            Inst::ReMatch { re, a, .. }
            | Inst::ReExtract { re, a, .. }
            | Inst::ReReplace { re, a, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                if *re as usize >= p.regexes.len() {
                    err(errs, Some(bi), i, format!("regex @{re} out of range"));
                }
                if matches!(inst, Inst::ReReplace { .. })
                    && p.regexes
                        .get(*re as usize)
                        .is_some_and(|r| r.rewrite.is_none())
                {
                    err(
                        errs,
                        Some(bi),
                        i,
                        format!("rereplace on regex @{re} without a rewrite template"),
                    );
                }
            }
            Inst::Sconcat { a, b: rhs, .. }
            | Inst::Strim { a, chars: rhs, .. }
            | Inst::Str2 { a, b: rhs, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *rhs, Ty::Str, "operand", bi, i, errs);
            }
            Inst::Str1 { a, .. } | Inst::SLen { a, .. } | Inst::Sord { a, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs)
            }
            Inst::Str3 { a, b, c, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *b, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *c, Ty::Str, "operand", bi, i, errs);
            }
            Inst::Str2i { a, n, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *n, Ty::I64, "operand", bi, i, errs);
            }
            Inst::Spad { a, len, pad, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *len, Ty::I64, "operand", bi, i, errs);
                want(&in_scope, def_types, *pad, Ty::Str, "operand", bi, i, errs);
            }
            Inst::Sslice { a, lo, hi, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *lo, Ty::I64, "operand", bi, i, errs);
                want(&in_scope, def_types, *hi, Ty::I64, "operand", bi, i, errs);
            }
            Inst::Slike { a, p, esc, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(&in_scope, def_types, *p, Ty::Str, "operand", bi, i, errs);
                if let Some(e) = esc {
                    want(&in_scope, def_types, *e, Ty::Str, "operand", bi, i, errs);
                }
            }
            Inst::Round2f { a, n, .. } => {
                want(&in_scope, def_types, *a, Ty::F64, "operand", bi, i, errs);
                want(&in_scope, def_types, *n, Ty::I64, "operand", bi, i, errs);
            }
            Inst::Round2i { a, n, .. } => {
                want(&in_scope, def_types, *a, Ty::I64, "operand", bi, i, errs);
                want(&in_scope, def_types, *n, Ty::I64, "operand", bi, i, errs);
            }
            Inst::Ssubstr { a, start, len, .. } => {
                want(&in_scope, def_types, *a, Ty::Str, "operand", bi, i, errs);
                want(
                    &in_scope,
                    def_types,
                    *start,
                    Ty::I64,
                    "operand",
                    bi,
                    i,
                    errs,
                );
                if let Some(len) = len {
                    want(&in_scope, def_types, *len, Ty::I64, "operand", bi, i, errs);
                }
            }
            Inst::Num1 { op, a, .. } => {
                want(&in_scope, def_types, *a, op.sig(), "operand", bi, i, errs)
            }
            Inst::Load { col, .. } => match p.in_cols.get(*col as usize) {
                None => err(errs, Some(bi), i, format!("unknown in column {col}")),
                Some(Col { ty, name }) if ty.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("in.{name} is nullable: use load.opt"),
                ),
                Some(_) => {}
            },
            Inst::LoadOpt { col, .. } => match p.in_cols.get(*col as usize) {
                None => err(errs, Some(bi), i, format!("unknown in column {col}")),
                Some(Col { ty, name }) if !ty.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("in.{name} is not nullable: use load"),
                ),
                Some(_) => {}
            },
            Inst::Store { col, val } => match p.out_cols.get(*col as usize) {
                None => err(errs, Some(bi), i, format!("unknown out column {col}")),
                Some(c) if c.ty.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("out.{} is nullable: use store.opt", c.name),
                ),
                Some(c) => want(
                    &in_scope,
                    def_types,
                    *val,
                    c.ty.ty,
                    "stored value",
                    bi,
                    i,
                    errs,
                ),
            },
            Inst::StoreOpt { col, flag, val } => match p.out_cols.get(*col as usize) {
                None => err(errs, Some(bi), i, format!("unknown out column {col}")),
                Some(c) if !c.ty.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("out.{} is not nullable: use store", c.name),
                ),
                Some(c) => {
                    want(
                        &in_scope,
                        def_types,
                        *flag,
                        Ty::I1,
                        "validity flag",
                        bi,
                        i,
                        errs,
                    );
                    want(
                        &in_scope,
                        def_types,
                        *val,
                        c.ty.ty,
                        "stored value",
                        bi,
                        i,
                        errs,
                    );
                }
            },
            Inst::Probe {
                static_id,
                dsts,
                keys,
                ..
            } => match p.statics.get(*static_id as usize) {
                None => err(errs, Some(bi), i, format!("unknown static @{static_id}")),
                Some(StaticTy::Scalar(_)) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is a scalar: use sload"),
                ),
                Some(StaticTy::Map {
                    keys: kts,
                    values: vts,
                }) => {
                    if keys.len() != kts.len() {
                        err(
                            errs,
                            Some(bi),
                            i,
                            format!(
                                "@{static_id} has {} key(s), probe passes {}",
                                kts.len(),
                                keys.len()
                            ),
                        );
                    } else {
                        for (k, kt) in keys.iter().zip(kts.iter()) {
                            want(&in_scope, def_types, *k, *kt, "probe key", bi, i, errs);
                        }
                    }
                    if dsts.len() != vts.len() {
                        err(
                            errs,
                            Some(bi),
                            i,
                            format!(
                                "@{static_id} has {} value column(s), probe defines {}",
                                vts.len(),
                                dsts.len()
                            ),
                        );
                    }
                }
                Some(StaticTy::MultiMap { .. }) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is a multimap: use probe.range"),
                ),
            },
            Inst::ProbeRange {
                static_id, keys, ..
            } => match p.statics.get(*static_id as usize) {
                None => err(errs, Some(bi), i, format!("unknown static @{static_id}")),
                Some(StaticTy::MultiMap { keys: kts, .. }) => {
                    if keys.len() != kts.len() {
                        err(
                            errs,
                            Some(bi),
                            i,
                            format!(
                                "@{static_id} has {} key(s), probe.range passes {}",
                                kts.len(),
                                keys.len()
                            ),
                        );
                    } else {
                        for (k, kt) in keys.iter().zip(kts.iter()) {
                            want(&in_scope, def_types, *k, *kt, "probe key", bi, i, errs);
                        }
                    }
                }
                Some(_) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is not a multimap: probe.range needs one"),
                ),
            },
            Inst::ProbeRead {
                static_id,
                idx,
                dsts,
            } => match p.statics.get(*static_id as usize) {
                None => err(errs, Some(bi), i, format!("unknown static @{static_id}")),
                Some(StaticTy::MultiMap { values: vts, .. }) => {
                    want(&in_scope, def_types, *idx, Ty::I64, "probe index", bi, i, errs);
                    if dsts.len() != vts.len() {
                        err(
                            errs,
                            Some(bi),
                            i,
                            format!(
                                "@{static_id} has {} value column(s), probe.read defines {}",
                                vts.len(),
                                dsts.len()
                            ),
                        );
                    }
                }
                Some(_) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is not a multimap: probe.read needs one"),
                ),
            },
            Inst::Sload { static_id, .. } => match p.statics.get(*static_id as usize) {
                None => err(errs, Some(bi), i, format!("unknown static @{static_id}")),
                Some(StaticTy::Map { .. }) | Some(StaticTy::MultiMap { .. }) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is a map: use probe"),
                ),
                Some(StaticTy::Scalar(ct)) if ct.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is nullable: use sload.opt"),
                ),
                Some(_) => {}
            },
            Inst::SloadOpt { static_id, .. } => match p.statics.get(*static_id as usize) {
                None => err(errs, Some(bi), i, format!("unknown static @{static_id}")),
                Some(StaticTy::Map { .. }) | Some(StaticTy::MultiMap { .. }) => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is a map: use probe"),
                ),
                Some(StaticTy::Scalar(ct)) if !ct.nullable => err(
                    errs,
                    Some(bi),
                    i,
                    format!("@{static_id} is not nullable: use sload"),
                ),
                Some(_) => {}
            },
        }

        // Definitions become visible AFTER the instruction (no self-use).
        for d in inst.dsts() {
            let ty = def_types.get(&d.0).map(|(t, _)| *t).unwrap_or(Ty::I1);
            // Select already inserted its corrected type above; keep it.
            in_scope.entry(d.0).or_insert(ty);
        }
    }

    // Terminator: cond and branch args are uses in this block's final scope.
    if let Term::Brif { cond, .. } = &b.term {
        want(
            &in_scope,
            def_types,
            *cond,
            Ty::I1,
            "branch condition",
            bi,
            None,
            errs,
        );
    }
    for (target, args) in b.term.successors() {
        match p.blocks.get(target.0 as usize) {
            None => err(
                errs,
                Some(bi),
                None,
                format!("branch to unknown block b{}", target.0),
            ),
            Some(tb) => {
                if args.len() != tb.params.len() {
                    err(
                        errs,
                        Some(bi),
                        None,
                        format!(
                            "b{} expects {} arg(s), got {}",
                            target.0,
                            tb.params.len(),
                            args.len()
                        ),
                    );
                } else {
                    for (arg, (_, pty)) in args.iter().zip(tb.params.iter()) {
                        want(
                            &in_scope,
                            def_types,
                            *arg,
                            *pty,
                            "branch arg",
                            bi,
                            None,
                            errs,
                        );
                    }
                }
                if target.0 == 0 {
                    err(errs, Some(bi), None, "branch to entry block".to_string());
                }
            }
        }
    }
}

/// CFG shape (reachability, acyclicity) and the store-completeness dataflow.
fn check_cfg_and_stores(p: &Program, errs: &mut Vec<VerifyError>) {
    let n = p.blocks.len();
    if n == 0 {
        return;
    }

    // Iterative DFS for reachability + BACK-EDGE detection (0 white, 1 gray,
    // 2 black). Explicit stack, NOT recursion: a deep-but-legal CFG (large
    // CASE/decision-tree lowerings) must not abort the process — recursion
    // stack-overflowed at ~8k blocks under adversarial fuzzing.
    //
    // Stage-B: cycles are LEGAL (multiplicity loops jump back to their
    // header via EmitTo — emit-and-continue — or a plain Jump on a
    // residual-filtered iteration). Back-edges are excluded from the topo
    // order below; the store dataflow stays sound because a back-edge's
    // state must MATCH the header's already-known entry state (EmitTo
    // propagates the RESET all-zero state; a filtered-continue Jump must
    // arrive store-free), so no fixpoint iteration is ever needed.
    let mut color = vec![0u8; n];
    let mut back_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)]; // (block, next successor index)
    color[0] = 1;
    while let Some(frame) = stack.last_mut() {
        let (b, si) = *frame;
        let succs = p.blocks[b].term.successors();
        if si < succs.len() {
            frame.1 += 1;
            let s = succs[si].0 .0 as usize;
            if s >= n {
                continue; // reported by check_block
            }
            match color[s] {
                0 => {
                    color[s] = 1;
                    stack.push((s, 0));
                }
                1 => {
                    back_edges.insert((b, s));
                }
                _ => {}
            }
        } else {
            color[b] = 2;
            stack.pop();
        }
    }
    let reachable: Vec<bool> = color.iter().map(|c| *c != 0).collect();
    for (bi, r) in reachable.iter().enumerate() {
        if !r {
            err(errs, Some(bi), None, "unreachable block".to_string());
        }
    }

    // Cycles must TERMINATE: every reachable block must reach a row-ENDING
    // terminator (emit/skip/trap — emit.to continues, so it doesn't count).
    // This is the old acyclicity rule relaxed exactly enough for
    // multiplicity loops; a cycle with no exit still errors.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (bi, b) in p.blocks.iter().enumerate() {
        for (succ, _) in b.term.successors() {
            let s = succ.0 as usize;
            if s < n {
                preds[s].push(bi);
            }
        }
    }
    let mut reaches_end = vec![false; n];
    let mut work: Vec<usize> = (0..n)
        .filter(|&b| {
            matches!(
                p.blocks[b].term,
                Term::Emit | Term::Skip | Term::Trap { .. }
            )
        })
        .collect();
    for &b in &work {
        reaches_end[b] = true;
    }
    while let Some(b) = work.pop() {
        for &pb in &preds[b] {
            if !reaches_end[pb] {
                reaches_end[pb] = true;
                work.push(pb);
            }
        }
    }
    let mut nonterminating = false;
    for bi in 0..n {
        if reachable[bi] && !reaches_end[bi] {
            nonterminating = true;
            err(
                errs,
                Some(bi),
                None,
                "control-flow cycle with no path to emit/skip/trap (cannot terminate)"
                    .to_string(),
            );
        }
    }
    if nonterminating {
        return; // the store dataflow's topo order needs terminating loops
    }

    // Store dataflow over the DAG in topological order. State: per out
    // column, how many times it has been stored on every path reaching this
    // point; all paths into a join must agree.
    let ncols = p.out_cols.len();
    let mut entry_state: Vec<Option<Vec<u8>>> = vec![None; n];
    entry_state[0] = Some(vec![0; ncols]);

    for bi in topo_order(p, n, &reachable, &back_edges) {
        let Some(state) = entry_state[bi].clone() else {
            continue; // unreachable; already reported
        };
        let mut state_out = state;
        for (ii, inst) in p.blocks[bi].insts.iter().enumerate() {
            let col = match inst {
                Inst::Store { col, .. } | Inst::StoreOpt { col, .. } => *col as usize,
                _ => continue,
            };
            if col < ncols {
                state_out[col] = state_out[col].saturating_add(1);
                if state_out[col] > 1 {
                    err(
                        errs,
                        Some(bi),
                        Some(ii),
                        format!(
                            "out.{} stored more than once on this path",
                            p.out_cols[col].name
                        ),
                    );
                }
            }
        }
        match &p.blocks[bi].term {
            Term::Emit => {
                for (ci, count) in state_out.iter().enumerate() {
                    if *count == 0 {
                        err(
                            errs,
                            Some(bi),
                            None,
                            format!("emit without storing out.{}", p.out_cols[ci].name),
                        );
                    }
                }
            }
            Term::EmitTo { .. } => {
                for (ci, count) in state_out.iter().enumerate() {
                    if *count == 0 {
                        err(
                            errs,
                            Some(bi),
                            None,
                            format!("emit without storing out.{}", p.out_cols[ci].name),
                        );
                    }
                }
                // The emitted row is complete; the CONTINUATION starts the
                // next output row from scratch.
                let zero = vec![0u8; state_out.len()];
                for (succ, _) in p.blocks[bi].term.successors() {
                    let s = succ.0 as usize;
                    if s >= n {
                        continue;
                    }
                    match &entry_state[s] {
                        None => entry_state[s] = Some(zero.clone()),
                        Some(existing) if *existing != zero => {
                            err(
                                errs,
                                Some(s),
                                None,
                                "paths joining here disagree on which out columns are stored"
                                    .to_string(),
                            );
                        }
                        Some(_) => {}
                    }
                }
            }
            Term::Skip => {
                if state_out.iter().any(|c| *c > 0) {
                    err(
                        errs,
                        Some(bi),
                        None,
                        "skip after storing (a skipped row must store nothing)".to_string(),
                    );
                }
            }
            Term::Trap { .. } => {}
            _ => {
                for (succ, _) in p.blocks[bi].term.successors() {
                    let s = succ.0 as usize;
                    if s >= n {
                        continue;
                    }
                    match &entry_state[s] {
                        None => entry_state[s] = Some(state_out.clone()),
                        Some(existing) if *existing != state_out => {
                            err(
                                errs,
                                Some(s),
                                None,
                                "paths joining here disagree on which out columns are stored"
                                    .to_string(),
                            );
                        }
                        Some(_) => {}
                    }
                }
            }
        }
    }
}

/// Topological order of the reachable, acyclicity-checked subgraph via
/// Kahn's algorithm. Indegrees count only edges whose SOURCE is reachable:
/// an edge out of an unreachable island (which may itself be cyclic and
/// never drain) must not starve a reachable join of its dataflow visit —
/// that would mask the join's store errors behind the island's
/// unreachable-block errors and make them reappear one fix later.
fn topo_order(
    p: &Program,
    n: usize,
    reachable: &[bool],
    back_edges: &std::collections::HashSet<(usize, usize)>,
) -> Vec<usize> {
    let mut indegree = vec![0usize; n];
    for (bi, b) in p.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        for (succ, _) in b.term.successors() {
            let s = succ.0 as usize;
            if s < n && !back_edges.contains(&(bi, s)) {
                indegree[s] += 1;
            }
        }
    }
    let mut stack: Vec<usize> = (0..n)
        .filter(|&i| reachable[i] && indegree[i] == 0)
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(b) = stack.pop() {
        order.push(b);
        for (succ, _) in p.blocks[b].term.successors() {
            let s = succ.0 as usize;
            if s < n && !back_edges.contains(&(b, s)) {
                indegree[s] -= 1;
                if indegree[s] == 0 {
                    stack.push(s);
                }
            }
        }
    }
    order
}
