//! Cranelift-jit backend: the same verified IR, compiled to native code.
//!
//! Coverage-first (TASK-44 stretch plan): trivially-safe ops map inline to
//! CLIF (const scalars, float arith, int compares, logic, select, itof,
//! fabs); everything with nontrivial semantics — checked int arith, every
//! string op, loads/stores, statics — calls an `extern "C"` helper that
//! delegates to the SAME functions the interpreter uses (casemap,
//! substr_window, duck_fcmp, DuckF64, the arena). The two backends cannot
//! drift where they share code; the random-IR differential in tests.rs
//! guards the rest. Inlining hot helpers is a later, measured optimization.
//!
//! ABI: one JIT'd function per program, called once per row:
//! `extern "C" fn(*mut Cx) -> i64` returning 0 = emit, 1 = skip, 2 = a
//! helper trapped (message in `Cx::trap`), 3+k = the program's k-th
//! `Term::Trap`. `Cx` is `#[repr(C)]` and the generated code reads exactly
//! one field directly: `trap_flag` at offset 0, checked after every
//! fallible helper call. Str values live in the arena and travel through
//! the JIT'd code as two i64s (offset, length).
//!
//! A `CraneliftFn` owns a compiled `InterpFn` too: it provides the
//! input/state checks, `new_state`, the prepared statics the helpers read,
//! and an always-available fallback.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value as CVal,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use super::super::ir::{
    BinOp, CmpPred, Inst, Lit, NumOp1, Program, RoundMode, StaticTy, StrOp1, Term, TrimSide, Ty,
};
use super::interp::{
    self, apply_ord, col_valid, substr_range_ok, substr_window, CompileError, DuckF64, InterpFn,
    PreparedStatic,
};
use super::{casemap, Arena, Batch, ColData, KeyBits, OutCol, RunState, ScalarVal, StrRef, Trap};

// ------------------------------------------------------- runtime context --

/// Everything a helper can touch during one row. `#[repr(C)]` because the
/// JIT'd code loads `trap_flag` (guaranteed offset 0) directly; every other
/// field is only ever dereferenced from Rust inside helpers.
#[repr(C)]
struct Cx {
    trap_flag: u8,
    row: usize,
    arena: *mut Arena,
    out: *mut OutCol,
    out_len: usize,
    input: *const Batch,
    statics: *const PreparedStatic,
    statics_len: usize,
    trap: Option<Trap>,
}

impl Cx {
    fn arena(&mut self) -> &mut Arena {
        unsafe { &mut *self.arena }
    }
    fn out(&mut self) -> &mut [OutCol] {
        unsafe { std::slice::from_raw_parts_mut(self.out, self.out_len) }
    }
    fn input(&self) -> &Batch {
        unsafe { &*self.input }
    }
    fn statics(&self) -> &[PreparedStatic] {
        unsafe { std::slice::from_raw_parts(self.statics, self.statics_len) }
    }
    fn get(&mut self, off: i64, len: i64) -> String {
        // Owned copy: several helpers read one span and push another; a
        // borrow across arena mutation is not expressible here.
        self.arena()
            .get(StrRef {
                off: off as usize,
                len: len as usize,
            })
            .to_string()
    }
    fn set_trap(&mut self, msg: String) {
        self.trap_flag = 1;
        self.trap = Some(Trap(msg));
    }
}

/// Per-probe-site descriptor owned by the `CraneliftFn`; the JIT'd code
/// passes its absolute address to `h_probe`.
struct ProbeDesc {
    static_id: usize,
    key_tys: Vec<Ty>,
    val_tys: Vec<Ty>,
}

/// One 16-byte scratch cell: scalar payload in `[0]` (f64 as bits, bool as
/// 0/1), strings as (offset, length).
type Cell = [u64; 2];

// --------------------------------------------------------------- helpers --
// Each mirrors the interpreter closure for the same instruction — most by
// calling the identical shared function.

unsafe fn cx<'a>(p: *mut Cx) -> &'a mut Cx {
    unsafe { &mut *p }
}

extern "C" fn h_load_i1(p: *mut Cx, col: i64) -> u8 {
    let c = unsafe { cx(p) };
    match &c.input().cols[col as usize] {
        ColData::I1 { data, .. } => data[c.row] as u8,
        _ => unreachable!("load type checked by the verifier"),
    }
}

extern "C" fn h_load_i64(p: *mut Cx, col: i64) -> i64 {
    let c = unsafe { cx(p) };
    match &c.input().cols[col as usize] {
        ColData::I64 { data, .. } => data[c.row],
        _ => unreachable!("load type checked by the verifier"),
    }
}

extern "C" fn h_load_f64(p: *mut Cx, col: i64) -> f64 {
    let c = unsafe { cx(p) };
    match &c.input().cols[col as usize] {
        ColData::F64 { data, .. } => data[c.row],
        _ => unreachable!("load type checked by the verifier"),
    }
}

extern "C" fn h_load_str(p: *mut Cx, col: i64, len_out: *mut i64) -> i64 {
    let c = unsafe { cx(p) };
    // SAFETY: input and arena are disjoint allocations behind separate raw
    // pointers; materializing both references keeps the borrow checker out
    // of a copy that never aliases (and skips the old per-load String clone).
    let input = unsafe { &*c.input };
    let arena = unsafe { &mut *c.arena };
    let r = arena.push_str(input.cols[col as usize].str_at(c.row));
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_load_valid(p: *mut Cx, col: i64) -> u8 {
    let c = unsafe { cx(p) };
    col_valid(&c.input().cols[col as usize], c.row) as u8
}

extern "C" fn h_const_str(p: *mut Cx, ptr: i64, len: i64) -> i64 {
    let c = unsafe { cx(p) };
    let s = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr as *const u8, len as usize))
    };
    c.arena().push_str(s).off as i64
}

macro_rules! checked_bin {
    ($name:ident, $method:ident, $msg:literal) => {
        extern "C" fn $name(p: *mut Cx, a: i64, b: i64) -> i64 {
            match a.$method(b) {
                Some(v) => v,
                None => {
                    unsafe { cx(p) }.set_trap(format!($msg));
                    0
                }
            }
        }
    };
}
checked_bin!(h_iadd, checked_add, "i64 overflow in iadd");
checked_bin!(h_isub, checked_sub, "i64 overflow in isub");
checked_bin!(h_imul, checked_mul, "i64 overflow in imul");

macro_rules! checked_div {
    ($name:ident, $method:ident, $opname:literal) => {
        extern "C" fn $name(p: *mut Cx, a: i64, b: i64) -> i64 {
            if b == 0 {
                unsafe { cx(p) }.set_trap(format!("division by zero in {}", $opname));
                return 0;
            }
            match a.$method(b) {
                Some(v) => v,
                None => {
                    unsafe { cx(p) }.set_trap(format!("i64 overflow in {}", $opname));
                    0
                }
            }
        }
    };
}
checked_div!(h_idiv, checked_div, "idiv");
checked_div!(h_irem, checked_rem, "irem");

extern "C" fn h_frem(a: f64, b: f64) -> f64 {
    a % b
}

extern "C" fn h_fround(a: f64) -> f64 {
    a.round()
}

extern "C" fn h_iabs(p: *mut Cx, a: i64) -> i64 {
    match a.checked_abs() {
        Some(v) => v,
        None => {
            unsafe { cx(p) }.set_trap("i64 overflow in iabs".to_string());
            0
        }
    }
}

extern "C" fn h_fcmp(a: f64, b: f64, pred: i64) -> u8 {
    apply_ord(decode_pred(pred), super::duck_fcmp(a, b)) as u8
}

extern "C" fn h_scmp(p: *mut Cx, ao: i64, al: i64, bo: i64, bl: i64, pred: i64) -> u8 {
    let c = unsafe { cx(p) };
    let (a, b) = (c.get(ao, al), c.get(bo, bl));
    apply_ord(decode_pred(pred), a.as_str().cmp(b.as_str())) as u8
}

fn decode_pred(p: i64) -> CmpPred {
    match p {
        0 => CmpPred::Eq,
        1 => CmpPred::Ne,
        2 => CmpPred::Lt,
        3 => CmpPred::Le,
        4 => CmpPred::Gt,
        _ => CmpPred::Ge,
    }
}

fn encode_pred(p: CmpPred) -> i64 {
    match p {
        CmpPred::Eq => 0,
        CmpPred::Ne => 1,
        CmpPred::Lt => 2,
        CmpPred::Le => 3,
        CmpPred::Gt => 4,
        CmpPred::Ge => 5,
    }
}

extern "C" fn h_ftoi(p: *mut Cx, x: f64, round: i64) -> i64 {
    let r = if round != 0 { x.round() } else { x.trunc() };
    // Mirrors interp: 2^63 is exactly representable; [-2^63, 2^63) fits.
    if r.is_finite() && r >= -(2f64.powi(63)) && r < 2f64.powi(63) {
        r as i64
    } else {
        unsafe { cx(p) }.set_trap(format!("f64 value {x:?} out of i64 range in ftoi"));
        0
    }
}

extern "C" fn h_itos(p: *mut Cx, v: i64, len_out: *mut i64) -> i64 {
    let c = unsafe { cx(p) };
    let r = c.arena().push_str(&format!("{v}"));
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_ftos(p: *mut Cx, v: f64, len_out: *mut i64) -> i64 {
    let c = unsafe { cx(p) };
    let r = c.arena().push_str(&format!("{}", DuckF64(v)));
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_stoi(p: *mut Cx, off: i64, len: i64, valid_out: *mut u8) -> i64 {
    let c = unsafe { cx(p) };
    let s = c.get(off, len);
    match s.trim_ascii().parse::<i64>() {
        Ok(v) => {
            unsafe { *valid_out = 1 };
            v
        }
        Err(_) => {
            unsafe { *valid_out = 0 };
            0
        }
    }
}

extern "C" fn h_stof(p: *mut Cx, off: i64, len: i64, valid_out: *mut u8) -> f64 {
    let c = unsafe { cx(p) };
    let s = c.get(off, len);
    match s.trim_ascii().parse::<f64>() {
        Ok(v) => {
            unsafe { *valid_out = 1 };
            v
        }
        Err(_) => {
            unsafe { *valid_out = 0 };
            0.0
        }
    }
}

extern "C" fn h_sconcat(p: *mut Cx, ao: i64, al: i64, bo: i64, bl: i64, len_out: *mut i64) -> i64 {
    let c = unsafe { cx(p) };
    let r = c.arena().concat(
        StrRef {
            off: ao as usize,
            len: al as usize,
        },
        StrRef {
            off: bo as usize,
            len: bl as usize,
        },
    );
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_scase(p: *mut Cx, off: i64, len: i64, upper: i64, len_out: *mut i64) -> i64 {
    let c = unsafe { cx(p) };
    let s = c.get(off, len);
    let out: String = if upper != 0 {
        s.chars().map(casemap::simple_upper).collect()
    } else {
        s.chars().map(casemap::simple_lower).collect()
    };
    let r = c.arena().push_str(&out);
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_strim(
    p: *mut Cx,
    side: i64,
    ao: i64,
    al: i64,
    co: i64,
    cl: i64,
    len_out: *mut i64,
) -> i64 {
    let c = unsafe { cx(p) };
    let (s, set_s) = (c.get(ao, al), c.get(co, cl));
    let set: Vec<char> = set_s.chars().collect();
    let hit = |ch: char| set.contains(&ch);
    let t = match side {
        0 => s.trim_matches(hit),
        1 => s.trim_start_matches(hit),
        _ => s.trim_end_matches(hit),
    }
    .to_string();
    let r = c.arena().push_str(&t);
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

extern "C" fn h_ssubstr(
    p: *mut Cx,
    ao: i64,
    al: i64,
    start: i64,
    len: i64,
    has_len: i64,
    len_out: *mut i64,
) -> i64 {
    let c = unsafe { cx(p) };
    if !substr_range_ok(start) {
        c.set_trap("substring offset outside of supported range".to_string());
        unsafe { *len_out = 0 };
        return 0;
    }
    if has_len != 0 && !substr_range_ok(len) {
        c.set_trap("substring length outside of supported range".to_string());
        unsafe { *len_out = 0 };
        return 0;
    }
    let s = c.get(ao, al);
    let out = substr_window(&s, start, (has_len != 0).then_some(len));
    let r = c.arena().push_str(&out);
    unsafe { *len_out = r.len as i64 };
    r.off as i64
}

macro_rules! store_h {
    ($name:ident, $variant:ident, $ty:ty, $conv:expr) => {
        extern "C" fn $name(p: *mut Cx, col: i64, valid: u8, val: $ty) {
            let c = unsafe { cx(p) };
            match &mut c.out()[col as usize] {
                OutCol::$variant(v) => v.push((valid != 0, $conv(val))),
                _ => unreachable!("store type checked by the verifier"),
            }
        }
    };
}
store_h!(h_store_i1, I1, u8, |v: u8| v != 0);
store_h!(h_store_i64, I64, i64, |v| v);
store_h!(h_store_f64, F64, f64, |v| v);

extern "C" fn h_store_str(p: *mut Cx, col: i64, valid: u8, off: i64, len: i64) {
    let c = unsafe { cx(p) };
    let r = StrRef {
        off: off as usize,
        len: len as usize,
    };
    match &mut c.out()[col as usize] {
        OutCol::Str(v) => v.push((valid != 0, r)),
        _ => unreachable!("store type checked by the verifier"),
    }
}

/// Scalar static read; covers `sload` (flag ignored) and `sload.opt`.
extern "C" fn h_sload(p: *mut Cx, sid: i64, valid_out: *mut u8, cell_out: *mut Cell) {
    let c = unsafe { cx(p) };
    let PreparedStatic::Scalar { valid, val } = &c.statics()[sid as usize] else {
        unreachable!("static kind checked at compile");
    };
    let (valid, val) = (*valid, val.clone());
    // Mirrors interp: an invalid scalar's payload is the type default, the
    // stored value never leaks through a false flag.
    let val = if valid {
        val
    } else {
        match val.ty() {
            Ty::I1 => ScalarVal::I1(false),
            Ty::I64 => ScalarVal::I64(0),
            Ty::F64 => ScalarVal::F64(0.0),
            Ty::Str => ScalarVal::Str(String::new()),
        }
    };
    let cell = match val {
        ScalarVal::I1(b) => [b as u64, 0],
        ScalarVal::I64(i) => [i as u64, 0],
        ScalarVal::F64(f) => [f.to_bits(), 0],
        ScalarVal::Str(s) => {
            let r = c.arena().push_str(&s);
            [r.off as u64, r.len as u64]
        }
    };
    unsafe {
        *valid_out = valid as u8;
        *cell_out = cell;
    }
}

extern "C" fn h_probe(
    p: *mut Cx,
    desc: *const ProbeDesc,
    keys: *const Cell,
    outs: *mut Cell,
) -> u8 {
    let c = unsafe { cx(p) };
    let desc = unsafe { &*desc };
    let keys = unsafe { std::slice::from_raw_parts(keys, desc.key_tys.len()) };
    let PreparedStatic::Map { entries } = &c.statics()[desc.static_id] else {
        unreachable!("static kind checked at compile");
    };

    // Mirrors interp::cmp_key: canonical f64 bits, arena strings by byte
    // order. The arena is only read here, never written, so the borrow is
    // fine to hold across the search.
    let arena = unsafe { &*c.arena };
    let cmp = |stored: &[KeyBits]| -> std::cmp::Ordering {
        for (kb, cell) in stored.iter().zip(keys.iter()) {
            let ord = match kb {
                KeyBits::I1(s) => s.cmp(&(cell[0] != 0)),
                KeyBits::I64(s) => s.cmp(&(cell[0] as i64)),
                KeyBits::F64(s) => s.cmp(&super::canon_f64_bits(f64::from_bits(cell[0]))),
                KeyBits::Str(s) => {
                    let v = arena.get(StrRef {
                        off: cell[0] as usize,
                        len: cell[1] as usize,
                    });
                    s.as_str().cmp(v)
                }
            };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    };

    let found = entries.binary_search_by(|(k, _)| cmp(k)).ok();
    let hit = found.is_some();
    let values: Vec<ScalarVal> = match found {
        Some(idx) => entries[idx].1.clone(),
        None => desc
            .val_tys
            .iter()
            .map(|ty| match ty {
                Ty::I1 => ScalarVal::I1(false),
                Ty::I64 => ScalarVal::I64(0),
                Ty::F64 => ScalarVal::F64(0.0),
                Ty::Str => ScalarVal::Str(String::new()),
            })
            .collect(),
    };
    for (i, v) in values.into_iter().enumerate() {
        let cell = match v {
            ScalarVal::I1(b) => [b as u64, 0],
            ScalarVal::I64(x) => [x as u64, 0],
            ScalarVal::F64(f) => [f.to_bits(), 0],
            ScalarVal::Str(s) => {
                let r = c.arena().push_str(&s);
                [r.off as u64, r.len as u64]
            }
        };
        unsafe { *outs.add(i) = cell };
    }
    hit as u8
}

// ------------------------------------------------------------ the JIT'd fn --

type RowFn = extern "C" fn(*mut Cx) -> i64;

pub struct CraneliftFn {
    /// Keeps the executable memory alive; also the fallback + checker.
    _module: JITModule,
    row_fn: RowFn,
    interp: InterpFn,
    trap_msgs: Vec<String>,
    /// Owned backing for absolute addresses baked into the code.
    _const_strs: Vec<Box<str>>,
    _probe_descs: Vec<Box<ProbeDesc>>,
}

/// A JIT'd value: scalars are one CLIF value, strings are (offset, length).
#[derive(Clone, Copy)]
enum V {
    S(CVal),
    Str(CVal, CVal),
}

impl V {
    fn s(self) -> CVal {
        match self {
            V::S(v) => v,
            V::Str(..) => unreachable!("scalar use of a str value"),
        }
    }
    fn str2(self) -> (CVal, CVal) {
        match self {
            V::Str(o, l) => (o, l),
            V::S(_) => unreachable!("str use of a scalar value"),
        }
    }
}

fn clif_ty(ty: Ty) -> types::Type {
    match ty {
        Ty::I1 => types::I8,
        Ty::I64 => types::I64,
        Ty::F64 => types::F64,
        Ty::Str => unreachable!("str values are two i64s, expanded at use sites"),
    }
}

pub fn compile(p: &Program, statics: Vec<super::StaticData>) -> Result<CraneliftFn, CompileError> {
    // The interpreter compile also runs verify + prepare_statics; its
    // prepared statics are the ones the helpers read.
    let interp = interp::compile(p, statics)?;

    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
        .map_err(|e| CompileError::Static(format!("cranelift: no host ISA: {e}")))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| CompileError::Static(format!("cranelift: ISA: {e}")))?;

    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, ptr) in HELPERS {
        jb.symbol(*name, *ptr);
    }
    let mut module = JITModule::new(jb);
    let ptr_ty = module.target_config().pointer_type();

    // Declare helper signatures once.
    let mut helper_ids: HashMap<&'static str, FuncId> = HashMap::new();
    for (name, _) in HELPERS {
        let mut sig = module.make_signature();
        helper_sig(name, &mut sig, ptr_ty);
        let id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| CompileError::Static(format!("cranelift declare {name}: {e}")))?;
        helper_ids.insert(name, id);
    }

    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    ctx.func.signature.returns.push(AbiParam::new(types::I64));

    let mut const_strs: Vec<Box<str>> = Vec::new();
    let mut probe_descs: Vec<Box<ProbeDesc>> = Vec::new();
    let mut trap_msgs: Vec<String> = Vec::new();

    {
        let mut fbc = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbc);

        // Scratch stack slots: one 16-byte out-cell for helper out-params,
        // plus room for the widest probe's keys and values.
        let max_keys = p
            .statics
            .iter()
            .map(|s| match s {
                StaticTy::Map { keys, .. } => keys.len(),
                _ => 1,
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let max_vals = p
            .statics
            .iter()
            .map(|s| match s {
                StaticTy::Map { values, .. } => values.len(),
                _ => 1,
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let slot_out =
            b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 3));
        let slot_keys = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16 * max_keys as u32,
            3,
        ));
        let slot_vals = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16 * max_vals as u32,
            3,
        ));

        let entry = b.create_block();
        b.append_block_param(entry, ptr_ty);

        // One CLIF block per IR block, with params per the IR's block params
        // (strings expand to two).
        let blocks: Vec<_> = p.blocks.iter().map(|_| b.create_block()).collect();
        let mut vals: HashMap<u32, V> = HashMap::new();
        for (ib, cb) in p.blocks.iter().zip(blocks.iter()) {
            for (v, ty) in &ib.params {
                let pv = match ty {
                    Ty::Str => {
                        let o = b.append_block_param(*cb, types::I64);
                        let l = b.append_block_param(*cb, types::I64);
                        V::Str(o, l)
                    }
                    t => V::S(b.append_block_param(*cb, clif_ty(*t))),
                };
                vals.insert(v.0, pv);
            }
        }
        let trap_exit = b.create_block();

        b.switch_to_block(entry);
        let cxp = b.block_params(entry)[0];
        b.ins().jump(blocks[0], &[]);

        b.switch_to_block(trap_exit);
        let two = b.ins().iconst(types::I64, 2);
        b.ins().return_(&[two]);

        let call_h = |b: &mut FunctionBuilder,
                      module: &mut JITModule,
                      name: &str,
                      args: &[CVal]|
         -> Option<CVal> {
            let id = helper_ids[name];
            let fref = module.declare_func_in_func(id, b.func);
            let call = b.ins().call(fref, args);
            b.inst_results(call).first().copied()
        };

        for (bi, ib) in p.blocks.iter().enumerate() {
            b.switch_to_block(blocks[bi]);
            for inst in &ib.insts {
                translate_inst(
                    &mut b,
                    &mut module,
                    &call_h,
                    inst,
                    p,
                    cxp,
                    &mut vals,
                    slot_out,
                    slot_keys,
                    slot_vals,
                    &mut const_strs,
                    &mut probe_descs,
                    trap_exit,
                );
            }
            // Terminator.
            let arg_list = |vals: &HashMap<u32, V>,
                            args: &[super::super::ir::Value]|
             -> Vec<cranelift_codegen::ir::BlockArg> {
                let mut out = Vec::new();
                for a in args {
                    match vals[&a.0] {
                        V::S(v) => out.push(v.into()),
                        V::Str(o, l) => {
                            out.push(o.into());
                            out.push(l.into());
                        }
                    }
                }
                out
            };
            match &ib.term {
                Term::Jump { to, args } => {
                    let a = arg_list(&vals, args);
                    b.ins().jump(blocks[to.0 as usize], &a);
                }
                Term::Brif {
                    cond,
                    then_to,
                    then_args,
                    else_to,
                    else_args,
                } => {
                    let c = vals[&cond.0].s();
                    let ta = arg_list(&vals, then_args);
                    let ea = arg_list(&vals, else_args);
                    b.ins().brif(
                        c,
                        blocks[then_to.0 as usize],
                        &ta,
                        blocks[else_to.0 as usize],
                        &ea,
                    );
                }
                Term::Emit => {
                    let z = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[z]);
                }
                Term::Skip => {
                    let o = b.ins().iconst(types::I64, 1);
                    b.ins().return_(&[o]);
                }
                Term::Trap { msg } => {
                    let code = 3 + trap_msgs.len() as i64;
                    trap_msgs.push(msg.clone());
                    let v = b.ins().iconst(types::I64, code);
                    b.ins().return_(&[v]);
                }
            }
        }

        b.seal_all_blocks();
        b.finalize();
    }

    let fid = module
        .declare_function("row", Linkage::Export, &ctx.func.signature)
        .map_err(|e| CompileError::Static(format!("cranelift declare row: {e}")))?;
    module
        .define_function(fid, &mut ctx)
        .map_err(|e| CompileError::Static(format!("cranelift define: {e}")))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| CompileError::Static(format!("cranelift finalize: {e}")))?;
    let code = module.get_finalized_function(fid);
    let row_fn: RowFn = unsafe { std::mem::transmute(code) };

    Ok(CraneliftFn {
        _module: module,
        row_fn,
        interp,
        trap_msgs,
        _const_strs: const_strs,
        _probe_descs: probe_descs,
    })
}

impl CraneliftFn {
    pub fn new_state(&self) -> RunState {
        self.interp.new_state()
    }

    /// Same contract as [`InterpFn::run`].
    pub fn run(&self, input: &Batch, st: &mut RunState) -> Result<(), Trap> {
        self.interp.check_input(input)?;
        self.interp.check_state(st)?;
        st.arena.clear();
        st.emitted = 0;
        for col in st.out.iter_mut() {
            col.clear();
        }
        interp::reserve_out(&mut st.out, input.rows);

        let statics = self.interp.statics();
        let mut emitted = 0usize;
        for row in 0..input.rows {
            let mut cx = Cx {
                trap_flag: 0,
                row,
                arena: &mut st.arena,
                out: st.out.as_mut_ptr(),
                out_len: st.out.len(),
                input,
                statics: statics.as_ptr(),
                statics_len: statics.len(),
                trap: None,
            };
            match (self.row_fn)(&mut cx) {
                0 => emitted += 1,
                1 => {}
                2 => return Err(cx.trap.take().expect("helper trap sets the message")),
                k => return Err(Trap(self.trap_msgs[(k - 3) as usize].clone())),
            }
        }
        st.emitted = emitted;
        Ok(())
    }
}

// ------------------------------------------------------- inst translation --

#[allow(clippy::too_many_arguments)]
fn translate_inst(
    b: &mut FunctionBuilder,
    module: &mut JITModule,
    call_h: &dyn Fn(&mut FunctionBuilder, &mut JITModule, &str, &[CVal]) -> Option<CVal>,
    inst: &Inst,
    p: &Program,
    cxp: CVal,
    vals: &mut HashMap<u32, V>,
    slot_out: cranelift_codegen::ir::StackSlot,
    slot_keys: cranelift_codegen::ir::StackSlot,
    slot_vals: cranelift_codegen::ir::StackSlot,
    const_strs: &mut Vec<Box<str>>,
    probe_descs: &mut Vec<Box<ProbeDesc>>,
    trap_exit: cranelift_codegen::ir::Block,
) {
    // After any fallible helper: check trap_flag (offset 0 of Cx) and bail.
    let trap_check = |b: &mut FunctionBuilder| {
        let flag = b.ins().load(types::I8, MemFlags::trusted(), cxp, 0);
        let cont = b.create_block();
        b.ins().brif(flag, trap_exit, &[], cont, &[]);
        b.switch_to_block(cont);
    };
    let icon = |b: &mut FunctionBuilder, v: i64| b.ins().iconst(types::I64, v);

    match inst {
        Inst::Const { dst, lit } => {
            let v = match lit {
                Lit::I1(x) => V::S(b.ins().iconst(types::I8, *x as i64)),
                Lit::I64(x) => V::S(b.ins().iconst(types::I64, *x)),
                Lit::F64(x) => V::S(b.ins().f64const(*x)),
                Lit::Str(s) => {
                    let boxed: Box<str> = s.clone().into_boxed_str();
                    let ptr = boxed.as_ptr() as i64;
                    let len = boxed.len() as i64;
                    const_strs.push(boxed);
                    let pv = icon(b, ptr);
                    let lv = icon(b, len);
                    let off = call_h(b, module, "h_const_str", &[cxp, pv, lv]).unwrap();
                    V::Str(off, lv)
                }
            };
            vals.insert(dst.0, v);
        }
        Inst::Bin { op, dst, a, b: rhs } => {
            let (x, y) = (vals[&a.0].s(), vals[&rhs.0].s());
            let v = match op {
                BinOp::Iadd => call_h(b, module, "h_iadd", &[cxp, x, y]).unwrap(),
                BinOp::Isub => call_h(b, module, "h_isub", &[cxp, x, y]).unwrap(),
                BinOp::Imul => call_h(b, module, "h_imul", &[cxp, x, y]).unwrap(),
                BinOp::Idiv => call_h(b, module, "h_idiv", &[cxp, x, y]).unwrap(),
                BinOp::Irem => call_h(b, module, "h_irem", &[cxp, x, y]).unwrap(),
                BinOp::Fadd => b.ins().fadd(x, y),
                BinOp::Fsub => b.ins().fsub(x, y),
                BinOp::Fmul => b.ins().fmul(x, y),
                BinOp::Fdiv => b.ins().fdiv(x, y),
                BinOp::Frem => call_h(b, module, "h_frem", &[x, y]).unwrap(),
                BinOp::And => b.ins().band(x, y),
                BinOp::Or => b.ins().bor(x, y),
                BinOp::Xor => b.ins().bxor(x, y),
            };
            if matches!(
                op,
                BinOp::Iadd | BinOp::Isub | BinOp::Imul | BinOp::Idiv | BinOp::Irem
            ) {
                trap_check(b);
            }
            vals.insert(dst.0, V::S(v));
        }
        Inst::Cmp {
            pred,
            ty,
            dst,
            a,
            b: rhs,
        } => {
            let v = match ty {
                Ty::I64 => {
                    let cc = match pred {
                        CmpPred::Eq => IntCC::Equal,
                        CmpPred::Ne => IntCC::NotEqual,
                        CmpPred::Lt => IntCC::SignedLessThan,
                        CmpPred::Le => IntCC::SignedLessThanOrEqual,
                        CmpPred::Gt => IntCC::SignedGreaterThan,
                        CmpPred::Ge => IntCC::SignedGreaterThanOrEqual,
                    };
                    let (x, y) = (vals[&a.0].s(), vals[&rhs.0].s());
                    b.ins().icmp(cc, x, y)
                }
                Ty::F64 => {
                    let (x, y) = (vals[&a.0].s(), vals[&rhs.0].s());
                    let pv = icon(b, encode_pred(*pred));
                    call_h(b, module, "h_fcmp", &[x, y, pv]).unwrap()
                }
                Ty::Str => {
                    let (ao, al) = vals[&a.0].str2();
                    let (bo, bl) = vals[&rhs.0].str2();
                    let pv = icon(b, encode_pred(*pred));
                    call_h(b, module, "h_scmp", &[cxp, ao, al, bo, bl, pv]).unwrap()
                }
                Ty::I1 => unreachable!("cmp on i1 is rejected by the verifier"),
            };
            vals.insert(dst.0, V::S(v));
        }
        Inst::Not { dst, a } => {
            let v = b.ins().bxor_imm(vals[&a.0].s(), 1);
            vals.insert(dst.0, V::S(v));
        }
        Inst::Select {
            dst,
            cond,
            a,
            b: rhs,
        } => {
            let c = vals[&cond.0].s();
            let v = match (vals[&a.0], vals[&rhs.0]) {
                (V::S(x), V::S(y)) => V::S(b.ins().select(c, x, y)),
                (V::Str(xo, xl), V::Str(yo, yl)) => {
                    let o = b.ins().select(c, xo, yo);
                    let l = b.ins().select(c, xl, yl);
                    V::Str(o, l)
                }
                _ => unreachable!("select arms share a type"),
            };
            vals.insert(dst.0, v);
        }
        Inst::Itof { dst, a } => {
            let v = b.ins().fcvt_from_sint(types::F64, vals[&a.0].s());
            vals.insert(dst.0, V::S(v));
        }
        Inst::Ftoi { mode, dst, a } => {
            let r = icon(b, matches!(mode, RoundMode::Round) as i64);
            let v = call_h(b, module, "h_ftoi", &[cxp, vals[&a.0].s(), r]).unwrap();
            trap_check(b);
            vals.insert(dst.0, V::S(v));
        }
        Inst::Itos { dst, a } => {
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_itos", &[cxp, vals[&a.0].s(), lp]).unwrap();
            let len = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, len));
        }
        Inst::Ftos { dst, a } => {
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_ftos", &[cxp, vals[&a.0].s(), lp]).unwrap();
            let len = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, len));
        }
        Inst::StoiOpt { flag, dst, a } => {
            let (o, l) = vals[&a.0].str2();
            let vp = b.ins().stack_addr(types::I64, slot_out, 0);
            let v = call_h(b, module, "h_stoi", &[cxp, o, l, vp]).unwrap();
            let f = b.ins().stack_load(types::I8, slot_out, 0);
            vals.insert(flag.0, V::S(f));
            vals.insert(dst.0, V::S(v));
        }
        Inst::StofOpt { flag, dst, a } => {
            let (o, l) = vals[&a.0].str2();
            let vp = b.ins().stack_addr(types::I64, slot_out, 0);
            let v = call_h(b, module, "h_stof", &[cxp, o, l, vp]).unwrap();
            let f = b.ins().stack_load(types::I8, slot_out, 0);
            vals.insert(flag.0, V::S(f));
            vals.insert(dst.0, V::S(v));
        }
        Inst::Sconcat { dst, a, b: rhs } => {
            let (ao, al) = vals[&a.0].str2();
            let (bo, bl) = vals[&rhs.0].str2();
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_sconcat", &[cxp, ao, al, bo, bl, lp]).unwrap();
            let len = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, len));
        }
        Inst::Str1 { op, dst, a } => {
            let (o, l) = vals[&a.0].str2();
            let up = icon(b, matches!(op, StrOp1::Upper) as i64);
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_scase", &[cxp, o, l, up, lp]).unwrap();
            let len = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, len));
        }
        Inst::Strim {
            side,
            dst,
            a,
            chars,
        } => {
            let (ao, al) = vals[&a.0].str2();
            let (co, cl) = vals[&chars.0].str2();
            let sv = icon(
                b,
                match side {
                    TrimSide::Both => 0,
                    TrimSide::Lead => 1,
                    TrimSide::Trail => 2,
                },
            );
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_strim", &[cxp, sv, ao, al, co, cl, lp]).unwrap();
            let len = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, len));
        }
        Inst::Ssubstr { dst, a, start, len } => {
            let (ao, al) = vals[&a.0].str2();
            let sv = vals[&start.0].s();
            let (lv, has) = match len {
                Some(l) => (vals[&l.0].s(), icon(b, 1)),
                None => (icon(b, 0), icon(b, 0)),
            };
            let lp = b.ins().stack_addr(types::I64, slot_out, 0);
            let off = call_h(b, module, "h_ssubstr", &[cxp, ao, al, sv, lv, has, lp]).unwrap();
            trap_check(b);
            let lnew = b.ins().stack_load(types::I64, slot_out, 0);
            vals.insert(dst.0, V::Str(off, lnew));
        }
        Inst::Num1 { op, dst, a } => {
            let x = vals[&a.0].s();
            let v = match op {
                NumOp1::Iabs => {
                    let v = call_h(b, module, "h_iabs", &[cxp, x]).unwrap();
                    trap_check(b);
                    v
                }
                NumOp1::Fabs => b.ins().fabs(x),
                NumOp1::Fround => call_h(b, module, "h_fround", &[x]).unwrap(),
            };
            vals.insert(dst.0, V::S(v));
        }
        Inst::Load { dst, col } => {
            let ty = p.in_cols[*col as usize].ty.ty;
            let cv = icon(b, *col as i64);
            let v = match ty {
                Ty::I1 => V::S(call_h(b, module, "h_load_i1", &[cxp, cv]).unwrap()),
                Ty::I64 => V::S(call_h(b, module, "h_load_i64", &[cxp, cv]).unwrap()),
                Ty::F64 => V::S(call_h(b, module, "h_load_f64", &[cxp, cv]).unwrap()),
                Ty::Str => {
                    let lp = b.ins().stack_addr(types::I64, slot_out, 0);
                    let off = call_h(b, module, "h_load_str", &[cxp, cv, lp]).unwrap();
                    let len = b.ins().stack_load(types::I64, slot_out, 0);
                    V::Str(off, len)
                }
            };
            vals.insert(dst.0, v);
        }
        Inst::LoadOpt { flag, dst, col } => {
            // Mirrors interp: on a false flag the payload is NORMALIZED to
            // the type default (the input batch may carry garbage there).
            let ty = p.in_cols[*col as usize].ty.ty;
            let cv = icon(b, *col as i64);
            let f = call_h(b, module, "h_load_valid", &[cxp, cv]).unwrap();
            vals.insert(flag.0, V::S(f));
            let v = match ty {
                Ty::I1 => {
                    let raw = call_h(b, module, "h_load_i1", &[cxp, cv]).unwrap();
                    let zero = b.ins().iconst(types::I8, 0);
                    V::S(b.ins().select(f, raw, zero))
                }
                Ty::I64 => {
                    let raw = call_h(b, module, "h_load_i64", &[cxp, cv]).unwrap();
                    let zero = icon(b, 0);
                    V::S(b.ins().select(f, raw, zero))
                }
                Ty::F64 => {
                    let raw = call_h(b, module, "h_load_f64", &[cxp, cv]).unwrap();
                    let zero = b.ins().f64const(0.0);
                    V::S(b.ins().select(f, raw, zero))
                }
                Ty::Str => {
                    let lp = b.ins().stack_addr(types::I64, slot_out, 0);
                    let off = call_h(b, module, "h_load_str", &[cxp, cv, lp]).unwrap();
                    let len = b.ins().stack_load(types::I64, slot_out, 0);
                    let zero = icon(b, 0);
                    let o = b.ins().select(f, off, zero);
                    let l = b.ins().select(f, len, zero);
                    V::Str(o, l)
                }
            };
            vals.insert(dst.0, v);
        }
        Inst::Store { col, val } => {
            let one = b.ins().iconst(types::I8, 1);
            store(b, module, call_h, p, cxp, vals, *col, one, *val);
        }
        Inst::StoreOpt { col, flag, val } => {
            let f = vals[&flag.0].s();
            store(b, module, call_h, p, cxp, vals, *col, f, *val);
        }
        Inst::Probe {
            static_id,
            hit,
            dsts,
            keys,
        } => {
            let StaticTy::Map {
                keys: key_tys,
                values: val_tys,
            } = &p.statics[*static_id as usize]
            else {
                unreachable!("probe on non-map is rejected by the verifier");
            };
            // Write the key cells.
            for (i, k) in keys.iter().enumerate() {
                let base = (16 * i) as i32;
                match vals[&k.0] {
                    V::S(v) => {
                        let ty = key_tys[i];
                        let as64 = match ty {
                            Ty::I1 => b.ins().uextend(types::I64, v),
                            Ty::I64 => v,
                            Ty::F64 => b.ins().bitcast(types::I64, MemFlags::new(), v),
                            Ty::Str => unreachable!(),
                        };
                        b.ins().stack_store(as64, slot_keys, base);
                    }
                    V::Str(o, l) => {
                        b.ins().stack_store(o, slot_keys, base);
                        b.ins().stack_store(l, slot_keys, base + 8);
                    }
                }
            }
            let desc = Box::new(ProbeDesc {
                static_id: *static_id as usize,
                key_tys: key_tys.clone(),
                val_tys: val_tys.clone(),
            });
            let desc_ptr = &*desc as *const ProbeDesc as i64;
            probe_descs.push(desc);
            let dp = icon(b, desc_ptr);
            let kp = b.ins().stack_addr(types::I64, slot_keys, 0);
            let vp = b.ins().stack_addr(types::I64, slot_vals, 0);
            let h = call_h(b, module, "h_probe", &[cxp, dp, kp, vp]).unwrap();
            vals.insert(hit.0, V::S(h));
            for (i, d) in dsts.iter().enumerate() {
                let base = (16 * i) as i32;
                let v = match val_tys[i] {
                    Ty::I1 => {
                        let x = b.ins().stack_load(types::I64, slot_vals, base);
                        V::S(b.ins().ireduce(types::I8, x))
                    }
                    Ty::I64 => V::S(b.ins().stack_load(types::I64, slot_vals, base)),
                    Ty::F64 => {
                        let x = b.ins().stack_load(types::I64, slot_vals, base);
                        V::S(b.ins().bitcast(types::F64, MemFlags::new(), x))
                    }
                    Ty::Str => {
                        let o = b.ins().stack_load(types::I64, slot_vals, base);
                        let l = b.ins().stack_load(types::I64, slot_vals, base + 8);
                        V::Str(o, l)
                    }
                };
                vals.insert(d.0, v);
            }
        }
        Inst::Sload { static_id, dst } => {
            sload(
                b, module, call_h, p, cxp, vals, *static_id, None, *dst, slot_out, slot_vals,
            );
        }
        Inst::SloadOpt {
            static_id,
            flag,
            dst,
        } => {
            sload(
                b,
                module,
                call_h,
                p,
                cxp,
                vals,
                *static_id,
                Some(*flag),
                *dst,
                slot_out,
                slot_vals,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn store(
    b: &mut FunctionBuilder,
    module: &mut JITModule,
    call_h: &dyn Fn(&mut FunctionBuilder, &mut JITModule, &str, &[CVal]) -> Option<CVal>,
    p: &Program,
    cxp: CVal,
    vals: &HashMap<u32, V>,
    col: u32,
    valid: CVal,
    val: super::super::ir::Value,
) {
    let ty = p.out_cols[col as usize].ty.ty;
    let cv = b.ins().iconst(types::I64, col as i64);
    match ty {
        Ty::I1 => {
            let v = vals[&val.0].s();
            call_h(b, module, "h_store_i1", &[cxp, cv, valid, v]);
        }
        Ty::I64 => {
            let v = vals[&val.0].s();
            call_h(b, module, "h_store_i64", &[cxp, cv, valid, v]);
        }
        Ty::F64 => {
            let v = vals[&val.0].s();
            call_h(b, module, "h_store_f64", &[cxp, cv, valid, v]);
        }
        Ty::Str => {
            let (o, l) = vals[&val.0].str2();
            call_h(b, module, "h_store_str", &[cxp, cv, valid, o, l]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sload(
    b: &mut FunctionBuilder,
    module: &mut JITModule,
    call_h: &dyn Fn(&mut FunctionBuilder, &mut JITModule, &str, &[CVal]) -> Option<CVal>,
    p: &Program,
    cxp: CVal,
    vals: &mut HashMap<u32, V>,
    static_id: u32,
    flag: Option<super::super::ir::Value>,
    dst: super::super::ir::Value,
    slot_out: cranelift_codegen::ir::StackSlot,
    slot_vals: cranelift_codegen::ir::StackSlot,
) {
    let StaticTy::Scalar(ct) = &p.statics[static_id as usize] else {
        unreachable!("sload on non-scalar is rejected by the verifier");
    };
    let sv = b.ins().iconst(types::I64, static_id as i64);
    let fp = b.ins().stack_addr(types::I64, slot_out, 0);
    let cp = b.ins().stack_addr(types::I64, slot_vals, 0);
    call_h(b, module, "h_sload", &[cxp, sv, fp, cp]);
    if let Some(f) = flag {
        let fv = b.ins().stack_load(types::I8, slot_out, 0);
        vals.insert(f.0, V::S(fv));
    }
    let v = match ct.ty {
        Ty::I1 => {
            let x = b.ins().stack_load(types::I64, slot_vals, 0);
            V::S(b.ins().ireduce(types::I8, x))
        }
        Ty::I64 => V::S(b.ins().stack_load(types::I64, slot_vals, 0)),
        Ty::F64 => {
            let x = b.ins().stack_load(types::I64, slot_vals, 0);
            V::S(b.ins().bitcast(types::F64, MemFlags::new(), x))
        }
        Ty::Str => {
            let o = b.ins().stack_load(types::I64, slot_vals, 0);
            let l = b.ins().stack_load(types::I64, slot_vals, 8);
            V::Str(o, l)
        }
    };
    vals.insert(dst.0, v);
}

// -------------------------------------------------- helper symbol table --

const HELPERS: &[(&str, *const u8)] = &[
    ("h_load_i1", h_load_i1 as *const u8),
    ("h_load_i64", h_load_i64 as *const u8),
    ("h_load_f64", h_load_f64 as *const u8),
    ("h_load_str", h_load_str as *const u8),
    ("h_load_valid", h_load_valid as *const u8),
    ("h_const_str", h_const_str as *const u8),
    ("h_iadd", h_iadd as *const u8),
    ("h_isub", h_isub as *const u8),
    ("h_imul", h_imul as *const u8),
    ("h_idiv", h_idiv as *const u8),
    ("h_irem", h_irem as *const u8),
    ("h_frem", h_frem as *const u8),
    ("h_fround", h_fround as *const u8),
    ("h_iabs", h_iabs as *const u8),
    ("h_fcmp", h_fcmp as *const u8),
    ("h_scmp", h_scmp as *const u8),
    ("h_ftoi", h_ftoi as *const u8),
    ("h_itos", h_itos as *const u8),
    ("h_ftos", h_ftos as *const u8),
    ("h_stoi", h_stoi as *const u8),
    ("h_stof", h_stof as *const u8),
    ("h_sconcat", h_sconcat as *const u8),
    ("h_scase", h_scase as *const u8),
    ("h_strim", h_strim as *const u8),
    ("h_ssubstr", h_ssubstr as *const u8),
    ("h_store_i1", h_store_i1 as *const u8),
    ("h_store_i64", h_store_i64 as *const u8),
    ("h_store_f64", h_store_f64 as *const u8),
    ("h_store_str", h_store_str as *const u8),
    ("h_sload", h_sload as *const u8),
    ("h_probe", h_probe as *const u8),
];

fn helper_sig(name: &str, sig: &mut cranelift_codegen::ir::Signature, ptr: types::Type) {
    use types::{F64, I64, I8};
    let (params, ret): (&[types::Type], Option<types::Type>) = match name {
        "h_load_i1" | "h_load_valid" => (&[ptr, I64], Some(I8)),
        "h_load_i64" => (&[ptr, I64], Some(I64)),
        "h_load_f64" => (&[ptr, I64], Some(F64)),
        "h_load_str" => (&[ptr, I64, I64], Some(I64)),
        "h_const_str" => (&[ptr, I64, I64], Some(I64)),
        "h_iadd" | "h_isub" | "h_imul" | "h_idiv" | "h_irem" => (&[ptr, I64, I64], Some(I64)),
        "h_frem" => (&[F64, F64], Some(F64)),
        "h_fround" => (&[F64], Some(F64)),
        "h_iabs" => (&[ptr, I64], Some(I64)),
        "h_fcmp" => (&[F64, F64, I64], Some(I8)),
        "h_scmp" => (&[ptr, I64, I64, I64, I64, I64], Some(I8)),
        "h_ftoi" => (&[ptr, F64, I64], Some(I64)),
        "h_itos" => (&[ptr, I64, I64], Some(I64)),
        "h_ftos" => (&[ptr, F64, I64], Some(I64)),
        "h_stoi" => (&[ptr, I64, I64, I64], Some(I64)),
        "h_stof" => (&[ptr, I64, I64, I64], Some(F64)),
        "h_sconcat" => (&[ptr, I64, I64, I64, I64, I64], Some(I64)),
        "h_scase" => (&[ptr, I64, I64, I64, I64], Some(I64)),
        "h_strim" => (&[ptr, I64, I64, I64, I64, I64, I64], Some(I64)),
        "h_ssubstr" => (&[ptr, I64, I64, I64, I64, I64, I64], Some(I64)),
        "h_store_i1" => (&[ptr, I64, I8, I8], None),
        "h_store_i64" => (&[ptr, I64, I8, I64], None),
        "h_store_f64" => (&[ptr, I64, I8, F64], None),
        "h_store_str" => (&[ptr, I64, I8, I64, I64], None),
        "h_sload" => (&[ptr, I64, I64, I64], None),
        "h_probe" => (&[ptr, I64, I64, I64], Some(I8)),
        _ => unreachable!("unknown helper {name}"),
    };
    for p in params {
        sig.params.push(AbiParam::new(*p));
    }
    if let Some(r) = ret {
        sig.returns.push(AbiParam::new(r));
    }
}

#[cfg(test)]
mod tests {
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    #[test]
    fn jit_smoke_add_two_i64() {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").unwrap();
        flags.set("is_pic", "false").unwrap();
        let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap();
        let mut module = JITModule::new(JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        ));

        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (x, y) = (b.block_params(entry)[0], b.block_params(entry)[1]);
        let sum = b.ins().iadd(x, y);
        b.ins().return_(&[sum]);
        b.finalize();

        let id = module
            .declare_function("add", Linkage::Export, &ctx.func.signature)
            .unwrap();
        module.define_function(id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
        module.finalize_definitions().unwrap();

        let code = module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(40, 2), 42);
    }
}
