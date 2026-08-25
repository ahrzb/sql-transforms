//! Interpreter-backend tests: every M-ir fixture executes against
//! hand-computed expectations; unverified IR and mismatched statics are
//! rejected; the steady state performs zero heap allocations (counting
//! global allocator); generated programs execute deterministically.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use super::super::ir::{fixtures, gen, parse::parse, verify::verify, Program, StaticTy, Ty};
use super::interp::{compile, CompileError};
use super::testutil::{batch, built, c_f64, c_i1, c_i64, c_str, rows, run_snapshot, snapshot};
use super::{tree_ensemble, Batch, ColData, KeyBits, OutCol, ScalarVal, StaticData, Trap};

// ------------------------------------------------- counting allocator --
// Thread-local counter so parallel tests don't disturb each other; const-
// initialized Cell so the TLS access itself never allocates. try_with
// swallows accesses during thread teardown.

struct CountingAlloc;

std::thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

fn alloc_count() -> u64 {
    ALLOCS.with(|c| c.get())
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static GA: CountingAlloc = CountingAlloc;

// ------------------------------------------------------------- helpers --

// ------------------------------------------------------------ fixtures --

#[test]
fn projection_fixture_executes() {
    let p = built(fixtures::PROJECTION);
    let statics = vec![StaticData::Map(vec![(
        vec![KeyBits::Str("a".into())],
        vec![ScalarVal::F64(10.0)],
    )])];
    let f = compile(&p, statics).unwrap();
    let input = batch(
        3,
        vec![
            c_i64(&[Some(40), None, Some(40)]),
            c_str(&[Some("a"), Some("a"), Some("x")]),
        ],
    );
    // Row 1: 40/10 = 4. Row 2: age NULL -> NULL. Row 3: probe miss -> NULL.
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["4.0"], &["NULL"], &["NULL"]])
    );
}

#[test]
fn filter_fixture_drops_rows() {
    let p = built(fixtures::FILTER);
    let f = compile(&p, vec![]).unwrap();
    let input = batch(3, vec![c_f64(&[Some(1.5), Some(-2.0), Some(0.0)])]);
    assert_eq!(run_snapshot(&f, &input).unwrap(), rows(&[&["1.5"]]));
}

#[test]
fn case_diamond_fixture_executes() {
    let p = built(fixtures::CASE_DIAMOND);
    let f = compile(&p, vec![]).unwrap();
    let input = batch(3, vec![c_i64(&[Some(31), Some(30), Some(5)])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["old", "false"], &["old", "false"], &["young", "false"],])
    );
}

#[test]
fn casts_fixture_executes_and_traps() {
    let p = built(fixtures::CASTS);
    let statics = |snd: StaticData| {
        vec![
            StaticData::Scalar {
                valid: true,
                val: ScalarVal::F64(3.5),
            },
            snd,
        ]
    };
    // @1 NULL: n = nearest(3.5) - trunc(3.5) = 4 - 3 = 1; msg = select over
    // scmp.eq("3.5:", ":") = false -> ":". 3.5 rather than 2.5 so the two
    // opcodes still disagree now that `nearest` is half-to-EVEN (TASK-70) —
    // nearest(2.5) == trunc(2.5) == 2 would make this fixture blind to a
    // collapse of the two modes.
    let f = compile(
        &p,
        statics(StaticData::Scalar {
            valid: false,
            val: ScalarVal::I64(0),
        }),
    )
    .unwrap();
    let input = batch(1, vec![c_str(&[Some("12")])]);
    assert_eq!(run_snapshot(&f, &input).unwrap(), rows(&[&["1", ":"]]));

    // @1 = 7: the select picks the static instead.
    let f7 = compile(
        &p,
        statics(StaticData::Scalar {
            valid: true,
            val: ScalarVal::I64(7),
        }),
    )
    .unwrap();
    assert_eq!(run_snapshot(&f7, &input).unwrap(), rows(&[&["7", ":"]]));

    // Unparseable input routes to the trap block and aborts the call.
    let bad = batch(1, vec![c_str(&[Some("xx")])]);
    assert_eq!(
        run_snapshot(&f, &bad).unwrap_err(),
        Trap("cast to i64 failed".into())
    );
}

#[test]
fn kitchen_fixture_executes() {
    let p = built(fixtures::KITCHEN);
    let statics = vec![StaticData::Map(vec![(
        vec![KeyBits::I64(2), KeyBits::Str("k1".into())],
        vec![ScalarVal::F64(0.5), ScalarVal::I64(9)],
    )])];
    let f = compile(&p, statics).unwrap();
    let input = batch(
        2,
        vec![
            c_i64(&[Some(4), Some(10)]),
            c_i64(&[Some(2), None]),
            c_str(&[Some("k1"), Some("nope")]),
            c_f64(&[Some(1.5), Some(-1.0)]),
        ],
    );
    // Row 1: probe hit (2,"k1"); r = select(2.25<1.5, ..., x)=1.5 valid;
    //        s = ftos(stof(itos(9))) = "9.0".
    // Row 2: b NULL -> r NULL; probe miss -> defaults -> s = "0.0".
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["1.5", "9.0"], &["NULL", "0.0"]])
    );
}

// ------------------------------------------------------------ rejects --

#[test]
fn rejects_unverified_program() {
    // Parses fine, fails verification (iadd on f64).
    let p = parse(
        r#"fn f(in: batch{a: f64}, out: batch{o: f64}) {
entry:
  %x = load in.a
  %y = iadd %x, %x
  %z = itof %y
  store out.o, %z
  emit
}"#,
    )
    .unwrap();
    assert!(
        verify(&p).is_err(),
        "precondition: program must be unverifiable"
    );
    match compile(&p, vec![]) {
        Err(CompileError::Verify(errs)) => assert!(!errs.is_empty()),
        Err(CompileError::Static(m)) | Err(CompileError::Regex(m)) => {
            panic!("wrong error kind: {m}")
        }
        Ok(_) => panic!("compile accepted an unverified program"),
    }
}

#[test]
fn rejects_mismatched_statics() {
    let p = built(fixtures::PROJECTION); // declares one map(str) -> (f64)
    for (data, needle) in [
        (vec![], "declares 1 static(s), 0 provided"),
        (
            vec![StaticData::Scalar {
                valid: true,
                val: ScalarVal::F64(1.0),
            }],
            "declared map, got scalar",
        ),
        (
            vec![StaticData::Map(vec![(
                vec![KeyBits::I64(1)],
                vec![ScalarVal::F64(1.0)],
            )])],
            "entry 0 has shape",
        ),
        (
            vec![StaticData::Map(vec![
                (vec![KeyBits::Str("a".into())], vec![ScalarVal::F64(1.0)]),
                (vec![KeyBits::Str("a".into())], vec![ScalarVal::F64(2.0)]),
            ])],
            "duplicate map key",
        ),
    ] {
        match compile(&p, data) {
            Err(CompileError::Static(msg)) => {
                assert!(msg.contains(needle), "expected '{needle}' in '{msg}'")
            }
            Err(CompileError::Verify(_)) | Err(CompileError::Regex(_)) => {
                panic!("fixture failed verify?")
            }
            Ok(_) => panic!("compile accepted bad statics (wanted '{needle}')"),
        }
    }
}

#[test]
fn rejects_input_shape_mismatch() {
    let p = built(fixtures::FILTER); // one f64 column
    let f = compile(&p, vec![]).unwrap();
    let mut st = f.new_state();
    let wrong_ty = batch(1, vec![c_i64(&[Some(1)])]);
    assert!(f.run(&wrong_ty, &mut st).unwrap_err().0.contains("is i64"));
    let wrong_count = batch(1, vec![]);
    assert!(f
        .run(&wrong_count, &mut st)
        .unwrap_err()
        .0
        .contains("0 column(s)"));
    let wrong_len = Batch {
        rows: 2,
        cols: vec![c_f64(&[Some(1.0)])],
    };
    assert!(f
        .run(&wrong_len, &mut st)
        .unwrap_err()
        .0
        .contains("1 row(s)"));
}

// ---------------------------------------------------------- allocation --

/// AC#3: after warmup, run() performs ZERO heap allocations — registers,
/// arena, and output builders are all reused; varlen goes arena-only.
/// PROJECTION covers str loads, a probe, arithmetic, and store.opt.
#[test]
fn steady_state_run_allocates_nothing() {
    let p = built(fixtures::PROJECTION);
    let statics = vec![StaticData::Map(vec![(
        vec![KeyBits::Str("a".into())],
        vec![ScalarVal::F64(10.0)],
    )])];
    let f = compile(&p, statics).unwrap();
    let input = batch(
        3,
        vec![
            c_i64(&[Some(40), None, Some(40)]),
            c_str(&[Some("a"), Some("a"), Some("x")]),
        ],
    );
    let mut st = f.new_state();
    // Warm: buffers reach steady capacity.
    f.run(&input, &mut st).unwrap();
    f.run(&input, &mut st).unwrap();

    let before = alloc_count();
    for _ in 0..5 {
        f.run(&input, &mut st).unwrap();
    }
    let delta = alloc_count() - before;
    assert_eq!(delta, 0, "steady-state run heap-allocated {delta} time(s)");

    // The cranelift backend honors the same contract — its helpers share
    // the interpreter's semantic functions, so they must also share the
    // no-alloc property (TASK-45 AC #4).
    let statics = vec![StaticData::Map(vec![(
        vec![KeyBits::Str("a".into())],
        vec![ScalarVal::F64(10.0)],
    )])];
    let cf = super::cranelift::compile(&p, statics).unwrap();
    let mut cst = cf.new_state();
    cf.run(&input, &mut cst).unwrap();
    cf.run(&input, &mut cst).unwrap();

    let before = alloc_count();
    for _ in 0..5 {
        cf.run(&input, &mut cst).unwrap();
    }
    let delta = alloc_count() - before;
    assert_eq!(
        delta, 0,
        "cranelift steady state heap-allocated {delta} time(s)"
    );
}

/// The string surface — case map, trim, substr, concat, int/float text,
/// parse, compare — is arena-only in steady state on BOTH backends. These
/// ops all used to build temp Strings per row (TASK-45 AC #4).
#[test]
fn steady_state_string_ops_allocate_nothing() {
    let p = built(
        r#"
fn stringy(in: batch{s: str, t: str, n: i64}, out: batch{a: str, b: str}) {
entry:
  %s = load in.s
  %t = load in.t
  %up = supper %s
  %lo = slower %up
  %set = const.str " x"
  %tr = strim.both %lo, %set
  %one = const.i64 1
  %three = const.i64 3
  %sub = ssubstr %tr, %one, %three
  %cat = sconcat %sub, %up
  %n = load in.n
  %ns = itos %n
  %cat2 = sconcat %cat, %ns
  store out.a, %cat2
  %ok, %iv = stoi.opt %ns
  %f = itof %iv
  %fs = ftos %f
  %eq = scmp.eq %s, %t
  %sel = select %eq, %fs, %tr
  %sel2 = select %ok, %sel, %up
  store out.b, %sel2
  emit
}
"#,
    );
    let input = batch(
        3,
        vec![
            c_str(&[Some("  héLLo x"), Some("wörld"), Some("")]),
            c_str(&[Some("wörld"), Some("wörld"), Some("a")]),
            c_i64(&[Some(42), Some(-7), Some(0)]),
        ],
    );

    let f = compile(&p, vec![]).unwrap();
    let mut st = f.new_state();
    f.run(&input, &mut st).unwrap();
    f.run(&input, &mut st).unwrap();
    let before = alloc_count();
    for _ in 0..5 {
        f.run(&input, &mut st).unwrap();
    }
    let delta = alloc_count() - before;
    assert_eq!(
        delta, 0,
        "interp string steady state allocated {delta} time(s)"
    );

    let cf = super::cranelift::compile(&p, vec![]).unwrap();
    let mut cst = cf.new_state();
    cf.run(&input, &mut cst).unwrap();
    cf.run(&input, &mut cst).unwrap();
    let before = alloc_count();
    for _ in 0..5 {
        cf.run(&input, &mut cst).unwrap();
    }
    let delta = alloc_count() - before;
    assert_eq!(
        delta, 0,
        "cranelift string steady state allocated {delta} time(s)"
    );
}

// ----------------------------------------- adversarial-pass regressions --
// Each pins a fix for a confirmed finding from the 2026-07-26 adversarial
// workflow against the interpreter.

/// store.opt with a false flag stores the TYPE DEFAULT payload, never the
/// live register (spec pin in ir/mod.rs).
#[test]
fn store_opt_false_flag_stores_type_default() {
    let p = built(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64?, s: str?}) {
entry:
  %x = load in.a
  %f = const.i1 false
  store.opt out.o, %f, %x
  %t = itos %x
  store.opt out.s, %f, %t
  emit
}"#,
    );
    let f = compile(&p, vec![]).unwrap();
    let mut st = f.new_state();
    f.run(&batch(1, vec![c_i64(&[Some(42)])]), &mut st).unwrap();
    match &st.out[0] {
        OutCol::I64(v) => assert_eq!(v[0], (false, 0), "payload must be the default, not 42"),
        _ => unreachable!(),
    }
    match &st.out[1] {
        OutCol::Str(v) => {
            assert!(!v[0].0);
            assert_eq!(
                st.arena.get(v[0].1),
                "",
                "payload must be the empty default"
            );
        }
        _ => unreachable!(),
    }
}

/// A nullable column's validity vector is part of the input shape.
#[test]
fn short_validity_vector_traps() {
    let p = built(
        r#"fn f(in: batch{a: i64?}, out: batch{o: i64}) {
entry:
  %f, %v = load.opt in.a
  %z = select %f, %v, %v
  store out.o, %z
  emit
}"#,
    );
    let f = compile(&p, vec![]).unwrap();
    let mut st = f.new_state();
    let bad = Batch {
        rows: 1,
        cols: vec![ColData::I64 {
            valid: vec![],
            data: vec![7],
        }],
    };
    let err = f.run(&bad, &mut st).unwrap_err();
    assert!(err.0.contains("validity vector"), "wrong trap: {}", err.0);
}

/// Register slots are dense: sparse value ids must not inflate the frame.
#[test]
fn sparse_value_ids_use_dense_register_slots() {
    use super::super::ir::{Block, Col, ColTy, Inst, Lit, Program, Term, Ty, Value};
    let p = Program {
        statics: vec![],
        regexes: vec![],
        externs: vec![],
        name: "sparse".into(),
        in_cols: vec![],
        out_cols: vec![Col {
            name: "o".into(),
            ty: ColTy {
                ty: Ty::I64,
                nullable: false,
            },
        }],
        blocks: vec![Block {
            params: vec![],
            insts: vec![
                Inst::Const {
                    dst: Value(9_999_999),
                    lit: Lit::I64(5),
                },
                Inst::Store {
                    col: 0,
                    val: Value(9_999_999),
                },
            ],
            term: Term::Emit,
        }],
    };
    let f = compile(&p, vec![]).unwrap();
    let st = f.new_state();
    assert!(
        st.regs.len() <= 4,
        "frame sized by ids, not defs: {}",
        st.regs.len()
    );
}

/// The emitted counter reports the row count even without reading columns.
#[test]
fn emitted_counts_rows() {
    let p = built(fixtures::FILTER);
    let f = compile(&p, vec![]).unwrap();
    let mut st = f.new_state();
    f.run(
        &batch(3, vec![c_f64(&[Some(1.5), Some(-2.0), Some(0.0)])]),
        &mut st,
    )
    .unwrap();
    assert_eq!(st.emitted, 1);
}

// ------------------------------------------------------- semantics pins --
// One test per documented pin in interp.rs that the design review found
// untested. Each tiny program computes one edge through the text form.

/// Run a one-output program over no input rows... rather, over one dummy row.
fn eval1(body: &str, out_col: &str) -> Result<Vec<Vec<String>>, Trap> {
    let text = format!(
        "fn f(in: batch{{d: i64}}, out: batch{{{out_col}}}) {{\nentry:\n{body}\n  emit\n}}"
    );
    let p = built(&text);
    let f = compile(&p, vec![]).unwrap();
    run_snapshot(&f, &batch(1, vec![c_i64(&[Some(0)])]))
}

#[test]
fn pin_integer_overflow_and_division_traps() {
    // Overflow texts are DuckDB's own, verbatim with operand values
    // (wave-3 pins); division-by-zero texts are internal (unreachable
    // through SQL — the frontend CASE guard yields NULL first).
    for (expr, needle) in [
        (
            "  %a = const.i64 9223372036854775807\n  %b = const.i64 1\n  %r = iadd %a, %b",
            "Overflow in addition of INT64 (9223372036854775807 + 1)!",
        ),
        (
            "  %a = const.i64 -9223372036854775808\n  %b = const.i64 1\n  %r = isub %a, %b",
            "Overflow in subtraction of INT64 (-9223372036854775808 - 1)!",
        ),
        (
            "  %a = const.i64 4611686018427387904\n  %b = const.i64 4\n  %r = imul %a, %b",
            "Overflow in multiplication of INT64 (4611686018427387904 * 4)!",
        ),
        (
            "  %a = const.i64 1\n  %b = const.i64 0\n  %r = idiv %a, %b",
            "division by zero in idiv",
        ),
        (
            "  %a = const.i64 1\n  %b = const.i64 0\n  %r = irem %a, %b",
            "division by zero in irem",
        ),
        (
            "  %a = const.i64 -9223372036854775808\n  %b = const.i64 -1\n  %r = idiv %a, %b",
            "Overflow in division of -9223372036854775808 / -1",
        ),
        (
            "  %a = const.i64 -9223372036854775808\n  %b = const.i64 -1\n  %r = irem %a, %b",
            "Overflow in division of -9223372036854775808 / -1",
        ),
    ] {
        let body = format!("{expr}\n  store out.o, %r");
        let err = eval1(&body, "o: i64").unwrap_err();
        assert!(
            err.0.contains(needle),
            "expected '{needle}', got '{}'",
            err.0
        );
    }
}

#[test]
fn pin_fcmp_nan_ordering() {
    // DuckDB DOUBLE order (measured 1.5.5, stretch-4 pins): NaN sorts ABOVE
    // everything, so NaN vs 1.0 is gt/ge/ne — not the IEEE all-false.
    let body = "  %n = const.f64 nan\n  %x = const.f64 1.0\n\
                \x20 %eq = fcmp.eq %n, %x\n  %ne = fcmp.ne %n, %x\n\
                \x20 %lt = fcmp.lt %n, %x\n  %le = fcmp.le %n, %x\n\
                \x20 %gt = fcmp.gt %n, %x\n  %ge = fcmp.ge %n, %x\n\
                \x20 store out.eq, %eq\n  store out.ne, %ne\n  store out.lt, %lt\n\
                \x20 store out.le, %le\n  store out.gt, %gt\n  store out.ge, %ge";
    let got = eval1(body, "eq: i1, ne: i1, lt: i1, le: i1, gt: i1, ge: i1").unwrap();
    assert_eq!(
        got,
        rows(&[&["false", "true", "false", "false", "true", "true"]])
    );
}

#[test]
fn pin_fcmp_nan_eq_nan_and_zero_order() {
    // DuckDB DOUBLE order: nan = nan TRUE, nan > inf TRUE, -0.0 = 0.0 TRUE,
    // -0.0 < 0.0 FALSE (measured 1.5.5).
    let body = "  %n = const.f64 nan\n  %i = const.f64 inf\n\
                \x20 %nz = const.f64 -0.0\n  %pz = const.f64 0.0\n\
                \x20 %a = fcmp.eq %n, %n\n  %b = fcmp.gt %n, %i\n\
                \x20 %c = fcmp.eq %nz, %pz\n  %d = fcmp.lt %nz, %pz\n\
                \x20 store out.a, %a\n  store out.b, %b\n\
                \x20 store out.c, %c\n  store out.d, %d";
    let got = eval1(body, "a: i1, b: i1, c: i1, d: i1").unwrap();
    assert_eq!(got, rows(&[&["true", "true", "true", "false"]]));
}

#[test]
fn pin_frem_is_ieee_and_new_unaries() {
    // frem: sign of the dividend, x % 0.0 = NaN, never traps. iabs traps on
    // MIN (covered below); fabs clears the sign bit; fround is half away
    // from zero (all measured DuckDB 1.5.5).
    let body = "  %a = const.f64 -5.5\n  %b = const.f64 2.5\n  %z = const.f64 0.0\n\
                \x20 %r = frem %a, %b\n  %rz = frem %a, %z\n\
                \x20 %nz = const.f64 -0.0\n  %ab = fabs %nz\n\
                \x20 %h = const.f64 -2.5\n  %ro = fround %h\n\
                \x20 %i = const.i64 -5\n  %ia = iabs %i\n\
                \x20 %s = const.str \"  hi  \"\n  %sp = const.str \" \"\n\
                \x20 %t = strim.both %s, %sp\n  %u = supper %t\n\
                \x20 %one = const.i64 1\n  %sub = ssubstr %u, %one, %one\n\
                \x20 store out.r, %r\n  store out.rz, %rz\n  store out.ab, %ab\n\
                \x20 store out.ro, %ro\n  store out.ia, %ia\n  store out.sub, %sub";
    let got = eval1(body, "r: f64, rz: f64, ab: f64, ro: f64, ia: i64, sub: str").unwrap();
    assert_eq!(got, rows(&[&["-0.5", "NaN", "0.0", "-3.0", "5", "H"]]));
}

#[test]
fn pin_iabs_min_traps() {
    let body = "  %m = const.i64 -9223372036854775808\n  %a = iabs %m\n  store out.o, %a";
    let err = eval1(body, "o: i64").unwrap_err();
    assert!(err.0.contains("Overflow on abs"), "got '{}'", err.0);
}

#[test]
fn pin_ssubstr_window_arithmetic() {
    // DuckDB virtual-window semantics (measured 1.5.5 + adversarial census):
    // start 0 and negative starts map through n+start+1; a non-negative len
    // runs forward, a negative len slices BACKWARDS from the resolved start;
    // ssubstr.rest is the 2-arg "rest of the string" form.
    for (start, len, expect) in [
        (2i64, Some(3i64), "ell"),
        (0, Some(3), "he"),
        (-2, None, "lo"),
        (-6, Some(3), "hel"),
        (-10, Some(8), "hello"),
        (1, Some(0), ""),
        (1, Some(-1), ""),
        (3, Some(-2), "he"),
        (2, Some(-1), "h"),
        (6, Some(-5), "hello"),
        (-2, Some(-3), "hel"),
        (10, None, ""),
        (0, None, "hello"),
        (-4294967296, None, "hello"),
    ] {
        let op = match len {
            Some(l) => format!("  %ln = const.i64 {l}\n  %r = ssubstr %s, %st, %ln\n"),
            None => "  %r = ssubstr.rest %s, %st\n".to_string(),
        };
        let body =
            format!("  %s = const.str \"hello\"\n  %st = const.i64 {start}\n{op}  store out.o, %r");
        assert_eq!(
            eval1(&body, "o: str").unwrap(),
            rows(&[&[expect]]),
            "substr('hello', {start}, {len:?})"
        );
    }
}

#[test]
fn pin_ssubstr_range_guards_trap() {
    // DuckDB errors for offsets/lengths outside [-2^32, 2^32-1]; the 2-arg
    // form has no length to guard.
    for (start, len, needle) in [
        (4294967296i64, Some(2i64), "offset outside"),
        (-4294967297, Some(2), "offset outside"),
        (1, Some(4294967296), "length outside"),
        (1, Some(-4294967297), "length outside"),
    ] {
        let op = match len {
            Some(l) => format!("  %ln = const.i64 {l}\n  %r = ssubstr %s, %st, %ln\n"),
            None => "  %r = ssubstr.rest %s, %st\n".to_string(),
        };
        let body =
            format!("  %s = const.str \"hello\"\n  %st = const.i64 {start}\n{op}  store out.o, %r");
        let err = eval1(&body, "o: str").unwrap_err();
        assert!(
            err.0.contains(needle),
            "({start}, {len:?}): got '{}'",
            err.0
        );
    }
    // Boundary values inside the guard execute.
    for start in [4294967295i64, -4294967296] {
        let body = format!(
            "  %s = const.str \"hello\"\n  %st = const.i64 {start}\n\
             \x20 %r = ssubstr.rest %s, %st\n  store out.o, %r"
        );
        assert!(
            eval1(&body, "o: str").is_ok(),
            "start {start} should not trap"
        );
    }
}

#[test]
fn pin_ftoi_rounding_and_traps() {
    // `nearest` is half-to-EVEN: it is DuckDB's DOUBLE->BIGINT cast, and it
    // is deliberately NOT the SQL round() builtin, which is
    // half-away-from-zero and never reaches this opcode (TASK-70).
    for (lit, mode, expect) in [
        ("2.5", "nearest", "2"),
        ("-2.5", "nearest", "-2"),
        ("0.5", "nearest", "0"),
        ("-0.5", "nearest", "0"),
        ("1.5", "nearest", "2"),
        ("-1.5", "nearest", "-2"),
        ("3.5", "nearest", "4"),
        ("2.4", "nearest", "2"),
        ("2.6", "nearest", "3"),
        ("-2.6", "nearest", "-3"),
        ("2.5", "trunc", "2"),
        ("-2.5", "trunc", "-2"),
    ] {
        let body = format!("  %a = const.f64 {lit}\n  %r = ftoi.{mode} %a\n  store out.o, %r");
        assert_eq!(
            eval1(&body, "o: i64").unwrap(),
            rows(&[&[expect]]),
            "ftoi.{mode}({lit})"
        );
    }
    for lit in ["nan", "inf", "-inf", "1e19"] {
        let body = format!("  %a = const.f64 {lit}\n  %r = ftoi.trunc %a\n  store out.o, %r");
        let err = eval1(&body, "o: i64").unwrap_err();
        assert!(err.0.contains("out of i64 range"), "ftoi({lit}): {}", err.0);
    }
}

#[test]
fn pin_ieee_flow_and_scmp_and_concat() {
    let body = "  %z = const.f64 0.0\n  %o = const.f64 1.0\n  %m = const.f64 -1.0\n\
                \x20 %nan = fdiv %z, %z\n  %pinf = fdiv %o, %z\n  %ninf = fdiv %m, %z\n\
                \x20 store out.a, %nan\n  store out.b, %pinf\n  store out.c, %ninf\n\
                \x20 %s1 = const.str \"Z\"\n  %s2 = const.str \"a\"\n  %lt = scmp.lt %s1, %s2\n\
                \x20 store out.d, %lt\n\
                \x20 %e1 = const.str \"\"\n  %cc = sconcat %e1, %e1\n  store out.e, %cc";
    let got = eval1(body, "a: f64, b: f64, c: f64, d: i1, e: str").unwrap();
    assert_eq!(got, rows(&[&["NaN", "inf", "-inf", "true", ""]]));
}

#[test]
fn pin_stoi_trims_whitespace_like_duckdb_cast() {
    for (s, ok) in [
        (" 5", true),
        ("5 ", true),
        ("+5", true),
        ("0x10", false),
        ("", false),
        ("  ", false),
    ] {
        let body = format!(
            "  %s = const.str \"{s}\"\n  %f, %v = stoi.opt %s\n  store out.f, %f\n  store out.v, %v"
        );
        let got = eval1(&body, "f: i1, v: i64").unwrap();
        assert_eq!(got[0][0], ok.to_string(), "stoi({s:?})");
    }
}

#[test]
fn pin_load_opt_normalizes_garbage_payloads() {
    // Invalid slot carries 999; the flag is false so downstream must see the
    // type default, not the garbage.
    let p = built(
        r#"fn f(in: batch{a: i64?}, out: batch{o: i64}) {
entry:
  %f, %v = load.opt in.a
  store out.o, %v
  emit
}"#,
    );
    let f = compile(&p, vec![]).unwrap();
    let input = Batch {
        rows: 1,
        cols: vec![ColData::I64 {
            valid: vec![false],
            data: vec![999],
        }],
    };
    assert_eq!(run_snapshot(&f, &input).unwrap(), rows(&[&["0"]]));
}

// ---------------------------------------------------------------- fuzz --

fn gen_scalar(rng: &mut gen::Rng, ty: Ty) -> ScalarVal {
    match ty {
        Ty::I1 => ScalarVal::I1(rng.chance(50)),
        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ScalarVal::I64(rng.next() as i64 % 1000),
        Ty::F64 => ScalarVal::F64((rng.next() as i64 % 1000) as f64 / 4.0),
        Ty::Str => ScalarVal::Str(format!("s{}", rng.below(5))),
        Ty::Dec(p, s) => ScalarVal::Dec((rng.next() as i64 % 1000) as i128, p, s),
    }
}

/// Static data matching a program's declarations; entry keys made distinct
/// through the first component's index.
fn gen_statics(rng: &mut gen::Rng, p: &Program) -> Vec<StaticData> {
    p.statics
        .iter()
        .map(|st| match st {
            StaticTy::Scalar(ct) => StaticData::Scalar {
                valid: !ct.nullable || rng.chance(70),
                val: gen_scalar(rng, ct.ty),
            },
            StaticTy::Map { keys, values } => {
                let n = if keys[0] == Ty::I1 {
                    2
                } else {
                    1 + rng.below(3) as usize
                };
                let entries = (0..n)
                    .map(|j| {
                        let key: Vec<KeyBits> = keys
                            .iter()
                            .enumerate()
                            .map(|(pos, kt)| match (pos, kt.lane()) {
                                (0, Ty::I1) => KeyBits::I1(j % 2 == 1),
                                (0, Ty::I64) => KeyBits::I64(j as i64),
                                (0, Ty::F64) => KeyBits::F64((j as f64).to_bits()),
                                (0, Ty::Str) => KeyBits::Str(format!("k{j}")),
                                (_, Ty::I1) => KeyBits::I1(true),
                                (_, Ty::I64) => KeyBits::I64(7),
                                (_, Ty::F64) => KeyBits::F64(1.5f64.to_bits()),
                                _ => KeyBits::Str("fix".into()),
                            })
                            .collect();
                        let vals = values.iter().map(|vt| gen_scalar(rng, *vt)).collect();
                        (key, vals)
                    })
                    .collect();
                StaticData::Map(entries)
            }
            // A small random ensemble whose model 0 always exists, since the
            // generated program always scores id 0 (see gen.rs). This is what
            // puts `predict` under the backend-equality differential.
            StaticTy::Model { n_features } => {
                StaticData::Model(Box::new(gen_ensemble(rng, *n_features)))
            }
            // The generator never declares multimaps (see gen.rs).
            StaticTy::MultiMap { .. } | StaticTy::BatchMap { .. } => StaticData::Map(Vec::new()),
        })
        .collect()
}

/// A random 1-model ensemble of `1..=3` stumps-or-deeper trees over
/// `n_features`. Deliberately hand-rolled rather than extracted from a
/// library: confit never imports one.
fn gen_ensemble(rng: &mut gen::Rng, n_features: u32) -> tree_ensemble::TreeEnsemble {
    let (mut model_id, mut tree_id, mut node_id) = (Vec::new(), Vec::new(), Vec::new());
    let (mut feature, mut threshold) = (Vec::new(), Vec::new());
    let (mut left, mut right, mut missing_left, mut value) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in 0..1 + rng.below(3) as i64 {
        // Either a bare leaf or a depth-1 split, so both shapes appear.
        let split = rng.chance(70);
        let n = if split { 3 } else { 1 };
        for k in 0..n {
            model_id.push(0);
            tree_id.push(t);
            node_id.push(k);
            let is_split = split && k == 0;
            feature.push(if is_split {
                rng.below(n_features as u64) as i32
            } else {
                -1
            });
            threshold.push(if is_split {
                (rng.below(5) as f64) - 2.0
            } else {
                0.0
            });
            left.push(if is_split { 1 } else { -1 });
            right.push(if is_split { 2 } else { -1 });
            missing_left.push(rng.chance(50));
            value.push((rng.below(9) as f64) - 4.0);
        }
    }
    tree_ensemble::TreeEnsemble::new(
        &tree_ensemble::NodeRows {
            model_id: &model_id,
            tree_id: &tree_id,
            node_id: &node_id,
            feature: &feature,
            threshold: &threshold,
            left: &left,
            right: &right,
            missing_left: &missing_left,
            value: &value,
        },
        &tree_ensemble::ModelRows {
            model_id: &[0],
            base: &[(rng.below(5) as f64) - 2.0],
            agg: &[if rng.chance(50) { "sum" } else { "mean" }],
            link: &[if rng.chance(50) { "identity" } else { "sigmoid" }],
        },
        n_features,
    )
    .expect("the generator only builds valid ensembles")
}

fn gen_input(rng: &mut gen::Rng, p: &Program) -> Batch {
    let rows = rng.below(5) as usize;
    let cols = p
        .in_cols
        .iter()
        .map(|c| {
            let mk_valid = |rng: &mut gen::Rng| !c.ty.nullable || rng.chance(70);
            match c.ty.ty.lane() {
                // A decimal ROW column is opaque, so the generator never
                // produces one (see schema.rs, Policy::Row).
                Ty::Dec(..) => unreachable!("a decimal row column is opaque"),
                Ty::I1 => c_i1(
                    &(0..rows)
                        .map(|_| mk_valid(rng).then(|| rng.chance(50)))
                        .collect::<Vec<_>>(),
                ),
                Ty::I64 => c_i64(
                    &(0..rows)
                        .map(|_| mk_valid(rng).then(|| rng.next() as i64 % 100_000))
                        .collect::<Vec<_>>(),
                ),
                Ty::F64 => c_f64(
                    &(0..rows)
                        .map(|_| {
                            mk_valid(rng).then(|| match rng.below(5) {
                                0 => f64::NAN,
                                1 => f64::INFINITY,
                                _ => (rng.next() as i64 % 1000) as f64 / 8.0,
                            })
                        })
                        .collect::<Vec<_>>(),
                ),
                Ty::Str => {
                    let opts: Vec<Option<String>> = (0..rows)
                        .map(|_| {
                            mk_valid(rng).then(|| match rng.below(4) {
                                0 => String::new(),
                                1 => "k1".to_string(),
                                2 => "unicode é".to_string(),
                                _ => format!("v{}", rng.below(9)),
                            })
                        })
                        .collect();
                    let refs: Vec<Option<&str>> = opts.iter().map(|o| o.as_deref()).collect();
                    c_str(&refs)
                }
                Ty::I8 | Ty::I16 | Ty::I32 => {
                    unreachable!("lane() never returns a narrow width")
                }
            }
        })
        .collect();
    Batch { rows, cols }
}

/// Generated programs compile and execute without panics; when they run to
/// completion the output is rectangular, |out| <= |in|, and a second run is
/// byte-identical (determinism). Traps are a legal outcome (i64::MAX consts
/// meeting iadd, ftoi on inf, ...).
#[test]
fn fuzz_generated_programs_execute_deterministically() {
    for seed in 0..gen::fuzz_seeds(150) {
        let p = gen::gen_program(seed);
        let mut rng = gen::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let statics = gen_statics(&mut rng, &p);
        let f = match compile(&p, statics) {
            Ok(f) => f,
            Err(e) => panic!("seed {seed}: generated program failed to compile: {e}"),
        };
        let input = gen_input(&mut rng, &p);
        let first = run_snapshot(&f, &input);
        let second = run_snapshot(&f, &input);
        match (&first, &second) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "seed {seed}: nondeterministic output");
                assert!(a.len() <= input.rows, "seed {seed}: |out| > |in|");
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "seed {seed}: nondeterministic trap"),
            _ => panic!("seed {seed}: run flipped between Ok and Err"),
        }
    }
}

#[test]
fn casemap_tables_sorted_and_marquee_pins() {
    use super::casemap::{simple_lower, simple_upper, LOWER_EXCEPTIONS, UPPER_EXCEPTIONS};
    // binary_search precondition: strictly sorted by codepoint.
    for table in [UPPER_EXCEPTIONS, LOWER_EXCEPTIONS] {
        assert!(
            table.windows(2).all(|w| w[0].0 < w[1].0),
            "table not sorted"
        );
    }
    // The two exception classes (see scripts/gen_casemap.py): simple-vs-full
    // divergence, and Unicode version skew where duckdb's utf8proc predates
    // the case pair (identity there).
    assert_eq!(simple_upper('ß'), 'ẞ');
    assert_eq!(simple_lower('İ'), 'i');
    assert_eq!(simple_upper('ᾀ'), 'ᾈ');
    assert_eq!(simple_upper('ƛ'), 'ƛ');
    // The fallback path stays exact where full mapping is 1:1.
    assert_eq!(simple_upper('a'), 'A');
    assert_eq!(simple_lower('É'), 'é');
    assert_eq!(simple_upper('ﬁ'), 'ﬁ');
}

/// THE backend contract (TASK-44): cranelift and the interpreter agree
/// byte-for-byte on every generated program — outputs, emitted counts, and
/// traps. Same seeds as the determinism fuzz, both backends fed identical
/// statics and inputs.
#[test]
fn fuzz_cranelift_agrees_with_interpreter() {
    use super::cranelift;
    use super::super::ir::Inst;
    // Counting programs is not coverage — assert from inside the loop that
    // the opcodes we care about actually reached BOTH backends. The
    // cranelift fallback is silent, so an unimplemented opcode would
    // otherwise look green while only the interpreter ever ran it.
    let mut predicts = 0usize;
    // More seeds than the determinism fuzz: this is the backend contract.
    for seed in 0..gen::fuzz_seeds(500) {
        let p = gen::gen_program(seed);
        predicts += p
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, Inst::Predict { .. }))
            .count();
        let mut rng = gen::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let statics_i = gen_statics(&mut rng, &p);
        let mut rng2 = gen::Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let statics_c = gen_statics(&mut rng2, &p);
        let input = gen_input(&mut rng, &p);

        let fi = compile(&p, statics_i).expect("interp compile");
        let fc = match cranelift::compile(&p, statics_c) {
            Ok(f) => f,
            Err(e) => panic!("seed {seed}: cranelift failed to compile: {e}"),
        };
        let a = run_snapshot(&fi, &input);
        let mut st = fc.new_state();
        let b = fc.run(&input, &mut st).map(|_| snapshot(&st));
        match (a, b) {
            (Ok(x), Ok(y)) => assert_eq!(x, y, "seed {seed}: outputs diverge"),
            (Err(x), Err(y)) => assert_eq!(x, y, "seed {seed}: traps diverge"),
            (x, y) => panic!("seed {seed}: outcome diverged: interp {x:?} vs cranelift {y:?}"),
        }
    }
    assert!(
        predicts > 20,
        "the generator emitted only {predicts} predict instruction(s) — this test \
         is not covering the tree kernel on either backend"
    );
}

/// The `predict` semantics the design froze, run END TO END on BOTH
/// backends: a NULL feature is not a NULL result (the model has a defined
/// answer for missing), whereas an unseen group is.
#[test]
fn tree_score_fixture_pins_null_and_unseen_group_on_both_backends() {
    use super::cranelift;
    let p = built(fixtures::TREE_SCORE);

    // region -> model id ('north' = 0, 'south' = 1); 'west' is absent, so it
    // is the probe-miss row. compile() consumes the statics, so both are
    // rebuilt per backend.
    let statics = || {
        let map = StaticData::Map(vec![
            (vec![KeyBits::Str("north".into())], vec![ScalarVal::I64(0)]),
            (vec![KeyBits::Str("south".into())], vec![ScalarVal::I64(1)]),
        ]);
        // Both models split `price <= 100` -> 1.0 else 2.0. They differ only
        // in where a MISSING price goes: model 0 right (the untrained
        // default), model 1 left (a tree that learned a missing branch).
        let model = tree_ensemble::TreeEnsemble::new(
            &tree_ensemble::NodeRows {
                model_id: &[0, 0, 0, 1, 1, 1],
                tree_id: &[0, 0, 0, 0, 0, 0],
                node_id: &[0, 1, 2, 0, 1, 2],
                feature: &[0, -1, -1, 0, -1, -1],
                threshold: &[100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
                left: &[1, -1, -1, 1, -1, -1],
                right: &[2, -1, -1, 2, -1, -1],
                missing_left: &[false, false, false, true, false, false],
                value: &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            },
            &tree_ensemble::ModelRows {
                model_id: &[0, 1],
                base: &[0.0, 0.0],
                agg: &["sum", "sum"],
                link: &["identity", "identity"],
            },
            2,
        )
        .expect("fixture ensemble builds");
        vec![map, StaticData::Model(Box::new(model))]
    };

    let input = batch(
        4,
        vec![
            c_str(&[
                Some("north"),
                Some("north"),
                Some("south"),
                Some("west"),
            ]),
            c_f64(&[Some(50.0), None, None, Some(50.0)]),
            // `rooms` is an INTEGER feature: it reaches predict through
            // itof.f32, one rounding, matching sklearn (TASK-77).
            c_i64(&[Some(1), Some(1), Some(1), Some(1)]),
        ],
    );
    let want = rows(&[
        // present feature, below the threshold -> left leaf
        &["1.0"],
        // NULL price -> NaN -> model 0 sends missing RIGHT
        &["2.0"],
        // same NULL, but model 1 learned missing goes LEFT
        &["1.0"],
        // unseen region: a missing MODEL is NULL, unlike a missing feature
        &["NULL"],
    ]);

    let fi = compile(&p, statics()).expect("interp compile");
    assert_eq!(run_snapshot(&fi, &input).unwrap(), want, "interpreter");

    // Directly, NOT through the fallback-guarded path: a missing cranelift
    // binding must fail here rather than silently drop to the interpreter.
    let fc = cranelift::compile(&p, statics()).expect("cranelift compile");
    let mut st = fc.new_state();
    fc.run(&input, &mut st).expect("cranelift run");
    assert_eq!(snapshot(&st), want, "cranelift");
}

/// TASK-91: the Dec lane on BOTH backends, end to end -- probe a
/// decimal128(38,0) and a decimal128(6,2) value column, select between the
/// probed payload and a zero default on the miss (the LEFT-join shape),
/// compare at one scale, convert down with DuckDB's div/mod algorithm, and
/// emit. 2^53+1 must survive all of it as ITSELF, which is the whole
/// ticket; the interpreter and the JIT must agree, which is the backend
/// contract.
#[test]
fn interp_and_cranelift_agree_on_a_decimal_probe_select_and_emit() {
    use super::cranelift;
    let p = parse(
        r#"static @0: map(str) -> (dec(38,0), dec(6,2))

fn f(in: batch{g: str}, out: batch{sk: dec(38,0)?, d: dec(6,2)?, big: i1?, x: f64?}) {
entry:
  %g = load in.g
  %hit, %sk, %d = probe @0, %g
  %z38 = const.dec(38,0) 0
  %z6 = const.dec(6,2) 0
  %skv = select %hit, %sk, %z38
  %dv = select %hit, %d, %z6
  %lim = const.dec(38,0) 9007199254740992
  %big = dcmp(38,0).gt %skv, %lim
  %x = dtof(6,2) %dv
  store.opt out.sk, %hit, %skv
  store.opt out.d, %hit, %dv
  store.opt out.big, %hit, %big
  store.opt out.x, %hit, %x
  emit
}"#,
    )
    .unwrap();
    let statics = || {
        vec![StaticData::Map(vec![
            (
                vec![KeyBits::Str("a".into())],
                vec![
                    ScalarVal::Dec(9_007_199_254_740_993, 38, 0),
                    ScalarVal::Dec(-1234, 6, 2),
                ],
            ),
            (
                vec![KeyBits::Str("b".into())],
                vec![
                    ScalarVal::Dec(-9_007_199_254_740_993, 38, 0),
                    ScalarVal::Dec(50, 6, 2),
                ],
            ),
        ])]
    };
    let input = batch(3, vec![c_str(&[Some("a"), Some("b"), Some("zz")])]);
    // The snapshot renders a Dec lane as its SCALED integer; the decimal
    // point is the arrow boundary's business, not the lane's.
    let want = rows(&[
        &["9007199254740993", "-1234", "true", "-12.34"],
        &["-9007199254740993", "50", "false", "0.5"],
        &["NULL", "NULL", "NULL", "NULL"],
    ]);

    let fi = compile(&p, statics()).expect("interp compile");
    assert_eq!(run_snapshot(&fi, &input).unwrap(), want, "interpreter");

    let fc = cranelift::compile(&p, statics()).expect("cranelift compile");
    let mut st = fc.new_state();
    fc.run(&input, &mut st).expect("cranelift run");
    assert_eq!(snapshot(&st), want, "cranelift");
}

/// Raw compute, no Python boundary: `cargo test --release backend_compute_
/// bench -- --ignored --nocapture`. Informational, not a gate assertion.
#[test]
#[ignore = "bench, run explicitly"]
fn backend_compute_bench() {
    use super::cranelift;
    use std::time::Instant;
    let p = parse(
        r#"fn f(in: batch{a: i64, b: f64}, out: batch{x: i64, h: f64, k: i1}) {
entry:
  %a = load in.a
  %b = load in.b
  %two = const.i64 2
  %one = const.i64 1
  %m = imul %a, %two
  %x = iadd %m, %one
  %hf = const.f64 0.5
  %h = fmul %b, %hf
  %z = const.f64 10.0
  %k = fcmp.gt %h, %z
  store out.x, %x
  store out.h, %h
  store out.k, %k
  emit
}"#,
    )
    .unwrap();
    let n = 100_000usize;
    let a: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64 % 1000)).collect();
    let b: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64 / 7.0)).collect();
    let input = batch(n, vec![c_i64(&a), c_f64(&b)]);

    let fi = compile(&p, vec![]).unwrap();
    let fc = cranelift::compile(&p, vec![]).unwrap();
    let mut sti = fi.new_state();
    let mut stc = fc.new_state();
    for (name, run) in [
        (
            "interp",
            &mut (|| fi.run(&input, &mut sti).unwrap()) as &mut dyn FnMut(),
        ),
        ("cranelift", &mut (|| fc.run(&input, &mut stc).unwrap())),
    ] {
        run(); // warm
        let mut best = u128::MAX;
        for _ in 0..20 {
            let t = Instant::now();
            run();
            best = best.min(t.elapsed().as_nanos());
        }
        println!(
            "{name:>10}: {:>6.2} ns/row (best of 20, n={n})",
            best as f64 / n as f64
        );
    }
}

#[test]
fn multimap_expand_fixture_executes() {
    // Stage-B machinery end to end: dup keys fan out one output row per
    // match (probe order outer, INSERTION order inner — the stable sort
    // keeps equal keys in materialization order), zero matches skip, and
    // the cyclic CFG verifies + roundtrips through the text format.
    let p = built(fixtures::MULTI_EXPAND);
    let printed = super::super::ir::print::print(&p);
    assert_eq!(parse(&printed).unwrap(), p, "print/parse roundtrip");

    // Entries deliberately unsorted; key 1's values inserted 10 then 11.
    let data = StaticData::Map(vec![
        (vec![KeyBits::I64(2)], vec![ScalarVal::I64(20)]),
        (vec![KeyBits::I64(1)], vec![ScalarVal::I64(10)]),
        (vec![KeyBits::I64(1)], vec![ScalarVal::I64(11)]),
    ]);
    let f = compile(&p, vec![data]).unwrap();
    let got = run_snapshot(&f, &batch(3, vec![c_i64(&[Some(1), Some(2), Some(3)])])).unwrap();
    assert_eq!(
        got,
        rows(&[&["1", "10"], &["1", "11"], &["2", "20"]]),
        "id=1 fans out to both matches in insertion order; id=3 skips"
    );
}

#[test]
fn keyless_multimap_is_a_cross_join() {
    // Zero-key probe.range covers the whole table — the cross/inequality
    // join primitive.
    let text = r#"
static @0: multimap() -> (i64)

fn cross(in: batch{id: i64}, out: batch{id: i64, v: i64}) {
entry:
  %id = load in.id
  %lo, %hi = probe.range @0
  jump head(%lo, %hi, %id)
head(%i: i64, %end: i64, %rid: i64):
  %more = icmp.lt %i, %end
  brif %more, body(%i, %end, %rid), done
body(%j: i64, %e: i64, %rid2: i64):
  %v = probe.read @0, %j
  store out.id, %rid2
  store out.v, %v
  %one = const.i64 1
  %j2 = iadd %j, %one
  emit.to head(%j2, %e, %rid2)
done:
  skip
}
"#;
    let p = built(text);
    let data = StaticData::Map(vec![
        (vec![], vec![ScalarVal::I64(7)]),
        (vec![], vec![ScalarVal::I64(8)]),
    ]);
    let f = compile(&p, vec![data]).unwrap();
    let got = run_snapshot(&f, &batch(2, vec![c_i64(&[Some(1), Some(2)])])).unwrap();
    assert_eq!(
        got,
        rows(&[&["1", "7"], &["1", "8"], &["2", "7"], &["2", "8"]])
    );
}

// -------------------------------------------------------------- externs --
// DRAFT-22 step 2: ecall executes the supplied ExternImpl; both backends
// route through the same call_extern, so results and traps are identical.

/// A width-1 scaler-shaped extern: nullable i64 id + f64 feature -> f64.
const EXTERN_SCALER: &str = r#"extern @0: "tf" (i64, f64) -> (f64)

fn f(in: batch{id: i64?, x: f64?}, out: batch{z: f64?}) {
b0:
  %idf, %idv = load.opt in.id
  %xf, %xv = load.opt in.x
  %w, %zf, %zv = ecall @0, %idf, %idv, %xf, %xv
  store.opt out.z, %zf, %zv
  emit
}"#;

fn imp(
    name: &str,
    f: impl Fn(&[Option<ScalarVal>]) -> Result<Option<Vec<Option<ScalarVal>>>, String> + 'static,
) -> super::ExternImpl {
    super::ExternImpl {
        name: name.into(),
        fun: Box::new(f),
    }
}

/// (id, x) -> x * (id + 1); any NULL arg -> whole-call NULL (the
/// PythonTransform NULL-id convention lives in the callable, not here).
fn scaler_impl() -> super::ExternImpl {
    imp("tf", |args| {
        let (Some(ScalarVal::I64(id)), Some(ScalarVal::F64(x))) = (&args[0], &args[1]) else {
            return Ok(None);
        };
        Ok(Some(vec![Some(ScalarVal::F64(x * (*id as f64 + 1.0)))]))
    })
}

fn scaler_input() -> Batch {
    batch(
        3,
        vec![
            c_i64(&[Some(1), None, Some(2)]),
            c_f64(&[Some(2.0), Some(3.0), None]),
        ],
    )
}

#[test]
fn extern_call_interp_executes() {
    let p = built(EXTERN_SCALER);
    let f = super::interp::compile_ext(&p, vec![], vec![scaler_impl()]).unwrap();
    assert_eq!(
        run_snapshot(&f, &scaler_input()).unwrap(),
        rows(&[&["4.0"], &["NULL"], &["NULL"]])
    );
}

#[test]
fn extern_call_traps_are_named() {
    let p = built(EXTERN_SCALER);
    // Raised error -> trap with the callable's message.
    let f =
        super::interp::compile_ext(&p, vec![], vec![imp("tf", |_| Err("boom".into()))]).unwrap();
    let err = run_snapshot(&f, &scaler_input()).unwrap_err();
    assert!(err.0.contains("boom"), "got: {err:?}");
    // Wrong arity -> named trap.
    let f = super::interp::compile_ext(
        &p,
        vec![],
        vec![imp("tf", |_| {
            Ok(Some(vec![
                Some(ScalarVal::F64(1.0)),
                Some(ScalarVal::F64(2.0)),
            ]))
        })],
    )
    .unwrap();
    let err = run_snapshot(&f, &scaler_input()).unwrap_err();
    assert!(
        err.0.contains("returned 2 value(s), declared 1"),
        "got: {err:?}"
    );
    // Wrong type -> named trap.
    let f = super::interp::compile_ext(
        &p,
        vec![],
        vec![imp("tf", |_| Ok(Some(vec![Some(ScalarVal::Str("x".into()))])))],
    )
    .unwrap();
    let err = run_snapshot(&f, &scaler_input()).unwrap_err();
    assert!(err.0.contains("is str, declared f64"), "got: {err:?}");
}

#[test]
fn extern_compile_validates_impl_list() {
    let p = built(EXTERN_SCALER);
    let Err(err) = super::interp::compile_ext(&p, vec![], vec![]) else {
        panic!("compiled with a missing extern impl");
    };
    assert!(
        err.to_string().contains("1 extern(s), 0 implementation(s)"),
        "got: {err}"
    );
    let Err(err) = super::interp::compile_ext(&p, vec![], vec![imp("other", |_| Ok(None))]) else {
        panic!("compiled with a misnamed extern impl");
    };
    assert!(
        err.to_string()
            .contains("declared 'tf', implementation is named 'other'"),
        "got: {err}"
    );
}

#[test]
fn extern_width_two_with_str_and_component_nulls() {
    // Width-2 (str, f64): distinguishes the whole-NULL call from a result
    // with NULL components, and exercises arena strings both directions.
    let text = r#"extern @0: "pair" (str) -> (str, f64)

fn f(in: batch{s: str?}, out: batch{w: i1, a: str?, b: f64?}) {
b0:
  %sf, %sv = load.opt in.s
  %w, %af, %av, %bf, %bv = ecall @0, %sf, %sv
  store out.w, %w
  store.opt out.a, %af, %av
  store.opt out.b, %bf, %bv
  emit
}"#;
    let p = built(text);
    let pair = || {
        imp("pair", |args| match &args[0] {
            None => Ok(None),
            Some(ScalarVal::Str(s)) if s == "half" => Ok(Some(vec![None, Some(ScalarVal::F64(0.5))])),
            Some(ScalarVal::Str(s)) => Ok(Some(vec![
                Some(ScalarVal::Str(format!("{s}!"))),
                Some(ScalarVal::F64(s.len() as f64)),
            ])),
            _ => Err("bad arg".into()),
        })
    };
    let input = || batch(3, vec![c_str(&[Some("ab"), Some("half"), None])]);
    let f = super::interp::compile_ext(&p, vec![], vec![pair()]).unwrap();
    let want = rows(&[
        &["true", "ab!", "2.0"],
        &["true", "NULL", "0.5"],
        &["false", "NULL", "NULL"],
    ]);
    assert_eq!(run_snapshot(&f, &input()).unwrap(), want);

    // Cranelift: same program, same impls, byte-identical output.
    let cf = super::cranelift::compile_ext(&p, vec![], vec![pair()]).unwrap();
    let mut cst = cf.new_state();
    cf.run(&input(), &mut cst).unwrap();
    assert_eq!(snapshot(&cst), want);
}

#[test]
fn extern_call_cranelift_agrees_with_interp() {
    let p = built(EXTERN_SCALER);
    let cf = super::cranelift::compile_ext(&p, vec![], vec![scaler_impl()]).unwrap();
    let mut cst = cf.new_state();
    cf.run(&scaler_input(), &mut cst).unwrap();
    assert_eq!(snapshot(&cst), rows(&[&["4.0"], &["NULL"], &["NULL"]]));
    // Traps agree too.
    let cf =
        super::cranelift::compile_ext(&p, vec![], vec![imp("tf", |_| Err("boom".into()))]).unwrap();
    let mut cst = cf.new_state();
    let err = cf.run(&scaler_input(), &mut cst).unwrap_err();
    assert!(err.0.contains("boom"), "got: {err:?}");
}
