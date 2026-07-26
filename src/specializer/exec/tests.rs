//! Interpreter-backend tests: every M-ir fixture executes against
//! hand-computed expectations; unverified IR and mismatched statics are
//! rejected; the steady state performs zero heap allocations (counting
//! global allocator); generated programs execute deterministically.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use super::super::ir::{fixtures, gen, parse::parse, verify::verify, Program, StaticTy, Ty};
use super::interp::{compile, CompileError};
use super::testutil::{batch, built, c_f64, c_i1, c_i64, c_str, rows, run_snapshot, snapshot};
use super::{Batch, ColData, KeyBits, OutCol, ScalarVal, StaticData, Trap};

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
                val: ScalarVal::F64(2.5),
            },
            snd,
        ]
    };
    // @1 NULL: n = round(2.5) - trunc(2.5) = 3 - 2 = 1; msg = select over
    // scmp.eq("2.5:", ":") = false -> ":".
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
        Err(CompileError::Static(m)) => panic!("wrong error kind: {m}"),
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
            Err(CompileError::Verify(_)) => panic!("fixture failed verify?"),
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
    for (expr, needle) in [
        (
            "  %a = const.i64 9223372036854775807\n  %b = const.i64 1\n  %r = iadd %a, %b",
            "overflow in iadd",
        ),
        (
            "  %a = const.i64 -9223372036854775808\n  %b = const.i64 1\n  %r = isub %a, %b",
            "overflow in isub",
        ),
        (
            "  %a = const.i64 4611686018427387904\n  %b = const.i64 4\n  %r = imul %a, %b",
            "overflow in imul",
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
            "overflow in idiv",
        ),
        (
            "  %a = const.i64 -9223372036854775808\n  %b = const.i64 -1\n  %r = irem %a, %b",
            "overflow in irem",
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
    assert!(err.0.contains("overflow"), "got '{}'", err.0);
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
    for (lit, mode, expect) in [
        ("2.5", "round", "3"),
        ("-2.5", "round", "-3"),
        ("0.5", "round", "1"),
        ("-0.5", "round", "-1"),
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
        Ty::I64 => ScalarVal::I64(rng.next() as i64 % 1000),
        Ty::F64 => ScalarVal::F64((rng.next() as i64 % 1000) as f64 / 4.0),
        Ty::Str => ScalarVal::Str(format!("s{}", rng.below(5))),
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
                            .map(|(pos, kt)| match (pos, kt) {
                                (0, Ty::I1) => KeyBits::I1(j % 2 == 1),
                                (0, Ty::I64) => KeyBits::I64(j as i64),
                                (0, Ty::F64) => KeyBits::F64((j as f64).to_bits()),
                                (0, Ty::Str) => KeyBits::Str(format!("k{j}")),
                                (_, Ty::I1) => KeyBits::I1(true),
                                (_, Ty::I64) => KeyBits::I64(7),
                                (_, Ty::F64) => KeyBits::F64(1.5f64.to_bits()),
                                (_, Ty::Str) => KeyBits::Str("fix".into()),
                            })
                            .collect();
                        let vals = values.iter().map(|vt| gen_scalar(rng, *vt)).collect();
                        (key, vals)
                    })
                    .collect();
                StaticData::Map(entries)
            }
        })
        .collect()
}

fn gen_input(rng: &mut gen::Rng, p: &Program) -> Batch {
    let rows = rng.below(5) as usize;
    let cols = p
        .in_cols
        .iter()
        .map(|c| {
            let mk_valid = |rng: &mut gen::Rng| !c.ty.nullable || rng.chance(70);
            match c.ty.ty {
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
    for seed in 0..150u64 {
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
    // More seeds than the determinism fuzz: this is the backend contract.
    for seed in 0..500u64 {
        let p = gen::gen_program(seed);
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
