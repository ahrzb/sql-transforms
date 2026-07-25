//! Interpreter-backend tests: every M-ir fixture executes against
//! hand-computed expectations; unverified IR and mismatched statics are
//! rejected; the steady state performs zero heap allocations (counting
//! global allocator); generated programs execute deterministically.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use super::super::ir::{fixtures, gen, parse::parse, verify::verify, Program, StaticTy, Ty};
use super::interp::{compile, CompileError, InterpFn};
use super::{Batch, ColData, KeyBits, OutCol, RunState, ScalarVal, StaticData, Trap};

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

fn c_i1(vals: &[Option<bool>]) -> ColData {
    ColData::I1 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(false)).collect(),
    }
}

fn c_i64(vals: &[Option<i64>]) -> ColData {
    ColData::I64 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(0)).collect(),
    }
}

fn c_f64(vals: &[Option<f64>]) -> ColData {
    ColData::F64 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(0.0)).collect(),
    }
}

fn c_str(vals: &[Option<&str>]) -> ColData {
    ColData::Str {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or("").to_string()).collect(),
    }
}

fn batch(rows: usize, cols: Vec<ColData>) -> Batch {
    Batch { rows, cols }
}

fn built(text: &str) -> Program {
    let p = parse(text).expect("fixture parses");
    verify(&p).expect("fixture verifies");
    p
}

/// Snapshot the output as strings, masking NULL payloads (a NULL's payload
/// is meaningless downstream by contract). Allocates — test side only.
fn snapshot(st: &RunState) -> Vec<Vec<String>> {
    let ncols = st.out.len();
    let nrows = st.out.first().map(|c| c.len()).unwrap_or(0);
    (0..nrows)
        .map(|r| {
            (0..ncols)
                .map(|c| match &st.out[c] {
                    OutCol::I1(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::I64(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::F64(v) => render(v[r].0, format!("{:?}", v[r].1)),
                    OutCol::Str(v) => render(v[r].0, st.arena.get(v[r].1).to_string()),
                })
                .collect()
        })
        .collect()
}

fn render(valid: bool, s: String) -> String {
    if valid {
        s
    } else {
        "NULL".to_string()
    }
}

fn run_snapshot(f: &InterpFn, input: &Batch) -> Result<Vec<Vec<String>>, Trap> {
    let mut st = f.new_state();
    f.run(input, &mut st)?;
    Ok(snapshot(&st))
}

fn rows(v: &[&[&str]]) -> Vec<Vec<String>> {
    v.iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}

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
        rows(&[
            &["old", "false"],
            &["old", "false"],
            &["young", "false"],
        ])
    );
}

#[test]
fn casts_fixture_executes_and_traps() {
    let p = built(fixtures::CASTS);
    let statics = |snd: StaticData| {
        vec![
            StaticData::Scalar { valid: true, val: ScalarVal::F64(2.5) },
            snd,
        ]
    };
    // @1 NULL: n = round(2.5) - trunc(2.5) = 3 - 2 = 1; msg = select over
    // scmp.eq("2.5:", ":") = false -> ":".
    let f = compile(
        &p,
        statics(StaticData::Scalar { valid: false, val: ScalarVal::I64(0) }),
    )
    .unwrap();
    let input = batch(1, vec![c_str(&[Some("12")])]);
    assert_eq!(run_snapshot(&f, &input).unwrap(), rows(&[&["1", ":"]]));

    // @1 = 7: the select picks the static instead.
    let f7 = compile(
        &p,
        statics(StaticData::Scalar { valid: true, val: ScalarVal::I64(7) }),
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
    assert!(verify(&p).is_err(), "precondition: program must be unverifiable");
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
            vec![StaticData::Scalar { valid: true, val: ScalarVal::F64(1.0) }],
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
    assert!(f.run(&wrong_count, &mut st).unwrap_err().0.contains("0 column(s)"));
    let wrong_len = Batch { rows: 2, cols: vec![c_f64(&[Some(1.0)])] };
    assert!(f.run(&wrong_len, &mut st).unwrap_err().0.contains("1 row(s)"));
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
                let n = if keys[0] == Ty::I1 { 2 } else { 1 + rng.below(3) as usize };
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
