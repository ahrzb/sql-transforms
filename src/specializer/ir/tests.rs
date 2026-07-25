//! IR boundary tests: fixtures round-trip and verify; every verifier rule
//! has a program that trips it; every parser guard has a text that trips it;
//! seeded fuzz pins `parse(print(p)) == p` across the generator's whole
//! surface.

use super::{fixtures, gen, parse::parse, print::print, verify::verify, Program};

fn parsed(text: &str) -> Program {
    match parse(text) {
        Ok(p) => p,
        Err(e) => panic!("parse failed: {e}\n---\n{text}"),
    }
}

fn verified(text: &str) -> Program {
    let p = parsed(text);
    if let Err(errs) = verify(&p) {
        let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        panic!("verify failed: {}\n---\n{text}", msgs.join("; "));
    }
    p
}

// ------------------------------------------------------------- fixtures --

#[test]
fn fixtures_verify_and_round_trip() {
    for (name, text) in fixtures::all() {
        let p = verified(text);
        let printed = print(&p);
        let p2 = parsed(&printed);
        assert_eq!(p2, p, "round-trip changed '{name}':\n{printed}");
        assert_eq!(print(&p2), printed, "printing is not a fixpoint for '{name}'");
        verify(&p2).unwrap_or_else(|_| panic!("canonical form of '{name}' fails verify"));
    }
}

/// Every opcode and terminator must appear in at least one fixture — this is
/// the acceptance criterion "hand-written programs covering every
/// instruction", kept honest mechanically.
#[test]
fn fixtures_cover_every_opcode() {
    let all: String = fixtures::all().iter().map(|(_, t)| *t).collect();
    let opcodes = [
        "const.i1", "const.i64", "const.f64", "const.str", "iadd", "isub", "imul", "idiv",
        "irem", "fadd", "fsub", "fmul", "fdiv", "and", "or", "xor", "not", "icmp.", "fcmp.",
        "scmp.", "select", "itof", "ftoi.trunc", "ftoi.round", "itos", "ftos", "stoi.opt",
        "stof.opt", "sconcat", "load in.", "load.opt in.", "store out.", "store.opt out.",
        "probe @", "sload @", "sload.opt @", "jump", "brif", "emit", "skip", "trap",
    ];
    let missing: Vec<&str> = opcodes.iter().filter(|op| !all.contains(**op)).copied().collect();
    assert!(missing.is_empty(), "no fixture covers: {missing:?}");
}

// ------------------------------------------------------ verifier rejects --

/// Parse must succeed and verify must fail with a message containing `needle`.
fn assert_verify_rejects(text: &str, needle: &str) {
    let p = parsed(text);
    match verify(&p) {
        Ok(()) => panic!("verifier accepted a bad program (wanted '{needle}'):\n{text}"),
        Err(errs) => {
            let all: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            assert!(
                all.iter().any(|m| m.contains(needle)),
                "expected an error containing '{needle}', got: {all:?}\n---\n{text}"
            );
        }
    }
}

#[test]
fn rejects_cross_block_use() {
    // %x is defined in entry and used in b1 without riding a branch arg.
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  %t = const.i1 true
  brif %t, one, two
one:
  store out.o, %x
  emit
two:
  skip
}"#,
        "cross blocks only as branch args",
    );
}

#[test]
fn rejects_use_before_def() {
    // Text form: the parser's name table already refuses a forward %use...
    assert_parse_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %y = iadd %x, %x
  %x = load in.a
  store out.o, %y
  emit
}"#,
        "undefined value '%x'",
    );
    // ...and the verifier independently rejects the same shape when the
    // program is constructed through the API (the path lowering will use).
    use super::{Block, Col, ColTy, Inst, Program, Term, Ty, Value, BinOp};
    let p = Program {
        statics: vec![],
        name: "f".into(),
        in_cols: vec![Col { name: "a".into(), ty: ColTy { ty: Ty::I64, nullable: false } }],
        out_cols: vec![Col { name: "o".into(), ty: ColTy { ty: Ty::I64, nullable: false } }],
        blocks: vec![Block {
            params: vec![],
            insts: vec![
                Inst::Bin { op: BinOp::Iadd, dst: Value(0), a: Value(1), b: Value(1) },
                Inst::Load { dst: Value(1), col: 0 },
                Inst::Store { col: 0, val: Value(0) },
            ],
            term: Term::Emit,
        }],
    };
    let errs = verify(&p).expect_err("use-before-def must not verify");
    assert!(
        errs.iter().any(|e| e.to_string().contains("used before any definition")),
        "wrong errors: {:?}",
        errs.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn rejects_arith_type_mismatch() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: f64}, out: batch{o: f64}) {
entry:
  %x = load in.a
  %y = iadd %x, %x
  %z = itof %y
  store out.o, %z
  emit
}"#,
        "must be i64, got f64",
    );
}

#[test]
fn rejects_skipping_the_null_lane() {
    // A nullable column read with the non-opt load: the arithmetic below it
    // would silently operate on a maybe-NULL — exactly the 3VL bug class the
    // IR is designed to make unrepresentable.
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64?}, out: batch{o: i64}) {
entry:
  %x = load in.a
  store out.o, %x
  emit
}"#,
        "is nullable: use load.opt",
    );
}

#[test]
fn rejects_inventing_a_null_lane() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %f, %x = load.opt in.a
  store out.o, %x
  emit
}"#,
        "is not nullable: use load",
    );
}

#[test]
fn rejects_store_without_flag_on_nullable_out() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64?}) {
entry:
  %x = load in.a
  store out.o, %x
  emit
}"#,
        "is nullable: use store.opt",
    );
}

#[test]
fn rejects_probe_on_scalar_static() {
    assert_verify_rejects(
        r#"static @0: scalar<f64>
fn f(in: batch{k: str}, out: batch{o: f64}) {
entry:
  %k = load in.k
  %h, %v = probe @0, %k
  store out.o, %v
  emit
}"#,
        "is a scalar: use sload",
    );
}

#[test]
fn rejects_probe_key_arity_and_type() {
    assert_verify_rejects(
        r#"static @0: map(i64, str) -> (f64)
fn f(in: batch{k: str}, out: batch{o: f64}) {
entry:
  %k = load in.k
  %h, %v = probe @0, %k
  store out.o, %v
  emit
}"#,
        "has 2 key(s), probe passes 1",
    );
    assert_verify_rejects(
        r#"static @0: map(i64) -> (f64)
fn f(in: batch{k: str}, out: batch{o: f64}) {
entry:
  %k = load in.k
  %h, %v = probe @0, %k
  store out.o, %v
  emit
}"#,
        "probe key %v0 must be i64, got str",
    );
}

#[test]
fn rejects_probe_value_arity() {
    assert_verify_rejects(
        r#"static @0: map(str) -> (f64, i64)
fn f(in: batch{k: str}, out: batch{o: f64}) {
entry:
  %k = load in.k
  %h, %v = probe @0, %k
  store out.o, %v
  emit
}"#,
        "has 2 value column(s), probe defines 1",
    );
}

#[test]
fn rejects_missing_store_at_emit() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64, p: i64}) {
entry:
  %x = load in.a
  store out.o, %x
  emit
}"#,
        "emit without storing out.p",
    );
}

#[test]
fn rejects_double_store() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  store out.o, %x
  store out.o, %x
  emit
}"#,
        "stored more than once",
    );
}

#[test]
fn rejects_store_before_skip() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  store out.o, %x
  skip
}"#,
        "skip after storing",
    );
}

#[test]
fn rejects_join_with_disagreeing_stores() {
    // One arm stores out.o before the join, the other doesn't.
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %t = const.i1 true
  brif %t, one, two
one:
  %x = const.i64 1
  store out.o, %x
  jump join
two:
  jump join
join:
  emit
}"#,
        "disagree on which out columns are stored",
    );
}

#[test]
fn rejects_cycle() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %t = const.i1 true
  brif %t, one, two
one:
  jump two
two:
  jump one
}"#,
        "control-flow cycle",
    );
}

#[test]
fn rejects_unreachable_block() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  store out.o, %x
  emit
island:
  skip
}"#,
        "unreachable block",
    );
}

#[test]
fn rejects_branch_arg_mismatch() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  jump join(%x)
join(%p: f64):
  %t = ftoi.trunc %p
  store out.o, %t
  emit
}"#,
        "branch arg %v0 must be f64, got i64",
    );
}

#[test]
fn rejects_select_arm_mismatch() {
    assert_verify_rejects(
        r#"fn f(in: batch{a: i64, b: f64}, out: batch{o: i64}) {
entry:
  %x = load in.a
  %y = load in.b
  %t = const.i1 true
  %s = select %t, %x, %y
  store out.o, %s
  emit
}"#,
        "select arms differ",
    );
}

// -------------------------------------------------------- parser rejects --

fn assert_parse_rejects(text: &str, needle: &str) {
    match parse(text) {
        Ok(_) => panic!("parser accepted bad text (wanted '{needle}'):\n{text}"),
        Err(e) => assert!(
            e.to_string().contains(needle),
            "expected a parse error containing '{needle}', got '{e}'"
        ),
    }
}

#[test]
fn parser_rejects_unknown_opcode() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  %x = alloc 8\n  emit\n}",
        "unknown opcode 'alloc'",
    );
}

#[test]
fn parser_rejects_value_redefinition() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  %x = load in.a\n  %x = load in.a\n  emit\n}",
        "defined twice",
    );
}

#[test]
fn parser_rejects_unknown_value() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  store out.o, %ghost\n  emit\n}",
        "undefined value '%ghost'",
    );
}

#[test]
fn parser_rejects_unknown_column() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  %x = load in.b\n  emit\n}",
        "unknown column 'in.b'",
    );
}

#[test]
fn parser_rejects_unknown_label() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  jump nowhere\n}",
        "unknown block 'nowhere'",
    );
}

#[test]
fn parser_rejects_unknown_static() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  %x = sload @0\n  emit\n}",
        "unknown static '@0'",
    );
}

#[test]
fn parser_rejects_sparse_static_ids() {
    assert_parse_rejects(
        "static @1: scalar<f64>\nfn f(in: batch{a: i64}, out: batch{o: i64}) {\nentry:\n  emit\n}",
        "dense and in order",
    );
}

#[test]
fn parser_rejects_bad_escape_and_unterminated_string() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{o: str}) {\nentry:\n  %x = const.str \"\\q\"\n  emit\n}",
        "unknown escape",
    );
    assert_parse_rejects("fn f(in: batch{a: i64}, out: batch{o: str}) { \"", "unterminated");
}

#[test]
fn parser_rejects_trailing_garbage() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64}, out: batch{}) {\nentry:\n  emit\n}\nextra",
        "trailing input",
    );
}

#[test]
fn parser_rejects_wrong_dst_count() {
    assert_parse_rejects(
        "fn f(in: batch{a: i64?}, out: batch{o: i64}) {\nentry:\n  %x = load.opt in.a\n  emit\n}",
        "'load.opt' defines 2 value(s), found 1",
    );
}

// ------------------------------------------------------------- literals --

#[test]
fn literal_edge_cases_round_trip() {
    let text = r#"fn lits(in: batch{a: i64}, out: batch{o: str}) {
entry:
  %nan = const.f64 nan
  %pinf = const.f64 inf
  %ninf = const.f64 -inf
  %nzero = const.f64 -0.0
  %tiny = const.f64 1e-5
  %big = const.f64 1e300
  %min = const.i64 -9223372036854775808
  %max = const.i64 9223372036854775807
  %esc = const.str "q\"b\\n\nl\tt\u{7}"
  store out.o, %esc
  emit
}"#;
    let p = verified(text);
    let printed = print(&p);
    let p2 = parsed(&printed);
    assert_eq!(p2, p, "literal round-trip changed the program:\n{printed}");
    // -0.0 must survive as -0.0 (bitwise equality, not IEEE ==).
    let has_neg_zero = printed.contains("-0.0");
    assert!(has_neg_zero, "printer lost the sign of -0.0:\n{printed}");
}

// ----------------------------------------------------------------- fuzz --

#[test]
fn fuzz_round_trip() {
    for seed in 0..300u64 {
        let p = gen::gen_program(seed);
        if let Err(errs) = verify(&p) {
            let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            panic!("generator produced an invalid program (seed {seed}): {msgs:?}\n{}", print(&p));
        }
        let text = print(&p);
        let p2 = match parse(&text) {
            Ok(p2) => p2,
            Err(e) => panic!("canonical text failed to parse (seed {seed}): {e}\n{text}"),
        };
        assert_eq!(p2, p, "round-trip changed the program (seed {seed}):\n{text}");
        assert_eq!(print(&p2), text, "printing is not a fixpoint (seed {seed})");
    }
}
