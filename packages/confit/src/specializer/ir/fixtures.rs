//! Hand-written IR programs, one habitat per instruction. Written in the
//! human-friendly form (named values, named labels) that the parser accepts
//! and the printer canonicalizes — the round-trip tests exercise exactly that
//! path. M-interp reuses these as its first executable programs.

/// The design-doc example: nullable arithmetic against a probed static.
/// Covers: load, load.opt, probe, itof, fdiv, and, store.opt, emit.
pub const PROJECTION: &str = r#"
static @0: map(str) -> (f64)

fn run(in: batch{age: i64?, seg: str}, out: batch{z: f64?}) {
entry:
  %age_f, %age_v = load.opt in.age
  %seg = load in.seg
  %hit, %mean = probe @0, %seg
  %num = itof %age_v
  %q = fdiv %num, %mean
  %zf = and %age_f, %hit
  store.opt out.z, %zf, %q
  emit
}
"#;

/// A WHERE clause: the one construct allowed to change |out|.
/// Covers: const.f64, fcmp, brif (no args), skip, store.
pub const FILTER: &str = r#"
fn keep_positive(in: batch{score: f64}, out: batch{score: f64}) {
entry:
  %s = load in.score
  %zero = const.f64 0.0
  %pos = fcmp.gt %s, %zero
  brif %pos, keep, drop
keep:
  %s2 = load in.score
  store out.score, %s2
  emit
drop:
  skip
}
"#;

/// A CASE diamond joining through a block param, plus the i1 algebra.
/// Covers: const.i64, const.i1, const.str, icmp, jump with args, block
/// params, xor, not, select, store of i1.
pub const CASE_DIAMOND: &str = r#"
fn bucket(in: batch{age: i64}, out: batch{label: str, flag: i1}) {
entry:
  %age = load in.age
  %lim = const.i64 30
  %old = icmp.ge %age, %lim
  brif %old, old_b, young_b
old_b:
  %a = const.str "old"
  jump join(%a)
young_b:
  %b = const.str "young"
  jump join(%b)
join(%label: str):
  store out.label, %label
  %t = const.i1 true
  %f = const.i1 false
  %x = xor %t, %f
  %n = not %x
  %pick = select %n, %t, %f
  store out.flag, %pick
  emit
}
"#;

/// The cast family and scalar statics; a failed parse traps (SQL CAST
/// semantics — TRY_CAST lowers to a select instead).
/// Covers: stoi.opt, sload, sload.opt, ftoi.nearest, ftoi.trunc, isub, ftos,
/// sconcat, scmp, trap.
pub const CASTS: &str = r#"
static @0: scalar<f64>
static @1: scalar<i64?>

fn casts(in: batch{s: str}, out: batch{n: i64, msg: str}) {
entry:
  %raw = load in.s
  %ok, %parsed = stoi.opt %raw
  brif %ok, good, bad
good:
  %base = sload @0
  %r1 = ftoi.nearest %base
  %t1 = ftoi.trunc %base
  %d = isub %r1, %t1
  %pf, %pv = sload.opt @1
  %n2 = select %pf, %pv, %d
  store out.n, %n2
  %ftxt = ftos %base
  %sep = const.str ":"
  %msg1 = sconcat %ftxt, %sep
  %same = scmp.eq %msg1, %sep
  %sel = select %same, %msg1, %sep
  store out.msg, %sel
  emit
bad:
  trap "cast to i64 failed"
}
"#;

/// Everything else: the full integer/float binary set, a compound-key probe
/// with multiple value columns, and the remaining casts.
/// Covers: iadd, imul, idiv, irem, fadd, fsub, fmul, or, icmp.ne, fcmp.lt,
/// itos, stof.opt, multi-key/multi-value probe, quoted column names.
pub const KITCHEN: &str = r#"
static @0: map(i64, str) -> (f64, i64)

fn kitchen(in: batch{a: i64, b: i64?, k: str, x: f64}, out: batch{"r?": f64?, s: str}) {
entry:
  %a = load in.a
  %bf, %bv = load.opt in.b
  %sum = iadd %a, %bv
  %prod = imul %sum, %a
  %quot = idiv %prod, %a
  %rem = irem %quot, %a
  %k = load in.k
  %h, %vf, %vi = probe @0, %rem, %k
  %x = load in.x
  %fa = fadd %x, %vf
  %fs = fsub %fa, %vf
  %fm = fmul %fs, %x
  %lt = fcmp.lt %fm, %x
  %ne = icmp.ne %vi, %a
  %both = or %lt, %ne
  %rf = and %both, %bf
  %rv = select %lt, %fm, %x
  store.opt out."r?", %rf, %rv
  %istr = itos %vi
  %ff, %fv = stof.opt %istr
  %num2 = select %ff, %fv, %fm
  %ns = ftos %num2
  store out.s, %ns
  emit
}
"#;

/// Per-group tree scoring: the group's model id comes from a params-map
/// probe, the features come from nullable columns. Covers: model statics,
/// predict, const.f64 nan, the NULL-to-NaN select.
///
/// Only the probe miss gates the output — a NULL *feature* is a value the
/// model has a defined answer for (it takes the node's missing direction),
/// while a missing *model* is not.
pub const TREE_SCORE: &str = r#"
static @0: map(str) -> (i64)
static @1: model<tree_ensemble(2)>

fn score(in: batch{region: str, price: f64?, sqft: f64?}, out: batch{pred: f64?}) {
entry:
  %region = load in.region
  %hit, %mid = probe @0, %region
  %pf, %pv = load.opt in.price
  %sf, %sv = load.opt in.sqft
  %nan = const.f64 nan
  %f0 = select %pf, %pv, %nan
  %f1 = select %sf, %sv, %nan
  brif %hit, scored(%mid, %f0, %f1), unseen
scored(%m: i64, %a: f64, %b: f64):
  %p = predict @1, %m, %a, %b
  %t = const.i1 true
  store.opt out.pred, %t, %p
  emit
unseen:
  %f = const.i1 false
  %z = const.f64 0.0
  store.opt out.pred, %f, %z
  emit
}
"#;

pub fn all() -> Vec<(&'static str, &'static str)> {
    vec![
        ("projection", PROJECTION),
        ("filter", FILTER),
        ("case_diamond", CASE_DIAMOND),
        ("casts", CASTS),
        ("kitchen", KITCHEN),
        ("tree_score", TREE_SCORE),
    ]
}

/// Stage-B multiplicity: a multimap probe loop — per input row, one output
/// row per matching entry (emit.to back-edge), zero matches skip. Covers:
/// multimap statics, probe.range/probe.read, emit.to, a legal CFG cycle.
pub const MULTI_EXPAND: &str = r#"
static @0: multimap(i64) -> (i64)

fn expand(in: batch{id: i64}, out: batch{id: i64, v: i64}) {
entry:
  %id = load in.id
  %lo, %hi = probe.range @0, %id
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
