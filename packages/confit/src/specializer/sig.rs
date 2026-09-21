//! The function-signature registry.
//!
//! What a builtin ACCEPTS and RETURNS is one declarative table; HOW its
//! node is built stays a small arm in `Binder::function`. Rows marked
//! [`NullArg::WholeCallNull`] are resolved entirely by the head there
//! (arity, eager binding, the bare-NULL short-circuit, per-arg type
//! checks, the result type). Custom rows and builtins without a table row
//! keep their validation in `Binder::function`; parser desugaring can also
//! lower a builtin to another expression class before dispatch.

use super::ir::Ty;

/// Accepted type of one parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgTy {
    /// Exactly this type — no implicit casts (measured on every row).
    Exact(Ty),
    /// Any integer width — see [`arg_ok`], the ONE place that fact lives.
    Int,
    /// Any integer or f64.
    Num,
}

/// Result-type rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ret {
    Fixed(Ty),
    /// That argument's own type (abs, round, nullif).
    Arg(usize),
    /// Integer-width promotion across the args; f64 is contagious
    /// (operators: + - * // % and the bitwise family).
    Widen,
    /// Full numeric unification incl. f64 — coalesce/least/greatest's
    /// rule. Those use guarded lazy binding, so no row constructs this yet;
    /// the width branch's Unify helper will.
    #[allow(dead_code)]
    Unify,
}

/// Who owns bare-NULL argument handling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NullArg {
    /// Any bare-NULL argument makes the whole call NULL of the resolved
    /// result type, BEFORE per-arg type checks — the audited dominant
    /// pattern (e.g. replace(NULL, 1, 2) binds NULL::VARCHAR).
    WholeCallNull,
    /// The arm keeps its own NULL/gate logic (the pad-count and
    /// bare-NULL-BLOB refusals, skip-NULL desugars, lazy guarded binding).
    Custom,
}

/// One overload of one function name.
pub struct Sig {
    pub params: &'static [ArgTy],
    pub variadic: bool,
    pub ret: Ret,
    pub null_arg: NullArg,
}

const fn whole(params: &'static [ArgTy], ret: Ret) -> Sig {
    Sig {
        params,
        variadic: false,
        ret,
        null_arg: NullArg::WholeCallNull,
    }
}

const fn custom(params: &'static [ArgTy], ret: Ret) -> Sig {
    Sig {
        params,
        variadic: false,
        ret,
        null_arg: NullArg::Custom,
    }
}

const STR1: &[ArgTy] = &[ArgTy::Exact(Ty::Str)];
const STR2: &[ArgTy] = &[ArgTy::Exact(Ty::Str), ArgTy::Exact(Ty::Str)];
const STR3: &[ArgTy] = &[
    ArgTy::Exact(Ty::Str),
    ArgTy::Exact(Ty::Str),
    ArgTy::Exact(Ty::Str),
];
const NUM1: &[ArgTy] = &[ArgTy::Num];
const NUM2: &[ArgTy] = &[ArgTy::Num, ArgTy::Num];

/// Declarative signatures for overloads dispatched by `Binder::function`.
pub const SIGS: &[(&[&str], Sig)] = &[
    (
        &["upper", "lower", "ucase", "lcase"],
        whole(STR1, Ret::Fixed(Ty::Str)),
    ),
    (
        &["length", "len", "char_length", "character_length", "strlen"],
        whole(STR1, Ret::Fixed(Ty::I64)),
    ),
    // audit 2026-08-13: DuckDB's parser refuses the bare 2-arg call form
    // position(h, n) (only POSITION(n IN h) parses); binding that spelling
    // here is laxer than the oracle — preserved.
    (
        &["instr", "strpos", "position"],
        whole(STR2, Ret::Fixed(Ty::I64)),
    ),
    (
        &["starts_with", "prefix", "ends_with", "suffix"],
        whole(STR2, Ret::Fixed(Ty::I1)),
    ),
    (
        &[
            "ln", "log2", "log10", "exp", "sqrt", "cbrt", "sin", "cos", "tan", "floor", "ceil",
            "ceiling",
        ],
        whole(NUM1, Ret::Fixed(Ty::F64)),
    ),
    // audit 2026-08-13: the whole-call-NULL short-circuit runs before the
    // per-arg type checks, so pow(s, NULL) binds NULL::DOUBLE here where
    // DuckDB refuses the VARCHAR sibling — looser than the oracle for this
    // math2 family (pow/power, fdiv, fmod, nextafter; log's 2-arg form
    // shares it via the math2 helper). Preserved.
    (
        &["pow", "power", "fdiv", "fmod", "nextafter"],
        whole(NUM2, Ret::Fixed(Ty::F64)),
    ),
    (&["pi"], whole(&[], Ret::Fixed(Ty::F64))),
    (&["abs"], whole(NUM1, Ret::Arg(0))),
    (
        &[
            "levenshtein",
            "editdist3",
            "damerau_levenshtein",
            "hamming",
            "mismatches",
        ],
        whole(STR2, Ret::Fixed(Ty::I64)),
    ),
    (&["jaccard"], whole(STR2, Ret::Fixed(Ty::F64))),
    (&["replace", "translate"], whole(STR3, Ret::Fixed(Ty::Str))),
    // m-8 phase 2: a codepoint is INTEGER on DuckDB, all three names.
    (
        &["unicode", "ord", "ascii"],
        whole(STR1, Ret::Fixed(Ty::I32)),
    ),
    (&["bit_length"], whole(STR1, Ret::Fixed(Ty::I64))),
    (&["strip_accents"], whole(STR1, Ret::Fixed(Ty::Str))),
    (&["reverse"], whole(STR1, Ret::Fixed(Ty::Str))),
    // ---- Custom rows: facts recorded, arm keeps every gate verbatim ----
    (
        &["repeat"], // arg0 bare NULL picks DuckDB's BLOB overload: refused
        custom(&[ArgTy::Exact(Ty::Str), ArgTy::Int], Ret::Fixed(Ty::Str)),
    ),
    (
        &["contains"], // NULL-literal-needle ambiguity gate (MAP/LIST)
        custom(STR2, Ret::Fixed(Ty::I1)),
    ),
    (
        // The count is additionally gated on int32-literal SHAPE
        // (a syntactic predicate no bound type can express) BEFORE the
        // NULL short-circuit, plus the 1 GiB budget refusal.
        &["lpad", "rpad"],
        custom(
            &[ArgTy::Exact(Ty::Str), ArgTy::Int, ArgTy::Exact(Ty::Str)],
            Ret::Fixed(Ty::Str),
        ),
    ),
    (
        &["array_extract", "list_extract"], // shared with bracket s[i]
        custom(&[ArgTy::Exact(Ty::Str), ArgTy::Int], Ret::Fixed(Ty::Str)),
    ),
    (
        &["array_slice", "list_slice"], // 4-arg step refusal precedes arity
        custom(
            &[ArgTy::Exact(Ty::Str), ArgTy::Int, ArgTy::Int],
            Ret::Fixed(Ty::Str),
        ),
    ),
];

/// The single place a bound argument type meets its declared [`ArgTy`].
/// m-8 phase 2: `Int` is any width of DuckDB's integer lattice — narrow
/// widths upcast implicitly into a wider slot, never the reverse.
pub fn arg_ok(want: ArgTy, got: Ty) -> bool {
    match want {
        ArgTy::Exact(t) => got == t,
        ArgTy::Int => got.is_int(),
        ArgTy::Num => got.is_int() || got == Ty::F64,
    }
}

/// The `WholeCallNull` row for a lowercased call name, if any.
pub fn lookup(name: &str) -> Option<&'static Sig> {
    SIGS.iter()
        .find(|(aliases, _)| aliases.contains(&name))
        .map(|(_, s)| s)
}

/// Operator result-type rules: the RULE lookup consumed by
/// `numeric_promote` and `cmp`; all machinery (constant folds/refusals,
/// NULL-op-NULL, zero-divisor guards) stays with the operators. m-8
/// phase 5 turns DECIMAL scale propagation into more data here.
pub const OPS: &[(&str, Ret)] = &[
    ("+", Ret::Widen),
    ("-", Ret::Widen),
    ("*", Ret::Widen),
    ("//", Ret::Widen),
    ("%", Ret::Widen),
    ("&", Ret::Widen),
    ("|", Ret::Widen),
    ("<<", Ret::Widen),
    (">>", Ret::Widen),
    ("xor", Ret::Widen),
    ("/", Ret::Fixed(Ty::F64)),
    ("=", Ret::Fixed(Ty::I1)),
    ("<>", Ret::Fixed(Ty::I1)),
    ("<", Ret::Fixed(Ty::I1)),
    ("<=", Ret::Fixed(Ty::I1)),
    (">", Ret::Fixed(Ty::I1)),
    (">=", Ret::Fixed(Ty::I1)),
];

pub fn op_ret(sym: &str) -> Ret {
    OPS.iter()
        .find(|(s, _)| *s == sym)
        .map(|(_, r)| *r)
        .expect("every operator symbol has a table row")
}
