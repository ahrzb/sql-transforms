//! TASK-92: the function-signature registry.
//!
//! What a builtin ACCEPTS and RETURNS is one declarative table; HOW its
//! node is built stays a small arm in `Binder::function`. Rows marked
//! [`NullArg::WholeCallNull`] are resolved entirely by the head there
//! (arity, eager binding, the bare-NULL short-circuit, per-arg type
//! checks, the result type); rows marked [`NullArg::Custom`] and the
//! names in [`CUSTOM_NAMES`] keep every gate in their arm verbatim — for
//! those the table only records the audited facts.

use super::ir::Ty;

/// Accepted type of one parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgTy {
    /// Exactly this type — no implicit casts (measured on every row).
    Exact(Ty),
    /// Any integer. Today's lattice has a single integer lane, so this
    /// means exactly I64 — see [`arg_ok`], the ONE place that fact lives.
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
    /// rule. Those stay CUSTOM_NAMES today (guarded lazy binding), so no
    /// row constructs this yet; the width branch's Unify helper will.
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
    /// The arm keeps its own NULL/gate logic (TASK-82/86 refusals,
    /// skip-NULL desugars, lazy guarded binding).
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

/// The signature table: `(aliases, Sig)`. Every alias is a
/// `BUILTIN_NAMES` entry; together with [`CUSTOM_NAMES`] the aliases
/// partition the catalogue exactly (enforced by the totality test below).
/// The rows ARE the audited catalogue (fleet audit 2026-08-13).
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
    (&["pow", "power", "fdiv", "fmod", "nextafter"], whole(NUM2, Ret::Fixed(Ty::F64))),
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
    // audit 2026-08-13: DuckDB types unicode/ord/ascii INTEGER, all three;
    // the single-int-lane I64 here diverges on width (the m-8 phase-2
    // catalogue owns the fix; ascii -> I32 is a task-79 change).
    (&["unicode", "ord", "ascii"], whole(STR1, Ret::Fixed(Ty::I64))),
    (&["bit_length"], whole(STR1, Ret::Fixed(Ty::I64))),
    (&["strip_accents"], whole(STR1, Ret::Fixed(Ty::Str))),
    (&["reverse"], whole(STR1, Ret::Fixed(Ty::Str))),
    // ---- Custom rows: facts recorded, arm keeps every gate verbatim ----
    (
        &["repeat"], // arg0 bare NULL is the TASK-86 BLOB-face refusal
        custom(&[ArgTy::Exact(Ty::Str), ArgTy::Int], Ret::Fixed(Ty::Str)),
    ),
    (
        &["contains"], // NULL-literal-needle ambiguity gate (MAP/LIST)
        custom(STR2, Ret::Fixed(Ty::I1)),
    ),
    (
        // TASK-82: the count is additionally gated on int32-literal SHAPE
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

/// Builtins whose arm owns everything (variadic desugars, arity ranges,
/// AST-shape gates, unconditional refusals) — no expressible fixed Sig.
/// Consumed only by the totality test, which is its whole job.
#[cfg_attr(not(test), allow(dead_code))]
pub const CUSTOM_NAMES: &[&str] = &[
    // arity-range rows (1-or-2 / 2-to-4 args)
    "ltrim", "rtrim", "log", "round", "trunc",
    // variadic desugars and unification
    "concat", "concat_ws", "coalesce", "least", "greatest",
    // cmp-delegated comparability, Arg(0) result
    "nullif",
    // operator aliases: NULL adopts the OTHER operand's type (not the
    // whole-call rule), arity error is unsup, arith owns folds/guards
    "add", "subtract", "multiply", "divide", "mod", "xor",
    // constant-pattern/option gates, pinned NULL asymmetries
    "regexp_matches", "regexp_full_match", "regexp_extract", "regexp_replace",
    // unconditional named refusals
    "regexp_split_to_array", "regexp_extract_all",
    "sum", "count", "avg", "min", "max", "geomean", "product", "string_agg",
    "first", "last", "any_value",
    // AST-shape gate over a declared extern
    "struct_extract",
];

/// The single place a bound argument type meets its declared [`ArgTy`].
/// Today `Int` checks `== Ty::I64` exactly; the integer-width branch
/// relaxes THIS function, nowhere else.
pub fn arg_ok(want: ArgTy, got: Ty) -> bool {
    match want {
        ArgTy::Exact(t) => got == t,
        ArgTy::Int => got == Ty::I64,
        ArgTy::Num => matches!(got, Ty::I64 | Ty::F64),
    }
}

/// The `WholeCallNull` row for a lowercased call name, if any.
pub fn lookup(name: &str) -> Option<&'static Sig> {
    SIGS.iter()
        .find(|(aliases, _)| aliases.contains(&name))
        .map(|(_, s)| s)
}

/// Operator result-type rules (TASK-92): the RULE lookup consumed by
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specializer::frontend::BUILTIN_NAMES;

    /// TASK-92 totality: SIGS aliases plus CUSTOM_NAMES partition the
    /// builtin catalogue exactly — every builtin is either a table alias
    /// or explicitly custom, every alias is a real builtin, and no alias
    /// appears twice.
    #[test]
    fn table_and_custom_partition_the_catalogue() {
        let mut aliases: Vec<&str> = SIGS
            .iter()
            .flat_map(|(names, _)| names.iter().copied())
            .collect();
        aliases.extend(CUSTOM_NAMES.iter().copied());
        for a in &aliases {
            assert!(
                BUILTIN_NAMES.contains(a),
                "alias {a} is not in BUILTIN_NAMES"
            );
        }
        let mut deduped = aliases.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            deduped.len(),
            aliases.len(),
            "an alias appears twice across SIGS/CUSTOM_NAMES"
        );
        for b in BUILTIN_NAMES {
            assert!(
                aliases.contains(b),
                "builtin {b} has neither a table row nor a CUSTOM_NAMES entry"
            );
        }
    }
}
