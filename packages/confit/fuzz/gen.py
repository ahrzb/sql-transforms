"""Seeded Case generator: a tiny query AST rendered to SQL at the edge.

The AST — not the SQL string — is what the shrinker edits, so every node is a
mutable dataclass with a `kids`/`swap` protocol. Generation is type-directed
(`expr(rng, env, ty, depth)`) over the SEMANTIC vocabulary
(int/float/str/bool) and weighted toward the expression spine; exotic
productions (refused clauses, hostile identifiers, bare decimal literals) run
at low weight so a divergence in one of them surfaces as one deduped class
instead of drowning the campaign.

Two axes, deliberately separate. A column's SEMANTIC type picks the
operators; its STORAGE type (`SCALARS`, plus `Struct`, plus `OPAQUE`) is what
the table actually declares. They were fused once — every generated column
was int64/double/string/bool — and the cost was invisible: narrow widths and
struct columns were unreachable by ANY seed, so whole families of live bugs
could not be found no matter how long a campaign ran. `leaves()` is the
bridge: it flattens a storage schema to the referenceable lanes the
expression layer binds. test_fuzz_smoke.py's parity tests keep the two axes
from drifting apart again.
"""

from __future__ import annotations

import decimal
import random
from dataclasses import dataclass, field

TYPES = ("int", "float", "str", "bool")
"""The SEMANTIC vocabulary — what the expression layer reasons about when it
picks operators and builtin overloads. Deliberately four-wide; a column's
storage width is a separate axis (below) so widening storage does not
multiply the expression grammar."""

# The STORAGE vocabulary: exactly what schema.rs::arrow_field_to_row_field
# serves, at exactly its width. int32 IS Ty::I32, never a collapse to i64 —
# so `a + a` over an int32 column is INTEGER arithmetic on DuckDB and traps
# at INT32, which is a different program from the int64 one. A spec string
# carries nullability as a trailing `?`.
SCALARS = ("bool", "int8", "int16", "int32", "int64", "double", "string")

# Outside the vocabulary on purpose. The boundary rule is opaque-unless-
# referenced: an unreferenced foreign column must never block a build.
OPAQUE = ("float32", "timestamp")

# STATIC-ONLY storage. Decimals serve exactly as decimal128(p,s) from a
# static table and stay opaque in a row table, so the generator emits them
# only on the static side -- one spelling per DuckDB storage tier
# (int16 / int32 / int64 / int128 by precision), plus the (38,0) that an
# ordinary fit-time sum(BIGINT) actually produces.
STATIC_ONLY = ("decimal(4,2)", "decimal(9,4)", "decimal(18,6)", "decimal(38,0)")

SEMANTIC = {
    "bool": "bool",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "double": "float",
    "string": "str",
}

# Inclusive value range per integer width, so generated data actually fits
# the column it is declared in.
INT_RANGE = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}


@dataclass(frozen=True)
class Struct:
    """A struct-typed table column. Its leaves are lanes: `w.mean` binds
    exactly like a top-level double column, in a row table AND in a static
    one. `fields` is a tuple so a Case stays hashable/comparable."""

    fields: tuple[tuple[str, object], ...]  # (name, spec) — spec is str | Struct
    nullable: bool = False


def leaves(schema: dict, prefix: str = "") -> list[tuple[str, str, bool]]:
    """(dotted path, semantic type, nullable) for every REFERENCEABLE leaf.

    Opaque columns are omitted: they exist in the table and must not break
    the build, but nothing may bind them. A struct's nullability flows down
    to its leaves — a null parent nulls every lane under it."""
    out: list[tuple[str, str, bool]] = []
    for name, spec in schema.items():
        path = f"{prefix}{name}"
        if isinstance(spec, Struct):
            out += [
                (p, t, spec.nullable or n)
                for p, t, n in leaves(dict(spec.fields), f"{path}.")
            ]
        elif spec.rstrip("?") in SEMANTIC:
            out.append((path, SEMANTIC[spec.rstrip("?")], spec.endswith("?")))
    return out


# The engine's builtin catalogue (frontend.rs BUILTIN_NAMES). Signatures are
# hand-written for the common core below; the rest are called through the
# wildcard production with random args — wrong shapes refuse, which is itself
# the surface under test.
BUILTIN_NAMES = [
    "abs",
    "add",
    "any_value",
    "array_extract",
    "array_slice",
    "ascii",
    "avg",
    "bit_length",
    "cbrt",
    "ceil",
    "ceiling",
    "char_length",
    "character_length",
    "coalesce",
    "concat",
    "concat_ws",
    "contains",
    "cos",
    "count",
    "damerau_levenshtein",
    "divide",
    "editdist3",
    "ends_with",
    "exp",
    "fdiv",
    "first",
    "floor",
    "fmod",
    "geomean",
    "greatest",
    "hamming",
    "instr",
    "jaccard",
    "last",
    "lcase",
    "least",
    "len",
    "length",
    "levenshtein",
    "list_extract",
    "list_slice",
    "ln",
    "log",
    "log10",
    "log2",
    "lower",
    "lpad",
    "ltrim",
    "max",
    "min",
    "mismatches",
    "mod",
    "multiply",
    "nextafter",
    "nullif",
    "ord",
    "pi",
    "position",
    "pow",
    "power",
    "prefix",
    "product",
    "regexp_extract",
    "regexp_extract_all",
    "regexp_full_match",
    "regexp_matches",
    "regexp_replace",
    "regexp_split_to_array",
    "repeat",
    "replace",
    "reverse",
    "round",
    "rpad",
    "rtrim",
    "sin",
    "sqrt",
    "starts_with",
    "string_agg",
    "strip_accents",
    "strlen",
    "strpos",
    "struct_extract",
    "subtract",
    "suffix",
    "sum",
    "tan",
    "translate",
    "trunc",
    "ucase",
    "unicode",
    "upper",
    "xor",
]

# name -> (arg types, result type). Only the differential core; arity-variable
# builtins get a fixed common arity here and the wildcard covers the rest.
SIGS = {
    "abs": (("float",), "float"),
    "ceil": (("float",), "float"),
    "floor": (("float",), "float"),
    "round": (("float",), "float"),
    "trunc": (("float",), "float"),
    "sqrt": (("float",), "float"),
    "exp": (("float",), "float"),
    "ln": (("float",), "float"),
    "sin": (("float",), "float"),
    "cos": (("float",), "float"),
    "pow": (("float", "float"), "float"),
    # greatest patrols the int width lane, least the float lane.
    "greatest": (("int", "int"), "int"),
    "least": (("float", "float"), "float"),
    "upper": (("str",), "str"),
    "lower": (("str",), "str"),
    "ltrim": (("str",), "str"),
    "rtrim": (("str",), "str"),
    "reverse": (("str",), "str"),
    "concat": (("str", "str"), "str"),
    "replace": (("str", "str", "str"), "str"),
    "length": (("str",), "int"),
    "strpos": (("str", "str"), "int"),
    "ascii": (("str",), "int"),
    # INTEGER-returning names patrol the width catalogue.
    "ord": (("str",), "int"),
    "unicode": (("str",), "int"),
    "repeat": (("str", "int"), "str"),
    "lpad": (("str", "int", "str"), "str"),
    "contains": (("str", "str"), "bool"),
    "starts_with": (("str", "str"), "bool"),
    "ends_with": (("str", "str"), "bool"),
    "levenshtein": (("str", "str"), "int"),
    "coalesce": (("float", "float"), "float"),
    "nullif": (("float", "float"), "float"),
}

AGGS = {  # statics-side aggregation only (fit-time surface)
    "sum": "float",
    "avg": "float",
    "min": "float",
    "max": "float",
    "count": "int",
}

# ---------------------------------------------------------------- AST nodes


@dataclass
class Node:
    """An expression node, and the edit protocol the shrinker drives it by:
    `kids()` lists the child expressions and `swap(i, new)` replaces the
    i-th of THAT list. The two orders must agree — a fold addresses a child
    by its `kids()` index and then swaps at the same index. A leaf keeps
    both defaults, so swapping into one raises."""

    def kids(self) -> list[Node]:
        return []

    def swap(self, i: int, new: Node) -> None:
        raise IndexError


@dataclass
class Lit(Node):
    val: object
    ty: str
    bare_decimal: bool = False  # render 2.5 (DECIMAL on duck) instead of 2.5e0


@dataclass
class Col(Node):
    name: str
    table: str | None = None
    ty: str = "float"


@dataclass
class Bin(Node):
    op: str
    lhs: Node
    rhs: Node

    def kids(self):
        return [self.lhs, self.rhs]

    def swap(self, i, new):
        setattr(self, ("lhs", "rhs")[i], new)


@dataclass
class Un(Node):
    op: str  # "NOT" | "-"
    e: Node

    def kids(self):
        return [self.e]

    def swap(self, i, new):
        self.e = new


@dataclass
class CaseW(Node):
    whens: list[tuple[Node, Node]]
    els: Node | None

    def kids(self):
        out = [x for w in self.whens for x in w]
        return out + ([self.els] if self.els else [])

    def swap(self, i, new):
        flat = self.kids()
        if self.els is not None and i == len(flat) - 1:
            self.els = new
            return
        w, half = divmod(i, 2)
        old = self.whens[w]
        self.whens[w] = (new, old[1]) if half == 0 else (old[0], new)


@dataclass
class Cast(Node):
    e: Node
    to: str
    try_: bool = False

    def kids(self):
        return [self.e]

    def swap(self, i, new):
        self.e = new


@dataclass
class Call(Node):
    name: str
    args: list[Node]
    modifier: str | None = None  # "over" | "filter" | "ignore_nulls" (refused)

    def kids(self):
        return list(self.args)

    def swap(self, i, new):
        self.args[i] = new


@dataclass
class LaneRead(Node):
    call: Call
    fld: str

    def kids(self):
        return [self.call]

    def swap(self, i, new):
        if isinstance(new, Call):
            self.call = new
        else:
            raise IndexError


@dataclass
class StructPack(Node):
    fields: list[tuple[str, Node]]

    def kids(self):
        return [e for _, e in self.fields]

    def swap(self, i, new):
        self.fields[i] = (self.fields[i][0], new)


@dataclass
class Between(Node):
    e: Node
    lo: Node
    hi: Node
    neg: bool = False

    def kids(self):
        return [self.e, self.lo, self.hi]

    def swap(self, i, new):
        setattr(self, ("e", "lo", "hi")[i], new)


@dataclass
class InList(Node):
    e: Node
    items: list[Node]
    neg: bool = False

    def kids(self):
        return [self.e, *self.items]

    def swap(self, i, new):
        if i == 0:
            self.e = new
        else:
            self.items[i - 1] = new


@dataclass
class IsNull(Node):
    e: Node
    neg: bool = False

    def kids(self):
        return [self.e]

    def swap(self, i, new):
        self.e = new


@dataclass
class Like(Node):
    e: Node
    pat: str
    neg: bool = False

    def kids(self):
        return [self.e]

    def swap(self, i, new):
        self.e = new


@dataclass
class Star(Node):
    exclude: list[str] = field(default_factory=list)
    replace: list[tuple[Node, str]] = field(default_factory=list)
    columns_re: str | None = None  # COLUMNS('re')
    qualifier: str | None = None  # `t.*` — a star over one joined relation


@dataclass
class Join:
    kind: str  # INNER|LEFT|RIGHT|FULL|CROSS (only the first two build)
    table: str
    on: Node | None


@dataclass
class Sel:
    items: list[tuple[Node, str | None]]
    frm: str | None  # "__THIS__" | static name | None (sub-select in FROM)
    sub: Sel | None = None
    joins: list[Join] = field(default_factory=list)
    where: Node | None = None
    group_by: list[Node] = field(default_factory=list)
    # hostile clauses — each is refused by the engine today; a build is a bug
    distinct: bool = False
    order_by: str | None = None
    limit: int | None = None
    qualify: Node | None = None
    top: int | None = None
    fetch: int | None = None


@dataclass
class Q:
    ctes: list[tuple[str, Sel]]
    body: Sel


# --------------------------------------------------------------- case shape


@dataclass
class UdfSpec:
    name: str
    takes: list[tuple[str, str]]  # (name, ty)
    ret: tuple  # ("scalar", ty) | ("struct", [(name, ty)]) | ("list", ty, k)
    instances: int = 0  # 0 = plain udf; k = instance-bearing, ids 0..k-1


@dataclass
class TreeSpec:
    kind: str  # "dtr" | "rf" | "gbr"
    n_features: int
    int_features: tuple[int, ...]  # indices declared int64 (rest double)
    depth: int
    instances: int


@dataclass
class Case:
    seed: int
    row_schema: dict[str, str]  # column -> storage spec (a name, or a Struct)
    rows: list[dict]  # TOTAL dicts against row_schema — every column present
    statics: dict[str, tuple[dict[str, str], list[dict]]]  # name -> (schema, rows)
    udfs: list[UdfSpec]
    tree: TreeSpec | None
    query: Q
    shape: str | None  # the declared row-shape contract; None = undeclared
    output: str | None  # nothing consumes this: dict rows are the only mode
    # the constructs this case exercises, for the coverage histogram
    tags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ values

INTS = [
    0,
    1,
    -1,
    2,
    7,
    -13,
    100,
    2**31 - 1,
    -(2**31),
    2**53 - 1,
    2**53,
    2**53 + 1,
    -(2**53) - 1,
    2**63 - 1,
    -(2**63),
]
FLOATS = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    2.5,
    -2.5,
    0.1,
    1e-300,
    1e300,
    1234.5678,
    float("inf"),
    float("-inf"),
    float("nan"),
]
STRS = [
    "",
    "a",
    "One",
    "one",
    "NULL",
    "%_",
    "O'Brien",
    "é☃",
    "  pad  ",
    "0",
    "abcdefghij" * 3,
]
QUANT = [i / 4 for i in range(-8, 9)]  # quantised grid — the sklearn lesson


def value(rng: random.Random, ty: str, nullable: bool, storage: str = "") -> object:
    """One data value of semantic type `ty`. `storage` is the declared
    integer width ("" = int64), and the value never leaves its range."""
    if nullable and rng.random() < 0.3:
        return None
    if ty == "int":
        # Clamped to the declared width, and biased to that width's OWN
        # boundary values — an int8 column whose extremes are +-127 is what
        # makes a narrow-width overflow reachable at all.
        lo, hi = INT_RANGE.get(storage, INT_RANGE["int64"])
        pool = [v for v in INTS if lo <= v <= hi] + [lo, hi]
        if rng.random() < 0.5:
            return rng.choice(pool)
        return rng.randrange(max(lo, -50), min(hi, 50) + 1)
    if ty == "float":
        r = rng.random()
        if r < 0.35:
            return rng.choice(QUANT)
        if r < 0.7:
            return rng.choice(FLOATS)
        return round(rng.uniform(-100, 100), 3)
    if ty == "str":
        return rng.choice(STRS)
    return rng.random() < 0.5


def _finite(rng: random.Random, ty: str) -> object:
    """A literal-safe value: finite floats (NaN/inf live in DATA, not SQL)."""
    while True:
        v = value(rng, ty, nullable=False)
        if ty != "float" or (v == v and abs(v) != float("inf")):
            return v


def lit(rng: random.Random, ty: str) -> Lit:
    if rng.random() < 0.06:
        return Lit(None, ty)
    v = _finite(rng, ty)
    bare = ty == "float" and rng.random() < 0.05  # DECIMAL-typed on duck
    return Lit(v, ty, bare_decimal=bare)


# ------------------------------------------------------------- environments


@dataclass
class Env:
    """Columns in scope, by type: list of (table-or-None, name, ty)."""

    cols: list[tuple[str | None, str, str]]
    udfs: list[UdfSpec]
    tree: TreeSpec | None

    def of(self, ty: str) -> list[tuple[str | None, str, str]]:
        return [c for c in self.cols if c[2] == ty]


CMP = ["=", "<>", "<", ">", "<=", ">="]
ARITH_I = ["+", "-", "*", "%"]  # int/int division truncates; % traps on 0
ARITH_F = ["+", "-", "*", "/"]


def expr(rng: random.Random, env: Env, ty: str, depth: int) -> Node:
    """A type-directed random expression of type `ty`."""
    cols = env.of(ty)
    if depth <= 0:
        if cols and rng.random() < 0.6:
            t, n, _ = rng.choice(cols)
            return Col(n, t, ty)
        return lit(rng, ty)
    r = rng.random()
    if cols and r < 0.25:
        t, n, _ = rng.choice(cols)
        return Col(n, t, ty)
    if r < 0.32:
        return lit(rng, ty)
    if ty in ("int", "float"):
        return _num(rng, env, ty, depth)
    if ty == "bool":
        return _bool(rng, env, depth)
    return _str(rng, env, depth)


def _num(rng, env, ty, depth):
    r = rng.random()
    if r < 0.40:
        ops = ARITH_I if ty == "int" else ARITH_F
        return Bin(
            rng.choice(ops),
            expr(rng, env, ty, depth - 1),
            expr(rng, env, ty, depth - 1),
        )
    if r < 0.48:
        return Un("-", expr(rng, env, ty, depth - 1))
    if r < 0.60:
        n = rng.randrange(1, 3)
        whens = [
            (expr(rng, env, "bool", depth - 1), expr(rng, env, ty, depth - 1))
            for _ in range(n)
        ]
        els = expr(rng, env, ty, depth - 1) if rng.random() < 0.7 else None
        return CaseW(whens, els)
    if r < 0.70:
        src = rng.choice(TYPES)
        # Narrow targets are real: CAST/TRY_CAST AS INTEGER exercises the
        # typed-width lane and its range semantics.
        to = "int32" if ty == "int" and rng.random() < 0.4 else ty
        return Cast(expr(rng, env, src, depth - 1), to, try_=rng.random() < 0.3)
    if r < 0.78:
        name = rng.choice(["coalesce", "nullif"])
        return Call(
            name, [expr(rng, env, ty, depth - 1), expr(rng, env, ty, depth - 1)]
        )
    if r < 0.92:
        sigs = [(n, s) for n, s in SIGS.items() if s[1] == ty]
        name, (args, _) = rng.choice(sigs)
        return Call(name, [expr(rng, env, a, depth - 1) for a in args])
    return _udf_call(rng, env, ty, depth) or lit(rng, ty)


def _bool(rng, env, depth):
    r = rng.random()
    if r < 0.35:
        t = rng.choice(("int", "float", "str", "bool"))
        return Bin(
            rng.choice(CMP), expr(rng, env, t, depth - 1), expr(rng, env, t, depth - 1)
        )
    if r < 0.55:
        return Bin(
            rng.choice(["AND", "OR"]),
            expr(rng, env, "bool", depth - 1),
            expr(rng, env, "bool", depth - 1),
        )
    if r < 0.63:
        return Un("NOT", expr(rng, env, "bool", depth - 1))
    if r < 0.73:
        t = rng.choice(("int", "float"))
        return Between(
            expr(rng, env, t, depth - 1),
            expr(rng, env, t, 0),
            expr(rng, env, t, 0),
            neg=rng.random() < 0.3,
        )
    if r < 0.81:
        t = rng.choice(("int", "str"))
        items = [lit(rng, t) for _ in range(rng.randrange(1, 4))]
        return InList(expr(rng, env, t, depth - 1), items, neg=rng.random() < 0.3)
    if r < 0.90:
        return IsNull(
            expr(rng, env, rng.choice(TYPES), depth - 1), neg=rng.random() < 0.5
        )
    pat = rng.choice(["%", "a%", "%e%", "_", "O''Brien", "%☃%", "a_c", ""])
    return Like(expr(rng, env, "str", depth - 1), pat, neg=rng.random() < 0.3)


def _str(rng, env, depth):
    r = rng.random()
    if r < 0.5:
        sigs = [(n, s) for n, s in SIGS.items() if s[1] == "str"]
        name, (args, _) = rng.choice(sigs)
        return Call(name, [expr(rng, env, a, depth - 1) for a in args])
    if r < 0.7:
        return CaseW(
            [(expr(rng, env, "bool", depth - 1), expr(rng, env, "str", depth - 1))],
            expr(rng, env, "str", 0),
        )
    return Cast(expr(rng, env, rng.choice(TYPES), depth - 1), "str")


def _udf_call(rng, env, ty, depth) -> Node | None:
    """A declared-UDF (or tree) call whose scalar type is `ty`, else None."""
    if ty == "float" and env.tree and rng.random() < 0.5:
        t = env.tree
        iid = _instance_arg(rng, env, t.instances)
        args = [
            expr(rng, env, "int" if i in t.int_features else "float", 0)
            for i in range(t.n_features)
        ]
        return Call("trees", [iid, *args])
    scalars = [u for u in env.udfs if u.ret[0] == "scalar" and u.ret[1] == ty]
    named = [
        u
        for u in env.udfs
        if u.ret[0] == "struct" and any(t == ty for _, t in u.ret[1])
    ]
    if named and rng.random() < 0.4:
        u = rng.choice(named)
        fld = rng.choice([n for n, t in u.ret[1] if t == ty])
        return LaneRead(Call(u.name, _udf_args(rng, env, u)), fld)
    if scalars:
        u = rng.choice(scalars)
        return Call(u.name, _udf_args(rng, env, u))
    return None


def _udf_args(rng, env, u: UdfSpec) -> list[Node]:
    args = [expr(rng, env, t, 1) for _, t in u.takes]
    if u.instances:
        args.insert(0, _instance_arg(rng, env, u.instances))
    return args


def _instance_arg(rng, env, k: int) -> Node:
    r = rng.random()
    if r < 0.6:
        return Lit(rng.randrange(k), "int")
    if r < 0.75:
        return Lit(None, "int")  # NULL id -> NULL output
    if r < 0.85:
        return Lit(k + 3, "int")  # unseen id
    ints = env.of("int")
    if ints:
        t, n, _ = rng.choice(ints)
        return Col(n, t, "int")
    return Lit(0, "int")


# ---------------------------------------------------------------- rendering


def _ident(name: str) -> str:
    if name.replace("_", "").isalnum() and name.isascii() and not name[0].isdigit():
        return name
    return '"' + name.replace('"', '""') + '"'


def _rlit(v, ty, bare=False) -> str:
    if v is None:
        return "NULL"
    if ty == "int":
        return str(v)
    if ty == "float":
        s = repr(v)
        # `e0` forces DOUBLE typing on DuckDB; a bare 2.5 is DECIMAL(2,1)
        # there (its own divergence class, generated deliberately at low
        # weight via bare_decimal). repr may already carry an exponent.
        return s if (bare or "e" in s) else s + "e0"
    if ty == "bool":
        return "TRUE" if v else "FALSE"
    return "'" + str(v).replace("'", "''") + "'"


def rexpr(e: Node) -> str:
    if isinstance(e, Lit):
        return _rlit(e.val, e.ty, e.bare_decimal)
    if isinstance(e, Col):
        # A lane name is a dotted path (`w.mean`); each segment quotes on its
        # own, or the whole path becomes one identifier that exists nowhere.
        q = ".".join(_ident(p) for p in e.name.split("."))
        return f"{_ident(e.table)}.{q}" if e.table else q
    if isinstance(e, Bin):
        return f"({rexpr(e.lhs)} {e.op} {rexpr(e.rhs)})"
    if isinstance(e, Un):
        return f"({e.op} {rexpr(e.e)})"
    if isinstance(e, CaseW):
        w = " ".join(f"WHEN {rexpr(c)} THEN {rexpr(v)}" for c, v in e.whens)
        els = f" ELSE {rexpr(e.els)}" if e.els is not None else ""
        return f"(CASE {w}{els} END)"
    if isinstance(e, Cast):
        to = {
            "int": "BIGINT",
            "int32": "INTEGER",
            "float": "DOUBLE",
            "str": "VARCHAR",
            "bool": "BOOLEAN",
        }[e.to]
        f = "TRY_CAST" if e.try_ else "CAST"
        return f"{f}({rexpr(e.e)} AS {to})"
    if isinstance(e, Call):
        args = ", ".join(rexpr(a) for a in e.args)
        base = f"{e.name}({args})"
        if e.modifier == "over":
            return f"{base} OVER ()"
        if e.modifier == "filter":
            return f"{base} FILTER (WHERE TRUE)"
        if e.modifier == "ignore_nulls":
            return f"{e.name}({args} IGNORE NULLS)"
        return base
    if isinstance(e, LaneRead):
        return f"({rexpr(e.call)}).{_ident(e.fld)}"
    if isinstance(e, StructPack):
        inner = ", ".join(f"{_ident(n)} := {rexpr(v)}" for n, v in e.fields)
        return f"struct_pack({inner})"
    if isinstance(e, Between):
        n = "NOT " if e.neg else ""
        return f"({rexpr(e.e)} {n}BETWEEN {rexpr(e.lo)} AND {rexpr(e.hi)})"
    if isinstance(e, InList):
        n = "NOT " if e.neg else ""
        return f"({rexpr(e.e)} {n}IN ({', '.join(rexpr(i) for i in e.items)}))"
    if isinstance(e, IsNull):
        return f"({rexpr(e.e)} IS {'NOT ' if e.neg else ''}NULL)"
    if isinstance(e, Like):
        n = "NOT " if e.neg else ""
        return f"({rexpr(e.e)} {n}LIKE '{e.pat}')"
    raise TypeError(e)


def _ritem(item: tuple[Node, str | None]) -> str:
    e, alias = item
    if isinstance(e, Star):
        if e.columns_re is not None:
            return f"COLUMNS('{e.columns_re}')"
        s = "*" if e.qualifier is None else f"{_ident(e.qualifier)}.*"
        if e.exclude:
            s += f" EXCLUDE ({', '.join(_ident(c) for c in e.exclude)})"
        if e.replace:
            rep = ", ".join(f"{rexpr(x)} AS {_ident(a)}" for x, a in e.replace)
            s += f" REPLACE ({rep})"
        return s
    return rexpr(e) + (f" AS {_ident(alias)}" if alias else "")


def _rsel(s: Sel) -> str:
    parts = ["SELECT"]
    if s.distinct:
        parts.append("DISTINCT")
    if s.top is not None:
        parts.append(f"TOP {s.top}")
    parts.append(", ".join(_ritem(i) for i in s.items))
    if s.sub is not None:
        parts.append(f"FROM ({_rsel(s.sub)}) AS sub")
    elif s.frm is not None:
        parts.append(f"FROM {_ident(s.frm)}")
    for j in s.joins:
        k = {
            "INNER": "JOIN",
            "LEFT": "LEFT JOIN",
            "RIGHT": "RIGHT JOIN",
            "FULL": "FULL OUTER JOIN",
            "CROSS": "CROSS JOIN",
        }[j.kind]
        on = f" ON {rexpr(j.on)}" if j.on is not None else ""
        parts.append(f"{k} {_ident(j.table)}{on}")
    if s.where is not None:
        parts.append(f"WHERE {rexpr(s.where)}")
    if s.group_by:
        parts.append("GROUP BY " + ", ".join(rexpr(g) for g in s.group_by))
    if s.qualify is not None:
        parts.append(f"QUALIFY {rexpr(s.qualify)}")
    if s.order_by is not None:
        parts.append(f"ORDER BY {_ident(s.order_by)}")
    if s.limit is not None:
        parts.append(f"LIMIT {s.limit}")
    if s.fetch is not None:
        parts.append(f"FETCH FIRST {s.fetch} ROWS ONLY")
    return " ".join(parts)


def render(q: Q) -> str:
    body = _rsel(q.body)
    if not q.ctes:
        return body
    ctes = ", ".join(f"{_ident(n)} AS ({_rsel(s)})" for n, s in q.ctes)
    return f"WITH {ctes} {body}"


# ------------------------------------------------------------------- cases

HOSTILE_NAMES = ["Weird Name", "sélect", "from", "__param_0", "K", 'a"b']


def _colspec(rng: random.Random, depth: int = 0, static: bool = False):
    """One column's STORAGE type. Structs nest (a lane is addressed by its
    full ordered path, so depth is part of the surface, not decoration) and
    opaque columns appear at low weight — they must never block a build.

    `static` widens the vocabulary by the STATIC-ONLY types (the decimals):
    a decimal row column is opaque, so generating one there would only ever
    exercise the opaque path."""
    r = rng.random()
    if static and depth == 0 and 0.20 <= r < 0.32:
        return rng.choice(STATIC_ONLY) + ("?" if rng.random() < 0.5 else "")
    if depth < 2 and r < (0.16 if depth == 0 else 0.25):
        k = rng.randrange(1, 4)
        return Struct(
            tuple((f"f{j}", _colspec(rng, depth + 1)) for j in range(k)),
            nullable=rng.random() < 0.5,
        )
    if depth == 0 and r < 0.20:
        return rng.choice(OPAQUE)
    return rng.choice(SCALARS) + ("?" if rng.random() < 0.5 else "")


def _schema(
    rng: random.Random, n: int, hostile: bool = False, static: bool = False
) -> dict:
    out: dict = {}
    for i in range(n):
        name = f"c{i}"
        if hostile and rng.random() < 0.5:
            cand = rng.choice(HOSTILE_NAMES)
            if cand not in out:
                name = cand
        out[name] = _colspec(rng, static=static)
    return out


def _cell(rng: random.Random, spec) -> object:
    if isinstance(spec, Struct):
        if spec.nullable and rng.random() < 0.3:
            return None
        return {n: _cell(rng, s) for n, s in spec.fields}
    storage = spec.rstrip("?")
    if storage.startswith("decimal("):
        # A STATIC-ONLY type. Deliberately NOT in SEMANTIC, so the expression
        # grammar never binds one -- every operator over a decimal except a
        # comparison, a join and CAST-to-DOUBLE refuses by name until lattice
        # phase 5, and a grammar full of them would bury the signal in
        # REFUSED. Star expansion and the schema path are what
        # exercise the lane, which is exactly where it is observable.
        if spec.endswith("?") and rng.random() < 0.3:
            return None
        return _decimal_cell(rng, storage)
    if storage not in SEMANTIC:  # opaque: pyarrow fills these, not us
        return None
    return value(rng, SEMANTIC[storage], spec.endswith("?"), storage)


def _decimal_cell(rng: random.Random, storage: str) -> decimal.Decimal:
    """An exact payload at the declared (p,s), biased PAST 2^53 a third of
    the time -- that is where f64 and the exact value part ways, and the
    whole reason the lane is an i128."""
    p, s = (int(x) for x in storage[len("decimal(") : -1].split(","))
    hi = 10**p - 1
    if rng.random() < 0.35 and hi > 9_007_199_254_740_992:
        m = rng.randrange(9_007_199_254_740_992, hi + 1)
    else:
        m = rng.randrange(0, min(hi, 10**9) + 1)
    if rng.random() < 0.5:
        m = -m
    return decimal.Decimal(m).scaleb(-s)


def _rows(rng: random.Random, schema: dict, n: int) -> list[dict]:
    return [{c: _cell(rng, s) for c, s in schema.items()} for _ in range(n)]


def _equi_on(rng, env: Env, table: str, tschema: dict) -> Node | None:
    """key = table.col [AND residual] — the residual may carry CASE/AND/OR,
    which is where the ON path has diverged before (see
    tests/known_divergences/test_join_residual.py). Joins on a struct LANE
    too: a lane is a column, so `w.mean = c0` is an ordinary equi-key."""
    tleaves = leaves(tschema)
    pairs = [
        (rc, tc)
        for _, rc, rt in env.cols
        for tc, tt, _ in tleaves
        if tt == rt and rt in ("int", "float", "str")
    ]
    if not pairs:
        return None
    rc, tc = rng.choice(pairs)
    on: Node = Bin("=", Col(rc, None, "float"), Col(tc, table, "float"))
    if rng.random() < 0.4:
        resid_env = Env(
            env.cols + [(table, c, t) for c, t, _ in tleaves],
            env.udfs,
            env.tree,
        )
        on = Bin("AND", on, expr(rng, resid_env, "bool", 2))
    return on


def gen(seed: int) -> Case:
    """The whole case for `seed`: schemas, data, UDF/tree specs and the query
    AST. Seeded end to end, so the repro for any finding is its seed alone."""
    # The static-order twins ride a stream of their own (static_order_case
    # says why), so a seed they do not claim reaches `rng` untouched and
    # generates exactly what it generated before.
    planted = static_order_case(seed)
    if planted is not None:
        return planted

    rng = random.Random(seed)  # noqa: S311 — fuzzing, not crypto
    tags: list[str] = []

    hostile_ids = rng.random() < 0.08
    row_schema = _schema(rng, rng.randrange(1, 5))
    rows = _rows(rng, row_schema, rng.choice([0, 1, 2, 3, 4, 6, 8]))

    statics: dict[str, tuple[dict[str, str], list[dict]]] = {}
    for i in range(rng.choice([0, 0, 1, 1, 2])):
        name = f"s{i}" if not hostile_ids else rng.choice(["S0", "Dim Table", "s0"])
        if name.lower() in {k.lower() for k in statics}:
            continue  # DuckDB resolves table names case-insensitively
        sch = _schema(rng, rng.randrange(1, 4), hostile=hostile_ids, static=True)
        # 8% of statics are WIDE (50-200 rows, spread int values)
        # -- an unordered GROUP BY over 4 groups cannot show hash-order
        # variance; over 50+ it measurably does (12 orders / 12 connections).
        # %97 keeps every width honest (96 fits int8) and still gives ~97
        # distinct groups. Names are hygienic on purpose: this branch sits
        # inside gen(), whose own `rows` is the ROW TABLE.
        if rng.random() < 0.08:
            srows = _rows(rng, sch, rng.randrange(50, 201))
            for i, r in enumerate(srows):
                for c, s in sch.items():
                    if isinstance(s, str) and s.rstrip("?").startswith("int"):
                        r[c] = (i * 7919) % 97
            statics[name] = (sch, srows)
        else:
            statics[name] = (sch, _rows(rng, sch, rng.randrange(0, 6)))

    udfs: list[UdfSpec] = []
    if rng.random() < 0.30:
        for i in range(rng.randrange(1, 3)):
            takes = [(f"x{j}", rng.choice(TYPES)) for j in range(rng.randrange(1, 3))]
            r = rng.random()
            if r < 0.5:
                ret = ("scalar", rng.choice(TYPES))
            elif r < 0.8:
                k = rng.randrange(2, 4)
                ret = ("struct", [(f"f{j}", rng.choice(TYPES)) for j in range(k)])
            else:
                w = rng.choice([1, 2, 3])  # width-1 list must REFUSE
                ret = ("list", rng.choice(("int", "float")), w)
            udfs.append(
                UdfSpec(f"udf{i}", takes, ret, instances=rng.choice([0, 0, 2, 3]))
            )
        tags.append("udf")

    tree = None
    if rng.random() < 0.10:
        nf = rng.randrange(1, 4)
        tree = TreeSpec(
            kind=rng.choice(["dtr", "rf", "gbr"]),
            n_features=nf,
            int_features=tuple(i for i in range(nf) if rng.random() < 0.4),
            depth=rng.randrange(1, 5),
            instances=rng.choice([1, 2]),
        )
        tags.append("tree")

    env = Env([(None, c, t) for c, t, _ in leaves(row_schema)], udfs, tree)
    query = _query(rng, env, statics, tags, hostile_ids)

    shape = rng.choice([None] * 6 + ["map", "filter", "many"])
    output = rng.choice([None] * 4 + ["dict", "model"])
    if shape:
        tags.append(f"shape={shape}")
    return Case(seed, row_schema, rows, statics, udfs, tree, query, shape, output, tags)


def _items(rng, env: Env, n: int, aliased=True) -> list[tuple[Node, str | None]]:
    out = []
    for i in range(n):
        ty = rng.choice(TYPES + ("float",))
        alias = f"o{i}" if aliased or rng.random() < 0.8 else None
        out.append((expr(rng, env, ty, rng.randrange(1, 4)), alias))
    return out


def _query(rng, env: Env, statics, tags, hostile_ids) -> Q:
    """One query off a weighted menu of templates, retrying another template
    when this case cannot meet the one drawn (no statics, no bool column).

    APPENDS the template's name to `tags`, which is what makes a template
    that stopped firing at all visible in the coverage histogram.
    """
    r = rng.random()
    body: Sel
    ctes: list[tuple[str, Sel]] = []

    if r < 0.40 or (not statics and r < 0.70):  # plain spine
        tags.append("spine")
        body = Sel(_items(rng, env, rng.randrange(1, 4)), "__THIS__")
    elif r < 0.55 and statics:  # joins
        tags.append("join")
        body = Sel([], "__THIS__")
        jenv = env
        for name in list(statics)[: rng.randrange(1, len(statics) + 1)]:
            sch = statics[name][0]
            kind = rng.choice(["INNER", "LEFT"] * 4 + ["RIGHT", "FULL", "CROSS"])
            on = None if kind == "CROSS" else _equi_on(rng, jenv, name, sch)
            if on is None and kind != "CROSS":
                continue
            body.joins.append(Join(kind, name, on))
            jenv = Env(
                jenv.cols + [(name, c, t) for c, t, _ in leaves(sch)],
                env.udfs,
                env.tree,
            )
        body.items = _items(rng, jenv, rng.randrange(1, 4))
        env = jenv
    elif r < 0.60:  # selection-context forms
        # a nullable-bool conjunct with a sibling that can trap, nested under
        # OR / NOT or used as a CASE condition -- a class 4000 seeds of this
        # generator could not reach before this arm existed.
        bools = [c for _, c, tt in env.cols if tt == "bool" and "." not in c]
        ints = [c for _, c, tt in env.cols if tt == "int" and "." not in c]
        if not bools or not ints:
            return _query(rng, env, statics, tags, hostile_ids)
        tags.append("selctx")
        trap = Bin(
            ">",
            Bin("+", Lit(2**63 - 1, "int"), Col(rng.choice(ints), None, "int")),
            Lit(0, "int"),
        )
        conj = Bin("AND", Col(rng.choice(bools), None, "bool"), trap)
        true = Lit(True, "bool")
        m = rng.random()
        if m < 0.30:
            pred = Bin("OR", conj, true)
        elif m < 0.45:
            pred = Bin("AND", Bin("OR", conj, true), true)
        elif m < 0.60:
            pred = CaseW([(conj, true)], true)
        elif m < 0.80:
            # projection CASE: the condition is selection ctx there too
            body = Sel(
                [(CaseW([(conj, Lit(1, "int"))], Lit(2, "int")), "o0")], "__THIS__"
            )
            return Q([], body)
        else:
            pred = Un("NOT", conj)
        body = Sel(
            [(Col(rng.choice(bools), None, "bool"), "o0")], "__THIS__", where=pred
        )
        return Q([], body)
    elif r < 0.63:  # star forms
        tags.append("star")
        star = Star()
        m = rng.random()
        body = Sel([(star, None)], "__THIS__")
        if statics and m < 0.35:
            # a star over a JOINED STATIC: struct and opaque columns must
            # refuse by name under `s.*`/`*`, never expand as leaves or
            # drop -- unreachable before this arm existed.
            sname = rng.choice(list(statics))
            sch = statics[sname][0]
            on = _equi_on(rng, env, sname, sch)
            if on is not None:
                tags.append("static-star")
                body.joins.append(Join(rng.choice(["INNER", "LEFT"]), sname, on))
                if rng.random() < 0.6:
                    star.qualifier = sname
                if rng.random() < 0.4:
                    # EXCLUDE a random static column -- struct/opaque ones
                    # exercise the take-it-out-before-the-refusal path
                    star.exclude = [rng.choice(list(sch))]
        # EXCLUDE/REPLACE name a top-level output column, so a lane path is
        # not a legal target — pick from the undotted names only.
        flat = [c for c in env.cols if "." not in c[1]]
        if not star.exclude and not body.joins:
            if m < 0.3 and flat:
                star.exclude = [rng.choice(flat)[1]]
            elif m < 0.6 and flat:
                _, name, ty = rng.choice(flat)
                star.replace = [(expr(rng, env, ty, 1), name)]
            elif m < 0.8:
                star.columns_re = rng.choice(["c.*", "^c", "0$"])
    elif r < 0.73 and statics:  # CTE over a static, joined back
        tags.append("cte")
        sname = rng.choice(list(statics))
        sch, _ = statics[sname]
        cname = rng.choice(["agg", "Agg", "AGG"])  # names bind case-insensitively
        sleaves = leaves(sch)
        if not sleaves:
            return _query(rng, env, {}, tags, hostile_ids)  # opaque-only static
        nums = [c for c, t, _ in sleaves if t in ("int", "float")]
        if rng.random() < 0.5 and nums:
            gname, gty, _ = rng.choice(sleaves)
            agg, aty = rng.choice(list(AGGS.items()))
            csel = Sel(
                [
                    (Col(gname, None, "float"), "g"),
                    (Call(agg, [Col(rng.choice(nums), None, "float")]), "v"),
                ],
                sname,
                group_by=[Col(gname, None, "float")],
            )
            cschema = {"g": gty, "v": aty}
        else:
            # A lane projects out of a CTE under an alias, which is how a
            # struct column reaches code that only ever sees flat columns.
            csel = Sel(
                [(Col(c, None, t), c.replace(".", "_")) for c, t, _ in sleaves], sname
            )
            cschema = {c.replace(".", "_"): t for c, t, _ in sleaves}
        ctes.append((cname, csel))
        body = Sel([], "__THIS__")
        on = _equi_on(rng, env, cname.lower() if rng.random() < 0.3 else cname, cschema)
        if on is not None:
            body.joins.append(Join(rng.choice(["INNER", "LEFT"]), cname, on))
            env = Env(
                env.cols + [(cname, c, t) for c, t in cschema.items()],
                env.udfs,
                env.tree,
            )
        body.items = _items(rng, env, rng.randrange(1, 3))
    elif r < 0.81:  # sub-select in FROM
        tags.append("subq")
        inner = Sel(
            _items(rng, env, rng.randrange(1, 4)),
            "__THIS__",
            where=expr(rng, env, "bool", 2) if rng.random() < 0.5 else None,
        )
        oenv = Env(
            [(None, a, "float") for _, a in inner.items if a], env.udfs, env.tree
        )
        body = Sel(
            [
                (Col(a, None, "float"), f"o{i}")
                for i, (_, a) in enumerate(inner.items)
                if a
            ]
            or [(Lit(1, "int"), "o0")],
            None,
            sub=inner,
        )
        env = oenv
    elif r < 0.87 and statics:  # static-only aggregation
        tags.append("static_agg")
        sname = rng.choice(list(statics))
        sch, _ = statics[sname]
        sleaves = leaves(sch)
        if not sleaves:
            return _query(rng, env, {}, tags, hostile_ids)  # opaque-only static
        nums = [c for c, t, _ in sleaves if t in ("int", "float")]
        col = rng.choice(nums) if nums else sleaves[0][0]
        agg, _ = rng.choice(list(AGGS.items()))
        body = Sel([(Call(agg, [Col(col, None, "float")]), "v")], sname)
        if rng.random() < 0.5:
            g = rng.choice(sleaves)[0]
            body.items.insert(0, (Col(g, None, "float"), "g"))
            body.group_by = [Col(g, None, "float")]
    else:  # struct_pack projection
        tags.append("struct")
        k = rng.randrange(1, 4)
        fields = [(f"f{i}", expr(rng, env, rng.choice(TYPES), 2)) for i in range(k)]
        body = Sel([(StructPack(fields), "s")], "__THIS__")

    if body.frm == "__THIS__" and rng.random() < 0.45:
        body.where = expr(rng, env, "bool", rng.randrange(1, 4))

    # hostile clause / modifier — constructs the engine must REFUSE, never
    # parse and drop
    h = rng.random()
    if h < 0.10:
        tags.append("hostile_clause")
        pick = rng.randrange(6)
        if pick == 0:
            body.distinct = True
        elif pick == 1 and body.items and body.items[0][1]:
            body.order_by = body.items[0][1]
        elif pick == 2:
            body.limit = rng.randrange(0, 4)
        elif pick == 3:
            body.qualify = Lit(True, "bool")
        elif pick == 4:
            body.top = 2
        else:
            body.fetch = 1
    elif h < 0.16:
        tags.append("hostile_modifier")
        calls = [n for it in body.items for n in _walk(it[0]) if isinstance(n, Call)]
        if calls:
            rng.choice(calls).modifier = rng.choice(["over", "filter", "ignore_nulls"])
        else:
            body.items.append((Call("abs", [Lit(1.0, "float")], modifier="over"), "hm"))
    return Q(ctes, body)


def _walk(n: Node):
    yield n
    for k in n.kids():
        yield from _walk(k)


# The salt for the static-order stream. Any constant does; what matters is
# that the stream is the seed's OWN and separate from `gen`'s, so the seeds
# it does not claim generate byte-for-byte what they generated before.
_ORDER_STREAM = 0x71E5


def static_order_case(seed: int) -> Case | None:
    """The static-tables-only ORDER BY twins, or None when this seed is not
    one of them: the same query over a sort key that repeats (REFUSED) and
    over one that does not (AGREE).

    Planted because the grammar cannot reach either: a generated ORDER BY is
    always the FIRST output item's alias, which in a static-only case is the
    group key (unique by construction) or the lone aggregate of a one-row
    result -- so no generated case can tie, and none can be a witness that a
    non-tie still serves. Measured over seeds 0-39999: 28 static-only ORDER
    BY cases, no tie among them.

    Both twins matter. The tie one is the rule; the unique one is the guard
    against paying for it with an over-refusal, which the oracle turns into a
    finding rather than a quiet REFUSED.
    """
    aux = random.Random(seed ^ _ORDER_STREAM)  # noqa: S311 -- fuzzing
    r = aux.random()
    if r >= 0.02:
        return None
    ties = r < 0.01
    n = aux.randrange(3, 7)
    vals = aux.sample(range(1, 40), n)
    if ties:
        # one group totals what another already does, and the ORDER BY says
        # nothing about which of the two comes first
        vals[aux.randrange(1, n)] = vals[0]
    rows = [{"g": f"g{i}", "v": v} for i, v in enumerate(vals)]
    aux.shuffle(rows)  # the tie apart in scan order as often as beside it
    q = Q(
        [],
        Sel(
            [
                (Col("g", None, "str"), "o"),
                (Call("sum", [Col("v", None, "int")]), "t"),
            ],
            "ties",
        ),
    )
    q.body.group_by = [Col("g", None, "str")]
    q.body.order_by = "t"
    return Case(
        seed,
        {"k": "int"},
        [{"k": 1}],
        {"ties": ({"g": "string", "v": "int64"}, rows)},
        [],
        None,
        q,
        None,
        None,
        ["static_tie_order" if ties else "static_tie_unique"],
    )


def planted_over_case() -> Case:
    """The planted case that must never come back AGREE: `abs(k) OVER ()` is
    a call-node modifier on a scalar call, which DuckDB refuses. Confit
    refuses it as well, so the verdict is REFUSED — and were the modifier
    ever parsed and dropped instead, the same case would come back
    DIVERGE_BUILD. The smoke test accepts either, because what it really
    watches is that the pipeline still SEES a silent drop."""
    q = Q(
        [],
        Sel([(Call("abs", [Col("k", None, "int")], modifier="over"), "c")], "__THIS__"),
    )
    return Case(
        -1,
        {"k": "int"},
        [{"k": 1}, {"k": -2}],
        {},
        [],
        None,
        q,
        None,
        None,
        ["planted_over"],
    )
