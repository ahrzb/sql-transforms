//! Canonical text form of a [`Program`]. The inverse of [`parse`]; every
//! choice here (names, spacing, literal forms) is pinned by the round-trip
//! property, so change both sides together or not at all.
//!
//! [`parse`]: super::parse

use std::fmt::Write;

use super::{Block, ColTy, Inst, Lit, Program, StaticTy, Term, Ty, Value};

pub fn print(p: &Program) -> String {
    let mut s = String::new();
    for (i, st) in p.statics.iter().enumerate() {
        let _ = write!(s, "static @{i}: ");
        match st {
            StaticTy::Scalar(ct) => {
                let _ = writeln!(s, "scalar<{}>", col_ty(*ct));
            }
            StaticTy::Map { keys, values } => {
                let _ = writeln!(s, "map({}) -> ({})", tys(keys), tys(values));
            }
        }
    }
    if !p.statics.is_empty() {
        s.push('\n');
    }
    let _ = writeln!(
        s,
        "fn {}(in: {}, out: {}) {{",
        p.name,
        batch(&p.in_cols),
        batch(&p.out_cols)
    );
    for (bi, b) in p.blocks.iter().enumerate() {
        let _ = write!(s, "b{bi}");
        if !b.params.is_empty() {
            let params: Vec<String> = b
                .params
                .iter()
                .map(|(v, t)| format!("{}: {}", val(*v), t.name()))
                .collect();
            let _ = write!(s, "({})", params.join(", "));
        }
        s.push_str(":\n");
        for inst in &b.insts {
            s.push_str("  ");
            print_inst(&mut s, p, inst);
            s.push('\n');
        }
        s.push_str("  ");
        print_term(&mut s, b);
        s.push('\n');
    }
    s.push_str("}\n");
    s
}

fn print_inst(s: &mut String, p: &Program, inst: &Inst) {
    let dsts = inst.dsts();
    if !dsts.is_empty() {
        let names: Vec<String> = dsts.iter().map(|v| val(*v)).collect();
        let _ = write!(s, "{} = ", names.join(", "));
    }
    match inst {
        Inst::Const { lit, .. } => match lit {
            Lit::I1(b) => {
                let _ = write!(s, "const.i1 {b}");
            }
            Lit::I64(i) => {
                let _ = write!(s, "const.i64 {i}");
            }
            Lit::F64(f) => {
                let _ = write!(s, "const.f64 {}", f64_text(*f));
            }
            Lit::Str(t) => {
                let _ = write!(s, "const.str {}", quote(t));
            }
        },
        Inst::Bin { op, a, b, .. } => {
            let _ = write!(s, "{} {}, {}", op.name(), val(*a), val(*b));
        }
        Inst::Cmp { pred, ty, a, b, .. } => {
            let prefix = match ty {
                Ty::I64 => "icmp",
                Ty::F64 => "fcmp",
                Ty::Str => "scmp",
                // Unreachable in verified programs; printed anyway so a bad
                // program still prints for diagnostics.
                Ty::I1 => "icmp",
            };
            let _ = write!(s, "{prefix}.{} {}, {}", pred.name(), val(*a), val(*b));
        }
        Inst::Not { a, .. } => {
            let _ = write!(s, "not {}", val(*a));
        }
        Inst::Select { cond, a, b, .. } => {
            let _ = write!(s, "select {}, {}, {}", val(*cond), val(*a), val(*b));
        }
        Inst::Itof { a, .. } => {
            let _ = write!(s, "itof {}", val(*a));
        }
        Inst::Ftoi { mode, a, .. } => {
            let m = match mode {
                super::RoundMode::Trunc => "trunc",
                super::RoundMode::Round => "round",
            };
            let _ = write!(s, "ftoi.{m} {}", val(*a));
        }
        Inst::Itos { a, .. } => {
            let _ = write!(s, "itos {}", val(*a));
        }
        Inst::Ftos { a, .. } => {
            let _ = write!(s, "ftos {}", val(*a));
        }
        Inst::StoiOpt { a, .. } => {
            let _ = write!(s, "stoi.opt {}", val(*a));
        }
        Inst::StofOpt { a, .. } => {
            let _ = write!(s, "stof.opt {}", val(*a));
        }
        Inst::Sconcat { a, b, .. } => {
            let _ = write!(s, "sconcat {}, {}", val(*a), val(*b));
        }
        Inst::Str1 { op, a, .. } => {
            let _ = write!(s, "{} {}", op.name(), val(*a));
        }
        Inst::Strim { side, a, chars, .. } => {
            let _ = write!(s, "strim.{} {}, {}", side.name(), val(*a), val(*chars));
        }
        Inst::Ssubstr { a, start, len, .. } => match len {
            Some(len) => {
                let _ = write!(s, "ssubstr {}, {}, {}", val(*a), val(*start), val(*len));
            }
            None => {
                let _ = write!(s, "ssubstr.rest {}, {}", val(*a), val(*start));
            }
        },
        Inst::Num1 { op, a, .. } => {
            let _ = write!(s, "{} {}", op.name(), val(*a));
        }
        Inst::Load { col, .. } => {
            let _ = write!(s, "load in.{}", col_name(p, false, *col));
        }
        Inst::LoadOpt { col, .. } => {
            let _ = write!(s, "load.opt in.{}", col_name(p, false, *col));
        }
        Inst::Store { col, val: v } => {
            let _ = write!(s, "store out.{}, {}", col_name(p, true, *col), val(*v));
        }
        Inst::StoreOpt { col, flag, val: v } => {
            let _ = write!(
                s,
                "store.opt out.{}, {}, {}",
                col_name(p, true, *col),
                val(*flag),
                val(*v)
            );
        }
        Inst::Probe {
            static_id, keys, ..
        } => {
            let ks: Vec<String> = keys.iter().map(|k| val(*k)).collect();
            let _ = write!(s, "probe @{static_id}, {}", ks.join(", "));
        }
        Inst::Sload { static_id, .. } => {
            let _ = write!(s, "sload @{static_id}");
        }
        Inst::SloadOpt { static_id, .. } => {
            let _ = write!(s, "sload.opt @{static_id}");
        }
    }
}

fn print_term(s: &mut String, b: &Block) {
    match &b.term {
        Term::Jump { to, args } => {
            let _ = write!(s, "jump {}", target(to.0, args));
        }
        Term::Brif {
            cond,
            then_to,
            then_args,
            else_to,
            else_args,
        } => {
            let _ = write!(
                s,
                "brif {}, {}, {}",
                val(*cond),
                target(then_to.0, then_args),
                target(else_to.0, else_args)
            );
        }
        Term::Emit => s.push_str("emit"),
        Term::Skip => s.push_str("skip"),
        Term::Trap { msg } => {
            let _ = write!(s, "trap {}", quote(msg));
        }
    }
}

fn target(block: u32, args: &[Value]) -> String {
    if args.is_empty() {
        format!("b{block}")
    } else {
        let a: Vec<String> = args.iter().map(|v| val(*v)).collect();
        format!("b{block}({})", a.join(", "))
    }
}

fn val(v: Value) -> String {
    format!("%v{}", v.0)
}

fn tys(list: &[Ty]) -> String {
    list.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")
}

fn col_ty(ct: ColTy) -> String {
    if ct.nullable {
        format!("{}?", ct.ty.name())
    } else {
        ct.ty.name().to_string()
    }
}

fn batch(cols: &[super::Col]) -> String {
    let inner: Vec<String> = cols
        .iter()
        .map(|c| format!("{}: {}", ident_or_quoted(&c.name), col_ty(c.ty)))
        .collect();
    format!("batch{{{}}}", inner.join(", "))
}

fn col_name(p: &Program, out: bool, idx: u32) -> String {
    let cols = if out { &p.out_cols } else { &p.in_cols };
    match cols.get(idx as usize) {
        Some(c) => ident_or_quoted(&c.name),
        // Out-of-range in an unverified program; keep printing diagnosable.
        None => format!("\"<bad col {idx}>\""),
    }
}

/// Column names print bare when they are identifiers, quoted otherwise
/// (SQL-derived output names like `COALESCE(a, b)` survive). Function names
/// are NOT routed through this: the verifier requires them to be
/// identifiers, so the printer writes them raw.
fn ident_or_quoted(name: &str) -> String {
    if is_ident(name) {
        name.to_string()
    } else {
        quote(name)
    }
}

pub(super) fn is_ident(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `f64` text form: `{:?}` for finite values (guaranteed shortest
/// round-trip), dedicated tokens for the specials. NaN payloads collapse to
/// the single canonical `nan`.
pub(super) fn f64_text(f: f64) -> String {
    if f.is_nan() {
        "nan".to_string()
    } else if f == f64::INFINITY {
        "inf".to_string()
    } else if f == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{f:?}")
    }
}

pub(super) fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
