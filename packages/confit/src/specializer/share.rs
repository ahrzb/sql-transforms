//! One stage's repeated pure subexpressions, computed once per row.
//!
//! A struct SQL function whose every field repeats one subexpression (the
//! row norm of a Normalizer: lane j is `x_j / g(norm(x))`) binds to n lanes
//! that each carry the whole norm, so the projection is quadratic in the
//! row and so was the code built from it. This pass hash-conses a stage's
//! projection into a DAG and names each subexpression that would be
//! evaluated more than once and CANNOT TRAP ([`can_trap`]: its kinds
//! are all on the trap-free allowlist, which also keeps out everything
//! impure — extern calls, tree models, `error()`). Lowering evaluates the
//! named ones first, once per row reaching the projection, and every
//! occurrence reads the value ([`SKind::Shared`]).
//!
//! Evaluating such a value earlier than its occurrence, or where its CASE
//! arm is not taken, is invisible: it cannot trap and has no effect. A
//! subexpression that can trap is never named, so every trap stays where
//! the query put it, evaluated as often as before.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::ir::{CmpPred, Lit};
use super::plan::{can_trap, SExpr, SKind};

/// A stage projection with its shared subexpressions named.
pub struct Shared {
    /// The named subexpressions, each reading only earlier ones: lowering
    /// evaluates them in this order before the first item.
    pub defs: Vec<SExpr>,
    /// The projection items, reading the defs through [`SKind::Shared`].
    pub items: Vec<SExpr>,
    /// Per def, the last evaluation step that reads it: step `k` is def `k`
    /// for `k < defs.len()`, else item `k - defs.len()`. A def's value is
    /// dead after that step.
    pub last_use: Vec<usize>,
}

/// The projection's shared subexpressions, or `None` when nothing is
/// evaluated twice.
pub fn share(items: &[SExpr]) -> Option<Shared> {
    let mut dag = Dag::default();
    let roots: Vec<u32> = items.iter().map(|e| dag.intern(e.clone())).collect();
    let n = dag.nodes.len();

    // How many times each node would be evaluated: once per occurrence
    // under parents that are themselves evaluated, and once in all when the
    // node is shared. Parents are interned after their children, so a
    // descending sweep sees every parent of a node before the node.
    let mut occ = vec![0u64; n];
    for &r in &roots {
        occ[r as usize] += 1;
    }
    let mut shared = vec![false; n];
    for id in (0..n).rev() {
        let node = &dag.nodes[id];
        shared[id] = occ[id] >= 2 && node.free && dag.worth_sharing(node);
        let runs = if shared[id] { 1 } else { occ[id] };
        if runs == 0 {
            continue;
        }
        for &c in &node.children {
            occ[c as usize] = occ[c as usize].saturating_add(runs);
        }
    }
    if !shared.iter().any(|s| *s) {
        return None;
    }

    // Children before parents: ascending id is a topological order.
    let mut def_of = vec![u32::MAX; n];
    let mut order = Vec::new();
    for id in 0..n {
        if shared[id] {
            def_of[id] = order.len() as u32;
            order.push(id as u32);
        }
    }
    let build = Build {
        dag: &dag,
        def_of: &def_of,
    };
    let mut defs: Vec<SExpr> = order.iter().map(|&id| build.expr(id, true)).collect();
    let mut items: Vec<SExpr> = roots.iter().map(|&r| build.expr(r, false)).collect();

    let mut last_use = vec![0usize; defs.len()];
    for (step, e) in defs.iter_mut().chain(items.iter_mut()).enumerate() {
        mark_reads(e, step, &mut last_use);
    }
    Some(Shared {
        defs,
        items,
        last_use,
    })
}

fn mark_reads(e: &mut SExpr, step: usize, last_use: &mut [usize]) {
    stacker::maybe_grow(
        super::frontend::RED_ZONE,
        super::frontend::STACK_SEGMENT,
        || {
            if let SKind::Shared(k) = e.kind {
                last_use[k as usize] = step;
                return;
            }
            for c in e.children_mut() {
                mark_reads(c, step, last_use);
            }
        },
    )
}

/// The fewest nodes a shared subexpression has (see `Dag::worth_sharing`).
const MIN_SHARED_SIZE: u32 = 6;

struct Node {
    /// The node with every child replaced by `Shared(child id)`.
    shallow: SExpr,
    children: Vec<u32>,
    /// Nodes in the subtree, as written (saturating).
    size: u32,
    /// No node in this subtree can trap.
    free: bool,
}

#[derive(Default)]
struct Dag {
    nodes: Vec<Node>,
    index: HashMap<u64, Vec<u32>>,
}

impl Dag {
    /// A small subexpression is cheaper to evaluate again than to carry:
    /// a shared value rides every block transition until its last read, and
    /// a Normalizer lane's own CASE splits blocks, so sharing each lane's
    /// `coalesce(x_i)` (five nodes) made serving quadratic in the row. An
    /// equality against a constant stays in place so a CASE can still
    /// dispatch on it (`lower::dispatch_table` reads the condition's shape).
    fn worth_sharing(&self, node: &Node) -> bool {
        if node.size < MIN_SHARED_SIZE {
            return false;
        }
        match &node.shallow.kind {
            SKind::Col(_)
            | SKind::Slot(_)
            | SKind::StaticCol { .. }
            | SKind::Lit(_)
            | SKind::NullOf
            | SKind::JoinHit(_)
            | SKind::Shared(_) => false,
            SKind::Cmp {
                pred: CmpPred::Eq, ..
            } => !node
                .children
                .iter()
                .any(|&c| matches!(self.nodes[c as usize].shallow.kind, SKind::Lit(_))),
            _ => true,
        }
    }

    fn intern(&mut self, e: SExpr) -> u32 {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || self.intern_here(e),
        )
    }

    fn intern_here(&mut self, mut e: SExpr) -> u32 {
        let mut children = Vec::new();
        let mut free = true;
        let mut size = 1u32;
        for c in e.children_mut() {
            let (ty, nullable) = (c.ty, c.nullable);
            let child = std::mem::replace(
                c,
                SExpr {
                    kind: SKind::NullOf,
                    ty,
                    nullable,
                },
            );
            let id = self.intern(child);
            free &= self.nodes[id as usize].free;
            size = size.saturating_add(self.nodes[id as usize].size);
            children.push(id);
            *c = SExpr {
                kind: SKind::Shared(id),
                ty,
                nullable,
            };
        }
        // With every child a `Shared` placeholder (trap-free to
        // `can_trap`), this asks only whether the node itself can trap.
        free &= !can_trap(&e);
        let h = shallow_hash(&e, &children);
        let bucket = self.index.entry(h).or_default();
        if let Some(&id) = bucket.iter().find(|&&id| self.nodes[id as usize].shallow == e) {
            return id;
        }
        let id = self.nodes.len() as u32;
        bucket.push(id);
        self.nodes.push(Node {
            shallow: e,
            children,
            size,
            free,
        });
        id
    }
}

/// A hash consistent with `==` on shallow nodes: the kind, the type, the
/// children, and a leaf's payload. Kinds that differ only in an operator
/// collide and are told apart by `==`.
fn shallow_hash(e: &SExpr, children: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::mem::discriminant(&e.kind).hash(&mut h);
    e.ty.hash(&mut h);
    e.nullable.hash(&mut h);
    children.hash(&mut h);
    match &e.kind {
        SKind::Col(i) | SKind::Slot(i) | SKind::JoinHit(i) => i.hash(&mut h),
        SKind::StaticCol { join, col } => (join, col).hash(&mut h),
        SKind::Lit(l) => match l {
            Lit::I1(b) => b.hash(&mut h),
            Lit::I64(v) => v.hash(&mut h),
            // `Lit`'s equality: bitwise, except NaNs of one sign are equal.
            Lit::F64(v) if v.is_nan() => v.is_sign_negative().hash(&mut h),
            Lit::F64(v) => v.to_bits().hash(&mut h),
            Lit::Str(s) => s.hash(&mut h),
            Lit::Dec(v, p, s) => (v, p, s).hash(&mut h),
        },
        _ => {}
    }
    h.finish()
}

struct Build<'a> {
    dag: &'a Dag,
    def_of: &'a [u32],
}

impl Build<'_> {
    /// Node `id` as an expression: a read of its def when it is shared
    /// (unless `top`, which builds the def's own body).
    fn expr(&self, id: u32, top: bool) -> SExpr {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || self.expr_here(id, top),
        )
    }

    fn expr_here(&self, id: u32, top: bool) -> SExpr {
        let node = &self.dag.nodes[id as usize];
        let k = self.def_of[id as usize];
        if k != u32::MAX && !top {
            return SExpr {
                kind: SKind::Shared(k),
                ty: node.shallow.ty,
                nullable: node.shallow.nullable,
            };
        }
        let mut e = node.shallow.clone();
        for (c, &cid) in e.children_mut().into_iter().zip(&node.children) {
            *c = self.expr(cid, false);
        }
        e
    }
}
