//! One stage's repeated pure subexpressions, computed once per row.
//!
//! A struct SQL function whose every field repeats one subexpression (the
//! row norm of a Normalizer: lane j is `x_j / g(norm(x))`) binds to n lanes
//! that each carry the whole norm, so the projection is quadratic in the
//! row and so was the code built from it. This pass hash-conses a stage's
//! projection into a DAG and names each subexpression that would be
//! evaluated more than once and CANNOT TRAP ([`can_trap`]: its kinds
//! are all on the trap-free allowlist, which also keeps out everything
//! impure — extern calls, tree models, `error()`). Lowering evaluates each
//! named one once per row reaching the projection, just before the first
//! item that reads it, and every occurrence reads the value
//! ([`SKind::Shared`]).
//!
//! Evaluating such a value earlier than its occurrence, or where its CASE
//! arm is not taken, is invisible: it cannot trap and has no effect. A
//! subexpression that can trap is never named, so every trap stays where
//! the query put it.
//!
//! One kind of trap is evaluated less often: a check, an item of a
//! [`SKind::Seq`] kept only for its traps (a sibling field of a struct
//! that a field read still evaluates), that an earlier item of the
//! projection already ran on every row that reaches this one. The same
//! expression on the same row traps again exactly where it trapped before,
//! and there the row stopped, so the later check is dropped
//! ([`repeated_checks`]). Every field read of a struct whose first field
//! guards the inputs carried that guard: W reads of a guard over T inputs
//! built and ran W copies of it.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use super::ir::{CmpPred, Lit};
use super::plan::{can_trap, SExpr, SKind};

/// A stage projection with its shared subexpressions named.
pub struct Shared {
    /// The named subexpressions, each reading only earlier ones: lowering
    /// evaluates each just before the first item that reads it.
    pub defs: Vec<SExpr>,
    /// The projection items, reading the defs through [`SKind::Shared`].
    pub items: Vec<SExpr>,
}

/// The projection's shared subexpressions, or `None` when nothing is
/// evaluated twice. `lets` are the values the items read through
/// [`SKind::Let`] (`frontend/lets.rs`): each is interned once, where it is
/// read, so the DAG is the one the text spelled out would give.
pub fn share(items: &[SExpr], lets: &[SExpr]) -> Option<Shared> {
    let mut dag = Dag::default();
    for e in lets {
        let id = dag.intern(e.clone());
        dag.lets.push(id);
    }
    let roots: Vec<u32> = items.iter().map(|e| dag.intern(e.clone())).collect();
    let n = dag.nodes.len();
    let dropped = repeated_checks(&dag, &roots);

    // Where each node is evaluated. An occurrence is GATED by the CASE
    // positions on its path: a CASE evaluates its first condition on every
    // row, and a later condition, an arm or the default only when the
    // conditions before it went one way (`Dag::gate`). Conditions are
    // hash-consed, so CASEs on the same conditions gate alike: every field
    // read of a `null_when` function sits behind its own `CASE WHEN c THEN
    // NULL ELSE ..`, all on one `c`. Per node, `occ` counts evaluations
    // under each gate (the empty gate is unconditional), once per
    // occurrence under parents that are themselves evaluated.
    //
    // A node is shared when computing it up front costs no row anything
    // and saves one: evaluated unconditionally and more than once, or
    // twice behind one gate (a row that opens the gate evaluates it twice;
    // one that does not pays for it once). A subtree held only by
    // different CASE arms stays lazy: hoisting it ran every instance arm
    // of a catalog step on every row. Parents are interned after their
    // children, so a descending sweep sees every parent of a node first.
    let mut occ: Vec<HashMap<Vec<u64>, u64>> = vec![HashMap::new(); n];
    for &r in &roots {
        *occ[r as usize].entry(Vec::new()).or_default() += 1;
    }
    let mut shared = vec![false; n];
    for id in (0..n).rev() {
        let node = &dag.nodes[id];
        let here = std::mem::take(&mut occ[id]);
        let total: u64 = here.values().fold(0, |a, &b| a.saturating_add(b));
        let must = here.get(&Vec::new()).copied().unwrap_or(0);
        let gated_twice = here.iter().any(|(g, &k)| !g.is_empty() && k >= 2);
        shared[id] = ((must >= 1 && total >= 2) || gated_twice)
            && node.free
            && dag.worth_sharing(node);
        let here: Vec<(Vec<u64>, u64)> = if shared[id] {
            vec![(Vec::new(), 1)]
        } else {
            here.into_iter().collect()
        };
        for (i, &c) in node.children.iter().enumerate() {
            if dropped.contains(&(id as u32, i)) {
                continue;
            }
            let step = dag.gate(node, i);
            for (g, k) in &here {
                let mut g = g.clone();
                if let Some(step) = step {
                    g.push(step);
                }
                let slot = occ[c as usize].entry(g).or_default();
                *slot = slot.saturating_add(*k);
            }
        }
    }
    if !shared.iter().any(|s| *s) && lets.is_empty() && dropped.is_empty() {
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
        dropped: &dropped,
    };
    let defs: Vec<SExpr> = order.iter().map(|&id| build.expr(id, true)).collect();
    let items: Vec<SExpr> = roots.iter().map(|&r| build.expr(r, false)).collect();

    Some(Shared { defs, items })
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
    /// Per let, its node.
    lets: Vec<u32>,
}

impl Dag {
    /// The gate step child `i` of `node` adds (see `share`), or `None`
    /// when every evaluation of the node evaluates it. A CASE's children run
    /// condition 0, result 0, condition 1, ..., the default: condition `k`
    /// runs when conditions `0..k` were not TRUE, result `k` when condition
    /// `k` was, the default when none was. The step names those conditions
    /// (by node id) and which of the three it is.
    fn gate(&self, node: &Node, i: usize) -> Option<u64> {
        let SKind::Case { arms, .. } = &node.shallow.kind else {
            return None;
        };
        if i == 0 {
            return None;
        }
        let n_arms = arms.len();
        let (upto, role) = if i >= 2 * n_arms {
            (n_arms, 2u8)
        } else if i % 2 == 1 {
            (i / 2 + 1, 1u8)
        } else {
            (i / 2, 0u8)
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        role.hash(&mut h);
        node.children
            .iter()
            .step_by(2)
            .take(upto)
            .for_each(|c| c.hash(&mut h));
        Some(h.finish())
    }

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
        if let SKind::Let(k) = e.kind {
            return self.lets[k as usize];
        }
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
            Lit::I128(v) => v.hash(&mut h),
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
    /// The checks [`repeated_checks`] drops.
    dropped: &'a HashSet<(u32, usize)>,
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
        if let SKind::Seq { pick, .. } = node.shallow.kind {
            if (0..node.children.len()).any(|i| self.dropped.contains(&(id, i))) {
                let mut items = Vec::new();
                let mut at = 0;
                for (i, &cid) in node.children.iter().enumerate() {
                    if self.dropped.contains(&(id, i)) {
                        continue;
                    }
                    if i == pick {
                        at = items.len();
                    }
                    items.push(self.expr(cid, false));
                }
                if items.len() == 1 {
                    return items.pop().expect("the picked item");
                }
                return SExpr {
                    kind: SKind::Seq { items, pick: at },
                    ty: node.shallow.ty,
                    nullable: node.shallow.nullable,
                };
            }
        }
        let mut e = node.shallow.clone();
        for (c, &cid) in e.children_mut().into_iter().zip(&node.children) {
            *c = self.expr(cid, false);
        }
        e
    }
}

/// The checks to drop, as (Seq node, item index): each check an earlier
/// run of the same expression already covers on every row that reaches it
/// (see the module doc). Items run in order, and so does each item's tree,
/// so a walk in that order meets the earlier run first. A run counts only
/// where it is certain: reached from its item through Seq items and CASE
/// positions ([`Gates`]) behind conditions that call no UDF, never through
/// an operator that may skip an operand (AND and OR in a condition skip
/// their right one). A later check is redundant where an earlier certain
/// run's CASE positions are a prefix of its own: every row that reaches it
/// reached that run.
///
/// Only a Seq node that stands in one place drops a check, so that the
/// decision, which holds for the node, holds for its one occurrence; and a
/// check that calls a UDF stays, since a second call may answer otherwise.
fn repeated_checks(dag: &Dag, roots: &[u32]) -> HashSet<(u32, usize)> {
    let mut w = Walk {
        dag,
        gates: Gates::default(),
        seqs: HashMap::new(),
        ran: HashMap::new(),
        dropped: HashSet::new(),
        calls: HashMap::new(),
    };
    for &r in roots {
        w.count(r);
    }
    let mut gate = Vec::new();
    for &r in roots {
        w.walk(r, &mut gate, true);
    }
    w.dropped
}

/// The CASE positions on a path, each named exactly (not by a hash), as
/// `Dag::gate` names them: by the conditions before it, and by whether it
/// is a later condition, the result of the last of them, or the default.
#[derive(Default)]
struct Gates {
    /// Per sequence of conditions, by (the sequence one shorter, its last
    /// condition): its number, from 1 (0 is the empty sequence).
    prefixes: HashMap<(u32, u32), u32>,
    /// Per (role, sequence of conditions): the position's number.
    ids: HashMap<(u8, u32), u32>,
}

impl Gates {
    fn longer(&mut self, prefix: u32, cond: u32) -> u32 {
        let next = self.prefixes.len() as u32 + 1;
        *self.prefixes.entry((prefix, cond)).or_insert(next)
    }

    fn id(&mut self, role: u8, prefix: u32) -> u32 {
        let next = self.ids.len() as u32;
        *self.ids.entry((role, prefix)).or_insert(next)
    }
}

struct Walk<'a> {
    dag: &'a Dag,
    gates: Gates,
    /// Per Seq node that can trap: in how many places it stands.
    seqs: HashMap<u32, u32>,
    /// Per check node: the CASE positions of each certain run of it.
    ran: HashMap<u32, Vec<Vec<u32>>>,
    dropped: HashSet<(u32, usize)>,
    /// Per node: whether its tree calls a UDF.
    calls: HashMap<u32, bool>,
}

impl Walk<'_> {
    /// Count the places of each Seq node under `id` that can trap.
    fn count(&mut self, id: u32) {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || {
                let node = &self.dag.nodes[id as usize];
                if node.free {
                    return;
                }
                if let SKind::Seq { .. } = node.shallow.kind {
                    *self.seqs.entry(id).or_default() += 1;
                }
                for &c in &node.children {
                    self.count(c);
                }
            },
        )
    }

    fn walk(&mut self, id: u32, gate: &mut Vec<u32>, certain: bool) {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || self.walk_here(id, gate, certain),
        )
    }

    fn walk_here(&mut self, id: u32, gate: &mut Vec<u32>, certain: bool) {
        let dag = self.dag;
        let node = &dag.nodes[id as usize];
        if node.free {
            // Nothing in it traps.
            return;
        }
        match &node.shallow.kind {
            SKind::Seq { pick, .. } => {
                let once = self.seqs.get(&id) == Some(&1);
                for (i, &c) in node.children.iter().enumerate() {
                    if i == *pick {
                        self.walk(c, gate, certain);
                        continue;
                    }
                    if once && self.ran_before(c, gate) && !self.calls_udf(c) {
                        self.dropped.insert((id, i));
                        continue;
                    }
                    self.walk(c, gate, certain);
                    if certain {
                        self.ran.entry(c).or_default().push(gate.clone());
                    }
                }
            }
            SKind::Case { arms, .. } => {
                let n_arms = arms.len();
                // The conditions before the child, and whether one calls a
                // UDF: one that does may answer otherwise the next time, so
                // a run behind it is not certain to come again.
                let mut prefix = 0;
                let mut udf = false;
                for (i, &c) in node.children.iter().enumerate() {
                    if i == 0 {
                        // The first condition runs wherever the CASE does.
                        self.walk(c, gate, certain);
                        continue;
                    }
                    let role = if i >= 2 * n_arms {
                        2
                    } else if i % 2 == 1 {
                        let cond = node.children[i - 1];
                        prefix = self.gates.longer(prefix, cond);
                        udf = udf || self.calls_udf(cond);
                        1
                    } else {
                        0
                    };
                    gate.push(self.gates.id(role, prefix));
                    self.walk(c, gate, certain && !udf);
                    gate.pop();
                }
            }
            _ => {
                for &c in &node.children {
                    self.walk(c, gate, false);
                }
            }
        }
    }

    /// Whether a certain run of `check` came before, under a prefix of
    /// `gate`.
    fn ran_before(&self, check: u32, gate: &[u32]) -> bool {
        self.ran
            .get(&check)
            .is_some_and(|runs| runs.iter().any(|g| gate.starts_with(g)))
    }

    fn calls_udf(&mut self, id: u32) -> bool {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || {
                if let Some(&b) = self.calls.get(&id) {
                    return b;
                }
                let node = &self.dag.nodes[id as usize];
                let mut b = matches!(node.shallow.kind, SKind::ExternCall { .. });
                for &c in &node.children {
                    b = b || self.calls_udf(c);
                }
                self.calls.insert(id, b);
                b
            },
        )
    }
}
