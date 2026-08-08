//! Packed tree-ensemble scoring: the routine behind the `predict`
//! instruction. A sibling of `strip_accents.rs` — a large table-driven native
//! routine reached through one opcode, called identically by the interpreter
//! and by cranelift so the two cannot drift.
//!
//! One layout covers `DecisionTree*`, `RandomForest*` and `GradientBoosting*`,
//! and (same node shape) XGBoost and LightGBM. Nothing here knows about
//! sklearn: the caller hands over flat columns, which is what the Python side
//! builds from two Arrow batches. Confit never imports an ML library.
//!
//! # Layout
//!
//! Struct-of-arrays over all nodes of all trees of all models. A tree's nodes
//! are contiguous (`tree_span`), a model's trees are contiguous
//! (`model_span`), and a model id is a dense index into `model_span`.
//! Child pointers arrive tree-local (sklearn's `children_left`) and are
//! rebased to absolute node indices at build, so traversal is one indexed
//! load per level with no per-step offset arithmetic.

use super::Trap;

/// How a model combines its trees' leaf values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Agg {
    Sum,
    Mean,
}

/// The function applied to the aggregated score.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Link {
    Identity,
    Sigmoid,
}

/// The node table, one entry per row, grouped by `(model_id, tree_id)`.
/// `left`/`right` are tree-local node ids (`-1` on a leaf), matching what
/// every library emits.
pub struct NodeRows<'a> {
    pub model_id: &'a [i64],
    pub tree_id: &'a [i64],
    pub node_id: &'a [i64],
    pub feature: &'a [i32],
    pub threshold: &'a [f64],
    pub left: &'a [i32],
    pub right: &'a [i32],
    pub missing_left: &'a [bool],
    pub value: &'a [f64],
}

/// The per-model header table, one entry per model, `model_id` dense from 0.
pub struct ModelRows<'a> {
    pub model_id: &'a [i64],
    pub base: &'a [f64],
    pub agg: &'a [&'a str],
    pub link: &'a [&'a str],
}

/// A prepared ensemble. Every invariant `predict` relies on — dense model
/// ids, in-range features, terminating traversal, both-children-or-neither —
/// is established by [`TreeEnsemble::new`] and never re-checked per row.
#[derive(Debug)]
pub struct TreeEnsemble {
    feature: Vec<i32>, // -1 = leaf
    threshold: Vec<f64>,
    left: Vec<u32>, // absolute node index; unread on a leaf
    right: Vec<u32>,
    missing_left: Vec<bool>,
    value: Vec<f64>,
    tree_span: Vec<(u32, u32)>,  // tree  -> node range
    model_span: Vec<(u32, u32)>, // model -> tree range
    base: Vec<f64>,
    agg: Vec<Agg>,
    link: Vec<Link>,
    n_features: u32,
}

impl TreeEnsemble {
    /// Validate and pack. Every refusal names the offending row or field
    /// (P7): a build that gets this far cannot produce a wrong number at run
    /// for a structural reason.
    pub fn new(
        nodes: &NodeRows<'_>,
        models: &ModelRows<'_>,
        n_features: u32,
    ) -> Result<Self, String> {
        let nn = nodes.model_id.len();
        if [
            nodes.tree_id.len(),
            nodes.node_id.len(),
            nodes.feature.len(),
            nodes.threshold.len(),
            nodes.left.len(),
            nodes.right.len(),
            nodes.missing_left.len(),
            nodes.value.len(),
        ]
        .iter()
        .any(|l| *l != nn)
        {
            return Err("node columns must all have the same length".into());
        }
        let nm = models.model_id.len();
        if models.base.len() != nm || models.agg.len() != nm || models.link.len() != nm {
            return Err("model columns must all have the same length".into());
        }
        if nm == 0 {
            return Err("model set has no models".into());
        }
        for (i, id) in models.model_id.iter().enumerate() {
            if *id != i as i64 {
                return Err(format!(
                    "model ids must be dense from 0: header row {i} has model_id {id}"
                ));
            }
        }

        let mut agg = Vec::with_capacity(nm);
        let mut link = Vec::with_capacity(nm);
        for (m, (a, l)) in models.agg.iter().zip(models.link.iter()).enumerate() {
            agg.push(match *a {
                "sum" => Agg::Sum,
                "mean" => Agg::Mean,
                other => return Err(format!("model {m}: unknown agg '{other}'")),
            });
            link.push(match *l {
                "identity" => Link::Identity,
                "sigmoid" => Link::Sigmoid,
                other => return Err(format!("model {m}: unknown link '{other}'")),
            });
        }

        // Spans, from one forward scan: rows arrive grouped by model then by
        // tree, which is what the extractor emits and what keeps a model's
        // trees contiguous in memory.
        let mut tree_span: Vec<(u32, u32)> = Vec::new();
        let mut model_span = vec![(0u32, 0u32); nm];
        let mut seen = vec![false; nm];
        let mut i = 0usize;
        while i < nn {
            let mid = nodes.model_id[i];
            let m = usize::try_from(mid)
                .ok()
                .filter(|m| *m < nm)
                .ok_or_else(|| format!("node row {i}: model {mid} has no header row"))?;
            if seen[m] {
                return Err(format!("node rows for model {m} are not contiguous"));
            }
            seen[m] = true;
            let first_tree = tree_span.len() as u32;
            while i < nn && nodes.model_id[i] == mid {
                let tid = nodes.tree_id[i];
                let lo = i;
                while i < nn && nodes.model_id[i] == mid && nodes.tree_id[i] == tid {
                    i += 1;
                }
                tree_span.push((lo as u32, i as u32));
            }
            model_span[m] = (first_tree, tree_span.len() as u32);
        }
        if let Some(m) = seen.iter().position(|ok| !ok) {
            return Err(format!("model {m} has no nodes"));
        }

        // Per tree: dense node ids, leaf/split consistency, in-range features,
        // and children that strictly follow their parent. That last rule is
        // what makes the traversal loop provably terminate without a depth
        // counter — every library we target already emits nodes in that order,
        // and it rules out cycles by construction.
        //
        // `parented` counts parents, saturating at one, and the two ends of it
        // together are a COMPLETE tree check: given children that strictly
        // follow their parent, a table is a tree exactly when every non-root
        // node has exactly one parent — one parent each makes the parent
        // function total, and the ordering makes walking parents strictly
        // decrease, so every node has a unique path back to node 0.
        //
        // Zero parents is "unreachable from the root" (checked after the
        // loop); two is a shared child, a decision DAG rather than a tree
        // (checked in it). A DAG scores perfectly well — one path, still
        // terminating — but nothing we target emits one, so it means the
        // table is malformed, and rejecting only the zero case would be an
        // arbitrary place to stop (TASK-76).
        let mut left = vec![0u32; nn];
        let mut right = vec![0u32; nn];
        for &(lo, hi) in &tree_span {
            let (lo, hi) = (lo as usize, hi as usize);
            let len = hi - lo;
            let mut parented = vec![false; len];
            for k in 0..len {
                let i = lo + k;
                if nodes.node_id[i] != k as i64 {
                    return Err(format!(
                        "node row {i}: node id {} out of dense order (want {k})",
                        nodes.node_id[i]
                    ));
                }
                let f = nodes.feature[i];
                let (l, r) = (nodes.left[i], nodes.right[i]);
                if f < 0 {
                    if l != -1 || r != -1 {
                        return Err(format!(
                            "node row {i}: leaf (feature -1) with children {l}/{r}"
                        ));
                    }
                    continue;
                }
                if f as u32 >= n_features {
                    return Err(format!(
                        "node row {i}: feature {f} beyond the declared width {n_features}"
                    ));
                }
                if l < 0 || r < 0 {
                    return Err(format!("node row {i}: split node missing a child ({l}/{r})"));
                }
                for c in [l, r] {
                    let c = c as usize;
                    if c >= len {
                        return Err(format!(
                            "node row {i}: child {c} out of range (tree has {len} node(s))"
                        ));
                    }
                    if c <= k {
                        return Err(format!("node row {i}: child {c} must follow its parent {k}"));
                    }
                    if std::mem::replace(&mut parented[c], true) {
                        return Err(format!(
                            "node row {i}: child {c} already has a parent (not a tree)"
                        ));
                    }
                }
                left[i] = (lo + l as usize) as u32;
                right[i] = (lo + r as usize) as u32;
            }
            if let Some(k) = (1..len).find(|k| !parented[*k]) {
                return Err(format!(
                    "node row {}: unreachable from its tree's root",
                    lo + k
                ));
            }
        }

        Ok(TreeEnsemble {
            feature: nodes.feature.to_vec(),
            threshold: nodes.threshold.to_vec(),
            left,
            right,
            missing_left: nodes.missing_left.to_vec(),
            value: nodes.value.to_vec(),
            tree_span,
            model_span,
            base: models.base.to_vec(),
            agg,
            link,
            n_features,
        })
    }

    pub fn n_features(&self) -> u32 {
        self.n_features
    }

    pub fn n_models(&self) -> usize {
        self.model_span.len()
    }

    /// Score one row. `feats.len()` must equal `n_features` — the verifier
    /// and the prepare-time kind check guarantee it, so this is a debug
    /// assertion rather than a per-row branch.
    pub fn predict(&self, id: i64, feats: &[f64]) -> Result<f64, Trap> {
        debug_assert_eq!(feats.len(), self.n_features as usize);
        let m = usize::try_from(id)
            .ok()
            .filter(|m| *m < self.model_span.len())
            .ok_or_else(|| Trap(format!("predict: no model with id {id}")))?;
        let (t0, t1) = self.model_span[m];
        // Sequential in tree_span order. Measured against sklearn 1.9.0 /
        // numpy 2.5.1 over 2000 rows at 10/100/500 trees: NEITHER estimator
        // uses numpy's pairwise summation — both accumulate left-to-right in
        // tree order, and `arr.sum(axis=1)` disagrees with that on up to 1853
        // of 2000 rows (max 320 ULP). Sequential is what makes us bit-exact,
        // not a concession.
        //
        // Where the base term enters is NOT cosmetic, and it differs by mode:
        //  * Sum (boosted): `_raw_predict_init` SEEDS the accumulator with the
        //    init prediction, then each stage does `+= lr * leaf`. Adding the
        //    base afterwards instead diverges on up to 1365/2000 rows, 632 ULP.
        //  * Mean (forest): `y_hat` starts at 0.0, `+=` per tree, then `/= n`.
        let score = match self.agg[m] {
            Agg::Sum => {
                let mut acc = self.base[m];
                for t in t0..t1 {
                    acc += self.leaf_value(t as usize, feats);
                }
                acc
            }
            Agg::Mean => {
                let mut acc = 0.0f64;
                for t in t0..t1 {
                    acc += self.leaf_value(t as usize, feats);
                }
                // t1 > t0 always: a model with no nodes was refused at build.
                self.base[m] + acc / (t1 - t0) as f64
            }
        };
        Ok(match self.link[m] {
            Link::Identity => score,
            Link::Sigmoid => 1.0 / (1.0 + (-score).exp()),
        })
    }

    fn leaf_value(&self, t: usize, feats: &[f64]) -> f64 {
        let mut n = self.tree_span[t].0 as usize;
        loop {
            let f = self.feature[n];
            if f < 0 {
                return self.value[n];
            }
            let x = feats[f as usize];
            // NaN is tested FIRST: `x <= threshold` is false for NaN, which
            // would silently mean "go right" instead of the node's learned
            // direction (sklearn's tree_.missing_go_to_left).
            n = if x.is_nan() {
                if self.missing_left[n] {
                    self.left[n]
                } else {
                    self.right[n]
                }
            } else if x <= self.threshold[n] {
                self.left[n]
            } else {
                self.right[n]
            } as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owned node columns, so a test can name the one field it corrupts.
    #[derive(Clone)]
    struct Nodes {
        model_id: Vec<i64>,
        tree_id: Vec<i64>,
        node_id: Vec<i64>,
        feature: Vec<i32>,
        threshold: Vec<f64>,
        left: Vec<i32>,
        right: Vec<i32>,
        missing_left: Vec<bool>,
        value: Vec<f64>,
    }

    impl Nodes {
        /// A stump: `x0 <= 0.5` -> 1.0, else 2.0.
        fn stump() -> Nodes {
            Nodes {
                model_id: vec![0, 0, 0],
                tree_id: vec![0, 0, 0],
                node_id: vec![0, 1, 2],
                feature: vec![0, -1, -1],
                threshold: vec![0.5, 0.0, 0.0],
                left: vec![1, -1, -1],
                right: vec![2, -1, -1],
                missing_left: vec![false; 3],
                value: vec![0.0, 1.0, 2.0],
            }
        }

        fn empty() -> Nodes {
            Nodes {
                model_id: vec![],
                tree_id: vec![],
                node_id: vec![],
                feature: vec![],
                threshold: vec![],
                left: vec![],
                right: vec![],
                missing_left: vec![],
                value: vec![],
            }
        }

        /// Append a copy of the first tree under `(model_id, tree_id)`.
        fn plus_stump(mut self, model_id: i64, tree_id: i64) -> Nodes {
            let s = Nodes::stump();
            for k in 0..3 {
                self.model_id.push(model_id);
                self.tree_id.push(tree_id);
                self.node_id.push(s.node_id[k]);
                self.feature.push(s.feature[k]);
                self.threshold.push(s.threshold[k]);
                self.left.push(s.left[k]);
                self.right.push(s.right[k]);
                self.missing_left.push(s.missing_left[k]);
                self.value.push(s.value[k]);
            }
            self
        }

        fn build(
            &self,
            base: &[f64],
            agg: &[&str],
            link: &[&str],
            n_features: u32,
        ) -> Result<TreeEnsemble, String> {
            let ids: Vec<i64> = (0..base.len() as i64).collect();
            self.build_with_ids(&ids, base, agg, link, n_features)
        }

        fn build_with_ids(
            &self,
            ids: &[i64],
            base: &[f64],
            agg: &[&str],
            link: &[&str],
            n_features: u32,
        ) -> Result<TreeEnsemble, String> {
            TreeEnsemble::new(
                &NodeRows {
                    model_id: &self.model_id,
                    tree_id: &self.tree_id,
                    node_id: &self.node_id,
                    feature: &self.feature,
                    threshold: &self.threshold,
                    left: &self.left,
                    right: &self.right,
                    missing_left: &self.missing_left,
                    value: &self.value,
                },
                &ModelRows {
                    model_id: ids,
                    base,
                    agg,
                    link,
                },
                n_features,
            )
        }

        fn one(&self) -> TreeEnsemble {
            self.build(&[0.0], &["sum"], &["identity"], 1)
                .expect("fixture builds")
        }
    }

    /// Corrupt a stump and take the refusal message.
    fn refusal(mutate: impl Fn(&mut Nodes)) -> String {
        let mut n = Nodes::stump();
        mutate(&mut n);
        n.build(&[0.0], &["sum"], &["identity"], 1)
            .expect_err("must refuse")
    }

    #[test]
    fn threshold_comparison_sends_equal_left() {
        let m = Nodes::stump().one();
        assert_eq!(m.predict(0, &[0.0]).unwrap(), 1.0);
        // x <= threshold goes LEFT, so the boundary itself goes left.
        assert_eq!(m.predict(0, &[0.5]).unwrap(), 1.0);
        assert_eq!(m.predict(0, &[0.51]).unwrap(), 2.0);
    }

    /// sklearn's rule, measured on 1.9.0: a NaN feature takes the node's
    /// declared direction — right by default, left where the tree learned a
    /// missing branch. NOT a house rule, and never an error.
    #[test]
    fn nan_takes_the_nodes_declared_missing_direction() {
        assert_eq!(
            Nodes::stump().one().predict(0, &[f64::NAN]).unwrap(),
            2.0,
            "default is right"
        );

        let mut n = Nodes::stump();
        n.missing_left[0] = true;
        let m = n.one();
        assert_eq!(m.predict(0, &[f64::NAN]).unwrap(), 1.0, "missing_left routes left");
        assert_eq!(m.predict(0, &[9.0]).unwrap(), 2.0, "a present value still splits");
    }

    #[test]
    fn root_is_a_leaf() {
        let n = Nodes {
            model_id: vec![0],
            tree_id: vec![0],
            node_id: vec![0],
            feature: vec![-1],
            threshold: vec![0.0],
            left: vec![-1],
            right: vec![-1],
            missing_left: vec![false],
            value: vec![7.5],
        };
        let m = n.one();
        assert_eq!(m.predict(0, &[1.0]).unwrap(), 7.5);
        assert_eq!(m.predict(0, &[f64::NAN]).unwrap(), 7.5);
    }

    /// The base enters the accumulator differently per mode, matching where
    /// each estimator puts it. Pinned with values whose float addition is
    /// order-sensitive, so a refactor that "simplifies" to `base + acc`
    /// fails here rather than 632 ULP downstream.
    #[test]
    fn the_base_seeds_a_sum_but_is_added_after_a_mean() {
        let n = Nodes::stump().plus_stump(0, 1);
        let big = 1e16;
        let sum = n.build(&[big], &["sum"], &["identity"], 1).unwrap();
        assert_eq!(
            sum.predict(0, &[0.0]).unwrap(),
            (big + 1.0) + 1.0,
            "sum seeds the accumulator with the base"
        );
        let mean = n.build(&[big], &["mean"], &["identity"], 1).unwrap();
        assert_eq!(
            mean.predict(0, &[0.0]).unwrap(),
            big + (0.0 + 1.0 + 1.0) / 2.0,
            "mean divides first, then adds the base"
        );
    }

    #[test]
    fn base_and_aggregation_modes() {
        // Two identical stumps in one model: sum doubles, mean does not.
        let n = Nodes::stump().plus_stump(0, 1);
        assert_eq!(
            n.build(&[10.0], &["sum"], &["identity"], 1)
                .unwrap()
                .predict(0, &[0.0])
                .unwrap(),
            12.0
        );
        assert_eq!(
            n.build(&[10.0], &["mean"], &["identity"], 1)
                .unwrap()
                .predict(0, &[0.0])
                .unwrap(),
            11.0
        );
    }

    #[test]
    fn sigmoid_link_matches_expit() {
        let m = Nodes::stump()
            .build(&[0.0], &["sum"], &["sigmoid"], 1)
            .unwrap();
        assert_eq!(m.predict(0, &[0.0]).unwrap(), 1.0 / (1.0 + (-1.0f64).exp()));
        assert_eq!(m.predict(0, &[1.0]).unwrap(), 1.0 / (1.0 + (-2.0f64).exp()));
    }

    /// Several fitted models coexist in one prepared static — that is the
    /// whole point: one function name, one entry, many fitted groups.
    #[test]
    fn models_are_addressed_by_dense_id() {
        let m = Nodes::stump()
            .plus_stump(1, 0)
            .build(
                &[0.0, 100.0],
                &["sum", "sum"],
                &["identity", "identity"],
                1,
            )
            .unwrap();
        assert_eq!(m.n_models(), 2);
        assert_eq!(m.predict(0, &[0.0]).unwrap(), 1.0);
        assert_eq!(m.predict(1, &[0.0]).unwrap(), 101.0);
    }

    #[test]
    fn unknown_model_id_traps_by_name() {
        let m = Nodes::stump().one();
        for bad in [1i64, -1, i64::MAX] {
            let t = m
                .predict(bad, &[0.0])
                .expect_err("out-of-range id must trap");
            assert!(t.0.contains(&bad.to_string()), "trap should name the id: {}", t.0);
        }
    }

    // ------------------------------------------------- build-time refusals --

    #[test]
    fn refuses_child_index_out_of_range() {
        assert!(refusal(|n| n.left[0] = 99).contains("out of range"));
    }

    #[test]
    fn refuses_a_child_that_does_not_move_forward() {
        // Children always follow their parent in every layout we accept;
        // requiring it is what makes the traversal loop provably terminate.
        assert!(refusal(|n| n.left[0] = 0).contains("must follow its parent"));
    }

    #[test]
    fn refuses_an_unreachable_node() {
        // The root turned into a leaf, orphaning both of its children. NOT
        // spelled as "point the root's right child back at node 1" any more:
        // that makes node 1 a SHARED child, which trips its own refusal
        // first, and this test would then pass for the wrong reason.
        assert!(
            refusal(|n| {
                n.feature[0] = -1;
                n.left[0] = -1;
                n.right[0] = -1;
            })
            .contains("unreachable")
        );
    }

    #[test]
    fn refuses_a_shared_child() {
        // Both of the root's children are node 2: a decision DAG, not a
        // tree. It scores perfectly well — one path, terminating — but
        // "exactly one parent per non-root node" is what MAKES the table a
        // tree, and we check both ends of it rather than only zero (TASK-76).
        assert!(refusal(|n| n.left[0] = 2).contains("already has a parent"));
    }

    #[test]
    fn refuses_a_leaf_with_children() {
        assert!(refusal(|n| n.left[1] = 2).contains("leaf"));
    }

    #[test]
    fn refuses_a_split_without_both_children() {
        assert!(refusal(|n| n.feature[1] = 0).contains("split node"));
    }

    #[test]
    fn refuses_a_feature_beyond_the_declared_width() {
        assert!(refusal(|n| n.feature[0] = 1).contains("feature 1"));
    }

    #[test]
    fn refuses_unknown_agg_or_link_spelling() {
        let n = Nodes::stump();
        assert!(n
            .build(&[0.0], &["median"], &["identity"], 1)
            .expect_err("must refuse")
            .contains("median"));
        assert!(n
            .build(&[0.0], &["sum"], &["softmax"], 1)
            .expect_err("must refuse")
            .contains("softmax"));
    }

    #[test]
    fn refuses_a_model_id_present_on_only_one_side() {
        // Nodes name a model with no header row.
        let mut n = Nodes::stump();
        n.model_id[2] = 1;
        assert!(n
            .build(&[0.0], &["sum"], &["identity"], 1)
            .expect_err("must refuse")
            .contains("model 1"));
        // A header row with no nodes.
        assert!(Nodes::stump()
            .build(&[0.0, 0.0], &["sum", "sum"], &["identity", "identity"], 1)
            .expect_err("must refuse")
            .contains("no nodes"));
    }

    #[test]
    fn refuses_non_dense_model_ids() {
        let mut n = Nodes::stump();
        for v in n.model_id.iter_mut() {
            *v = 3;
        }
        let e = n
            .build_with_ids(&[3], &[0.0], &["sum"], &["identity"], 1)
            .expect_err("must refuse");
        assert!(e.contains("dense"), "{e}");
    }

    #[test]
    fn refuses_misordered_or_sparse_node_ids() {
        assert!(refusal(|n| n.node_id[1] = 5).contains("node id"));
    }

    #[test]
    fn refuses_ragged_columns() {
        let mut n = Nodes::stump();
        n.threshold.pop();
        assert!(n
            .build(&[0.0], &["sum"], &["identity"], 1)
            .expect_err("must refuse")
            .contains("same length"));
    }

    #[test]
    fn refuses_an_empty_ensemble() {
        assert!(Nodes::empty()
            .build(&[], &[], &[], 1)
            .expect_err("an ensemble with no models cannot serve anything")
            .contains("no models"));
    }
}
