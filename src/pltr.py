import numpy as np
import itertools
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

def node_depths(t):
    depth = np.zeros(t.node_count, int)
    for node in range(t.node_count):
        for c in (t.children_left[node], t.children_right[node]):
            if c != -1:
                depth[c] = depth[node] + 1
    return depth

def leaf_rule(tree, leaf, nm):
    t, parent = tree.tree_, {}
    for node  in range(t.node_count):
        for child, sign in ((t.children_left[node], "<="), (t.children_right[node], ">")):
            if child != -1: parent[child] = (node, sign)
    conds, node = [], leaf
    while node in parent:
        p, sign = parent[node]
        conds.append(f"{nm[t.feature[p]]} {sign} {t.threshold[p]:.2f}")
        node = p
    return " AND ".join(reversed(conds))

class RuleExtractor:
    """One rule per variable (2-leaf tree) + one rule per variable pair
    (3-leaf tree, one leaf of the longer branch, both variables used)."""

    def __init__(self, min_leaf_frac=.01, seed=0):
        self.min_leaf_frac, self.seed = min_leaf_frac, seed

    def fit(self, X, y):
        n, d = X.shape
        msl = max(1, int(self.min_leaf_frac * n))
        self.rules_, seen = [], set()
        st = dict(single_tried=0, single_kept=0, pair_tried=0,
                  pair_no_second_split=0, pair_one_variable=0,
                  pair_duplicate=0, pair_kept=0)

        for cols in [(j,) for j in range(d)] + list(itertools.combinations(range(d), 2)):
            single = len(cols) == 1                      # <- was `-- 1`
            tree = DecisionTreeClassifier(
                max_leaf_nodes=2 if single else 3,
                min_samples_leaf=msl, random_state=self.seed
            ).fit(X[:, cols], y)
            t = tree.tree_
            leaves = np.where(t.children_left == -1)[0]
            st["single_tried" if single else "pair_tried"] += 1

            if single:
                if len(leaves) < 1: continue
                leaf = leaves[0]
            else:
                deep = leaves[node_depths(t)[leaves] == 2]
                if len(deep) == 0:
                    st["pair_no_second_split"] += 1; continue
                if len(set(t.feature[t.feature >= 0])) < 2:   # same variable twice
                    st["pair_one_variable"] += 1; continue
                leaf = deep[0]

            key = (tree.apply(X[:, cols]) == leaf).tobytes()
            if key in seen:
                if not single: st["pair_duplicate"] += 1
                continue
            seen.add(key)
            self.rules_.append((cols, tree, leaf))
            st["single_kept" if single else "pair_kept"] += 1

        self.stats_ = st
        return self

    def transform(self, X):
        out = np.empty((X.shape[0], len(self.rules_)), dtype=np.float32)
        cache = {}
        for k, (cols, tree, leaf) in enumerate(self.rules_):
            if id(tree) not in cache:
                cache[id(tree)] = tree.apply(X[:, cols])
            out[:, k] = (cache[id(tree)] == leaf)
        return out

class PLTR:
    """
    Base design + optional tree rules -> adaptive-lasso logistic regression.
    """

    def __init__(self, base, rule_cols, use_rules=True, min_leaf_frac=.01, Cs=np.logspace(-2, 2, 9), cv=10):
        self.base, self.rule_cols, self.use_rules = base, rule_cols, use_rules
        self.min_leaf_frac, self.Cs, self.cv = min_leaf_frac, Cs, cv

    @staticmethod
    def _dense(M):
        return M.toarray() if hasattr(M, "toarray") else np.asarray(M)

    def _design(self, X):
        B = self._dense(self.base_.transform(X))
        if not self.use_rules: return B
        R = self.rules_.transform(X[self.rule_cols].values.astype(float))
        return np.hstack([B, R])

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y)
        self.nb_ = self._dense(self.base_.transform(X)).shape[1]

        if self.use_rules: # rules learned on training data only
            self.rules_ = RuleExtractor(self.min_leaf_frac).fit(
                X[self.rule_cols].values.astype(float), y
            )
        D = self._design(X)
        ridge = LogisticRegression(C=1.0, max_iter=3000).fit(D, y) # stage 1: adaptive weights
        self.w_ = np.abs(ridge.coef_.ravel()) + 1e-6
        self.lasso_ = LogisticRegressionCV(
            Cs=list(self.Cs), cv=StratifiedKFold(self.cv, shuffle=True, random_state=0),
            penalty="l1", solver="liblinear", scoring="roc_auc",
            max_iter=500, n_jobs=-1
        ).fit(D * self.w_, y)
        return self

    def predict_proba(self, X):
        return self.lasso_.predict_proba(self._design(X) * self.w_)
    
    def explain(self, base_names, k=12):
        coef, nb = self.lasso_.coef_.ravel() * self.w_, self.nb_
        kept = [j for j in range(nb, len(coef)) if coef[j] != 0] if self.use_rules else []
        n_pair = sum(len(self.rules_.rules_[j - nb][0]) == 2 for j in kept)
        print(f"non-zero: {(coef != 0).sum()} of {len(coef)} | rules kept: {len(kept)} "
            f"(pair trees: {n_pair}, single-variable: {len(kept) - n_pair})")
        for j in np.argsort(-np.abs(coef))[:k]:
            if coef[j] == 0: break
            if j < nb:
                print(f"{coef[j]:+.3f}  {base_names[j]}")
            else:
                cols, tree, leaf = self.rules_.rules_[j - nb]
                print(f"{coef[j]:+.3f}  RULE: {leaf_rule(tree, leaf, [self.rule_cols[c] for c in cols])}")
