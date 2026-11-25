import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

class WoE(BaseEstimator, TransformerMixin):

    def __init__(self, n_bins=10, max_cat=12): self.n_bins, self.max_cat = n_bins, max_cat

    def _prep(self, X):
        X = X.copy()
        for c in X.columns:
            if c.startswith("PAY_") and not c.startswith("PAY_AMT"): X[c] = X[c].clip(upper=3)
        return X
    
    def _codes(self, v, edges): return v if edges is None else np.searchsorted(edges, v, side="right")

    def fit(self, X, y):
        X, y = self._prep(X), np.asarray(y)
        B = y.sum(); G = len(y) - B
        self.maps_ = {}
        for c in X.columns:
            v = X[c].values.astype(float)
            edges = None if len(np.unique(v)) <= self.max_cat else \
                np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            codes, table = self._codes(v, edges), {}
            for k in np.unique(codes):
                msk = codes == k; b = y[msk].sum(); g = msk.sum() - b
                table[k] = np.log(((g + .5) / G) / ((b + .5) / B))
            self.maps_[c] = (edges, table)
        return self
    
    def transform(self, X):
        X = self._prep(X)
        out = np.empty((len(X), X.shape[1]))
        for j, c in enumerate(X.columns):
            edges, table = self.maps_[c]
            codes = self._codes(X[c].values.astype(float), edges)
            out[:, j] = [table.get(k, 0.0) for k in codes]
        return out