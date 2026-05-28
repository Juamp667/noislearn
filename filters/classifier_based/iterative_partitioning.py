"""
Iterative partitioning label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y


c45_like = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=33)


@dataclass
class IPFIterationInfo:
    iter_idx: int
    n_samples_before: int
    n_flagged: int
    frac_flagged: float
    threshold_p: float
    vote_rule: str


@dataclass
class IterativePartitioningFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int
    n_iters: int
    history: List[IPFIterationInfo]


class IterativePartitioningFilter(BaseEstimator):
    def __init__(self, estimator=c45_like, n_partitions: int = 10, vote_rule: str = "majority", action: str = "remove", p_stop: float = 0.01, k_patience: int = 3, max_iter: int = 20, random_state: int = 33):
        self.estimator = estimator
        self.n_partitions = n_partitions
        self.vote_rule = vote_rule
        self.action = action
        self.p_stop = p_stop
        self.k_patience = k_patience
        self.max_iter = max_iter
        self.random_state = random_state

    def _flag_by_votes(self, disagree_counts: np.ndarray, n_models: int) -> np.ndarray:
        if self.vote_rule == "consensus":
            return disagree_counts == n_models
        if self.vote_rule == "majority":
            return disagree_counts > (n_models / 2.0)
        raise ValueError("vote_rule must be 'majority' or 'consensus'")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        n0 = X.shape[0]
        orig_idx = np.arange(n0)
        alive = np.ones(n0, dtype=bool)
        noisy_votes_global = np.zeros(n0, dtype=int)
        history: List[IPFIterationInfo] = []
        patience_counter = 0
        y_out = y.copy()

        for it in range(1, self.max_iter + 1):
            E_idx = orig_idx[alive]
            X_E = X[E_idx]
            y_E = y_out[E_idx]
            nE = X_E.shape[0]
            if nE < 2 * self.n_partitions:
                break
            skf = StratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state + it)
            preds = np.empty((self.n_partitions, nE), dtype=object)
            for m, (_, part_idx) in enumerate(skf.split(X_E, y_E)):
                model = clone(self.estimator)
                model.fit(X_E[part_idx], y_E[part_idx])
                preds[m] = model.predict(X_E)
            disagree_counts = (preds != y_E[None, :]).sum(axis=0).astype(int)
            maj = None
            if self.action == "relabel":
                maj = np.empty(nE, dtype=object)
                for j in range(nE):
                    vals, cnts = np.unique(preds[:, j], return_counts=True)
                    maj[j] = vals[np.argmax(cnts)]
            noisy_local = self._flag_by_votes(disagree_counts, self.n_partitions)
            noisy_votes_global[E_idx] = disagree_counts
            n_flagged = int(noisy_local.sum())
            frac_flagged = float(n_flagged / max(nE, 1))
            patience_counter = patience_counter + 1 if n_flagged < (self.p_stop * n0) else 0
            history.append(IPFIterationInfo(it, nE, n_flagged, frac_flagged, self.p_stop, self.vote_rule))
            if n_flagged == 0:
                break
            if self.action == "remove":
                alive[E_idx[noisy_local]] = False
            elif self.action == "relabel":
                flagged_orig = E_idx[noisy_local]
                y_out[flagged_orig] = maj[noisy_local]
            else:
                raise ValueError("action must be 'remove' or 'relabel'")
            if patience_counter >= self.k_patience:
                break

        self.result_ = IterativePartitioningFilterResult(keep_mask=alive.copy(), noisy_fraction=float((~alive).mean()) if self.action == "remove" else float((history[-1].n_flagged / n0) if history else 0.0), noisy_votes=noisy_votes_global, n_models=int(self.n_partitions), n_iters=int(len(history)), history=history)
        self.X_ = X
        self.y_ = y_out
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]
        return self.X_, self.y_

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        last = r.history[-1] if r.history else None
        return {"n_samples": int(self.X_.shape[0]), "n_models_per_iter": int(r.n_models), "n_iters": int(r.n_iters), "vote_rule": self.vote_rule, "action": self.action, "fraction_removed_or_flagged": float(r.noisy_fraction), "last_iter_flagged": int(last.n_flagged) if last else 0, "last_iter_frac_flagged": float(last.frac_flagged) if last else 0.0, "p_stop": float(self.p_stop), "k_patience": int(self.k_patience), "max_iter": int(self.max_iter)}
