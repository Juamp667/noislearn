"""
Iterative class Noise Filter based on the Fusion of Classifiers (INFFC).
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y


c45_like = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=33)
knn1_like = KNeighborsClassifier(n_neighbors=1)
lda_like = LinearDiscriminantAnalysis()


@dataclass
class INFFCIterationInfo:
    iter_idx: int
    n_samples_before: int
    n_flagged: int
    frac_flagged: float
    decision_rule: str


@dataclass
class INFFCFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int
    n_iters: int
    history: List[INFFCIterationInfo]


class INFFCFilter(BaseEstimator):
    def __init__(self, estimators=None, cv: int = 10, decision_rule: str = "majority", threshold: float = 0.5, action: str = "remove", max_iter: int = 20, max_removed_frac: float = 0.5, random_state: int = 33):
        self.estimators = estimators
        self.cv = cv
        self.decision_rule = decision_rule
        self.threshold = threshold
        self.action = action
        self.max_iter = max_iter
        self.max_removed_frac = max_removed_frac
        self.random_state = random_state

    def _default_estimators(self):
        return [clone(c45_like), clone(knn1_like), clone(lda_like)]

    def _flag_by_votes(self, disagree_count: np.ndarray, n_models: int) -> np.ndarray:
        '''
        Flags instances as noisy depending on `self.decision_rule` criteria.
        '''
        frac = disagree_count / float(n_models)
        if self.decision_rule == "consensus":
            return frac >= 1.0
        if self.decision_rule == "majority":
            return frac >= 0.5
        if self.decision_rule == "threshold":
            return frac >= float(self.threshold)
        raise ValueError("decision_rule must be 'consensus', 'majority', or 'threshold'")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)

        n0 = X.shape[0]
        if int(self.cv) < 2:
            raise ValueError("cv must be >= 2")
        if n0 < self.cv:
            raise ValueError(f"Need n_samples >= cv. Got n_samples={n0}, cv={self.cv}.")
        if self.action not in {"remove", "relabel"}:
            raise ValueError("action must be 'remove' or 'relabel'")
        if not (0.0 <= float(self.max_removed_frac) <= 1.0):
            raise ValueError("max_removed_frac must be in [0, 1]")

        estimators = self._default_estimators() if self.estimators is None else list(self.estimators)
        m = len(estimators)
        if m < 2:
            raise ValueError("Provide at least 2 estimators for INFFC.")

        orig_idx = np.arange(n0)
        alive = np.ones(n0, dtype=bool)
        noisy_votes_global = np.zeros(n0, dtype=int)
        history: List[INFFCIterationInfo] = []
        y_out = y.copy()

        for it in range(1, int(self.max_iter) + 1):
            # Extract remanining instances
            E_idx = orig_idx[alive]
            X_E = X[E_idx]
            y_E = y_out[E_idx]
            nE = X_E.shape[0]
            # Check there are more instances than partitions planned
            if nE < self.cv: 
                break
            
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state + it)
            preds = np.empty((m, nE), dtype=object) # -> shape = (n_estimators, n_remaining_instances)

            # Make oof predictions for each estimator 
            for est_idx, est in enumerate(estimators):
                for train_idx, test_idx in skf.split(X_E, y_E):
                    model = clone(est)
                    # Fit with train_set
                    model.fit(X_E[train_idx], y_E[train_idx])
                    # Comute and store oof preds associated to the `est` estimator
                    preds[est_idx, test_idx] = model.predict(X_E[test_idx])

            # Compute the number of disagreements between the set estimators for each remaining instance 
            disagree_counts = (preds != y_E[None, :]).sum(axis=0).astype(int)
            # Flag remaining samples as noisy or not
            noisy_local = self._flag_by_votes(disagree_counts, m)
            
            noisy_votes_global[E_idx] = disagree_counts # Update number of disagreements for remaining instances
            n_flagged = int(noisy_local.sum())  # Number of instances flagged as noisy in this iteration
            frac_flagged = float(n_flagged / max(nE, 1))    # Fraction (with respect to the currently remaining) of instances flagged as noisy in this iteration
            history.append(INFFCIterationInfo(it, nE, n_flagged, frac_flagged, self.decision_rule))

            # Stop if no sample has been flagged as noisy
            if n_flagged == 0:
                break
            
            # Remove reamining instances flagged as noisy in this iteration
            if self.action == "remove":
                alive[E_idx[noisy_local]] = False
            elif self.action == "relabel":
                raise ValueError("action='relabel' is not implemented yet.")

            removed_frac = float((~alive).sum() / n0)   # Fraction (with respect to the initial training dataset) removed up to this iteration
            # Stop if a pct higher than `self.max_removed_frac` has been removed up to this iteration
            if removed_frac >= float(self.max_removed_frac):
                break

        keep_mask = alive.copy()
        noisy_fraction = float((~keep_mask).mean())

        self.result_ = INFFCFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=noisy_fraction,
            noisy_votes=noisy_votes_global,
            n_models=int(m),
            n_iters=int(len(history)),
            history=history,
        )
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
        return {"n_samples": int(self.X_.shape[0]), "n_models": int(r.n_models), "n_iters": int(r.n_iters), "decision_rule": self.decision_rule, "action": self.action, "fraction_removed_or_flagged": float(r.noisy_fraction), "last_iter_flagged": int(last.n_flagged) if last else 0, "last_iter_frac_flagged": float(last.frac_flagged) if last else 0.0, "cv": int(self.cv), "threshold": float(self.threshold) if self.decision_rule == "threshold" else None, "max_iter": int(self.max_iter), "max_removed_frac": float(self.max_removed_frac)}
