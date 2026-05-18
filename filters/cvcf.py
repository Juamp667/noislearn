"""
Cross-Validated Committees Filter (CVCF).
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y


c45_like = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=33)


@dataclass
class CVCFFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    fold_preds: np.ndarray
    disagree_count: np.ndarray
    noisy_votes: np.ndarray
    n_models: int


class CVCFFilter(BaseEstimator):
    def __init__(self, estimator=c45_like, cv: int = 10, vote_rule: str = "threshold", threshold: float = 0.5, action: str = "remove", random_state: int = 33):
        self.estimator = estimator
        self.cv = cv
        self.vote_rule = vote_rule
        self.threshold = threshold
        self.action = action
        self.random_state = random_state

    def _flag_by_votes(self, disagree_count: np.ndarray, n_models: int) -> np.ndarray:
        '''
        Flags instances as noisy or not based on the voting criteria given by `self.vote_rule`.
        '''
        if self.vote_rule == "consensus":
            return disagree_count == n_models
        if self.vote_rule == "threshold":
            return (disagree_count / float(n_models)) >= self.threshold
        raise ValueError("vote_rule must be 'threshold' or 'consensus'")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]

        if int(self.cv) < 2:
            raise ValueError("cv must be >= 2")
        if n < self.cv:
            raise ValueError(f"Need n_samples >= cv. Got n_samples={n}, cv={self.cv}.")
        if self.action not in {"remove", "relabel"}:
            raise ValueError("action must be 'remove' or 'relabel'")

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        fold_preds = np.empty((self.cv, n), dtype=int)

        for fold_idx, (train_idx, _) in enumerate(skf.split(X, y_idx)):
            # Train a model on the train_fold
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])
            # Predict in all the training set
            fold_preds[fold_idx] = model.predict(X)

        # Compute the agreement between models
        disagree_count = (fold_preds != y_idx[None, :]).sum(axis=0).astype(int)
        # Flag instances as noisy or not
        noisy_mask = self._flag_by_votes(disagree_count, n_models=self.cv)
        keep_mask = ~noisy_mask

        self.result_ = CVCFFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            fold_preds=self.classes_[fold_preds],
            disagree_count=disagree_count,
            noisy_votes=noisy_mask.astype(int),
            n_models=int(self.cv),
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]

        # Elif "relabel"
        raise Exception("Relabeling has no been implemented yet.")

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(r.n_models),
            "removed_or_flagged": int((~r.keep_mask).sum()),
            "fraction_flagged": float(r.noisy_fraction),
            "vote_rule": self.vote_rule,
            "threshold": float(self.threshold) if self.vote_rule == "threshold" else None,
            "action": self.action,
        }
