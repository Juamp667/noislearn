"""
Single-classifier label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y


@dataclass
class ClassificationFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    oof_pred: np.ndarray


class ClassificationFilter(BaseEstimator):
    def __init__(self, estimator, cv=10, action="remove", random_state=33):
        self.estimator = estimator
        self.cv = cv
        self.action = action
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        oof_pred_idx = np.empty(n, dtype=int)
        for train_idx, test_idx in skf.split(X, y_idx):
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])
            oof_pred_idx[test_idx] = model.predict(X[test_idx])

        oof_pred = self.classes_[oof_pred_idx]
        noisy_mask = (oof_pred_idx != y_idx)
        keep_mask = ~noisy_mask

        self.result_ = ClassificationFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), oof_pred=oof_pred)
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]
        if self.action == "relabel":
            y_new = np.asarray(self.y_).copy()
            noisy_idx = np.where(~self.result_.keep_mask)[0]
            y_new[noisy_idx] = self.result_.oof_pred[noisy_idx]
            return self.X_, y_new
        raise ValueError("action must be 'remove' or 'relabel'")

    def get_filter_report(self):
        return {
            "n_samples": int(self.X_.shape[0]),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "cv": int(self.cv),
            "action": self.action,
        }
