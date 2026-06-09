"""
Ensemble label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y


@dataclass
class EnsembleFilterResult:
    """Summary of an ensemble-based noise filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int


class EnsembleFiltering(BaseEstimator):
    """Ensemble-based noise filter using multiple classifiers.

    Parameters
    ----------
    estimators : sequence of estimators
        Base learners combined in the ensemble committee.
    cv : int, default=10
        Number of stratified folds used to compute out-of-fold predictions.
    mode : str, default="S"
        Decision rule used to flag samples as noisy. The current implementation
        accepts ``"threshold"`` and ``"consensus"``; the signature default is kept for compatibility.
    threshold : float, default=0.5
        Minimum fraction of disagreeing estimators required when ``mode="threshold"``.
    action : {"remove", "relabel"}, default="remove"
        Whether noisy samples are dropped or relabelled.
    random_state : int, default=33
        Seed used by the stratified splitter.
    return_noisy_samples : bool, default=False
        Stored on the instance for compatibility; the current implementation does not branch on it.

    Notes
    -----
    A sample is flagged as noisy when enough estimators disagree with its observed label.
    """

    def __init__(self, estimators, cv=10, mode="S", threshold=0.5, action="remove", random_state=33, return_noisy_samples=False):
        self.estimators = estimators
        self.cv = cv
        self.mode = mode
        self.threshold = threshold
        self.action = action
        self.random_state = random_state
        self.return_noisy_samples = return_noisy_samples

    def fit(self, X, y):
        """Fit the filter and cache ensemble disagreement counts."""

        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]
        m = len(self.estimators)
        if m < 2:
            raise ValueError("Provide at least 2 estimators for ensemble filtering.")

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        oof_preds = np.empty((m, n), dtype=int)

        for est_idx, est in enumerate(self.estimators):
            for train_indices, test_indices in skf.split(X, y_idx):
                model = clone(est)
                model.fit(X[train_indices], y_idx[train_indices])
                oof_preds[est_idx, test_indices] = model.predict(X[test_indices])

        wrong_votes = (oof_preds != y_idx[None, :]).sum(axis=0)
        wrong_frac = wrong_votes / m

        if self.mode == "consensus":
            noisy_mask = (wrong_votes == m)
        elif self.mode == "threshold":
            noisy_mask = (wrong_frac >= self.threshold)
        else:
            raise ValueError("mode must be 'threshold' or 'consensus'")

        keep_mask = ~noisy_mask

        self.keep_mask = keep_mask
        self.sample_indices_ = np.flatnonzero(keep_mask)
        self.result_ = EnsembleFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), noisy_votes=wrong_votes, n_models=m)
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered data."""

        self.fit(X, y)
        if self.action == "remove":
            return self.X_[self.result_.keep_mask], self.y_[self.result_.keep_mask]
        raise ValueError("action='relabel' is not implemented yet")

    def get_filter_report(self):
        """Return a dictionary with the main fit diagnostics."""

        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(self.result_.n_models),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "mode": self.mode,
            "threshold": self.threshold if self.mode == "threshold" else None,
            "action": self.action,
        }
