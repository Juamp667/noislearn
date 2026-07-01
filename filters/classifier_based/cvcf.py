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

from .._detection import attach_detection_report, resample_by_action, validate_action


# Default C4.5-like tree used by the committee filter.
c45_like = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=33)


@dataclass
class CVCFFilterResult:
    """Summary of a cross-validated committees filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    fold_preds: np.ndarray
    fold_confidence: np.ndarray
    disagree_count: np.ndarray
    noisy_votes: np.ndarray
    n_models: int
    support: np.ndarray
    committee_pred_idx: np.ndarray
    committee_pred: np.ndarray
    committee_confidence: np.ndarray
    noise_score: np.ndarray


class CVCFFilter(BaseEstimator):
    """Cross-validated committees noise filter.

    Parameters
    ----------
    estimator : estimator, default=c45_like
        Base learner cloned for each fold of the committee.
    cv : int, default=10
        Number of stratified folds used to build the committee.
    vote_rule : {"threshold", "majority", "consensus"}, default="threshold"
        Rule used to flag samples as noisy from the fold disagreements.
    threshold : float, default=0.5
        Minimum fraction of disagreeing folds required when ``vote_rule="threshold"``.
    action : {"remove", "detect"}, default="remove"
        Whether noisy samples are dropped or only detected.
    random_state : int, default=33
        Seed used by the stratified splitter.

    Notes
    -----
    Relabel is not implemented yet.
    """

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
        if self.vote_rule == "majority":
            return disagree_count > (n_models / 2.0)
        if self.vote_rule == "threshold":
            return (disagree_count / float(n_models)) >= self.threshold
        raise ValueError("vote_rule must be 'threshold', 'majority', or 'consensus'")

    def fit(self, X, y):
        """Fit the filter and cache fold-wise predictions and agreement scores."""

        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]

        if int(self.cv) < 2:
            raise ValueError("cv must be >= 2")
        if n < self.cv:
            raise ValueError(f"Need n_samples >= cv. Got n_samples={n}, cv={self.cv}.")
        validate_action(self.action)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        fold_pred_idx = np.empty((self.cv, n), dtype=int)
        fold_confidence = np.ones((self.cv, n), dtype=float)

        for fold_idx, (train_idx, _) in enumerate(skf.split(X, y_idx)):
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])
            pred_idx = np.asarray(model.predict(X), dtype=int)
            fold_pred_idx[fold_idx] = pred_idx

            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X), dtype=float)
                if proba.ndim == 2 and proba.shape[0] == n and proba.shape[1] == self.classes_.shape[0]:
                    fold_confidence[fold_idx] = np.take_along_axis(proba, pred_idx[:, None], axis=1).ravel()

        # Compute the agreement between models
        disagree_count = (fold_pred_idx != y_idx[None, :]).sum(axis=0).astype(int)
        # Flag instances as noisy or not
        noisy_mask = self._flag_by_votes(disagree_count, n_models=self.cv)
        keep_mask = ~noisy_mask

        total_conf = np.sum(fold_confidence, axis=0)
        support = np.zeros((n, self.classes_.shape[0]), dtype=float)
        sample_range = np.arange(n)
        for fold_idx in range(self.cv):
            np.add.at(support, (sample_range, fold_pred_idx[fold_idx]), fold_confidence[fold_idx])

        support = np.divide(support, total_conf[:, None], out=np.zeros_like(support), where=total_conf[:, None] > 0.0)
        zero_conf_mask = total_conf <= 0.0
        if np.any(zero_conf_mask):
            support[zero_conf_mask] = 1.0 / float(self.classes_.shape[0])

        committee_pred_idx = np.argmax(support, axis=1)
        committee_pred = self.classes_[committee_pred_idx]
        committee_confidence = support[np.arange(n), committee_pred_idx]
        committee_noise_score = 1.0 - committee_confidence
        masked_support = support.copy()
        masked_support[np.arange(n), y_idx] = -np.inf
        alt_support = np.max(masked_support, axis=1)
        alt_support[~np.isfinite(alt_support)] = 0.0
        noise_score = (1.0 - support[np.arange(n), y_idx]) * np.sqrt(alt_support)

        self.result_ = CVCFFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            fold_preds=self.classes_[fold_pred_idx],
            fold_confidence=fold_confidence,
            disagree_count=disagree_count,
            noisy_votes=noisy_mask.astype(int),
            n_models=int(self.cv),
            support=support,
            committee_pred_idx=committee_pred_idx,
            committee_pred=committee_pred,
            committee_confidence=committee_confidence,
            noise_score=noise_score,
        )
        self.fold_confidence_ = fold_confidence
        self.support_ = support
        self.noise_score_ = noise_score
        self.committee_pred_idx_ = committee_pred_idx
        self.committee_pred_ = committee_pred
        self.committee_confidence_ = committee_confidence
        self.committee_noise_score_ = committee_noise_score
        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=committee_pred,
            fold_preds=self.result_.fold_preds,
            fold_confidence=fold_confidence,
            disagree_count=disagree_count,
            vote_fraction=disagree_count / float(self.cv),
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered data."""

        self.fit(X, y)
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def get_filter_report(self) -> Dict[str, Any]:
        """Return a dictionary with the main fit diagnostics."""

        r = self.result_
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(r.n_models),
            "removed_or_flagged": int((~r.keep_mask).sum()),
            "fraction_flagged": float(r.noisy_fraction),
            "support_mean": float(np.mean(self.support_)),
            "noise_score_mean": float(np.mean(self.noise_score_)),
            "vote_rule": self.vote_rule,
            "threshold": float(self.threshold) if self.vote_rule == "threshold" else None,
            "action": self.action,
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
