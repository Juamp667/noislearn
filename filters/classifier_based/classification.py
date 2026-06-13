"""
Single-classifier label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y

from .._detection import attach_detection_report, resample_by_action, validate_action


@dataclass
class ClassificationFilterResult:
    """Summary of a single-classifier noise filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    oof_pred: np.ndarray


class ClassificationFilter(BaseEstimator):
    """Cross-validated single-classifier noise filter.

    Parameters
    ----------
    estimator : estimator
        Base learner cloned and trained on each fold.
    cv : int, default=10
        Number of stratified folds used to generate out-of-fold predictions.
    action : {"remove", "detect"}, default="remove"
        Whether noisy samples are dropped or only detected.
    random_state : int, default=33
        Seed used by the stratified splitter.

    Notes
    -----
    A sample is flagged as noisy when its out-of-fold prediction differs from the observed label.
    """

    def __init__(self, estimator, cv=10, action="remove", random_state=33):
        self.estimator = estimator
        self.cv = cv
        self.action = action
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the filter and cache out-of-fold predictions."""

        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]
        validate_action(self.action)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        oof_pred_idx = np.empty(n, dtype=int)
        oof_confidence = np.full(n, np.nan, dtype=float)
        oof_proba = None
        has_oof_proba = True
        for train_idx, test_idx in skf.split(X, y_idx):
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])
            pred_idx = np.asarray(model.predict(X[test_idx]), dtype=int)
            oof_pred_idx[test_idx] = pred_idx

            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X[test_idx]), dtype=float)
                if proba.ndim == 2 and proba.shape[0] == test_idx.shape[0]:
                    if oof_proba is None:
                        oof_proba = np.full((n, proba.shape[1]), np.nan, dtype=float)
                    if proba.shape[1] != self.classes_.shape[0]:
                        has_oof_proba = False
                    else:
                        oof_proba[test_idx] = proba
                        oof_confidence[test_idx] = np.take_along_axis(proba, pred_idx[:, None], axis=1).ravel()
                else:
                    has_oof_proba = False
            else:
                has_oof_proba = False

        oof_pred = self.classes_[oof_pred_idx]
        noisy_mask = (oof_pred_idx != y_idx)
        keep_mask = ~noisy_mask
        noise_score = None
        if has_oof_proba and oof_proba is not None and np.isfinite(oof_proba).all():
            p_true = oof_proba[np.arange(n), y_idx]
            masked = oof_proba.copy()
            masked[np.arange(n), y_idx] = -np.inf
            p_alt = np.max(masked, axis=1)
            p_alt[~np.isfinite(p_alt)] = 0.0
            noise_score = (1.0 - p_true) * np.sqrt(p_alt)
        else:
            oof_proba = None

        self.result_ = ClassificationFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), oof_pred=oof_pred)
        self.oof_confidence_ = oof_confidence
        self.oof_proba_ = oof_proba
        self.noise_score_ = noise_score
        self.X_ = X
        self.y_ = y
        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=oof_pred,
            confidence=oof_confidence,
            oof_proba=oof_proba,
        )
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered or detected data."""

        self.fit(X, y)
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def get_filter_report(self):
        """Return a dictionary with the main fit diagnostics."""

        return {
            "n_samples": int(self.X_.shape[0]),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "cv": int(self.cv),
            "action": self.action,
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
