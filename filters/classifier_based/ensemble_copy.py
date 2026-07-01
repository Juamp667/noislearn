"""
Ensemble label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y

from .._detection import attach_detection_report, resample_by_action, validate_action


@dataclass
class EnsembleFilterResult:
    """Summary of an ensemble-based noise filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int
    support: np.ndarray
    noise_score: np.ndarray


class EnsembleFiltering_moded(BaseEstimator):
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
    action : {"remove", "detect"}, default="remove"
        Whether noisy samples are dropped or only detected.
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
        validate_action(self.action)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        n_classes = self.classes_.shape[0]

        oof_preds = np.empty((m, n), dtype=int)
        oof_confidence = np.ones((m, n), dtype=float)
        oof_proba = np.zeros((m, n, n_classes), dtype=float) # For each instance, each model computes a probability for each possible class

        for est_idx, est in enumerate(self.estimators):
            for train_indices, test_indices in skf.split(X, y_idx):
                model = clone(est)
                model.fit(X[train_indices], y_idx[train_indices])
                pred_idx = np.asarray(model.predict(X[test_indices]), dtype=int)
                oof_preds[est_idx, test_indices] = pred_idx

                has_valid_proba = False

                if hasattr(model, "predict_proba"):
                    proba = np.asarray(model.predict_proba(X[test_indices]), dtype=float) # -> dim(n, n_classes)
                    model_classes = np.asarray(model.classes_, dtype=int)

                    if (
                        proba.ndim == 2
                        and proba.shape[0] == test_indices.shape[0]
                        and proba.shape[1] == model_classes.shape[0]
                        and np.all((model_classes >= 0) & (model_classes < n_classes))
                    ):
                        # Se alinean las columnas de predict_proba con las clases globales.
                        proba_full = np.zeros((test_indices.shape[0], n_classes), dtype=float)
                        proba_full[:, model_classes] = proba

                        # Cada modelo conserva su distribución de probabilidades completa.
                        oof_proba[est_idx, test_indices] = proba_full

                        # Peso del modelo: confianza en su propia clase predicha.
                        oof_confidence[est_idx, test_indices] = proba_full[
                            np.arange(test_indices.shape[0]),
                            pred_idx,
                        ]

                        has_valid_proba = True

                if not has_valid_proba:
                    # Un modelo sin probabilidades contribuye mediante un voto duro.
                    oof_proba[est_idx, test_indices, pred_idx] = 1.0


        wrong_votes = (oof_preds != y_idx[None, :]).sum(axis=0)
        wrong_frac = wrong_votes / m
        total_conf = np.sum(oof_confidence, axis=0) #Suma de las confianzas de cada modelo en su predicción
        # Voto blando completo ponderado por la confianza de cada modelo.
        support = np.einsum(
            "mn,mnc->nc",
            oof_confidence,
            oof_proba,
        ) # Suma de einstein sobre índice m (el asociado a los clasificadores del ensemble)
        
        zero_conf_mask = total_conf <= 0.0
        support = np.divide(support, total_conf[:, None], out=np.zeros_like(support), where=total_conf[:, None] > 0.0)
        if np.any(zero_conf_mask):
            support[zero_conf_mask] = 1.0 / float(self.classes_.shape[0])

        committee_pred_idx = np.argmax(support, axis=1)
        committee_support = support[np.arange(n), committee_pred_idx]
        masked_support = support.copy()
        masked_support[np.arange(n), y_idx] = -np.inf
        alt_support = np.max(masked_support, axis=1)
        alt_support[~np.isfinite(alt_support)] = 0.0
        noise_score = (1.0 - support[np.arange(n), y_idx]) * np.sqrt(alt_support)

        if self.mode == "consensus":
            noisy_mask = (wrong_votes == m)
        elif self.mode == "threshold":
            noisy_mask = (wrong_frac >= self.threshold)
        else:
            raise ValueError("mode must be 'threshold' or 'consensus'")

        keep_mask = ~noisy_mask

        self.keep_mask = keep_mask
        self.sample_indices_ = np.flatnonzero(keep_mask)
        self.result_ = EnsembleFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), noisy_votes=wrong_votes, n_models=m, support=support, noise_score=noise_score)
        self.oof_preds_ = oof_preds
        self.oof_confidence_ = oof_confidence
        self.support_ = support
        self.noise_score_ = noise_score
        self.committee_pred_idx_ = committee_pred_idx
        self.committee_pred_ = self.classes_[committee_pred_idx]
        self.committee_confidence_ = committee_support
        self.committee_noise_score_ = 1.0 - committee_support
        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=self.committee_pred_,
            estimator_votes=self.classes_[oof_preds],
            estimator_confidence=oof_confidence,
            wrong_votes=wrong_votes,
            wrong_fraction=wrong_frac,
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered data."""

        self.fit(X, y)
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def get_filter_report(self):
        """Return a dictionary with the main fit diagnostics."""

        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(self.result_.n_models),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "support_mean": float(np.mean(self.support_)),
            "noise_score_mean": float(np.mean(self.noise_score_)),
            "mode": self.mode,
            "threshold": self.threshold if self.mode == "threshold" else None,
            "action": self.action,
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
