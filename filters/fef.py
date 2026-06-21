"""Filter ensemble label-noise filtering.

This module implements :class:`FilterEnsembleFilter`, a committee-style noise
filter that combines the decisions and optional noise scores of several
existing filters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_is_fitted

from .._detection import attach_detection_report, resample_by_action


_ALLOWED_STRATEGIES = {
    "union",
    "majority",
    "consensus",
    "k_of_m",
    "threshold",
    "weighted_threshold",
}
_RESERVED_ACTIONS = {"weight", "relabel"}


@dataclass
class FilterEnsembleFilterResult:
    """Summary of a filter-ensemble noise filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_mask: np.ndarray
    noisy_indices: np.ndarray
    noise_score: np.ndarray
    ensemble_score: np.ndarray
    filter_votes: np.ndarray
    filter_scores: np.ndarray
    support: np.ndarray
    support_fraction: np.ndarray
    filter_names: tuple[str, ...]
    filter_weights: np.ndarray
    base_reports: tuple[dict[str, Any], ...]
    strategy: str
    n_filters: int
    class_protection_applied: bool
    protected_indices: np.ndarray


def _copy_training_data(X, y):
    """Create a safe copy of the training data for a base filter fit."""

    if hasattr(X, "copy"):
        try:
            X_copy = X.copy()
        except TypeError:
            X_copy = copy.deepcopy(X)
    else:
        X_copy = copy.deepcopy(X)
    y_copy = np.array(y, copy=True)
    return X_copy, y_copy


def _safe_clone(estimator, random_state: int | None = None):
    """Clone an estimator when possible, otherwise deep-copy it."""

    try:
        fitted = clone(estimator)
    except Exception:
        fitted = copy.deepcopy(estimator)

    if random_state is not None:
        params = None
        if hasattr(fitted, "get_params"):
            try:
                params = fitted.get_params(deep=False)
            except Exception:
                params = None

        if params is not None and "random_state" in params and params.get("random_state") is None:
            try:
                fitted.set_params(random_state=random_state)
            except Exception:
                if hasattr(fitted, "random_state") and getattr(fitted, "random_state") is None:
                    setattr(fitted, "random_state", random_state)
        elif hasattr(fitted, "random_state") and getattr(fitted, "random_state", None) is None:
            try:
                setattr(fitted, "random_state", random_state)
            except Exception:
                pass

    return fitted


def _validate_action(action: str) -> None:
    if action not in ({"remove", "detect"} | _RESERVED_ACTIONS):
        raise ValueError("action must be one of 'remove', 'detect', 'weight', or 'relabel'.")


def _normalize_score_vector(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return scores

    if np.any(~np.isfinite(scores)):
        raise ValueError("noise scores must be finite.")

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_range = score_max - score_min
    if score_range <= 1e-12:
        if 0.0 <= score_min and score_max <= 1.0:
            return np.clip(scores, 0.0, 1.0)
        return np.zeros_like(scores, dtype=float)

    scaled = (scores - score_min) / score_range
    return np.clip(scaled, 0.0, 1.0)


def _extract_report(fitted_filter):
    if hasattr(fitted_filter, "get_detection_report"):
        try:
            report = fitted_filter.get_detection_report()
            if report is not None:
                return dict(report)
        except Exception:
            pass

    report = getattr(fitted_filter, "detection_report_", None)
    if report is None:
        return None
    try:
        return dict(report)
    except Exception:
        return None


def _extract_noisy_mask(report: dict[str, Any] | None, fitted_filter, n_samples: int) -> np.ndarray:
    candidates: list[np.ndarray] = []

    if report is not None:
        if "noisy_mask" in report and report["noisy_mask"] is not None:
            candidates.append(np.asarray(report["noisy_mask"], dtype=bool).ravel())
        if "noisy_indices" in report and report["noisy_indices"] is not None:
            noisy_indices = np.asarray(report["noisy_indices"], dtype=int).ravel()
            mask = np.zeros(n_samples, dtype=bool)
            mask[noisy_indices] = True
            candidates.append(mask)

    if hasattr(fitted_filter, "noisy_mask_"):
        candidates.append(np.asarray(getattr(fitted_filter, "noisy_mask_"), dtype=bool).ravel())

    result = getattr(fitted_filter, "result_", None)
    if result is not None:
        if hasattr(result, "keep_mask"):
            candidates.append(~np.asarray(result.keep_mask, dtype=bool).ravel())
        if hasattr(result, "noisy_mask"):
            candidates.append(np.asarray(result.noisy_mask, dtype=bool).ravel())

    if hasattr(fitted_filter, "keep_mask"):
        candidates.append(~np.asarray(getattr(fitted_filter, "keep_mask"), dtype=bool).ravel())

    if hasattr(fitted_filter, "sample_indices_"):
        mask = np.zeros(n_samples, dtype=bool)
        mask[np.asarray(getattr(fitted_filter, "sample_indices_"), dtype=int).ravel()] = True
        candidates.append(mask)

    for candidate in candidates:
        if candidate.shape[0] == n_samples:
            return candidate.astype(bool, copy=False)

    raise ValueError(f"Base filter noisy_mask length does not match n_samples={n_samples}.")


def _extract_noise_score(report: dict[str, Any] | None, fitted_filter):
    score = None
    if report is not None and report.get("noise_score") is not None:
        score = report["noise_score"]
    elif hasattr(fitted_filter, "noise_score_"):
        score = getattr(fitted_filter, "noise_score_")
    return score


class FilterEnsembleFilter(BaseEstimator):
    """Combine several noise filters using vote and score aggregation.

    Parameters
    ----------
    base_filters : list
        Sequence of fitted or unfitted filters. Each item can be either a
        filter instance or a ``(name, filter)`` pair.
    strategy : str, default="majority"
        Decision rule used to flag samples as noisy. Supported values are
        ``"union"``, ``"majority"``, ``"consensus"``, ``"k_of_m"``,
        ``"threshold"``, and ``"weighted_threshold"``.
    min_votes : int or None, default=None
        Minimum number of votes required when ``strategy="k_of_m"``.
    vote_threshold : float, default=0.5
        Fraction of filters that must vote noisy when ``strategy="threshold"``.
    score_threshold : float, default=0.5
        Minimum aggregated ensemble score when ``strategy="weighted_threshold"``.
    filter_weights : list of float or None, default=None
        Optional weight per base filter. If omitted, all filters receive equal
        weight. The weights are normalized internally.
    use_filter_scores : bool, default=True
        If True, use each base filter's ``noise_score`` when available.
        Otherwise every filter contributes only its binary vote.
    score_fallback : str, default="binary"
        Fallback strategy used when a base filter does not provide
        ``noise_score``. Only ``"binary"`` is implemented.
    normalize_scores : bool, default=False
        Whether to normalize each base filter score to ``[0, 1]`` before
        aggregation.
    min_class_count : int, default=2
        Minimum number of samples to keep per class after filtering.
    action : str, default="remove"
        Post-fit action. ``"remove"`` and ``"detect"`` are executed now.
        ``"weight"`` and ``"relabel"`` are reserved for future support.
    random_state : int or None, default=None
        Optional seed propagated to cloned base filters when they expose a
        ``random_state`` parameter set to ``None``.
    n_jobs : int or None, default=None
        Reserved for future parallel execution.

    Notes
    -----
    The final detection report preserves the common structure used by the rest
    of the library and adds ensemble-specific diagnostics such as the vote and
    score matrices.
    """

    def __init__(
        self,
        base_filters,
        strategy: str = "majority",
        min_votes: int | None = None,
        vote_threshold: float = 0.5,
        score_threshold: float = 0.5,
        filter_weights: list[float] | None = None,
        use_filter_scores: bool = True,
        score_fallback: str = "binary",
        normalize_scores: bool = False,
        min_class_count: int = 2,
        action: str = "remove",
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.base_filters = base_filters
        self.strategy = strategy
        self.min_votes = min_votes
        self.vote_threshold = vote_threshold
        self.score_threshold = score_threshold
        self.filter_weights = filter_weights
        self.use_filter_scores = use_filter_scores
        self.score_fallback = score_fallback
        self.normalize_scores = normalize_scores
        self.min_class_count = min_class_count
        self.action = action
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _resolve_base_filters(self):
        try:
            base_filters = list(self.base_filters)
        except TypeError as exc:
            raise TypeError("base_filters must be a sequence of filters or (name, filter) pairs.") from exc

        if not base_filters:
            raise ValueError("base_filters cannot be empty.")

        resolved = []
        for item in base_filters:
            if isinstance(item, tuple):
                if len(item) != 2 or not isinstance(item[0], str):
                    raise TypeError("Each base filter tuple must be of the form (name, filter).")
                name, estimator = item
            else:
                name, estimator = type(item).__name__, item

            if not hasattr(estimator, "fit"):
                raise TypeError(f"Base filter '{name}' must expose a fit method.")
            resolved.append((str(name), estimator))

        return resolved

    def _validate_parameters(self, n_filters: int) -> None:
        _validate_action(self.action)

        if self.strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(
                "strategy must be one of 'union', 'majority', 'consensus', 'k_of_m', "
                "'threshold', or 'weighted_threshold'."
            )

        if self.strategy == "k_of_m":
            if self.min_votes is None:
                raise ValueError("min_votes must be provided when strategy='k_of_m'.")
            min_votes = int(self.min_votes)
            if not (1 <= min_votes <= n_filters):
                raise ValueError("min_votes must satisfy 1 <= min_votes <= n_filters.")

        if self.strategy == "threshold":
            vote_threshold = float(self.vote_threshold)
            if not np.isfinite(vote_threshold) or not (0.0 <= vote_threshold <= 1.0):
                raise ValueError("vote_threshold must be in [0, 1].")

        if self.strategy == "weighted_threshold":
            score_threshold = float(self.score_threshold)
            if not np.isfinite(score_threshold) or not (0.0 <= score_threshold <= 1.0):
                raise ValueError("score_threshold must be in [0, 1].")

        if int(self.min_class_count) < 1:
            raise ValueError("min_class_count must be >= 1.")

        if self.score_fallback != "binary":
            raise NotImplementedError("Only score_fallback='binary' is implemented for now.")

        if self.filter_weights is not None:
            weights = np.asarray(self.filter_weights, dtype=float).ravel()
            if weights.shape[0] != n_filters:
                raise ValueError("filter_weights length must match the number of base filters.")
            if np.any(~np.isfinite(weights)):
                raise ValueError("filter_weights must be finite.")
            if np.any(weights < 0.0):
                raise ValueError("filter_weights must be non-negative.")
            if float(np.sum(weights)) <= 0.0:
                raise ValueError("filter_weights must sum to a value greater than zero.")

    def _fit_base_filter(self, name: str, estimator, X, y, n_samples: int):
        fitted = _safe_clone(estimator, random_state=self.random_state)
        X_fit, y_fit = _copy_training_data(X, y)
        fit_result = fitted.fit(X_fit, y_fit)
        if fit_result is not None:
            fitted = fit_result

        report = _extract_report(fitted)
        noisy_mask = _extract_noisy_mask(report, fitted, n_samples)
        if noisy_mask.shape[0] != n_samples:
            raise ValueError(f"Base filter '{name}' noisy_mask length does not match n_samples={n_samples}.")

        noise_score = _extract_noise_score(report, fitted)
        if noise_score is not None:
            noise_score = np.asarray(noise_score, dtype=float).ravel()
            if noise_score.shape[0] != n_samples:
                raise ValueError(f"Base filter '{name}' noise_score length does not match n_samples={n_samples}.")
            if np.any(~np.isfinite(noise_score)):
                raise ValueError(f"Base filter '{name}' returned non-finite values in noise_score.")

        if report is None:
            report = {}
        else:
            report = dict(report)

        report.setdefault("n_samples", int(n_samples))
        report.setdefault("n_noisy", int(noisy_mask.sum()))
        report.setdefault("noisy_indices", np.flatnonzero(noisy_mask))
        report.setdefault("noisy_mask", noisy_mask.copy())
        report.setdefault("noisy_fraction", float(noisy_mask.mean()))
        report.setdefault("observed_labels", np.asarray(y).copy())
        report.setdefault("predicted_labels", None)
        if noise_score is not None:
            report["noise_score"] = noise_score.copy()

        return fitted, noisy_mask, noise_score, report

    def _prepare_scores(self, noisy_mask: np.ndarray, noise_score: np.ndarray | None):
        if not self.use_filter_scores:
            return noisy_mask.astype(float)

        if noise_score is None:
            return noisy_mask.astype(float)

        scores = np.asarray(noise_score, dtype=float).ravel()
        if self.normalize_scores:
            scores = _normalize_score_vector(scores)
        return scores

    def _compute_final_mask(self, support: np.ndarray, support_fraction: np.ndarray, ensemble_score: np.ndarray):
        if self.strategy == "union":
            noisy_mask = support >= 1
        elif self.strategy == "majority":
            noisy_mask = support > (self.n_filters_ / 2.0)
        elif self.strategy == "consensus":
            noisy_mask = support == self.n_filters_
        elif self.strategy == "k_of_m":
            noisy_mask = support >= int(self.min_votes)
        elif self.strategy == "threshold":
            noisy_mask = support_fraction >= float(self.vote_threshold)
        elif self.strategy == "weighted_threshold":
            noisy_mask = ensemble_score >= float(self.score_threshold)
        else:
            raise ValueError("strategy must be one of 'union', 'majority', 'consensus', 'k_of_m', 'threshold', or 'weighted_threshold'.")

        return np.asarray(noisy_mask, dtype=bool)

    def _apply_min_class_count(self, noisy_mask: np.ndarray, y: np.ndarray, ensemble_score: np.ndarray):
        if int(self.min_class_count) <= 1:
            return noisy_mask, np.array([], dtype=int), False

        final_mask = np.asarray(noisy_mask, dtype=bool).copy()
        protected_indices: list[int] = []
        class_protection_applied = False

        for class_label in np.unique(y):
            class_indices = np.flatnonzero(y == class_label)
            class_noisy = class_indices[final_mask[class_indices]]
            if class_noisy.size == 0:
                continue

            allowed_remove = max(0, class_indices.size - int(self.min_class_count))
            if class_noisy.size <= allowed_remove:
                continue

            n_to_unmark = class_noisy.size - allowed_remove
            order = np.lexsort((class_noisy, ensemble_score[class_noisy]))
            protected = class_noisy[order[:n_to_unmark]]
            final_mask[protected] = False
            protected_indices.extend(int(idx) for idx in protected.tolist())
            class_protection_applied = True

        protected_indices_arr = np.array(sorted(set(protected_indices)), dtype=int)
        return final_mask, protected_indices_arr, class_protection_applied

    def fit(self, X, y):
        """Fit the ensemble filter and cache the detection report."""

        feature_names = None
        if hasattr(X, "columns"):
            feature_names = np.asarray(list(X.columns), dtype=object)

        X, y = check_X_y(X, y, accept_sparse=True)
        y = np.asarray(y)
        n_samples = int(X.shape[0])

        if feature_names is not None and feature_names.shape[0] == X.shape[1]:
            self.feature_names_in_ = feature_names
        self.n_features_in_ = int(X.shape[1])
        self.classes_ = np.unique(y)

        resolved = self._resolve_base_filters()
        self._validate_parameters(len(resolved))

        self.n_filters_ = int(len(resolved))
        self.filter_names_ = tuple(name for name, _ in resolved)

        if self.filter_weights is None:
            weights = np.ones(self.n_filters_, dtype=float)
        else:
            weights = np.asarray(self.filter_weights, dtype=float).ravel()
        weights = weights / float(np.sum(weights))
        self.filter_weights_ = weights

        base_reports: list[dict[str, Any]] = []
        filter_votes = np.zeros((n_samples, self.n_filters_), dtype=int)    # rowXsample, colXfilter
        filter_scores = np.zeros((n_samples, self.n_filters_), dtype=float)

        for filter_idx, (name, estimator) in enumerate(resolved):
            _, noisy_mask, noise_score, report = self._fit_base_filter(name, estimator, X, y, n_samples)
            base_reports.append(copy.deepcopy(report))
            filter_votes[:, filter_idx] = noisy_mask.astype(int)
            filter_scores[:, filter_idx] = self._prepare_scores(noisy_mask, noise_score)

        support = filter_votes.sum(axis=1).astype(int) # number of filters tagging as noisy
        support_fraction = support / float(self.n_filters_) # fraction of filters tagging as noisy

        # Compute the mean noise_score for each sample #TODO: extend the aggregation with other techniques
        ensemble_score = np.average(filter_scores, axis=1, weights=self.filter_weights_)

        final_noisy_mask = self._compute_final_mask(support, support_fraction, ensemble_score)
        final_noisy_mask, protected_indices, class_protection_applied = self._apply_min_class_count(
            final_noisy_mask,
            y,
            ensemble_score,
        )

        self.keep_mask_ = ~final_noisy_mask
        self.filter_votes_ = filter_votes
        self.filter_scores_ = filter_scores
        self.support_ = support
        self.support_fraction_ = support_fraction
        self.ensemble_score_ = np.asarray(ensemble_score, dtype=float)
        self.noise_score_ = self.ensemble_score_.copy()
        self.base_reports_ = tuple(base_reports)
        self.protected_indices_ = protected_indices
        self.class_protection_applied_ = bool(class_protection_applied)

        result = FilterEnsembleFilterResult(
            keep_mask=self.keep_mask_,
            noisy_fraction=float(final_noisy_mask.mean()),
            noisy_mask=final_noisy_mask,
            noisy_indices=np.flatnonzero(final_noisy_mask),
            noise_score=self.ensemble_score_,
            ensemble_score=self.ensemble_score_,
            filter_votes=filter_votes,
            filter_scores=filter_scores,
            support=support,
            support_fraction=support_fraction,
            filter_names=self.filter_names_,
            filter_weights=self.filter_weights_,
            base_reports=self.base_reports_,
            strategy=self.strategy,
            n_filters=self.n_filters_,
            class_protection_applied=self.class_protection_applied_,
            protected_indices=self.protected_indices_,
        )
        self.result_ = result

        attach_detection_report(
            self,
            final_noisy_mask,
            noise_score=self.ensemble_score_,
            observed_labels=y,
            predicted_labels=None,
            strategy=self.strategy,
            strategy_used=self.strategy,
            n_filters=self.n_filters_,
            filter_names=list(self.filter_names_),
            filter_votes=filter_votes,
            filter_scores=filter_scores,
            support=support,
            support_fraction=support_fraction,
            filter_weights=self.filter_weights_,
            base_reports=self.base_reports_,
            min_class_count=int(self.min_class_count),
            class_protection_applied=self.class_protection_applied_,
            protected_indices=self.protected_indices_,
        )

        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered data."""

        self.fit(X, y)
        if self.action in _RESERVED_ACTIONS:
            raise NotImplementedError("action='weight' and action='relabel' are not implemented yet.")
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def fit_filter(self, X, y):
        """Fit the filter and return the filtered data plus the report."""

        self.fit(X, y)
        if self.action in _RESERVED_ACTIONS:
            raise NotImplementedError("action='weight' and action='relabel' are not implemented yet.")
        X_clean, y_clean = resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)
        return X_clean, y_clean, dict(self.detection_report_)

    def get_support_matrix(self):
        """Return the binary vote matrix of the base filters."""

        check_is_fitted(self, ["filter_votes_"])
        return np.array(self.filter_votes_, copy=True)

    def get_score_matrix(self):
        """Return the per-filter score matrix used by the ensemble."""

        check_is_fitted(self, ["filter_scores_"])
        return np.array(self.filter_scores_, copy=True)

    def get_ensemble_score(self):
        """Return the aggregated ensemble noise score."""

        check_is_fitted(self, ["ensemble_score_"])
        return np.array(self.ensemble_score_, copy=True)

    def get_sample_weight(self, mode: str = "linear"):
        """Return instance weights derived from the ensemble noise score."""

        check_is_fitted(self, ["ensemble_score_"])
        if mode != "linear":
            raise ValueError("Only mode='linear' is implemented.")
        return np.clip(1.0 - np.asarray(self.ensemble_score_, dtype=float), 0.0, 1.0)

    def get_filter_report(self) -> Dict[str, Any]:
        """Return a compact summary of the ensemble run."""

        check_is_fitted(self, ["result_"])
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_filters": int(self.n_filters_),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "strategy": self.strategy,
            "action": self.action,
            "min_class_count": int(self.min_class_count),
            "class_protection_applied": bool(self.class_protection_applied_),
            "noise_score_mean": float(np.mean(self.noise_score_)),
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        check_is_fitted(self, ["detection_report_"])
        return dict(self.detection_report_)


FEF = FilterEnsembleFilter


__all__ = ["FilterEnsembleFilter", "FilterEnsembleFilterResult", "FEF"]
