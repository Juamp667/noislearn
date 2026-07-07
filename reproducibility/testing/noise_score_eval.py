"""Evaluation helpers for continuous noise scores.

The goal is to compare noise filters without collapsing their output into a
single arbitrary threshold.

Three complementary views are provided:

* ROC-AUC and Average Precision sweep all possible thresholds, so they evaluate
  the ranking quality of the score itself.
* Precision@k / Recall@k / F1@k check whether the k most suspicious instances
  coincide with the k truly corrupted ones.
* Filtering curves measure the practical effect of removing progressively more
  suspicious samples before training a classifier.

This module is designed to be used from notebooks or library code.
"""

from __future__ import annotations

from numbers import Integral
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_consistent_length

__all__ = [
    "evaluate_noise_score_ranking",
    "evaluate_all_noise_scores",
    "filtering_curve_evaluation",
    "compare_filtering_curves",
    "summarize_filtering_curve",
]


_RANKING_COLUMNS = [
    "roc_auc",
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
]


def _as_1d_array(values: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    """Convert `values` to a strict 1D numpy array."""

    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Replace non-finite values with ordered sentinels.

    NaN and -inf are treated as the lowest possible scores. +inf is treated as
    the highest possible score. This keeps the ranking stable while still
    allowing sklearn metrics to consume the vector.
    """

    scores = np.asarray(scores, dtype=float).copy()
    finite_mask = np.isfinite(scores)

    if np.any(finite_mask):
        finite_scores = scores[finite_mask]
        low = np.nextafter(float(np.min(finite_scores)), -np.inf)
        high = np.nextafter(float(np.max(finite_scores)), np.inf)
    else:
        # All values are NaN/inf. Use a constant vector so ranking metrics still
        # have a deterministic ordering.
        low = 0.0
        high = 0.0

    clean = scores
    low_mask = np.isnan(clean) | np.isneginf(clean)
    high_mask = np.isposinf(clean)
    clean[low_mask] = low
    clean[high_mask] = high
    clean[~np.isfinite(clean)] = low
    return clean


def _safe_scalar(value: Any) -> float:
    """Convert a metric result into a finite float or NaN."""

    arr = np.asarray(value)
    if arr.size != 1:
        raise TypeError("metric_fn must return a scalar value.")
    scalar = float(arr.reshape(()))
    return scalar if np.isfinite(scalar) else np.nan


def _evaluate_metric(metric_fn: Callable[..., Any], estimator: Any, X_eval: Any, y_eval: Any) -> float:
    """Evaluate a metric callable with several common signatures.

    The helper supports:
    * sklearn scorers: `metric_fn(estimator, X_eval, y_eval)`
    * sklearn-style metrics: `metric_fn(y_eval, y_pred)`
    * external closures: `metric_fn(estimator)`, or `metric_fn()` when the
      callable closes over its own external evaluation set
    """

    try:
        return _safe_scalar(metric_fn(estimator, X_eval, y_eval))
    except TypeError:
        pass
    except Exception:
        return np.nan

    y_pred = None
    if hasattr(estimator, "predict"):
        try:
            y_pred = estimator.predict(X_eval)
        except Exception:
            y_pred = None

    if y_pred is not None:
        try:
            return _safe_scalar(metric_fn(y_eval, y_pred))
        except TypeError:
            pass
        except Exception:
            return np.nan

    try:
        return _safe_scalar(metric_fn(estimator))
    except Exception:
        pass

    try:
        return _safe_scalar(metric_fn())
    except Exception:
        return np.nan


def _subset_rows(obj: Any, mask: np.ndarray) -> Any:
    """Subset rows without mutating the original object.

    pandas objects keep their original indices. Numpy arrays are sliced by
    position.
    """

    mask = np.asarray(mask)

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if mask.dtype == bool:
            return obj.iloc[mask].copy()
        return obj.iloc[np.asarray(mask, dtype=int)].copy()
    return obj[mask]


def _to_filtered_counts(n_total: int, removal_percentage: float) -> int:
    """Convert a removal percentage to an integer count."""

    return int(np.ceil((float(removal_percentage) / 100.0) * n_total))


def _normalize_removal_percentages(removal_percentages: Sequence[float]) -> np.ndarray:
    """Normalize removal percentages to the 0-100 scale.

    Values in [0, 1] are interpreted as fractions. Larger values are treated as
    percentages.
    """

    arr = np.asarray(list(removal_percentages), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("removal_percentages must be a non-empty 1D sequence.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("removal_percentages must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError("removal_percentages must be non-negative.")

    if np.max(arr) <= 1.0:
        arr = arr * 100.0
    if np.any(arr > 100.0):
        raise ValueError("removal_percentages must be within [0, 100].")
    return arr


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.nan
    return float(np.mean(arr[valid]))


def _nanstd_ddof1(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        return np.nan
    return float(np.std(arr[valid], ddof=1))


def _resolve_cv(cv: Any, random_state: int) -> Any:
    """Resolve `cv` into a stratified splitter."""

    if isinstance(cv, Integral):
        if cv < 2:
            raise ValueError("cv must be >= 2.")
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if hasattr(cv, "split"):
        return cv
    raise TypeError("cv must be an int, a splitter with split(), or None.")


def _fit_filtered_model(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    keep_mask: np.ndarray,
) -> tuple[Any | None, Any | None, bool]:
    """Clone, fit and validate a filtered model.

    Returns
    -------
    fitted_estimator, y_filtered, invalid
    """

    y_train_arr = _as_1d_array(y_train, name="y_train")
    keep_mask = np.asarray(keep_mask, dtype=bool)

    if keep_mask.shape[0] != y_train_arr.shape[0]:
        raise ValueError("keep_mask must match the training sample count.")

    y_filtered = _subset_rows(y_train, keep_mask)
    y_filtered_arr = _as_1d_array(y_filtered, name="y_filtered")

    if np.unique(y_filtered_arr).size < 2:
        return None, y_filtered, True

    X_filtered = _subset_rows(X_train, keep_mask)

    try:
        fitted = clone(estimator)
        fitted.fit(X_filtered, y_filtered)
    except Exception:
        return None, y_filtered, True

    return fitted, y_filtered, False


def evaluate_noise_score_ranking(noisy_ground_truth: Any, scores: Any) -> dict[str, float]:
    """Evaluate a continuous noise score as a ranking of suspicious samples.

    ROC-AUC and Average Precision evaluate the full ranking across all possible
    thresholds. Precision@k / Recall@k / F1@k check whether the k most suspicious
    instances match the k truly corrupted ones, where k is the number of actual
    noisy samples.
    """

    y_true = _as_1d_array(noisy_ground_truth, name="noisy_ground_truth", dtype=bool)
    score_arr = _as_1d_array(scores, name="scores", dtype=float)
    check_consistent_length(y_true, score_arr)

    score_arr = _sanitize_scores(score_arr)
    n_samples = int(y_true.shape[0])
    n_positives = int(np.sum(y_true))
    k = n_positives

    roc_auc = np.nan
    if 0 < n_positives < n_samples:
        try:
            roc_auc = float(roc_auc_score(y_true, score_arr))
        except Exception:
            roc_auc = np.nan

    average_precision = np.nan
    if n_positives > 0:
        try:
            average_precision = float(average_precision_score(y_true, score_arr))
        except Exception:
            average_precision = np.nan

    if k == 0:
        precision_at_k = np.nan
        recall_at_k = np.nan
        f1_at_k = np.nan
    else:
        top_idx = np.argsort(-score_arr, kind="mergesort")[:k]
        tp_at_k = int(np.sum(y_true[top_idx]))
        precision_at_k = tp_at_k / float(k)
        recall_at_k = tp_at_k / float(n_positives)
        denom = precision_at_k + recall_at_k
        f1_at_k = 0.0 if denom == 0.0 else (2.0 * precision_at_k * recall_at_k) / denom

    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "precision_at_k": float(precision_at_k) if np.isfinite(precision_at_k) else np.nan,
        "recall_at_k": float(recall_at_k) if np.isfinite(recall_at_k) else np.nan,
        "f1_at_k": float(f1_at_k) if np.isfinite(f1_at_k) else np.nan,
    }


def evaluate_all_noise_scores(noisy_ground_truth: Any, noise_scores: Mapping[str, Any]) -> pd.DataFrame:
    """Evaluate every method in `noise_scores` with `evaluate_noise_score_ranking`."""

    rows: list[dict[str, Any]] = []
    for method, scores in noise_scores.items():
        metrics = evaluate_noise_score_ranking(noisy_ground_truth, scores)
        rows.append({"method": method, **metrics})
    return pd.DataFrame(rows, columns=["method", *_RANKING_COLUMNS])


def filtering_curve_evaluation(
    X: Any,
    y_noisy: Any,
    scores: Any,
    estimator: Any,
    metric_fn: Callable[..., Any],
    removal_percentages: Sequence[float],
    cv: Any | None = None,
    random_state: int = 33,
) -> pd.DataFrame:
    """Evaluate how useful a score is as a progressive filtering criterion.

    For each removal percentage, the top-scoring samples are removed, the
    estimator is refit, and the metric is computed.

    If `cv` is provided, the procedure is repeated fold-by-fold using
    `StratifiedKFold`. The filtering is always computed only on the training fold
    to avoid test leakage.

    When `cv` is None, the function fits on the full input and leaves the metric
    evaluation flexible: `metric_fn` may be a regular sklearn metric, a scorer,
    or a closure that only needs the fitted estimator.
    """

    if not callable(metric_fn):
        raise TypeError("metric_fn must be callable.")

    y_arr = _as_1d_array(y_noisy, name="y_noisy")
    score_arr = _as_1d_array(scores, name="scores", dtype=float)
    check_consistent_length(X, y_arr, score_arr)

    removal_percentages_arr = _normalize_removal_percentages(removal_percentages)
    rows: list[dict[str, Any]] = []

    if cv is None:
        for removal_percentage in removal_percentages_arr:
            n_remove = _to_filtered_counts(len(y_arr), float(removal_percentage))
            order = np.argsort(-_sanitize_scores(score_arr), kind="mergesort")
            keep_mask = np.ones(len(y_arr), dtype=bool)
            if n_remove > 0:
                keep_mask[order[:n_remove]] = False

            fitted, _, invalid = _fit_filtered_model(estimator, X, y_noisy, keep_mask)
            metric_value = np.nan
            if not invalid and fitted is not None:
                X_eval = _subset_rows(X, keep_mask)
                y_eval = _subset_rows(y_noisy, keep_mask)
                metric_value = _evaluate_metric(metric_fn, fitted, X_eval, y_eval)

            rows.append(
                {
                    "removal_percentage": float(removal_percentage),
                    "n_removed_mean": float(n_remove),
                    "n_train_remaining_mean": float(len(y_arr) - n_remove),
                    "metric_mean": float(metric_value) if np.isfinite(metric_value) else np.nan,
                    "metric_std": np.nan,
                }
            )
    else:
        splitter = _resolve_cv(cv, random_state=random_state)
        y_split = y_arr
        X_split = X

        for removal_percentage in removal_percentages_arr:
            fold_metrics: list[float] = []
            removed_counts: list[int] = []
            remaining_counts: list[int] = []

            for train_idx, test_idx in splitter.split(X_split, y_split):
                X_train = _subset_rows(X_split, train_idx)
                y_train = _subset_rows(y_noisy, train_idx)
                scores_train = score_arr[train_idx]

                n_remove = _to_filtered_counts(len(train_idx), float(removal_percentage))
                order = train_idx[np.argsort(-_sanitize_scores(scores_train), kind="mergesort")]
                keep_mask_full = np.ones(len(y_split), dtype=bool)
                if n_remove > 0:
                    keep_mask_full[order[:n_remove]] = False

                train_keep_mask = keep_mask_full[train_idx]

                # Never touch the test fold. The mask is only applied to the
                # training subset.
                fitted, _, invalid = _fit_filtered_model(
                    estimator,
                    X_train,
                    y_train,
                    train_keep_mask,
                )

                metric_value = np.nan
                if not invalid and fitted is not None:
                    X_test = _subset_rows(X_split, test_idx)
                    y_test = _subset_rows(y_noisy, test_idx)
                    metric_value = _evaluate_metric(metric_fn, fitted, X_test, y_test)

                fold_metrics.append(float(metric_value) if np.isfinite(metric_value) else np.nan)
                removed_counts.append(int(n_remove))
                remaining_counts.append(int(len(train_idx) - n_remove))

            rows.append(
                {
                    "removal_percentage": float(removal_percentage),
                    "n_removed_mean": _nanmean(removed_counts),
                    "n_train_remaining_mean": _nanmean(remaining_counts),
                    "metric_mean": _nanmean(fold_metrics),
                    "metric_std": _nanstd_ddof1(fold_metrics),
                }
            )

    curve_df = pd.DataFrame(rows, columns=["removal_percentage", "n_removed_mean", "n_train_remaining_mean", "metric_mean", "metric_std"])
    curve_df = curve_df.sort_values("removal_percentage", kind="mergesort").reset_index(drop=True)
    return curve_df


def compare_filtering_curves(
    X: Any,
    y_noisy: Any,
    noise_scores: Mapping[str, Any],
    estimator: Any,
    metric_fn: Callable[..., Any],
    removal_percentages: Sequence[float],
    cv: Any,
    random_state: int = 33,
) -> pd.DataFrame:
    """Evaluate the filtering curve for every method in `noise_scores`."""

    frames: list[pd.DataFrame] = []
    for method, scores in noise_scores.items():
        curve_df = filtering_curve_evaluation(
            X=X,
            y_noisy=y_noisy,
            scores=scores,
            estimator=estimator,
            metric_fn=metric_fn,
            removal_percentages=removal_percentages,
            cv=cv,
            random_state=random_state,
        ).copy()
        curve_df.insert(0, "method", method)
        frames.append(curve_df)

    if not frames:
        return pd.DataFrame(
            columns=["method", "removal_percentage", "n_removed_mean", "n_train_remaining_mean", "metric_mean", "metric_std"]
        )

    return pd.concat(frames, ignore_index=True)


def summarize_filtering_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each method's filtering curve.

    The AUC is computed over the normalized removal fraction, so the score is
    comparable whether the curve was expressed in [0, 1] or [0, 100].
    """

    required = {
        "method",
        "removal_percentage",
        "metric_mean",
    }
    missing = required - set(curve_df.columns)
    if missing:
        raise ValueError(f"curve_df is missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for method, group in curve_df.groupby("method", dropna=False, sort=False):
        group = group.copy()
        removal = pd.to_numeric(group["removal_percentage"], errors="coerce")
        metric = pd.to_numeric(group["metric_mean"], errors="coerce")

        valid = metric.notna()
        if valid.any():
            best_idx = metric[valid].idxmax()
            best_metric = float(metric.loc[best_idx])
            best_removal = float(removal.loc[best_idx])
            mean_metric = float(metric[valid].mean())
        else:
            best_metric = np.nan
            best_removal = np.nan
            mean_metric = np.nan

        zero_mask = np.isclose(removal.to_numpy(dtype=float), 0.0, equal_nan=False)
        metric_at_0 = float(metric[zero_mask].mean()) if np.any(zero_mask & metric.notna().to_numpy()) else np.nan
        max_delta_vs_no_filter = best_metric - metric_at_0 if np.isfinite(best_metric) and np.isfinite(metric_at_0) else np.nan

        auc_metric_curve = np.nan
        valid_curve = valid & removal.notna()
        if int(valid_curve.sum()) >= 2:
            x = removal[valid_curve].to_numpy(dtype=float)
            y = metric[valid_curve].to_numpy(dtype=float)
            order = np.argsort(x, kind="mergesort")
            x = x[order]
            y = y[order]
            if np.max(x) <= 1.0:
                x = x
            else:
                x = x / 100.0
            auc_metric_curve = float(np.trapz(y, x=x))

        rows.append(
            {
                "method": method,
                "best_metric": best_metric,
                "best_removal_percentage": best_removal,
                "mean_metric_across_percentages": mean_metric,
                "auc_metric_curve": auc_metric_curve,
                "metric_at_0_removal": metric_at_0,
                "max_delta_vs_no_filter": max_delta_vs_no_filter,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "best_metric",
            "best_removal_percentage",
            "mean_metric_across_percentages",
            "auc_metric_curve",
            "metric_at_0_removal",
            "max_delta_vs_no_filter",
        ],
    )


def _smoke_test() -> None:
    """Minimal synthetic example used as a built-in smoke test."""

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(33)
    X, y_clean = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        class_sep=1.0,
        random_state=33,
    )
    y_noisy = y_clean.copy()
    noisy_idx = rng.choice(len(y_clean), size=36, replace=False)
    labels = np.unique(y_clean)
    for idx in noisy_idx:
        candidates = labels[labels != y_noisy[idx]]
        y_noisy[idx] = rng.choice(candidates)

    noisy_ground_truth = np.zeros(len(y_clean), dtype=bool)
    noisy_ground_truth[noisy_idx] = True

    good_score = noisy_ground_truth.astype(float) + rng.normal(scale=0.1, size=len(y_clean))
    good_score[5] = np.nan
    random_score = rng.random(len(y_clean))
    constant_score = np.full(len(y_clean), 0.5)

    ranking_df = evaluate_all_noise_scores(
        noisy_ground_truth,
        {
            "good": good_score,
            "random": random_score,
            "constant": constant_score,
        },
    )
    assert set(ranking_df["method"]) == {"good", "random", "constant"}
    assert ranking_df["roc_auc"].notna().any()

    curve_df = compare_filtering_curves(
        X,
        y_noisy,
        {"good": good_score, "random": random_score},
        LogisticRegression(max_iter=500),
        balanced_accuracy_score,
        removal_percentages=[0, 10, 20, 30],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=33),
        random_state=33,
    )
    summary_df = summarize_filtering_curve(curve_df)
    assert not curve_df.empty
    assert not summary_df.empty


if __name__ == "__main__":  # pragma: no cover
    _smoke_test()
