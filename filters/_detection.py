"""Helpers shared by noise filters that support detection mode."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_action(action: str) -> None:
    """Validate a filter action.

    Detection is supported, relabel is not implemented yet.
    """

    if action == "relabel":
        raise NotImplementedError("action='relabel' is not implemented yet.")
    if action not in {"remove", "detect"}:
        raise ValueError("action must be 'remove' or 'detect'")


def resample_by_action(X, y, action: str, keep_mask: np.ndarray):
    """Apply the post-fit action to the data."""

    if action == "remove":
        return X[keep_mask], y[keep_mask]
    if action == "detect":
        return X, y
    validate_action(action)
    raise AssertionError("Unreachable")


def attach_detection_report(estimator, noisy_mask: np.ndarray, *, noise_score=None, **details: Any):
    """Store common detection attributes and a full report."""

    noisy_mask = np.asarray(noisy_mask, dtype=bool)
    noisy_indices = np.flatnonzero(noisy_mask)
    report: dict[str, Any] = {
        "n_samples": int(noisy_mask.shape[0]),
        "n_noisy": int(noisy_indices.size),
        "noisy_indices": noisy_indices,
        "noisy_mask": noisy_mask,
        "noisy_fraction": float(noisy_mask.mean()),
        "action": getattr(estimator, "action", None),
    }

    if noise_score is not None:
        noise_score_arr = np.asarray(noise_score)
        estimator.noise_score_ = noise_score_arr
        report["noise_score"] = noise_score_arr
    else:
        estimator.noise_score_ = None

    report.update(details)

    estimator.noisy_mask_ = noisy_mask
    estimator.noisy_indices_ = noisy_indices
    estimator.noisy_fraction_ = float(noisy_mask.mean())
    estimator.detection_report_ = report
    return report
