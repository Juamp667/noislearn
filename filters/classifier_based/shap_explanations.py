"""External SHAP explanations for classification-based noise filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.validation import check_consistent_length, check_is_fitted
except Exception:  # pragma: no cover - fallback for lightweight environments.
    def clone(estimator):
        return estimator


    class StratifiedKFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None):
            self.n_splits = int(n_splits)
            self.shuffle = bool(shuffle)
            self.random_state = random_state

        def split(self, X, y):
            y_arr = np.asarray(y)
            n_samples = int(y_arr.shape[0])
            if self.n_splits < 2:
                raise ValueError("n_splits must be >= 2")

            rng = np.random.default_rng(None if self.random_state is None else int(self.random_state))
            all_indices = np.arange(n_samples)
            fold_indices = [[] for _ in range(self.n_splits)]

            for label in np.unique(y_arr):
                label_indices = np.flatnonzero(y_arr == label).tolist()
                if self.shuffle:
                    rng.shuffle(label_indices)
                for pos, idx in enumerate(label_indices):
                    fold_indices[pos % self.n_splits].append(int(idx))

            for fold in fold_indices:
                test_idx = np.asarray(sorted(fold), dtype=int)
                train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
                yield train_idx, test_idx


    def check_consistent_length(*arrays):
        lengths = [len(np.asarray(arr)) for arr in arrays if arr is not None]
        if len(set(lengths)) > 1:
            raise ValueError("Found input variables with inconsistent numbers of samples.")


    def check_is_fitted(estimator, attributes):
        missing = [attr for attr in attributes if not hasattr(estimator, attr)]
        if missing:
            raise AttributeError(f"{type(estimator).__name__} is not fitted yet. Missing attributes: {missing}")

try:
    from .classification import ClassificationFilter
except Exception:  # pragma: no cover - fallback when sklearn-based modules are unavailable.
    class ClassificationFilter:  # type: ignore[too-many-ancestors]
        pass


@dataclass
class ClassificationFilterSHAPExplanation:
    """Local SHAP explanation for one noisy or selected sample."""

    sample_idx: int
    fold_idx: int
    true_label_idx: int
    true_label: Any
    oof_pred_idx: int
    oof_pred: Any
    target_class_idx: int
    target_class: Any
    confidence: float | None
    noise_score: float | None
    is_noisy: bool
    base_value: float | None
    shap_values: np.ndarray
    top_k: list[tuple[Any, float]]
    figure: Any | None = None


@dataclass
class ClassificationFilterSHAPReport:
    """Container for SHAP explanations computed from a ClassificationFilter."""

    items: list[ClassificationFilterSHAPExplanation]
    sample_indices: np.ndarray
    noisy_indices: np.ndarray
    noisy_only: bool
    sort_by: str
    ascending: bool
    class_index: Any
    top_k: int | None
    algorithm: str
    background_size: int | None
    background_random_state: int | None
    max_evals: int | None
    max_display: int
    return_figures: bool
    show_figures: bool
    feature_names: np.ndarray
    n_samples: int
    n_features: int
    cv: int
    random_state: int | None
    estimator_name: str

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def by_fold(self):
        grouped: dict[int, list[ClassificationFilterSHAPExplanation]] = {}
        for item in self.items:
            grouped.setdefault(item.fold_idx, []).append(item)
        return grouped


def explain_classification_filter_noisy_instances(
    fitted_filter: ClassificationFilter,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    *,
    noisy_only: bool = True,
    class_index: Any = "predicted",
    top_k: int | None = 5,
    feature_names: Sequence[Any] | None = None,
    background_size: int | None = None,
    background_random_state: int | None = None,
    algorithm: str = "auto",
    max_evals: int | None = None,
    return_figures: bool = False,
    show_figures: bool = False,
    max_display: int = 10,
    sort_by: str = "confidence",
    ascending: bool = True,
):
    """Explain OOF decisions from a fitted ``ClassificationFilter`` with SHAP.

    The helper refits each fold model, computes SHAP values on the requested
    samples and stores the resulting report on ``fitted_filter.shap_report_``.

    Parameters
    ----------
    fitted_filter : ClassificationFilter
        A fitted filter instance.
    sample_indices : sequence of int, ndarray, or None, default=None
        Optional subset of samples to explain.
    noisy_only : bool, default=True
        If True, only explain samples flagged as noisy by the filter.
    class_index : {"predicted", "true"}, int, label, default="predicted"
        Target class to explain.
    top_k : int or None, default=5
        Number of strongest feature contributions to keep per sample, ranked by
        absolute SHAP value.
        The same ordering is used in the optional figures.
    feature_names : sequence, default=None
        Optional names for the input features.
    background_size : int or None, default=None
        Number of training samples used as SHAP background per fold. If None,
        the whole training fold is used.
    background_random_state : int or None, default=None
        Seed used to subsample the SHAP background.
    algorithm : str, default="auto"
        SHAP explainer algorithm forwarded to ``shap.Explainer``.
    max_evals : int or None, default=None
        Maximum SHAP evaluations per explained sample. If omitted, SHAP decides
        the exact number automatically.
    return_figures : bool, default=False
        Store a Matplotlib figure in each explanation item.
    show_figures : bool, default=False
        Display each figure as it is generated.
    max_display : int, default=10
        Maximum number of features shown in the waterfall plot.
    sort_by : {"confidence", "noise_score", "fold_idx", "sample_idx"}, default="confidence"
        Sort key for the returned explanations.
    ascending : bool, default=True
        Sort direction.

    Notes
    -----
    Each fold is refit internally. For stochastic estimators, set their
    ``random_state`` to a fixed value to make the explanation reproducible.
    """

    check_is_fitted(fitted_filter, ["result_", "X_", "y_", "classes_"])

    try:
        import shap
    except ModuleNotFoundError as e:
        raise ImportError("shap is required for SHAP explanations. Install shap and retry.") from e

    if not hasattr(fitted_filter.estimator, "predict_proba"):
        raise ValueError("ClassificationFilter SHAP explanations require an estimator with predict_proba().")

    if background_size is not None and int(background_size) < 1:
        raise ValueError("background_size must be >= 1 when provided")
    if max_evals is not None and int(max_evals) < 1:
        raise ValueError("max_evals must be >= 1 when provided")
    if int(max_display) < 1:
        raise ValueError("max_display must be >= 1")
    if sort_by not in {"confidence", "noise_score", "fold_idx", "sample_idx"}:
        raise ValueError("sort_by must be 'confidence', 'noise_score', 'fold_idx' or 'sample_idx'")

    X = np.asarray(fitted_filter.X_)
    y = np.asarray(fitted_filter.y_)
    classes = np.asarray(fitted_filter.classes_, dtype=object)
    n_samples, n_features = X.shape
    filter_random_state = fitted_filter.random_state

    if int(fitted_filter.cv) < 2:
        raise ValueError("cv must be >= 2")

    feature_names_arr = _resolve_feature_names(feature_names, n_features)
    y_idx = _labels_to_indices(y, classes)
    oof_pred_idx = _labels_to_indices(np.asarray(fitted_filter.result_.oof_pred, dtype=object), classes)
    noisy_mask = ~np.asarray(fitted_filter.result_.keep_mask, dtype=bool)

    max_evals_eff = None if max_evals is None else int(max_evals)

    selected_indices = _normalize_sample_indices(sample_indices, n_samples)
    if noisy_only:
        selected_indices = [idx for idx in selected_indices if bool(noisy_mask[idx])]

    if not selected_indices:
        report = ClassificationFilterSHAPReport(
            items=[],
            sample_indices=np.asarray(selected_indices, dtype=int),
            noisy_indices=np.asarray([], dtype=int),
            noisy_only=bool(noisy_only),
            sort_by=sort_by,
            ascending=bool(ascending),
            class_index=class_index,
            top_k=None if top_k is None else int(top_k),
            algorithm=algorithm,
            background_size=None if background_size is None else int(background_size),
            background_random_state=None if background_random_state is None else int(background_random_state),
            max_evals=max_evals_eff,
            max_display=int(max_display),
            return_figures=bool(return_figures),
            show_figures=bool(show_figures),
            feature_names=feature_names_arr,
            n_samples=int(n_samples),
            n_features=int(n_features),
            cv=int(fitted_filter.cv),
            random_state=None if filter_random_state is None else int(filter_random_state),
            estimator_name=type(fitted_filter.estimator).__name__,
        )
        fitted_filter.shap_report_ = report
        fitted_filter.explanation_report_ = report
        return report


    # Refit each fold to match the original OOF split and explain the selected samples.
    skf = StratifiedKFold(n_splits=int(fitted_filter.cv), shuffle=True, random_state=filter_random_state)
    fold_lookup = np.full(n_samples, -1, dtype=int)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_idx)):
        fold_lookup[test_idx] = fold_idx
        fold_splits.append((np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)))

    if np.any(fold_lookup < 0):
        raise RuntimeError("Internal error: some samples were not assigned to any fold.")

    grouped_indices: dict[int, list[int]] = {}
    for sample_idx in selected_indices:
        fold_idx = int(fold_lookup[sample_idx])
        grouped_indices.setdefault(fold_idx, []).append(int(sample_idx))

    seed_base = filter_random_state if background_random_state is None else background_random_state
    if seed_base is not None:
        seed_base = int(seed_base)

    items: list[ClassificationFilterSHAPExplanation] = []
    for fold_idx in sorted(grouped_indices):
        train_idx, _ = fold_splits[fold_idx]
        fold_y_idx = y_idx[train_idx]
        rng = np.random.default_rng(None if seed_base is None else seed_base + fold_idx)
        background_idx = _select_background_indices(train_idx, fold_y_idx, background_size, rng, classes.shape[0])

        model = clone(fitted_filter.estimator)
        model.fit(X[train_idx], y_idx[train_idx])

        background = X[background_idx]
        masker = shap.maskers.Independent(background, max_samples=background.shape[0])
        explainer = shap.Explainer(
            model.predict_proba, 
            masker, 
            algorithm=algorithm, 
            output_names=classes
        )

        batch_indices = np.asarray(grouped_indices[fold_idx], dtype=int)
        batch_proba = model.predict_proba(X[batch_indices])
        batch_explanation = explainer(X[batch_indices], max_evals=max_evals_eff)

        for pos, sample_idx in enumerate(batch_indices):
            sample_explanation = batch_explanation[pos]
            target_class_idx = _resolve_target_class_index(
                class_index=class_index,
                sample_idx=int(sample_idx),
                oof_pred_idx=oof_pred_idx,
                y_idx=y_idx,
                classes=classes,
            )
            shap_values, base_value = _extract_sample_shap_values(sample_explanation, target_class_idx)
            top_features = _top_k_from_values(shap_values, feature_names_arr, top_k)
            confidence = _sample_confidence(batch_proba, pos, target_class_idx)
            noise_score = _sample_noise_score(fitted_filter, int(sample_idx))

            figure = None
            if return_figures or show_figures:
                figure = _plot_shap_waterfall(
                    shap_values,
                    data=X[sample_idx],
                    feature_names=feature_names_arr,
                    title=f"Sample {int(sample_idx)} | fold {int(fold_idx)} | class {_scalar(classes[target_class_idx])}",
                    base_value=base_value,
                    max_display=int(max_display),
                )
                if figure is not None and (show_figures or not return_figures):
                    try:
                        import matplotlib.pyplot as plt
                    except ModuleNotFoundError as e:
                        raise ImportError("matplotlib is required when figures are requested.") from e
                    if show_figures:
                        plt.show()
                    if not return_figures:
                        plt.close(figure)
                        figure = None

            items.append(
                ClassificationFilterSHAPExplanation(
                    sample_idx=int(sample_idx),
                    fold_idx=int(fold_idx),
                    true_label_idx=int(y_idx[sample_idx]),
                    true_label=_scalar(classes[y_idx[sample_idx]]),
                    oof_pred_idx=int(oof_pred_idx[sample_idx]),
                    oof_pred=_scalar(classes[oof_pred_idx[sample_idx]]),
                    target_class_idx=int(target_class_idx),
                    target_class=_scalar(classes[target_class_idx]),
                    confidence=confidence,
                    noise_score=noise_score,
                    is_noisy=bool(noisy_mask[sample_idx]),
                    base_value=base_value,
                    shap_values=shap_values,
                    top_k=top_features,
                    figure=figure,
                )
            )

    items = _sort_items(items, sort_by=sort_by, ascending=ascending)
    report = ClassificationFilterSHAPReport(
        items=items,
        sample_indices=np.asarray(selected_indices, dtype=int),
        noisy_indices=np.asarray([item.sample_idx for item in items if item.is_noisy], dtype=int),
        noisy_only=bool(noisy_only),
        sort_by=sort_by,
        ascending=bool(ascending),
        class_index=class_index,
        top_k=None if top_k is None else int(top_k),
        algorithm=algorithm,
        background_size=None if background_size is None else int(background_size),
        background_random_state=None if background_random_state is None else int(background_random_state),
        max_evals=max_evals_eff,
        max_display=int(max_display),
        return_figures=bool(return_figures),
        show_figures=bool(show_figures),
        feature_names=feature_names_arr,
        n_samples=int(n_samples),
        n_features=int(n_features),
        cv=int(fitted_filter.cv),
        random_state=None if filter_random_state is None else int(filter_random_state),
        estimator_name=type(fitted_filter.estimator).__name__,
    )
    fitted_filter.shap_report_ = report
    fitted_filter.explanation_report_ = report
    return report


def _resolve_feature_names(feature_names: Sequence[Any] | None, n_features: int):
    if feature_names is None:
        return np.asarray(range(n_features), dtype=object)
    names = np.asarray(list(feature_names), dtype=object)
    if names.shape[0] != n_features:
        raise ValueError("feature name count does not match n_features")
    return names


def _normalize_sample_indices(sample_indices: Sequence[int] | np.ndarray | None, n_samples: int):
    if sample_indices is None:
        return list(range(n_samples))

    indices = np.asarray(sample_indices)
    if indices.ndim == 0:
        indices = indices.reshape(1)

    if indices.dtype == bool:
        if indices.shape[0] != n_samples:
            raise ValueError("Boolean sample_indices mask must match n_samples")
        return list(np.flatnonzero(indices))

    indices = indices.astype(int, copy=False).ravel()
    if np.any((indices < 0) | (indices >= n_samples)):
        raise IndexError("sample_indices contains out-of-range values")
    return list(indices)


def _labels_to_indices(labels: np.ndarray, classes: np.ndarray):
    labels = np.asarray(labels, dtype=object).ravel()
    classes = np.asarray(classes, dtype=object).ravel()
    indices = np.empty(labels.shape[0], dtype=int)
    for i, label in enumerate(labels):
        matches = np.flatnonzero(classes == label)
        if matches.size == 0:
            raise ValueError(f"Label {label!r} is not present in the fitted classes.")
        indices[i] = int(matches[0])
    return indices


def _resolve_target_class_index(class_index: Any, sample_idx: int, oof_pred_idx: np.ndarray, y_idx: np.ndarray, classes: np.ndarray):
    if isinstance(class_index, str):
        key = class_index.lower()
        if key in {"predicted", "prediction", "oof_pred", "oof_predicted"}:
            return int(oof_pred_idx[sample_idx])
        if key in {"true", "label", "y_true", "oof_true"}:
            return int(y_idx[sample_idx])
        raise ValueError("class_index must be 'predicted', 'true', an int or a known label")

    if class_index is None:
        return 1 if int(classes.shape[0]) > 1 else 0

    matches = np.flatnonzero(np.asarray(classes, dtype=object) == class_index)
    if matches.size:
        return int(matches[0])

    idx = int(class_index)
    if idx < 0 or idx >= int(classes.shape[0]):
        raise ValueError("class_index is out of range for the fitted classes")
    return idx


def _select_background_indices(train_idx: np.ndarray, y_train_idx: np.ndarray, background_size: int | None, rng: np.random.Generator, n_classes: int):
    train_idx = np.asarray(train_idx, dtype=int)
    y_train_idx = np.asarray(y_train_idx, dtype=int)

    if background_size is None or int(background_size) >= train_idx.shape[0]:
        return train_idx

    size = int(background_size)
    if size < 1:
        raise ValueError("background_size must be >= 1 when provided")

    if size < n_classes:
        return np.asarray(rng.choice(train_idx, size=size, replace=False), dtype=int)

    selected: list[int] = []
    for class_idx in np.unique(y_train_idx):
        class_positions = np.flatnonzero(y_train_idx == class_idx)
        if class_positions.size == 0:
            continue
        pos = int(rng.choice(class_positions))
        selected.append(int(train_idx[pos]))

    if len(selected) < size:
        chosen = np.asarray(selected, dtype=int)
        remaining_pool = train_idx[~np.isin(train_idx, chosen)]
        if remaining_pool.size > 0:
            extra = rng.choice(remaining_pool, size=min(size - len(selected), remaining_pool.size), replace=False)
            selected.extend(np.asarray(extra, dtype=int).tolist())

    return np.asarray(selected[:size], dtype=int)


def _sample_confidence(probabilities: np.ndarray, sample_pos: int, target_class_idx: int):
    value = probabilities[sample_pos, target_class_idx]
    return None if not np.isfinite(value) else float(value)


def _sample_noise_score(fitted_filter: ClassificationFilter, sample_idx: int):
    noise_score = getattr(fitted_filter, "noise_score_", None)
    if noise_score is None:
        return None
    value = noise_score[sample_idx]
    return None if not np.isfinite(value) else float(value)


def _extract_sample_shap_values(sample_explanation, target_class_idx: int):
    values = sample_explanation.values # Valores shap de cada atributo
    base_values = getattr(sample_explanation, "base_values", None)  # Valor shap base (background)

    if isinstance(values, (list, tuple)):
        # Extraigo los valores shap asociados a la clase cuya predicción deseo explicar
        contrib = np.asarray(values[target_class_idx], dtype=float)
        return contrib.reshape(-1), _select_base_value(base_values, target_class_idx)

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1), _select_base_value(base_values, target_class_idx)
    if arr.ndim != 2:
        raise TypeError("Unsupported SHAP value shape for a single sample.")

    base_size = None
    if base_values is not None:
        base_size = np.asarray(base_values).reshape(-1).size

    # Si se trata de un problema multiclase (más de un base_value)
    if base_size is not None and base_size > 1:
        if arr.shape[1] == base_size and target_class_idx < arr.shape[1]:
            contrib = arr[:, target_class_idx]
        elif arr.shape[0] == base_size and target_class_idx < arr.shape[0]:
            contrib = arr[target_class_idx, :]
        elif target_class_idx < arr.shape[1]:
            contrib = arr[:, target_class_idx]
        elif target_class_idx < arr.shape[0]:
            contrib = arr[target_class_idx, :]
        else:
            raise IndexError("target_class_idx is out of range for the SHAP values")
    else:
        if arr.shape[1] == 1:
            contrib = arr[:, 0]
        elif arr.shape[0] == 1:
            contrib = arr[0, :]
        elif target_class_idx < arr.shape[1]:
            contrib = arr[:, target_class_idx]
        elif target_class_idx < arr.shape[0]:
            contrib = arr[target_class_idx, :]
        else:
            contrib = arr.reshape(-1)

    return np.asarray(contrib, dtype=float).reshape(-1), _select_base_value(base_values, target_class_idx)


def _select_base_value(base_values, target_class_idx: int):
    if base_values is None:
        return None
    arr = np.asarray(base_values, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return float(arr[0])
    if target_class_idx < arr.size:
        return float(arr[target_class_idx])
    return float(arr[0])


def _top_k_from_values(shap_values: np.ndarray, feature_names: np.ndarray, top_k: int | None):
    if not top_k or int(top_k) <= 0:
        return []

    values = np.asarray(shap_values, dtype=float).reshape(-1)
    k = min(int(top_k), values.shape[0])
    order = np.argsort(np.abs(values))[::-1][:k]
    return [(_scalar(feature_names[int(idx)]), float(values[int(idx)])) for idx in order]


def _plot_shap_waterfall(
    shap_values: np.ndarray,
    data: np.ndarray,
    feature_names: np.ndarray,
    *,
    title: str,
    base_value: float | None = None,
    max_display: int = 10,
):
    try:
        import shap
    except ModuleNotFoundError as e:
        raise ImportError("shap is required when figures are requested.") from e

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ImportError("matplotlib is required when figures are requested.") from e

    values = np.asarray(shap_values, dtype=float).reshape(-1)
    if values.size == 0:
        return None

    display_n = min(int(max_display), values.size)
    explanation = shap.Explanation(
        values=values,
        base_values=0.0 if base_value is None else float(base_value),
        data=np.asarray(data, dtype=float).reshape(-1),
        feature_names=[_scalar(name) for name in feature_names],
    )

    axes = None
    plotter = getattr(explanation, "plot_waterfall", None)
    if callable(plotter):
        axes = plotter(max_display=display_n, show=False)
    else:
        axes = shap.plots.waterfall(explanation, max_display=display_n, show=False)

    if axes is None:
        axes = plt.gca()

    if title:
        axes.set_title(title)
    return getattr(axes, "figure", axes)


def _sort_items(items: list[ClassificationFilterSHAPExplanation], *, sort_by: str, ascending: bool):
    if sort_by == "confidence":
        valid = [item for item in items if item.confidence is not None and not np.isnan(item.confidence)]
        missing = [item for item in items if item.confidence is None or np.isnan(item.confidence)]
        valid.sort(key=lambda item: float(item.confidence), reverse=not ascending)
        return valid + missing

    if sort_by == "noise_score":
        valid = [item for item in items if item.noise_score is not None and not np.isnan(item.noise_score)]
        missing = [item for item in items if item.noise_score is None or np.isnan(item.noise_score)]
        valid.sort(key=lambda item: float(item.noise_score), reverse=not ascending)
        return valid + missing

    if sort_by == "fold_idx":
        return sorted(items, key=lambda item: item.fold_idx, reverse=not ascending)

    if sort_by == "sample_idx":
        return sorted(items, key=lambda item: item.sample_idx, reverse=not ascending)

    raise ValueError("sort_by must be 'confidence', 'noise_score', 'fold_idx' or 'sample_idx'")


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ShapDifferenceInstanceExplanation:
    """Local explanation for one potentially noisy instance."""

    instance_index: int
    observed_label: Any
    predicted_label: Any
    explanatory_noise_score: float
    class_difference: dict[str, Any]
    top_features: pd.DataFrame
    fold_idx: int | None = None
    observed_label_idx: int | None = None
    predicted_label_idx: int | None = None
    confidence: float | None = None
    noise_score: float | None = None
    is_noisy: bool | None = None
    base_value_observed: float | None = None
    base_value_predicted: float | None = None


@dataclass
class ShapDifferenceReport:
    """Collection of SHAP difference explanations for noisy instances."""

    items: list[ShapDifferenceInstanceExplanation]
    summary: pd.DataFrame
    feature_names: np.ndarray
    class_labels: np.ndarray
    noisy_mask: np.ndarray
    top_k: int | None
    n_samples: int
    n_features: int
    sample_indices: np.ndarray | None = None
    noisy_indices: np.ndarray | None = None
    noisy_only: bool | None = None
    sort_by: str | None = None
    ascending: bool | None = None
    algorithm: str | None = None
    background_size: int | None = None
    background_random_state: int | None = None
    max_evals: int | None = None
    cv: int | None = None
    random_state: int | None = None
    estimator_name: str | None = None

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def to_frame(self):
        return self.summary.copy()


_EXPLANATION_TOL = 1e-12


def _normalize_label_array(labels: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(labels, dtype=object).ravel()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence.")
    return arr


def _resolve_label_index(labels: np.ndarray, label: Any, *, label_name: str) -> int:
    matches = np.flatnonzero(labels == label)
    if matches.size == 0:
        raise ValueError(f"{label_name} {label!r} is not present in class_labels.")
    return int(matches[0])


def _sanitize_vector(values: Any, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be numeric and 1D-like.") from exc
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_classwise_shap_values(shap_values: Any, class_labels: Sequence[Any]) -> np.ndarray:
    """Normalize per-instance SHAP values to a (n_classes, n_features) matrix."""

    class_labels_arr = _normalize_label_array(class_labels, name="class_labels")
    n_classes = int(class_labels_arr.size)

    if hasattr(shap_values, "values") and not isinstance(shap_values, (list, tuple, np.ndarray)):
        shap_values = shap_values.values

    if isinstance(shap_values, (list, tuple)):
        rows = [_sanitize_vector(values, name=f"shap_values[{idx}]") for idx, values in enumerate(shap_values)]
        if not rows:
            raise ValueError("shap_values must contain at least one class-specific vector.")
        n_features = rows[0].shape[0]
        for idx, row in enumerate(rows[1:], start=1):
            if row.shape[0] != n_features:
                raise ValueError(
                    f"All class-specific SHAP vectors must have the same length. "
                    f"Got shap_values[0].shape[-1]={n_features} and shap_values[{idx}].shape[-1]={row.shape[0]}."
                )
        if len(rows) != n_classes:
            raise ValueError(
                f"The number of SHAP class vectors ({len(rows)}) does not match class_labels ({n_classes})."
            )
        return np.vstack(rows)

    try:
        arr = np.asarray(shap_values, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("shap_values must be array-like or expose a numeric .values attribute.") from exc

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if n_classes == 1:
            return np.nan_to_num(arr.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        raise ValueError(
            "shap_values must provide class-specific contributions; a single 1D vector is not enough for class comparison."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported SHAP value shape {arr.shape}. Expected a 2D class-by-feature matrix after squeezing."
        )

    if arr.shape[0] == n_classes and arr.shape[1] != n_classes:
        classwise = arr
    elif arr.shape[1] == n_classes and arr.shape[0] != n_classes:
        classwise = arr.T
    elif arr.shape[0] == n_classes and arr.shape[1] == n_classes:
        # Ambiguous square matrix: assume the first axis indexes classes.
        classwise = arr
    else:
        raise ValueError(
            f"Could not infer the class axis from SHAP shape {arr.shape} and {n_classes} class labels."
        )

    return np.nan_to_num(np.asarray(classwise, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _class_difference_direction(delta_phi: float) -> str:
    if abs(float(delta_phi)) <= _EXPLANATION_TOL:
        return "neutral"
    return "supports_predicted" if float(delta_phi) > 0.0 else "supports_observed"


def _class_difference_interpretation(phi_predicted: float, phi_observed: float) -> str:
    pred_pos = float(phi_predicted) > _EXPLANATION_TOL
    pred_neg = float(phi_predicted) < -_EXPLANATION_TOL
    obs_pos = float(phi_observed) > _EXPLANATION_TOL
    obs_neg = float(phi_observed) < -_EXPLANATION_TOL
    pred_zero = not (pred_pos or pred_neg)
    obs_zero = not (obs_pos or obs_neg)

    if pred_zero and obs_zero:
        return "Atributo localmente poco relevante para diferenciar ambas clases."
    if pred_pos and obs_neg:
        return "Favorece la clase predicha y penaliza la etiqueta observada."
    if pred_pos and obs_pos:
        return "Favorece ambas clases, pero mas intensamente la clase con mayor SHAP."
    if pred_neg and obs_pos:
        return "Penaliza la clase predicha y favorece la etiqueta observada."
    if pred_neg and obs_neg:
        return "Penaliza ambas clases, pero con distinta intensidad."
    if pred_pos and obs_zero:
        return "Favorece la clase predicha y es neutro para la etiqueta observada."
    if pred_neg and obs_zero:
        return "Penaliza la clase predicha y es neutro para la etiqueta observada."
    if pred_zero and obs_pos:
        return "Es neutro para la clase predicha y favorece la etiqueta observada."
    if pred_zero and obs_neg:
        return "Es neutro para la clase predicha y penaliza la etiqueta observada."
    return "Atributo localmente poco relevante para diferenciar ambas clases."


def _build_difference_frame(
    class_difference: dict[str, Any],
    feature_names: Sequence[Any],
    feature_values: Any,
    *,
    top_k: int | None,
) -> pd.DataFrame:
    phi_observed = np.asarray(class_difference["phi_observed"], dtype=float).reshape(-1)
    phi_predicted = np.asarray(class_difference["phi_predicted"], dtype=float).reshape(-1)
    delta_phi = np.asarray(class_difference["delta_phi"], dtype=float).reshape(-1)
    abs_delta_phi = np.asarray(class_difference["abs_delta_phi"], dtype=float).reshape(-1)

    feature_names_arr = np.asarray(feature_names, dtype=object).ravel()
    feature_values_arr = np.asarray(feature_values).reshape(-1)

    if feature_names_arr.size != phi_observed.size:
        raise ValueError(
            f"feature_names length ({feature_names_arr.size}) does not match the SHAP feature dimension ({phi_observed.size})."
        )
    if feature_values_arr.size != phi_observed.size:
        raise ValueError(
            f"feature_values length ({feature_values_arr.size}) does not match the SHAP feature dimension ({phi_observed.size})."
        )

    if top_k is None:
        keep_n = int(phi_observed.size)
    else:
        keep_n = int(top_k)
        if keep_n < 1:
            raise ValueError("top_k must be >= 1 when provided.")
        keep_n = min(keep_n, int(phi_observed.size))

    order = np.argsort(np.nan_to_num(abs_delta_phi, nan=-np.inf), kind="mergesort")[::-1]
    order = order[:keep_n]

    rows = []
    for idx in order:
        phi_obs = float(phi_observed[idx])
        phi_pred = float(phi_predicted[idx])
        delta = float(delta_phi[idx])
        abs_delta = float(abs_delta_phi[idx])
        rows.append(
            {
                "feature": _scalar(feature_names_arr[idx]),
                "value": _scalar(feature_values_arr[idx]),
                "phi_observed": phi_obs,
                "phi_predicted": phi_pred,
                "delta_phi": delta,
                "abs_delta_phi": abs_delta,
                "direction": _class_difference_direction(delta),
                "interpretation": _class_difference_interpretation(phi_pred, phi_obs),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "value",
            "phi_observed",
            "phi_predicted",
            "delta_phi",
            "abs_delta_phi",
            "direction",
            "interpretation",
        ],
    )


def compute_shap_class_difference(
    shap_values: Any,
    class_labels: Sequence[Any],
    observed_label: Any,
    predicted_label: Any,
) -> dict[str, Any]:
    """Compare SHAP attributions for the observed and predicted class.

    The returned vectors are sanitized to keep the explanation robust when the
    underlying SHAP output contains NaN or inf values.
    """

    class_labels_arr = _normalize_label_array(class_labels, name="class_labels")
    observed_idx = _resolve_label_index(class_labels_arr, observed_label, label_name="observed_label")
    predicted_idx = _resolve_label_index(class_labels_arr, predicted_label, label_name="predicted_label")

    classwise = _normalize_classwise_shap_values(shap_values, class_labels_arr)
    phi_observed = np.asarray(classwise[observed_idx], dtype=float).reshape(-1)
    phi_predicted = np.asarray(classwise[predicted_idx], dtype=float).reshape(-1)
    delta_phi = phi_predicted - phi_observed
    abs_delta_phi = np.abs(delta_phi)

    return {
        "observed_label": _scalar(observed_label),
        "predicted_label": _scalar(predicted_label),
        "phi_observed": phi_observed.copy(),
        "phi_predicted": phi_predicted.copy(),
        "delta_phi": delta_phi.copy(),
        "abs_delta_phi": abs_delta_phi.copy(),
    }


def explain_instance_shap_difference(
    shap_values: Any,
    class_labels: Sequence[Any],
    feature_names: Sequence[Any],
    feature_values: Any,
    observed_label: Any,
    predicted_label: Any,
    top_k: int | None = 5,
) -> pd.DataFrame:
    """Explain one instance by contrasting observed-vs-predicted SHAP values."""

    class_difference = compute_shap_class_difference(
        shap_values=shap_values,
        class_labels=class_labels,
        observed_label=observed_label,
        predicted_label=predicted_label,
    )
    return _build_difference_frame(
        class_difference,
        feature_names,
        feature_values,
        top_k=top_k,
    )


def compute_explanatory_noise_score(
    delta_phi: Any,
    phi_observed: Any,
    phi_predicted: Any,
    eps: float = 1e-12,
) -> float:
    """Measure the strength of the SHAP shift between two classes.

    This score complements the filter's ``noise_score``: the filter decides how
    suspicious an instance looks, while this score measures how much the local
    attribution pattern changes between the observed and predicted classes.
    """

    delta_arr = np.nan_to_num(np.asarray(delta_phi, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    obs_arr = np.nan_to_num(np.asarray(phi_observed, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    pred_arr = np.nan_to_num(np.asarray(phi_predicted, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    if not (delta_arr.size == obs_arr.size == pred_arr.size):
        raise ValueError("delta_phi, phi_observed and phi_predicted must have the same length.")

    numerator = float(np.sum(np.abs(delta_arr)))
    denominator = float(np.sum(np.abs(obs_arr)) + np.sum(np.abs(pred_arr)) + float(eps))
    return numerator / denominator


def _sort_difference_items(
    items: list[ShapDifferenceInstanceExplanation],
    *,
    sort_by: str,
    ascending: bool,
):
    if sort_by in {"sample_idx", "instance_index"}:
        return sorted(items, key=lambda item: item.instance_index, reverse=not ascending)

    if sort_by == "fold_idx":
        return sorted(items, key=lambda item: -1 if item.fold_idx is None else item.fold_idx, reverse=not ascending)

    if sort_by in {"confidence", "noise_score", "explanatory_noise_score"}:
        valid = []
        missing = []
        for item in items:
            value = getattr(item, sort_by)
            if value is None or np.isnan(value):
                missing.append(item)
            else:
                valid.append(item)
        valid.sort(key=lambda item: float(getattr(item, sort_by)), reverse=not ascending)
        return valid + missing

    raise ValueError("sort_by must be 'confidence', 'noise_score', 'explanatory_noise_score', 'fold_idx' or 'sample_idx'")


def _classification_filter_difference_summary_columns(summary_depth: int):
    columns = [
        "instance_index",
        "fold_idx",
        "observed_label",
        "predicted_label",
        "confidence",
        "noise_score",
        "explanatory_noise_score",
        "is_noisy",
    ]
    for rank in range(1, int(summary_depth) + 1):
        columns.extend([f"top_feature_{rank}", f"top_delta_{rank}"])
    return columns


def _classification_filter_difference_summary_row(item: ShapDifferenceInstanceExplanation, summary_depth: int):
    row = {
        "instance_index": int(item.instance_index),
        "fold_idx": item.fold_idx,
        "observed_label": item.observed_label,
        "predicted_label": item.predicted_label,
        "confidence": item.confidence,
        "noise_score": item.noise_score,
        "explanatory_noise_score": float(item.explanatory_noise_score),
        "is_noisy": item.is_noisy,
    }
    for rank, (_, feature_row) in enumerate(item.top_features.iterrows(), start=1):
        if rank > int(summary_depth):
            break
        row[f"top_feature_{rank}"] = feature_row["feature"]
        row[f"top_delta_{rank}"] = float(feature_row["delta_phi"])
    return row


def explain_classification_filter_shap_difference(
    fitted_filter: ClassificationFilter,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    *,
    noisy_only: bool = True,
    top_k: int | None = 5,
    feature_names: Sequence[Any] | None = None,
    background_size: int | None = None,
    background_random_state: int | None = None,
    algorithm: str = "auto",
    max_evals: int | None = None,
    sort_by: str = "confidence",
    ascending: bool = True,
) -> ShapDifferenceReport:
    """Explain observed-vs-predicted OOF SHAP differences from a fitted filter.

    The function mirrors ``explain_classification_filter_noisy_instances``: it
    reconstructs the original cross-validation splits, refits the fold model and
    computes SHAP on the selected out-of-fold samples. For every sample, it then
    contrasts the SHAP vector of the observed label with the SHAP vector of the
    out-of-fold predicted label.

    By default, ``background_size=None`` uses the full training fold as SHAP
    background for each model. Pass an integer only when a smaller background is
    explicitly desired for speed.
    """

    check_is_fitted(fitted_filter, ["result_", "X_", "y_", "classes_"])

    try:
        import shap
    except ModuleNotFoundError as e:
        raise ImportError("shap is required for SHAP explanations. Install shap and retry.") from e

    if not hasattr(fitted_filter.estimator, "predict_proba"):
        raise ValueError("ClassificationFilter SHAP difference explanations require an estimator with predict_proba().")
    if background_size is not None and int(background_size) < 1:
        raise ValueError("background_size must be >= 1 when provided")
    if max_evals is not None and int(max_evals) < 1:
        raise ValueError("max_evals must be >= 1 when provided")
    if sort_by not in {"confidence", "noise_score", "explanatory_noise_score", "fold_idx", "sample_idx", "instance_index"}:
        raise ValueError(
            "sort_by must be 'confidence', 'noise_score', 'explanatory_noise_score', 'fold_idx' or 'sample_idx'"
        )

    X = np.asarray(fitted_filter.X_)
    y = np.asarray(fitted_filter.y_)
    classes = np.asarray(fitted_filter.classes_, dtype=object)
    n_samples, n_features = X.shape
    filter_random_state = fitted_filter.random_state

    if int(fitted_filter.cv) < 2:
        raise ValueError("cv must be >= 2")

    feature_names_arr = _resolve_feature_names(feature_names, n_features)
    y_idx = _labels_to_indices(y, classes)
    oof_pred_idx = _labels_to_indices(np.asarray(fitted_filter.result_.oof_pred, dtype=object), classes)
    noisy_mask = ~np.asarray(fitted_filter.result_.keep_mask, dtype=bool)
    max_evals_eff = None if max_evals is None else int(max_evals)

    selected_indices = _normalize_sample_indices(sample_indices, n_samples)
    if noisy_only:
        selected_indices = [idx for idx in selected_indices if bool(noisy_mask[idx])]

    summary_depth = n_features if top_k is None else min(int(top_k), n_features)
    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be >= 1 when provided.")
    summary_columns = _classification_filter_difference_summary_columns(summary_depth)

    if not selected_indices:
        summary_df = pd.DataFrame([], columns=summary_columns)
        report = ShapDifferenceReport(
            items=[],
            summary=summary_df,
            feature_names=feature_names_arr,
            class_labels=classes,
            noisy_mask=np.asarray(noisy_mask, dtype=bool),
            top_k=None if top_k is None else int(top_k),
            n_samples=int(n_samples),
            n_features=int(n_features),
            sample_indices=np.asarray(selected_indices, dtype=int),
            noisy_indices=np.asarray([], dtype=int),
            noisy_only=bool(noisy_only),
            sort_by=sort_by,
            ascending=bool(ascending),
            algorithm=algorithm,
            background_size=None if background_size is None else int(background_size),
            background_random_state=None if background_random_state is None else int(background_random_state),
            max_evals=max_evals_eff,
            cv=int(fitted_filter.cv),
            random_state=None if filter_random_state is None else int(filter_random_state),
            estimator_name=type(fitted_filter.estimator).__name__,
        )
        fitted_filter.shap_difference_report_ = report
        return report

    skf = StratifiedKFold(n_splits=int(fitted_filter.cv), shuffle=True, random_state=filter_random_state)
    fold_lookup = np.full(n_samples, -1, dtype=int)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_idx)):
        fold_lookup[test_idx] = fold_idx
        fold_splits.append((np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)))

    if np.any(fold_lookup < 0):
        raise RuntimeError("Internal error: some samples were not assigned to any fold.")

    grouped_indices: dict[int, list[int]] = {}
    for sample_idx in selected_indices:
        fold_idx = int(fold_lookup[sample_idx])
        grouped_indices.setdefault(fold_idx, []).append(int(sample_idx))

    seed_base = filter_random_state if background_random_state is None else background_random_state
    if seed_base is not None:
        seed_base = int(seed_base)

    items: list[ShapDifferenceInstanceExplanation] = []
    for fold_idx in sorted(grouped_indices):
        train_idx, _ = fold_splits[fold_idx]
        fold_y_idx = y_idx[train_idx]
        rng = np.random.default_rng(None if seed_base is None else seed_base + fold_idx)
        background_idx = _select_background_indices(train_idx, fold_y_idx, background_size, rng, classes.shape[0])

        model = clone(fitted_filter.estimator)
        model.fit(X[train_idx], y_idx[train_idx])

        background = X[background_idx]
        masker = shap.maskers.Independent(background, max_samples=background.shape[0])
        explainer = shap.Explainer(
            model.predict_proba,
            masker,
            algorithm=algorithm,
            output_names=classes,
        )

        batch_indices = np.asarray(grouped_indices[fold_idx], dtype=int)
        batch_proba = model.predict_proba(X[batch_indices])
        batch_explanation = explainer(X[batch_indices], max_evals=max_evals_eff)

        for pos, sample_idx in enumerate(batch_indices):
            observed_idx = int(y_idx[sample_idx])
            predicted_idx = int(oof_pred_idx[sample_idx])
            sample_explanation = batch_explanation[pos]
            phi_observed, base_observed = _extract_sample_shap_values(sample_explanation, observed_idx)
            phi_predicted, base_predicted = _extract_sample_shap_values(sample_explanation, predicted_idx)
            phi_observed = np.nan_to_num(phi_observed, nan=0.0, posinf=0.0, neginf=0.0)
            phi_predicted = np.nan_to_num(phi_predicted, nan=0.0, posinf=0.0, neginf=0.0)

            if phi_observed.shape[0] != phi_predicted.shape[0]:
                raise ValueError("Observed and predicted SHAP vectors must have the same feature dimension.")

            delta_phi = phi_predicted - phi_observed
            abs_delta_phi = np.abs(delta_phi)
            class_difference = {
                "observed_label": _scalar(classes[observed_idx]),
                "predicted_label": _scalar(classes[predicted_idx]),
                "phi_observed": phi_observed.copy(),
                "phi_predicted": phi_predicted.copy(),
                "delta_phi": delta_phi.copy(),
                "abs_delta_phi": abs_delta_phi.copy(),
                "base_value_observed": base_observed,
                "base_value_predicted": base_predicted,
            }
            feature_table = _build_difference_frame(
                class_difference,
                feature_names_arr,
                X[sample_idx],
                top_k=top_k,
            )
            explanatory_noise_score = compute_explanatory_noise_score(
                delta_phi,
                phi_observed,
                phi_predicted,
            )
            confidence = _sample_confidence(batch_proba, pos, predicted_idx)
            noise_score = _sample_noise_score(fitted_filter, int(sample_idx))

            items.append(
                ShapDifferenceInstanceExplanation(
                    instance_index=int(sample_idx),
                    observed_label=_scalar(classes[observed_idx]),
                    predicted_label=_scalar(classes[predicted_idx]),
                    explanatory_noise_score=float(explanatory_noise_score),
                    class_difference=class_difference,
                    top_features=feature_table,
                    fold_idx=int(fold_idx),
                    observed_label_idx=observed_idx,
                    predicted_label_idx=predicted_idx,
                    confidence=confidence,
                    noise_score=noise_score,
                    is_noisy=bool(noisy_mask[sample_idx]),
                    base_value_observed=base_observed,
                    base_value_predicted=base_predicted,
                )
            )

    items = _sort_difference_items(items, sort_by=sort_by, ascending=bool(ascending))
    summary_df = pd.DataFrame(
        [_classification_filter_difference_summary_row(item, summary_depth) for item in items],
        columns=summary_columns,
    ).reset_index(drop=True)

    report = ShapDifferenceReport(
        items=items,
        summary=summary_df,
        feature_names=feature_names_arr,
        class_labels=classes,
        noisy_mask=np.asarray(noisy_mask, dtype=bool),
        top_k=None if top_k is None else int(top_k),
        n_samples=int(n_samples),
        n_features=int(n_features),
        sample_indices=np.asarray(selected_indices, dtype=int),
        noisy_indices=np.asarray([item.instance_index for item in items if bool(item.is_noisy)], dtype=int),
        noisy_only=bool(noisy_only),
        sort_by=sort_by,
        ascending=bool(ascending),
        algorithm=algorithm,
        background_size=None if background_size is None else int(background_size),
        background_random_state=None if background_random_state is None else int(background_random_state),
        max_evals=max_evals_eff,
        cv=int(fitted_filter.cv),
        random_state=None if filter_random_state is None else int(filter_random_state),
        estimator_name=type(fitted_filter.estimator).__name__,
    )
    fitted_filter.shap_difference_report_ = report
    return report


def explain_noisy_instances_with_shap(
    X: Any,
    y_observed: Any,
    y_pred: Any,
    shap_values_all: Any,
    class_labels: Sequence[Any],
    feature_names: Sequence[Any] | None = None,
    noisy_mask: Any | None = None,
    top_k: int | None = 5,
) -> ShapDifferenceReport:
    """Build local SHAP difference explanations for the instances flagged as noisy.

    The filter noise score still tells us which samples are suspicious. This
    report complements it by showing which attributes shift the classifier from
    the observed label toward the predicted one.
    """

    if isinstance(X, pd.DataFrame):
        X_values = X.to_numpy(copy=False)
        inferred_feature_names = np.asarray(list(X.columns), dtype=object)
    else:
        X_values = np.asarray(X)
        if X_values.ndim != 2:
            raise ValueError("X must be a 2D array or pandas DataFrame.")
        inferred_feature_names = np.asarray([f"feature_{idx}" for idx in range(X_values.shape[1])], dtype=object)

    y_observed_arr = np.asarray(y_observed, dtype=object).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=object).ravel()

    if noisy_mask is None:
        noisy_mask_arr = y_observed_arr != y_pred_arr
    else:
        noisy_mask_arr = np.asarray(noisy_mask, dtype=bool).ravel()

    check_consistent_length(X_values, y_observed_arr, y_pred_arr, noisy_mask_arr, shap_values_all)

    if feature_names is None:
        feature_names_arr = inferred_feature_names
    else:
        feature_names_arr = np.asarray(feature_names, dtype=object).ravel()

    if feature_names_arr.shape[0] != X_values.shape[1]:
        raise ValueError(
            f"feature_names length ({feature_names_arr.shape[0]}) does not match the number of features ({X_values.shape[1]})."
        )

    class_labels_arr = _normalize_label_array(class_labels, name="class_labels")

    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be >= 1 when provided.")

    selected_indices = np.flatnonzero(noisy_mask_arr)
    summary_depth = X_values.shape[1] if top_k is None else min(int(top_k), X_values.shape[1])
    summary_columns = ["instance_index", "observed_label", "predicted_label", "explanatory_noise_score"]
    for rank in range(1, summary_depth + 1):
        summary_columns.extend([f"top_feature_{rank}", f"top_delta_{rank}"])

    items: list[ShapDifferenceInstanceExplanation] = []
    summary_rows: list[dict[str, Any]] = []

    for instance_index in selected_indices:
        sample_shap_values = shap_values_all[instance_index]
        class_difference = compute_shap_class_difference(
            sample_shap_values,
            class_labels_arr,
            y_observed_arr[instance_index],
            y_pred_arr[instance_index],
        )
        feature_table = _build_difference_frame(
            class_difference,
            feature_names_arr,
            X_values[instance_index],
            top_k=top_k,
        )
        explanatory_noise_score = compute_explanatory_noise_score(
            class_difference["delta_phi"],
            class_difference["phi_observed"],
            class_difference["phi_predicted"],
        )

        items.append(
            ShapDifferenceInstanceExplanation(
                instance_index=int(instance_index),
                observed_label=_scalar(y_observed_arr[instance_index]),
                predicted_label=_scalar(y_pred_arr[instance_index]),
                explanatory_noise_score=float(explanatory_noise_score),
                class_difference=class_difference,
                top_features=feature_table,
            )
        )

        summary_row = {
            "instance_index": int(instance_index),
            "observed_label": _scalar(y_observed_arr[instance_index]),
            "predicted_label": _scalar(y_pred_arr[instance_index]),
            "explanatory_noise_score": float(explanatory_noise_score),
        }
        for rank, (_, feature_row) in enumerate(feature_table.iterrows(), start=1):
            summary_row[f"top_feature_{rank}"] = feature_row["feature"]
            summary_row[f"top_delta_{rank}"] = float(feature_row["delta_phi"])
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    summary_df = summary_df.reset_index(drop=True)

    return ShapDifferenceReport(
        items=items,
        summary=summary_df,
        feature_names=feature_names_arr,
        class_labels=class_labels_arr,
        noisy_mask=np.asarray(noisy_mask_arr, dtype=bool),
        top_k=None if top_k is None else int(top_k),
        n_samples=int(X_values.shape[0]),
        n_features=int(X_values.shape[1]),
    )


__all__ = [
    "ClassificationFilterSHAPExplanation",
    "ClassificationFilterSHAPReport",
    "ShapDifferenceInstanceExplanation",
    "ShapDifferenceReport",
    "compute_explanatory_noise_score",
    "compute_shap_class_difference",
    "explain_classification_filter_noisy_instances",
    "explain_classification_filter_shap_difference",
    "explain_instance_shap_difference",
    "explain_noisy_instances_with_shap",
]
