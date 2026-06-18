"""External SHAP explanations for classification-based noise filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from .classification import ClassificationFilter


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
            background, 
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


__all__ = [
    "ClassificationFilterSHAPExplanation",
    "ClassificationFilterSHAPReport",
    "explain_classification_filter_noisy_instances",
]
