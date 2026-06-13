"""TabPFN-based cross-validated noise filtering with local explanations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y

from tabpfn import TabPFNClassifier

from ..classification import ClassificationFilter, ClassificationFilterResult
from ..._detection import attach_detection_report, resample_by_action, validate_action


@dataclass
class TabPFNFoldInfo:
    """Per-fold diagnostics collected during a TabPFN cross-validation run."""

    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    model: Any
    y_true_idx: np.ndarray
    y_true: np.ndarray
    oof_pred_idx: np.ndarray
    oof_pred: np.ndarray
    oof_proba: np.ndarray | None
    confidence: np.ndarray | None
    noisy_mask: np.ndarray
    noisy_fraction: float


@dataclass
class TabPFNNoiseExplanation:
    """Local explanation for a single sample flagged by TabPFN."""

    sample_idx: int
    fold_idx: int
    true_label_idx: int
    true_label: Any
    oof_pred_idx: int
    oof_pred: Any
    target_class_idx: int
    target_class: Any
    confidence: float | None
    is_noisy: bool
    top_k: list[tuple[Any, float]]
    oof_proba: np.ndarray | None = None
    interaction_values: Any | None = None
    figure: Any | None = None


@dataclass
class TabPFNExplanationReport:
    """Container for TabPFN filter explanations and fold diagnostics."""

    items: list[TabPFNNoiseExplanation]
    folds: list[TabPFNFoldInfo]
    sample_indices: np.ndarray
    noisy_indices: np.ndarray
    sort_by: str
    ascending: bool
    class_index: Any
    budget: int
    index: str
    max_order: int
    imputer: str

    def __iter__(self):
        """Iterate over the stored explanations."""

        return iter(self.items)

    def __len__(self):
        """Return the number of stored explanations."""

        return len(self.items)

    def __getitem__(self, item):
        """Return the explanation at the given position."""

        return self.items[item]

    def by_fold(self):
        """Group the stored explanations by fold index."""

        grouped = {}
        for item in self.items:
            grouped.setdefault(item.fold_idx, []).append(item)
        return grouped


class TabPFN_CF(ClassificationFilter):
    """Cross-validated TabPFN label-noise filter with fold-aware explanations.

    Parameters
    ----------
    cv : int, default=10
        Number of stratified folds used to generate out-of-fold predictions.
    random_state : int, default=33
        Seed used by the stratified splitter and forwarded to TabPFN.
    action : {"remove", "detect"}, default="remove"
        Whether noisy samples are dropped or only detected. Relabel is not implemented yet.
    tabpfn_params : dict or None, default=None
        Keyword arguments forwarded to :class:`tabpfn.TabPFNClassifier`.

    Notes
    -----
    Explanations are computed with SHAP-based tooling from ``tabpfn_extensions``.
    Using ``fit_mode="fit_with_cache"`` is recommended for faster and more stable explanations.
    """

    def __init__(self, cv=10, random_state=33, action="remove", tabpfn_params=None):
        params = {} if tabpfn_params is None else dict(tabpfn_params)
        params.setdefault("device", "auto")
        params.setdefault("random_state", random_state)

        self.tabpfn_params = params
        super().__init__(
            estimator=TabPFNClassifier(**params),
            cv=cv,
            action=action,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Fit the filter and cache fold-wise predictions and diagnostics."""

        feature_names = None
        if hasattr(X, "columns"):
            feature_names = np.asarray(list(X.columns), dtype=object)

        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape

        if int(self.cv) < 2:
            raise ValueError("cv must be >= 2")
        if n_samples < self.cv:
            raise ValueError(f"Need n_samples >= cv. Got n_samples={n_samples}, cv={self.cv}.")
        validate_action(self.action)

        self.n_features_in_ = int(n_features)
        if feature_names is None:
            self.feature_names_in_ = np.arange(self.n_features_in_)
        else:
            if feature_names.shape[0] != self.n_features_in_:
                raise ValueError("feature name count does not match n_features")
            self.feature_names_in_ = feature_names

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        oof_pred_idx = np.empty(n_samples, dtype=int)
        oof_fold_idx = np.full(n_samples, -1, dtype=int)
        oof_confidence = np.full(n_samples, np.nan, dtype=float)
        oof_proba = None
        has_oof_proba = True
        # Initilize a history for each fold
        fold_history = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_idx)):
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])

            fold_pred_idx = np.asarray(model.predict(X[test_idx]), dtype=int)
            fold_pred = self.classes_[fold_pred_idx]
            fold_true_idx = y_idx[test_idx]
            fold_true = self.classes_[fold_true_idx]
            fold_noisy_mask = fold_pred_idx != fold_true_idx

            fold_proba = None
            fold_confidence = None
            if hasattr(model, "predict_proba"):
                fold_proba = np.asarray(model.predict_proba(X[test_idx]), dtype=float)
                if fold_proba.ndim == 2 and fold_proba.shape[0] == test_idx.shape[0]:
                    if oof_proba is None:
                        oof_proba = np.full((n_samples, fold_proba.shape[1]), np.nan, dtype=float)
                    if fold_proba.shape[1] != self.classes_.shape[0]:
                        has_oof_proba = False
                    else:
                        # Store probabilities
                        oof_proba[test_idx] = fold_proba
                        # Store probability associated to predicted label
                        fold_confidence = np.take_along_axis(fold_proba, fold_pred_idx[:, None], axis=1).ravel()
                        oof_confidence[test_idx] = fold_confidence
                else:
                    has_oof_proba = False
            else:
                has_oof_proba = False

            oof_pred_idx[test_idx] = fold_pred_idx
            oof_fold_idx[test_idx] = fold_idx

            # Store the history of the current fold iteration
            fold_history.append(
                TabPFNFoldInfo(
                    fold_idx=fold_idx,
                    train_idx=np.asarray(train_idx),
                    test_idx=np.asarray(test_idx),
                    model=model,
                    y_true_idx=np.asarray(fold_true_idx),
                    y_true=np.asarray(fold_true),
                    oof_pred_idx=np.asarray(fold_pred_idx),
                    oof_pred=np.asarray(fold_pred),
                    oof_proba=fold_proba,
                    confidence=fold_confidence,
                    noisy_mask=np.asarray(fold_noisy_mask),
                    noisy_fraction=float(fold_noisy_mask.mean()),
                )
            )

        if np.any(oof_fold_idx < 0):
            raise RuntimeError("Internal error: some samples were not assigned to any fold.")

        oof_pred = self.classes_[oof_pred_idx]
        noisy_mask = oof_pred_idx != y_idx
        noise_score = None
        if has_oof_proba and oof_proba is not None and np.isfinite(oof_proba).all():
            p_true = oof_proba[np.arange(n_samples), y_idx]
            masked = oof_proba.copy()
            masked[np.arange(n_samples), y_idx] = -np.inf
            p_alt = np.max(masked, axis=1)
            p_alt[~np.isfinite(p_alt)] = 0.0
            noise_score = (1.0 - p_true) * np.sqrt(p_alt)
        else:
            oof_proba = None

        self.oof_pred_idx_ = oof_pred_idx
        self.oof_pred_ = oof_pred
        self.oof_proba_ = oof_proba
        self.oof_confidence_ = oof_confidence
        self.noise_score_ = noise_score
        self.oof_fold_idx_ = oof_fold_idx
        self.noisy_mask_ = noisy_mask
        self.noisy_indices_ = np.flatnonzero(noisy_mask)
        self.fold_history_ = fold_history
        self.y_idx_ = y_idx

        self.result_ = ClassificationFilterResult(
            keep_mask=~noisy_mask,
            noisy_fraction=float(noisy_mask.mean()),
            oof_pred=oof_pred,
        )

        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=oof_pred,
            confidence=oof_confidence,
            oof_fold_idx=oof_fold_idx,
            oof_proba=oof_proba,
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

        check_is_fitted(self, ["result_", "fold_history_"])
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_folds": int(self.cv),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "action": self.action,
            "cv": int(self.cv),
            "tabpfn_params": dict(self.tabpfn_params),
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)

    def get_fold_history(self):
        """Return the stored per-fold diagnostics."""

        check_is_fitted(self, ["fold_history_"])
        return list(self.fold_history_)

    def explain_noisy_instances(
        self,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        *,
        noisy_only: bool = True,
        class_index: Any = "predicted",
        index: str = "SV",
        max_order: int = 1,
        imputer: str = "baseline",
        budget: int = 128,
        top_k: int | None = 5,
        sort_by: str = "confidence",
        ascending: bool = True,
        feature_names: Sequence[Any] | None = None,
        return_interaction_values: bool = False,
        return_figures: bool = False,
        max_display: int = 10,
    ):
        """Explain OOF decisions made by the filter and aggregate the result.

        Parameters
        ----------
        sample_indices:
            Optional subset of sample indices to explain.
        noisy_only:
            If True, ignore samples that were not flagged as noisy by the filter.
        class_index:
            "predicted" (default) explains the OOF predicted class for each sample,
            "true" explains the true class, or pass an int/label to use a fixed class.
        index, max_order, imputer, budget:
            Forwarded to ``get_tabpfn_imputation_explainer`` and ``explain``.
        top_k:
            Number of strongest contributions to report per sample. Set to ``None`` or
            ``0`` to skip.
        sort_by:
            ``"confidence"`` (default), ``"fold_idx"`` or ``"sample_idx"``.
        ascending:
            Sort direction. For confidence, ascending means less confident first.
        feature_names:
            Optional feature names used in the returned contributions and plots.
        return_interaction_values:
            If True, store the raw ``InteractionValues`` object per explanation.
        return_figures:
            If True, attach a waterfall figure for each explanation.
        max_display:
            Max number of interactions shown in the waterfall plot.
        """

        check_is_fitted(self, ["fold_history_", "oof_fold_idx_", "result_", "X_", "y_"])

        try:
            from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer
        except ModuleNotFoundError as e:
            raise ImportError(
                "tabpfn-extensions[interpretability] is required for explanations."
            ) from e

        if self.tabpfn_params.get("fit_mode") != "fit_with_cache":
            warnings.warn(
                "TabPFN explanations are faster and more stable with fit_mode='fit_with_cache'.",
                UserWarning,
                stacklevel=2,
            )

        # Extract feature names (column_id if pandas else int_id)
        feature_names_arr = self._resolve_feature_names(feature_names)

        # Initilize sample_indices to study
        selected_indices = self._normalize_sample_indices(sample_indices)

        # Leave just noisy indices if noisy_only=True
        if noisy_only:
            selected_indices = [idx for idx in selected_indices if bool(self.noisy_mask_[idx])]

        items = []
        # For each instance to study
        for sample_idx in selected_indices:
            # Extract the corresponding fold history and associated info
            fold_idx = int(self.oof_fold_idx_[sample_idx])
            fold = self.fold_history_[fold_idx]

            target_class_idx = self._resolve_target_class_index(class_index, sample_idx)
            target_class = self._label_at(target_class_idx)
            confidence = self._sample_confidence(sample_idx, target_class_idx)

            # Initilize the explain with the model and training data
            explainer = get_tabpfn_imputation_explainer(
                model=fold.model,
                data=self.X_[fold.train_idx],
                index=index,
                max_order=max_order,
                imputer=imputer,
                class_index=target_class_idx,
            )

            # Compute explanation for the instance associated to sample_idx
            interaction_values = explainer.explain(self.X_[sample_idx : sample_idx + 1], budget=budget)
            top_features = self._extract_top_k(interaction_values, top_k, feature_names_arr)

            figure = None
            if return_figures:
                try:
                    axes = interaction_values.plot_waterfall(
                        feature_names=feature_names_arr,
                        show=False,
                        max_display=max_display,
                    )
                    figure = axes.figure if axes is not None else None
                except ModuleNotFoundError as e:
                    raise ImportError("shap is required when return_figures=True.") from e

            items.append(
                TabPFNNoiseExplanation(
                    sample_idx=int(sample_idx),
                    fold_idx=fold_idx,
                    true_label_idx=int(self.y_idx_[sample_idx]),
                    true_label=self._label_at(self.y_idx_[sample_idx]),
                    oof_pred_idx=int(self.oof_pred_idx_[sample_idx]),
                    oof_pred=self._label_at(self.oof_pred_idx_[sample_idx]),
                    target_class_idx=int(target_class_idx),
                    target_class=target_class,
                    confidence=confidence,
                    is_noisy=bool(self.noisy_mask_[sample_idx]),
                    top_k=top_features,
                    oof_proba=self.oof_proba_[sample_idx] if self.oof_proba_ is not None else None,
                    interaction_values=interaction_values if return_interaction_values else None,
                    figure=figure,
                )
            )

        items = self._sort_explanations(items, sort_by=sort_by, ascending=ascending)
        report = TabPFNExplanationReport(
            items=items,
            folds=list(self.fold_history_),
            sample_indices=np.asarray(selected_indices, dtype=int),
            noisy_indices=np.asarray([item.sample_idx for item in items if item.is_noisy], dtype=int),
            sort_by=sort_by,
            ascending=ascending,
            class_index=class_index,
            budget=int(budget),
            index=index,
            max_order=int(max_order),
            imputer=imputer,
        )
        self.explanation_report_ = report
        return report

    def _resolve_feature_names(self, feature_names: Sequence[Any] | None):
        if feature_names is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        names = np.asarray(list(feature_names), dtype=object)
        if names.shape[0] != self.n_features_in_:
            raise ValueError("feature name count does not match n_features")
        return names

    def _normalize_sample_indices(self, sample_indices: Sequence[int] | np.ndarray | None):
        # If no indices are passed, return an arange
        if sample_indices is None:
            return list(range(self.X_.shape[0]))

        indices = np.asarray(sample_indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)

        # If boolean list is passed
        if indices.dtype == bool:
            if indices.shape[0] != self.X_.shape[0]:
                raise ValueError("Boolean sample_indices mask must match n_samples")
            return list(np.flatnonzero(indices))

        # If interger list is passed
        indices = indices.astype(int, copy=False).ravel()
        if np.any((indices < 0) | (indices >= self.X_.shape[0])):
            raise IndexError("sample_indices contains out-of-range values")
        return list(indices)

    def _resolve_target_class_index(self, class_index: Any, sample_idx: int):
        if isinstance(class_index, str):
            key = class_index.lower()
            if key in {"predicted", "prediction", "oof_pred", "oof_predicted"}:
                return int(self.oof_pred_idx_[sample_idx])
            if key in {"true", "label", "y_true", "oof_true"}:
                return int(self.y_idx_[sample_idx])
            raise ValueError("class_index must be 'predicted', 'true', an int or a known label")

        if class_index is None:
            return 1 if len(self.classes_) > 1 else 0

        matches = np.flatnonzero(np.asarray(self.classes_, dtype=object) == class_index)
        if matches.size:
            return int(matches[0])

        idx = int(class_index)
        if idx < 0 or idx >= len(self.classes_):
            raise ValueError("class_index is out of range for the fitted classes")
        return idx

    def _label_at(self, class_idx: int):
        value = self.classes_[int(class_idx)]
        return value.item() if isinstance(value, np.generic) else value

    def _sample_confidence(self, sample_idx: int, class_idx: int):
        if self.oof_proba_ is None:
            return None
        value = self.oof_proba_[sample_idx, int(class_idx)]
        return float(value) if not np.isnan(value) else None

    def _extract_top_k(self, interaction_values, top_k: int | None, feature_names):
        if not top_k or int(top_k) <= 0:
            return []

        _, ranked = interaction_values.get_top_k(int(top_k), as_interaction_values=False)
        top_features = []
        for interaction, value in ranked:
            top_features.append((self._format_interaction(interaction, feature_names), float(value)))
        return top_features

    def _format_interaction(self, interaction, feature_names):
        if not isinstance(interaction, tuple):
            interaction = (interaction,)

        labels = tuple(self._scalar(feature_names[int(idx)]) for idx in interaction)
        return labels[0] if len(labels) == 1 else labels

    def _scalar(self, value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _sort_explanations(self, items, *, sort_by: str, ascending: bool):
        if sort_by == "confidence":
            valid = [item for item in items if item.confidence is not None and not np.isnan(item.confidence)]
            missing = [item for item in items if item.confidence is None or np.isnan(item.confidence)]
            valid.sort(key=lambda item: float(item.confidence), reverse=not ascending)
            return valid + missing

        if sort_by == "fold_idx":
            return sorted(items, key=lambda item: item.fold_idx, reverse=not ascending)

        if sort_by == "sample_idx":
            return sorted(items, key=lambda item: item.sample_idx, reverse=not ascending)

        raise ValueError("sort_by must be 'confidence', 'fold_idx' or 'sample_idx'")
