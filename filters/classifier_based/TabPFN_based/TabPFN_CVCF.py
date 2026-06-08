from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y

from tabpfn import TabPFNClassifier

from ..cvcf import CVCFFilter, CVCFFilterResult


@dataclass
class TabPFNCommitteeFoldInfo:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    model: Any
    pred_idx: np.ndarray
    pred: np.ndarray
    proba: np.ndarray | None
    confidence: np.ndarray | None
    wrong_mask: np.ndarray
    wrong_fraction: float


@dataclass
class TabPFNCommitteeFoldExplanation:
    sample_idx: int
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    predicted_class_idx: int
    predicted_class: Any
    target_class_idx: int
    target_class: Any
    confidence: float | None
    is_correct: bool
    main_effects: np.ndarray
    normalized_effects: np.ndarray
    top_k: list[tuple[Any, float]]
    interaction_values: Any | None = None
    figure: Any | None = None


@dataclass
class TabPFNCommitteeAggregatedView:
    name: str
    fold_indices: np.ndarray
    target_class_idx: int | None
    target_class: Any | None
    weights: np.ndarray
    contributions: np.ndarray
    top_k: list[tuple[Any, float]]
    n_folds: int
    figure: Any | None = None


@dataclass
class TabPFNCommitteeSampleExplanation:
    sample_idx: int
    true_label_idx: int
    true_label: Any
    vote_counts_idx: np.ndarray
    vote_counts: dict[Any, int]
    committee_pred_idx: int
    committee_pred: Any
    confidence: float
    noise_score: float
    vote_margin: float
    vote_entropy: float
    is_noisy: bool
    supporting_folds: np.ndarray
    dissenting_folds: np.ndarray
    fold_explanations: list[TabPFNCommitteeFoldExplanation]
    all_view: TabPFNCommitteeAggregatedView
    majority_view: TabPFNCommitteeAggregatedView


@dataclass
class TabPFNCommitteeExplanationReport:
    items: list[TabPFNCommitteeSampleExplanation]
    folds: list[TabPFNCommitteeFoldInfo]
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
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def by_fold(self):
        grouped = {}
        for item in self.items:
            for fold_exp in item.fold_explanations:
                grouped.setdefault(fold_exp.fold_idx, []).append(fold_exp)
        for fold_idx in grouped:
            grouped[fold_idx].sort(key=lambda exp: exp.sample_idx)
        return grouped

    def by_sample(self):
        return {item.sample_idx: item for item in self.items}


class TabPFN_CVCF(CVCFFilter):
    """Cross-validated TabPFN committee filter with fold-aware explanations."""

    def __init__(self, cv=10, vote_rule="consensus", threshold=0.5, random_state=33, action="remove", tabpfn_params=None):
        params = {} if tabpfn_params is None else dict(tabpfn_params)
        params.setdefault("device", "auto")
        params.setdefault("random_state", random_state)

        self.tabpfn_params = params
        super().__init__(
            estimator=TabPFNClassifier(**params),
            cv=cv,
            vote_rule=vote_rule,
            threshold=threshold,
            action=action,
            random_state=random_state,
        )

    def fit(self, X, y):
        feature_names = None
        if hasattr(X, "columns"):
            feature_names = np.asarray(list(X.columns), dtype=object)

        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_splits = int(self.cv)

        if n_splits < 2:
            raise ValueError("cv must be >= 2")
        if n_samples < n_splits:
            raise ValueError(f"Need n_samples >= cv. Got n_samples={n_samples}, cv={self.cv}.")

        self.n_features_in_ = int(n_features)
        if feature_names is None:
            self.feature_names_in_ = np.arange(self.n_features_in_)
        else:
            if feature_names.shape[0] != self.n_features_in_:
                raise ValueError("feature name count does not match n_features")
            self.feature_names_in_ = feature_names

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        fold_history = []
        fold_pred_matrix = np.empty((n_splits, n_samples), dtype=int)
        fold_confidence_matrix = np.full((n_splits, n_samples), np.nan, dtype=float)
        has_confidence = False

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_idx)):
            model = clone(self.estimator)
            model.fit(X[train_idx], y_idx[train_idx])

            pred_idx = np.asarray(model.predict(X), dtype=int)
            pred = self.classes_[pred_idx]
            wrong_mask = pred_idx != y_idx
            wrong_fraction = float(wrong_mask.mean())

            proba = None
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X), dtype=float)
                if proba.ndim == 2 and proba.shape[0] == n_samples:
                    confidence = np.take_along_axis(proba, pred_idx[:, None], axis=1).ravel()
                    fold_confidence_matrix[fold_idx] = confidence
                    has_confidence = True

            fold_pred_matrix[fold_idx] = pred_idx

            fold_history.append(
                TabPFNCommitteeFoldInfo(
                    fold_idx=fold_idx,
                    train_idx=np.asarray(train_idx),
                    test_idx=np.asarray(test_idx),
                    model=model,
                    pred_idx=np.asarray(pred_idx),
                    pred=np.asarray(pred),
                    proba=proba,
                    confidence=confidence,
                    wrong_mask=np.asarray(wrong_mask),
                    wrong_fraction=wrong_fraction,
                )
            )

        fold_confidence_matrix_ = fold_confidence_matrix if has_confidence else None
        wrong_votes = (fold_pred_matrix != y_idx[None, :]).sum(axis=0).astype(int)
        noisy_mask = self._flag_by_votes(wrong_votes, n_models=n_splits)
        keep_mask = ~noisy_mask

        vote_counts = np.zeros((n_samples, len(self.classes_)), dtype=int)
        sample_range = np.arange(n_samples)
        for pred_idx in fold_pred_matrix:
            np.add.at(vote_counts, (sample_range, pred_idx), 1)

        committee_pred_idx = np.empty(n_samples, dtype=int)
        committee_majority_votes = vote_counts.max(axis=1)
        second_votes = np.zeros(n_samples, dtype=int)
        if len(self.classes_) > 1:
            second_votes = np.partition(vote_counts, -2, axis=1)[:, -2]

        vote_margin = np.empty(n_samples, dtype=float)
        vote_entropy = np.empty(n_samples, dtype=float)

        # For each instance to study
        for sample_idx in range(n_samples):
            committee_pred_idx[sample_idx] = self._resolve_committee_prediction(
                vote_counts[sample_idx],
                fold_pred_matrix[:, sample_idx],
                None if fold_confidence_matrix_ is None else fold_confidence_matrix_[:, sample_idx],
            )
            # Compute an score associated to between-fold agreement
            vote_margin[sample_idx] = float((committee_majority_votes[sample_idx] - second_votes[sample_idx]) / float(n_splits))
            probs = vote_counts[sample_idx] / float(n_splits)
            mask = probs > 0
            vote_entropy[sample_idx] = float(-(probs[mask] * np.log(probs[mask])).sum() / np.log(len(self.classes_))) if len(self.classes_) > 1 else 0.0

        committee_pred = self.classes_[committee_pred_idx]
        committee_confidence = committee_majority_votes / float(n_splits)
        committee_noise_score = 1.0 - committee_confidence

        self.fold_history_ = fold_history
        self.fold_pred_matrix_ = fold_pred_matrix
        self.fold_confidence_matrix_ = fold_confidence_matrix_
        self.vote_counts_ = vote_counts
        self.committee_pred_idx_ = committee_pred_idx
        self.committee_pred_ = committee_pred
        self.committee_confidence_ = committee_confidence
        self.committee_noise_score_ = committee_noise_score
        self.committee_vote_margin_ = vote_margin
        self.committee_vote_entropy_ = vote_entropy
        self.wrong_votes_ = wrong_votes
        self.noisy_mask_ = noisy_mask
        self.noisy_indices_ = np.flatnonzero(noisy_mask)
        self.y_idx_ = y_idx

        self.result_ = CVCFFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            fold_preds=self.classes_[fold_pred_matrix],
            disagree_count=wrong_votes,
            noisy_votes=noisy_mask.astype(int),
            n_models=int(n_splits),
        )

        self.keep_mask_ = keep_mask
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]
        raise ValueError("action='relabel' is not implemented yet.")

    def get_filter_report(self):
        check_is_fitted(self, ["result_", "fold_history_", "committee_confidence_"])
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(self.cv),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "vote_rule": self.vote_rule,
            "threshold": float(self.threshold) if self.vote_rule == "threshold" else None,
            "action": self.action,
            "agreement_mean": float(np.mean(self.committee_confidence_)),
            "noise_score_mean": float(np.mean(self.committee_noise_score_)),
            "vote_margin_mean": float(np.mean(self.committee_vote_margin_)),
            "vote_entropy_mean": float(np.mean(self.committee_vote_entropy_)),
            "tabpfn_params": dict(self.tabpfn_params),
        }

    def get_fold_history(self):
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
        """Explain committee predictions fold by fold and aggregate them.

        The all-view aggregates every fold explanation. The majority-view only
        aggregates folds that vote for the committee majority class of the sample.
        By default, each fold is explained with its own predicted class.
        """

        check_is_fitted(self, ["fold_history_", "fold_pred_matrix_", "vote_counts_", "X_", "y_"])

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

        feature_names_arr = self._resolve_feature_names(feature_names)
        selected_indices = self._normalize_sample_indices(sample_indices)

        if noisy_only:
            selected_indices = [idx for idx in selected_indices if bool(self.noisy_mask_[idx])]

        noisy_selected_indices = np.asarray([idx for idx in selected_indices if bool(self.noisy_mask_[idx])], dtype=int)

        items = []
        # For each instance to study
        for sample_idx in selected_indices:
            committee_idx = int(self.committee_pred_idx_[sample_idx])
            fold_explanations = []

            # Analyze what each fold said about it
            for fold in self.fold_history_:
                predicted_class_idx = int(fold.pred_idx[sample_idx])
                target_class_idx = self._resolve_target_class_index(
                    class_index=class_index,
                    sample_idx=sample_idx,
                    fold_pred_idx=predicted_class_idx,
                    committee_pred_idx=committee_idx,
                )

                # Initilize the explainer
                explainer = get_tabpfn_imputation_explainer(
                    model=fold.model,
                    data=self.X_[fold.train_idx],
                    index=index,
                    max_order=max_order,
                    imputer=imputer,
                    class_index=target_class_idx,
                )

                # Extract explanation info
                interaction_values = explainer.explain(self.X_[sample_idx : sample_idx + 1], budget=budget)
                main_effects = self._interaction_values_to_main_effects(interaction_values)
                normalized_effects = self._normalize_vector(main_effects)
                top_features = self._vector_top_k(normalized_effects, top_k, feature_names_arr)
                confidence = self._fold_confidence_for_sample(fold, sample_idx, target_class_idx)

                figure = None
                if return_figures:
                    figure = self._plot_top_contributions(
                        normalized_effects,
                        feature_names_arr,
                        title=f"Fold {fold.fold_idx} sample {sample_idx}",
                        max_display=max_display,
                    )

                fold_explanations.append(
                    TabPFNCommitteeFoldExplanation(
                        sample_idx=int(sample_idx),
                        fold_idx=int(fold.fold_idx),
                        train_idx=np.asarray(fold.train_idx),
                        test_idx=np.asarray(fold.test_idx),
                        predicted_class_idx=predicted_class_idx,
                        predicted_class=self._label_at(predicted_class_idx),
                        target_class_idx=int(target_class_idx),
                        target_class=self._label_at(target_class_idx),
                        confidence=confidence,
                        is_correct=bool(predicted_class_idx == self.y_idx_[sample_idx]),
                        main_effects=main_effects,
                        normalized_effects=normalized_effects,
                        top_k=top_features,
                        interaction_values=interaction_values if return_interaction_values else None,
                        figure=figure,
                    )
                )

            supporting_folds = np.asarray(
                [exp.fold_idx for exp in fold_explanations if exp.predicted_class_idx == committee_idx],
                dtype=int,
            )
            dissenting_folds = np.asarray(
                [exp.fold_idx for exp in fold_explanations if exp.predicted_class_idx != committee_idx],
                dtype=int,
            )

            all_view = self._build_aggregated_view(
                name="all",
                fold_explanations=fold_explanations,
                feature_names=feature_names_arr,
                top_k=top_k,
                title=f"All folds sample {sample_idx}",
                return_figures=return_figures,
                max_display=max_display,
            )

            majority_fold_explanations = [exp for exp in fold_explanations if exp.predicted_class_idx == committee_idx]
            if not majority_fold_explanations:
                majority_fold_explanations = fold_explanations

            majority_view = self._build_aggregated_view(
                name="majority",
                fold_explanations=majority_fold_explanations,
                feature_names=feature_names_arr,
                top_k=top_k,
                title=f"Majority folds sample {sample_idx}",
                return_figures=return_figures,
                max_display=max_display,
            )

            vote_counts_row = self.vote_counts_[sample_idx]
            vote_counts_map = {
                self._label_at(class_idx): int(count)
                for class_idx, count in enumerate(vote_counts_row)
                if int(count) > 0
            }

            items.append(
                TabPFNCommitteeSampleExplanation(
                    sample_idx=int(sample_idx),
                    true_label_idx=int(self.y_idx_[sample_idx]),
                    true_label=self._label_at(self.y_idx_[sample_idx]),
                    vote_counts_idx=np.asarray(vote_counts_row, dtype=int),
                    vote_counts=vote_counts_map,
                    committee_pred_idx=committee_idx,
                    committee_pred=self._label_at(committee_idx),
                    confidence=float(self.committee_confidence_[sample_idx]),
                    noise_score=float(self.committee_noise_score_[sample_idx]),
                    vote_margin=float(self.committee_vote_margin_[sample_idx]),
                    vote_entropy=float(self.committee_vote_entropy_[sample_idx]),
                    is_noisy=bool(self.noisy_mask_[sample_idx]),
                    supporting_folds=supporting_folds,
                    dissenting_folds=dissenting_folds,
                    fold_explanations=fold_explanations,
                    all_view=all_view,
                    majority_view=majority_view,
                )
            )

        items = self._sort_sample_explanations(items, sort_by=sort_by, ascending=ascending)
        report = TabPFNCommitteeExplanationReport(
            items=items,
            folds=list(self.fold_history_),
            sample_indices=np.asarray(selected_indices, dtype=int),
            noisy_indices=noisy_selected_indices,
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
        if sample_indices is None:
            return list(range(self.X_.shape[0]))

        indices = np.asarray(sample_indices)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.dtype == bool:
            if indices.shape[0] != self.X_.shape[0]:
                raise ValueError("Boolean sample_indices mask must match n_samples")
            return list(np.flatnonzero(indices))

        indices = indices.astype(int, copy=False).ravel()
        if np.any((indices < 0) | (indices >= self.X_.shape[0])):
            raise IndexError("sample_indices contains out-of-range values")
        return list(indices)

    def _resolve_target_class_index(self, class_index: Any, sample_idx: int, fold_pred_idx: int, committee_pred_idx: int):
        if isinstance(class_index, str):
            key = class_index.lower()
            if key in {"predicted", "prediction", "fold_pred", "model_pred"}:
                return int(fold_pred_idx)
            if key in {"committee", "majority", "consensus", "vote"}:
                return int(committee_pred_idx)
            if key in {"true", "label", "y_true"}:
                return int(self.y_idx_[sample_idx])
            raise ValueError(
                "class_index must be 'predicted', 'committee', 'true', an int or a known label"
            )

        if class_index is None:
            return int(fold_pred_idx)

        matches = np.flatnonzero(np.asarray(self.classes_, dtype=object) == class_index)
        if matches.size:
            return int(matches[0])

        idx = int(class_index)
        if idx < 0 or idx >= len(self.classes_):
            raise ValueError("class_index is out of range for the fitted classes")
        return idx

    def _resolve_committee_prediction(self, vote_counts_row, fold_pred_col, fold_conf_col):
        max_votes = int(np.max(vote_counts_row))
        tied = np.flatnonzero(vote_counts_row == max_votes)
        # Return the winner if there's just one
        if tied.size == 1 or fold_conf_col is None or not np.any(np.isfinite(fold_conf_col)):
            return int(tied[0])
        
        # If there's a draw between 2 or + classes, use the confidence if available to compute class_pred_scoring
        scores = []
        for class_idx in tied:
            mask = fold_pred_col == class_idx
            if conf.size and np.any(np.isfinite(conf)):
                scores.append(float(np.nanmean(conf)))
            else:
                scores.append(-np.inf)

        scores = np.asarray(scores, dtype=float)
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        return int(tied[best[0]])

    def _interaction_values_to_main_effects(self, interaction_values):
        first_order = interaction_values.get_n_order(order=1) if hasattr(interaction_values, "get_n_order") else interaction_values
        if hasattr(first_order, "to_first_order_array"):
            return np.asarray(first_order.to_first_order_array(), dtype=float)
        if hasattr(interaction_values, "to_first_order_array"):
            return np.asarray(interaction_values.to_first_order_array(), dtype=float)
        raise TypeError("interaction_values does not expose first-order contributions")

    def _normalize_vector(self, values):
        values = np.asarray(values, dtype=float)
        scale = float(np.sum(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0.0:
            return np.zeros_like(values, dtype=float)
        return values / scale

    def _vector_top_k(self, values, top_k, feature_names):
        if not top_k or int(top_k) <= 0:
            return []

        values = np.asarray(values, dtype=float)
        k = min(int(top_k), values.shape[0])
        order = np.argsort(np.abs(values))[::-1][:k]
        return [(self._scalar(feature_names[int(idx)]), float(values[int(idx)])) for idx in order]

    def _plot_top_contributions(self, values, feature_names, title, max_display):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            raise ImportError("matplotlib is required when return_figures=True.") from e

        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return None

        display_n = min(int(max_display), values.size)
        order = np.argsort(np.abs(values))[::-1][:display_n]
        order = order[::-1]
        scores = values[order]
        labels = [str(self._scalar(feature_names[int(idx)])) for idx in order]
        colors = ["#d62728" if score < 0 else "#1f77b4" for score in scores]

        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.35 * display_n + 1.0)))
        y = np.arange(display_n)
        ax.barh(y, scores, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("normalized contribution")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    def _build_aggregated_view(self, name, fold_explanations, feature_names, top_k, title, return_figures, max_display):
        if fold_explanations:
            vectors = np.vstack([exp.normalized_effects for exp in fold_explanations])
            weights = np.asarray([self._fold_weight(exp) for exp in fold_explanations], dtype=float)
            if not np.any(np.isfinite(weights)) or np.all(weights <= 0.0):
                weights = np.ones(len(fold_explanations), dtype=float)
            contributions = np.average(vectors, axis=0, weights=weights)
            contributions = self._normalize_vector(contributions)
            fold_indices = np.asarray([exp.fold_idx for exp in fold_explanations], dtype=int)
        else:
            weights = np.asarray([], dtype=float)
            contributions = np.zeros(self.n_features_in_, dtype=float)
            fold_indices = np.asarray([], dtype=int)

        target_class_idx = self._common_target_class_idx(fold_explanations)
        target_class = self._label_at(target_class_idx) if target_class_idx is not None else None
        figure = self._plot_top_contributions(contributions, feature_names, title, max_display) if return_figures else None

        return TabPFNCommitteeAggregatedView(
            name=name,
            fold_indices=fold_indices,
            target_class_idx=target_class_idx,
            target_class=target_class,
            weights=weights,
            contributions=contributions,
            top_k=self._vector_top_k(contributions, top_k, feature_names),
            n_folds=int(len(fold_explanations)),
            figure=figure,
        )

    def _common_target_class_idx(self, fold_explanations):
        if not fold_explanations:
            return None
        first = fold_explanations[0].target_class_idx
        if all(exp.target_class_idx == first for exp in fold_explanations):
            return int(first)
        return None

    def _fold_weight(self, fold_explanation):
        if fold_explanation.confidence is None:
            return 1.0
        value = float(fold_explanation.confidence)
        if not np.isfinite(value) or value < 0.0:
            return 1.0
        return value

    def _fold_confidence_for_sample(self, fold, sample_idx, target_class_idx):
        if fold.confidence is None:
            return None
        if target_class_idx < 0 or target_class_idx >= fold.proba.shape[1]:
            return None
        value = float(fold.proba[sample_idx, target_class_idx])
        return None if not np.isfinite(value) else value

    def _label_at(self, class_idx: int):
        value = self.classes_[int(class_idx)]
        return value.item() if isinstance(value, np.generic) else value

    def _scalar(self, value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _sort_sample_explanations(self, items, *, sort_by: str, ascending: bool):
        if sort_by == "confidence":
            key = lambda item: item.confidence
        elif sort_by == "noise_score":
            key = lambda item: item.noise_score
        elif sort_by == "vote_margin":
            key = lambda item: item.vote_margin
        elif sort_by == "sample_idx":
            key = lambda item: item.sample_idx
        else:
            raise ValueError("sort_by must be 'confidence', 'noise_score', 'vote_margin' or 'sample_idx'")

        return sorted(items, key=key, reverse=not ascending)
