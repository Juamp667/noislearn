import sys
import types

import numpy as np
import pandas as pd

from filters import (
    compute_explanatory_noise_score,
    compute_shap_class_difference,
    explain_classification_filter_shap_difference,
    explain_instance_shap_difference,
    explain_noisy_instances_with_shap,
)


class _FakeEstimator:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.tile(np.array([[0.25, 0.75]], dtype=float), (X.shape[0], 1))


class _FakeFittedFilter:
    def __init__(self, X, y, result):
        self.estimator = _FakeEstimator()
        self.cv = 2
        self.random_state = 0
        self.X_ = X
        self.y_ = y
        self.classes_ = np.array(["a", "b"], dtype=object)
        self.result_ = result
        self.noise_score_ = np.array([0.9, 0.1, 0.2, 0.8], dtype=float)

    def fit(self, X=None, y=None):
        return self


class _FakeSampleExplanation:
    def __init__(self, values):
        self.values = values
        self.base_values = np.array([0.25, 0.75], dtype=float)


class _FakeBatchExplanation:
    def __init__(self, X):
        self._items = []
        for row in np.asarray(X, dtype=float):
            self._items.append(_FakeSampleExplanation(np.column_stack([row, -row])))

    def __getitem__(self, item):
        return self._items[item]


class _FakeIndependentMasker:
    max_samples_seen = []

    def __init__(self, data, max_samples=None):
        self.data = np.asarray(data)
        self.max_samples = max_samples
        type(self).max_samples_seen.append(max_samples)


class _FakeExplainer:
    background_sizes = []
    max_evals_seen = []

    def __init__(self, predict_fn, background, algorithm="auto", output_names=None):
        self.predict_fn = predict_fn
        self.background = np.asarray(getattr(background, "data", background))
        self.algorithm = algorithm
        self.output_names = output_names
        type(self).background_sizes.append(self.background.shape[0])

    def __call__(self, X, max_evals=None):
        type(self).max_evals_seen.append(max_evals)
        return _FakeBatchExplanation(X)


def _install_fake_shap(monkeypatch):
    _FakeExplainer.background_sizes = []
    _FakeExplainer.max_evals_seen = []
    _FakeIndependentMasker.max_samples_seen = []
    monkeypatch.setitem(
        sys.modules,
        "shap",
        types.SimpleNamespace(
            Explainer=_FakeExplainer,
            maskers=types.SimpleNamespace(Independent=_FakeIndependentMasker),
        ),
    )


def _sample_fitted_filter():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    y = np.array(["a", "b", "a", "b"], dtype=object)
    result = types.SimpleNamespace(
        keep_mask=np.array([False, True, True, False], dtype=bool),
        oof_pred=np.array(["b", "b", "a", "a"], dtype=object),
    )
    return _FakeFittedFilter(X, y, result)


def _sample_shap_values():
    class_labels = np.array(["clean", "noisy"], dtype=object)
    shap_values = np.array(
        [
            [0.2, -0.1, 0.4],
            [-0.4, 0.3, 0.1],
        ],
        dtype=float,
    ).T
    return class_labels, shap_values


def test_compute_shap_class_difference_supports_transposed_input():
    class_labels, shap_values = _sample_shap_values()

    result = compute_shap_class_difference(
        shap_values=shap_values,
        class_labels=class_labels,
        observed_label="clean",
        predicted_label="noisy",
    )

    assert result["observed_label"] == "clean"
    assert result["predicted_label"] == "noisy"
    assert np.allclose(result["phi_observed"], np.array([0.2, -0.1, 0.4]))
    assert np.allclose(result["phi_predicted"], np.array([-0.4, 0.3, 0.1]))
    assert np.allclose(result["delta_phi"], np.array([-0.6, 0.4, -0.3]))
    assert np.allclose(result["abs_delta_phi"], np.array([0.6, 0.4, 0.3]))


def test_explain_instance_shap_difference_orders_features_by_discrepancy():
    class_labels, shap_values = _sample_shap_values()
    feature_names = ["f1", "f2", "f3"]
    feature_values = [10, 20, 30]

    frame = explain_instance_shap_difference(
        shap_values=shap_values,
        class_labels=class_labels,
        feature_names=feature_names,
        feature_values=feature_values,
        observed_label="clean",
        predicted_label="noisy",
        top_k=2,
    )

    assert list(frame.columns) == [
        "feature",
        "value",
        "phi_observed",
        "phi_predicted",
        "delta_phi",
        "abs_delta_phi",
        "direction",
        "interpretation",
    ]
    assert list(frame["feature"]) == ["f1", "f2"]
    assert list(frame["direction"]) == ["supports_observed", "supports_predicted"]
    assert frame.iloc[0]["interpretation"] == "Penaliza la clase predicha y favorece la etiqueta observada."
    assert frame.iloc[1]["interpretation"] == "Favorece la clase predicha y penaliza la etiqueta observada."


def test_compute_explanatory_noise_score_is_finite_with_nan_inputs():
    score = compute_explanatory_noise_score(
        delta_phi=[np.nan, 1.0],
        phi_observed=[0.0, np.nan],
        phi_predicted=[0.0, 2.0],
    )

    assert np.isfinite(score)
    assert np.isclose(score, 0.5, atol=1e-9)


def test_explain_noisy_instances_with_shap_builds_report_and_summary():
    X = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        columns=["f1", "f2", "f3"],
    )
    y_observed = np.array(["clean", "noisy", "clean"], dtype=object)
    y_pred = np.array(["noisy", "noisy", "noisy"], dtype=object)
    class_labels = np.array(["clean", "noisy"], dtype=object)
    shap_values_all = np.array(
        [
            [[0.2, -0.1, 0.4], [-0.4, 0.3, 0.1]],
            [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]],
            [[0.3, 0.2, -0.2], [-0.2, 0.1, 0.0]],
        ],
        dtype=float,
    )

    report = explain_noisy_instances_with_shap(
        X=X,
        y_observed=y_observed,
        y_pred=y_pred,
        shap_values_all=shap_values_all,
        class_labels=class_labels,
        feature_names=None,
        noisy_mask=None,
        top_k=1,
    )

    assert len(report) == 2
    assert report.noisy_mask.tolist() == [True, False, True]
    assert report.feature_names.tolist() == ["f1", "f2", "f3"]
    assert list(report.summary["instance_index"]) == [0, 2]
    assert list(report.summary.columns) == [
        "instance_index",
        "observed_label",
        "predicted_label",
        "explanatory_noise_score",
        "top_feature_1",
        "top_delta_1",
    ]
    assert report[0].top_features.shape[0] == 1
    assert report[0].top_features.iloc[0]["feature"] == "f1"


def test_explain_classification_filter_shap_difference_uses_full_fold_background(monkeypatch):
    _install_fake_shap(monkeypatch)
    fitted_filter = _sample_fitted_filter()

    report = explain_classification_filter_shap_difference(
        fitted_filter,
        sample_indices=[0],
        noisy_only=False,
        feature_names=["f1", "f2", "f3"],
        top_k=2,
        sort_by="sample_idx",
    )

    assert len(report) == 1
    assert _FakeExplainer.background_sizes == [2]
    assert _FakeIndependentMasker.max_samples_seen == [2]
    assert report.background_size is None
    assert report.sample_indices.tolist() == [0]
    assert fitted_filter.shap_difference_report_ is report

    item = report[0]
    assert item.instance_index == 0
    assert item.observed_label == "a"
    assert item.predicted_label == "b"
    assert item.confidence == 0.75
    assert item.noise_score == 0.9
    assert item.is_noisy is True
    np.testing.assert_allclose(item.class_difference["phi_observed"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(item.class_difference["phi_predicted"], np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_allclose(item.class_difference["delta_phi"], np.array([-2.0, -4.0, -6.0]))
    assert list(item.top_features["feature"]) == ["f3", "f2"]
    assert list(report.summary["top_feature_1"]) == ["f3"]
    assert list(report.summary["top_delta_1"]) == [-6.0]
