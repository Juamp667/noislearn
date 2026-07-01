import numpy as np
import pytest
from sklearn.base import BaseEstimator

from filters import FilterEnsembleFilter


class _StaticDummyFilter(BaseEstimator):
    def __init__(self, noisy_mask, noise_score=None, random_state=None):
        self.noisy_mask = noisy_mask
        self.noise_score = noise_score
        self.random_state = random_state
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.noisy_mask_ = np.asarray(self.noisy_mask, dtype=bool).copy()
        self.noisy_indices_ = np.flatnonzero(self.noisy_mask_)
        if self.noise_score is None:
            self.noise_score_ = None
        else:
            self.noise_score_ = np.asarray(self.noise_score, dtype=float).copy()
        self.detection_report_ = {
            "n_samples": int(len(y)),
            "n_noisy": int(self.noisy_mask_.sum()),
            "noisy_indices": self.noisy_indices_.copy(),
            "noisy_mask": self.noisy_mask_.copy(),
            "noise_score": None if self.noise_score_ is None else self.noise_score_.copy(),
            "observed_labels": np.asarray(y).copy(),
            "predicted_labels": None,
        }
        return self

    def get_detection_report(self):
        return dict(self.detection_report_)


class _RandomDummyFilter(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        rng = np.random.default_rng(self.random_state)
        self.noise_score_ = rng.random(len(y))
        self.noisy_mask_ = self.noise_score_ >= 0.5
        self.noisy_indices_ = np.flatnonzero(self.noisy_mask_)
        self.detection_report_ = {
            "n_samples": int(len(y)),
            "n_noisy": int(self.noisy_mask_.sum()),
            "noisy_indices": self.noisy_indices_.copy(),
            "noisy_mask": self.noisy_mask_.copy(),
            "noise_score": self.noise_score_.copy(),
            "observed_labels": np.asarray(y).copy(),
            "predicted_labels": None,
        }
        return self

    def get_detection_report(self):
        return dict(self.detection_report_)


class _PrefittedDummyFilter(BaseEstimator):
    def __init__(self, noisy_mask, noise_score=None, random_state=None):
        self.noisy_mask = noisy_mask
        self.noise_score = noise_score
        self.random_state = random_state

        self.noisy_mask_ = np.asarray(noisy_mask, dtype=bool).copy()
        self.noisy_indices_ = np.flatnonzero(self.noisy_mask_)
        self.keep_mask_ = ~self.noisy_mask_
        self.noise_score_ = None if noise_score is None else np.asarray(noise_score, dtype=float).copy()
        self.detection_report_ = {
            "n_samples": int(self.noisy_mask_.size),
            "n_noisy": int(self.noisy_mask_.sum()),
            "noisy_indices": self.noisy_indices_.copy(),
            "noisy_mask": self.noisy_mask_.copy(),
            "noise_score": None if self.noise_score_ is None else self.noise_score_.copy(),
            "predicted_labels": None,
        }

    def fit(self, X, y):
        raise AssertionError("fit should not be called for a prefit filter")

    def get_detection_report(self):
        return dict(self.detection_report_)


def _vote_filters():
    return [
        _StaticDummyFilter([1, 0, 1, 0]),
        _StaticDummyFilter([0, 1, 1, 0]),
        _StaticDummyFilter([0, 0, 1, 0]),
    ]


def _score_filters():
    return [
        _StaticDummyFilter([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]),
        _StaticDummyFilter([0, 1, 1, 0], [0.9, 0.2, 0.8, 0.1]),
        _StaticDummyFilter([0, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]),
    ]


def _min_class_filters():
    return [
        _StaticDummyFilter([0, 0, 0, 1, 1, 1], [0.0, 0.0, 0.0, 0.1, 0.9, 0.2]),
        _StaticDummyFilter([0, 0, 0, 1, 1, 1], [0.0, 0.0, 0.0, 0.1, 0.9, 0.2]),
        _StaticDummyFilter([0, 0, 0, 1, 1, 1], [0.0, 0.0, 0.0, 0.1, 0.9, 0.2]),
    ]


def test_fef_vote_strategies_union_majority_consensus_and_k_of_m():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    cases = [
        ("union", {}, [True, True, True, False]),
        ("majority", {}, [False, False, True, False]),
        ("consensus", {}, [False, False, True, False]),
        ("k_of_m", {"min_votes": 2}, [False, False, True, False]),
    ]

    for strategy, kwargs, expected in cases:
        fef = FilterEnsembleFilter(
            base_filters=_vote_filters(),
            strategy=strategy,
            use_filter_scores=False,
            normalize_scores=False,
            **kwargs,
        )

        fef.fit(X, y)

        assert np.array_equal(fef.noisy_mask_, np.asarray(expected, dtype=bool))


def test_fef_threshold_uses_vote_fraction():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    fef = FilterEnsembleFilter(
        base_filters=_vote_filters(),
        strategy="threshold",
        vote_threshold=0.3,
        use_filter_scores=False,
        normalize_scores=False,
    )

    fef.fit(X, y)

    assert np.array_equal(fef.noisy_mask_, np.array([True, True, True, False], dtype=bool))


def test_fef_weighted_threshold_uses_scores_instead_of_only_votes():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    fef = FilterEnsembleFilter(
        base_filters=_score_filters(),
        strategy="weighted_threshold",
        score_threshold=0.5,
        use_filter_scores=True,
        normalize_scores=False,
    )

    fef.fit(X, y)

    assert np.array_equal(fef.noisy_mask_, np.array([True, False, True, False], dtype=bool))
    assert np.allclose(fef.get_ensemble_score(), np.array([0.9, 0.2, 0.8, 0.1], dtype=float))


def test_fef_uses_binary_fallback_and_provided_filter_scores():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    base_filters = [
        _StaticDummyFilter([1, 0, 0, 0]),
        _StaticDummyFilter([0, 1, 0, 0], [0.2, 0.8, 0.1, 0.1]),
    ]

    fef = FilterEnsembleFilter(
        base_filters=base_filters,
        strategy="weighted_threshold",
        score_threshold=0.5,
        use_filter_scores=True,
        normalize_scores=False,
    )

    fef.fit(X, y)

    expected_scores = np.array(
        [
            [1.0, 0.2],
            [0.0, 0.8],
            [0.0, 0.1],
            [0.0, 0.1],
        ],
        dtype=float,
    )

    assert np.array_equal(fef.get_support_matrix(), np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=int))
    assert np.allclose(fef.get_score_matrix(), expected_scores)
    assert np.allclose(fef.get_ensemble_score(), expected_scores.mean(axis=1))
    assert np.array_equal(fef.noisy_mask_, np.array([True, False, False, False], dtype=bool))


def test_fef_min_class_count_protects_small_classes():
    X = np.zeros((6, 1), dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    fef = FilterEnsembleFilter(
        base_filters=_min_class_filters(),
        strategy="union",
        min_class_count=2,
        use_filter_scores=True,
        normalize_scores=False,
    )

    fef.fit(X, y)

    assert np.array_equal(fef.noisy_mask_, np.array([False, False, False, False, True, False], dtype=bool))
    assert np.array_equal(fef.protected_indices_, np.array([3, 5], dtype=int))
    assert fef.class_protection_applied_ is True


def test_fef_fit_filter_report_contains_expected_fields_and_tuple_names():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    fef = FilterEnsembleFilter(
        base_filters=[
            ("ENN", _StaticDummyFilter([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])),
            ("CF", _StaticDummyFilter([0, 1, 0, 0], [0.2, 0.9, 0.8, 0.7])),
        ],
        strategy="union",
        filter_weights=[2.0, 1.0],
        min_class_count=1,
        use_filter_scores=True,
        normalize_scores=False,
    )

    X_clean, y_clean, report = fef.fit_filter(X, y)

    expected_keys = {
        "n_samples",
        "n_noisy",
        "noisy_indices",
        "noisy_mask",
        "noisy_fraction",
        "action",
        "observed_labels",
        "predicted_labels",
        "noise_score",
        "strategy",
        "strategy_used",
        "n_filters",
        "filter_names",
        "filter_votes",
        "filter_scores",
        "support",
        "support_fraction",
        "filter_weights",
        "base_reports",
        "min_class_count",
        "class_protection_applied",
        "protected_indices",
    }

    assert X_clean.shape[0] == 2
    assert y_clean.shape[0] == 2
    assert expected_keys.issubset(report.keys())
    assert report["filter_names"] == ["ENN", "CF"]
    assert report["n_samples"] == 4
    assert report["n_filters"] == 2
    assert report["strategy_used"] == "union"
    assert np.allclose(report["filter_weights"], np.array([2.0 / 3.0, 1.0 / 3.0], dtype=float))
    assert len(report["base_reports"]) == 2
    assert report["predicted_labels"] is None
    assert report["class_protection_applied"] is False
    assert report["protected_indices"].size == 0
    assert report["filter_votes"].shape == (4, 2)
    assert report["filter_scores"].shape == (4, 2)
    assert report["support"].shape == (4,)
    assert report["support_fraction"].shape == (4,)
    assert report["n_noisy"] == int(report["noisy_mask"].sum())
    assert np.allclose(fef.get_sample_weight(), np.clip(1.0 - report["noise_score"], 0.0, 1.0))


def test_fef_reproducible_when_random_state_is_fixed():
    X = np.zeros((6, 1), dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)

    base_filters = [_RandomDummyFilter(), _RandomDummyFilter()]

    fef1 = FilterEnsembleFilter(
        base_filters=base_filters,
        strategy="majority",
        random_state=17,
        use_filter_scores=True,
        normalize_scores=False,
    )
    fef2 = FilterEnsembleFilter(
        base_filters=base_filters,
        strategy="majority",
        random_state=17,
        use_filter_scores=True,
        normalize_scores=False,
    )

    fef1.fit(X, y)
    fef2.fit(X, y)

    assert np.array_equal(fef1.noisy_mask_, fef2.noisy_mask_)
    assert np.allclose(fef1.get_ensemble_score(), fef2.get_ensemble_score())


def test_fef_does_not_modify_original_base_filters():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    base_filter = _StaticDummyFilter([1, 0, 1, 0], [0.8, 0.1, 0.7, 0.2])
    fef = FilterEnsembleFilter(base_filters=[base_filter], strategy="union", normalize_scores=False)

    fef.fit(X, y)

    assert base_filter.fit_calls == 0
    assert not hasattr(base_filter, "detection_report_")
    assert not hasattr(base_filter, "noisy_mask_")


def test_fef_can_use_prefitted_base_filters_without_refitting():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    base_filters = [
        _PrefittedDummyFilter([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]),
        _PrefittedDummyFilter([0, 1, 1, 0], [0.9, 0.2, 0.8, 0.1]),
    ]

    fef_union = FilterEnsembleFilter(
        base_filters=base_filters,
        strategy="union",
        refit_base_filters=False,
        use_filter_scores=True,
        normalize_scores=False,
    )
    fef_threshold = FilterEnsembleFilter(
        base_filters=base_filters,
        strategy="threshold",
        vote_threshold=0.5,
        refit_base_filters=False,
        use_filter_scores=True,
        normalize_scores=False,
    )

    fef_union.fit(X, y)
    fef_threshold.fit(X, y)

    assert np.array_equal(fef_union.noisy_mask_, np.array([True, True, True, False], dtype=bool))
    assert np.array_equal(fef_threshold.noisy_mask_, np.array([True, True, True, False], dtype=bool))
    assert all(filter_.noise_score_ is not None for filter_ in base_filters)


def test_fef_refits_base_filters_by_default():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    base_filters = [
        _PrefittedDummyFilter([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]),
        _PrefittedDummyFilter([0, 1, 1, 0], [0.9, 0.2, 0.8, 0.1]),
    ]

    fef = FilterEnsembleFilter(base_filters=base_filters, strategy="union")

    with pytest.raises(AssertionError, match="fit should not be called"):
        fef.fit(X, y)
