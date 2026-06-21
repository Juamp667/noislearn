import numpy as np
import pytest

from filters import NoiseScoreFilter


class _DummyNoiseFilter:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.noise_score_ = np.linspace(0.0, 1.0, len(y), dtype=float)
        return self


def _fit_rational_valley(scores):
    scores = np.asarray(scores, dtype=float).ravel()
    X = np.zeros((scores.size, 1), dtype=float)
    y = np.zeros(scores.size, dtype=int)
    return NoiseScoreFilter(noise_scores=scores, threshold="rational_valley").fit(X, y)


def test_rational_valley_finds_internal_threshold_on_bimodal_density():
    rng = np.random.default_rng(42)
    scores = np.r_[
        rng.normal(0.2, 0.04, 200),
        rng.normal(0.8, 0.04, 200),
    ]
    scores = np.clip(scores, 0.0, 1.0)

    fitted = _fit_rational_valley(scores)

    assert fitted.threshold_report_["threshold_strategy_used"] == "rational_valley"
    assert fitted.threshold_report_["fallback_used"] is False
    assert 0.35 < fitted.threshold_ < 0.65
    assert fitted.threshold_report_["separability_score"] is not None
    assert fitted.threshold_report_["separability_score"] > 0.10
    assert fitted.threshold_report_["valid_minima"]


def test_rational_valley_falls_back_on_monotone_beta():
    rng = np.random.default_rng(7)
    scores = rng.beta(1, 8, 300)

    fitted = _fit_rational_valley(scores)

    assert fitted.threshold_report_["fallback_used"] is True
    assert fitted.threshold_report_["threshold_strategy_used"] == "rational_valley_fallback"
    assert 0.0 <= fitted.threshold_ <= 1.0
    assert fitted.threshold_report_["fallback_reason"] in {
        "fit_failed",
        "invalid_denominator",
        "no_internal_critical_point",
        "no_internal_minimum",
        "threshold_out_of_bounds",
        "low_separability",
    }


def test_rational_valley_falls_back_on_constant_scores():
    scores = np.ones(100, dtype=float) * 0.3

    fitted = _fit_rational_valley(scores)

    assert fitted.threshold_report_["fallback_used"] is True
    assert fitted.threshold_report_["fallback_reason"] == "constant_scores"
    assert np.isclose(fitted.threshold_, 0.3)


def test_rational_valley_falls_back_on_nearly_constant_scores():
    rng = np.random.default_rng(13)
    scores = 0.3 + 1e-6 * rng.standard_normal(100)

    fitted = _fit_rational_valley(scores)

    assert fitted.threshold_report_["fallback_used"] is True
    assert fitted.threshold_report_["fallback_reason"] == "constant_scores"
    assert np.isclose(fitted.threshold_, float(np.mean(np.clip(scores, 0.0, 1.0))), atol=1e-4)


def test_rational_valley_handles_small_sample_with_controlled_fallback():
    scores = np.array([0.1, 0.2, 0.9], dtype=float)

    fitted = _fit_rational_valley(scores)

    assert fitted.threshold_report_["fallback_used"] is True
    assert fitted.threshold_report_["fallback_reason"] == "density_estimation_failed"
    assert np.isclose(fitted.threshold_, float(np.mean(scores)))


def test_rational_valley_rejects_non_finite_scores():
    scores = np.array([0.1, 0.2, np.nan, 0.8], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        _fit_rational_valley(scores)


def test_rational_valley_clips_scores_outside_unit_interval():
    scores = np.array([-0.1, 0.2, 0.7, 1.2], dtype=float)

    fitted = _fit_rational_valley(scores)

    assert np.allclose(fitted.noise_score_, np.clip(scores, 0.0, 1.0))
    assert 0.0 <= fitted.threshold_ <= 1.0


def test_noise_score_filter_can_fit_nested_filter_and_use_quantile_threshold():
    X = np.zeros((5, 1), dtype=float)
    y = np.array([0, 1, 0, 1, 0], dtype=int)
    noise_filter = _DummyNoiseFilter()

    fitted = NoiseScoreFilter(
        noise_filter=noise_filter,
        fit_filter=True,
        threshold="quantile",
        quantile=0.5,
    ).fit(X, y)

    expected_scores = np.linspace(0.0, 1.0, len(y), dtype=float)
    expected_threshold = float(np.quantile(expected_scores, 0.5))

    assert noise_filter.fit_calls == 1
    assert fitted.noise_filter_ is noise_filter
    assert np.allclose(fitted.noise_score_, expected_scores)
    assert np.isclose(fitted.threshold_, expected_threshold)
    assert fitted.threshold_report_["threshold_strategy_used"] == "quantile"
    assert np.isclose(fitted.threshold_report_["quantile"], 0.5)


def test_noise_score_filter_reuses_pretrained_nested_filter():
    X = np.zeros((4, 1), dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    noise_filter = _DummyNoiseFilter()
    noise_filter.noise_score_ = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    fitted = NoiseScoreFilter(
        noise_filter=noise_filter,
        fit_filter=True,
        threshold="mean",
    ).fit(X, y)

    assert noise_filter.fit_calls == 0
    assert fitted.noise_filter_ is noise_filter
    assert np.allclose(fitted.noise_score_, noise_filter.noise_score_)
