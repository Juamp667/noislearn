import numpy as np
import pandas as pd

from filters import (
    compute_explanatory_noise_score,
    compute_shap_class_difference,
    explain_instance_shap_difference,
    explain_noisy_instances_with_shap,
)


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
