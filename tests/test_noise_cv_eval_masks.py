from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from noise_cv_eval import (
    _class_noise_mask_from_clean_reference,
    _load_fold_views,
    _load_persisted_noise_mask,
    _removal_metrics,
    _row_noise_mask,
    _true_noisy_mask,
    _true_noisy_mask_and_known,
)
from testFuncs import load_dataset_df


DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"


def _raw_train_with_source(tmp_path, values=(0, 1, 2)):
    df = pd.DataFrame({"x": list(values), "target": ["a"] * len(values)})
    df.attrs["source_path"] = str(tmp_path / "toy-5-1tra.dat")
    return df


def test_persisted_noise_mask_npy_is_loaded(tmp_path):
    noisy_train = _raw_train_with_source(tmp_path)
    expected = np.array([False, True, False], dtype=bool)
    np.save(Path(noisy_train.attrs["source_path"]).with_suffix(".noise_mask.npy"), expected)

    actual = _load_persisted_noise_mask(noisy_train)

    np.testing.assert_array_equal(actual, expected)


def test_persisted_noise_mask_csv_is_loaded(tmp_path):
    noisy_train = _raw_train_with_source(tmp_path)
    expected = np.array([False, True, False], dtype=bool)
    mask_path = Path(noisy_train.attrs["source_path"]).with_suffix(".noise_mask.csv")
    pd.DataFrame({"is_noisy": expected}).to_csv(mask_path, index=False)

    actual = _load_persisted_noise_mask(noisy_train)

    np.testing.assert_array_equal(actual, expected)


def test_persisted_noise_mask_length_must_match_fold(tmp_path):
    noisy_train = _raw_train_with_source(tmp_path)
    np.save(Path(noisy_train.attrs["source_path"]).with_suffix(".noise_mask.npy"), np.array([True]))

    with pytest.raises(ValueError, match="has 1 row"):
        _load_persisted_noise_mask(noisy_train)


def test_true_noisy_mask_prefers_persisted_mask(tmp_path):
    clean_train = _raw_train_with_source(tmp_path, values=(9, 8, 7))
    noisy_train = _raw_train_with_source(tmp_path)
    expected = np.array([True, False, True], dtype=bool)
    np.save(Path(noisy_train.attrs["source_path"]).with_suffix(".noise_mask.npy"), expected)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _true_noisy_mask(
            dataset="toy",
            noise_type="cla_rand",
            clean_train_raw=clean_train,
            noisy_train_raw=noisy_train,
            root=tmp_path,
        )

    np.testing.assert_array_equal(actual, expected)
    assert caught == []


def test_true_noisy_mask_and_known_marks_persisted_mask_as_exact(tmp_path):
    clean_train = _raw_train_with_source(tmp_path, values=(9, 8, 7))
    noisy_train = _raw_train_with_source(tmp_path)
    expected = np.array([True, False, True], dtype=bool)
    np.save(Path(noisy_train.attrs["source_path"]).with_suffix(".noise_mask.npy"), expected)

    true_noisy_mask, known_ground_truth_mask = _true_noisy_mask_and_known(
        dataset="toy",
        noise_type="cla_rand",
        clean_train_raw=clean_train,
        noisy_train_raw=noisy_train,
        root=tmp_path,
    )

    np.testing.assert_array_equal(true_noisy_mask, expected)
    np.testing.assert_array_equal(known_ground_truth_mask, np.ones(expected.shape[0], dtype=bool))


def test_ambiguous_class_noise_rows_are_marked_unknown():
    clean_reference = pd.DataFrame(
        {
            "x": [1, 1, 2],
            "target": ["a", "b", "a"],
        }
    )
    noisy_train = pd.DataFrame(
        {
            "x": [1, 1, 2],
            "target": ["a", "c", "b"],
        }
    )

    true_noisy_mask, known_ground_truth_mask = _class_noise_mask_from_clean_reference(
        clean_reference,
        noisy_train,
        return_known=True,
        warn_on_ambiguous=False,
    )

    np.testing.assert_array_equal(true_noisy_mask, np.array([False, True, True]))
    np.testing.assert_array_equal(known_ground_truth_mask, np.array([False, True, True]))


def test_removal_metrics_exclude_unknown_ground_truth_rows():
    metrics = _removal_metrics(
        true_noisy_mask=np.array([False, True, True]),
        pred_removed_mask=np.array([True, True, False]),
        known_ground_truth_mask=np.array([False, True, True]),
    )

    assert metrics["n_known_ground_truth"] == 2
    assert metrics["n_unknown_ground_truth"] == 1
    assert metrics["ground_truth_exact"] is False
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["n_removed_pred"] == 1
    assert metrics["precision_removal"] == 1.0
    assert metrics["recall_removal"] == 0.5


def test_class_noise_mask_uses_noisy_fold_alignment_for_shuffled_seed():
    noisy_fold = DATASET_ROOT / "cla_rand" / "seed_02" / "25" / "ecoli" / "ecoli-5-1tra.dat"
    if not noisy_fold.exists():
        pytest.skip("Local KEEL noisy datasets are not available.")

    clean_train = load_dataset_df(
        dataset="ecoli",
        noise_type="data_base",
        split="tra",
        fold=1,
        encoding=None,
        root=DATASET_ROOT,
    )
    noisy_train = load_dataset_df(
        dataset="ecoli",
        noise_type="cla_rand",
        seed=2,
        k=25,
        split="tra",
        fold=1,
        encoding=None,
        root=DATASET_ROOT,
    )

    old_positional_mask = _row_noise_mask(clean_train, noisy_train)
    _, _, true_noisy_mask = _load_fold_views(
        dataset="ecoli",
        noise_type="cla_rand",
        seed=2,
        k=25,
        fold=1,
        encoding=None,
        root=DATASET_ROOT,
        test_source="clean",
    )

    assert old_positional_mask.shape == true_noisy_mask.shape == (268,)
    assert int(old_positional_mask.sum()) == 264
    assert int(true_noisy_mask.sum()) == 62
