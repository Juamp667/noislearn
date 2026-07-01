"""
5-fold evaluation helpers for noisy KEEL datasets.

This module builds a pipeline with:
- preprocessing
- optional noise filter
- classifier

It returns two tables:
- classification metrics by fold/method
- removal metrics for the filters
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from time import perf_counter

from testFuncs import load_dataset_df

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


CLASSIFICATION_COLUMNS = [
    "experiment",
    "dataset",
    "noise_type",
    "noise_pct",
    "seed",
    "k",
    "fold",
    "method",
    "encoding",
    "test_source",
    "preprocess_before_filter",
    "n_train_input",
    "n_train_used",
    "n_test",
    "elapsed_s",
    "valid_classification",
    "params",
    "accuracy",
    "bal_acc",
    "f1_macro",
    "precision_macro",
    "recall_macro",
]

REMOVAL_COLUMNS = [
    "experiment",
    "dataset",
    "noise_type",
    "noise_pct",
    "seed",
    "k",
    "fold",
    "filter",
    "encoding",
    "test_source",
    "preprocess_before_filter",
    "n_train_input",
    "n_train_used",
    "elapsed_s",
    "valid_classification",
    "params",
    "n_true_noisy",
    "n_known_ground_truth",
    "n_unknown_ground_truth",
    "ground_truth_exact",
    "n_removed_pred",
    "removed_pct",
    "acc_removal",
    "precision_removal",
    "recall_removal",
    "f1_removal",
    "specificity",
    "mcc",
    "tp",
    "tn",
    "fp",
    "fn",
]


def _coerce_int(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def _default_experiment_name(dataset, noise_type, seed, k, test_source):
    return f"{dataset}|{noise_type}|seed={seed}|k={k}|test={test_source}"


def _noise_pct(noise_type: str, k) -> int:
    if noise_type == "data_base":
        return 0
    if k is None:
        raise ValueError("k is required for noisy datasets.")
    return int(_coerce_int(k))


def _row_noise_mask(clean_train_raw: pd.DataFrame, noisy_train_raw: pd.DataFrame) -> np.ndarray:
    if clean_train_raw.shape != noisy_train_raw.shape:
        raise ValueError(
            "Clean and noisy training folds must have the same shape to compute the ground-truth noise mask."
        )

    clean = clean_train_raw.reset_index(drop=True)
    noisy = noisy_train_raw.reset_index(drop=True)
    noisy = noisy.reindex(columns=clean.columns)

    clean_cmp = clean.fillna("__nan__")
    noisy_cmp = noisy.fillna("__nan__")
    return noisy_cmp.ne(clean_cmp).any(axis=1).to_numpy(dtype=bool)


def _target_noise_mask(clean_train_raw: pd.DataFrame, noisy_train_raw: pd.DataFrame) -> np.ndarray:
    if clean_train_raw.shape != noisy_train_raw.shape:
        raise ValueError(
            "Clean and noisy training folds must have the same shape to compute the ground-truth noise mask."
        )

    clean = clean_train_raw.reset_index(drop=True)
    noisy = noisy_train_raw.reset_index(drop=True).reindex(columns=clean.columns)
    clean_labels = clean.iloc[:, -1].to_numpy(dtype=object)
    noisy_labels = noisy.iloc[:, -1].to_numpy(dtype=object)
    return np.asarray(
        [
            _normalise_key_value(noisy_label) != _normalise_key_value(clean_label)
            for clean_label, noisy_label in zip(clean_labels, noisy_labels)
        ],
        dtype=bool,
    )


def _features_match_positionally(clean_train_raw: pd.DataFrame, noisy_train_raw: pd.DataFrame) -> bool:
    if clean_train_raw.shape != noisy_train_raw.shape:
        return False

    clean = clean_train_raw.reset_index(drop=True)
    noisy = noisy_train_raw.reset_index(drop=True).reindex(columns=clean.columns)
    clean_features = clean.iloc[:, :-1].fillna("__nan__")
    noisy_features = noisy.iloc[:, :-1].fillna("__nan__")
    return not bool(noisy_features.ne(clean_features).to_numpy().any())


_MISSING_KEY_VALUE = object()


def _is_missing_value(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalise_key_value(value):
    if _is_missing_value(value):
        return _MISSING_KEY_VALUE
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_key(values) -> tuple:
    return tuple(_normalise_key_value(value) for value in values)


_NOISE_MASK_SUFFIXES = (".noise_mask.npy", ".noise_mask.csv", ".noise_mask.txt")
_NOISE_MASK_COLUMNS = ("is_noisy", "noise_mask", "noisy", "mask")
_NOISE_MASK_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_NOISE_MASK_FALSE_VALUES = {"0", "false", "f", "no", "n"}


def _noise_mask_candidate_paths(source_path) -> list[Path]:
    source_path = Path(source_path)
    return [source_path.with_suffix(suffix) for suffix in _NOISE_MASK_SUFFIXES]


def _read_tabular_noise_mask(mask_path: Path) -> np.ndarray:
    raw_df = pd.read_csv(mask_path, header=None)
    if raw_df.shape[1] == 1:
        values = raw_df.iloc[:, 0]
        if len(values) and str(values.iloc[0]).strip().lower() in _NOISE_MASK_COLUMNS:
            values = values.iloc[1:]
        return values.to_numpy()

    named_df = pd.read_csv(mask_path)
    for column in _NOISE_MASK_COLUMNS:
        if column in named_df.columns:
            return named_df[column].to_numpy()

    raise ValueError(
        f"Noise mask file {mask_path} must contain one mask column, or one of: "
        f"{', '.join(_NOISE_MASK_COLUMNS)}."
    )


def _coerce_noise_mask_values(values, expected_len: int, mask_path: Path) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim > 1:
        if 1 not in values.shape:
            raise ValueError(f"Noise mask file {mask_path} must be one-dimensional.")
        values = values.reshape(-1)

    if values.shape[0] != expected_len:
        raise ValueError(
            f"Noise mask file {mask_path} has {values.shape[0]} row(s), "
            f"but the noisy training fold has {expected_len}."
        )

    mask = np.zeros(expected_len, dtype=bool)
    for idx, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            mask[idx] = bool(value)
            continue

        if _is_missing_value(value):
            raise ValueError(f"Noise mask file {mask_path} contains a missing value at row {idx}.")

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, (int, float, np.integer, np.floating)):
            if value in (0, 1):
                mask[idx] = bool(value)
                continue

        text = str(value).strip().lower()
        if text in _NOISE_MASK_TRUE_VALUES:
            mask[idx] = True
        elif text in _NOISE_MASK_FALSE_VALUES:
            mask[idx] = False
        else:
            raise ValueError(
                f"Noise mask file {mask_path} contains an invalid boolean value "
                f"at row {idx}: {value!r}."
            )

    return mask


def _load_persisted_noise_mask(noisy_train_raw: pd.DataFrame) -> Optional[np.ndarray]:
    source_path = noisy_train_raw.attrs.get("source_path")
    if not source_path:
        return None

    for mask_path in _noise_mask_candidate_paths(source_path):
        if not mask_path.exists():
            continue

        if mask_path.suffix == ".npy":
            values = np.load(mask_path, allow_pickle=False)
        elif mask_path.suffix in {".csv", ".txt"}:
            values = _read_tabular_noise_mask(mask_path)
        else:  # pragma: no cover - guarded by _NOISE_MASK_SUFFIXES
            continue

        return _coerce_noise_mask_values(
            values,
            expected_len=int(noisy_train_raw.shape[0]),
            mask_path=mask_path,
        )

    return None


def _class_noise_mask_from_clean_reference(
    clean_reference_raw: pd.DataFrame,
    noisy_train_raw: pd.DataFrame,
    return_known: bool = False,
    warn_on_ambiguous: bool = True,
):
    if clean_reference_raw.shape[1] != noisy_train_raw.shape[1]:
        raise ValueError(
            "Clean reference and noisy training fold must have the same number of columns "
            "to compute the ground-truth class-noise mask."
        )

    clean_reference = clean_reference_raw.reset_index(drop=True)
    noisy = noisy_train_raw.reset_index(drop=True).reindex(columns=clean_reference.columns)
    feature_cols = list(clean_reference.columns[:-1])
    target_col = clean_reference.columns[-1]

    clean_labels_by_features = {}
    for _, row in clean_reference.iterrows():
        key = _feature_key(row[feature_cols].to_numpy(dtype=object))
        label = _normalise_key_value(row[target_col])
        clean_labels_by_features.setdefault(key, set()).add(label)

    true_noisy_mask = np.zeros(noisy.shape[0], dtype=bool)
    known_ground_truth_mask = np.ones(noisy.shape[0], dtype=bool)
    missing_keys = 0
    ambiguous_matches = 0
    for row_idx, row in noisy.iterrows():
        key = _feature_key(row[feature_cols].to_numpy(dtype=object))
        if key not in clean_labels_by_features:
            missing_keys += 1
            continue
        noisy_label = _normalise_key_value(row[target_col])
        clean_labels = clean_labels_by_features[key]
        if len(clean_labels) == 1:
            clean_label = next(iter(clean_labels))
            true_noisy_mask[int(row_idx)] = noisy_label != clean_label
        elif noisy_label not in clean_labels:
            true_noisy_mask[int(row_idx)] = True
        else:
            known_ground_truth_mask[int(row_idx)] = False
            ambiguous_matches += 1

    if missing_keys:
        raise ValueError(
            "Cannot compute the ground-truth class-noise mask because "
            f"{missing_keys} noisy training row(s) were not found in the clean reference."
        )

    if ambiguous_matches and warn_on_ambiguous:
        warnings.warn(
            "Some class-noise ground-truth rows are ambiguous because their feature vectors "
            "match multiple clean labels. They are excluded from removal metrics when the "
            "extended ground-truth mask is used. Persist a row identifier or a noise_mask "
            "when generating these noisy datasets for fully exact removal metrics. "
            f"Ambiguous rows: {ambiguous_matches}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_known:
        return true_noisy_mask, known_ground_truth_mask

    return true_noisy_mask


def _true_noisy_mask_and_known(
    dataset: str,
    noise_type: str,
    clean_train_raw: pd.DataFrame,
    noisy_train_raw: pd.DataFrame,
    root,
    warn_on_ambiguous: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    known_ground_truth_mask = np.ones(noisy_train_raw.shape[0], dtype=bool)

    if noise_type == "data_base":
        return np.zeros(noisy_train_raw.shape[0], dtype=bool), known_ground_truth_mask

    persisted_mask = _load_persisted_noise_mask(noisy_train_raw)
    if persisted_mask is not None:
        return persisted_mask, known_ground_truth_mask

    if str(noise_type).startswith("cla_"):
        if _features_match_positionally(clean_train_raw, noisy_train_raw):
            return _target_noise_mask(clean_train_raw, noisy_train_raw), known_ground_truth_mask

        clean_reference_raw = load_dataset_df(
            dataset=dataset,
            noise_type="data_base",
            split="cc",
            encoding=None,
            root=root,
        )
        return _class_noise_mask_from_clean_reference(
            clean_reference_raw,
            noisy_train_raw,
            return_known=True,
            warn_on_ambiguous=warn_on_ambiguous,
        )

    return _row_noise_mask(clean_train_raw, noisy_train_raw), known_ground_truth_mask


def _true_noisy_mask(
    dataset: str,
    noise_type: str,
    clean_train_raw: pd.DataFrame,
    noisy_train_raw: pd.DataFrame,
    root,
) -> np.ndarray:
    true_noisy_mask, _ = _true_noisy_mask_and_known(
        dataset=dataset,
        noise_type=noise_type,
        clean_train_raw=clean_train_raw,
        noisy_train_raw=noisy_train_raw,
        root=root,
        warn_on_ambiguous=True,
    )
    return true_noisy_mask


def _load_fold_views(
    dataset: str,
    noise_type: str,
    seed,
    k,
    fold: int,
    encoding,
    root,
    test_source: str,
    compute_ground_truth: bool = True,
    return_ground_truth_known: bool = False,
):
    clean_train_raw = load_dataset_df(
        dataset=dataset,
        noise_type="data_base",
        split="tra",
        fold=fold,
        encoding=None,
        root=root,
    )

    if noise_type == "data_base":
        noisy_train_raw = clean_train_raw
        train_df = load_dataset_df(
            dataset=dataset,
            noise_type="data_base",
            split="tra",
            fold=fold,
            encoding=encoding,
            root=root,
        )
        test_df = load_dataset_df(
            dataset=dataset,
            noise_type="data_base",
            split="tst",
            fold=fold,
            encoding=encoding,
            root=root,
        )
    else:
        noisy_train_raw = load_dataset_df(
            dataset=dataset,
            noise_type=noise_type,
            seed=seed,
            k=k,
            split="tra",
            fold=fold,
            encoding=None,
            root=root,
        )
        train_df = load_dataset_df(
            dataset=dataset,
            noise_type=noise_type,
            seed=seed,
            k=k,
            split="tra",
            fold=fold,
            encoding=encoding,
            root=root,
        )

        if test_source == "clean":
            test_df = load_dataset_df(
                dataset=dataset,
                noise_type="data_base",
                split="tst",
                fold=fold,
                encoding=encoding,
                root=root,
            )
        elif test_source == "noisy":
            test_df = load_dataset_df(
                dataset=dataset,
                noise_type=noise_type,
                seed=seed,
                k=k,
                split="tst",
                fold=fold,
                encoding=encoding,
                root=root,
            )
        else:
            raise ValueError("test_source must be 'clean' or 'noisy'.")

    if not compute_ground_truth:
        if return_ground_truth_known:
            return train_df, test_df, None, None
        return train_df, test_df, None

    true_noisy_mask, known_ground_truth_mask = _true_noisy_mask_and_known(
        dataset=dataset,
        noise_type=noise_type,
        clean_train_raw=clean_train_raw,
        noisy_train_raw=noisy_train_raw,
        root=root,
    )

    if return_ground_truth_known:
        return train_df, test_df, true_noisy_mask, known_ground_truth_mask

    return train_df, test_df, true_noisy_mask


def _lazy_ml_imports():
    try:
        from imblearn.pipeline import Pipeline
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn and imbalanced-learn are required to run the 5CV evaluation."
        ) from exc

    return {
        "Pipeline": Pipeline,
        "clone": clone,
        "RandomForestClassifier": RandomForestClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "SkPipeline": SkPipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def _build_default_preprocessor(metrics, X: pd.DataFrame):
    numeric_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    num_pipe = metrics["SkPipeline"](
        steps=[
            ("imputer", metrics["SimpleImputer"](strategy="median")),
            ("scaler", metrics["StandardScaler"]()),
        ]
    )

    cat_encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        cat_encoder = metrics["OneHotEncoder"](**cat_encoder_kwargs, sparse_output=False)
    except TypeError:  # older sklearn
        cat_encoder = metrics["OneHotEncoder"](**cat_encoder_kwargs, sparse=False)

    cat_pipe = metrics["SkPipeline"](
        steps=[
            ("imputer", metrics["SimpleImputer"](strategy="most_frequent")),
            ("onehot", cat_encoder),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    return metrics["ColumnTransformer"](transformers=transformers, remainder="drop")


def _resolve_preprocessor(preprocessor, metrics, X_train):
    if preprocessor is None:
        return _build_default_preprocessor(metrics, X_train)
    if preprocessor == "passthrough":
        return None
    return preprocessor


def _build_pipeline(clone, Pipeline, preprocessor, filter_estimator, classifier, preprocess_before_filter: bool):
    steps = []

    if preprocess_before_filter:
        if preprocessor is not None:
            steps.append(("preprocess", clone(preprocessor)))
        if filter_estimator is not None:
            steps.append(("filter", clone(filter_estimator)))
    else:
        if filter_estimator is not None:
            steps.append(("filter", clone(filter_estimator)))
        if preprocessor is not None:
            steps.append(("preprocess", clone(preprocessor)))

    steps.append(("classifier", clone(classifier)))
    return Pipeline(steps)


def _summarize_estimator(estimator):
    if estimator is None:
        return None

    estimator_type = type(estimator).__name__
    params = {}
    if hasattr(estimator, "get_params"):
        try:
            params = estimator.get_params(deep=False)
        except Exception:
            params = {}

    return {
        "type": estimator_type,
        "params": params,
    }


def _pipeline_params(preprocessor, filter_estimator, classifier, preprocess_before_filter):
    return {
        "preprocess_before_filter": bool(preprocess_before_filter),
        "preprocessor": _summarize_estimator(preprocessor),
        "filter": _summarize_estimator(filter_estimator),
        "classifier": _summarize_estimator(classifier),
    }


def _extract_keep_mask(fitted_pipeline, original_n: int) -> np.ndarray:
    if "filter" not in fitted_pipeline.named_steps:
        return np.ones(original_n, dtype=bool)

    fitted_filter = fitted_pipeline.named_steps["filter"]

    if hasattr(fitted_filter, "result_") and hasattr(fitted_filter.result_, "keep_mask"):
        mask = np.asarray(fitted_filter.result_.keep_mask, dtype=bool)
    elif hasattr(fitted_filter, "keep_mask"):
        mask = np.asarray(fitted_filter.keep_mask, dtype=bool)
    elif hasattr(fitted_filter, "sample_indices_"):
        mask = np.zeros(original_n, dtype=bool)
        mask[np.asarray(fitted_filter.sample_indices_, dtype=int)] = True
    else:
        raise AttributeError(
            "The filter must expose a keep_mask (via result_.keep_mask or keep_mask) or sample_indices_."
        )

    if mask.shape[0] != original_n:
        raise ValueError("The filter keep mask does not match the training fold size.")

    return mask


def _removal_metrics(
    true_noisy_mask: np.ndarray,
    pred_removed_mask: np.ndarray,
    known_ground_truth_mask: Optional[np.ndarray] = None,
):
    true_noisy_mask = np.asarray(true_noisy_mask, dtype=bool)
    pred_removed_mask = np.asarray(pred_removed_mask, dtype=bool)

    if true_noisy_mask.shape != pred_removed_mask.shape:
        raise ValueError("The true and predicted removal masks must have the same shape.")

    if known_ground_truth_mask is None:
        known_ground_truth_mask = np.ones(true_noisy_mask.shape[0], dtype=bool)
    else:
        known_ground_truth_mask = np.asarray(known_ground_truth_mask, dtype=bool)
        if known_ground_truth_mask.shape != true_noisy_mask.shape:
            raise ValueError("The known ground-truth mask must match the true noise mask shape.")

    n_total = int(true_noisy_mask.shape[0])
    n_known_ground_truth = int(known_ground_truth_mask.sum())
    n_unknown_ground_truth = n_total - n_known_ground_truth
    ground_truth_exact = n_unknown_ground_truth == 0

    true_eval = true_noisy_mask[known_ground_truth_mask]
    pred_eval = pred_removed_mask[known_ground_truth_mask]

    tp = int(np.sum(pred_eval & true_eval))
    tn = int(np.sum((~pred_eval) & (~true_eval)))
    fp = int(np.sum(pred_eval & (~true_eval)))
    fn = int(np.sum((~pred_eval) & true_eval))

    n = n_known_ground_truth
    n_true_noisy = tp + fn
    n_removed_pred = tp + fp

    acc_removal = (tp + tn) / n if n else np.nan
    removed_pct = 100.0 * n_removed_pred / n if n else np.nan

    precision_removal = tp / n_removed_pred if n_removed_pred > 0 else np.nan
    recall_removal = tp / n_true_noisy if n_true_noisy > 0 else np.nan
    f1_removal = (
        (2.0 * precision_removal * recall_removal) / (precision_removal + recall_removal)
        if np.isfinite(precision_removal)
        and np.isfinite(recall_removal)
        and (precision_removal + recall_removal) > 0
        else np.nan
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else np.nan

    return {
        "removed_pct": removed_pct,
        "acc_removal": acc_removal,
        "precision_removal": precision_removal,
        "recall_removal": recall_removal,
        "f1_removal": f1_removal,
        "specificity": specificity,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_true_noisy": n_true_noisy,
        "n_known_ground_truth": n_known_ground_truth,
        "n_unknown_ground_truth": n_unknown_ground_truth,
        "ground_truth_exact": ground_truth_exact,
        "n_removed_pred": n_removed_pred,
    }


def _classification_metrics(y_true, y_pred, metrics):
    return {
        "acc": metrics["accuracy_score"](y_true, y_pred),
        "accuracy": metrics["accuracy_score"](y_true, y_pred),
        "bal_acc": metrics["balanced_accuracy_score"](y_true, y_pred),
        "f1_macro": metrics["f1_score"](y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": metrics["precision_score"](y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics["recall_score"](y_true, y_pred, average="macro", zero_division=0),
    }


def _safe_fit_predict(pipe, X_train, y_train, X_test):
    y_train_arr = np.asarray(y_train)
    if np.unique(y_train_arr).shape[0] < 2:
        # print("Erroraco1")
        return None, True
    try:
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test), False
    except ValueError:
        # print("Erroraco2")
        return None, True


def _safe_metric_rows(metric_names):
    return {name: np.nan for name in metric_names}


def _print_invalid_experiment(*, experiment, dataset, fold, noise_type, seed, k, method, reason, n_train=None):
    parts = [
        "[INVALID]",
        f"experiment={experiment}",
        f"dataset={dataset}",
        f"fold={int(fold)}",
        f"noise_type={noise_type}",
        f"seed={seed}",
        f"k={k}",
        f"method={method}",
        f"reason={reason}",
    ]
    if n_train is not None:
        parts.append(f"n_train={int(n_train)}")
    print(" ".join(parts), flush=True)


def _resolve_protected_mask(y_train, protect_classes=None):
    y_arr = np.asarray(y_train)
    if protect_classes is None:
        return np.zeros(y_arr.shape[0], dtype=bool)

    if protect_classes == "minority":
        values, counts = np.unique(y_arr, return_counts=True)
        protected_class = values[np.argmin(counts)]
        return y_arr == protected_class

    if isinstance(protect_classes, str):
        protect_classes = [protect_classes]

    protected_set = set(protect_classes)
    return np.isin(y_arr, list(protected_set))


def _ensure_two_classes_after_protection(y_train, keep_mask, protected_mask):
    y_arr = np.asarray(y_train)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    protected_mask = np.asarray(protected_mask, dtype=bool)

    keep_mask = keep_mask | protected_mask

    y_kept = y_arr[keep_mask]
    if np.unique(y_kept).shape[0] >= 2:
        return keep_mask, False

    protected_classes = np.unique(y_arr[protected_mask])
    if protected_classes.size == 0:
        return keep_mask, True

    for cls in protected_classes:
        cls_idx = np.where(y_arr == cls)[0]
        if cls_idx.size > 0:
            keep_mask[cls_idx[0]] = True

    y_kept = y_arr[keep_mask]
    return keep_mask, np.unique(y_kept).shape[0] < 2


def run_5cv_baseline(
    dataset: str,
    noise_type: str = "data_base",
    seed=None,
    k=None,
    classifier=None,
    preprocessor=None,
    encoding: Optional[str] = "onehot",
    test_source: str = "clean",
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    root=None,
    preprocess_before_filter: bool = True,
    protect_classes=None,
    experiment_name: Optional[str] = None,
    verbose=0
):
    ml = _lazy_ml_imports()
    clone = ml["clone"]
    Pipeline = ml["Pipeline"]

    if classifier is None:
        classifier = ml["RandomForestClassifier"](random_state=33, n_jobs=-1)

    noise_pct = _noise_pct(noise_type, k)
    seed_int = _coerce_int(seed)
    k_int = _coerce_int(k)
    experiment = experiment_name or _default_experiment_name(dataset, noise_type, seed_int, k_int, test_source)

    classification_rows = []

    for fold in tqdm(list(folds), desc=f"5CV baseline {dataset} {noise_type}", disable=~verbose):
        train_df, test_df, _ = _load_fold_views(
            dataset=dataset,
            noise_type=noise_type,
            seed=seed,
            k=k,
            fold=int(fold),
            encoding=encoding,
            root=root,
            test_source=test_source,
            compute_ground_truth=False,
        )

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        protected_mask = _resolve_protected_mask(y_train, protect_classes=protect_classes)
        fold_invalid = y_train.nunique(dropna=False) < 2

        resolved_preprocessor = _resolve_preprocessor(preprocessor, ml, X_train)
        n_train_input = int(X_train.shape[0])
        n_test = int(X_test.shape[0])
        row_start = perf_counter()

        if fold_invalid:
            # _print_invalid_experiment(
            #     experiment=experiment,
            #     dataset=dataset,
            #     fold=fold,
            #     noise_type=noise_type,
            #     seed=seed_int,
            #     k=k_int,
            #     method="baseline",
            #     reason="single_class_in_training_fold",
            #     n_train=n_train_input,
            # )
            baseline_metrics = _safe_metric_rows(["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro"])
            elapsed_s = perf_counter() - row_start
            classification_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_pct": noise_pct,
                    "seed": seed_int,
                    "k": k_int,
                    "fold": int(fold),
                    "method": "baseline",
                    "encoding": encoding,
                    "test_source": test_source,
                    "preprocess_before_filter": preprocess_before_filter,
                    "n_train_input": n_train_input,
                    "n_train_used": n_train_input,
                    "n_test": n_test,
                    "elapsed_s": elapsed_s,
                    "valid_classification": False,
                    "params": _pipeline_params(resolved_preprocessor, None, classifier, True),
                    **baseline_metrics,
                }
            )

            continue

        base_pipe = _build_pipeline(
            clone=clone,
            Pipeline=Pipeline,
            preprocessor=resolved_preprocessor,
            filter_estimator=None,
            classifier=classifier,
            preprocess_before_filter=True,
        )
        base_pred, invalid = _safe_fit_predict(base_pipe, X_train, y_train, X_test)
        if not invalid and protected_mask.any():
            keep_mask = np.ones(n_train_input, dtype=bool)
            keep_mask, invalid = _ensure_two_classes_after_protection(y_train, keep_mask, protected_mask)
        # if invalid:
        #     _print_invalid_experiment(
        #         experiment=experiment,
        #         dataset=dataset,
        #         fold=fold,
        #         noise_type=noise_type,
        #         seed=seed_int,
        #         k=k_int,
        #         method="baseline",
        #         reason="fit_failed_or_too_few_classes_after_protection",
        #         n_train=n_train_input,
        #     )
        baseline_metrics = _classification_metrics(y_test, base_pred, ml) if not invalid else _safe_metric_rows(["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro"])
        elapsed_s = perf_counter() - row_start
        classification_rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_pct": noise_pct,
                "seed": seed_int,
                "k": k_int,
                "fold": int(fold),
                "method": "baseline",
                "encoding": encoding,
                "test_source": test_source,
                "preprocess_before_filter": preprocess_before_filter,
                "n_train_input": n_train_input,
                "n_train_used": n_train_input,
                "n_test": n_test,
                "elapsed_s": elapsed_s,
                "params": _pipeline_params(resolved_preprocessor, None, classifier, True),
                **baseline_metrics,
            }
        )

    classification_df = pd.DataFrame(classification_rows, columns=CLASSIFICATION_COLUMNS)
    return classification_df


def run_5cv_filters(
    dataset: str,
    noise_type: str = "data_base",
    seed=None,
    k=None,
    filters: Optional[Mapping[str, Any]] = None,
    classifier=None,
    preprocessor=None,
    encoding: Optional[str] = "onehot",
    test_source: str = "clean",
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    root=None,
    preprocess_before_filter: bool = True,
    protect_classes=None,
    experiment_name: Optional[str] = None,
    summarize: bool = False,
    verbose=0
):
    ml = _lazy_ml_imports()
    clone = ml["clone"]
    Pipeline = ml["Pipeline"]

    if classifier is None:
        classifier = ml["RandomForestClassifier"](random_state=33, n_jobs=-1)

    if filters is None:
        filters = {}

    noise_pct = _noise_pct(noise_type, k)
    seed_int = _coerce_int(seed)
    k_int = _coerce_int(k)
    experiment = experiment_name or _default_experiment_name(dataset, noise_type, seed_int, k_int, test_source)

    classification_rows = []
    removal_rows = []

    for fold in tqdm(list(folds), desc=f"5CV filters {dataset} {noise_type}", disable=~verbose):
        train_df, test_df, true_noisy_mask, known_ground_truth_mask = _load_fold_views(
            dataset=dataset,
            noise_type=noise_type,
            seed=seed,
            k=k,
            fold=int(fold),
            encoding=encoding,
            root=root,
            test_source=test_source,
            return_ground_truth_known=True,
        )

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        protected_mask = _resolve_protected_mask(y_train, protect_classes=protect_classes)

        resolved_preprocessor = _resolve_preprocessor(preprocessor, ml, X_train)
        n_train_input = int(X_train.shape[0])
        n_test = int(X_test.shape[0])

        for filter_name, filter_estimator in filters.items():
            row_start = perf_counter()
            pipe = _build_pipeline(
                clone=clone,
                Pipeline=Pipeline,
                preprocessor=resolved_preprocessor,
                filter_estimator=filter_estimator,
                classifier=classifier,
                preprocess_before_filter=preprocess_before_filter,
            )
            y_pred, invalid = _safe_fit_predict(pipe, X_train, y_train, X_test)
            class_metrics = _classification_metrics(y_test, y_pred, ml) if not invalid else _safe_metric_rows(["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro"])
            if invalid:
                keep_mask = np.ones(n_train_input, dtype=bool)
                removal_metrics = _safe_metric_rows([
                    "n_true_noisy",
                    "n_known_ground_truth",
                    "n_unknown_ground_truth",
                    "ground_truth_exact",
                    "n_removed_pred",
                    "removed_pct",
                    "acc_removal",
                    "precision_removal",
                    "recall_removal",
                    "f1_removal",
                    "specificity",
                    "mcc",
                ])
            else:
                keep_mask = _extract_keep_mask(pipe, original_n=n_train_input)
                if protected_mask.any():
                    keep_mask, invalid = _ensure_two_classes_after_protection(y_train, keep_mask, protected_mask)
                pred_removed_mask = ~keep_mask
                removal_metrics = _removal_metrics(
                    true_noisy_mask=true_noisy_mask,
                    pred_removed_mask=pred_removed_mask,
                    known_ground_truth_mask=known_ground_truth_mask,
                )
            # if invalid:
            #     _print_invalid_experiment(
            #         experiment=experiment,
            #         dataset=dataset,
            #         fold=fold,
            #         noise_type=noise_type,
            #         seed=seed_int,
            #         k=k_int,
            #         method=filter_name,
            #         reason="fit_failed_or_too_few_classes_after_protection",
            #         n_train=n_train_input,
            #     )
            if np.unique(np.asarray(y_train)[keep_mask]).shape[0] < 2:
                if not invalid:
                    _print_invalid_experiment(
                        experiment=experiment,
                        dataset=dataset,
                        fold=fold,
                        noise_type=noise_type,
                        seed=seed_int,
                        k=k_int,
                        method=filter_name,
                        reason="too_few_classes_after_filter",
                        n_train=n_train_input,
                    )
                invalid = True
                class_metrics = _safe_metric_rows(["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro"])
                removal_metrics = _safe_metric_rows([
                    "n_true_noisy",
                    "n_known_ground_truth",
                    "n_unknown_ground_truth",
                    "ground_truth_exact",
                    "n_removed_pred",
                    "removed_pct",
                    "acc_removal",
                    "precision_removal",
                    "recall_removal",
                    "f1_removal",
                    "specificity",
                    "mcc",
                ])
            elapsed_s = perf_counter() - row_start
            fitted_filter = pipe.named_steps.get("filter")
            if fitted_filter is not None and getattr(fitted_filter, "action", "remove") == "remove":
                n_train_used = int(keep_mask.sum())
            else:
                n_train_used = n_train_input

            pipeline_params = _pipeline_params(resolved_preprocessor, filter_estimator, classifier, preprocess_before_filter)

            classification_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_pct": noise_pct,
                    "seed": seed_int,
                    "k": k_int,
                    "fold": int(fold),
                    "method": filter_name,
                    "encoding": encoding,
                    "test_source": test_source,
                    "preprocess_before_filter": preprocess_before_filter,
                    "n_train_input": n_train_input,
                    "n_train_used": n_train_used,
                    "n_test": n_test,
                    "elapsed_s": elapsed_s,
                    "valid_classification": not invalid,
                    "params": pipeline_params,
                    **class_metrics,
                }
            )

            removal_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_pct": noise_pct,
                    "seed": seed_int,
                    "k": k_int,
                    "fold": int(fold),
                    "filter": filter_name,
                    "encoding": encoding,
                    "test_source": test_source,
                    "preprocess_before_filter": preprocess_before_filter,
                    "n_train_input": n_train_input,
                    "n_train_used": n_train_used,
                    "elapsed_s": elapsed_s,
                    "valid_classification": not invalid,
                    "params": pipeline_params,
                    **removal_metrics,
                }
            )

    classification_df = pd.DataFrame(classification_rows, columns=CLASSIFICATION_COLUMNS)
    removal_df = pd.DataFrame(removal_rows, columns=REMOVAL_COLUMNS)

    if not summarize:
        return classification_df, removal_df

    class_group_cols = ["dataset", "noise_type", "noise_pct", "seed", "k", "method"]
    removal_group_cols = ["dataset", "noise_type", "noise_pct", "seed", "k", "filter"]
    class_metric_cols = ["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro", "elapsed_s"]
    removal_metric_cols = [
        "n_true_noisy",
        "n_known_ground_truth",
        "n_unknown_ground_truth",
        "ground_truth_exact",
        "n_removed_pred",
        "removed_pct",
        "acc_removal",
        "precision_removal",
        "recall_removal",
        "f1_removal",
        "specificity",
        "mcc",
        "elapsed_s",
    ]

    expected_folds = len(tuple(folds))

    valid_classification_groups = (
        classification_df.groupby(class_group_cols, dropna=False)["fold"].nunique().reset_index(name="fold_count")
    )
    valid_classification_groups = valid_classification_groups[
        valid_classification_groups["fold_count"] == expected_folds
    ][class_group_cols]
    classification_df = classification_df.merge(valid_classification_groups, on=class_group_cols, how="inner")

    valid_removal_groups = (
        removal_df.groupby(removal_group_cols, dropna=False)["fold"].nunique().reset_index(name="fold_count")
    )
    valid_removal_groups = valid_removal_groups[
        valid_removal_groups["fold_count"] == expected_folds
    ][removal_group_cols]
    removal_df = removal_df.merge(valid_removal_groups, on=removal_group_cols, how="inner")

    class_summary_df = classification_df.groupby(class_group_cols, dropna=False)[class_metric_cols].agg(["mean", "std"]).reset_index()
    class_summary_df.columns = [col if isinstance(col, str) else "_".join([part for part in col if part]).rstrip("_") for col in class_summary_df.columns]
    class_params = classification_df.groupby(class_group_cols, dropna=False)["params"].first().reset_index(drop=True)
    class_summary_df.insert(len(class_group_cols), "params", class_params)

    removal_summary_df = removal_df.groupby(removal_group_cols, dropna=False)[removal_metric_cols].agg(["mean", "std"]).reset_index()
    removal_summary_df.columns = [col if isinstance(col, str) else "_".join([part for part in col if part]).rstrip("_") for col in removal_summary_df.columns]
    removal_params = removal_df.groupby(removal_group_cols, dropna=False)["params"].first().reset_index(drop=True)
    removal_summary_df.insert(len(removal_group_cols), "params", removal_params)

    return classification_df, removal_df, class_summary_df, removal_summary_df


def run_5cv_experiment(
    dataset: str,
    noise_type: str = "data_base",
    seed=None,
    k=None,
    filters: Optional[Mapping[str, Any]] = None,
    classifier=None,
    preprocessor=None,
    encoding: Optional[str] = "onehot",
    test_source: str = "clean",
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    root=None,
    preprocess_before_filter: bool = True,
    protect_classes=None,
    experiment_name: Optional[str] = None,
    summarize: bool = False,
):
    """Run one 5CV experiment for a dataset/noise setting.

    Parameters
    ----------
    dataset : str
        Dataset folder name (e.g. ``banana``).
    noise_type : str
        ``data_base`` or a noisy folder prefix such as ``att_gaus``.
    seed, k : int | str | None
        Noise folder identifiers. They are ignored for ``data_base``.
    filters : mapping[str, estimator], optional
        Dictionary of filters to evaluate. The baseline is always computed.
    classifier : estimator, optional
        Classifier placed at the end of the pipeline. Defaults to RandomForest.
    preprocessor : estimator, optional
        Preprocessing step before the filter. Defaults to StandardScaler when
        ``encoding`` is not None. Use ``"passthrough"`` to skip it.
    encoding : {None, "ordinal", "onehot"}
        Passed to the dataset loader.
    test_source : {"clean", "noisy"}
        ``clean`` evaluates on the clean test fold from ``data_base``.
        ``noisy`` evaluates on the matching noisy test fold.
    folds : sequence[int]
        Fold ids to evaluate.
    preprocess_before_filter : bool
        If True, use ``preprocess -> filter -> classifier``.
    experiment_name : str, optional
        Label stored in the output tables.

    Returns
    -------
    classification_df, removal_df : pandas.DataFrame
    class_summary_df, removal_summary_df : pandas.DataFrame, optional
    """

    ml = _lazy_ml_imports()
    clone = ml["clone"]
    Pipeline = ml["Pipeline"]

    if classifier is None:
        classifier = ml["RandomForestClassifier"](random_state=33, n_jobs=-1)

    if filters is None:
        filters = {}

    noise_pct = _noise_pct(noise_type, k)
    seed_int = _coerce_int(seed)
    k_int = _coerce_int(k)
    experiment = experiment_name or _default_experiment_name(dataset, noise_type, seed_int, k_int, test_source)

    classification_rows = []
    removal_rows = []

    for fold in tqdm(list(folds), desc=f"5CV {dataset} {noise_type}"):
        train_df, test_df, true_noisy_mask, known_ground_truth_mask = _load_fold_views(
            dataset=dataset,
            noise_type=noise_type,
            seed=seed,
            k=k,
            fold=int(fold),
            encoding=encoding,
            root=root,
            test_source=test_source,
            return_ground_truth_known=True,
        )

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        protected_mask = _resolve_protected_mask(y_train, protect_classes=protect_classes)

        resolved_preprocessor = _resolve_preprocessor(preprocessor, ml, X_train)

        n_train_input = int(X_train.shape[0])
        n_test = int(X_test.shape[0])

        base_pipe = _build_pipeline(
            clone=clone,
            Pipeline=Pipeline,
            preprocessor=resolved_preprocessor,
            filter_estimator=None,
            classifier=classifier,
            preprocess_before_filter=True,
        )
        base_pipe.fit(X_train, y_train)
        base_pred = base_pipe.predict(X_test)

        baseline_metrics = _classification_metrics(y_test, base_pred, ml)
        classification_rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_pct": noise_pct,
                "seed": seed_int,
                "k": k_int,
                "fold": int(fold),
                "method": "baseline",
                "encoding": encoding,
                "test_source": test_source,
                "preprocess_before_filter": preprocess_before_filter,
                "n_train_input": n_train_input,
                "n_train_used": n_train_input,
                "n_test": n_test,
                "params": _pipeline_params(resolved_preprocessor, None, classifier, True),
                **baseline_metrics,
            }
        )

        for filter_name, filter_estimator in filters.items():
            pipe = _build_pipeline(
                clone=clone,
                Pipeline=Pipeline,
                preprocessor=resolved_preprocessor,
                filter_estimator=filter_estimator,
                classifier=classifier,
                preprocess_before_filter=preprocess_before_filter,
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            class_metrics = _classification_metrics(y_test, y_pred, ml)
            keep_mask = _extract_keep_mask(pipe, original_n=n_train_input)
            if protected_mask.any():
                keep_mask = keep_mask | protected_mask
            pred_removed_mask = ~keep_mask
            removal_metrics = _removal_metrics(
                true_noisy_mask=true_noisy_mask,
                pred_removed_mask=pred_removed_mask,
                known_ground_truth_mask=known_ground_truth_mask,
            )

            fitted_filter = pipe.named_steps.get("filter")
            if fitted_filter is not None and getattr(fitted_filter, "action", "remove") == "remove":
                n_train_used = int(keep_mask.sum())
            else:
                n_train_used = n_train_input

            classification_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_pct": noise_pct,
                    "seed": seed_int,
                    "k": k_int,
                    "fold": int(fold),
                    "method": filter_name,
                    "encoding": encoding,
                    "test_source": test_source,
                    "preprocess_before_filter": preprocess_before_filter,
                    "n_train_input": n_train_input,
                    "n_train_used": n_train_used,
                    "n_test": n_test,
                    "valid_classification": not invalid,
                    "params": _pipeline_params(resolved_preprocessor, filter_estimator, classifier, preprocess_before_filter),
                    **class_metrics,
                }
            )

            removal_rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_pct": noise_pct,
                    "seed": seed_int,
                    "k": k_int,
                    "fold": int(fold),
                    "filter": filter_name,
                    "encoding": encoding,
                    "test_source": test_source,
                    "preprocess_before_filter": preprocess_before_filter,
                    "n_train_input": n_train_input,
                    "n_train_used": n_train_used,
                    "valid_classification": not invalid,
                    "params": _pipeline_params(resolved_preprocessor, filter_estimator, classifier, preprocess_before_filter),
                    **removal_metrics,
                }
            )

    classification_df = pd.DataFrame(classification_rows, columns=CLASSIFICATION_COLUMNS)
    removal_df = pd.DataFrame(removal_rows, columns=REMOVAL_COLUMNS)

    if not summarize:
        return classification_df, removal_df

    class_group_cols = ["dataset", "noise_type", "noise_pct", "seed", "k", "method"]
    removal_group_cols = ["dataset", "noise_type", "noise_pct", "seed", "k", "filter"]

    class_metric_cols = ["accuracy", "bal_acc", "f1_macro", "precision_macro", "recall_macro"]
    removal_metric_cols = [
        "n_true_noisy",
        "n_known_ground_truth",
        "n_unknown_ground_truth",
        "ground_truth_exact",
        "n_removed_pred",
        "removed_pct",
        "acc_removal",
        "precision_removal",
        "recall_removal",
        "f1_removal",
        "specificity",
        "mcc",
    ]

    class_summary_df = (
        classification_df.groupby(class_group_cols, dropna=False)[class_metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    class_summary_df.columns = [
        col if isinstance(col, str) else "_".join([part for part in col if part]).rstrip("_")
        for col in class_summary_df.columns
    ]
    class_params = (
        classification_df.groupby(class_group_cols, dropna=False)["params"].first().reset_index(drop=True)
    )
    class_summary_df.insert(len(class_group_cols), "params", class_params)

    removal_summary_df = (
        removal_df.groupby(removal_group_cols, dropna=False)[removal_metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    removal_summary_df.columns = [
        col if isinstance(col, str) else "_".join([part for part in col if part]).rstrip("_")
        for col in removal_summary_df.columns
    ]
    removal_params = (
        removal_df.groupby(removal_group_cols, dropna=False)["params"].first().reset_index(drop=True)
    )
    removal_summary_df.insert(len(removal_group_cols), "params", removal_params)

    return classification_df, removal_df, class_summary_df, removal_summary_df



def run_5cv_grid(
    experiments: Iterable[Mapping[str, Any]],
    save_params: bool = True,
    **shared_kwargs,
):
    """Run several 5CV experiments and concatenate the results.

    Each item in ``experiments`` is a dict with the arguments of
    ``run_5cv_experiment``. Shared keyword arguments are applied to all of them.
    Set ``save_params=False`` to omit the ``params`` columns from saved files.
    """

    classification_frames = []
    removal_frames = []
    class_summary_frames = []
    removal_summary_frames = []

    save_path = shared_kwargs.pop("save_path", None)
    save_format = shared_kwargs.pop("save_format", "pickle")
    save_each = shared_kwargs.pop("save_each", False)

    warnings_path = shared_kwargs.pop("warnings_path", None)
    clear_warnings_file = shared_kwargs.pop("clear_warnings_file", False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    if warnings_path is not None:
        warnings_path = Path(warnings_path)
        warnings_path.parent.mkdir(parents=True, exist_ok=True)

        if clear_warnings_file:
            warnings_path.write_text("", encoding="utf-8")

    def _save_experiment(exp_name, class_df, rem_df, class_summary_df=None, removal_summary_df=None):
        if save_path is None:
            return

        safe_name = str(exp_name).replace("/", "_")

        if save_params:
            class_df_to_save = class_df
            rem_df_to_save = rem_df
            class_summary_df_to_save = class_summary_df
            removal_summary_df_to_save = removal_summary_df
        else:
            class_df_to_save = class_df.drop(columns=["params"], errors="ignore")
            rem_df_to_save = rem_df.drop(columns=["params"], errors="ignore")
            class_summary_df_to_save = (
                None if class_summary_df is None else class_summary_df.drop(columns=["params"], errors="ignore")
            )
            removal_summary_df_to_save = (
                None if removal_summary_df is None else removal_summary_df.drop(columns=["params"], errors="ignore")
            )

        payload = {
            "classification_df": class_df_to_save,
            "removal_df": rem_df_to_save,
            "class_summary_df": class_summary_df_to_save,
            "removal_summary_df": removal_summary_df_to_save,
        }

        if save_format == "pickle":
            out_file = save_path / f"{safe_name}.pkl"
            pd.to_pickle(payload, out_file)
        elif save_format == "csv":
            exp_dir = save_path / safe_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            class_df_to_save.to_csv(exp_dir / "classification.csv", index=False)
            rem_df_to_save.to_csv(exp_dir / "removal.csv", index=False)
            if class_summary_df_to_save is not None:
                class_summary_df_to_save.to_csv(exp_dir / "class_summary.csv", index=False)
            if removal_summary_df_to_save is not None:
                removal_summary_df_to_save.to_csv(exp_dir / "removal_summary.csv", index=False)
        else:
            raise ValueError("save_format must be 'pickle' or 'csv'.")

    def _save_warnings(exp_name, captured_warnings):
        if warnings_path is None:
            return

        if len(captured_warnings) == 0:
            return

        with warnings_path.open("a", encoding="utf-8") as f:
            for w in captured_warnings:
                if issubclass(w.category, DeprecationWarning):
                    continue
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"{w.category.__name__}: {w.message}\n")
                f.write(f"File: {w.filename}\n")
                f.write(f"Line: {w.lineno}\n")
                f.write("-" * 80 + "\n")

    for exp in tqdm(list(experiments), desc=f"5CV experiments"):
        exp = dict(exp)
        exp_name = exp.pop("experiment_name", None)
        exp_name = exp.pop("label", exp_name)

        if warnings_path is not None:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")

                result = run_5cv_filters(
                    experiment_name=exp_name,
                    **shared_kwargs,
                    **exp,
                )

            _save_warnings(exp_name, captured_warnings)

        else:
            result = run_5cv_filters(
                experiment_name=exp_name,
                **shared_kwargs,
                **exp,
            )

        if len(result) == 4:
            class_df, rem_df, class_summary_df, removal_summary_df = result
            class_summary_frames.append(class_summary_df)
            removal_summary_frames.append(removal_summary_df)
        else:
            class_df, rem_df = result
            class_summary_df = None
            removal_summary_df = None

        if save_each:
            _save_experiment(
                exp_name or f"exp_{len(classification_frames)}",
                class_df,
                rem_df,
                class_summary_df,
                removal_summary_df,
            )

        classification_frames.append(class_df)

        if not rem_df.empty:
            removal_frames.append(rem_df)

    classification_df = (
        pd.concat(classification_frames, ignore_index=True)
        if classification_frames
        else pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    )

    removal_df = (
        pd.concat(removal_frames, ignore_index=True)
        if removal_frames
        else pd.DataFrame(columns=REMOVAL_COLUMNS)
    )

    if class_summary_frames:
        class_summary_df = pd.concat(class_summary_frames, ignore_index=True)
        removal_summary_df = pd.concat(removal_summary_frames, ignore_index=True)

        if save_path is not None and not save_each:
            _save_experiment(
                "grid_summary",
                classification_df,
                removal_df,
                class_summary_df,
                removal_summary_df,
            )

        return classification_df, removal_df, class_summary_df, removal_summary_df

    if save_path is not None and not save_each:
        _save_experiment("grid_summary", classification_df, removal_df)

    return classification_df, removal_df
