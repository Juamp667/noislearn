'''
    Module with functions to test noise treatment.
'''

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    _ML_AVAILABLE = True
except ModuleNotFoundError:
    f1_score = precision_score = recall_score = make_scorer = None
    Pipeline = RandomForestClassifier = cross_validate = StandardScaler = None
    _ML_AVAILABLE = False

from noisers import *
from tqdm import tqdm
from tqdm.contrib import tzip

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"


@dataclass(frozen=True)
class _AttributeSpec:
    name: str
    kind: str
    categories: Tuple[str, ...] = ()


def _clean_token(token: str) -> str:
    return token.strip().strip('"\'')


def _csv_fields(text: str):
    return [_clean_token(value) for value in next(csv.reader([text], skipinitialspace=True))]


def _split_attribute_declaration(raw: str):
    raw = raw.strip()
    if not raw:
        raise ValueError("Malformed @attribute declaration.")

    if " " in raw:
        name, spec = raw.split(None, 1)
        spec_lower = spec.lower()
        if spec.startswith("{") or spec.startswith("[") or spec_lower.startswith(("real", "integer", "numeric", "continuous")):
            return name.strip(), spec.strip()

    for marker in ("{", "["):
        idx = raw.find(marker)
        if idx > 0:
            return raw[:idx].strip(), raw[idx:].strip()

    parts = raw.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return raw.strip(), ""


def _parse_attribute_line(line: str) -> _AttributeSpec:
    name, spec = _split_attribute_declaration(line.strip()[len("@attribute"):].strip())
    if spec.startswith("{"):
        inner = spec[1:-1].strip()
        categories = tuple(_csv_fields(inner)) if inner else ()
        return _AttributeSpec(name=name, kind="nominal", categories=categories)
    return _AttributeSpec(name=name, kind="numeric")


def _parse_name_list(text: str):
    text = text.strip()
    if not text:
        return []
    return _csv_fields(text)


def _normalize_seed(seed):
    if seed is None:
        return None
    seed_text = str(seed).strip()
    if seed_text.startswith("seed_"):
        return seed_text
    if seed_text.isdigit():
        return f"seed_{int(seed_text):02d}"
    return seed_text


def _normalize_level(level):
    if level is None:
        return None
    level_text = str(level).strip()
    if level_text.isdigit():
        return str(int(level_text))
    return level_text


def _normalize_dataset_names(datasets):
    if datasets is None:
        return None
    if isinstance(datasets, str):
        return [datasets]
    return list(datasets)


def _normalize_fold_filter(fold):
    if fold is None:
        return None
    if isinstance(fold, (list, tuple, set)):
        values = fold
    else:
        values = [fold]

    normalized = set()
    for value in values:
        text = str(value).strip()
        if text.isdigit():
            text = str(int(text))
        normalized.add(text)
    return normalized


def _parse_dataset_file_kind(file_path: Path):
    stem = file_path.stem
    if stem.endswith("-cc"):
        return "cc", None

    try:
        _, tail = stem.rsplit("-", 1)
    except ValueError:
        return None, None

    match = re.match(r"^(?P<fold>\d+)(?P<kind>tra|tst)$", tail)
    if not match:
        return None, None

    return match.group("kind"), int(match.group("fold"))


def _dataset_base_dir(noise_type: str = "data_base", seed=None, k=None, root=None) -> Path:
    root_path = Path(root) if root is not None else DATASET_ROOT
    if noise_type == "data_base":
        base_dir = root_path / "data_base"
    else:
        seed_folder = _normalize_seed(seed)
        level_folder = _normalize_level(k)
        if seed_folder is None or level_folder is None:
            raise ValueError("seed and k are required for noisy datasets.")
        base_dir = root_path / noise_type / seed_folder / level_folder

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {base_dir}")
    return base_dir


def _keel_dataframe_from_file(
    file_path: Path,
    encoding: Optional[str] = None,
    relative_root: Optional[Path] = None,
) -> pd.DataFrame:
    if encoding not in {None, "ordinal", "onehot"}:
        raise ValueError("encoding must be one of: None, 'ordinal', 'onehot'.")

    relation = None
    attribute_specs = []
    outputs = []
    rows = []
    data_started = False

    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if not data_started:
                if lower.startswith("@relation"):
                    parts = line.split(None, 1)
                    relation = parts[1].strip() if len(parts) > 1 else None
                elif lower.startswith("@attribute"):
                    attribute_specs.append(_parse_attribute_line(line))
                elif lower.startswith("@outputs"):
                    outputs = _parse_name_list(line[len("@outputs"):])
                elif lower.startswith("@data"):
                    data_started = True
                continue

            if line.startswith("@"):
                continue

            rows.append((line_no, _csv_fields(line)))

    if not attribute_specs:
        raise ValueError(f"No @attribute declarations found in {file_path}.")
    if not rows:
        raise ValueError(f"No data rows found in {file_path}.")

    expected_cols = len(attribute_specs)
    for line_no, row in rows:
        if len(row) != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} values in {file_path} at line {line_no}, got {len(row)}."
            )

    columns = [spec.name for spec in attribute_specs]
    raw_df = pd.DataFrame([row for _, row in rows], columns=columns)
    raw_df = raw_df.replace({"?": np.nan, "": np.nan})

    for spec in attribute_specs:
        if spec.kind == "numeric":
            raw_df[spec.name] = pd.to_numeric(raw_df[spec.name], errors="coerce")

    spec_by_name = {spec.name: spec for spec in attribute_specs}
    target_names = [name for name in outputs if name in spec_by_name]
    if not target_names:
        target_names = [attribute_specs[-1].name]

    feature_specs = [spec for spec in attribute_specs if spec.name not in target_names]

    if encoding is None:
        feature_df = raw_df[[spec.name for spec in feature_specs]].copy()
    elif encoding == "ordinal":
        feature_data = {}
        for spec in feature_specs:
            if spec.kind == "numeric":
                feature_data[spec.name] = pd.to_numeric(raw_df[spec.name], errors="coerce")
            else:
                mapping = {category: idx for idx, category in enumerate(spec.categories)}
                feature_data[spec.name] = raw_df[spec.name].map(mapping)
        feature_df = pd.DataFrame(feature_data, index=raw_df.index)
    else:
        encoded_parts = []
        for spec in feature_specs:
            if spec.kind == "numeric":
                encoded_parts.append(pd.to_numeric(raw_df[spec.name], errors="coerce").to_frame(name=spec.name))
            else:
                categorical = pd.Categorical(raw_df[spec.name], categories=spec.categories)
                dummies = pd.get_dummies(categorical, prefix=spec.name, prefix_sep="=", dtype=int)
                expected_cols = [f"{spec.name}={category}" for category in spec.categories]
                encoded_parts.append(dummies.reindex(columns=expected_cols, fill_value=0))

        feature_df = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=raw_df.index)

    target_df = raw_df[target_names].copy()
    result = pd.concat([feature_df, target_df], axis=1)
    relative_path = None
    if relative_root is not None:
        try:
            relative_path = str(file_path.relative_to(relative_root).with_suffix(""))
        except ValueError:
            relative_path = str(file_path.with_suffix(""))
    else:
        relative_path = str(file_path.with_suffix(""))

    result.attrs.update(
        {
            "source_path": str(file_path),
            "relative_path": relative_path,
            "relation": relation,
            "encoding": encoding,
            "target_names": tuple(target_names),
        }
    )
    return result


def _iter_dataset_files(dataset_dir: Path, split=None, fold=None):
    split_filter = None if split in {None, "all"} else {str(split).strip().lower()}
    fold_filter = _normalize_fold_filter(fold)

    for file_path in sorted(dataset_dir.glob("*.dat")):
        kind, file_fold = _parse_dataset_file_kind(file_path)
        if kind is None:
            continue

        if split_filter is not None and kind not in split_filter:
            continue

        if fold_filter is not None:
            if file_fold is None:
                continue
            if str(file_fold) not in fold_filter:
                continue

        yield file_path, kind, file_fold


def load_dataset_df(
    dataset: str,
    noise_type: str = "data_base",
    seed=None,
    k=None,
    split: Optional[str] = None,
    fold=None,
    encoding: Optional[str] = None,
    root=None,
) -> pd.DataFrame:
    """Load one KEEL `.dat` file as a DataFrame.

    For `data_base`, `split=None` defaults to the `cc` file.
    For noisy datasets, `split` and `fold` must identify a single file.
    """

    root_path = Path(root) if root is not None else DATASET_ROOT
    dataset_dir = _dataset_base_dir(noise_type=noise_type, seed=seed, k=k, root=root_path) / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    effective_split = split
    if noise_type == "data_base" and effective_split is None:
        effective_split = "cc"
    elif noise_type != "data_base" and effective_split is None:
        raise ValueError("For noisy datasets, split must be 'tra', 'tst', or 'cc'.")

    matches = list(_iter_dataset_files(dataset_dir, split=effective_split, fold=fold))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file for dataset={dataset!r}, noise_type={noise_type!r}, "
            f"seed={seed!r}, k={k!r}, split={effective_split!r}, fold={fold!r}. Found {len(matches)}."
        )

    file_path, kind, file_fold = matches[0]
    df = _keel_dataframe_from_file(file_path, encoding=encoding, relative_root=root_path)
    df.attrs.update(
        {
            "dataset": dataset,
            "noise_type": noise_type,
            "seed": _normalize_seed(seed),
            "k": _normalize_level(k),
            "split": kind,
            "fold": file_fold,
        }
    )
    return df


def load_dataset_dfs(
    datasets: Optional[Sequence[str]] = None,
    noise_type: str = "data_base",
    seed=None,
    k=None,
    split: Optional[str] = None,
    fold=None,
    encoding: Optional[str] = None,
    root=None,
):
    """Load many KEEL files and return `(dfs, dfs_names)`.

    If `datasets` is None, dataset folders are discovered automatically.
    `split=None` means all matching `.dat` files inside each dataset folder.
    """

    root_path = Path(root) if root is not None else DATASET_ROOT
    base_dir = _dataset_base_dir(noise_type=noise_type, seed=seed, k=k, root=root_path)
    dataset_names = _normalize_dataset_names(datasets)
    if dataset_names is None:
        dataset_names = sorted(entry.name for entry in base_dir.iterdir() if entry.is_dir())

    dfs = []
    dfs_names = []

    for dataset_name in dataset_names:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

        for file_path, kind, file_fold in _iter_dataset_files(dataset_dir, split=split, fold=fold):
            df = _keel_dataframe_from_file(file_path, encoding=encoding, relative_root=root_path)
            df.attrs.update(
                {
                    "dataset": dataset_name,
                    "noise_type": noise_type,
                    "seed": _normalize_seed(seed),
                    "k": _normalize_level(k),
                    "split": kind,
                    "fold": file_fold,
                }
            )
            dfs.append(df)
            dfs_names.append(str(file_path.relative_to(root_path).with_suffix("")))

    if not dfs:
        raise FileNotFoundError(
            f"No dataset files found for noise_type={noise_type!r}, seed={seed!r}, k={k!r}, split={split!r}, fold={fold!r}."
        )

    return dfs, dfs_names


def _build_scoring_default():
    if make_scorer is None:
        raise ImportError("scikit-learn is required for evaluation helpers.")

    return {
        "Acc": "accuracy",
        "BalAcc": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "Prec_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "Rec_macro": make_scorer(recall_score, average="macro", zero_division=0),
    }


SCORING_DEFAULT = _build_scoring_default() if _ML_AVAILABLE else None

def run_cv_and_store(
    res: dict,
    df_key: str,          # key in res (your df_name)
    row_name: str,        # what you want to store in res[df_key]["df_name"] (e.g. "iris_nf")
    noise_kw: float,     # -1 baseline, or nl
    X,
    y,
    estimator,
    k_cv: int = 5,
    scoring: dict = None,
    n_jobs: int = -1,
):
    """
    Run cross_validate and store mean test metrics into res[df_key].

    Parameters
    ----------
    res : dict
        Your nested results dict.
    df_key : str
        Which dataset bucket to store into (e.g. df_name).
    row_name : str
        Label stored in the 'df_name' list (e.g. df_name, df_name+'_nf', df_name+'_f').
    noise_pct : float
        Noise level stored in 'noise_pct' list (use -1 for baseline).
    X, y : array-like
        Data and labels.
    estimator : sklearn estimator
        Pipeline / model to be evaluated by CV.
    k_cv : int
        Number of folds.
    scoring : dict
        Scoring dict for cross_validate.
    n_jobs : int
        Parallel jobs for cross_validate.

    Returns
    -------
    cv : dict
        cross_validate output dict (so you can inspect raw fold scores if needed).
    """
    if scoring is None:
        if SCORING_DEFAULT is None:
            raise ImportError("scikit-learn is required for run_cv_and_store.")
        scoring = SCORING_DEFAULT

    if cross_validate is None:
        raise ImportError("scikit-learn is required for run_cv_and_store.")

    cv = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=k_cv
    )

    # Store results
    res[df_key]["df_name"].append(row_name)
    res[df_key]["noise_kw"].append(noise_kw)
    res[df_key]["Acc"].append(cv["test_Acc"].mean())
    res[df_key]["BalAcc"].append(cv["test_BalAcc"].mean())
    res[df_key]["f1_macro"].append(cv["test_f1_macro"].mean())
    res[df_key]["Prec_macro"].append(cv["test_Prec_macro"].mean())
    res[df_key]["Rec_macro"].append(cv["test_Rec_macro"].mean())

    return cv


def urlf_test_in_dfs(
    dfs, 
    dfs_names, 
    noise_kw, 
    rs=33, 
    filtr = None, 
    noiser = None,
    model = None,
    sc = None,
    k_cv=5,
    verbose = 0
    ):

    if Pipeline is None or RandomForestClassifier is None or StandardScaler is None:
        raise ImportError("scikit-learn and imbalanced-learn are required for urlf_test_in_dfs.")

    if model is None:
        model = RandomForestClassifier(random_state=33, n_jobs=-1)
    if sc is None:
        sc = StandardScaler()

    # Initialize dict to store results
    res = {
        df_name : {
        "df_name":[],
        "noise_kw":[],
        "Acc":[],
        "BalAcc":[],
        "f1_macro":[],
        "Prec_macro":[],
        "Rec_macro":[]
        } for df_name in dfs_names
    }

    # Iter through dataframes
    
    for (df_name, df) in zip(tqdm(dfs_names, "Cycling through dataframes"), dfs):

        # Extract attributes and target from df
        X = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values

        # First compute baseline (no filter nor noise) results with df data
        pipe_base = Pipeline([("sc", sc), ("model", model)])

        run_cv_and_store(
            res=res,
            df_key=df_name,
            row_name=df_name,
            noise_kw={},
            X=X,
            y=y,
            estimator=pipe_base,
            k_cv=k_cv
        )

        # Iter through noise_params
        for np in noise_kw:
            if verbose == 1:
                print(f"Processing {df_name} with noise params={np}.")
                print("\n")

            # Initialize noiser
            noiser.set_params(**np)
            # Compute results without filter applied
            run_cv_and_store(
                res=res,
                df_key=df_name,
                row_name=df_name + "_nf",
                noise_kw=np,
                X=X,
                y=y,
                estimator=Pipeline(
                    [
                        ("noiser", noiser),
                        ("sc", sc),
                        ("model", model),
                    ]
                ),
                k_cv=k_cv
            )

            # Compute results with filter applied
            run_cv_and_store(
                res=res,
                df_key=df_name,
                row_name=df_name + "_f",
                noise_kw=np,
                X=X,
                y=y,
                estimator=Pipeline(
                    [
                        ("noiser", noiser),
                        ("filter", filtr),
                        ("sc", sc),
                        ("model", model),
                    ]
                ),
                k_cv=k_cv
            )
        

    return res
