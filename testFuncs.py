'''
    Module with functions to test noise treatment.
'''

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, make_scorer
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from noisers import *

SCORING_DEFAULT = {
    "Acc": "accuracy",
    "BalAcc": "balanced_accuracy",
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "Prec_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "Rec_macro": make_scorer(recall_score, average="macro", zero_division=0),
}

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
        scoring = SCORING_DEFAULT

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
    model = RandomForestClassifier(random_state=33, n_jobs=-1),
    sc = StandardScaler(),
    k_cv=5
    ):

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
    for (df_name, df) in zip(dfs_names,dfs):

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
            print(f"Processing {df_name} with noise params={np}.")

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
        print("\n")

    return res