'''
    Module with functions to test noise treatment.
'''

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from noiseData import *

def urlf_test_in_dfs(
    dfs, 
    dfs_names, 
    noise_levels, 
    rs=33, 
    filter = None, 
    model = RandomForestClassifier(random_state=33, n_jobs=-1),
    sc = StandardScaler(),
    k_cv=5
    ):

    # Initialize dict to store results
    res = {
        df_name : {
        "df_name":[],
        "noise_pct":[],
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
        cv = cross_validate(
            estimator=Pipeline(
                [
                    ("sc", sc),
                    ("model", model),
                ]
            ),
            X=X,
            y=y,
            scoring={
                "Acc":"accuracy",
                "BalAcc":"balanced_accuracy",
                "f1_macro":"f1_macro",
                "Prec_macro":"precision_macro",
                "Rec_macro":"recall_macro"
            },
            n_jobs=-1,
            cv=k_cv)

        # Store results
        res[df_name]["df_name"].append(df_name)
        res[df_name]["noise_pct"].append(-1)    # -1 means baseline with no filtering technique applied
        res[df_name]["Acc"].append(cv["test_Acc"].mean())
        res[df_name]["BalAcc"].append(cv["test_BalAcc"].mean())
        res[df_name]["f1_macro"].append(cv["test_f1_macro"].mean())
        res[df_name]["Prec_macro"].append(cv["test_Prec_macro"].mean())
        res[df_name]["Rec_macro"].append(cv["test_Rec_macro"].mean())

        # Iter through noise_levels
        for nl in noise_levels:
            print(f"Processing {df_name} with noise level={nl}.")
            # Apply random uniform noise with nl as noise level
            y_noisy = urlf(y, noise_level=nl, random_state=rs)

            # Compute results without filter
            cv = cross_validate(
                estimator=Pipeline(
                    [
                        ("sc", sc),
                        ("model", model),
                    ]
                ),
                X=X,
                y=y_noisy,
                scoring={
                    "Acc":"accuracy",
                    "BalAcc":"balanced_accuracy",
                    "f1_macro":"f1_macro",
                    "Prec_macro":"precision_macro",
                    "Rec_macro":"recall_macro"
                },
                n_jobs=-1,
                cv=k_cv)

            # Store results
            res[df_name]["df_name"].append(df_name+"_nf")   # nf means "not filtered"
            res[df_name]["noise_pct"].append(nl)    # -1 means baseline with no filtering technique applied
            res[df_name]["Acc"].append(cv["test_Acc"].mean())
            res[df_name]["BalAcc"].append(cv["test_BalAcc"].mean())
            res[df_name]["f1_macro"].append(cv["test_f1_macro"].mean())
            res[df_name]["Prec_macro"].append(cv["test_Prec_macro"].mean())
            res[df_name]["Rec_macro"].append(cv["test_Rec_macro"].mean())

            # Compute results with filter applied
            cv = cross_validate(
                estimator=Pipeline(
                    [
                        ("filter", filter),
                        ("sc", sc),
                        ("model", model),
                    ]
                ),
                X=X,
                y=y_noisy,
                scoring={
                    "Acc":"accuracy",
                    "BalAcc":"balanced_accuracy",
                    "f1_macro":"f1_macro",
                    "Prec_macro":"precision_macro",
                    "Rec_macro":"recall_macro"
                },
                n_jobs=-1,
                cv=k_cv)

            # Store results
            res[df_name]["df_name"].append(df_name+"_f")   # f means "filtered"
            res[df_name]["noise_pct"].append(nl)    # -1 means baseline with no filtering technique applied
            res[df_name]["Acc"].append(cv["test_Acc"].mean())
            res[df_name]["BalAcc"].append(cv["test_BalAcc"].mean())
            res[df_name]["f1_macro"].append(cv["test_f1_macro"].mean())
            res[df_name]["Prec_macro"].append(cv["test_Prec_macro"].mean())
            res[df_name]["Rec_macro"].append(cv["test_Rec_macro"].mean())
        print("\n")

    return res