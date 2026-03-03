'''
    Module with different filters designed to manage noisy data.
'''


import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y

@dataclass
class EnsembleFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray  # nº de modelos que discrepan con y
    n_models: int

class EnsembleFiltering(BaseEstimator, TransformerMixin):
    """
    Ensemble Filtering (label-noise filtering) sklearn-compatible.

    Idea: train multiple base classifiers with CV, obtain out-of-fold predictions,
    and mark an instance as 'noisy' if many classifiers disagree with its label.
    Then remove it (default) or optionally relabel it with the ensemble vote.

    Parameters
    ----------
    estimators : list
        List of sklearn classifiers (must implement fit/predict).
    cv : int
        Number of folds for out-of-fold predictions.
    mode : {"majority", "consensus"}
        - "majority": remove if (#wrong_votes / n_models) >= threshold (default threshold=0.5)
        - "consensus": remove if all models disagree with y (threshold ignored)
    threshold : float
        Only used for mode="majority". Typical: 0.5 or 0.6.
    action : {"remove", "relabel"}
        - "remove": drop flagged instances
        - "relabel": keep all, but replace y of flagged instances with ensemble majority vote
    random_state : int
        Seed for CV splits.
    """
    def __init__(
        self,
        estimators,
        cv=10,
        mode="majority",
        threshold=0.5,
        action="remove",
        random_state=33,
        return_noisy_samples=False, 
    ):
        self.estimators = estimators
        self.cv = cv
        self.mode = mode
        self.threshold = threshold
        self.action = action
        self.random_state = random_state
        self.return_noisy_samples = return_noisy_samples

    def fit(self, X, y):
        # Check correct form of X,y
        X, y = check_X_y(X, y, accept_sparse=True)
        # Get number of differente classes and turn y to ordinal (with classes 0,1,2,...)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        # Get number of samples
        n = X.shape[0]
        # Get number of estimators
        m = len(self.estimators)
        if m < 2:
            raise ValueError("Provide at least 2 estimators for ensemble filtering.")
        # Initialize StratifiedKFold strategy
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Initialize OutOfFold prediction matrix (m rows, n cols) -> (n_models, n_samples)
        oof_preds = np.empty((m, n), dtype=int)
        # Fill OutOfFold prediction matrix
        for est_idx, est in enumerate(self.estimators):
            for train_indices, test_indices in skf.split(X, y_idx):
                # Clone the estimator from zero (to avoid refitting)
                model = clone(est)
                # Fit the model with all folds but one out
                model.fit(X[train_indices], y_idx[train_indices])
                # Predict labels on the fold left above
                oof_preds[est_idx, test_indices] = model.predict(X[test_indices])
                
        # Compare each set of estimator predicitons (rows of oof_preds) to the real ones (y_idx)
        # and compute the pct of missclassification for each sample over all the estimators
        wrong_votes = (oof_preds != y_idx[None, :]).sum(axis=0)  # (n,)
        wrong_frac = wrong_votes / m

        # Compute action mask based on self.action
        if self.mode == "consensus":
            noisy_mask = (wrong_votes == m)
        elif self.mode == "majority":
            noisy_mask = (wrong_frac >= self.threshold)
        else:
            raise ValueError("mode must be 'majority' or 'consensus'")

        # Compute mask of samples to keep
        keep_mask = ~noisy_mask

        # TODO: Añadir apartado relabel, quizá compatible con otras librerías o similar
        # # If relabel: replace noisy labels by ensemble majority vote (mode)
        # self.y_relabel_ = None
        # if self.action == "relabel":
        #     # majority vote prediction per sample
        #     # bincount per column (fast enough for moderate sizes)
        #     y_new = y_idx.copy()
        #     for i in np.where(noisy_mask)[0]:
        #         counts = np.bincount(oof_preds[:, i], minlength=n_classes)
        #         y_new[i] = counts.argmax()
        #     self.y_relabel_ = self.classes_[y_new]

        self.result_ = EnsembleFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            noisy_votes=wrong_votes,
            n_models=m,
        )
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, X):
        # Turn X to array if it's not yet
        X = np.asarray(X) if not hasattr(X, "shape") else X

        # Remove noisy_samples following self.mode criterion if self.action=="remove"
        if self.action == "remove":
            return X[self.result_.keep_mask]

        return X # TODO: Añadir que pasa si relabel

    def fit_resample(self, X, y):
        # Fit the ensmble filter
        self.fit(X, y)
        # Once fitted return X,y following self.mode criterion
        if self.action == "remove":
            return self.X_[self.result_.keep_mask], self.y_[self.result_.keep_mask]
        else:  # relabel
            return self.X_, self.y_relabel_

    def get_filter_report(self):
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models": int(self.result_.n_models),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "mode": self.mode,
            "threshold": self.threshold if self.mode == "majority" else None,
            "action": self.action,
        }
