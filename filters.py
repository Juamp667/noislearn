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

class EnsembleFiltering(BaseEstimator): #, TransformerMixin TODO: Compatibilizar con pipeline de sklearn (ahora es compatible con pipeline de imblearn solo)
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

    # def transform(self, X):
    #     # Turn X to array if it's not yet
    #     X = np.asarray(X) if not hasattr(X, "shape") else X

    #     # Remove noisy_samples following self.mode criterion if self.action=="remove"
    #     if self.action == "remove":
    #         return X[self.result_.keep_mask]

    #     return X # TODO: Añadir que pasa si relabel

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




@dataclass
class ClassificationFilterResult:
    """
    Stores the outcome of the filtering process for a *single* classifier.

    Attributes
    ----------
    keep_mask : np.ndarray of shape (n_samples,)
        Boolean mask indicating which samples are kept (True) vs flagged as noisy (False).
    noisy_fraction : float
        Fraction of samples flagged as noisy.
    oof_pred : np.ndarray of shape (n_samples,)
        Out-of-fold predictions (in original label space) for each sample.
    """
    keep_mask: np.ndarray
    noisy_fraction: float
    oof_pred: np.ndarray


class ClassificationFilter(BaseEstimator):
    """
    ClassificationFilter (single-estimator label-noise filter), imblearn-compatible via `fit_resample`.

    Idea
    ----
    1) Train ONE base classifier using Stratified CV to obtain out-of-fold (OOF) predictions.
    2) Flag a sample as "noisy" if its OOF prediction != its label.
    3) Apply an action:
       - remove: drop flagged samples (returns filtered X, y)
       - relabel: replace noisy labels by the OOF prediction (optional)

    Notes
    -----
    - This class is meant to work as a *sampler* inside `imblearn.pipeline.Pipeline`
      because it can change both X and y via `fit_resample`.
    - A vanilla `sklearn.pipeline.Pipeline` expects transformers to only transform X,
      so `fit_resample`-based filters belong to imblearn pipelines.

    Parameters
    ----------
    estimator : sklearn estimator
        Base classifier implementing `fit` and `predict`.
    cv : int, default=10
        Number of Stratified folds used to generate OOF predictions.
    action : {"remove", "relabel"}, default="remove"
        - "remove": drop samples where OOF prediction != y
        - "relabel": keep all samples but replace y by OOF predictions where they disagree
    random_state : int, default=33
        Random seed for the StratifiedKFold shuffling.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels seen during fit (original label space).
    result_ : ClassificationFilterResult
        Filtering summary: keep mask, noisy fraction, and OOF predictions.
    X_ : array-like
        Cached input X seen during fit.
    y_ : array-like
        Cached input y seen during fit.
    """

    def __init__(self, estimator, cv=10, action="remove", random_state=33):
        self.estimator = estimator
        self.cv = cv
        self.action = action
        self.random_state = random_state

    def fit(self, X, y):
        # --------------------------
        # 1) Validate input (X, y)
        # --------------------------
        X, y = check_X_y(X, y, accept_sparse=True)

        # ------------------------------------------------------------
        # 2) Map labels to ordinal indices 0..K-1 (keeps original classes)
        # ------------------------------------------------------------
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n = X.shape[0]

        # ------------------------------------------------
        # 3) Build Stratified CV splitter for OOF predictions
        # ------------------------------------------------
        skf = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        # -------------------------------------------------------
        # 4) Compute OOF predictions for the SINGLE base estimator
        # -------------------------------------------------------
        oof_pred_idx = np.empty(n, dtype=int)

        for train_idx, test_idx in skf.split(X, y_idx):
            # clone to avoid information leakage / refit issues
            model = clone(self.estimator)

            # fit on train fold (ordinal labels)
            model.fit(X[train_idx], y_idx[train_idx])

            # predict on held-out fold
            oof_pred_idx[test_idx] = model.predict(X[test_idx])

        # Convert OOF predictions back to original label space
        oof_pred = self.classes_[oof_pred_idx]

        # ------------------------------------------
        # 5) Decide which samples are flagged as noisy
        # ------------------------------------------
        noisy_mask = (oof_pred_idx != y_idx)
        keep_mask = ~noisy_mask

        # ----------------------------------------
        # 6) Store results & cache X,y for resample
        # ----------------------------------------
        self.result_ = ClassificationFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            oof_pred=oof_pred,
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """
        imblearn sampler API: fit the filter and return (X_res, y_res).

        Returns
        -------
        X_res, y_res : filtered arrays
            If action="remove": returns only samples where keep_mask is True.
            If action="relabel": returns all X and y with noisy labels replaced by OOF preds.
        """
        self.fit(X, y)

        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]

        if self.action == "relabel":
            # relabel only the flagged ones (keep clean labels as-is)
            y_new = np.asarray(self.y_).copy()
            noisy_idx = np.where(~self.result_.keep_mask)[0]
            y_new[noisy_idx] = self.result_.oof_pred[noisy_idx]
            return self.X_, y_new

        raise ValueError("action must be 'remove' or 'relabel'")

    def get_filter_report(self):
        """
        Small summary dictionary for logging/debugging.
        """
        return {
            "n_samples": int(self.X_.shape[0]),
            "removed_or_flagged": int((~self.result_.keep_mask).sum()),
            "fraction_flagged": float(self.result_.noisy_fraction),
            "cv": int(self.cv),
            "action": self.action,
        }





import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y
from sklearn.tree import DecisionTreeClassifier

c45_like = DecisionTreeClassifier(
    criterion="entropy",   # information gain (closer to C4.5/ID3 than gini)
    splitter="best",
    random_state=33
)

@dataclass
class IPFIterationInfo:
    iter_idx: int
    n_samples_before: int
    n_flagged: int
    frac_flagged: float
    threshold_p: float
    vote_rule: str


@dataclass
class IterativePartitioningFilterResult:
    """
    Stores the outcome of the Iterative Partitioning Filter (IPF).

    Attributes
    ----------
    keep_mask : np.ndarray of shape (n_original_samples,)
        Boolean mask in the ORIGINAL sample indexing indicating which samples remain.
    noisy_fraction : float
        Fraction of ORIGINAL samples removed/flagged by the end.
    noisy_votes : np.ndarray of shape (n_original_samples,)
        Number of models that disagreed with the sample's label at the LAST iteration
        where the sample was still present. For samples removed earlier, this stores
        the last known votes before removal.
    n_models : int
        Number of partition models used per iteration.
    n_iters : int
        Number of IPF iterations executed.
    history : list[IPFIterationInfo]
        Per-iteration summary.
    """
    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int
    n_iters: int
    history: List[IPFIterationInfo]


class IterativePartitioningFilter(BaseEstimator):
    """
    Iterative Partitioning Filter (IPF) for label-noise filtering.

    High-level idea
    ---------------
    Repeatedly:
      1) Split current working set E into `n_partitions` stratified subsets.
      2) Train one classifier per subset.
      3) Each classifier predicts labels for ALL samples in E.
      4) Count how many classifiers disagree with each sample's current label.
      5) Flag as noisy using a voting rule:
         - "majority": flagged if disagreements > n_models/2
         - "consensus": flagged if disagreements == n_models
      6) Action:
         - remove: delete flagged samples from E
         - relabel: replace their labels by the ensemble majority vote (optional)

    Stopping criterion (classic IPF-style)
    --------------------------------------
    Stop when for `k_patience` consecutive iterations, the number of newly-flagged
    samples is < `p_stop` fraction of the ORIGINAL training size.

    Notes
    -----
    - "C4.5" is commonly approximated in sklearn with DecisionTreeClassifier(criterion="entropy")
      but here you can pass ANY sklearn classifier as `estimator`.
    - This implements an imblearn-like API via `fit_resample(X, y)`.

    Parameters
    ----------
    estimator : sklearn estimator
        Base classifier implementing fit/predict on X and y (any label space).
    n_partitions : int, default=10
        Number of partitions (= number of models per iteration).
    vote_rule : {"majority","consensus"}, default="majority"
        Rule to flag noisy samples.
    action : {"remove","relabel"}, default="remove"
        What to do with flagged samples.
    p_stop : float, default=0.01
        Threshold as a fraction of ORIGINAL training size for "few new removals".
        E.g. 0.01 means < 1% newly flagged triggers patience counting. TODO:¿Añadir/analizar_impacto_de un p_stop schema?
    k_patience : int, default=3
        Number of consecutive iterations satisfying the p_stop condition to stop.
    max_iter : int, default=20
        Safety cap on iterations.
    random_state : int, default=33
        Random seed for stratified splitting (shuffle=True).
    """

    def __init__(
        self,
        estimator = c45_like,
        n_partitions: int = 10,
        vote_rule: str = "majority",
        action: str = "remove",
        p_stop: float = 0.01,
        k_patience: int = 3,
        max_iter: int = 20,
        random_state: int = 33,
    ):
        self.estimator = estimator
        self.n_partitions = n_partitions
        self.vote_rule = vote_rule
        self.action = action
        self.p_stop = p_stop
        self.k_patience = k_patience
        self.max_iter = max_iter
        self.random_state = random_state

    def _flag_by_votes(self, disagree_counts: np.ndarray, n_models: int) -> np.ndarray:
        if self.vote_rule == "consensus":
            return disagree_counts == n_models  # Every model misclassifies the same sample
        if self.vote_rule == "majority":
            return disagree_counts > (n_models / 2.0)   # More than a half of the models misclassify the same sample 
        raise ValueError("vote_rule must be 'majority' or 'consensus'")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        
        n0 = X.shape[0] # n_smaples
        orig_idx = np.arange(n0)

        # Track which ORIGINAL samples 'i' are still in E <-> alive[i]=1
        alive = np.ones(n0, dtype=bool)

        # Keep last-known votes (in ORIGINAL index space)
        noisy_votes_global = np.zeros(n0, dtype=int)

        history: List[IPFIterationInfo] = []
        patience_counter = 0

        for it in range(1, self.max_iter + 1):
            # Current working set E
            E_idx = orig_idx[alive] # Indices of samples not removed (those alive)
            X_E = X[E_idx]
            y_E = y[E_idx]

            nE = X_E.shape[0]   # n_samples_alive
            if nE < 2 * self.n_partitions:  # if (just 2 samples alive)
                # not enough samples to partition meaningfully
                break

            # Stratified partitions on current labels
            skf = StratifiedKFold(
                n_splits=self.n_partitions,
                shuffle=True,
                random_state=self.random_state + it,  # vary split each iteration
            )

            # Predictions from each model across ALL samples_alive in iteration 'it'
            # shape: (n_partitions, nE)
            preds = np.empty((self.n_partitions, nE), dtype=object)

            for m, (_, part_idx) in enumerate(skf.split(X_E, y_E)):
                model = clone(self.estimator)
                model.fit(X_E[part_idx], y_E[part_idx])   # train only on the partition Ei
                preds[m] = model.predict(X_E)             # predict on all E
            
            # Sum the amount of disagreements of each model for each sample_alive
            disagree_counts = (preds != y_E[None, :]).sum(axis=0).astype(int)

            # TODO: ¡¡¡¡Repasar el apartado de relabel!!!!
            # Majority vote labels (for relabel action)
            # robust majority: per column, take most frequent value
            # (works for strings/objects too)
            if self.action == "relabel":
                maj = np.empty(nE, dtype=object)
                for j in range(nE):
                    col = preds[:, j]
                    vals, cnts = np.unique(col, return_counts=True)
                    maj[j] = vals[np.argmax(cnts)]
            else:
                maj = None

            # Flag noisy samples in samples_alive using vote rule
            noisy_local = self._flag_by_votes(disagree_counts, self.n_partitions)

            # Update global votes snapshot for those still alive
            noisy_votes_global[E_idx] = disagree_counts

            # Compute number and fraction of alive_samples flagged as noisy in iteration 'it'
            n_flagged = int(noisy_local.sum())
            frac_flagged = float(n_flagged / max(nE, 1))

            # Stopping criterion: newly flagged < p_stop * n0
            if n_flagged < (self.p_stop * n0):  # If (the pct of flagged samples is less than...)
                patience_counter += 1
            else:
                patience_counter = 0

            history.append(
                IPFIterationInfo(
                    iter_idx=it,
                    n_samples_before=nE,
                    n_flagged=n_flagged,
                    frac_flagged=frac_flagged,
                    threshold_p=self.p_stop,
                    vote_rule=self.vote_rule,
                )
            )

            if n_flagged == 0:  # If (no sample is classified as noisy)
                break

            # Apply action
            if self.action == "remove":
                # Remove flagged from alive set
                # ie: set noisy sample_idx as not_alive
                alive[E_idx[noisy_local]] = False

            # TODO: Repasar actuación RELABEL
            elif self.action == "relabel":
                # Relabel flagged in the ORIGINAL y array for samples in E
                flagged_orig = E_idx[noisy_local]
                y[flagged_orig] = maj[noisy_local]
                # Keep them (do not remove)
            else:
                raise ValueError("action must be 'remove' or 'relabel'")

            if patience_counter >= self.k_patience: # If (max_patience is reached)
                break
        
        self.result_ = IterativePartitioningFilterResult(
            keep_mask=alive.copy(),
            noisy_fraction=float((~alive).mean()) if self.action == "remove" else float(
                # if relabel, interpret "noisy fraction" as final-iteration flagged fraction over original
                # (you can redefine if you prefer)
                (history[-1].n_flagged / n0) if history else 0.0
            ),
            noisy_votes=noisy_votes_global,
            n_models=int(self.n_partitions),
            n_iters=int(len(history)),
            history=history,
        )

        # cache for resampling
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]
        # relabel keeps all samples
        return self.X_, self.y_

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        last = r.history[-1] if r.history else None
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_models_per_iter": int(r.n_models),
            "n_iters": int(r.n_iters),
            "vote_rule": self.vote_rule,
            "action": self.action,
            "fraction_removed_or_flagged": float(r.noisy_fraction),
            "last_iter_flagged": int(last.n_flagged) if last else 0,
            "last_iter_frac_flagged": float(last.frac_flagged) if last else 0.0,
            "p_stop": float(self.p_stop),
            "k_patience": int(self.k_patience),
            "max_iter": int(self.max_iter),
        }