"""
Iterative partitioning label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y

from .._detection import attach_detection_report, resample_by_action, validate_action


# Default C4.5-like tree used by the iterative partitioning filter.
c45_like = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=33)


@dataclass
class IPFIterationInfo:
    """Per-iteration diagnostics for iterative partitioning filtering."""

    iter_idx: int
    n_samples_before: int
    n_flagged: int
    frac_flagged: float
    threshold_p: float
    vote_rule: str


@dataclass
class IterativePartitioningFilterResult:
    """Summary of an iterative partitioning filtering run."""

    keep_mask: np.ndarray
    noisy_fraction: float
    noisy_votes: np.ndarray
    n_models: int
    n_iters: int
    history: List[IPFIterationInfo]


class IterativePartitioningFilter(BaseEstimator):
    """Iterative partitioning noise filter.

    Parameters
    ----------
    estimator : estimator, default=c45_like
        Base learner fitted on each partition.
    n_partitions : int, default=10
        Number of stratified partitions built at each iteration.
    vote_rule : {"majority", "consensus"}, default="majority"
        Rule used to flag a sample as noisy from the partition disagreements.
    action : {"remove", "detect"}, default="remove"
        Whether noisy samples are dropped or only detected.
    p_stop : float, default=0.01
        Patience threshold expressed as a fraction of the original dataset.
    k_patience : int, default=3
        Number of consecutive low-yield iterations tolerated before stopping.
    max_iter : int, default=20
        Maximum number of cleaning iterations.
    random_state : int, default=33
        Seed used by the stratified splitter in each iteration.

    Notes
    -----
    Relabel is not implemented yet.
    """

    def __init__(self, estimator=c45_like, n_partitions: int = 10, vote_rule: str = "majority", action: str = "remove", p_stop: float = 0.01, k_patience: int = 3, max_iter: int = 20, random_state: int = 33):
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
            return disagree_counts == n_models
        if self.vote_rule == "majority":
            return disagree_counts > (n_models / 2.0)
        raise ValueError("vote_rule must be 'majority' or 'consensus'")

    def fit(self, X, y):
        """Fit the filter and iteratively partition the training data."""

        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        validate_action(self.action)
        n0 = X.shape[0]
        orig_idx = np.arange(n0)
        alive = np.ones(n0, dtype=bool)
        noisy_votes_global = np.zeros(n0, dtype=int)
        history: List[IPFIterationInfo] = []
        patience_counter = 0
        y_out = y.copy()

        for it in range(1, self.max_iter + 1):
            E_idx = orig_idx[alive]
            X_E = X[E_idx]
            y_E = y_out[E_idx]
            nE = X_E.shape[0]
            if nE < 2 * self.n_partitions:
                break
            skf = StratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state + it)
            preds = np.empty((self.n_partitions, nE), dtype=object)
            for m, (_, part_idx) in enumerate(skf.split(X_E, y_E)):
                model = clone(self.estimator)
                model.fit(X_E[part_idx], y_E[part_idx])
                preds[m] = model.predict(X_E)
            disagree_counts = (preds != y_E[None, :]).sum(axis=0).astype(int)
            noisy_local = self._flag_by_votes(disagree_counts, self.n_partitions)
            noisy_votes_global[E_idx] = disagree_counts
            n_flagged = int(noisy_local.sum())
            frac_flagged = float(n_flagged / max(nE, 1))
            patience_counter = patience_counter + 1 if n_flagged < (self.p_stop * n0) else 0
            history.append(IPFIterationInfo(it, nE, n_flagged, frac_flagged, self.p_stop, self.vote_rule))
            if n_flagged == 0:
                break
            alive[E_idx[noisy_local]] = False
            if patience_counter >= self.k_patience:
                break

        self.result_ = IterativePartitioningFilterResult(keep_mask=alive.copy(), noisy_fraction=float((~alive).mean()), noisy_votes=noisy_votes_global, n_models=int(self.n_partitions), n_iters=int(len(history)), history=history)
        self.X_ = X
        self.y_ = y_out
        # noise_score is not implemented yet for IterativePartitioningFilter.
        attach_detection_report(
            self,
            ~alive,
            observed_labels=y,
            predicted_labels=None,
            noisy_votes=noisy_votes_global,
            history=history,
            vote_rule=self.vote_rule,
            p_stop=float(self.p_stop),
            k_patience=int(self.k_patience),
            max_iter=int(self.max_iter),
        )
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered or detected data."""

        self.fit(X, y)
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def get_filter_report(self) -> Dict[str, Any]:
        """Return a dictionary with the main fit diagnostics."""

        r = self.result_
        last = r.history[-1] if r.history else None
        return {"n_samples": int(self.X_.shape[0]), "n_models_per_iter": int(r.n_models), "n_iters": int(r.n_iters), "vote_rule": self.vote_rule, "action": self.action, "fraction_removed_or_flagged": float(r.noisy_fraction), "last_iter_flagged": int(last.n_flagged) if last else 0, "last_iter_frac_flagged": float(last.frac_flagged) if last else 0.0, "p_stop": float(self.p_stop), "k_patience": int(self.k_patience), "max_iter": int(self.max_iter)}

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
