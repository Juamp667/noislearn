"""
Multiedit label-noise filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y


@dataclass
class MultiEditFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    removed_total: int
    n_iters: int
    n_blocks: int
    nn_pred: np.ndarray


class MultiEditFilter(BaseEstimator):
    def __init__(self, n_neighbors: int = 3, n_blocks: int = 10, metric: str = "minkowski", p: int = 2, action: str = "remove", random_state: int = 33, n_jobs: Optional[int] = None, max_iter: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.n_blocks = n_blocks
        self.metric = metric
        self.p = p
        self.action = action
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def _predict_block(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
        if X_train.shape[0] == 0:
            return np.empty(X_test.shape[0], dtype=object)

        k_eff = min(k, X_train.shape[0]) # Exec KNN with min(n_neighbors, number_of_instances_in_Xtrain)
        knn = KNeighborsClassifier(
            n_neighbors=k_eff,
            metric=self.metric,
            p=self.p if self.metric == "minkowski" else None,
            n_jobs=self.n_jobs,
        )
        knn.fit(X_train, y_train)
        return knn.predict(X_test)

    def _make_stratified_blocks(self, y: np.ndarray, n_blocks: int, rng: np.random.Generator):
        '''
        Randomly but evenly (regarding y_labels) compute the indices splitting a set into n_blocks of equal size.
        '''
        classes = np.unique(y)  # Extract unique classes
        blocks = [[] for _ in range(n_blocks)] # Initialize a list where to store each block separately

        # Randomly but evenly divide each class instances into each block 
        for c in classes:
            # Extract the indices associated to class `c`
            cls_idx = np.where(y == c)[0]
            # Randomly shuffle those indices
            rng.shuffle(cls_idx)
            # Split those indices in a round-robin fashion (evenly distributed)
            for pos, idx in enumerate(cls_idx):
                blocks[pos % n_blocks].append(int(idx))

        # Randomly reorder the indices inside each block (so they are not ordered by the associated class)
        for block in blocks:
            rng.shuffle(block)

        return [np.asarray(block, dtype=int) for block in blocks]

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)

        n = X.shape[0]
        k = int(self.n_neighbors)
        b = int(self.n_blocks)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if b < 2:
            raise ValueError("n_blocks must be >= 2")
        if n < b:
            raise ValueError(f"Need n_samples >= n_blocks. Got n_samples={n}, n_blocks={b}.")
        if self.action not in {"remove", "relabel"}:
            raise ValueError("action must be 'remove' or 'relabel'")
        if self.max_iter is not None:
            max_iter = int(self.max_iter)
            if max_iter < 1:
                raise ValueError("max_iter must be >= 1 when provided")
        else:
            max_iter = None

        alive = np.ones(n, dtype=bool)  # Array indexing instances currently not removed
        nn_pred = np.empty(n, dtype=object)
        n_iters = 0
        rng = np.random.default_rng(self.random_state)

        while True:
            if max_iter is not None and n_iters >= max_iter:
                break
            # Check if remaining instances is lower than the number of partitions to make
            alive_idx = np.where(alive)[0]
            if alive_idx.size < b:
                break
            n_iters += 1
            # Extract remaning instances and randomly apply stratification
            X_a = X[alive_idx]
            y_a = y[alive_idx]
            # Randomly but evenly split remaining instances into a list of same size blocks
            blocks_rel = self._make_stratified_blocks(y_a, b, rng)

            # Proceed with the cleaning algorithm
            removed_this_iter = 0
            for i in range(b):
                # Extract train and test block indices
                test_rel = blocks_rel[i]
                train_rel = blocks_rel[(i + 1) % b]
                if test_rel.size == 0 or train_rel.size == 0:
                    continue
                # Extract instances associted to those indices
                train_idx = alive_idx[train_rel]
                test_idx = alive_idx[test_rel]
                # Predict labels on block R_i based on the model fitted with following block
                preds = self._predict_block(X[train_idx], y[train_idx], X[test_idx], k)
                # Store predictions and compute disagreement
                nn_pred[test_idx] = preds
                misclassified = (preds != y[test_idx])  # Mask with the disagreements for this iteration
                removed_this_iter += int(np.sum(misclassified)) # Number of instances to remove this iteration
                alive[test_idx[misclassified]] = False  # Set to 'remove' misclassified instances

            # End the loop when no new instance is removed
            if removed_this_iter == 0:
                break
        


        keep_mask = alive.copy()
        noisy_fraction = float((~keep_mask).mean())
        self.result_ = MultiEditFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=noisy_fraction,
            removed_total=int((~keep_mask).sum()),
            n_iters=int(n_iters),
            n_blocks=b,
            nn_pred=nn_pred,
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        if self.action == "remove":
            km = self.result_.keep_mask
            return self.X_[km], self.y_[km]

        y_new = np.asarray(self.y_).copy()
        noisy_idx = np.where(~self.result_.keep_mask)[0]
        y_new[noisy_idx] = self.result_.nn_pred[noisy_idx]
        return self.X_, y_new

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_neighbors_k": int(self.n_neighbors),
            "n_blocks": int(r.n_blocks),
            "n_iters": int(r.n_iters),
            "max_iter": None if self.max_iter is None else int(self.max_iter),
            "removed_or_flagged": int(r.removed_total),
            "fraction_flagged": float(r.noisy_fraction),
            "metric": self.metric,
            "p": int(self.p) if self.metric == "minkowski" else None,
            "action": self.action,
        }
