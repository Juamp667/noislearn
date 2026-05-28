"""
Nearest Centroid Neighbor Edition (NCNEdit).
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y


@dataclass
class NCNEditFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    ncn_pred: np.ndarray
    disagree_count: np.ndarray
    neighbor_count_used: np.ndarray


class NCNEdit(BaseEstimator):
    ''' 
    Similar to ENN, but selecting the neighborhood recursively by storing the points that minimize the 
    distance between the sample to filter and the centroid of its neighborhood (which does not consider 
    the point to filter).
    '''
    def __init__(self, n_neighbors: int = 3, metric: str = "minkowski", p: int = 2, action: str = "remove", n_jobs: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.action = action
        self.n_jobs = n_jobs

    @staticmethod
    def _majority_vote(labels_1d: np.ndarray):
        vals, cnts = np.unique(labels_1d, return_counts=True)
        return vals[np.argmax(cnts)]

    @staticmethod
    def _centroid(points: np.ndarray):
        ''' 
        Compute the centroid between `points`.
        '''
        return np.mean(points, axis=0)

    def _select_ncn_neighbors(self, x_i: np.ndarray, candidate_idxs: np.ndarray, X: np.ndarray, k: int):
        ''' 
        Compute the `k` closest neighbors to `x_i` using the iterative centroid approach.
        '''
        # Initilize list with indices of current neighbors
        selected = [int(candidate_idxs[0])]
        # Initilize list with remaining neighbors
        remaining = list(map(int, candidate_idxs[1:]))

        while len(selected) < k and remaining:
            best_idx = None
            best_dist = None
            # Compute the closest point to the current centroid
            for cand in remaining:
                # Compute the centroid with `selected`+[candidate] points
                trial_idxs = selected + [cand]
                centroid = self._centroid(X[trial_idxs])
                # Compute distance from `x_i` to the latter centroid
                dist = float(np.linalg.norm(x_i - centroid))
                # Update dist if improves and store best_idx
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = cand
            # Add to the neighborhood the best candidate (and rmv it from remaining) 
            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.asarray(selected, dtype=int)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)

        n = X.shape[0]
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors. Got n_samples={n}, n_neighbors={k}.")
        if self.action not in {"remove", "relabel"}:
            raise ValueError("action must be 'remove' or 'relabel'")

        # Compute the closest instances to each and every one of them
        nn = NearestNeighbors(
            n_neighbors=n,
            metric=self.metric,
            p=self.p if self.metric == "minkowski" else None,
            n_jobs=self.n_jobs,
        )
        nn.fit(X)
        idxs = nn.kneighbors(X, return_distance=False)


        ncn_pred = np.empty(n, dtype=object)
        disagree_count = np.empty(n, dtype=int)
        neighbor_count_used = np.empty(n, dtype=int)

        # For each instance
        for i in range(n):
            # Extract candidate closest points (in order)
            candidate_idxs = idxs[i][1:]
            # Compute neighborhood indices using closest centroid approach
            neigh_idx = self._select_ncn_neighbors(X[i], candidate_idxs, X, k)
            neigh_labels = y[neigh_idx]
            # Compute predicted label considering the neighborhood
            ncn_pred[i] = self._majority_vote(neigh_labels)
            # Store amount of disagreements
            disagree_count[i] = int(np.sum(neigh_labels != y[i]))
            neighbor_count_used[i] = int(len(neigh_idx))

        # Compute the masks with filtered samples  
        noisy_mask = (ncn_pred != y)
        keep_mask = ~noisy_mask

        self.result_ = NCNEditFilterResult(
            keep_mask=keep_mask,
            noisy_fraction=float(noisy_mask.mean()),
            ncn_pred=ncn_pred,
            disagree_count=disagree_count,
            neighbor_count_used=neighbor_count_used,
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
        y_new[noisy_idx] = self.result_.ncn_pred[noisy_idx]
        return self.X_, y_new

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_neighbors_k": int(self.n_neighbors),
            "avg_neighbors_used": float(np.mean(r.neighbor_count_used)),
            "max_neighbors_used": int(np.max(r.neighbor_count_used)),
            "removed_or_flagged": int((~r.keep_mask).sum()),
            "fraction_flagged": float(r.noisy_fraction),
            "metric": self.metric,
            "p": int(self.p) if self.metric == "minkowski" else None,
            "action": self.action,
        }
