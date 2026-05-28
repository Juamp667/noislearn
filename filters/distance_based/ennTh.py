"""
Edited nearest-neighbor probabilistic filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y


@dataclass
class ENNProbFilterResult:
    keep_mask: np.ndarray
    noisy_fraction: float
    nn_pred: np.ndarray
    max_prob: np.ndarray
    class_probabilities: np.ndarray
    neighbor_count_used: np.ndarray
    kth_distance: np.ndarray


class ENNProbFilter(BaseEstimator):
    def __init__(self, n_neighbors: int = 3, mode: str = "prob", threshold: float = 0.5, metric: str = "minkowski", p: int = 2, tie_eps: float = 1e-12, action: str = "remove", n_jobs: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.threshold = threshold
        self.metric = metric
        self.p = p
        self.tie_eps = tie_eps
        self.action = action
        self.n_jobs = n_jobs

    @staticmethod
    def _weighted_class_probabilities(labels_1d: np.ndarray, dists_1d: np.ndarray, classes: np.ndarray):
        ''' 
        Compute class probabilites based on the nn labels, distances and train_set_unique_classes.
        '''
        # The weight of each nn is equal to the inverse of its associated distance
        weights = 1.0 / np.maximum(dists_1d, 1e-12)
        probs = np.zeros(len(classes), dtype=float)
        # Compute the probability associated to each class as the sum of the nn weights
        for i, c in enumerate(classes):
            probs[i] = float(np.sum(weights[labels_1d == c]))
        total = float(np.sum(probs))
        # Normalize the probabilities
        if total > 0.0:
            probs /= total
        return probs

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
        if self.mode not in {"prob", "th"}:
            raise ValueError("mode must be 'prob' or 'th'")
        if self.action != "remove":
            raise ValueError("action must be 'remove'")
        if not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        classes = np.unique(y)
        n_query = k + 1 # The "+1" is due to the fact that NearestNeighbors returns the reciprocal-instance-distance (which is zero)
        # Compute the nn to each instance (as well as the corresponding distances)
        nn = NearestNeighbors(n_neighbors=n_query, metric=self.metric, p=self.p if self.metric == "minkowski" else None, n_jobs=self.n_jobs)
        nn.fit(X)
        dists, idxs = nn.kneighbors(X, return_distance=True)

        nn_pred = np.empty(n, dtype=object)
        max_prob = np.empty(n, dtype=float)
        neighbor_count_used = np.empty(n, dtype=int)
        kth_distance = np.empty(n, dtype=float)
        class_probabilities = np.empty((n, len(classes)), dtype=float)
        
        # For each instance
        for i in range(n):
            # Extract the nn distances and indices
            di = dists[i][1:]
            ii = idxs[i][1:]
            # Extract the distance of the further neighbour
            d_k = di[k - 1]
            mask = (di <= (d_k + float(self.tie_eps)))

            # Compute the probability of belonging to each one of the available classes
            neigh_idx = ii[mask]
            neigh_dists = di[mask]
            neigh_labels = y[neigh_idx]
            probs = self._weighted_class_probabilities(neigh_labels, neigh_dists, classes)
            class_probabilities[i] = probs
            
            # Select the one with a higher probability and store both
            best = int(np.argmax(probs))
            nn_pred[i] = classes[best]
            max_prob[i] = float(probs[best])
            neighbor_count_used[i] = int(len(neigh_idx))
            kth_distance[i] = float(d_k)

        # Compute disagreement
        noisy_mask = (nn_pred != y)
        # If threshold mode correct noisy instances if their associated prob is no high enough
        if self.mode == "th":
            noisy_mask = noisy_mask | (max_prob < float(self.threshold))
        keep_mask = ~noisy_mask


        self.result_ = ENNProbFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), nn_pred=nn_pred, max_prob=max_prob, class_probabilities=class_probabilities, neighbor_count_used=neighbor_count_used, kth_distance=kth_distance)
        self.X_ = X
        self.y_ = y
        self.classes_ = classes
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        km = self.result_.keep_mask
        return self.X_[km], self.y_[km]

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        return {"n_samples": int(self.X_.shape[0]), "mode": self.mode, "n_neighbors_k": int(self.n_neighbors), "threshold": float(self.threshold) if self.mode == "th" else None, "avg_neighbors_used": float(np.mean(r.neighbor_count_used)), "max_neighbors_used": int(np.max(r.neighbor_count_used)), "removed_or_flagged": int((~r.keep_mask).sum()), "fraction_flagged": float(r.noisy_fraction), "metric": self.metric, "p": int(self.p) if self.metric == "minkowski" else None, "tie_eps": float(self.tie_eps), "action": self.action}


# Backwards-friendly names.
ENNProb = ENNProbFilter
ENNTh = ENNProbFilter
