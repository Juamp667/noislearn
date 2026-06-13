"""
Nearest Centroid Neighbor Edition (NCNEdit).
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y

from .._detection import attach_detection_report, resample_by_action, validate_action


@dataclass
class NCNEditFilterResult:
    """Summary of an NCNEdit filtering run.

    Attributes
    ----------
    keep_mask : ndarray of bool
        Mask indicating which samples are kept after filtering.
    noisy_fraction : float
        Fraction of samples flagged as noisy.
    ncn_pred : ndarray
        Prediction obtained from the centroid-based neighborhood vote.
    disagree_count : ndarray
        Number of neighbors whose label differs from the observed label.
    neighbor_count_used : ndarray
        Number of neighbors actually used for each sample.
    """

    keep_mask: np.ndarray
    noisy_fraction: float
    ncn_pred: np.ndarray
    disagree_count: np.ndarray
    neighbor_count_used: np.ndarray


class NCNEdit(BaseEstimator):
    """Nearest-centroid-neighbor noise filter.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors used to build the centroid-based neighborhood.
    metric : str, default="minkowski"
        Distance metric used by :class:`sklearn.neighbors.NearestNeighbors`.
    p : int, default=2
        Minkowski power parameter, only used when ``metric="minkowski"``.
    action : {"remove", "detect"}, default="remove"
        Whether to drop noisy samples or only detect them.
    n_jobs : int or None, default=None
        Parallelism forwarded to the nearest-neighbor search.
    candidate_strategy : {"full", "expansive"}, default="expansive"
        Strategy used to grow the candidate set while selecting the centroid-nearest neighbors.

    Notes
    -----
    The neighborhood is built recursively by adding the candidate that minimizes the distance
    between the sample and the centroid of the partial neighborhood.
    """
    def __init__(self, n_neighbors: int = 3, metric: str = "minkowski", p: int = 2, action: str = "remove", n_jobs: Optional[int] = None, candidate_strategy: str = "expansive"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.action = action
        self.n_jobs = n_jobs
        self.candidate_strategy = candidate_strategy

    @staticmethod
    def _majority_vote(labels_1d: np.ndarray):
        vals, cnts = np.unique(labels_1d, return_counts=True)
        return vals[np.argmax(cnts)]

    @staticmethod
    def _alternative_majority_fraction(labels_1d: np.ndarray, self_label):
        if labels_1d.size == 0:
            return 0.0
        alt_labels = labels_1d[labels_1d != self_label]
        if alt_labels.size == 0:
            return 0.0
        _, cnts = np.unique(alt_labels, return_counts=True)
        return float(np.max(cnts) / float(labels_1d.size))

    @staticmethod
    def _centroid(points: np.ndarray):
        ''' 
        Compute the centroid between `points`.
        '''
        return np.mean(points, axis=0)

    @staticmethod
    def _expansive_candidate_sizes(n: int, k: int):
        max_candidates = n - 1
        sizes = []
        mult = 2
        while True:
            candidate_count = min(max_candidates, mult * k)
            if not sizes or candidate_count != sizes[-1]:
                sizes.append(candidate_count)
            if candidate_count >= max_candidates:
                break
            mult += 1
        return sizes

    def _select_ncn_neighbors(self, x_i: np.ndarray, candidate_idxs: np.ndarray, X: np.ndarray, k: int):
        ''' 
        Compute the `k` closest neighbors to `x_i` using the iterative centroid approach.
        '''
        # Initialize the neighborhood with the closest candidate and keep the running sum
        selected = [int(candidate_idxs[0])]
        selected_sum = X[selected[0]].astype(float, copy=True)
        remaining = np.asarray(candidate_idxs[1:], dtype=int)

        while len(selected) < k and remaining.size > 0:
            # Evaluate all candidate centroids at once.
            trial_centroids = (selected_sum + X[remaining]) / float(len(selected) + 1)
            dists = np.linalg.norm(trial_centroids - x_i, axis=1)
            best_pos = int(np.argmin(dists))
            best_idx = int(remaining[best_pos])

            selected.append(best_idx)
            selected_sum += X[best_idx]
            remaining = np.delete(remaining, best_pos)

        return np.asarray(selected, dtype=int)

    def fit(self, X, y):
        """Fit the filter and cache the NCN-based predictions."""

        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)

        n = X.shape[0]
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors. Got n_samples={n}, n_neighbors={k}.")
        if self.candidate_strategy not in {"full", "expansive"}:
            raise ValueError("candidate_strategy must be 'full' or 'expansive'")
        validate_action(self.action)

        # Compute the closest instances to each and every one of them once.
        nn = NearestNeighbors(
            n_neighbors=n,
            metric=self.metric,
            p=self.p if self.metric == "minkowski" else None,
            n_jobs=self.n_jobs,
        )
        nn.fit(X)
        idxs = nn.kneighbors(X, n_neighbors=n, return_distance=False)


        ncn_pred = np.empty(n, dtype=object)
        disagree_count = np.empty(n, dtype=int)
        neighbor_count_used = np.empty(n, dtype=int)
        neighborhood_labels_used = np.empty(n, dtype=object)
        neighborhood_alternative_majority_fraction = np.empty(n, dtype=float)

        expansive = self.candidate_strategy == "expansive"
        candidate_sizes = self._expansive_candidate_sizes(n, k) if expansive else None

        if expansive:
            prev_pred = None
            for candidate_count in candidate_sizes:
                current_pred = np.empty(n, dtype=object)
                current_disagree = np.empty(n, dtype=int)
                current_neighbor_count_used = np.empty(n, dtype=int)
                current_neighborhood_labels_used = np.empty(n, dtype=object)
                current_neighborhood_alternative_majority_fraction = np.empty(n, dtype=float)

                # For each instance
                for i in range(n):
                    # Extract candidate closest points (in order)
                    candidate_idxs = idxs[i][1:candidate_count + 1]
                    # Compute neighborhood indices using closest centroid approach
                    neigh_idx = self._select_ncn_neighbors(X[i], candidate_idxs, X, k)
                    neigh_labels = y[neigh_idx]
                    # Compute predicted label considering the neighborhood
                    current_pred[i] = self._majority_vote(neigh_labels)
                    # Store amount of disagreements
                    current_disagree[i] = int(np.sum(neigh_labels != y[i]))
                    current_neighbor_count_used[i] = int(len(neigh_idx))
                    current_neighborhood_labels_used[i] = np.asarray(neigh_labels, dtype=object)
                    current_neighborhood_alternative_majority_fraction[i] = self._alternative_majority_fraction(neigh_labels, y[i])

                ncn_pred = current_pred
                disagree_count = current_disagree
                neighbor_count_used = current_neighbor_count_used
                neighborhood_labels_used = current_neighborhood_labels_used
                neighborhood_alternative_majority_fraction = current_neighborhood_alternative_majority_fraction

                if prev_pred is not None and np.array_equal(current_pred, prev_pred):
                    break

                prev_pred = current_pred
        else:
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
                neighborhood_labels_used[i] = np.asarray(neigh_labels, dtype=object)
                neighborhood_alternative_majority_fraction[i] = self._alternative_majority_fraction(neigh_labels, y[i])

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
        self.neighborhood_labels_used_ = neighborhood_labels_used
        self.neighborhood_alternative_majority_fraction_ = neighborhood_alternative_majority_fraction
        noise_score = (disagree_count / np.maximum(neighbor_count_used, 1)) * np.sqrt(neighborhood_alternative_majority_fraction)
        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=ncn_pred,
            disagree_count=disagree_count,
            neighbor_count_used=neighbor_count_used,
            neighborhood_labels_used=neighborhood_labels_used,
            neighborhood_alternative_majority_fraction=neighborhood_alternative_majority_fraction,
        )
        self.X_ = X
        self.y_ = y
        return self

    def fit_resample(self, X, y):
        """Fit the filter and return the filtered or detected data."""

        self.fit(X, y)
        return resample_by_action(self.X_, self.y_, self.action, self.result_.keep_mask)

    def get_filter_report(self) -> Dict[str, Any]:
        """Return a dictionary with the main fit diagnostics."""

        r = self.result_
        return {
            "n_samples": int(self.X_.shape[0]),
            "candidate_strategy": self.candidate_strategy,
            "n_neighbors_k": int(self.n_neighbors),
            "avg_neighbors_used": float(np.mean(r.neighbor_count_used)),
            "max_neighbors_used": int(np.max(r.neighbor_count_used)),
            "removed_or_flagged": int((~r.keep_mask).sum()),
            "fraction_flagged": float(r.noisy_fraction),
            "metric": self.metric,
            "p": int(self.p) if self.metric == "minkowski" else None,
            "action": self.action,
        }

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
