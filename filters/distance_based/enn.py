"""
Edited nearest-neighbor filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y

from .._detection import attach_detection_report, resample_by_action, validate_action


@dataclass
class ENNFilterResult:
    """Summary of an edited nearest-neighbor filtering run.

    Attributes
    ----------
    keep_mask : ndarray of bool
        Mask indicating which samples are kept after filtering.
    noisy_fraction : float
        Fraction of samples flagged as noisy.
    nn_pred : ndarray
        Majority-vote label predicted from the selected neighbors.
    disagree_count : ndarray
        Number of neighbors whose label differs from the observed label.
    neighbor_count_used : ndarray
        Number of neighbors actually used for each sample.
    kth_distance : ndarray
        Distance to the kth neighbor used to define the neighborhood.
    """

    keep_mask: np.ndarray
    noisy_fraction: float
    nn_pred: np.ndarray
    disagree_count: np.ndarray
    neighbor_count_used: np.ndarray
    kth_distance: np.ndarray


class ENNFilter(BaseEstimator):
    """Edited nearest-neighbor noise filter.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors used to vote on each sample.
    mode : {"enn", "menn"}, default="enn"
        ``"enn"`` uses the fixed k nearest neighbors.
        ``"menn"`` expands the candidate neighborhood until predictions stabilize.
    metric : str, default="minkowski"
        Distance metric used by :class:`sklearn.neighbors.NearestNeighbors`.
    p : int, default=2
        Minkowski power parameter, only used when ``metric="minkowski"``.
    tie_eps : float, default=1e-12
        Tolerance used to include neighbors tied at the kth distance.
    action : {"remove", "detect"}, default="remove"
        Whether to drop noisy samples or only detect them.
    n_jobs : int or None, default=None
        Parallelism forwarded to the nearest-neighbor search.
    candidate_strategy : {"full", "expansive"}, default="expansive"
        Strategy used to grow the candidate set in ``"menn"`` mode.

    Notes
    -----
    A sample is flagged as noisy when the neighborhood vote differs from the observed label.
    """

    def __init__(self, n_neighbors: int = 3, mode: str = "enn", metric: str = "minkowski", p: int = 2, tie_eps: float = 1e-12, action: str = "remove", n_jobs: Optional[int] = None, candidate_strategy: str = "expansive"):
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self.p = p
        self.tie_eps = tie_eps
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

    def fit(self, X, y):
        """Fit the filter and cache nearest-neighbor predictions."""

        X, y = check_X_y(X, y, accept_sparse=True)
        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors. Got n_samples={n}, n_neighbors={k}.")
        if self.mode not in {"enn", "menn"}:
            raise ValueError("mode must be 'enn' or 'menn'")
        if self.candidate_strategy not in {"full", "expansive"}:
            raise ValueError("candidate_strategy must be 'full' or 'expansive'")
        validate_action(self.action)

        nn = NearestNeighbors(n_neighbors=n, metric=self.metric, p=self.p if self.metric == "minkowski" else None, n_jobs=self.n_jobs)
        nn.fit(X)
        nn_pred = np.empty(n, dtype=object)
        disagree_count = np.empty(n, dtype=int)
        neighbor_count_used = np.empty(n, dtype=int)
        kth_distance = np.empty(n, dtype=float)
        neighborhood_labels_used = np.empty(n, dtype=object)
        neighborhood_alternative_majority_fraction = np.empty(n, dtype=float)

        expansive = self.mode == "menn" and self.candidate_strategy == "expansive"
        candidate_sizes = self._expansive_candidate_sizes(n, k) if expansive else None

        if expansive:
            prev_pred = None
            for candidate_count in candidate_sizes:
                dists, idxs = nn.kneighbors(X, n_neighbors=candidate_count + 1, return_distance=True)
                current_pred = np.empty(n, dtype=object)
                current_disagree = np.empty(n, dtype=int)
                current_neighbor_count_used = np.empty(n, dtype=int)
                current_kth_distance = np.empty(n, dtype=float)
                current_neighborhood_labels_used = np.empty(n, dtype=object)
                current_neighborhood_alternative_majority_fraction = np.empty(n, dtype=float)

                for i in range(n):
                    di = dists[i][1:]
                    ii = idxs[i][1:]
                    d_k = di[k - 1]
                    mask = (di <= (d_k + float(self.tie_eps)))
                    neigh_idx = ii[mask]
                    neigh_labels = y[neigh_idx]
                    current_pred[i] = self._majority_vote(neigh_labels)
                    current_disagree[i] = int(np.sum(neigh_labels != y[i]))
                    current_neighbor_count_used[i] = int(len(neigh_idx))
                    current_kth_distance[i] = float(d_k)
                    current_neighborhood_labels_used[i] = np.asarray(neigh_labels, dtype=object)
                    current_neighborhood_alternative_majority_fraction[i] = self._alternative_majority_fraction(neigh_labels, y[i])

                if prev_pred is not None and np.array_equal(current_pred, prev_pred):
                    nn_pred = current_pred
                    disagree_count = current_disagree
                    neighbor_count_used = current_neighbor_count_used
                    kth_distance = current_kth_distance
                    neighborhood_labels_used = current_neighborhood_labels_used
                    neighborhood_alternative_majority_fraction = current_neighborhood_alternative_majority_fraction
                    break

                prev_pred = current_pred
                nn_pred = current_pred
                disagree_count = current_disagree
                neighbor_count_used = current_neighbor_count_used
                kth_distance = current_kth_distance
                neighborhood_labels_used = current_neighborhood_labels_used
                neighborhood_alternative_majority_fraction = current_neighborhood_alternative_majority_fraction
        else:
            n_query = k + 1 if self.mode == "enn" else n
            dists, idxs = nn.kneighbors(X, n_neighbors=n_query, return_distance=True)

            for i in range(n):
                di = dists[i][1:]
                ii = idxs[i][1:]
                if self.mode == "enn":
                    neigh_idx = ii
                else:
                    d_k = di[k - 1]
                    mask = (di <= (d_k + float(self.tie_eps)))
                    neigh_idx = ii[mask]
                neigh_labels = y[neigh_idx]
                nn_pred[i] = self._majority_vote(neigh_labels)
                disagree_count[i] = int(np.sum(neigh_labels != y[i]))
                neighbor_count_used[i] = int(len(neigh_idx))
                kth_distance[i] = float(di[k - 1])
                neighborhood_labels_used[i] = np.asarray(neigh_labels, dtype=object)
                neighborhood_alternative_majority_fraction[i] = self._alternative_majority_fraction(neigh_labels, y[i])

        noisy_mask = (nn_pred != y)
        keep_mask = ~noisy_mask

        noise_score = (disagree_count / np.maximum(neighbor_count_used, 1)) * np.sqrt(neighborhood_alternative_majority_fraction)
        self.result_ = ENNFilterResult(keep_mask=keep_mask, noisy_fraction=float(noisy_mask.mean()), nn_pred=nn_pred, disagree_count=disagree_count, neighbor_count_used=neighbor_count_used, kth_distance=kth_distance)
        self.neighborhood_labels_used_ = neighborhood_labels_used
        self.neighborhood_alternative_majority_fraction_ = neighborhood_alternative_majority_fraction
        attach_detection_report(
            self,
            noisy_mask,
            noise_score=noise_score,
            observed_labels=y,
            predicted_labels=nn_pred,
            disagree_count=disagree_count,
            neighbor_count_used=neighbor_count_used,
            kth_distance=kth_distance,
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
        return {"n_samples": int(self.X_.shape[0]), "mode": self.mode, "candidate_strategy": self.candidate_strategy, "n_neighbors_k": int(self.n_neighbors), "avg_neighbors_used": float(np.mean(r.neighbor_count_used)), "max_neighbors_used": int(np.max(r.neighbor_count_used)), "removed_or_flagged": int((~r.keep_mask).sum()), "fraction_flagged": float(r.noisy_fraction), "metric": self.metric, "p": int(self.p) if self.metric == "minkowski" else None, "tie_eps": float(self.tie_eps), "action": self.action}

    def get_detection_report(self):
        """Return the stored detection report."""

        return dict(self.detection_report_)
