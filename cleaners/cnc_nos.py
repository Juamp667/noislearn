"""
Class Noise Cleaner with Noise Scoring (CNC-NOS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils.validation import check_X_y

from filters.classifier_based import CVCFFilter, ClassificationFilter
from filters.distance_based import ENNProbFilter, MultiEditFilter, NCNEdit


@dataclass
class _BaseFilterOutput:
    name: str
    noisy_mask: np.ndarray
    predicted_labels: np.ndarray


@dataclass
class CNCNOSIterationInfo:
    iter_idx: int
    phase: str
    n_samples_before: int
    n_filters: int
    n_candidates: int
    n_kept: int
    n_relabelled: int
    n_removed: int
    mean_wns: float
    wns_threshold: float
    stagnation_counter: int


@dataclass
class CNCNOSCleanerResult:
    keep_mask: np.ndarray
    relabel_mask: np.ndarray
    remove_mask: np.ndarray
    noisy_fraction: float
    changed_fraction: float
    initial_candidate_fraction: float
    final_filtering_applied: bool
    n_iters: int
    history: List[CNCNOSIterationInfo]
    base_filter_names: Tuple[str, ...]
    candidate_mask: np.ndarray
    wns: np.ndarray


class CNCNOSCleaner(BaseEstimator):
    """CNC-NOS cleaner using the filters available in this project.

    Parameters
    ----------
    base_filters:
        Optional sequence of estimators or ``(name, estimator)`` pairs.
        When omitted, a heterogeneous default ensemble is built from the
        filters shipped with this repository.
    """

    def __init__(
        self,
        base_filters: Optional[Sequence[Any]] = None,
        base_neighbors: int = 3,
        score_neighbors: int = 5,
        cv: int = 10,
        metric: str = "minkowski",
        p: int = 2,
        max_iter: int = 10,
        stagnation_patience: int = 2,
        wns_tol: float = 1e-4,
        final_filtering: bool = True,
        final_filtering_min_fraction: float = 0.2,
        min_class_count: int = 2,
        random_state: int = 33,
        n_jobs: Optional[int] = None,
    ):
        self.base_filters = base_filters
        self.base_neighbors = base_neighbors
        self.score_neighbors = score_neighbors  # Number of neighbours used to compute the wNS
        self.cv = cv
        self.metric = metric
        self.p = p
        self.max_iter = max_iter
        self.stagnation_patience = stagnation_patience
        self.wns_tol = wns_tol
        self.final_filtering = final_filtering
        self.final_filtering_min_fraction = final_filtering_min_fraction
        self.min_class_count = min_class_count
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _as_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)

    @staticmethod
    def _majority_vote(labels: np.ndarray, fallback: Optional[Any] = None):
        labels = np.asarray(labels, dtype=object)
        values, counts = np.unique(labels, return_counts=True)
        if values.size == 1:
            return values[0], int(counts[0])

        best_idx = int(np.argmax(counts))
        best_count = int(counts[best_idx])
        best_label = values[best_idx]

        # A strict majority is required. If there is no majority, keep the
        # fallback label when provided and otherwise report "no consensus".
        if best_count > (labels.size / 2.0):
            return best_label, best_count
        if fallback is not None:
            return fallback, int(np.sum(labels == fallback))
        return None, best_count

    def _effective_cv(self, y: np.ndarray) -> Optional[int]:
        if self.cv is None:
            return None
        cv = int(self.cv)
        if cv < 2:
            raise ValueError("cv must be >= 2 when provided")
        _, counts = np.unique(y, return_counts=True)
        if counts.size == 0:
            return None
        min_count = int(np.min(counts))
        if min_count < 2:
            return None
        return max(2, min(cv, int(y.shape[0]), min_count))

    def _effective_k(self, n_samples: int) -> int:
        k = int(self.base_neighbors)
        if k < 1:
            raise ValueError("base_neighbors must be >= 1")
        return max(1, min(k, n_samples - 1))

    def _effective_score_k(self, n_samples: int) -> int:
        k = int(self.score_neighbors)
        if k < 1:
            raise ValueError("score_neighbors must be >= 1")
        return max(1, min(k, n_samples - 1))

    def _build_default_base_filters(self, X: np.ndarray, y: np.ndarray, iteration_idx: int):
        """Build the default heterogeneous ensemble.

        The original paper uses RNG, NCNEdit, CVCF, EF and IPF. We approximate
        that diversity using the filters available in this repository.
        """

        n_samples = int(X.shape[0])
        if n_samples < 2:
            return []

        k_eff = self._effective_k(n_samples)
        blocks_eff = max(2, min(10, n_samples))
        cv_eff = self._effective_cv(y)
        iter_rs = int(self.random_state) + int(iteration_idx)

        filters: List[Tuple[str, BaseEstimator]] = [
            (
                "NCNEdit",
                NCNEdit(
                    n_neighbors=k_eff,
                    metric=self.metric,
                    p=self.p,
                    action="remove",
                    n_jobs=self.n_jobs,
                    candidate_strategy="expansive",
                ),
            ),
            (
                "ENNProb",
                ENNProbFilter(
                    n_neighbors=k_eff,
                    mode="th",
                    threshold=0.5,
                    metric=self.metric,
                    p=self.p,
                    action="remove",
                    n_jobs=self.n_jobs,
                ),
            ),
            (
                "MultiEdit",
                MultiEditFilter(
                    n_neighbors=k_eff,
                    n_blocks=blocks_eff,
                    metric=self.metric,
                    p=self.p,
                    action="remove",
                    random_state=iter_rs,
                    n_jobs=self.n_jobs,
                ),
            ),
        ]

        if cv_eff is not None:
            knn = KNeighborsClassifier(
                n_neighbors=1,
                metric=self.metric,
                p=self.p,
                n_jobs=self.n_jobs,
            )
            filters.extend(
                [
                    (
                        "KNN1CV",
                        ClassificationFilter(
                            estimator=knn,
                            cv=cv_eff,
                            action="relabel",
                            random_state=iter_rs,
                        ),
                    ),
                    (
                        "CVCF",
                        CVCFFilter(
                            cv=cv_eff,
                            vote_rule="consensus",
                            action="remove",
                            random_state=iter_rs,
                        ),
                    ),
                ]
            )

        return filters

    def _resolve_base_filters(self, X: np.ndarray, y: np.ndarray, iteration_idx: int):
        # If no filters are introduced by the user use the default ones
        if self.base_filters is None:
            return self._build_default_base_filters(X, y, iteration_idx)

        resolved = []
        for idx, item in enumerate(self.base_filters):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, estimator = item
            else:
                name, estimator = type(item).__name__, item
            resolved.append((str(name), clone(estimator)))
        return resolved

    def _extract_output(self, fitted_filter: BaseEstimator, X: np.ndarray, y: np.ndarray) -> _BaseFilterOutput:
        # Extract `result_` atribute asociated to a previously fitted filter
        result = getattr(fitted_filter, "result_", None)

        # Extract instances labeled as noisy by the previously fitted filter
        if result is not None and hasattr(result, "keep_mask"):
            keep_mask = np.asarray(result.keep_mask, dtype=bool)
            noisy_mask = ~keep_mask
        elif hasattr(fitted_filter, "keep_mask"):
            keep_mask = np.asarray(fitted_filter.keep_mask, dtype=bool)
            noisy_mask = ~keep_mask
        elif hasattr(fitted_filter, "sample_indices_"):
            keep_mask = np.zeros(X.shape[0], dtype=bool)
            keep_mask[np.asarray(fitted_filter.sample_indices_, dtype=int)] = True
            noisy_mask = ~keep_mask
        else:
            noisy_mask = np.zeros(X.shape[0], dtype=bool)

        # Extract the labels predicted by the previously fitted filter if 
        # they were somehow saved in `fitted_filter`
        predicted_labels = None
        if result is not None:
            for attr in ("oof_pred", "nn_pred", "ncn_pred", "predicted_labels_", "predictions_", "labels_"):
                if hasattr(result, attr):
                    predicted_labels = np.asarray(getattr(result, attr), dtype=object)
                    break

            # If the filter is classifier ensemble based compute the predicted label with a majority vote
            if predicted_labels is None and hasattr(result, "fold_preds"):
                fold_preds = np.asarray(result.fold_preds, dtype=object)
                predicted_labels = np.empty(fold_preds.shape[1], dtype=object)
                for i in range(fold_preds.shape[1]):
                    pred, _ = self._majority_vote(fold_preds[:, i], fallback=y[i])
                    predicted_labels[i] = y[i] if pred is None else pred

        # If `predicted_labels` couldn't be retrieved before
        # estimate them with the preffited filter
        if predicted_labels is None and hasattr(fitted_filter, "predict"):
            try:
                predicted_labels = np.asarray(fitted_filter.predict(X), dtype=object)
            except Exception:
                predicted_labels = np.asarray(y, dtype=object).copy()

        if predicted_labels is None:
            raise ValueError(f"No predicted labels could be retrieved from {type(fitted_filter).__name__}.") 
            predicted_labels = np.asarray(y, dtype=object).copy()

        if predicted_labels.shape[0] != X.shape[0]:
            raise ValueError("Base filter predictions do not match the current training fold size.")

        # About `fitted_filter`, return:
        #   - Name
        #   - Instances flagged as noisy
        #   - Predicted classes for instances flagged as noisy
        return _BaseFilterOutput(
            name=type(fitted_filter).__name__,
            noisy_mask=noisy_mask,
            predicted_labels=predicted_labels,
        )

    def _fit_base_filters(self, X: np.ndarray, y: np.ndarray, iteration_idx: int):
        # Initilize list with data to be returned
        outputs: List[_BaseFilterOutput] = []
        # Retrieve filters to apply
        specs = self._resolve_base_filters(X, y, iteration_idx)
        # Apply each filter separately
        for _, estimator in specs:
            fitted = clone(estimator)
            fitted.fit(X, y)
            outputs.append(self._extract_output(fitted, X, y))
        
        # Return results after applying each filter
        return outputs, tuple(name for name, _ in specs)

    def _compute_wns(self, X: np.ndarray, y: np.ndarray, candidate_mask: np.ndarray):
        n_samples = int(X.shape[0])
        if n_samples < 2:
            return np.zeros(n_samples, dtype=float)


        k_eff = self._effective_score_k(n_samples)  # Retrieve the number of neighbours to consider 
        y_arr = np.asarray(y)
        # Compute `k_eff + 1` nn to each instance (awa the dist to it)
        nn = NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric=self.metric,
            p=self.p if self.metric == "minkowski" else self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(X)
        dists, idxs = nn.kneighbors(X, return_distance=True)
        
        # Extract just nn info (the first index is reciprocal) 
        neigh_idx = idxs[:, 1:] 
        neigh_dist = dists[:, 1:]

        # -----------------------------
        # Compute confidence(e_i)
        # -----------------------------
        appearances = np.zeros(n_samples, dtype=float)  # Array to store the number of times an instance is a neighbour of one flagged as noisy
        candidate_nns = neigh_idx[candidate_mask]   # Extract the neighbours of candidate instances
        if np.any(candidate_mask):
            # Add one for each time an instance is neighbour of a candidate one
            np.add.at(
                appearances, 
                candidate_nns.ravel(),
                1.0
            )
        confidence = 1.0 / (1.0 + np.square(appearances))


        # -----------------------------
        # Compute clean(e_i)
        # -----------------------------
        total_dist_sum = np.sum(neigh_dist, axis=1)

        if np.any(total_dist_sum <= 0.0):
            bad_idx = np.flatnonzero(total_dist_sum <= 0.0)
            raise ValueError(
                "No se puede calcular clean(e_i): hay instancias cuya suma "
                f"de distancias a sus vecinos es cero. Índices afectados: {bad_idx[:10]}"
            )

        noisy_neighbor_mask = candidate_mask[neigh_idx]
        noisy_dist_sum = np.sum(neigh_dist * noisy_neighbor_mask, axis=1)

        safe_total_dist_sum = total_dist_sum.copy()
        safe_total_dist_sum[safe_total_dist_sum <= 0.0] = 1.0

        is_noise = np.where(candidate_mask, 1.0, -1.0)

        clean = (
            total_dist_sum + is_noise * (noisy_dist_sum - total_dist_sum)
        ) / (2.0 * safe_total_dist_sum)


        # -----------------------------
        # Compute neighborhood'(e_i)
        # -----------------------------
        neighborhood = 1/k_eff*np.sum(clean[neigh_idx]*confidence[neigh_idx]*(-1)**(y[neigh_idx]==y[:,None]), axis=1)

        wns = confidence * neighborhood
        return np.clip(wns.astype(float), -1.0, 1.0)

    def _protected_mask(self, y: np.ndarray):
        if int(self.min_class_count) <= 1:
            return np.zeros(y.shape[0], dtype=bool)

        values, counts = np.unique(y, return_counts=True)
        count_map = {val: int(cnt) for val, cnt in zip(values, counts)}
        return np.array([count_map[label] <= int(self.min_class_count) for label in y], dtype=bool)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        X = self._as_dense(X)
        y = np.asarray(y)

        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if int(self.stagnation_patience) < 1:
            raise ValueError("stagnation_patience must be >= 1")
        if int(self.base_neighbors) < 1:
            raise ValueError("base_neighbors must be >= 1")
        if int(self.score_neighbors) < 1:
            raise ValueError("score_neighbors must be >= 1")
        if int(self.min_class_count) < 1:
            raise ValueError("min_class_count must be >= 1")
        if not (0.0 <= float(self.final_filtering_min_fraction) <= 1.0):
            raise ValueError("final_filtering_min_fraction must be in [0, 1]")

        n0 = int(X.shape[0])    # Number of instances
        classes, counts = np.unique(y, return_counts=True)  # Number of classes and associated counts
        if n0 < 2 or classes.size < 2:
            keep_mask = np.ones(n0, dtype=bool)
            relabel_mask = np.zeros(n0, dtype=bool)
            remove_mask = np.zeros(n0, dtype=bool)
            self.result_ = CNCNOSCleanerResult(
                keep_mask=keep_mask,
                relabel_mask=relabel_mask,
                remove_mask=remove_mask,
                noisy_fraction=0.0,
                changed_fraction=0.0,
                initial_candidate_fraction=0.0,
                final_filtering_applied=False,
                n_iters=0,
                history=[],
                base_filter_names=tuple(),
                candidate_mask=np.zeros(n0, dtype=bool),
                wns=np.zeros(n0, dtype=float),
            )
            self.X_ = X
            self.y_ = y
            self.classes_ = classes
            self.y_clean_ = y.copy()
            self.keep_mask_ = keep_mask
            self.relabel_mask_ = relabel_mask
            self.remove_mask_ = remove_mask
            self.clean_indices_ = np.arange(n0, dtype=int)
            self.base_filter_names_ = tuple()
            return self

        # Initialize remaining instances as all the initially available 
        X_curr = X.copy()
        y_curr = y.copy()
        current_indices = np.arange(n0, dtype=int)
        y_clean_full = y.copy()

        # Initialize a list with the history for each iteration of the algorithm
        history: List[CNCNOSIterationInfo] = []
        prev_mean_wns: Optional[float] = None
        stagnation_counter = 0
        initial_candidate_fraction = 0.0
        last_candidate_mask = np.zeros(X_curr.shape[0], dtype=bool)
        last_wns = np.zeros(X_curr.shape[0], dtype=float)
        base_filter_names: Tuple[str, ...] = tuple()
        final_filtering_applied = False

        # Iterate the algorithm up to `max_iter` times
        for iter_idx in range(1, int(self.max_iter) + 1):
            # Check there are more than 2 instances remaining
            if X_curr.shape[0] < 2:
                break
            
            # Apply each filter to the current training set and retrieve their individual
            # predictions (instances flagged as noisy and the predicted class)
            outputs, base_filter_names = self._fit_base_filters(X_curr, y_curr, iter_idx)
            if not outputs:
                break
            
            
            noisy_votes = np.sum([out.noisy_mask for out in outputs], axis=0)   # Number of noisy votes for each filter
            candidate_threshold = (len(outputs) // 2) + 1   # Minimum number of filters in agree for majority vote
            # Compute noisy instances as just those flagged like that by at least half of the filters
            candidate_mask = (noisy_votes >= candidate_threshold)  
            candidate_count = int(np.sum(candidate_mask))   # Number of noisy candidates
            if candidate_count == 0:
                last_candidate_mask = candidate_mask
                last_wns = np.zeros(X_curr.shape[0], dtype=float)
                break

            if iter_idx == 1:
                initial_candidate_fraction = float(candidate_count / max(X_curr.shape[0], 1))   # Pct of noisy candidates with respect to the current TR set size

            # Compute the weighted noise score for each and every instance in the current set
            wns = self._compute_wns(X_curr, y_curr, candidate_mask)
            last_candidate_mask = candidate_mask
            last_wns = wns
            candidate_wns = wns[candidate_mask]
            mean_wns = float(np.mean(candidate_wns))    # Mean wNS


            # Increase by one the stagnation_counter if the mean_wns is reduced (as the more its reduced the less useful it is)
            if prev_mean_wns is None:
                stagnation_counter = 0
            elif mean_wns <= (prev_mean_wns + float(self.wns_tol)):
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_mean_wns = mean_wns
            
            # Extract the labels proposed by each filter
            proposal_labels = np.empty((len(outputs), X_curr.shape[0]), dtype=object) # -> shape=(n_filters, n_curr_samples)
            for out_idx, out in enumerate(outputs):
                # For each model, associate the filter predicted label if its flagged as noisy by itself
                # else leave the original one
                proposal_labels[out_idx] = np.where(out.noisy_mask, out.predicted_labels, y_curr)

            iteration_labels = y_curr.copy()
            updated_labels = iteration_labels.copy()
            keep_mask_curr = np.ones(X_curr.shape[0], dtype=bool)
            relabel_mask_curr = np.zeros(X_curr.shape[0], dtype=bool)
            remove_mask_curr = np.zeros(X_curr.shape[0], dtype=bool)
            retain_mask_curr = np.zeros(X_curr.shape[0], dtype=bool)

            # Compute labels to protect (with min_class_count)
            values, counts = np.unique(iteration_labels, return_counts=True)
            remaining_counts = {val: int(cnt) for val, cnt in zip(values, counts)}
            protected_mask = self._protected_mask(iteration_labels)

            candidate_indices = np.where(candidate_mask)[0]
            candidate_order = candidate_indices[np.argsort(-wns[candidate_indices], kind="mergesort")]

            # For each noisy candidate
            for idx in candidate_order:
                # Do nothing if its class is  protected
                if protected_mask[idx]:
                    retain_mask_curr[idx] = True
                    continue
                source_label = iteration_labels[idx]
                if remaining_counts.get(source_label, 0) <= int(self.min_class_count):
                    retain_mask_curr[idx] = True
                    continue
                # Do nothing (maintain) if its wNS is lower than zero
                if wns[idx] <= 0.0:
                    retain_mask_curr[idx] = True
                    continue

                # Consensus gets priority, then majority voting if the score is
                # at least as high as the average candidate score.
                counts_idx = np.unique(proposal_labels[:, idx], return_counts=True) # Predicted class and associated counts made by filters to the candidate[idx]
                unique_labels = counts_idx[0]   # Predicted labels
                label_counts = counts_idx[1]    # Counts of each predicted label
                best_label = unique_labels[int(np.argmax(label_counts))]    # Label with higher counts
                best_count = int(np.max(label_counts))    # Higher label_count
                consensus = unique_labels.size == 1 # There is a consensus if and only if one class is predicted by all the filters (the same)
                majority = best_count > (len(outputs) / 2.0)    # There is a majority voted class if there is one voted more than half the times
                

                # Decide what to do with the candidate
                target_label = None
                # Relabel if consensus OR (majority AND wNS>=mean(wNS))
                if consensus:
                    target_label = best_label
                elif majority and wns[idx] >= mean_wns:
                    target_label = best_label

                # If there's not even majority vote continue with next candidate
                if target_label is None:
                    continue
                
                # If the candidate is associted the current class continue with next candidate
                if target_label == source_label:
                    retain_mask_curr[idx] = True
                    continue
                
                # Continue if the class has less classes than the minimun required
                if remaining_counts[source_label] - 1 < int(self.min_class_count):
                    continue
                
                # Compute iteration information
                updated_labels[idx] = target_label  # Reflag the label
                relabel_mask_curr[idx] = True   # Flag the candidate as relabeled
                # Update the counts associated to both the original and the relabeled classes
                remaining_counts[source_label] -= 1 
                remaining_counts[target_label] = remaining_counts.get(target_label, 0) + 1

            # If nothing changed, the process has converged.
            if not np.any(relabel_mask_curr) and not np.any(remove_mask_curr):
                history.append(
                    CNCNOSIterationInfo(
                        iter_idx=iter_idx,
                        phase="main",
                        n_samples_before=int(X_curr.shape[0]),
                        n_filters=int(len(outputs)),
                        n_candidates=candidate_count,
                        n_kept=int(keep_mask_curr.sum()),
                        n_relabelled=0,
                        n_removed=0,
                        mean_wns=mean_wns,
                        wns_threshold=mean_wns,
                        stagnation_counter=stagnation_counter,
                    )
                )
                break

            # Conservative removal pass for the remaining noisy candidates.
            for idx in candidate_order:
                if relabel_mask_curr[idx] or protected_mask[idx] or retain_mask_curr[idx]:
                    continue

                source_label = iteration_labels[idx]
                if remaining_counts.get(source_label, 0) <= int(self.min_class_count):
                    continue
                
                # Remove the example in any other case
                keep_mask_curr[idx] = False
                remove_mask_curr[idx] = True
                remaining_counts[source_label] -= 1

            n_relabelled = int(np.sum(relabel_mask_curr))
            n_removed = int(np.sum(remove_mask_curr))
            n_kept = int(np.sum(keep_mask_curr))

            history.append(
                CNCNOSIterationInfo(
                    iter_idx=iter_idx,
                    phase="main",
                    n_samples_before=int(X_curr.shape[0]),
                    n_filters=int(len(outputs)),
                    n_candidates=candidate_count,
                    n_kept=n_kept,
                    n_relabelled=n_relabelled,
                    n_removed=n_removed,
                    mean_wns=mean_wns,
                    wns_threshold=mean_wns,
                    stagnation_counter=stagnation_counter,
                )
            )

            # Apply the accepted changes.
            y_clean_full[current_indices[relabel_mask_curr]] = updated_labels[relabel_mask_curr]
            X_curr = X_curr[keep_mask_curr]
            y_curr = updated_labels[keep_mask_curr]
            current_indices = current_indices[keep_mask_curr]

            if n_relabelled == 0 and n_removed == 0:
                break
            if stagnation_counter >= int(self.stagnation_patience):
                break

        # # Optional final adaptive filtering: conservative consensus removal.
        # if (
        #     bool(self.final_filtering)
        #     and initial_candidate_fraction >= float(self.final_filtering_min_fraction)
        #     and X_curr.shape[0] >= 2
        # ):
        #     outputs, base_filter_names = self._fit_base_filters(X_curr, y_curr, len(history) + 1)
        #     if outputs:
        #         noisy_votes = np.sum([out.noisy_mask for out in outputs], axis=0)
        #         final_noisy_mask = noisy_votes == len(outputs)
        #         if np.any(final_noisy_mask):
        #             values, counts = np.unique(y_curr, return_counts=True)
        #             remaining_counts = {val: int(cnt) for val, cnt in zip(values, counts)}
        #             final_keep_mask = np.ones(X_curr.shape[0], dtype=bool)
        #             final_remove_mask = np.zeros(X_curr.shape[0], dtype=bool)

        #             for idx in np.where(final_noisy_mask)[0]:
        #                 source_label = y_curr[idx]
        #                 if remaining_counts.get(source_label, 0) <= int(self.min_class_count):
        #                     continue
        #                 final_keep_mask[idx] = False
        #                 final_remove_mask[idx] = True
        #                 remaining_counts[source_label] -= 1

        #             if np.any(final_remove_mask):
        #                 history.append(
        #                     CNCNOSIterationInfo(
        #                         iter_idx=len(history) + 1,
        #                         phase="final",
        #                         n_samples_before=int(X_curr.shape[0]),
        #                         n_filters=int(len(outputs)),
        #                         n_candidates=int(np.sum(final_noisy_mask)),
        #                         n_kept=int(np.sum(final_keep_mask)),
        #                         n_relabelled=0,
        #                         n_removed=int(np.sum(final_remove_mask)),
        #                         mean_wns=0.0,
        #                         wns_threshold=0.0,
        #                         stagnation_counter=stagnation_counter,
        #                     )
        #                 )
        #                 final_filtering_applied = True
        #                 X_curr = X_curr[final_keep_mask]
        #                 y_curr = y_curr[final_keep_mask]
        #                 current_indices = current_indices[final_keep_mask]

        # Build final masks aligned to the original order.
        keep_mask = np.zeros(n0, dtype=bool)
        keep_mask[current_indices] = True
        relabel_mask = np.zeros(n0, dtype=bool)
        relabel_mask[current_indices] = y_clean_full[current_indices] != y[current_indices]
        remove_mask = ~keep_mask

        cleaned_X = X[keep_mask]
        cleaned_y = y_clean_full[keep_mask]

        self.X_ = X
        self.y_ = y
        self.classes_ = classes
        self.y_clean_ = y_clean_full
        self.X_clean_ = cleaned_X
        self.y_clean_subset_ = cleaned_y
        self.keep_mask_ = keep_mask
        self.relabel_mask_ = relabel_mask
        self.remove_mask_ = remove_mask
        self.clean_indices_ = current_indices.copy()
        self.base_filter_names_ = base_filter_names

        self.result_ = CNCNOSCleanerResult(
            keep_mask=keep_mask,
            relabel_mask=relabel_mask,
            remove_mask=remove_mask,
            noisy_fraction=float(remove_mask.mean()),
            changed_fraction=float((relabel_mask | remove_mask).mean()),
            initial_candidate_fraction=float(initial_candidate_fraction),
            final_filtering_applied=bool(final_filtering_applied),
            n_iters=int(len(history)),
            history=history,
            base_filter_names=base_filter_names,
            candidate_mask=last_candidate_mask,
            wns=last_wns,
        )
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        return self.X_clean_, self.y_clean_subset_

    def get_filter_report(self) -> Dict[str, Any]:
        r = self.result_
        n_relabelled = int(np.sum(r.relabel_mask))
        n_removed = int(np.sum(r.remove_mask))
        return {
            "n_samples": int(self.X_.shape[0]),
            "n_kept": int(np.sum(r.keep_mask)),
            "n_relabelled": n_relabelled,
            "n_removed": n_removed,
            "fraction_removed": float(r.noisy_fraction),
            "fraction_changed": float(r.changed_fraction),
            "initial_candidate_fraction": float(r.initial_candidate_fraction),
            "final_filtering_applied": bool(r.final_filtering_applied),
            "n_iters": int(r.n_iters),
            "base_filters": list(r.base_filter_names),
            "base_neighbors": int(self.base_neighbors),
            "score_neighbors": int(self.score_neighbors),
            "cv": int(self.cv) if self.cv is not None else None,
            "metric": self.metric,
            "p": int(self.p) if self.metric == "minkowski" else int(self.p),
            "max_iter": int(self.max_iter),
            "stagnation_patience": int(self.stagnation_patience),
            "wns_tol": float(self.wns_tol),
            "final_filtering_min_fraction": float(self.final_filtering_min_fraction),
            "min_class_count": int(self.min_class_count),
            "random_state": int(self.random_state),
        }


# Backwards-friendly aliases.
CNCNOS = CNCNOSCleaner

__all__ = [
    "CNCNOS",
    "CNCNOSCleaner",
    "CNCNOSCleanerResult",
    "CNCNOSIterationInfo",
]
