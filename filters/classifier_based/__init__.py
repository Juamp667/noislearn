"""Public exports for classifier-based label-noise filters."""

from .classification import ClassificationFilter, ClassificationFilterResult
from .cvcf import CVCFFilter, CVCFFilterResult
from .ensemble import EnsembleFiltering, EnsembleFilterResult
from .inffc_old_wrong import INFFC_old_wrong, INFFC_old_wrongFilterResult, INFFC_old_wrongIterationInfo
from .iterative_partitioning import IPFIterationInfo, IterativePartitioningFilter, IterativePartitioningFilterResult, c45_like
from .shap_explanations import ClassificationFilterSHAPExplanation, ClassificationFilterSHAPReport, explain_classification_filter_noisy_instances
from .TabPFN_based import *
# Ordered list of the classifier-based filters exposed by this package.
CLASSIFIER_BASED_FILTERS = [
    "ClassificationFilter",
    "CVCFFilter",
    "EnsembleFiltering",
    "INFFC_old_wrong",
    "IterativePartitioningFilter",
]

if TABPFNClassificationFilter is not None:
    CLASSIFIER_BASED_FILTERS.append("TABPFNClassificationFilter")

__all__ = [
    "CLASSIFIER_BASED_FILTERS",
    "ClassificationFilter",
    "ClassificationFilterResult",
    "ClassificationFilterSHAPExplanation",
    "ClassificationFilterSHAPReport",
    "CVCFFilter",
    "CVCFFilterResult",
    "EnsembleFilterResult",
    "EnsembleFiltering",
    "INFFC_old_wrong",
    "INFFC_old_wrongFilterResult",
    "INFFC_old_wrongIterationInfo",
    "IPFIterationInfo",
    "IterativePartitioningFilter",
    "IterativePartitioningFilterResult",
    "explain_classification_filter_noisy_instances",
    "TABPFNClassificationFilter",
    "c45_like",
]
