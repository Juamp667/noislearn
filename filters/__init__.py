"""Public filter exports for the noise-cleaning toolkit."""

from .distance_based import (
    AllKNN,
    DISTANCE_BASED_FILTERS,
    ENNFilter,
    ENNFilterResult,
    ENNProb,
    ENNProbFilter,
    ENNProbFilterResult,
    ENNTh,
    MultiEditFilter,
    MultiEditFilterResult,
    NCNEdit,
    NCNEditFilterResult,
    TomekLinks,
)
from .classifier_based import (
    CLASSIFIER_BASED_FILTERS,
    ClassificationFilter,
    ClassificationFilterResult,
    ClassificationFilterSHAPExplanation,
    ClassificationFilterSHAPReport,
    CVCFFilter,
    CVCFFilterResult,
    EnsembleFilterResult,
    EnsembleFiltering,
    INFFC_old_wrong,
    INFFC_old_wrongFilterResult,
    INFFC_old_wrongIterationInfo,
    IPFIterationInfo,
    IterativePartitioningFilter,
    IterativePartitioningFilterResult,
    explain_classification_filter_noisy_instances,
    TABPFNClassificationFilter,
    c45_like,
)


def print_available_filters():
    """Return the names of all built-in filter families."""
    return DISTANCE_BASED_FILTERS + CLASSIFIER_BASED_FILTERS


__all__ = [
    "AllKNN",
    "CLASSIFIER_BASED_FILTERS",
    "ClassificationFilter",
    "ClassificationFilterResult",
    "ClassificationFilterSHAPExplanation",
    "ClassificationFilterSHAPReport",
    "CVCFFilter",
    "CVCFFilterResult",
    "DISTANCE_BASED_FILTERS",
    "ENNFilter",
    "ENNFilterResult",
    "ENNProb",
    "ENNProbFilter",
    "ENNProbFilterResult",
    "ENNTh",
    "EnsembleFilterResult",
    "EnsembleFiltering",
    "INFFC_old_wrong",
    "INFFC_old_wrongFilterResult",
    "INFFC_old_wrongIterationInfo",
    "IPFIterationInfo",
    "IterativePartitioningFilter",
    "IterativePartitioningFilterResult",
    "MultiEditFilter",
    "MultiEditFilterResult",
    "NCNEdit",
    "NCNEditFilterResult",
    "explain_classification_filter_noisy_instances",
    "TABPFNClassificationFilter",
    "TomekLinks",
    "c45_like",
    "print_available_filters",
]
