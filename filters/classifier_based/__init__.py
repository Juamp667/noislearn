"""Public exports for classifier-based label-noise filters."""

try:
    from .classification import ClassificationFilter, ClassificationFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    ClassificationFilter = None
    ClassificationFilterResult = None

try:
    from .cvcf import CVCFFilter, CVCFFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    CVCFFilter = None
    CVCFFilterResult = None

try:
    from .ensemble import EnsembleFiltering, EnsembleFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    EnsembleFiltering = None
    EnsembleFilterResult = None

try:
    from .fef import FEF, FilterEnsembleFilter, FilterEnsembleFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    FEF = None
    FilterEnsembleFilter = None
    FilterEnsembleFilterResult = None

try:
    from .inffc_old_wrong import INFFC_old_wrong, INFFC_old_wrongFilterResult, INFFC_old_wrongIterationInfo
except Exception:  # pragma: no cover - optional dependency fallback.
    INFFC_old_wrong = None
    INFFC_old_wrongFilterResult = None
    INFFC_old_wrongIterationInfo = None

try:
    from .iterative_partitioning import IPFIterationInfo, IterativePartitioningFilter, IterativePartitioningFilterResult, c45_like
except Exception:  # pragma: no cover - optional dependency fallback.
    IPFIterationInfo = None
    IterativePartitioningFilter = None
    IterativePartitioningFilterResult = None
    c45_like = None

from .shap_explanations import (
    ClassificationFilterSHAPExplanation,
    ClassificationFilterSHAPReport,
    ShapDifferenceInstanceExplanation,
    ShapDifferenceReport,
    compute_explanatory_noise_score,
    compute_shap_class_difference,
    explain_classification_filter_shap_difference,
    explain_classification_filter_noisy_instances,
    explain_instance_shap_difference,
    explain_noisy_instances_with_shap,
)
try:
    from .TabPFN_based import *
except Exception:  # pragma: no cover - optional dependency fallback.
    TABPFNClassificationFilter = None
# Ordered list of the classifier-based filters exposed by this package.
CLASSIFIER_BASED_FILTERS = [
    "ClassificationFilter",
    "CVCFFilter",
    "FilterEnsembleFilter",
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
    "ShapDifferenceInstanceExplanation",
    "ShapDifferenceReport",
    "CVCFFilter",
    "CVCFFilterResult",
    "FEF",
    "FilterEnsembleFilter",
    "FilterEnsembleFilterResult",
    "EnsembleFilterResult",
    "EnsembleFiltering",
    "INFFC_old_wrong",
    "INFFC_old_wrongFilterResult",
    "INFFC_old_wrongIterationInfo",
    "IPFIterationInfo",
    "IterativePartitioningFilter",
    "IterativePartitioningFilterResult",
    "explain_classification_filter_noisy_instances",
    "explain_classification_filter_shap_difference",
    "compute_explanatory_noise_score",
    "compute_shap_class_difference",
    "explain_instance_shap_difference",
    "explain_noisy_instances_with_shap",
    "TABPFNClassificationFilter",
    "c45_like",
]
