from .classification import ClassificationFilter, ClassificationFilterResult
from .cvcf import CVCFFilter, CVCFFilterResult
from .enn import ENNFilter, ENNFilterResult
from .ensemble import EnsembleFiltering, EnsembleFilterResult
from .ncnedit import NCNEdit, NCNEditFilterResult
from .iterative_partitioning import IPFIterationInfo, IterativePartitioningFilter, IterativePartitioningFilterResult, c45_like

def print_available_filters():
    return ["ClassificationFilter", "ENNFilter", "EnsembleFilter",
        "IterativePartitioningFilter", "NCNEdit", "CVCFFilter", ]
