# Common libraries along two or more filters
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import StratifiedKFold

# Available filters
from .classification import ClassificationFilter, ClassificationFilterResult
from .cvcf import CVCFFilter, CVCFFilterResult
from .enn import ENNFilter, ENNFilterResult
from .ennTh import ENNProb, ENNTh, ENNProbFilter, ENNProbFilterResult
from .ensemble import EnsembleFiltering, EnsembleFilterResult
from .ncnedit import NCNEdit, NCNEditFilterResult
from .iterative_partitioning import IPFIterationInfo, IterativePartitioningFilter, IterativePartitioningFilterResult, c45_like

def print_available_filters():
    return ["ClassificationFilter", "ENNFilter", "ENNProb", "ENNTh", "EnsembleFilter",
        "IterativePartitioningFilter", "NCNEdit", "CVCFFilter", ]
