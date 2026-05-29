try:
    from .aknn import AllKNN
except ModuleNotFoundError:
    AllKNN = None
from .enn import ENNFilter, ENNFilterResult
from .ennTh import ENNProb, ENNTh, ENNProbFilter, ENNProbFilterResult
from .multiedit import MultiEditFilter, MultiEditFilterResult
from .ncnedit import NCNEdit, NCNEditFilterResult
try:
    from .tomeklinks import TomekLinks
except ModuleNotFoundError:
    TomekLinks = None

DISTANCE_BASED_FILTERS = [
    "ENNFilter",
    "ENNProb",
    "ENNTh",
    "MultiEditFilter",
    "NCNEdit",
]

if AllKNN is not None:
    DISTANCE_BASED_FILTERS.insert(0, "AllKNN")
if TomekLinks is not None:
    DISTANCE_BASED_FILTERS.insert(1 if AllKNN is not None else 0, "TomekLinks")

__all__ = [
    "DISTANCE_BASED_FILTERS",
    "ENNFilter",
    "ENNFilterResult",
    "ENNProb",
    "ENNProbFilter",
    "ENNProbFilterResult",
    "ENNTh",
    "MultiEditFilter",
    "MultiEditFilterResult",
    "NCNEdit",
    "NCNEditFilterResult",
]

if AllKNN is not None:
    __all__.insert(0, "AllKNN")
if TomekLinks is not None:
    __all__.append("TomekLinks")
