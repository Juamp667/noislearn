"""Public exports for distance-based label-noise filters."""

try:
    from .aknn import AllKNN
except ModuleNotFoundError:
    AllKNN = None
try:
    from .enn import ENNFilter, ENNFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    ENNFilter = None
    ENNFilterResult = None

try:
    from .ennTh import ENNProb, ENNTh, ENNProbFilter, ENNProbFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    ENNProb = None
    ENNTh = None
    ENNProbFilter = None
    ENNProbFilterResult = None

try:
    from .multiedit import MultiEditFilter, MultiEditFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    MultiEditFilter = None
    MultiEditFilterResult = None

try:
    from .ncnedit import NCNEdit, NCNEditFilterResult
except Exception:  # pragma: no cover - optional dependency fallback.
    NCNEdit = None
    NCNEditFilterResult = None

try:
    from .tomeklinks import TomekLinks
except ModuleNotFoundError:
    TomekLinks = None
# Ordered list of the distance-based filters exposed by this package.
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
