"""
Tomek-links label-noise filtering.
"""

try:
    from imblearn.under_sampling import TomekLinks
except ModuleNotFoundError:
    TomekLinks = None
