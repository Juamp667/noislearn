"""
Tomek-links label-noise filtering.
"""

# Re-export the imbalanced-learn implementation under this namespace.
try:
    from imblearn.under_sampling import TomekLinks
except ModuleNotFoundError:
    TomekLinks = None
