# Getting started

The documentation site is built with MkDocs Material and reads the local Python modules directly from the repository root.

## Install docs dependencies

```bash
python -m pip install -r requirements-docs.txt
```

## Preview the site

```bash
mkdocs serve
```

## Build a static version

```bash
mkdocs build
```

## What you will find

- A conceptual guide for the main noise models handled by the library.
- A dedicated page for TabPFN-based filtering and local explainability.
- An evaluation guide for 5-fold experiments, persisted noise masks, and continuous noise scores.
- An API reference generated directly from the public docstrings.

## Minimal examples

Generate label noise and keep its true mask:

```python
from noisers.funcs import urlf

y_noisy, noise_mask = urlf(y, noise_level=0.2, random_state=42, return_mask=True)
```

Run a filter in detection mode:

```python
from filters import ENNFilter

filt = ENNFilter(action="detect")
filt.fit_resample(X, y_noisy)
report = filt.get_detection_report()
```

Combine multiple filters with FEF:

```python
from filters import ENNFilter, CVCFFilter, FilterEnsembleFilter

fef = FilterEnsembleFilter(
    base_filters=[("ENN", ENNFilter()), ("CVCF", CVCFFilter())],
    strategy="majority",
)
X_clean, y_clean = fef.fit_resample(X, y_noisy)
```

!!! tip
    Run MkDocs from the repository root so the local `filters` and `cleaners` packages can be imported without any extra packaging step.
