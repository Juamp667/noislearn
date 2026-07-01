<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)

Documentation site: `https://noislearn.org/`

`noislearn` is an scikit-learn compatible library for label-noise generation, noise filtering, iterative cleaning, noise-score evaluation, and explainability on tabular classification datasets.

## What is included

| Area                     | Contents                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Noise generation         | `URLFNoise`, `NARNoise`, `urlf`, `nar`, and persisted `noise_mask_` support                                             |
| Distance-based filters   | `AllKNN`, `TomekLinks`, `ENNFilter`, `ENNProbFilter` / `ENNTh`, `MultiEditFilter`, `NCNEdit`                            |
| Classifier-based filters | `ClassificationFilter` (CF), `CVCFFilter`, `FilterEnsembleFilter` (FEF), `EnsembleFiltering` (EF), `INFFC_old_wrong`, `IterativePartitioningFilter` (IPF) |
| TabPFN-based filters     | `TabPFN_CF`, `TabPFN_CVCF`, SHAP-based local explanation reports                                                        |
| High-level cleaner       | `CNCNOSCleaner`                                                                                                         |
| Noise-score filtering    | `NoiseScoreFilter` with mean, quantile, beta-adaptive, and rational-valley thresholds                                   |
| Evaluation tooling       | `noise_cv_eval.py`, `noise_score_eval.py`, `testFuncs.py`, notebooks for experiments and analysis                       |
| Documentation site       | MkDocs Material source and API reference                                                                                |

## Current status

- The main noise-filtering families are implemented and exported from `filters/`.
- CNC-NOS is available as a higher-level cleaner in `cleaners/`.
- TabPFN-based filters include local SHAP-based explanation reports for noisy-instance inspection.
- Filters expose a common detection report with `noisy_mask`, `noisy_indices`, `noisy_fraction`, optional `noise_score`, and `action` metadata when supported.
- The default filter action removes suspicious samples, while `action="detect"` preserves the original data and only reports detections.
- Noise generators can return the true corruption mask and estimator wrappers store it as `noise_mask_`.
- The 5-fold evaluation utilities can use persisted `.noise_mask.npy`, `.noise_mask.csv`, or `.noise_mask.txt` files and track unknown ground-truth rows separately.


## Repository structure

- `filters/`: distance-based and classifier-based filtering algorithms.
- `cleaners/`: higher-level noise cleaning pipelines.
- `noisers/`: label-noise generation utilities.
- `noise_score_eval.py`: ranking and filtering-curve evaluation for continuous noise scores.
- `mkdocs.yml`: documentation configuration.
- `docs/`: MkDocs source for the public documentation site.
- `requirements-docs.txt`: documentation dependencies.
- `noise_cv_eval.py`: evaluation script for 5-fold experiments.
- `testFuncs.py`: dataset loading and evaluation helpers.
- `*.ipynb`: notebooks for experimentation and result analysis.

## Roadmap

- RNGE.
- CEWS.
- Relabel support in filters where `action="relabel"` is still reserved.
- Weight-based downstream training support where `action="weight"` is still reserved.
- Continued CNC-NOS tuning and benchmark analysis.

## Common usage

### Add reproducible label noise

```python
from noisers.funcs import urlf

y_noisy, noise_mask = urlf(y, noise_level=0.2, random_state=42, return_mask=True)
```

The estimator wrappers expose the same information after `fit_resample`:

```python
from noisers.classes import URLFNoise

noiser = URLFNoise(noise_level=0.2, random_state=42)
X_same, y_noisy = noiser.fit_resample(X, y)
noise_mask = noiser.noise_mask_
```

### Detect without removing

```python
from filters import ENNFilter

filt = ENNFilter(n_neighbors=3, action="detect")
X_out, y_out = filt.fit_resample(X, y)
report = filt.get_detection_report()
```

With `action="detect"`, `X_out` and `y_out` keep the original samples and the report contains the detected noisy rows.

### Combine filters with FEF

```python
from filters import ENNFilter, CVCFFilter, FilterEnsembleFilter

fef = FilterEnsembleFilter(
    base_filters=[("ENN", ENNFilter()), ("CVCF", CVCFFilter())],
    strategy="weighted_threshold",
    score_threshold=0.5,
    use_filter_scores=True,
    random_state=42,
)
X_clean, y_clean = fef.fit_resample(X, y)
ensemble_score = fef.get_ensemble_score()
```

### Threshold an existing noise score

```python
from filters import NoiseScoreFilter

score_filter = NoiseScoreFilter(
    noise_scores=scores,
    threshold="rational_valley",
)
X_clean, y_clean = score_filter.fit_resample(X, y)
threshold_details = score_filter.threshold_report_
```

## Documentation

The API documentation is built with MkDocs Material and generated from the public docstrings.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview it locally.

The documentation is also configured for GitHub Pages and can be published under the custom domain `noislearn.org`.

To build the static site locally:

```bash
mkdocs build --strict
```
