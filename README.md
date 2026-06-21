<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)

Documentation site: `https://noislearn.org/`

`noislearn` is an scikit-learn compatible library for noise filtering, iterative cleaning, and explainability on tabular classification datasets.

## What is included

| Area                     | Contents                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Noise generation         | `URLFNoise`, `NARNoise`                                                                                                 |
| Distance-based filters   | `AllKNN`, `TomekLinks`, `ENNFilter`, `ENNProbFilter` / `ENNTh`, `MultiEditFilter`, `NCNEdit`                            |
| Classifier-based filters | `ClassificationFilter` (CF), `CVCFFilter`, `FilterEnsembleFilter` (FEF), `EnsembleFiltering` (EF), `INFFC_old_wrong`, `IterativePartitioningFilter` (IPF) |
| TabPFN-based filters     | `TabPFN_CF`, `TabPFN_CVCF`, SHAP-based local explanation reports                                                        |
| High-level cleaner       | `CNCNOSCleaner`                                                                                                         |
| Evaluation tooling       | `noise_cv_eval.py`, `testFuncs.py`, notebooks for experiments and analysis                                              |
| Documentation site       | MkDocs Material source and API reference                                                                                |

## Current status

- The main noise-filtering families are implemented and exported from `filters/`.
- CNC-NOS is available as a higher-level cleaner in `cleaners/`.
- TabPFN-based filters include local SHAP-based explanation reports for noisy-instance inspection.


## Repository structure

- `filters/`: distance-based and classifier-based filtering algorithms.
- `cleaners/`: higher-level noise cleaning pipelines.
- `noisers/`: label-noise generation utilities.
- `mkdocs.yml`: documentation configuration.
- `docs/`: MkDocs source for the public documentation site.
- `requirements-docs.txt`: documentation dependencies.
- `noise_cv_eval.py`: evaluation script for 5-fold experiments.
- `testFuncs.py`: dataset loading and evaluation helpers.
- `*.ipynb`: notebooks for experimentation and result analysis.

## Roadmap

- RNGE.
- CEWS.
- Relabel support in the remaining filters that still only remove noisy instances.
- Continued CNC-NOS tuning and benchmark analysis.

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
