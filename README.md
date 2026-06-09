<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
`noislearn` is an scikit-learn compatible library for noise filtering, iterative cleaning, and explainability on tabular classification datasets.

## What is included

| Area | Contents |
| --- | --- |
| Noise generation | `URLFNoise`, `NARNoise` |
| Distance-based filters | `AllKNN`, `TomekLinks`, `ENNFilter`, `ENNProbFilter` / `ENNTh`, `MultiEditFilter`, `NCNEdit` |
| Classifier-based filters | `ClassificationFilter` (CF), `CVCFFilter`, `EnsembleFiltering` (EF), `INFFCFilter`, `IterativePartitioningFilter` (IPF) |
| TabPFN-based filters | `TabPFN_CF`, `TabPFN_CVCF`, SHAP-based local explanation reports |
| High-level cleaner | `CNCNOSCleaner` |
| Evaluation tooling | `noise_cv_eval.py`, `testFuncs.py`, notebooks for experiments and analysis |
| Documentation site | MkDocs Material source under `docs/` and local static output under `site/` |

## Current status

- The main noise-filtering families are implemented and exported from `filters/`.
- CNC-NOS is available as a higher-level cleaner in `cleaners/`.
- TabPFN-based filters include local SHAP-based explanation reports for noisy-instance inspection.
- A MkDocs Material documentation site is available locally and built from the public docstrings.

## Repository structure

- `filters/`: distance-based and classifier-based filtering algorithms.
- `cleaners/`: higher-level noise cleaning pipelines.
- `noisers/`: label-noise generation utilities.
- `mkdocs.yml`: documentation configuration.
- `requirements-docs.txt`: documentation dependencies.
- `noise_cv_eval.py`: evaluation script for 5-fold experiments.
- `testFuncs.py`: dataset loading and evaluation helpers.
- `*.ipynb`: notebooks for experimentation and result analysis.

## Local / ignored artifacts

The workspace `.gitignore` excludes a number of generated or local-only paths:

- `docs/`: MkDocs source tree used for the local documentation site.
- `site/`: generated static documentation output.
- `dataset/`: local benchmark data copies.
- `results/`, `results_5cv/`, `results_5cv_cvc_nos/`, `resultsEvaluation/`, `all_results/`: experimental outputs.
- `temp/`: temporary scratch space.
- `*.pkl`: serialized models or intermediate artifacts.

If you want to version any of these assets, adjust `.gitignore` accordingly.

## Documentation

The API documentation is built with MkDocs Material and generated from the public docstrings.

Note that both `docs/` and the generated `site/` directory are ignored in this workspace, so the documentation is intended to be rendered locally rather than committed as build output.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview it locally.

To build the static site:

```bash
mkdocs build --strict
```

## Roadmap

- RNGE.
- CEWS.
- Relabel support in the remaining filters that still only remove noisy instances.
- Continued CNC-NOS tuning and benchmark analysis.
