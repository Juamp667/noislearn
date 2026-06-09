<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
`noislearn` is an scikit-learn compatible library for noise filtering, iterative cleaning, and explainability on tabular classification datasets.

## What is included

| Area                     | Contents                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Noise generation         | `URLFNoise`, `NARNoise`                                                                                                 |
| Distance-based filters   | `AllKNN`, `TomekLinks`, `ENNFilter`, `ENNProbFilter` / `ENNTh`, `MultiEditFilter`, `NCNEdit`                            |
| Classifier-based filters | `ClassificationFilter` (CF), `CVCFFilter`, `EnsembleFiltering` (EF), `INFFCFilter`, `IterativePartitioningFilter` (IPF) |
| TabPFN-based filters     | `TabPFN_CF`, `TabPFN_CVCF`, SHAP-based local explanation reports                                                        |
| High-level cleaner       | `CNCNOSCleaner`                                                                                                         |
| Evaluation tooling       | `noise_cv_eval.py`, `testFuncs.py`, notebooks for experiments and analysis                                              |
| Documentation site       | MkDocs Material source and API reference                                                                                |

## Current status

- The main noise-filtering families are implemented and exported from `filters/`.
- CNC-NOS is available as a higher-level cleaner in `cleaners/`.
- TabPFN-based filters include local SHAP-based explanation reports for noisy-instance inspection.
- 

## Repository structure

- `filters/`: distance-based and classifier-based filtering algorithms.
- `cleaners/`: higher-level noise cleaning pipelines.
- `noisers/`: label-noise generation utilities.
- `requirements-docs.txt`: documentation dependencies.
- `noise_cv_eval.py`: evaluation script for 5-fold experiments.
- `testFuncs.py`: dataset loading and evaluation helpers.
- `*.ipynb`: notebooks for experimentation and result analysis.

## ## Roadmap

- RNGE.
- CEWS.
- Relabel support in the remaining filters that still only remove noisy instances.
- Continued CNC-NOS tuning and benchmark analysis.
