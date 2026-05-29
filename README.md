<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
`noislearn` is an scikit-learn compatible library gathering a lot of noise-filtering techniques used for classification datasets.

## Features

- Noise filtering algorithms
- Noise generation utilities (currently NAR and NCAR noise generators)
- Notebook-based experimentation
- Helper functions for testing and evaluation with widely-used datsets.
- CNC-NOS cleaner for iterative noise detection, relabeling, and removal.

## Project Status

#### Implemented filters include:

- ENN
- MEEN
- CF
- AKNN
- IPF
- EF
- MultiEdit
- CVCF
- NCNEdit
- ENNTh
- INFFC
- TomekLinks
- CNC-NOS

#### Current work

- Testing CNC-NOS against noisy benchmark datasets.
- Comparing baseline prediction performance vs. cleaned training sets.
- Debugging `wNS` computation and iteration stopping criteria.
#### Planned or in progress:
- RNGE
- CEWS


## Repository Structure

- `filters/`: filtering algorithms
- `noisers/`: noise generation utilities
- `cleaners/`: higher-level noise cleaning pipelines
- `testing.ipynb`: experimentation notebook
- `testFuncs.py`: helper/test functions

## Documentation

The API documentation is being prepared with MkDocs Material.

- Local preview: `mkdocs serve`
- Build static site: `mkdocs build`
