<div style="display: flex; flex-direction: row; align-items: center; gap: 12px;">
  <img src="logo.png" alt="" width="62">
  <h1 style="margin: 0;">noislearn</h1>
</div>
`noislearn` is an scikit-learn compatible library gathering a lot of noise-filtering techniques used for classification datasets. As

## Features

- Noise filtering algorithms
- Noise generation utilities (currently NAR and NCAR noise generators)
- Notebook-based experimentation
- Helper functions for testing and evaluation with widely-used datsets.

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
- 
#### Planned or in progress:
- RNGE
- CEWS


## Repository Structure

- `filters/`: filtering algorithms
- `noisers/`: noise generation utilities
- `testing.ipynb`: experimentation notebook
- `testFuncs.py`: helper/test functions
