# Noise generators

The noise generators create controlled label corruption for tabular classification experiments.

## Functional API

::: noisers.funcs.urlf
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noisers.funcs.nar
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## Estimator API

::: noisers.classes.URLFNoise
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noisers.classes.NARNoise
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## Mask support

- `urlf(..., return_mask=True)` and `nar(..., return_mask=True)` return `(y_noisy, noise_mask)`.
- `URLFNoise.fit_resample` and `NARNoise.fit_resample` store the same mask in `noise_mask_`.
- The mask is `True` only where the final label differs from the original label.
