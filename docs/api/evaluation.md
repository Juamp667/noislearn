# Evaluation utilities

The evaluation scripts are kept as repository modules rather than packaged estimators. They are intended for experiments, notebooks, and benchmark runs.

## `noise_cv_eval.py`

The 5-fold evaluation module provides helpers for baseline runs, filtered runs, full experiments, and grid experiments:

- `run_5cv_baseline`
- `run_5cv_filters`
- `run_5cv_experiment`
- `run_5cv_grid`

The classification output includes fold metadata, preprocessing metadata, timing, validity flags, and common metrics such as accuracy, balanced accuracy, macro-F1, macro-precision, and macro-recall.

The removal output includes the predicted removals, true noisy rows when known, exact/partial ground-truth metadata, and confusion-matrix-derived removal metrics.

## `noise_score_eval.py`

::: noise_score_eval.evaluate_noise_score_ranking
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noise_score_eval.evaluate_all_noise_scores
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noise_score_eval.filtering_curve_evaluation
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noise_score_eval.compare_filtering_curves
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: noise_score_eval.summarize_filtering_curve
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
