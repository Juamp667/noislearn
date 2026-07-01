# Evaluation workflow

The repository includes two complementary evaluation paths: fold-based experiments over KEEL-style datasets and score-oriented evaluation for filters that expose a continuous `noise_score_`.

## 5-fold experiments

`noise_cv_eval.py` builds 5-fold experiments with preprocessing, an optional noise filter, and a final classifier. It returns two tables:

- Classification metrics by fold and method.
- Removal metrics for the filter decisions.

The removal table tracks `n_true_noisy`, `n_removed_pred`, `precision_removal`, `recall_removal`, `f1_removal`, `specificity`, `mcc`, and the confusion counts `tp`, `tn`, `fp`, `fn`.

## Ground-truth noise masks

When the noisy fold has a persisted mask next to the source data, the evaluator uses it as the exact ground truth. Supported suffixes are:

- `.noise_mask.npy`
- `.noise_mask.csv`
- `.noise_mask.txt`

CSV/TXT masks can contain a single unnamed column or one of the named columns `is_noisy`, `noise_mask`, `noisy`, or `mask`. Boolean values can be written as `true`/`false`, `yes`/`no`, or `1`/`0`.

If no persisted mask is available, the evaluator falls back to a clean-reference comparison. Ambiguous rows are marked as unknown, and removal metrics exclude those rows from the confusion matrix.

## Continuous noise-score evaluation

`noise_score_eval.py` evaluates filters without forcing a single arbitrary threshold.

- `evaluate_noise_score_ranking` reports ROC-AUC, average precision, Precision@k, Recall@k, and F1@k.
- `evaluate_all_noise_scores` compares several score vectors in one table.
- `filtering_curve_evaluation`, `compare_filtering_curves`, and `summarize_filtering_curve` measure downstream classifier behavior after removing increasing percentages of the most suspicious samples.

## Recommended reporting

For each experimental run, keep both decision metrics and ranking metrics when a filter exposes `noise_score_`. This makes it possible to separate the quality of the suspiciousness ranking from the quality of a particular threshold.
