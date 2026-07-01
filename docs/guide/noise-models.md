# Noise models and filtering families

`noislearn` focuses on classification settings where the training data may contain wrong labels, ambiguous instances, or local inconsistencies.

## Main noise families

| Family | Main idea | Typical use |
| --- | --- | --- |
| Noise generators | Inject controlled label corruption with URLF or class-dependent NAR | Reproducible benchmark setup |
| Distance-based | Compare each instance with its local neighborhood | Fast baselines and local consistency checks |
| Classifier-based | Compare observed labels against out-of-fold predictions | Label-noise detection from model disagreement |
| Filter ensemble | Aggregate several filters by votes or scores | Robust decisions across heterogeneous criteria |
| Noise-score filter | Threshold a continuous suspiciousness score | Controlled removal from ranking outputs |
| TabPFN-based | Use TabPFN as the base learner and inspect its local explanations | Strong tabular classifier plus explanation support |
| CNC-NOS | Combine several filters and a weighted noise score | Higher-level cleaning pipeline |

## Label noise

Label noise appears when an instance is assigned the wrong class. This is the main target of the filtering algorithms in the repository.

## Attribute noise

Attribute noise affects feature values rather than the target label. The current API is primarily centered on class-noise filtering, not on feature corruption repair.

## Why filtering can be useful

- It can remove or relabel suspicious samples before training the final classifier.
- It can improve downstream generalization when the training labels are unreliable.
- It can provide a structured report that helps audit what was removed and why.

## When each family is useful

- Use distance-based filters when you want a simple local consistency baseline.
- Use classifier-based filters when you want out-of-fold disagreement logic.
- Use `FilterEnsembleFilter` when you want to aggregate several detectors and inspect support, vote, and score matrices.
- Use `NoiseScoreFilter` when you already have a `noise_score_` vector and want a mean, quantile, adaptive, or rational-valley threshold.
- Use TabPFN-based filters when you want stronger tabular predictions and local SHAP reports.
- Use CNC-NOS when you want a higher-level cleaning strategy that combines several filters.

## Detection reports

Filters that support the shared detection interface expose `noisy_mask_`, `noisy_indices_`, `noisy_fraction_`, `detection_report_`, and `get_detection_report()`. When available, `noise_score_` stores a continuous suspiciousness score where larger values indicate more suspicious samples.

Use `action="remove"` to return only kept samples from `fit_resample`. Use `action="detect"` to keep the input unchanged and inspect the report.
