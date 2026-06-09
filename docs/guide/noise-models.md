# Noise models and filtering families

`noislearn` focuses on classification settings where the training data may contain wrong labels, ambiguous instances, or local inconsistencies.

## Main noise families

| Family | Main idea | Typical use |
| --- | --- | --- |
| Distance-based | Compare each instance with its local neighborhood | Fast baselines and local consistency checks |
| Classifier-based | Compare observed labels against out-of-fold predictions | Label-noise detection from model disagreement |
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
- Use TabPFN-based filters when you want stronger tabular predictions and local SHAP reports.
- Use CNC-NOS when you want a higher-level cleaning strategy that combines several filters.
