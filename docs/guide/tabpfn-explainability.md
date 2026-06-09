# TabPFN explainability

The TabPFN-based filters expose a local explanation report for noisy instances.

## What the report describes

For each instance, the report typically includes:

- `sample_idx`: original sample index.
- `fold_idx`: fold used to train the local model.
- `true_label`: observed label.
- `oof_pred`: out-of-fold predicted label.
- `confidence`: probability associated with the predicted label.
- `top_k`: the strongest SHAP contributions for the selected target class.

## Default explanation target

By default, the SHAP values explain the predicted class. This means the report shows which features push the model toward the class selected by the filter.

## How to read the signs

- Positive SHAP values increase the score of the explained class.
- Negative SHAP values decrease that score.
- Larger absolute values indicate stronger local influence.

## What the report does not say

- It does not prove that a label is causally wrong.
- It does not validate semantic correctness on its own.
- It does not replace human domain review when causal interpretation is required.

## Example interpretation

```text
true_label = 2
oof_pred    = 1
confidence  = 0.51
top_k       = [feature_a, feature_b, feature_c]
```

This means the filter considers the sample closer to class `1` than to the observed label `2`, but the decision is weak if the confidence is near 0.5.
