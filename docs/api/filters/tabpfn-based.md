# TabPFN-based filters

These filters use `TabPFNClassifier` as the base learner and expose a local explanation report for noisy samples.

## TabPFN_CF

::: filters.classifier_based.TabPFN_based.TabPFN_CF.TabPFN_CF
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CF.TabPFNFoldInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CF.TabPFNNoiseExplanation
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CF.TabPFNExplanationReport
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## TabPFN_CVCF

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFN_CVCF
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFNCommitteeFoldInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFNCommitteeFoldExplanation
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFNCommitteeAggregatedView
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFNCommitteeSampleExplanation
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.TabPFN_based.TabPFN_CVCF.TabPFNCommitteeExplanationReport
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## Reading the explanation report

- `class_index="predicted"` explains the class chosen by the model.
- `top_k` stores the strongest local SHAP contributions.
- `confidence` is the probability of the explained class.
- `all_view` aggregates every fold, while `majority_view` only keeps the folds aligned with the committee vote.
