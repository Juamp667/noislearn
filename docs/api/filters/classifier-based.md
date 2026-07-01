# Classifier-based filters

These filters compare the observed labels against predictions obtained from one or more base classifiers.

## Overview

- `ClassificationFilter` uses a single classifier and out-of-fold predictions.
- `CVCFFilter` aggregates fold-wise committee votes.
- `FilterEnsembleFilter` (FEF) combines several existing filters by vote and score.
- `EnsembleFiltering` compares several estimators.
- `INFFC_old_wrong` iteratively fuses a heterogeneous committee.
- `IterativePartitioningFilter` repeatedly partitions the data and checks agreement.

## ClassificationFilter

::: filters.classifier_based.classification.ClassificationFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.classification.ClassificationFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## CVCFFilter

::: filters.classifier_based.cvcf.CVCFFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.cvcf.CVCFFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## EnsembleFiltering

::: filters.classifier_based.ensemble.EnsembleFiltering
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.ensemble.EnsembleFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## FilterEnsembleFilter

FEF accepts either filter instances or `(name, filter)` pairs. It supports `union`, `majority`, `consensus`, `k_of_m`, `threshold`, and `weighted_threshold` strategies. The report includes per-filter votes, score matrices, normalized weights, support counts, support fractions, base reports, and class-protection metadata.

::: filters.classifier_based.fef.FilterEnsembleFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.fef.FilterEnsembleFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## NoiseScoreFilter

`NoiseScoreFilter` thresholds an existing score vector or the `noise_score_` exposed by another fitted filter. It supports fixed numeric thresholds, `mean`, `quantile`, beta-adaptive thresholding, and rational-valley thresholding with a detailed `threshold_report_`.

::: filters.noiseScoreFiltering.NoiseScoreFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## INFFC_old_wrong

::: filters.classifier_based.inffc_old_wrong.INFFC_old_wrong
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.inffc_old_wrong.INFFC_old_wrongFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.inffc_old_wrong.INFFC_old_wrongIterationInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## IterativePartitioningFilter

::: filters.classifier_based.iterative_partitioning.IterativePartitioningFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.iterative_partitioning.IterativePartitioningFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.classifier_based.iterative_partitioning.IPFIterationInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
