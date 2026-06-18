# Classifier-based filters

These filters compare the observed labels against predictions obtained from one or more base classifiers.

## Overview

- `ClassificationFilter` uses a single classifier and out-of-fold predictions.
- `CVCFFilter` aggregates fold-wise committee votes.
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
