# Distance-based filters

These filters rely on neighborhood geometry and local consistency.

## Overview

- `ENNFilter` and `ENNProbFilter` use nearest-neighbor voting.
- `MultiEditFilter` iteratively cleans the data in stratified blocks.
- `NCNEdit` selects neighbors by minimizing the centroid distance.

`AllKNN` and `TomekLinks` are re-exported from `imbalanced-learn` when that dependency is available.

## ENN

::: filters.distance_based.enn.ENNFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.distance_based.enn.ENNFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## ENNProb

::: filters.distance_based.ennTh.ENNProbFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.distance_based.ennTh.ENNProbFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## MultiEdit

::: filters.distance_based.multiedit.MultiEditFilter
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.distance_based.multiedit.MultiEditFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## NCNEdit

::: filters.distance_based.ncnedit.NCNEdit
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: filters.distance_based.ncnedit.NCNEditFilterResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
