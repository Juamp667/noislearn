# Cleaners

The cleaners combine several filters into higher-level cleaning strategies.

## CNCNOSCleaner

::: cleaners.cnc_nos.CNCNOSCleaner
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: cleaners.cnc_nos.CNCNOSCleanerResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: cleaners.cnc_nos.CNCNOSIterationInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## What to look at in the report

- `keep_mask`: samples preserved after cleaning.
- `relabel_mask`: samples whose label was changed.
- `remove_mask`: samples removed from the final cleaned set.
- `history`: per-iteration trace of the cleaning process.
