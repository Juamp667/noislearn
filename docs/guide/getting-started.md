# Getting started

The documentation site is built with MkDocs Material and reads the local Python modules directly from the repository root.

## Install docs dependencies

```bash
python -m pip install -r requirements-docs.txt
```

## Preview the site

```bash
mkdocs serve
```

## Build a static version

```bash
mkdocs build
```

## What you will find

- A conceptual guide for the main noise models handled by the library.
- A dedicated page for TabPFN-based filtering and local explainability.
- An API reference generated directly from the public docstrings.

!!! tip
    Run MkDocs from the repository root so the local `filters` and `cleaners` packages can be imported without any extra packaging step.
