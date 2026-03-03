# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Statistical Learning Notebooks for university courses (STAT312: Advanced Data Analytics, STAT321: Linear Models and Time Series Analysis, STAT420: Quantitative Data Analysis). Built with **Marimo** reactive notebooks, exported to **WASM** for GitHub Pages hosting.

## Development Commands

All marimo commands must be run via the `marimo` conda environment:

```bash
# Edit a notebook interactively (opens browser UI)
conda run -n marimo marimo edit "stat312/k-NN Classification/knn_marimo.py"

# Run a notebook in view-only mode
conda run -n marimo marimo run "stat312/KDE/kde_marimo.py"

# Export notebook to WASM for GitHub Pages deployment
conda run -n marimo marimo export html-wasm <notebook> -o <output_dir>

# Install dependencies into the conda environment
conda run -n marimo pip install -r requirements.txt
```

There is no test suite or linter configured for this project.

## Architecture

Each topic lives in its own directory under `stat312/`, `stat321/`, or `stat420/` with:
- `*_marimo.py` - the Marimo notebook source
- `*_wasm/` - exported WASM build (committed to repo for GitHub Pages)
- `README.md` - documentation with live demo links

Notebooks are structured as Marimo apps using `@app.cell` decorators. Each cell returns its exports as a tuple, and Marimo's reactivity graph automatically re-executes dependent cells when inputs change.

Key pattern: interactive controls (`mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.button`) drive visualisations built with **Plotly**. Scikit-learn provides the ML algorithm implementations.

## Conventions

- **Python version**: 3.12 or higher required
- **UK English spelling**: "colour", "visualise", "centre", "licence", "regularisation"
- **Random seeds**: use year-based seeds (e.g., `np.random.seed(2025)`) for reproducibility
- **Markdown in cells**: use raw strings `mo.md(r"""...""")` with LaTeX for mathematical formulas
- **Notebook App setup**: `marimo.App(width="medium", app_title="Title")`; use `width="full"` for 3D visualisations
- **Visualisations**: Plotly exclusively (works in WASM exports; matplotlib is a fallback dependency only)
- WASM exports are **not** gitignored - they are committed for GitHub Pages hosting
- `CLAUDE.md` itself **is** gitignored — use `git add -f CLAUDE.md` when it needs to be committed

## Git Workflow

- Feature branches per notebook (e.g., `stat321-fwl-theorem-notebook`)
- PR workflow with merge to main
- Commit messages: imperative mood, descriptive

## GitHub Pages Deployment

The site is deployed via legacy branch-based GitHub Pages from the `main` branch root. There is no custom CI workflow — GitHub's built-in `pages-build-deployment` runs automatically on push.

**Critical:** The repo root contains a `.nojekyll` file. This **must not be removed**. Without it, GitHub Pages runs Jekyll processing which silently drops all files starting with `_` (e.g. `_baseIsEqual-*.js`). The marimo WASM export produces underscore-prefixed JS chunks that are required at runtime — removing `.nojekyll` will cause notebooks to render as blank pages.

**WASM export cleanup:** The `marimo export html-wasm` command may copy `CLAUDE.md` into the output directory. Always check and remove it from the WASM output before committing.
