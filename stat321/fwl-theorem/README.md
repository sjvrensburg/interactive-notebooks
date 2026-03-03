# Frisch-Waugh-Lovell Theorem

An interactive visualisation of the FWL theorem and its implications for partial regression in multiple linear regression.

## Live Demo

[View Interactive Notebook](./fwl_wasm/) *(after WASM export)*

## Overview

This notebook illustrates the **Frisch-Waugh-Lovell (FWL) theorem**, a fundamental result in multiple regression that shows how the coefficient on a particular regressor can be obtained by:

1. **Partialling out** (residualising) all other regressors from both the dependent variable and the regressor of interest
2. Regressing the residualised dependent variable on the residualised regressor

### Key Concepts

- **Partial regression**: Isolating the relationship between $y$ and $X_1$ while controlling for $X_2$
- **Omitted variable bias**: How the simple regression coefficient differs from the multiple regression coefficient when predictors are correlated
- **Geometric interpretation**: Understanding FWL through projections in column space

## Interactive Features

- Adjust sample size, predictor correlation, true coefficients, and noise level
- Visualise the partialling out process step-by-step
- Compare coefficients from simple regression, full OLS, and FWL approaches
- See how omitted variable bias changes with predictor correlation

## Running Locally

```bash
conda run -n marimo marimo edit "stat321/fwl-theorem/fwl_marimo.py"
```

## Export for GitHub Pages

```bash
conda run -n marimo marimo export html-wasm "stat321/fwl-theorem/fwl_marimo.py" -o "stat321/fwl-theorem/fwl_wasm"
```
