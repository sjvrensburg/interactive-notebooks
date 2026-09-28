# Gaussian Classifiers: Naive Bayes, LDA & QDA

Interactive notebook comparing three probabilistic classifiers for a
2-feature, binary classification problem: **Gaussian Naive Bayes**, **Linear
Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)**.

**[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat312/Gaussian%20Classifiers/gaussian_classifiers_wasm/)**

## Overview

All three classifiers apply Bayes' theorem to Gaussian class-conditional
densities, but differ in how much structure they assume about the
class covariance matrices:

| Classifier | Covariance assumption | Boundary shape |
|:---|:---|:---|
| Gaussian Naive Bayes | Diagonal, possibly different per class | Quadratic, axis-aligned bias |
| LDA | Full, shared across classes | Linear |
| QDA | Full, separate per class | Quadratic |

The notebook grounds this comparison in the Bayes'-theorem derivation and
the LDA discriminant function from the STAT312 class notes (Chapter 6),
and extends it to QDA — not covered in the notes — to complete the
picture referenced in the notes' callout on why equal covariances produce
linear boundaries.

## Features

- **Population controls**: set the mean vector and (co)variance of two
  Gaussian classes directly via sliders (mean, standard deviations,
  correlation) and the class prior $P(Y=1)$. Covariance matrices are
  parametrised through standard deviations and a correlation coefficient,
  which guarantees positive semi-definiteness automatically.
- **Theoretical Bayes boundary**: the exact decision boundary implied by
  the population parameters, computed directly — no estimation involved.
- **Simulation**: draw a labelled sample from the population model (sample
  size and reseeding controlled interactively).
- **Estimated boundaries**: Gaussian Naive Bayes, LDA and QDA are fitted to
  the simulated sample and their decision boundaries overlaid on the true
  Bayes-optimal boundary.
- **Toggleable elements**: class density contours, each method's boundary,
  training points, and a shaded decision-region background for any one
  method.
- **Accuracy comparison**: out-of-sample accuracy of each fitted method
  against the theoretical Bayes-optimal ceiling.

## Running Locally

```bash
conda run -n marimo marimo edit "stat312/Gaussian Classifiers/gaussian_classifiers_marimo.py"
```

## Course

STAT312: Advanced Data Analytics — Nelson Mandela University.

## References

Class Notes, Chapter 6: *Classification Methods: Naive Bayes and Linear
Discriminant Analysis* (STAT312, 2026).
