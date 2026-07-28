# Ridge & LASSO Geometry

**Interactive visualisation of why LASSO produces sparse solutions and Ridge does not** — watch the RSS contour touch each constraint region!

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat312/Ridge%20and%20LASSO%20Geometry/ridge_lasso_geometry_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

This interactive tool demonstrates the **geometric origin of the difference between Ridge regression and LASSO** in coefficient (β) space:

- **Constrained optimisation**: Both methods minimise RSS subject to a budget constraint on the coefficients
- **RSS contours**: Nested ellipses centred at the OLS estimate β̂, stretched along the collinearity axis
- **Constraint regions**: Ridge uses an ℓ₂ ball (a circle); LASSO uses an ℓ₁ ball (a diamond)
- **The contact point**: The solution is where the expanding RSS ellipse first touches the constraint region
- **Why LASSO selects variables**: The diamond's corners lie on the axes, so contact frequently lands on a corner — producing an **exactly zero** coefficient

## 📊 Key Features

### Side-by-Side Constraint Geometry

Two panels share the same coefficient space:

- **Left (LASSO)**: the ℓ₁ diamond constraint region with the contact ellipse and solution
- **Right (Ridge)**: the ℓ₂ circular constraint region with the contact ellipse and solution
- Nested grey RSS contours show the objective function's level sets
- The bold red contour is the level set passing through the solution — the first ellipse to touch the constraint

### Coefficient Paths

A second tab plots β̂₁ and β̂₂ against λ for both methods:

- **Ridge** paths are smooth — coefficients shrink toward zero asymptotically but **never reach it**
- **LASSO** paths are piecewise linear — they **hit exactly zero** and stay there, performing variable selection
- A vertical line marks the current λ

### Interactive Controls
- **Correlation (ρ)**: Adjust collinearity between X₁ and X₂ — higher ρ stretches the ellipses and makes sparsity more likely
- **OLS β̂₁, β̂₂**: Set the centre of the RSS contours
- **Penalty (λ)**: Move from nearly-OLS (λ ≈ 0) to heavily regularised
- **Display toggles**: Show/hide the OLS point, nested contours, and labels

### Solutions Panel
- Side-by-side coefficient table for OLS, Ridge, and LASSO
- Equivalent constraint budget (t₂ vs t₁) and shrinkage ratio for each method
- Count of coefficients zeroed by LASSO at the current λ

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
conda run -n marimo marimo edit "stat312/Ridge and LASSO Geometry/ridge_lasso_geometry_marimo.py"

# View-only mode
conda run -n marimo marimo run "stat312/Ridge and LASSO Geometry/ridge_lasso_geometry_marimo.py"
```

## 🔬 Mathematical Foundation

Both methods solve a penalised problem, equivalent (for some budget t) to a constrained one:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}}\;(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}_{\text{ols}})' \mathbf{X}'\mathbf{X}\,(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}_{\text{ols}})$$

$$\text{subject to}\quad\begin{cases}\|\boldsymbol{\beta}\|_2 \le t & \text{Ridge}\\[2pt] \|\boldsymbol{\beta}\|_1 \le t & \text{LASSO}\end{cases}$$

The objective's level sets are **ellipses** centred at β̂_ols. The constraint defines a region (circle for Ridge, diamond for LASSO). The constrained optimum is the **first point of contact** between an expanding ellipse and the constraint region.

**Ridge** (closed form):
$$\hat{\boldsymbol{\beta}}^{\text{ridge}} = (\mathbf{X}'\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}'\mathbf{y}$$

The ℓ₂ ball is smooth, so contact is on an arc — coefficients shrink but remain non-zero.

**LASSO** (coordinate descent with soft-thresholding):
$$\beta_j \leftarrow \frac{S\!\left(\rho_j,\ \lambda\right)}{A_{jj}}, \qquad S(x,\gamma) = \operatorname{sign}(x)\max(|x|-\gamma, 0)$$

The ℓ₁ ball has corners on the coordinate axes. When an ellipse touches a corner, the corresponding coefficient is exactly zero.

## 📝 Educational Context

### STAT312: Advanced Data Analytics

This tool accompanies the STAT312 material on regularisation, demonstrating:
- The geometric meaning of the ℓ₁ vs ℓ₂ penalty
- Why LASSO performs variable selection while Ridge does not
- How collinearity between regressors shapes the RSS ellipses and influences sparsity
- The shrinkage-vs-selection trade-off across the regularisation path

### Learning Objectives
- Connect the algebraic penalty formulations to their geometric constraint regions
- Visualise the contact-point argument that explains LASSO sparsity
- Explore how correlation between predictors interacts with the penalty to drive coefficients to zero
- Read coefficient-path plots as a record of variable selection over λ

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: Interactive 2D visualisations (side-by-side subplots + coefficient paths)
- **NumPy**: Ridge via closed form; LASSO via coordinate descent with soft-thresholding (no external solver dependency)
