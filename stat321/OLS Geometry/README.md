# OLS Geometry Explorer

**Interactive 3D visualisation of the geometry of Ordinary Least Squares** - See how projection, residuals, and the column space relate in observation space!

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat321/OLS%20Geometry/ols_geometry_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

This interactive tool demonstrates the **geometric interpretation of OLS regression** in 3-observation space (ℝ³):

- **Column Space**: The plane spanned by the regressor vectors **x₁** and **x₂**
- **Projection**: How **ŷ** is the orthogonal projection of **y** onto the column space
- **Residuals**: The residual vector **e** = **y** − **ŷ** is perpendicular to the column space
- **Projection Matrices**: How **P** and **M** = **I** − **P** decompose **y**
- **Orthogonality**: Visual and numerical confirmation that **e** ⊥ C(**X**)

## 📊 Key Features

### Interactive Controls
- **Vector Entry**: Edit the entries of **x₁**, **x₂**, and **y** directly
- **Custom Vector**: Add an arbitrary vector to the plot (e.g., **ȳ·ι** to show the mean vector lies in C(**X**) when there is an intercept)
- **Display Toggles**: Show/hide the column space plane, individual vectors, residuals, right-angle marker, drop-lines, and observation axes

### Visualisation
- **3D Rotation**: Click and drag to rotate; scroll to zoom
- **Colour-Coded Vectors**: Distinct colours for **x₁**, **x₂**, **y**, **ŷ**, and **e**
- **Right-Angle Marker**: Confirms orthogonality of residuals to the column space
- **Drop-Lines**: Project **ŷ** onto each observation axis to read off fitted values

### Summary Tabs
- **Summary**: β̂, ŷ, residuals, SSR, ‖**e**‖, centred and uncentred R², orthogonality check
- **Matrices**: Design matrix **X**, projection matrix **P**, residual-maker **M**, and numerical verification of their properties (symmetry, idempotency, annihilation)

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
marimo edit "stat321/OLS Geometry/ols_geometry_marimo.py"

# View-only mode
marimo run "stat321/OLS Geometry/ols_geometry_marimo.py"
```

## 🔬 Mathematical Foundation

**OLS Estimator**:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X'X})^{-1}\mathbf{X'y}$$

**Projection Matrix**:
$$\mathbf{P_X} = \mathbf{X}(\mathbf{X'X})^{-1}\mathbf{X'}$$

**Fitted Values and Residuals**:
$$\hat{\mathbf{y}} = \mathbf{P_X y}, \quad \mathbf{e} = \mathbf{M_X y} = (\mathbf{I} - \mathbf{P_X})\mathbf{y}$$

**Key Properties**:
- $\mathbf{P_X}$ is symmetric and idempotent
- $\mathbf{M_X X} = \mathbf{0}$ (residuals orthogonal to column space)
- $\mathbf{e'X} = \mathbf{0}$ (orthogonality conditions)

## 📝 Educational Context

### STAT321: Linear Models and Time Series Analysis

This tool accompanies **Section 1.4** of the STAT321 lecture notes on OLS geometry, demonstrating:
- The column space interpretation of linear regression
- Orthogonal projection as the best linear approximation
- The role of **P** and **M** matrices
- Geometric meaning of R² and the Pythagorean decomposition

### Learning Objectives
- Visualise OLS as orthogonal projection in observation space
- Understand why residuals are perpendicular to the column space
- Connect algebraic properties of **P** and **M** to geometric intuition
- Explore how changing **X** or **y** affects the projection

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: Interactive 3D visualisations
- **NumPy**: Linear algebra computations
