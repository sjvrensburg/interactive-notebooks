# FWL Theorem Explorer

**Interactive 3D visualisation of the Frisch-Waugh-Lovell theorem** - Step through the partialling-out process and see how FWL recovers the same coefficients as full OLS!

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat321/fwl-theorem/fwl_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

This interactive tool demonstrates the **FWL theorem** geometrically in 3-observation space (ℝ³):

- **Partialling out**: How projecting both **y** and **X₁** onto **X₂** removes the influence of **X₂**
- **Residualised vectors**: The residuals M_{X₂} **y** and M_{X₂} **X₁** = **X̃₁** are orthogonal to **X₂**
- **FWL regression**: Regressing M_{X₂} **y** on **X̃₁** recovers the same β̂₁ as full OLS
- **Fitted value decomposition**: **ŷ** = P_{X₂} **ŷ** + β̂₁ **X̃₁** shows why FWL works

## 📊 Key Features

### Sequential Stepping (Steps 0–4)

Use the **FWL Step** slider to walk through the theorem:

- **Step 0**: Base OLS — see **y**, **X₁**, **X₂**, **ŷ**, and the residual **e**
- **Step 1**: Project **y** onto **X₂** — see P_{X₂} **y** and the residual M_{X₂} **y**
- **Step 2**: Project **X₁** onto **X₂** — see P_{X₂} **X₁** and the residual M_{X₂} **X₁** = **X̃₁**
- **Step 3**: FWL regression — regress M_{X₂} **y** on **X̃₁** to get β̂₁ **X̃₁**
- **Step 4**: Decompose **ŷ** = P_{X₂} **ŷ** + β̂₁ **X̃₁** — confirming the FWL fitted value is the component of **ŷ** orthogonal to **X₂**

### Two Visualisation Tabs

- **📐 Displaced Path**: Residuals drawn as displaced segments (e.g. M_{X₂} **y** from the tip of P_{X₂} **y** to **y**), showing vector subtraction geometrically
- **🎯 Origin Directions**: All residualised vectors drawn from the origin, making it easy to see directions and orthogonality

Each tab includes its own colour-coded step-by-step explanation that describes what the vectors represent in that particular view.

### Interactive Controls
- **Correlation (ρ)**: Adjust collinearity between **X₁** and **X₂**
- **True β₁, β₂**: Set the true regression coefficients
- **Noise (σ)**: Control the magnitude of the error term
- **Display toggles**: Show/hide axis spines and vector labels

### Statistics Panel
- Coefficient comparison: True, Full OLS, and FWL estimates side-by-side
- Numerical verification that β̂₁ from full OLS equals β̂₁ from FWL
- Collinearity warning when |ρ| > 0.9

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
conda run -n marimo marimo edit "stat321/fwl-theorem/fwl_marimo.py"

# View-only mode
conda run -n marimo marimo run "stat321/fwl-theorem/fwl_marimo.py"
```

## 🔬 Mathematical Foundation

**FWL Theorem**: In the model **y** = **X₁** β₁ + **X₂** β₂ + **ε**, the OLS estimator β̂₁ can equivalently be obtained by:

$$\tilde{\mathbf{y}} = \mathbf{M}_{X_2}\mathbf{y}, \quad \tilde{\mathbf{X}}_1 = \mathbf{M}_{X_2}\mathbf{X}_1$$

$$\hat{\beta}_1 = (\tilde{\mathbf{X}}_1'\tilde{\mathbf{X}}_1)^{-1}\tilde{\mathbf{X}}_1'\tilde{\mathbf{y}}$$

where $\mathbf{M}_{X_2} = \mathbf{I} - \mathbf{X}_2(\mathbf{X}_2'\mathbf{X}_2)^{-1}\mathbf{X}_2'$ is the residual-maker matrix for **X₂**.

**Fitted value decomposition**:
$$\hat{\mathbf{y}} = \mathbf{P}_{X_2}\hat{\mathbf{y}} + \hat{\beta}_1\tilde{\mathbf{X}}_1$$

## 📝 Educational Context

### STAT321: Linear Models and Time Series Analysis

This tool accompanies the STAT321 lecture notes on the FWL theorem, demonstrating:
- Why partialling out yields the same coefficient as full OLS
- The geometric meaning of residualisation as orthogonal projection
- How the fitted value decomposes into components along and orthogonal to **X₂**
- The effect of collinearity on the residualised regressors

### Learning Objectives
- Visualise the FWL theorem as a sequence of orthogonal projections
- Understand partialling out as removing the influence of control variables
- Connect the algebraic FWL result to geometric intuition in observation space
- Explore how correlation between regressors affects the partialling-out process

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: Interactive 3D visualisations
- **NumPy**: Linear algebra computations
