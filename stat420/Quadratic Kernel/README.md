# Quadratic Kernel Ridge Regression

**Interactive walkthrough of the kernel trick made visible** — verify $\kappa(a,b)=(1+a'b)^2$ against its explicit feature map, replay a fully-worked example by hand, and watch the map bend a circle flat in 3D.

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat420/Quadratic%20Kernel/quadratic_kernel_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

This tool accompanies §4.2–4.4 of the STAT420 kernel ridge regression notes (a companion to Exterkate, Groenen, Heij & van Dijk, 2016):

- **The kernel trick**: why $\kappa(a,b)=(1+a'b)^2$ equals $\varphi(a)'\varphi(b)$ for the feature map $\varphi(a)=(1,\sqrt2 a_1,\dots,a_1^2,\dots,\sqrt2 a_1a_2,\dots)$, without ever building $\varphi$
- **The computational saving**: $O(N)$ via the kernel vs $O(N^2)$ via the explicit feature map, quantified for any $N$
- **The representer property in action**: a forecast is a weighted sum of similarities to training points, $\hat y_*=\sum_t\hat\alpha_t\kappa(x_t,x_*)$
- **What the feature map actually does geometrically**: equal-radius points collapse onto a flat plane; same-sign-pattern points separate across another plane — the reason a *linear* method in feature space can fit a *curved* boundary in input space

## 📊 Key Features

### 🧮 Kernel Trick (§4.2)
- Pick $N$ and a random seed; two points $a,b\in\R^N$ are drawn and $\kappa(a,b)$ is computed both via the explicit feature map and the closed form — the two always agree
- A log-scale cost plot of $2N+1$ vs $N^2+3N+1$ operations per kernel-matrix entry, with the paper's own $N=132$ case marked

### 📐 Worked Example (§4.3)
- The notes' hand-worked $N{=}2, T{=}3$ pipeline — three fixed training points, editable $\lambda$, $y$-values, and a movable query point $x_*$
- Live-updated $\K$, $\K+\lambda\I$, $\hat{\boldsymbol\alpha}$, $\bk_*$, and the forecast $\hat y_*$
- Defaults reproduce the notes' answer $\hat y_*=43/120\approx0.3583$ exactly
- A 2D plot where edge width/opacity encodes how much each training point "votes" on the forecast

### 📦 Feature Geometry (§4.4)
- The pure-quadratic block $\varphi_3(a)=(a_1^2,a_2^2,\sqrt2\,a_1a_2)$ plotted in 3D alongside the original 2D input space
- Move a point by radius and angle and watch it land on the plane $q_1+q_2=r^2$ (radius becomes a flat coordinate) and see its side of the plane $q_3=0$ track the sign of $a_1a_2$

### 🎯 Putting It to Work
- Ridge regression with a **linear** kernel ($\kappa(a,b)=a'b$) vs the **quadratic** kernel on a circular target pattern, same $\lambda$, same solver
- The linear kernel is stuck with a boundary through the origin; the quadratic kernel traces the circle — a direct payoff of the §4.4 geometry

## 🚀 Running Locally

```bash
conda run -n marimo marimo edit "stat420/Quadratic Kernel/quadratic_kernel_marimo.py"

# Or view-only
conda run -n marimo marimo run "stat420/Quadratic Kernel/quadratic_kernel_marimo.py"
```

## 🔬 Mathematical Foundation

For $a,b\in\R^N$, the quadratic kernel is

$$\kappa(a,b) = (1+a'b)^2 = \varphi(a)'\varphi(b), \qquad
\varphi(a) = \bigl(1,\ \sqrt2 a_1,\dots,\sqrt2 a_N,\ a_1^2,\dots,a_N^2,\ \sqrt2 a_1a_2,\dots,\sqrt2 a_{N-1}a_N\bigr)'.$$

The $\sqrt2$ factors exist because $(\sum_n u_n)^2=\sum_n u_n^2+2\sum_{n<m}u_nu_m$ generates every cross term twice — tagging each cross-feature with $\sqrt2$ makes the square collapse back to $(1+a'b)^2$ exactly. The kernel ridge forecast

$$\hat y_* = \bk_*'(\K+\lambda\I_T)^{-1}\y, \qquad K_{st}=\kappa(x_s,x_t),\ (\bk_*)_t=\kappa(x_t,x_*)$$

is algebraically identical to ordinary ridge regression run on the $M=\binom{N+2}{2}$ explicit features, but costs $T\times T$ instead of $M\times M$ to invert — and never needs $\varphi$ written down.

## 📝 Educational Context

### STAT420: Quantitative Data Analysis

This notebook accompanies the kernel ridge regression material, demonstrating:
- How the kernel trick turns an intractable $M\times M$ problem into a tractable $T\times T$ one
- The representer property: fitted coefficients always lie in the span of the training feature vectors
- Why polynomial kernels buy nonlinearity without ever constructing the polynomial features explicitly
- The geometric reason a linear method in feature space can recover a curved boundary in input space

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: 2D and 3D interactive visualisations (subplots mixing `xy` and `scene` types)
- **NumPy**: Feature map, kernel matrix, and kernel ridge regression solved directly — no external ML dependency
