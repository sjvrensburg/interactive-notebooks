# PCA as a Series of Rotations

**Animated visualisation of principal component analysis as a rotation of a 3D data cloud onto its principal axes.**

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat312/PCA%20Rotations/pca_rotations_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

This interactive tool shows that **PCA is a change of coordinates by a rotation**, and breaks that rotation into three simple steps:

- **PCA as a rotation**: the scores z = V′x are obtained with an orthogonal matrix V, which is a pure rotation once the eigenvector signs give det V = +1
- **Givens rotations**: any 3D rotation factorises as V′ = G₃G₂G₁, where each Gₖ turns the space within a single coordinate plane
- **Diagonalising the covariance**: each rotation removes correlation, until Cov(z) = Λ = diag(λ₁, λ₂, λ₃)
- **Active vs passive views**: rotating the data onto the axes and rotating the axes onto the data give the same coordinates
- **Dimension reduction**: dropping PC3 flattens the cloud onto the PC1–PC2 plane with minimal loss of variance

## 📊 Key Features

### Animated 3D Rotation

A Plotly animation with **▶ Play / ❚❚ Pause** buttons and a scrubber:

| Step | Plane | Rotates about | What happens |
|:---:|:---:|:---:|:---|
| G₁ | x–y | z-axis | PC1 swings into the x–z plane |
| G₂ | x–z | y-axis | PC1 tips down onto the x-axis |
| G₃ | y–z | x-axis | the cloud spins about PC1 until PC2 lies on the y-axis |
| (optional) | — | — | PC3 is dropped: the cloud flattens onto the PC1–PC2 plane |

Points are coloured by their PC1 score so individual observations can be tracked, and dashed "target" arrows show where the moving arrows will land.

### Covariance Heatmap

The covariance matrix of the current coordinates, MSM′, is animated next to the 3D plot:

- After **G₂**, the first row and column are zero apart from λ₁
- After **G₃**, the matrix is diagonal: the principal component variances
- The trace (total variance) never changes

### Interactive Controls
- **Standard deviations** of x, y, z and **correlations** ρ(x, y), ρ(x, z), ρ(y, z) — invalid combinations are replaced by the nearest valid correlation matrix, with a warning
- **Sample size** n
- **Point of view**: rotate the data (active) or rotate the axes (passive)
- **Projection**: optionally finish by dropping PC3
- **Frames per rotation** and **frame duration** to control smoothness and speed

### Numerical Summary
- Sample covariance S, eigenvector matrix V and det V
- Eigenvalues and percentage of variance explained
- The three rotation angles θ₁, θ₂, θ₃

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
marimo edit "stat312/PCA Rotations/pca_rotations_marimo.py"

# View-only mode
marimo run "stat312/PCA Rotations/pca_rotations_marimo.py"
```

## 🔬 Mathematical Foundation

The sample covariance of the centred data has eigendecomposition

$$\mathbf{S} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}', \qquad \lambda_1 \ge \lambda_2 \ge \lambda_3,$$

and the principal component scores are $\mathbf{z}_i = \mathbf{V}'\mathbf{x}_i$, with $\operatorname{Cov}(\mathbf{z}) = \mathbf{V}'\mathbf{S}\mathbf{V} = \boldsymbol{\Lambda}$.

**Sign convention.** Eigenvectors are defined only up to sign. The notebook tries each sign for $\mathbf{v}_1$ and $\mathbf{v}_2$, sets $\mathbf{v}_3 = \mathbf{v}_1 \times \mathbf{v}_2$ so that $\det\mathbf{V} = +1$, and keeps the choice that needs the least total rotation.

**Givens factorisation.** A Givens rotation $\mathbf{G}(i, j, \theta)$ rotates by θ in the $(i, j)$ coordinate plane. Applying three of them to V zeros its below-diagonal entries (a QR decomposition):

$$\theta_1 = \operatorname{atan2}(V_{21}, V_{11}),\quad \theta_2 = \operatorname{atan2}(V_{31}, V_{11}'),\quad \theta_3 = \operatorname{atan2}(V_{32}', V_{22}')$$

(primes denote entries after the previous rotations). The result is orthogonal, upper triangular, with positive diagonal and determinant +1, so it is the identity:

$$\mathbf{G}_3\mathbf{G}_2\mathbf{G}_1\mathbf{V} = \mathbf{I} \quad\Longrightarrow\quad \mathbf{V}' = \mathbf{G}_3\mathbf{G}_2\mathbf{G}_1 .$$

The animation interpolates each angle from 0 to θₖ in turn. With M the rotation applied so far, the plotted points are $\mathbf{M}\mathbf{x}_i$ (active view) and the heatmap shows $\mathbf{M}\mathbf{S}\mathbf{M}'$.

## 📝 Educational Context

### STAT312: Advanced Data Analytics

This tool accompanies the STAT312 material on principal component analysis, demonstrating:
- Why PCA preserves distances and total variance
- How the principal components decorrelate the variables
- The link between eigenvectors, rotations and Euler angles
- What is lost when the smallest-variance component is discarded

### Learning Objectives
- Interpret PCA geometrically as a rotation of the coordinate system
- Read the eigenvector matrix V as a rotation and relate it to the principal directions
- Follow the covariance matrix as it becomes diagonal
- Distinguish rotating the data from rotating the axes

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: 3D scatter animation with frames, plus an animated covariance heatmap
- **NumPy**: Eigendecomposition and Givens rotations (no scikit-learn dependency)

The Plotly figure is embedded in an iframe so that its animation frames are rebuilt every time a control changes.
