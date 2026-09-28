# PCA as a Series of Rotations

**Animated visualisation of principal component analysis as turning a 3D data cloud until its longest axes line up with the coordinate axes.**

🌐 **[Live Demo](https://sjvrensburg.github.io/interactive-notebooks/stat312/PCA%20Rotations/pca_rotations_wasm/)** - Run the notebook directly in your browser!

## 🎯 What You'll Learn

- **PCA is a rotation**: finding the principal components is the same as turning the standardised data cloud until its longest direction (PC1) lies along the first axis, the next longest (PC2) along the second, and the last (PC3) along the third
- **Scores are new coordinates**: after the turning, each point's coordinates are its principal component scores
- **Nothing is lost by turning**: the cloud is never stretched or squashed, so the total variance stays at λ₁ + λ₂ + λ₃ = p
- **The components are uncorrelated**: the covariance matrix starts as the correlation matrix **R** and ends with zeros off the diagonal
- **Variance equals eigenvalue**: the diagonal ends as λ₁, λ₂, λ₃
- **Dimension reduction**: dropping PC3 flattens the cloud onto the PC1–PC2 plane while keeping most of the variance

## 📊 Key Features

### Animated 3D Rotation

The turning is done in three simple turns, each about one coordinate axis:

| Turn | What it achieves |
|:---:|:---|
| 1 | PC1 is swung round to point the same way as Z₁ (just tilted up or down) |
| 2 | PC1 is tipped onto Z₁ — **PC1 found** |
| 3 | The cloud spins about PC1 until PC2 lies on Z₂ — **PC2 and PC3 found** |
| (optional) | PC3 is dropped: the cloud flattens onto the PC1–PC2 plane |

- **⏭ Next step** plays one turn and pauses on a caption saying what has just been found
- **▶ Play all** runs straight through; **❚❚ Pause** and **⏮ Reset** as expected
- **What turns?**: watch the points turn onto fixed axes, or the axes turn onto the fixed cloud
- Points are coloured by their PC1 score so individual observations can be followed, and dashed lines show where the moving arrows will end up

### Covariance Heatmap

Animated alongside the 3D plot: it starts as the correlation matrix **R**, loses its off-diagonal entries turn by turn, and ends as diag(λ₁, λ₂, λ₃).

### Interactive Controls
- **Correlations** between the three standardised variables Z₁, Z₂, Z₃ (impossible combinations are replaced by the closest possible one, with a warning)
- **Sample size** n
- **Speed** of the animation

### Summary
- "What to notice" points linking the animation to the key properties of principal components
- Eigenvalues, proportion and cumulative proportion of variance, and the eigenvector weights

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
marimo edit "stat312/PCA Rotations/pca_rotations_marimo.py"

# View-only mode
marimo run "stat312/PCA Rotations/pca_rotations_marimo.py"
```

## 🔬 Mathematical Foundation

The data are standardised, so PCA works from the correlation matrix, with eigenvectors **v**ⱼ (directions) and eigenvalues λⱼ (variances along those directions):

$$\mathbf{R}\mathbf{v}_j = \lambda_j\mathbf{v}_j, \qquad \lambda_1 \ge \lambda_2 \ge \lambda_3 .$$

The score of observation *i* on component *j* is $\text{PC}_{ij} = \mathbf{z}_i^T\mathbf{v}_j$. Collecting the eigenvectors as the columns of **V**, all the scores together are $\mathbf{V}^T\mathbf{z}_i$. **V** is a rotation, and the animation builds it from three turns about the coordinate axes (a form of Euler angles, computed with Givens rotations; the signs of the eigenvectors are chosen so that the total turning is as small as possible).

## 📝 Educational Context

### STAT312: Advanced Data Analytics

This tool accompanies the STAT312 notes on principal component analysis (Chapter 10), in particular the "rugby ball" geometric intuition and the key properties of principal components.

### Learning Objectives
- Interpret PCA geometrically as turning the data cloud (or, equivalently, the axes)
- See principal component scores as coordinates on the new axes
- Connect the diagonalised covariance matrix to uncorrelated components and eigenvalues
- Understand what is lost when the smallest-variance component is dropped

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: 3D scatter animation with frames, plus an animated covariance heatmap
- **NumPy**: Eigendecomposition and rotations (no scikit-learn dependency)

The Plotly figure is embedded in an iframe so that its animation frames are rebuilt every time a control changes, and a small script provides the step-by-step playback.
