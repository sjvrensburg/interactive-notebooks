# Interactive Statistical Learning Notebooks

🎓 **Interactive demonstrations for Advanced Data Analytics, Linear Models, and Quantitative Data Analysis** - Built with [Marimo](https://marimo.io/)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demos-blue)](https://sjvrensburg.github.io/interactive-notebooks/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Live Interactive Demonstrations

Explore these interactive statistical learning concepts directly in your browser:

### STAT312: Advanced Data Analytics

#### 🔍 [k-Nearest Neighbours Classification](https://sjvrensburg.github.io/interactive-notebooks/stat312/k-NN%20Classification/knn_interactive_wasm/)

**Learn the fundamentals of k-NN classification through interactive exploration:**

- 📊 Visualise decision boundaries in real-time
- 🎛️ Adjust k values and see immediate effects
- 📈 Understand bias-variance trade-offs
- 🎯 Make predictions on custom data points
- 📋 Compare training vs. testing performance

[📖 View Documentation](./stat312/k-NN%20Classification/README.md) | [💻 Run Locally](./stat312/k-NN%20Classification/knn_marimo.py)

#### 📊 [Kernel Density Estimation](https://sjvrensburg.github.io/interactive-notebooks/stat312/KDE/kde_wasm/)

**Learn the fundamentals of KDE through interactive exploration:**

- 📊 Visualise kernel density estimates in real-time
- 🎛️ Adjust kernel bandwidths and see immediate effects
- 📈 Understand how kernels smooth out data distributions
- 🎯 Estimate densities at specific points
- 📋 Explore various kernel types like Gaussian, Epanechnikov etc.

[📖 View Documentation](./stat312/KDE/README.md) | [💻 Run Locally](./stat312/KDE/knn_marimo.py)

#### 📈 [Non-Parametric Regression](https://sjvrensburg.github.io/interactive-notebooks/stat312/Non-Parametric%20Regression/nonparam_regression_wasm/)

**Explore kernel methods and non-parametric regression techniques:**

- 🔧 Compare k-NN vs. Nadaraya-Watson regression
- 🎚️ Experiment with different kernel functions
- 📐 Adjust bandwidth parameters interactively
- 🎪 Visualise kernel weights and local smoothing
- 📊 Evaluate model performance on test data

[📖 View Documentation](./stat312/Non-Parametric%20Regression/README.md) | [💻 Run Locally](./stat312/Non-Parametric%20Regression/nonparam_regression_marimo.py)

#### 🎯 [K-Means Clustering](https://sjvrensburg.github.io/interactive-notebooks/stat312/K-Means%20Clustering/kmeans_wasm/)

**Watch K-Means clustering evolve step-by-step:**

- 🔄 Step through algorithm iterations with interactive slider
- 🎨 Visualise cluster formation with convex hulls
- 📍 Track centroid movement across iterations
- 🎲 Control random initialization with seed parameter
- 📊 Monitor convergence with Adjusted Rand Index (ARI)
- ⚙️ Adjust k, sample size, and cluster separation

[📖 View Documentation](./stat312/K-Means%20Clustering/README.md) | [💻 Run Locally](./stat312/K-Means%20Clustering/kmeans_marimo.py)

### STAT321: Linear Models and Time Series Analysis

#### 📐 [OLS Geometry Explorer](https://sjvrensburg.github.io/interactive-notebooks/stat321/OLS%20Geometry/ols_geometry_wasm/)

**Explore the geometric interpretation of Ordinary Least Squares in 3D:**

- 📊 Visualise the column space, projection, and residuals in observation space
- 🎛️ Edit regressor and response vectors interactively
- 📐 Confirm orthogonality of residuals with right-angle markers
- 🧮 Inspect projection and residual-maker matrices with property verification
- 📈 Track β̂, ŷ, SSR, and R² in real time

[📖 View Documentation](./stat321/OLS%20Geometry/README.md) | [💻 Run Locally](./stat321/OLS%20Geometry/ols_geometry_marimo.py)

### STAT420: Quantitative Data Analysis

#### 🌳 [Classification and Regression Trees (CART)](https://sjvrensburg.github.io/interactive-notebooks/stat420/cart_wasm/)

**Explore decision trees and cost-complexity pruning:**

- 🌱 Control tree growth with maximum depth parameter
- ✂️ Interactively prune trees using α parameter
- 📊 Visualise complete tree structure with Mermaid diagrams
- 🎨 Explore non-linear decision boundaries
- 📈 Understand bias-variance trade-offs through pruning
- 🔍 Zoom in/out on tree diagrams for detailed inspection

[📖 View Documentation](./stat420/README.md) | [💻 Run Locally](./stat420/cart_pruning_marimo.py)

## 🎯 Learning Objectives

These interactive notebooks help you:

- **Visualise complex algorithms** through dynamic, real-time demonstrations
- **Understand parameter effects** by adjusting values and seeing immediate results
- **Connect theory to practice** with mathematical formulations and hands-on exploration
- **Develop intuition** for machine learning concepts through interactive experimentation
- **Evaluate model performance** using proper train/test evaluation methods

## 🛠️ Technology Stack

- **[Marimo](https://marimo.io/)**: Reactive Python notebooks for interactive data science
- **[Plotly](https://plotly.com/)**: Interactive visualisations that work seamlessly in browsers
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning algorithms and utilities
- **[NumPy](https://numpy.org/) & [SciPy](https://scipy.org/)**: Scientific computing foundations
- **WebAssembly (WASM)**: Client-side execution for responsive interactions

## 🔧 Local Development

### Prerequisites

- Python 3.12 or higher
- Git for version control

### Setup

```bash
# Clone the repository
git clone https://github.com/sjvrensburg/interactive-notebooks.git
cd interactive-notebooks

# Install dependencies
pip install -r requirements.txt

# Verify marimo installation
marimo --help
```

### Running Notebooks Locally

```bash
# Run k-NN Classification demo
marimo run "stat312/k-NN Classification/knn_marimo.py"

# Run Non-Parametric Regression demo
marimo run "stat312/Non-Parametric Regression/nonparam_regression_marimo.py"

# Edit a notebook interactively
marimo edit "stat312/k-NN Classification/knn_marimo.py"
```

### Exporting to WASM

```bash
# Export a notebook to standalone WASM application
marimo export html-wasm notebook.py -o output_directory/
```

## 📁 Repository Structure

```
interactive-notebooks/
├── stat312/                              # STAT312: Advanced Data Analytics
│   ├── k-NN Classification/
│   │   ├── knn_marimo.py                # Interactive k-NN tutorial
│   │   ├── knn_interactive_wasm/        # WASM export for GitHub Pages
│   │   └── README.md                    # Demo documentation
│   ├── KDE/
│   │   ├── kde_marimo.py                # Kernel Density Estimation tutorial
│   │   ├── kde_wasm/                    # WASM export for GitHub Pages
│   │   └── README.md                    # Demo documentation
│   ├── Non-Parametric Regression/
│   │   ├── nonparam_regression_marimo.py # Kernel regression tutorial
│   │   ├── nonparam_regression_wasm/     # WASM export for GitHub Pages
│   │   └── README.md                     # Demo documentation
│   └── K-Means Clustering/
│       ├── kmeans_marimo.py             # K-Means evolution tutorial
│       ├── kmeans_wasm/                 # WASM export for GitHub Pages
│       └── README.md                    # Demo documentation
├── stat321/                              # STAT321: Linear Models and Time Series
│   └── OLS Geometry/
│       ├── ols_geometry_marimo.py        # OLS geometry explorer
│       ├── ols_geometry_wasm/            # WASM export for GitHub Pages
│       └── README.md                     # Demo documentation
├── stat420/                              # STAT420: Quantitative Data Analysis
│   ├── cart_pruning_marimo.py           # Decision trees and pruning tutorial
│   ├── cart_wasm/                       # WASM export for GitHub Pages
│   └── README.md                        # Demo documentation
├── requirements.txt                      # Python dependencies
├── CLAUDE.md                            # Development guide for AI assistants
└── README.md                            # This file
```

## 🎓 Educational Context

### STAT312: Advanced Data Analytics
**Focus**: Interactive exploration of machine learning fundamentals

**Key Topics**:
- k-Nearest Neighbours (Classification & Regression)
- Non-parametric regression methods
- Kernel density estimation
- Kernel methods and bandwidth selection
- K-Means clustering and unsupervised learning
- Bias-variance trade-off visualisation
- Cross-validation and model evaluation
- Decision boundary analysis

### STAT321: Linear Models and Time Series Analysis
**Focus**: Geometric interpretation of linear regression

**Key Topics**:
- Ordinary Least Squares geometry in observation space
- Projection matrices and residual-maker matrices
- Column space and orthogonal projection
- Centred and uncentred R²
- Pythagorean decomposition of sum of squares

### STAT420: Quantitative Data Analysis
**Focus**: Tree-based methods and model complexity

**Key Topics**:
- Classification and Regression Trees (CART)
- Decision tree construction and splitting criteria
- Gini impurity and information gain
- Cost-complexity pruning
- Overfitting and regularisation techniques
- Model interpretability

**Target Audience**:

- Statistics and data science students
- Machine learning practitioners
- Educators teaching statistical concepts
- Self-learners exploring ML algorithms

## 📄 Licence

This work is licenced under a [Creative Commons Attribution-ShareAlike 4.0 International Licence](https://creativecommons.org/licences/by-sa/4.0/).

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit and indicate if changes were made
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same licence

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs or suggest improvements via [GitHub Issues](https://github.com/sjvrensburg/interactive-notebooks/issues)
- Submit pull requests with enhancements
- Share feedback on the educational content

## 🔗 Links

- **Live Demos**: [https://sjvrensburg.github.io/interactive-notebooks/](https://sjvrensburg.github.io/interactive-notebooks/)
- **Repository**: [https://github.com/sjvrensburg/interactive-notebooks](https://github.com/sjvrensburg/interactive-notebooks)
- **Author's Website**: [https://sjvrensburg.github.io/](https://sjvrensburg.github.io/)
- **Marimo Documentation**: [https://docs.marimo.io/](https://docs.marimo.io/)

---

*Built with ❤️ using [Marimo](https://marimo.io/) for interactive statistical learning*
