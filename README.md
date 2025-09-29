# Interactive Statistical Learning Notebooks

🎓 **Interactive demonstrations for STAT312 (Statistical Learning)** - Built with [Marimo](https://marimo.io/)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demos-blue)](https://sjvrensburg.github.io/interactive-notebooks/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Live Interactive Demonstrations

Explore these interactive statistical learning concepts directly in your browser:

### 🔍 [k-Nearest Neighbors Classification](https://sjvrensburg.github.io/interactive-notebooks/stat312/k-NN%20Classification/knn_interactive_wasm/)

**Learn the fundamentals of k-NN classification through interactive exploration:**

- 📊 Visualize decision boundaries in real-time
- 🎛️ Adjust k values and see immediate effects
- 📈 Understand bias-variance tradeoffs
- 🎯 Make predictions on custom data points
- 📋 Compare training vs. testing performance

[📖 View Documentation](./stat312/k-NN%20Classification/README.md) | [💻 Run Locally](./stat312/k-NN%20Classification/knn_marimo.py)

### 📊 [Kernel Density Estimation](https://sjvrensburg.github.io/interactive-notebooks/stat312/KDE/kde_wasm/)

**Learn the fundamentals of KDE through interactive exploration:**

- 📊 Visualise kernel density estimates in real-time
- 🎛️ Adjust kernel bandwidths and see immediate effects
- 📈 Understand how kernels smooth out data distributions
- 🎯 Estimate densities at specific points
- 📋 Explore various kernel types like Gaussian, Epanechnikov etc.

[📖 View Documentation](./stat312/KDE/README.md) | [💻 Run Locally](./stat312/KDE/knn_marimo.py)

### 📈 [Non-Parametric Regression](https://sjvrensburg.github.io/interactive-notebooks/stat312/Non-Parametric%20Regression/nonparam_regression_wasm/)

**Explore kernel methods and non-parametric regression techniques:**

- 🔧 Compare k-NN vs. Nadaraya-Watson regression
- 🎚️ Experiment with different kernel functions
- 📐 Adjust bandwidth parameters interactively
- 🎪 Visualize kernel weights and local smoothing
- 📊 Evaluate model performance on test data

[📖 View Documentation](./stat312/Non-Parametric%20Regression/README.md) | [💻 Run Locally](./stat312/Non-Parametric%20Regression/nonparam_regression_marimo.py)

## 🎯 Learning Objectives

These interactive notebooks help you:

- **Visualize complex algorithms** through dynamic, real-time demonstrations
- **Understand parameter effects** by adjusting values and seeing immediate results
- **Connect theory to practice** with mathematical formulations and hands-on exploration
- **Develop intuition** for machine learning concepts through interactive experimentation
- **Evaluate model performance** using proper train/test evaluation methods

## 🛠️ Technology Stack

- **[Marimo](https://marimo.io/)**: Reactive Python notebooks for interactive data science
- **[Plotly](https://plotly.com/)**: Interactive visualizations that work seamlessly in browsers
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
marimo export html notebook.py --include-code -o output_directory/
```

## 📁 Repository Structure

```
interactive-notebooks/
├── stat312/                              # Course-specific demonstrations
│   ├── k-NN Classification/
│   │   ├── knn_marimo.py                # Interactive k-NN tutorial
│   │   ├── knn_interactive_wasm/        # WASM export for GitHub Pages
│   │   └── README.md                    # Demo documentation
│   └── Non-Parametric Regression/
│   |   ├── nonparam_regression_marimo.py # Kernel regression tutorial
│   |   ├── nonparam_regression_wasm/     # WASM export for GitHub Pages
│   |   └── README.md                     # Demo documentation
|   └── KDE/
│       ├── kde_marimo.py                 # Kernel Density Estimation tutorial
│       ├── kde_wasm/                     # WASM export for GitHub Pages
│       └── README.md                     # Demo documentation
├── requirements.txt                      # Python dependencies
├── WARP.md                              # Development guide for AI assistants
└── README.md                            # This file
```

## 🎓 Educational Context

**Course**: STAT312 (Statistical Learning)  
**Focus**: Interactive exploration of machine learning fundamentals

**Key Topics**:

- k-Nearest Neighbors (Classification & Regression)
- Non-parametric regression methods
- Kernel methods and bandwidth selection  
- Bias-variance tradeoff visualization
- Cross-validation and model evaluation
- Decision boundary analysis

**Target Audience**:

- Statistics and data science students
- Machine learning practitioners
- Educators teaching statistical concepts
- Self-learners exploring ML algorithms

## 📄 License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit and indicate if changes were made
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

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
