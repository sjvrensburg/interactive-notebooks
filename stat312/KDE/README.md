# Kernel Density Estimation

🧮 **Interactive tutorial on KDE techniques**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Interactive-brightgreen)](https://sjvrensburg.github.io/interactive-notebooks/stat312/KDE/kde_wasm/)
[![Marimo](https://img.shields.io/badge/Built%20with-Marimo-blue)](https://marimo.io/)

## 🚀 [**Launch Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/KDE/kde_wasm/)

Experience kernel density estimation directly in your browser - no installation required!

## 📚 What You'll Learn

This interactive demonstration covers kernel density estimation fundamentals:

### 🎯 Core Concepts

- **Density Estimation**: Non-parametric PDF approximation
- **Kernel Functions**: Smoothing tools for data distributions
- **Bandwidth Selection**: Bias-variance trade-off control
- **Smoothing Effects**: Parameter impacts on density shape

### 🎛️ Interactive Features

- **Real-time Visualization**: Density updates with parameter changes
- **Dynamic Data Generation**: Sample from various distributions
- **Kernel Comparison**: Side-by-side kernel experiments
- **Bandwidth Tuning**: Slider-based smoothing adjustments
- **Weight Visualization**: Kernel contributions at points
- **Performance Metrics**: Bias, variance, and error analysis

### 🧮 Mathematical Foundation

Kernel density estimator:
```
f̂(x) = (1/(n h)) Σ K((x - xᵢ)/h)   for i=1 to n
```

**Common Kernels:**
- **Gaussian**: K(u) = (1/√(2π)) exp(-u²/2)
- **Epanechnikov**: K(u) = (3/4)(1 - u²) for |u| ≤ 1
- **Triangular**: K(u) = (1 - |u|) for |u| ≤ 1
- **Uniform**: K(u) = 1/2 for |u| ≤ 1

## 📖 Learning Objectives

After this tutorial:
- ✅ Principles of non-parametric density estimation
- ✅ Kernel smoothing of empirical distributions
- ✅ Bandwidth's role in estimation
- ✅ Kernel choice trade-offs
- ✅ Bias-variance in KDE
- ✅ KDE vs. parametric methods

## 🎓 Educational Structure

1. **📚 Theory Overview**: Density estimation basics
2. **🔄 Data Generation**: Interactive sampling
3. **🔧 Kernel Functions**: Visual comparisons
4. **🎛️ Bandwidth Exploration**: Smoothing effects
5. **📈 Density Visualization**: Estimated vs. true densities
6. **⚖️ Performance Analysis**: Quality metrics

## 🎨 Visualization Features

**Interactive Plots:**
- **Density Curves**: Estimated and true functions
- **Histogram Comparison**: KDE vs. histograms
- **Kernel Shapes**: Individual contributions
- **Bandwidth Effects**: Real-time updates
- **Point-wise Analysis**: Density at locations
- **Error Metrics**: Bias/variance calculations

## 🛠️ Technical Details

**Built with:**
- **Marimo**: Reactive notebooks
- **Plotly**: Interactive visualizations
- **Scikit-learn**: KDE implementation
- **NumPy & SciPy**: Numerical functions
- **WebAssembly**: Client-side execution

**Features:**
- Distribution sampling
- Bandwidth selection
- Cross-validation
- PWA support

## 🚀 Running Locally

```bash
cd interactive-notebooks
pip install -r requirements.txt
marimo run "stat312/KDE/kde_marimo.py"
# Or edit: marimo edit "stat312/KDE/kde_marimo.py"
```

## 📱 Browser Compatibility

- ✅ Chrome/Chromium 90+
- ✅ Firefox 85+
- ✅ Safari 14+
- ✅ Edge 90+

Optimal: Desktop/tablet, ≥1200px width.

## 🎯 Target Audience

**Primary:**
- Statistics students (non-parametric methods)
- STAT312 participants (density estimation)
- Data scientists (distribution analysis)

**Secondary:**
- Educators (probability teaching)
- Researchers (empirical distributions)
- Practitioners (EDA)

## 🔗 Related Resources

- **[k-NN Classification](../k-NN%20Classification/)**
- **[Non-Parametric Regression](../Non-Parametric%20Regression/)**
- **[Main Repository](../../)**
- **[KDE Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation)**
- **[Marimo Docs](https://docs.marimo.io/)**

## 📄 License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

**Ready to explore KDE?** [🚀 **Launch the Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/KDE/kde_wasm/)