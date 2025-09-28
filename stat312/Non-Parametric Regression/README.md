# Non-Parametric Regression Methods

📈 **Interactive exploration of kernel-based regression techniques**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Interactive-brightgreen)](https://sjvrensburg.github.io/interactive-notebooks/stat312/Non-Parametric%20Regression/nonparam_regression_wasm/)
[![Marimo](https://img.shields.io/badge/Built%20with-Marimo-blue)](https://marimo.io/)

## 🚀 [**Launch Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/Non-Parametric%20Regression/nonparam_regression_wasm/)

Explore non-parametric regression methods directly in your browser - no installation required!

## 📚 What You'll Learn

This interactive demonstration provides comprehensive coverage of non-parametric regression techniques:

### 🎯 Core Concepts
- **Non-Parametric Philosophy**: Learning without assuming specific functional forms
- **Local Smoothing**: How nearby observations influence predictions
- **Kernel Methods**: Mathematical weighting schemes for local neighborhoods
- **Bandwidth Selection**: Controlling the smoothness-flexibility tradeoff

### 🎛️ Interactive Features
- **Method Comparison**: Side-by-side visualization of k-NN vs. Nadaraya-Watson regression
- **Dynamic Function Generation**: Create datasets with sine waves, polynomials, or step functions
- **Kernel Exploration**: Experiment with Gaussian, Epanechnikov, Triangular, and Uniform kernels
- **Parameter Tuning**: Real-time adjustment of k values and bandwidth parameters
- **Weight Visualization**: See exactly how kernel weights are applied to training data
- **Performance Analysis**: Compare methods using proper train/test evaluation

### 🧮 Mathematical Foundation

**k-Nearest Neighbors Regression:**
```
f̂(x₀) = (1/k) Σ yᵢ  for i ∈ Nₖ(x₀)
```
Where Nₖ(x₀) represents the k nearest neighbors of x₀.

**Nadaraya-Watson Kernel Regression:**
```
f̂(x₀) = Σᵢ₌₁ⁿ K((x₀-xᵢ)/h) yᵢ / Σᵢ₌₁ⁿ K((x₀-xᵢ)/h)
```
Where:
- K(·) is the kernel function
- h is the bandwidth parameter
- The weights decrease with distance from x₀

**Common Kernel Functions:**
- **Gaussian**: K(u) = (1/√2π) exp(-u²/2)
- **Epanechnikov**: K(u) = (3/4)(1-u²) for |u| ≤ 1
- **Triangular**: K(u) = (1-|u|) for |u| ≤ 1
- **Uniform**: K(u) = 1/2 for |u| ≤ 1

## 📖 Learning Objectives

After completing this tutorial, you will understand:

- ✅ The fundamental principles of non-parametric regression
- ✅ How k-NN regression creates step-wise predictions
- ✅ How kernel regression produces smooth, continuous estimates
- ✅ The role of bandwidth in controlling model flexibility
- ✅ Trade-offs between different kernel functions
- ✅ Proper model evaluation and comparison techniques
- ✅ When to choose non-parametric over parametric methods

## 🎓 Educational Structure

The demonstration follows a comprehensive learning path:

1. **📚 Theory Overview**: Introduction to non-parametric regression principles
2. **🔄 Data Generation**: Interactive creation of various functional relationships
3. **👥 k-NN Regression**: Understanding local averaging with nearest neighbors
4. **🔧 Kernel Functions**: Mathematical foundations and visual comparisons
5. **🎯 Nadaraya-Watson**: Smooth regression with weighted local averaging
6. **⚖️ Method Comparison**: Side-by-side evaluation of both approaches
7. **💡 Best Practices**: Guidelines for parameter selection and practical use

## 🎨 Visualization Features

**Interactive Plots Include:**
- **Function Fitting**: Compare predicted curves with true underlying functions
- **Training/Test Data**: Distinguish between model training and evaluation sets
- **Query Point Analysis**: Examine how predictions are made at specific locations
- **Kernel Weight Visualization**: See the relative influence of each training point
- **Performance Metrics**: Real-time MSE and R² calculations
- **Parameter Sensitivity**: Observe how changing parameters affects model behavior

## 🛠️ Technical Details

**Built with:**
- **Marimo**: Reactive Python notebooks enabling seamless interactivity
- **Plotly**: Advanced interactive visualizations with hover effects
- **Scikit-learn**: Professional-grade k-NN implementation
- **NumPy & SciPy**: Efficient numerical computing and distance calculations
- **WebAssembly**: Client-side execution for responsive interactions

**Advanced Features:**
- Train/test data splitting with stratification
- Cross-validation for robust performance estimation
- Multiple kernel function implementations
- Efficient neighborhood search algorithms
- Progressive Web App (PWA) capabilities for offline use

## 🚀 Running Locally

To run this demonstration on your local machine:

```bash
# Navigate to the repository root
cd interactive-notebooks

# Install dependencies
pip install -r requirements.txt

# Run the notebook
marimo run "stat312/Non-Parametric Regression/nonparam_regression_marimo.py"

# Or edit interactively
marimo edit "stat312/Non-Parametric Regression/nonparam_regression_marimo.py"
```

## 📊 Key Insights You'll Discover

**Parameter Effects:**
- **Small k/bandwidth**: More flexible, can capture local patterns but may overfit
- **Large k/bandwidth**: Smoother predictions but may miss important features
- **Optimal values**: Balance between bias and variance for best generalization

**Method Characteristics:**
- **k-NN**: Step-wise predictions, robust to outliers, simple interpretation
- **Kernel Methods**: Smooth continuous functions, differentiable, flexible weighting

**Practical Considerations:**
- **Computational Complexity**: How methods scale with dataset size
- **Curse of Dimensionality**: Performance in high-dimensional spaces
- **Boundary Effects**: Behavior near the edges of the data range

## 📱 Browser Compatibility

This demonstration works in all modern web browsers:
- ✅ Chrome/Chromium 90+
- ✅ Firefox 85+
- ✅ Safari 14+
- ✅ Edge 90+

For optimal experience, use a desktop or tablet with a screen width of at least 1200px to accommodate side-by-side comparisons.

## 🎯 Target Audience

**Primary:**
- Advanced statistics students studying non-parametric methods
- STAT312 course participants exploring regression techniques
- Graduate students in data science or machine learning

**Secondary:**
- Practitioners comparing local regression approaches
- Educators teaching statistical learning concepts  
- Researchers working with complex, non-linear relationships

## 🔗 Related Resources

- **[k-NN Classification Demo](../k-NN%20Classification/)**: Explore k-NN for classification tasks
- **[Main Repository](../../)**: Browse all interactive demonstrations
- **[The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)**: Comprehensive statistical learning textbook
- **[Kernel Regression Wikipedia](https://en.wikipedia.org/wiki/Kernel_regression)**: Mathematical background
- **[Marimo Documentation](https://docs.marimo.io/)**: Learn about reactive notebooks

## 📚 Recommended Reading

**Before the Demo:**
- Review basic regression concepts and least squares estimation
- Understand the bias-variance tradeoff in statistical learning
- Familiarize yourself with cross-validation techniques

**After the Demo:**
- Explore advanced kernel methods (local polynomial regression)
- Study bandwidth selection algorithms (cross-validation, plug-in methods)
- Investigate multi-dimensional kernel regression applications

## 📄 License

This educational content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

---

**Ready to explore non-parametric regression?** [🚀 **Launch the Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/Non-Parametric%20Regression/nonparam_regression_wasm/)