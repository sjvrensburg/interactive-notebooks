# k-Nearest Neighbors Classification

🔍 **Interactive tutorial on k-NN classification algorithms**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Interactive-brightgreen)](https://sjvrensburg.github.io/interactive-notebooks/stat312/k-NN%20Classification/knn_interactive_wasm/)
[![Marimo](https://img.shields.io/badge/Built%20with-Marimo-blue)](https://marimo.io/)

## 🚀 [**Launch Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/k-NN%20Classification/knn_interactive_wasm/)

Experience k-NN classification directly in your browser - no installation required!

## 📚 What You'll Learn

This interactive demonstration covers the fundamental concepts of k-Nearest Neighbors classification:

### 🎯 Core Concepts
- **Algorithm Mechanics**: How k-NN makes predictions using majority voting
- **Distance Metrics**: Understanding Euclidean, Manhattan, and other distance measures
- **Decision Boundaries**: Visualizing how classification regions form
- **Parameter Effects**: How k values affect model behavior and performance

### 🎛️ Interactive Features
- **Real-time Visualization**: Watch decision boundaries change as you adjust parameters
- **Dynamic Data Generation**: Create custom datasets with controllable parameters
- **Parameter Exploration**: Use sliders to experiment with different k values
- **Query Point Prediction**: Click anywhere to see how new points would be classified
- **Performance Analysis**: Compare training vs. testing accuracy with elbow plots

### 🧮 Mathematical Foundation

The k-NN algorithm classifies a new point by:

1. **Distance Calculation**: Compute distance to all training points
   ```
   d(x, xᵢ) = √[(x₁-xᵢ₁)² + (x₂-xᵢ₂)² + ...]
   ```

2. **Neighbor Selection**: Find the k closest training points

3. **Majority Vote**: Assign the most common class among the k neighbors

4. **Decision Rule**: 
   ```
   ŷ = argmax(count of class c among k neighbors)
   ```

## 📖 Learning Objectives

After completing this tutorial, you will understand:

- ✅ How k-NN classification works step-by-step
- ✅ The relationship between k values and model complexity
- ✅ How to visualize and interpret decision boundaries  
- ✅ The bias-variance tradeoff in k-NN models
- ✅ Proper model evaluation using train/test splits
- ✅ When k-NN is appropriate for classification tasks

## 🎓 Educational Structure

The demonstration follows a structured learning path:

1. **📊 Theory Introduction**: Mathematical foundations and key concepts
2. **🔄 Data Generation**: Interactive dataset creation with controllable parameters
3. **🤖 Algorithm Visualization**: Real-time decision boundary plotting
4. **📈 Performance Analysis**: Comprehensive evaluation with elbow plots
5. **🎯 Interactive Prediction**: Hands-on classification of custom points
6. **💡 Key Takeaways**: Summary of important insights and best practices

## 🛠️ Technical Details

**Built with:**
- **Marimo**: Reactive Python notebooks
- **Plotly**: Interactive visualizations
- **Scikit-learn**: k-NN implementation
- **NumPy**: Numerical computations
- **WebAssembly**: Browser-native execution

**Features:**
- Client-side execution (no server required)
- Responsive interactive controls
- Real-time parameter updates
- Progressive Web App (PWA) capabilities

## 🚀 Running Locally

To run this demonstration on your local machine:

```bash
# Navigate to the repository root
cd interactive-notebooks

# Install dependencies
pip install -r requirements.txt

# Run the notebook
marimo run "stat312/k-NN Classification/knn_marimo.py"

# Or edit interactively
marimo edit "stat312/k-NN Classification/knn_marimo.py"
```

## 📱 Browser Compatibility

This demonstration works in all modern web browsers:
- ✅ Chrome/Chromium 90+
- ✅ Firefox 85+
- ✅ Safari 14+
- ✅ Edge 90+

For the best experience, use a desktop or tablet with a screen width of at least 1024px.

## 🎯 Target Audience

**Primary:**
- Statistics and data science students learning classification algorithms
- STAT312 course participants exploring machine learning fundamentals

**Secondary:**
- Educators teaching machine learning concepts
- Practitioners seeking visual intuition for k-NN behavior
- Self-learners exploring classification methods

## 🔗 Related Resources

- **[Non-Parametric Regression Demo](../Non-Parametric%20Regression/)**: Explore k-NN for regression tasks
- **[Main Repository](../../)**: Browse all interactive demonstrations
- **[Marimo Documentation](https://docs.marimo.io/)**: Learn about reactive notebooks
- **[Scikit-learn k-NN Guide](https://scikit-learn.org/stable/modules/neighbors.html)**: Technical documentation

## 📄 License

This educational content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

---

**Ready to explore?** [🚀 **Launch the Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat312/k-NN%20Classification/knn_interactive_wasm/)