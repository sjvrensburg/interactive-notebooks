# Classification and Regression Trees (CART)

🌳 **Interactive exploration of decision trees and cost-complexity pruning**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Interactive-brightgreen)](https://sjvrensburg.github.io/interactive-notebooks/stat420/cart_wasm/)
[![Marimo](https://img.shields.io/badge/Built%20with-Marimo-blue)](https://marimo.io/)

## 🚀 [**Launch Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat420/cart_wasm/)

Explore decision trees and pruning techniques directly in your browser - no installation required!

## 📚 What You'll Learn

This interactive demonstration provides comprehensive coverage of Classification and Regression Trees (CART):

### 🎯 Core Concepts
- **Decision Tree Mechanics**: How trees recursively partition feature space
- **Impurity Measures**: Understanding Gini impurity for optimal splits
- **Tree Growth**: Controlling model complexity through maximum depth
- **Cost-Complexity Pruning**: Balancing accuracy against model simplicity

### 🎛️ Interactive Features
- **Dynamic Tree Growth**: Adjust maximum depth to control complexity
- **Interactive Pruning**: Use α parameter to remove weak branches in real-time
- **Tree Structure Visualisation**: See the complete tree structure with Mermaid diagrams
- **Decision Boundary Plots**: Visualise how trees partition 2D feature space
- **Train/Test Evaluation**: Compare performance metrics before and after pruning
- **Browser Zoom Support**: Zoom in/out on tree diagrams for detailed inspection

### 🧮 Mathematical Foundation

**Gini Impurity** (for classification):
```
Gini(t) = 1 - Σᵢ₌₁ᶜ pᵢ²
```
where pᵢ is the proportion of class i in node t.

**Cost-Complexity Pruning:**
```
Rα(T) = R(T) + α|T|
```
Where:
- R(T) is the misclassification rate
- |T| is the number of terminal nodes (complexity)
- α ≥ 0 is the complexity parameter controlling pruning strength

**Key Principles:**
- **α = 0**: No pruning (full tree up to max depth)
- **Small α**: Minimal pruning, removes only very weak branches
- **Large α**: Aggressive pruning, produces simpler trees
- **Optimal α**: Balances test accuracy and model simplicity

## 📖 Learning Objectives

After completing this tutorial, you will understand:

- ✅ How decision trees make predictions through recursive partitioning
- ✅ The role of impurity measures in determining optimal splits
- ✅ How tree depth controls model complexity and overfitting
- ✅ The mechanics of cost-complexity pruning (weakest-link pruning)
- ✅ The bias-variance trade-off in tree-based models
- ✅ How pruning improves generalisation performance
- ✅ When to use decision trees vs. linear classifiers

## 🎓 Educational Structure

The demonstration follows a comprehensive learning path:

1. **📚 Theory Introduction**: Mathematical foundations of CART algorithms
2. **🔄 Dataset Generation**: Two-moons synthetic data for non-linear boundaries
3. **🌱 Tree Growth**: Interactive control of maximum tree depth
4. **✂️ Cost-Complexity Pruning**: Real-time pruning with α parameter
5. **📊 Tree Visualisation**: Complete tree structure with Mermaid diagrams
6. **🎨 Decision Boundaries**: Visualise feature space partitioning
7. **📈 Performance Analysis**: Train/test accuracy comparison
8. **💡 Key Takeaways**: Summary of best practices and extensions

## 🎨 Visualisation Features

**Interactive Components:**
- **Horizontal Slider**: Controls maximum tree depth (1-10 levels)
- **Vertical Slider**: Controls pruning strength (α from 0.0 to 0.1)
- **Mermaid Tree Diagrams**: Full tree structure with split conditions and impurity
- **Decision Boundary Plots**: 2D feature space with training/test data
- **Real-time Metrics**: Training accuracy, test accuracy, and node count
- **Browser Zoom**: Zoom in/out on tree diagrams using browser controls (Ctrl/Cmd + scroll)

**Visual Conventions:**
- Filled markers: Training data points
- Hollow markers: Test data points
- Background colours: Predicted class regions
- Blue/Orange: Class 0 and Class 1

## 🛠️ Technical Details

**Built with:**
- **Marimo**: Reactive Python notebooks enabling seamless interactivity
- **Plotly**: Advanced interactive visualisations with hover effects
- **Scikit-learn**: Professional-grade decision tree implementation
- **NumPy**: Efficient numerical computing for meshgrid generation
- **Mermaid.js**: Tree structure visualisation
- **WebAssembly**: Client-side execution for responsive interactions

**Advanced Features:**
- Train/test data splitting with stratification
- Two-moons synthetic dataset for non-linear boundaries
- Cost-complexity pruning path calculation
- Interactive tree structure generation
- Real-time decision boundary rendering
- Progressive Web App (PWA) capabilities for offline use

## 🚀 Running Locally

To run this demonstration on your local machine:

```bash
# Navigate to the repository root
cd interactive-notebooks

# Install dependencies
pip install -r requirements.txt

# Run the notebook
marimo run "stat420/cart_pruning_marimo.py"

# Or edit interactively
marimo edit "stat420/cart_pruning_marimo.py"
```

## 📊 Key Insights You'll Discover

**Tree Growth Effects:**
- **Shallow trees (depth 1-3)**: Underfit the data, simple boundaries
- **Medium trees (depth 4-6)**: Balance complexity and generalisation
- **Deep trees (depth 7-10)**: May overfit, complex irregular boundaries

**Pruning Effects:**
- **α = 0.0**: Full unpruned tree with maximum complexity
- **α = 0.01-0.03**: Light pruning removes weakest branches
- **α = 0.05-0.10**: Heavy pruning produces simplified trees
- **Sweet spot**: Find α where test accuracy peaks

**Bias-Variance Trade-off:**
- **Unpruned deep trees**: Low bias, high variance (overfitting)
- **Heavily pruned trees**: Higher bias, lower variance (underfitting)
- **Optimal pruning**: Balances both for best generalisation

## 📱 Browser Compatibility

This demonstration works in all modern web browsers:
- ✅ Chrome/Chromium 90+
- ✅ Firefox 85+
- ✅ Safari 14+
- ✅ Edge 90+

For optimal experience, use a desktop or tablet with a screen width of at least 1200px to accommodate tree diagrams and decision boundaries.

## 🎯 Target Audience

**Primary:**
- Statistics students studying classification algorithms
- STAT420 course participants exploring tree-based methods
- Data science students learning supervised learning techniques

**Secondary:**
- Machine learning practitioners comparing model complexity
- Educators teaching decision tree concepts
- Researchers exploring non-linear classification methods

## 🔗 Related Resources

- **[k-NN Classification Demo](../stat312/k-NN%20Classification/)**: Explore instance-based learning
- **[Non-Parametric Regression Demo](../stat312/Non-Parametric%20Regression/)**: Compare local regression approaches
- **[Main Repository](../)**: Browse all interactive demonstrations
- **[The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)**: Comprehensive statistical learning textbook (Chapter 9)
- **[CART Paper (Breiman et al.)](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)**: Original CART methodology
- **[Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)**: Technical documentation
- **[Marimo Documentation](https://docs.marimo.io/)**: Learn about reactive notebooks

## 📚 Recommended Reading

**Before the Demo:**
- Review basic classification concepts and evaluation metrics
- Understand entropy and information gain concepts
- Familiarise yourself with overfitting and regularisation

**After the Demo:**
- Explore ensemble methods (Random Forests, Gradient Boosting)
- Study feature importance and variable selection in trees
- Investigate regression trees for continuous outcomes
- Learn about conditional inference trees (CTree)

## 💡 Practical Applications

**When to Use Decision Trees:**
- Non-linear relationships between features and outcome
- Mixed data types (categorical and continuous features)
- Need for interpretable models
- Feature interactions are important
- Robust to outliers required

**When to Consider Alternatives:**
- Linear relationships dominate
- High-dimensional sparse data
- Need for probabilistic predictions
- Computational efficiency critical for very large datasets

## 📄 Licence

This educational content is licenced under [CC BY-SA 4.0](https://creativecommons.org/licences/by-sa/4.0/).

---

**Ready to explore decision trees?** [🚀 **Launch the Interactive Demo**](https://sjvrensburg.github.io/interactive-notebooks/stat420/cart_wasm/)