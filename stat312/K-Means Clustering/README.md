# K-Means Clustering Evolution Tutorial

**Interactive demonstration of the K-Means clustering algorithm** - Watch clusters form and evolve step by step!

## 🎯 What You'll Learn

This interactive tutorial demonstrates the **K-Means clustering algorithm** through visual, step-by-step evolution:

- **Random Initialization**: How initial centroids are randomly placed
- **Assignment Step**: Points assigned to their nearest centroids
- **Update Step**: Centroids moved to the mean of their assigned points
- **Convergence**: When and how the algorithm stabilizes
- **Cluster Evolution**: How clusters form and boundaries solidify over iterations

## 📊 Key Features

### Interactive Controls
- **True Number of Clusters**: Configure the ground truth structure
- **Number of Data Points**: Adjust dataset size (50-500 points)
- **Cluster Separation**: Control how distinct clusters are
- **k Parameter**: Set the number of clusters to find
- **Maximum Iterations**: Limit the evolution process

### Visualization
- **Iteration Slider**: Step through each iteration manually
- **Convex Hulls**: Semi-transparent cluster boundaries at each step
- **Centroid Tracking**: Watch centroids move with black X markers
- **ARI Score**: See how well discovered clusters match ground truth
- **Color-Coded Points**: Clear visual distinction between clusters

## 🚀 Running Locally

```bash
# Interactive mode (recommended)
marimo edit "stat312/K-Means Clustering/kmeans_marimo.py"

# View-only mode
marimo run "stat312/K-Means Clustering/kmeans_marimo.py"
```

## 📈 Algorithm Steps Visualized

1. **Iteration 0**: Random initial centroid placement
2. **Early Iterations**: Watch clusters begin to form
3. **Middle Iterations**: See boundaries adjust and stabilize
4. **Final Iterations**: Observe convergence to stable configuration

## 🔬 Mathematical Foundation

**K-Means Objective**:
$$\min \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

where:
- $k$ = number of clusters
- $C_i$ = cluster $i$
- $\mu_i$ = centroid of cluster $i$

**Algorithm**:
1. Initialize $k$ centroids randomly
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Move each centroid to mean of assigned points
4. **Repeat** steps 2-3 until convergence

## 📝 Educational Context

### STAT312: Advanced Data Analytics

This tutorial is part of the STAT312 course, demonstrating:
- Unsupervised learning fundamentals
- Clustering algorithm mechanics
- Initialization sensitivity
- Convergence behavior
- Performance evaluation (Adjusted Rand Index)

### Learning Objectives
- Understand the iterative nature of K-Means
- Recognize the importance of initialization
- Visualize how clusters evolve over time
- Learn when the algorithm converges
- Evaluate clustering quality with ARI

## 🛠️ Technical Details

**Built with**:
- **Marimo**: Reactive Python notebooks
- **Plotly**: Interactive visualizations
- **scikit-learn**: K-Means implementation
- **NumPy/SciPy**: Numerical computing and convex hulls

**Performance Metrics**:
- **Adjusted Rand Index (ARI)**: Measures similarity to ground truth (-1 to 1, 1 = perfect)
- **Iteration Count**: Tracks convergence speed
- **Visual Inspection**: Convex hulls show cluster shape evolution
