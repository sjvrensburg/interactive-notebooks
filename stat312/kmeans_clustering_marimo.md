# K-Means Clustering Interactive Tutorial

This interactive Marimo notebook provides a comprehensive visualization of the k-means clustering algorithm, allowing you to observe how the algorithm converges step-by-step.

## Features

### 🎛️ Interactive Data Generation
- **True number of clusters**: Configure the underlying data structure (2-8 clusters)
- **Number of data points**: Adjust dataset size (50-500 points)
- **Cluster separation**: Control how well-separated the clusters are
- **Random seed**: Ensure reproducible results
- **Toggle labeling**: Switch between true cluster labels and uniform coloring

### 🎯 K-Means Algorithm Configuration
- **Number of clusters (k)**: Set the algorithm's cluster count (2-10)
- **Maximum iterations**: Limit the number of iterations (5-50)
- **Pre-calculated states**: All iteration states are computed in advance for smooth visualization

### 🎬 Iterative Visualization
- **Step-by-step control**: Slider to navigate through algorithm iterations (0 to convergence)
- **Dynamic cluster coloring**: Points colored by current cluster assignment
- **Centroid tracking**: Visual markers for cluster centers with labels
- **Convex hull option**: Toggle to show cluster boundaries

### 📊 Algorithm Analysis
- **Real-time convergence**: Watch how the algorithm stabilizes
- **Centroid movement**: Observe how centers move toward cluster means
- **Boundary formation**: See decision boundaries emerge between clusters
- **Mathematical insights**: Detailed explanation of the underlying math

## Key Learning Objectives

1. **Understanding Initialization**: How k-means++ improves starting positions
2. **Assignment Step**: Points assigned to nearest centroids using Euclidean distance
3. **Update Step**: Centroids recalculated as cluster means
4. **Convergence**: When the algorithm stops changing assignments
5. **Limitations**: When and why k-means struggles with certain data patterns

## Technical Implementation

- **Framework**: Marimo reactive notebooks
- **Libraries**: NumPy, SciPy, scikit-learn, Plotly
- **Visualization**: Interactive 2D scatter plots with hover information
- **State Management**: Pre-computed iteration states for smooth interactivity
- **Convex Hull**: Optional boundary visualization using scipy.spatial

## Usage Instructions

1. **Generate Data**: Adjust the data generation parameters and observe the initial dataset
2. **Configure K-Means**: Set the number of clusters and maximum iterations
3. **Iterate Through Algorithm**: Use the iteration slider to step through the algorithm
4. **Enable Convex Hull**: Toggle to visualize cluster boundaries
5. **Analyze Convergence**: Observe how the algorithm stabilizes

## Educational Value

This notebook is designed for statistics and machine learning students to:
- Develop intuition for unsupervised learning
- Understand iterative optimization algorithms
- Visualize mathematical concepts in action
- Explore algorithm limitations and edge cases
- Practice parameter tuning and interpretation

## Dependencies

- `marimo>=0.16.0`
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`
- `plotly>=5.0.0`