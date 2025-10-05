import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # K-Means Clustering Interactive Tutorial
        
        Welcome to this interactive tutorial on **K-Means Clustering**! This notebook provides a step-by-step visualization of how the k-means algorithm works, allowing you to see the algorithm converge in real-time.
        
        ## What is K-Means Clustering?
        
        **K-Means** is an unsupervised learning algorithm that partitions data into $k$ distinct clusters by minimizing the within-cluster sum of squares. The algorithm iteratively assigns data points to the nearest centroid and updates centroids based on the mean of assigned points.
        
        ### The Algorithm Steps:
        
        1. **Initialization**: Randomly select $k$ initial centroids
        2. **Assignment Step**: Assign each data point to the nearest centroid
        3. **Update Step**: Recalculate centroids as the mean of assigned points
        4. **Convergence**: Repeat steps 2-3 until centroids stop changing significantly
        
        ### Key Concepts:
        
        - **Centroid**: The center point of a cluster (mean of all points in that cluster)
        - **Within-cluster variance**: Sum of squared distances from points to their centroid
        - **Convergence**: When assignments no longer change between iterations
        
        ### Mathematical Objective:
        
        $$\min_{C_1,\ldots,C_k} \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$
        
        where $C_i$ is cluster $i$, $\mu_i$ is its centroid, and $\| \cdot \|$ is Euclidean distance.
        """
    )
    return


@app.cell
def __():
    # Import all necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from scipy.spatial import ConvexHull
    
    # Set random seed for reproducibility
    np.random.seed(42)
    return (
        ConvexHull,
        KMeans,
        go,
        make_blobs,
        make_subplots,
        np,
        plt,
        px,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Interactive Data Generation""")
    return


@app.cell
def __(mo):
    # UI controls for data generation
    n_true_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=3, label="True number of clusters (N_true):"
    )
    
    n_samples_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200, label="Number of data points:"
    )
    
    cluster_std_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.0, label="Cluster separation (std):"
    )
    
    random_state_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=42, label="Random seed:"
    )

    mo.md(
        f"""
        **Adjust the data generation parameters:**
        
        {n_true_clusters_slider}
        
        {n_samples_slider}
        
        {cluster_std_slider}
        
        {random_state_slider}
        """
    )
    return (
        cluster_std_slider,
        n_samples_slider,
        n_true_clusters_slider,
        random_state_slider,
    )


@app.cell
def __(cluster_std_slider, make_blobs, n_samples_slider, n_true_clusters_slider, random_state_slider, np):
    # Generate synthetic dataset based on UI controls
    def generate_data():
        X, y_true = make_blobs(
            n_samples=n_samples_slider.value,
            centers=n_true_clusters_slider.value,
            cluster_std=cluster_std_slider.value,
            random_state=random_state_slider.value,
            center_box=(-10, 10)
        )
        return X, y_true

    X, y_true = generate_data()
    return generate_data, X, y_true


@app.cell
def __(go, X, y_true, mo, np):
    # Toggle for coloring by true labels vs cluster assignments
    show_true_labels_toggle = mo.ui.checkbox(
        value=False, label="Color by true cluster labels (vs. current assignments)"
    )
    
    mo.md(
        f"""
        {show_true_labels_toggle}
        
        **Generated Dataset & K-Means Clustering:**
        """
    )
    
    return (
        show_true_labels_toggle,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🎯 K-Means Algorithm Setup""")
    return


@app.cell
def __(mo):
    # K-Means parameters
    k_clusters_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=3, label="Number of clusters (k):"
    )
    
    max_iterations_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=20, label="Maximum iterations:"
    )

    mo.md(
        f"""
        **Configure K-Means parameters:**
        
        {k_clusters_slider}
        
        {max_iterations_slider}
        """
    )
    return k_clusters_slider, max_iterations_slider


@app.cell
def __(KMeans, X, k_clusters_slider, max_iterations_slider, np):
    # Pre-calculate K-Means iterations
    def calculate_kmeans_iterations(X_data, k, max_iter):
        """Run K-Means and store all iteration states"""
        
        # Initialize with random centroids (not k-means++) for better visualization
        np.random.seed(42)  # For reproducible initial centroids
        random_indices = np.random.choice(len(X_data), k, replace=False)
        centroids = [X_data[random_indices].copy()]
        
        # Manual iteration to store all states
        current_centroids = centroids[0]
        converged = False
        
        for iteration in range(max_iter):
            # Assignment step
            distances = np.linalg.norm(X_data[:, np.newaxis] - current_centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update step
            new_centroids = np.array([
                X_data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else current_centroids[i]
                for i in range(k)
            ])
            
            centroids.append(new_centroids.copy())
            
            # Check for convergence
            if np.allclose(current_centroids, new_centroids, atol=1e-4):
                converged = True
                
            current_centroids = new_centroids
        
        # Continue with same centroids if converged early to fill max iterations
        if converged:
            remaining_iters = max_iter - len(centroids) + 1
            for _ in range(remaining_iters):
                centroids.append(current_centroids.copy())
        
        # Calculate all labels for each iteration
        all_labels = []
        for centroid_set in centroids:
            distances = np.linalg.norm(X_data[:, np.newaxis] - centroid_set, axis=2)
            labels = np.argmin(distances, axis=1)
            all_labels.append(labels)
        
        return centroids, all_labels

    # Calculate all iteration states
    kmeans_centroids, kmeans_labels = calculate_kmeans_iterations(
        X, k_clusters_slider.value, max_iterations_slider.value
    )
    
    return (
        calculate_kmeans_iterations,
        kmeans_centroids,
        kmeans_labels,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🎬 Iterative Visualization""")
    return


@app.cell
def __(kmeans_centroids, mo):
    # Iteration control slider
    iteration_slider = mo.ui.slider(
        start=0,
        stop=len(kmeans_centroids) - 1,
        step=1,
        value=0,
        label=f"K-Means Iteration (0-{len(kmeans_centroids) - 1}):"
    )
    
    # Show convex hull toggle
    show_convex_hull_toggle = mo.ui.checkbox(
        value=False, label="Show Convex Hull"
    )

    mo.md(
        f"""
        **Control the visualization:**
        
        {iteration_slider}
        
        {show_convex_hull_toggle}
        """
    )
    return iteration_slider, show_convex_hull_toggle


@app.cell
def __(ConvexHull, X, go, iteration_slider, kmeans_centroids, kmeans_labels, show_convex_hull_toggle, np, px, show_true_labels_toggle, y_true):
    # Dynamic plot update based on iteration
    def plot_kmeans_iteration(iteration_idx, show_hull=False, show_true_labels=False):
        fig = go.Figure()
        
        # Get current state
        current_labels = kmeans_labels[iteration_idx]
        current_centroids = kmeans_centroids[iteration_idx]
        k = len(current_centroids)
        
        # Determine coloring
        if show_true_labels:
            # Use true cluster labels for coloring
            plot_labels = y_true
            n_true_clusters = len(np.unique(y_true))
            colors = px.colors.qualitative.Plotly[:n_true_clusters]
            title_suffix = "(True Labels)"
        else:
            # Use current cluster assignments for coloring
            plot_labels = current_labels
            colors = px.colors.qualitative.Set1[:k]
            title_suffix = "(Current Assignments)"
        
        # Plot data points
        if show_true_labels:
            # Color by true clusters
            for cluster_id in np.unique(plot_labels):
                mask = plot_labels == cluster_id
                if np.sum(mask) > 0:
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[cluster_id],
                            opacity=0.7
                        ),
                        name=f'True Cluster {cluster_id}',
                        hovertemplate=f'True Cluster {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                    ))
        else:
            # Color by current cluster assignments and optionally show convex hulls
            for cluster_id in range(k):
                mask = plot_labels == cluster_id
                if np.sum(mask) > 0:
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[cluster_id],
                            opacity=0.7
                        ),
                        name=f'Cluster {cluster_id}',
                        hovertemplate=f'Cluster {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                    ))
                    
                    # Add convex hull if enabled and cluster has enough points
                    if show_hull and np.sum(mask) >= 3:
                        try:
                            cluster_points = X[mask]
                            hull = ConvexHull(cluster_points)
                            hull_points = cluster_points[hull.vertices]
                            hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                            
                            # Convert hex color to rgba for fill
                            hex_color = colors[cluster_id].lstrip('#')
                            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)'
                            
                            fig.add_trace(go.Scatter(
                                x=hull_points[:, 0],
                                y=hull_points[:, 1],
                                mode='lines',
                                line=dict(color=colors[cluster_id], width=2),
                                fill='toself',
                                fillcolor=fill_color,
                                name=f'Hull {cluster_id}',
                                hoverinfo='skip'
                            ))
                        except Exception:
                            pass  # Skip hull if calculation fails
        
        # Plot centroids
        fig.add_trace(go.Scatter(
            x=current_centroids[:, 0],
            y=current_centroids[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            hovertemplate='Centroid<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # Add centroid labels
        for i, centroid in enumerate(current_centroids):
            fig.add_annotation(
                x=centroid[0],
                y=centroid[1],
                text=f'C{i}',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black'
            )
        
        fig.update_layout(
            title=f'K-Means Clustering - Iteration {iteration_idx} {title_suffix}',
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            showlegend=True,
            width=800,
            height=600,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        return fig

    # Create plot for current iteration
    current_iteration_fig = plot_kmeans_iteration(
        iteration_slider.value, 
        show_convex_hull_toggle.value,
        show_true_labels_toggle.value
    )
    
    return (
        current_iteration_fig,
        plot_kmeans_iteration,
    )


@app.cell
def __(current_iteration_fig, mo):
    # Display the interactive plot
    current_iteration_fig
    return





@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📊 Algorithm Analysis
        
        ### What to Observe:
        
        1. **Iteration 0**: Initial random placement of centroids
        2. **Early Iterations**: Rapid reorganization as points are reassigned
        3. **Middle Iterations**: Fine-tuning of cluster boundaries
        4. **Final Iterations**: Convergence when assignments stabilize
        
        ### Key Insights:
        
        - **Centroid Movement**: Watch how centroids move toward the center of their assigned clusters
        - **Boundary Formation**: Notice how decision boundaries emerge between clusters
        - **Convergence**: The algorithm typically converges quickly (5-15 iterations)
        - **Local Optima**: Different random seeds can lead to different final clusters
        
        ### When the Algorithm Struggles:
        
        - **Different sized clusters**: K-means assumes equal-sized spherical clusters
        - **Non-globular shapes**: Elongated or irregular clusters are poorly handled
        - **Outliers**: Single points can significantly shift centroids
        - **Wrong k value**: Too few or too many clusters lead to poor assignments
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🧮 Mathematical Details
        
        ### Distance Calculation:
        
        The algorithm uses **Euclidean distance** between points and centroids:
        
        $$d(x, c_i) = \sqrt{\sum_{j=1}^p (x_j - c_{ij})^2}$$
        
        where $x$ is a data point, $c_i$ is centroid $i$, and $p$ is the number of features.
        
        ### Assignment Rule:
        
        $$\text{Assign } x \text{ to cluster } i \text{ where } i = \arg\min_{i} d(x, c_i)$$
        
        ### Centroid Update:
        
        $$c_i^{new} = \frac{1}{|C_i|} \sum_{x \in C_i} x$$
        
        where $C_i$ is the set of points currently assigned to cluster $i$.
        
        ### Convergence Criterion:
        
        The algorithm converges when:
        $$\|c_i^{new} - c_i^{old}\| < \epsilon \text{ for all } i$$
        
        where $\epsilon$ is a small tolerance (typically $10^{-4}$).
        """
    )
    return