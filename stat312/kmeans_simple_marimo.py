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
        # K-Means Clustering Tutorial
        
        Welcome to this **interactive K-Means clustering tutorial**! This notebook demonstrates:
        
        1. **Generate synthetic data** with configurable parameters
        2. **Randomly initialize centroids** with a button click
        3. **Run the K-Means algorithm** step-by-step
        4. **Visualize results** with convex hulls
        
        ## What is K-Means Clustering?
        
        **K-Means** partitions data into $k$ clusters by minimizing within-cluster sum of squares.
        
        ### Algorithm Steps:
        1. **Random Initialization**: Select $k$ random initial centroids
        2. **Assignment**: Assign each point to the nearest centroid
        3. **Update**: Recalculate centroids as cluster means
        4. **Convergence**: Repeat until centroids stabilize
        
        ### Mathematical Objective:
        
        $$\min_{C_1,\ldots,C_k} \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$$
        """
    )
    return


@app.cell
def __():
    # Import all necessary libraries
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import ConvexHull
    
    return (
        ConvexHull,
        KMeans,
        make_blobs,
        go,
        np,
        pd,
        px,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Data Generation Controls""")
    return


@app.cell
def __(mo):
    # UI controls for data generation
    n_true_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=3, label="True number of clusters"
    )
    
    n_samples_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200, label="Number of data points"
    )
    
    cluster_std_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.2, label="Cluster separation (std)"
    )

    mo.md(
        f"""
        **Generate data:**
        
        {n_true_clusters_slider}
        
        {n_samples_slider}
        
        {cluster_std_slider}
        """
    )
    return n_true_clusters_slider, n_samples_slider, cluster_std_slider


@app.cell
def __(make_blobs, n_samples_slider, n_true_clusters_slider, cluster_std_slider, pd):
    # Generate synthetic dataset
    X_data, y_true_data = make_blobs(
        n_samples=n_samples_slider.value,
        centers=n_true_clusters_slider.value,
        cluster_std=cluster_std_slider.value,
        center_box=(-10, 10)
    )
    
    # Create DataFrame
    df = pd.DataFrame(X_data, columns=['x', 'y'])
    df['true_cluster'] = y_true_data
    
    return df


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🎯 K-Means Configuration
        
        Press the button to randomly initialize centroids and run the algorithm!
        """
    )
    return


@app.cell
def __(mo):
    # K-Means configuration
    k_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=3, label="Number of clusters (k)"
    )
    
    max_iterations_slider = mo.ui.slider(
        start=5, stop=30, step=1, value=15, label="Maximum iterations"
    )
    
    # Button to run K-Means
    run_button = mo.ui.button(
        label="🚀 Run K-Means with Random Centroids",
        value=False
    )

    mo.md(
        f"""
        **Configuration:**
        
        {k_clusters_slider}
        
        {max_iterations_slider}
        
        {run_button}
        """
    )
    return k_clusters_slider, max_iterations_slider, run_button


@app.cell
def __(run_button, k_clusters_slider, max_iterations_slider, df, KMeans, np):
    # State management - only run when button is pressed
    kmeans = None
    labels = []
    final_centroids = []
    ari = 0
    iterations = 0
    
    if not run_button.value:
        status_md = "Press the button above to run K-Means!"
    else:
        # Prepare data
        _X = df[['x', 'y']].values
        _y_true = df['true_cluster'].values
        
        # Run K-Means with random initialization
        kmeans = KMeans(
            n_clusters=k_clusters_slider.value,
            init='random',  # Random initialization
            n_init=1,        # Single run with random start
            max_iter=max_iterations_slider.value,
            random_state=None,  # No seed for true randomness
        )
        
        # Fit the model
        kmeans.fit(_X)
        
        # Get results
        labels = kmeans.labels_
        final_centroids = kmeans.cluster_centers_
        iterations = kmeans.n_iter_
        
        # Calculate metrics
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(_y_true, labels)
        
        status_md = f"""
        ## 📊 K-Means Results
        
        **Performance:**
        - **Iterations to converge:** {iterations}
        - **Adjusted Rand Index:** {ari:.3f} (1.0 = perfect match)
        - **Final Centroids:** {len(final_centroids)}
        
        **Final Centroid Locations:**
        {chr(10).join([f"Centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})" for i, c in enumerate(final_centroids)])}
        """
    
    return status_md, kmeans, labels, final_centroids, ari, iterations


@app.cell
def __(status_md, mo):
    # Display the K-Means results
    mo.md(status_md)
    return


@app.cell
def __(kmeans, labels, df, ConvexHull, go, mo, np, px):
    # Create visualization with convex hulls
    
    if kmeans is None:
        viz_md = mo.md("⚠️ No K-Means results to display.")
    else:
        # Get original data
        X = df[['x', 'y']].values
        
        # Create figure
        fig = go.Figure()
        
        # Define colors for each cluster
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
        
        # Add convex hulls for each cluster
        for cluster_id in np.unique(labels):
            # Get points for this cluster
            cluster_points = X[labels == cluster_id]
            
            # Only add hull if we have enough points
            if len(cluster_points) >= 3:
                try:
                    # Calculate convex hull
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    
                    # Close the polygon by adding the first point at the end
                    hull_points_closed = np.vstack([hull_points, hull_points[0]])
                    
                    # Convert color to RGB for fillcolor
                    color_hex = px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)]
                    color_rgb = px.colors.hex_to_rgb(color_hex)
                    
                    # Add hull trace
                    fig.add_trace(go.Scatter(
                        x=hull_points_closed[:, 0],
                        y=hull_points_closed[:, 1],
                        mode='lines',
                        fill='toself',
                        fillcolor=f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.2)',
                        line=dict(color=colors[cluster_id % len(colors)], width=2),
                        name=f'Cluster {cluster_id}',
                        showlegend=False
                    ))
                except:
                    # Skip hull if calculation fails
                    pass
        
        # Add data points colored by cluster
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=[colors[i % len(colors)] for i in labels],
                opacity=0.8
            ),
            name='Data Points'
        ))
        
        # Add centroids
        fig.add_trace(go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(
                size=20,
                color='black',
                symbol='x',
                line=dict(width=3, color='white')
            ),
            name='Centroids'
        ))
        
        fig.update_layout(
            title=f'K-Means Clustering Results (k={len(kmeans.cluster_centers_)})',
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            height=600,
            showlegend=True,
            legend=dict(
                items=[
                    dict(label='Data Points', marker=dict(size=8)),
                    dict(label='Centroids', marker=dict(symbol='x', size=20)),
                ]
            )
        )
        
        viz_md = mo.ui.plotly(fig)
    
    return viz_md


if __name__ == "__main__":
    app.run()