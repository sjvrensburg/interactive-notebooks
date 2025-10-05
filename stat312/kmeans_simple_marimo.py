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
        # K-Means Clustering Evolution Tutorial
        
        Watch the **step-by-step evolution** of the K-Means algorithm! This notebook demonstrates:
        
        1. **Generate synthetic data** with configurable parameters
        2. **Run K-Means iteratively** with manual iteration capture
        3. **Step through iterations** with an interactive slider
        4. **Visualize cluster evolution** with convex hulls at each step
        
        ## What You'll See
        
        - **Random Initialization**: Initial random centroid placement
        - **Assignment Step**: Points assigned to nearest centroids
        - **Update Step**: Centroids moved to cluster means
        - **Convergence**: When centroids stop changing significantly
        - **Evolution**: How clusters grow and stabilize over iterations
        
        ### Algorithm Steps Visualized:
        
        1. **Iteration 0**: Random initial centroids
        2. **Each Subsequent**: Assignment → Update → Check Convergence
        3. **Final**: Stable cluster configuration
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
    from sklearn.metrics import pairwise_distances_argmin_min, adjusted_rand_score
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import ConvexHull
    
    return (
        ConvexHull,
        KMeans,
        make_blobs,
        pairwise_distances_argmin_min,
        adjusted_rand_score,
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
    data_x, data_y = make_blobs(
        n_samples=n_samples_slider.value,
        centers=n_true_clusters_slider.value,
        cluster_std=cluster_std_slider.value,
        center_box=(-10, 10)
    )
    
    # Create DataFrame
    df = pd.DataFrame(data_x, columns=['x', 'y'])
    df['true_cluster'] = data_y
    
    return df


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🚀 K-Means Evolution Setup
        
        Configure the algorithm and watch it evolve step by step!
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
    
    # Button to run K-Means evolution
    run_button = mo.ui.button(
        label="🎬 Run K-Means Evolution",
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
def __(run_button, k_clusters_slider, max_iterations_slider, df, KMeans, np, pairwise_distances_argmin_min, adjusted_rand_score, mo):
    # State management - only run when button is pressed
    iteration_history = []
    
    if not run_button.value:
        status_md = "Press the button above to run K-Means evolution!"
        iteration_slider = mo.ui.slider(0, 0, value=0, label="Iteration", disabled=True)
    else:
        # Prepare data
        X_data = df[['x', 'y']].values
        y_true = df['true_cluster'].values
        
        # Manual K-Means evolution to capture each iteration
        rng = np.random.default_rng()
        current_centroids = X_data[rng.choice(X_data.shape[0], k_clusters_slider.value, replace=False)]
        
        # Store initial state
        iteration_data = {
            'iteration': 0,
            'centroids': current_centroids.copy(),
            'labels': pairwise_distances_argmin_min(current_centroids, X_data)[1],
            'ari': adjusted_rand_score(y_true, pairwise_distances_argmin_min(current_centroids, X_data)[1])
        }
        iteration_history = [iteration_data]
        
        # Manual K-Means iterations
        actual_iterations = 0
        for i in range(max_iterations_slider.value):
            # Assignment step
            old_labels = pairwise_distances_argmin_min(current_centroids, X_data)[1]
            
            # Update step - calculate new centroids
            new_centroids = np.zeros_like(current_centroids)
            for j in range(k_clusters_slider.value):
                cluster_points = X_data[old_labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = current_centroids[j]
            
            # Check for convergence
            if np.allclose(current_centroids, new_centroids):
                actual_iterations = i + 1
                break
            
            current_centroids = new_centroids
            current_labels = pairwise_distances_argmin_min(current_centroids, X_data)[1]
            
            # Store this iteration
            iteration_data = {
                'iteration': i + 1,
                'centroids': current_centroids.copy(),
                'labels': current_labels.copy(),
                'ari': adjusted_rand_score(y_true, current_labels)
            }
            iteration_history.append(iteration_data)
        else:
            actual_iterations = max_iterations_slider.value
        
        # Create final kmeans object for reference
        kmeans = KMeans(
            n_clusters=k_clusters_slider.value,
            init=iteration_history[-1]['centroids'],
            n_init=1,
            max_iter=1,
            random_state=42
        )
        kmeans.fit(X_data)
        
        final_ari = iteration_history[-1]['ari']
        
        status_md = f"""
        ## 📊 K-Means Evolution Complete
        
        **Performance:**
        - **Total Iterations:** {actual_iterations}
        - **Final ARI:** {final_ari:.3f} (1.0 = perfect match)
        - **Final Centroids:** {len(kmeans.cluster_centers_)}
        
        **Evolution Summary:**
        - Start with random centroids (Iteration 0)
        - Each iteration: Assignment → Update → Check Convergence
        - Watch clusters form and stabilize over time
        """
        
        # Create iteration slider
        iteration_slider = mo.ui.slider(
            0, 
            len(iteration_history) - 1, 
            value=len(iteration_history) - 1, 
            label="Iteration", 
            show_value=True
        )
    
    return status_md, kmeans, iteration_history, iteration_slider


@app.cell
def __(status_md, mo):
    # Display the K-Means results
    mo.md(status_md)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🎬 Cluster Evolution Viewer
        
        **Step through each iteration to see how clusters evolve!**
        
        Use the slider below to:
        - **Iteration 0**: Random initial centroids
        - **Early Iterations**: Watch clusters form
        - **Later Iterations**: See convergence process
        - **Final State**: Stable cluster configuration
        
        Each iteration shows:
        - **Data Points**: Colored by current cluster assignment
        - **Centroids**: Black X marks (white outline)
        - **Convex Hulls**: Semi-transparent cluster boundaries
        - **ARI Score**: How well clusters match true labels
        """
    )
    return


@app.cell
def __(iteration_slider, mo, df, ConvexHull, go, np, px, iteration_history):
    # Create visualization for selected iteration
    if len(iteration_history) == 0:
        viz_md = mo.md("Run the algorithm first to see the evolution!")
    else:
        # Get selected iteration data
        selected_iter = iteration_slider.value
        iter_data = iteration_history[selected_iter]
        plot_X_data = df[['x', 'y']].values
        
        # Create figure
        evolution_fig = go.Figure()
        
        # Define colors for each cluster
        cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
        
        # Add convex hulls for each cluster in this iteration
        for cluster_idx in np.unique(iter_data['labels']):
            # Get points for this cluster
            cluster_pts = plot_X_data[iter_data['labels'] == cluster_idx]
            
            # Only add hull if we have enough points
            if len(cluster_pts) >= 3:
                try:
                    # Calculate convex hull
                    cluster_hull = ConvexHull(cluster_pts)
                    hull_pts = cluster_pts[cluster_hull.vertices]
                    
                    # Close the polygon
                    hull_pts_closed = np.vstack([hull_pts, hull_pts[0]])
                    
                    # Convert color to RGB for fillcolor
                    fill_color_hex = px.colors.qualitative.Plotly[cluster_idx % len(px.colors.qualitative.Plotly)]
                    fill_color_rgb = px.colors.hex_to_rgb(fill_color_hex)
                    
                    # Add hull trace
                    evolution_fig.add_trace(go.Scatter(
                        x=hull_pts_closed[:, 0],
                        y=hull_pts_closed[:, 1],
                        mode='lines',
                        fill='toself',
                        fillcolor=f'rgba({fill_color_rgb[0]}, {fill_color_rgb[1]}, {fill_color_rgb[2]}, 0.2)',
                        line=dict(color=cluster_colors[cluster_idx % len(cluster_colors)], width=2),
                        name=f'Cluster {cluster_idx}',
                        showlegend=False
                    ))
                except:
                    pass
        
        # Add data points colored by cluster
        evolution_fig.add_trace(go.Scatter(
            x=plot_X_data[:, 0],
            y=plot_X_data[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=[cluster_colors[i % len(cluster_colors)] for i in iter_data['labels']],
                opacity=0.8
            ),
            name='Data Points'
        ))
        
        # Add centroids for this iteration
        evolution_fig.add_trace(go.Scatter(
            x=iter_data['centroids'][:, 0],
            y=iter_data['centroids'][:, 1],
            mode='markers',
            marker=dict(
                size=20,
                color='black',
                symbol='x',
                line=dict(width=3, color='white')
            ),
            name='Centroids'
        ))
        
        # Add iteration information
        evolution_fig.add_annotation(
            text=f"<b>Iteration {selected_iter}</b><br>ARI: {iter_data['ari']:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        )
        
        evolution_fig.update_layout(
            title=f'K-Means Evolution - Iteration {selected_iter}',
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        viz_md = mo.ui.plotly(evolution_fig)
    
    return viz_md


if __name__ == "__main__":
    app.run()