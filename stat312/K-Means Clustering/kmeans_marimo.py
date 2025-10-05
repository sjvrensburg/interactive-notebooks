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
    mo.md(r"""## 🎛️ Configuration""")
    return


@app.cell
def __(mo):
    # UI controls for data generation and K-Means
    n_true_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=3, label="True number of clusters"
    )

    n_samples_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200, label="Number of data points"
    )

    cluster_std_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.2, label="Cluster separation (std)"
    )

    k_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=3, label="Number of clusters (k)"
    )

    max_iterations_slider = mo.ui.slider(
        start=5, stop=30, step=1, value=15, label="Maximum iterations"
    )

    random_state_slider = mo.ui.slider(
        start=0, stop=100, step=1, value=42, label="Random state (seed)"
    )

    mo.md(
        f"""
        **Data Generation:** {n_true_clusters_slider} {n_samples_slider} {cluster_std_slider}

        **K-Means Algorithm:** {k_clusters_slider} {max_iterations_slider} {random_state_slider}
        """
    )
    return n_true_clusters_slider, n_samples_slider, cluster_std_slider, k_clusters_slider, max_iterations_slider, random_state_slider


@app.cell
def __(make_blobs, n_samples_slider, n_true_clusters_slider, cluster_std_slider, random_state_slider, pd):
    # Generate synthetic dataset
    print(f"🎲 Generating {n_samples_slider.value} points with {n_true_clusters_slider.value} true clusters (seed={random_state_slider.value})...")

    data_x, data_y = make_blobs(
        n_samples=n_samples_slider.value,
        centers=n_true_clusters_slider.value,
        cluster_std=cluster_std_slider.value,
        center_box=(-10, 10),
        random_state=random_state_slider.value
    )

    # Create DataFrame
    df = pd.DataFrame(data_x, columns=['x', 'y'])
    df['true_cluster'] = data_y

    print(f"✅ Data generated: shape={data_x.shape}")

    return df, data_x, data_y


@app.cell
def __(k_clusters_slider, max_iterations_slider, random_state_slider, df, np, pairwise_distances_argmin_min, adjusted_rand_score):
    # Run K-Means evolution
    print(f"\n🎬 Running K-Means with k={k_clusters_slider.value}, max_iter={max_iterations_slider.value}, seed={random_state_slider.value}...")

    # Prepare data
    X_data = df[['x', 'y']].values
    y_true = df['true_cluster'].values

    # Manual K-Means evolution to capture each iteration
    rng = np.random.default_rng(random_state_slider.value)
    current_centroids = X_data[rng.choice(X_data.shape[0], k_clusters_slider.value, replace=False)]

    print(f"🎯 Initial centroids selected from data points")

    # Store initial state
    iteration_data = {
        'iteration': 0,
        'centroids': current_centroids.copy(),
        'labels': pairwise_distances_argmin_min(X_data, current_centroids)[0],
        'ari': adjusted_rand_score(y_true, pairwise_distances_argmin_min(X_data, current_centroids)[0])
    }
    iteration_history = [iteration_data]

    print(f"📊 Iteration 0: ARI = {iteration_data['ari']:.3f}")

    # Manual K-Means iterations
    actual_iterations = 0
    for i in range(max_iterations_slider.value):
        # Assignment step
        old_labels = pairwise_distances_argmin_min(X_data, current_centroids)[0]

        # Update step - calculate new centroids
        new_centroids = np.zeros_like(current_centroids)
        for j in range(k_clusters_slider.value):
            cluster_points = X_data[old_labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = current_centroids[j]

        # Check for convergence
        if np.allclose(current_centroids, new_centroids, atol=1e-4):
            actual_iterations = i + 1
            print(f"✨ Converged at iteration {actual_iterations}")
            break

        current_centroids = new_centroids
        current_labels = pairwise_distances_argmin_min(X_data, current_centroids)[0]

        # Store this iteration
        iteration_data = {
            'iteration': i + 1,
            'centroids': current_centroids.copy(),
            'labels': current_labels.copy(),
            'ari': adjusted_rand_score(y_true, current_labels)
        }
        iteration_history.append(iteration_data)

        if (i + 1) % 5 == 0:
            print(f"📊 Iteration {i + 1}: ARI = {iteration_data['ari']:.3f}")
    else:
        actual_iterations = max_iterations_slider.value
        print(f"⚠️  Reached maximum iterations ({actual_iterations})")

    final_ari = iteration_history[-1]['ari']

    print(f"🏁 Final ARI: {final_ari:.3f} ({len(iteration_history)} total iterations captured)")

    return iteration_history, X_data, y_true, current_centroids, actual_iterations, final_ari


@app.cell
def __(mo, actual_iterations, final_ari, iteration_history):
    # Create iteration slider
    iteration_slider = mo.ui.slider(
        0,
        len(iteration_history) - 1,
        value=len(iteration_history) - 1,
        label="Iteration",
        show_value=True
    )

    mo.md(
        f"""
        ## 📊 K-Means Evolution Results

        **Performance:** Total Iterations: {actual_iterations} | Final ARI: {final_ari:.3f} (1.0 = perfect)

        **Evolution Viewer:** {iteration_slider}
        """
    )
    return iteration_slider,


@app.cell
def __(iteration_slider, X_data, ConvexHull, go, np, px, iteration_history):
    # Create visualisation for selected iteration
    print(f"\n🎨 Rendering iteration {iteration_slider.value}...")

    # Get selected iteration data
    selected_iter = iteration_slider.value
    iter_data = iteration_history[selected_iter]

    # Create figure
    evolution_fig = go.Figure()

    # Define colours for each cluster
    cluster_colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']

    # Add convex hulls for each cluster in this iteration
    hull_count = 0
    for cluster_idx in np.unique(iter_data['labels']):
        # Get points for this cluster
        cluster_pts = X_data[iter_data['labels'] == cluster_idx]

        # Only add hull if we have enough points
        if len(cluster_pts) >= 3:
            try:
                # Calculate convex hull
                cluster_hull = ConvexHull(cluster_pts)
                hull_pts = cluster_pts[cluster_hull.vertices]

                # Close the polygon
                hull_pts_closed = np.vstack([hull_pts, hull_pts[0]])

                # Convert colour to RGB for fillcolor
                fill_colour_hex = px.colors.qualitative.Plotly[cluster_idx % len(px.colors.qualitative.Plotly)]
                fill_colour_rgb = px.colors.hex_to_rgb(fill_colour_hex)

                # Add hull trace
                evolution_fig.add_trace(go.Scatter(
                    x=hull_pts_closed[:, 0],
                    y=hull_pts_closed[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba({fill_colour_rgb[0]}, {fill_colour_rgb[1]}, {fill_colour_rgb[2]}, 0.2)',
                    line=dict(color=cluster_colours[cluster_idx % len(cluster_colours)], width=2),
                    name=f'Cluster {cluster_idx}',
                    showlegend=False
                ))
                hull_count += 1
            except Exception as e:
                print(f"⚠️  Could not create hull for cluster {cluster_idx}: {e}")

    print(f"✅ Created {hull_count} convex hulls")

    # Add data points coloured by cluster
    evolution_fig.add_trace(go.Scatter(
        x=X_data[:, 0],
        y=X_data[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=[cluster_colours[i % len(cluster_colours)] for i in iter_data['labels']],
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

    # Add iteration information in bottom-right corner
    evolution_fig.add_annotation(
        text=f"<b>Iteration {selected_iter}</b><br>ARI: {iter_data['ari']:.3f}",
        xref="paper", yref="paper",
        x=0.98, y=0.02, showarrow=False,
        xanchor="right", yanchor="bottom",
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

    print(f"✅ Plot created successfully with {len(X_data)} points")

    evolution_fig
    return evolution_fig, selected_iter, iter_data


if __name__ == "__main__":
    app.run()
