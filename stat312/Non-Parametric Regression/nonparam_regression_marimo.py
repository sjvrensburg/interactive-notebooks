import marimo

__generated_with = "0.8.5"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # Non-Parametric Regression Methods
        ## Interactive Demonstration of k-Nearest Neighbours and Kernel-Based Approaches

        Welcome to this interactive tutorial on **Non-Parametric Regression**! This notebook provides hands-on exploration of regression techniques that make minimal assumptions about the underlying functional relationship between predictor and response variables.

        ## What are Non-Parametric Regression Methods?

        Unlike parametric approaches that assume specific functional forms (like linear regression), non-parametric methods derive predictions directly from the data structure, allowing the relationship $f(x)$ in $E[Y|X=x] = f(x)$ to assume any form supported by the data.

        ### Key Methods Covered:

        **📊 k-Nearest Neighbours (k-NN) Regression:**

        - Predicts by averaging response values from the k most similar observations

        - Embodies the principle of **local smoothness**

        - Simple but powerful for complex relationships

        **🔧 Nadaraya-Watson Kernel Regression:**

        - Uses kernel-weighted local averaging for smooth function estimates

        - Provides continuous, differentiable predictions

        - Flexible bandwidth parameter controls smoothness

        **📈 Key Advantages:**

        - No assumptions about functional form

        - Can capture complex, non-linear relationships

        - Adaptable to local data patterns

        **⚠️ Considerations:**

        - Requires larger sample sizes

        - Can struggle in sparse data regions

        - Computational complexity increases with data size
        """
    )
    return


@app.cell
def __():
    # Import required libraries
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold
    from scipy.spatial.distance import cdist
    from typing import Callable
    import pandas as pd

    # Set random seed for reproducibility
    np.random.seed(2025)
    return (
        Callable,
        KFold,
        KNeighborsRegressor,
        cdist,
        go,
        make_subplots,
        np,
        pd,
        px,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Data Generation

        Let's start by generating synthetic data that exhibits a non-linear relationship. This will help us understand how different regression methods perform on complex patterns.
        """
    )
    return


@app.cell
def __(mo, np):
    # UI controls for data generation
    n_points_slider = mo.ui.slider(
        start=50, stop=200, step=10, value=100,
        label="Number of Data Points:"
    )

    noise_level_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.1, value=0.5,
        label="Noise Level:"
    )

    function_selector = mo.ui.dropdown(
        options=["Sine Wave", "Polynomial", "Step Function"],
        value="Sine Wave",
        label="True Function:"
    )

    test_split_slider = mo.ui.slider(
        start=0.1, stop=0.4, step=0.05, value=0.2,
        label="Test Set Proportion:"
    )

    mo.md(f"""
    **Data Generation Parameters:**

    {n_points_slider}
    {noise_level_slider}
    {function_selector}
    {test_split_slider}
    """)
    return function_selector, n_points_slider, noise_level_slider, test_split_slider


@app.cell
def __(function_selector, go, n_points_slider, noise_level_slider, np, test_split_slider):
    # Generate data based on selected parameters
    n_points = n_points_slider.value
    noise_level = noise_level_slider.value
    test_proportion = test_split_slider.value

    # Create x values
    x_all = np.linspace(0, 10, n_points)

    # Generate y values based on selected function
    if function_selector.value == "Sine Wave":
        y_true_func = lambda x: 2 * np.sin(x) + 0.5 * x
        y_all = y_true_func(x_all) + np.random.normal(0, noise_level, n_points)
    elif function_selector.value == "Polynomial":
        y_true_func = lambda x: 0.1 * x**3 - 0.5 * x**2 + x + 2
        y_all = y_true_func(x_all) + np.random.normal(0, noise_level, n_points)
    else:  # Step Function
        y_true_func = lambda x: np.where(x < 3, 1, np.where(x < 7, 3, 0.5))
        y_all = y_true_func(x_all) + np.random.normal(0, noise_level, n_points)

    # Create train/test split
    n_test = int(n_points * test_proportion)
    test_indices = np.random.choice(n_points, n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_points), test_indices)

    # Split the data
    x_train_initial = x_all[train_indices]
    y_train_initial = y_all[train_indices]
    x_test_initial = x_all[test_indices]
    y_test_initial = y_all[test_indices]

    # Create dense grid for smooth plotting
    x_grid = np.linspace(0, 10, 200)
    y_true = y_true_func(x_grid)

    # Create initial data visualization
    data_fig = go.Figure()

    # Add training data
    data_fig.add_trace(go.Scatter(
        x=x_train_initial,
        y=y_train_initial,
        mode='markers',
        marker=dict(color='blue', size=8, opacity=0.8),
        name='Training Data'
    ))

    # Add initial test data
    data_fig.add_trace(go.Scatter(
        x=x_test_initial,
        y=y_test_initial,
        mode='markers',
        marker=dict(color='red', size=10, symbol='diamond', opacity=0.8),
        name='Test Data'
    ))

    # Add true function
    data_fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_true,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Function'
    ))

    data_fig.update_layout(
        title=f'Generated Dataset: {function_selector.value} (n={n_points}, test={test_proportion:.0%}, noise={noise_level})',
        xaxis_title='Predictor (x)',
        yaxis_title='Response (y)',
        width=800,
        height=500,
        showlegend=True
    )

    data_fig

    return (
        data_fig,
        n_points,
        n_test,
        noise_level,
        test_indices,
        test_proportion,
        train_indices,
        x_all,
        x_grid,
        x_test_initial,
        x_train_initial,
        y_all,
        y_test_initial,
        y_train_initial,
        y_true,
        y_true_func,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Query Point Selection

        Use the slider below to select a query point for detailed analysis of nearest neighbours and kernel weights.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The train/test split is now fixed for consistent evaluation across different methods and parameters.
        """
    )
    return


@app.cell
def __(
    test_indices,
    train_indices,
    x_all,
    y_all,
):
    # Use the original automatic train/test split
    current_train_indices = train_indices
    current_test_indices = test_indices

    # Extract current train/test data
    x_train = x_all[current_train_indices]
    y_train = y_all[current_train_indices]
    x_test = x_all[current_test_indices]
    y_test = y_all[current_test_indices]

    return (
        current_test_indices,
        current_train_indices,
        x_test,
        x_train,
        y_test,
        y_train,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## k-Nearest Neighbours Regression

        k-NN regression predicts response values by averaging the k closest training observations. The mathematical formulation is:

        $$\hat{f}(x_0) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x_0)} y_i$$

        where $\mathcal{N}_k(x_0)$ represents the k nearest neighbours of $x_0$.
        """
    )
    return


@app.cell
def __(mo):
    # UI controls for k-NN regression
    k_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=5,
        label="Number of Neighbours (k):"
    )

    query_point_knn_slider = mo.ui.slider(
        start=0.0, stop=10.0, step=0.1, value=5.0,
        label="Query Point (x-value):"
    )

    show_neighbours = mo.ui.checkbox(
        value=False, label="Show Nearest Neighbours for Query Point"
    )

    mo.md(f"""
    **k-NN Regression Parameters:**

    {k_slider}
    {query_point_knn_slider}
    {show_neighbours}
    """)
    return k_slider, query_point_knn_slider, show_neighbours


@app.cell
def __(
    KNeighborsRegressor,
    go,
    k_slider,
    np,
    query_point_knn_slider,
    show_neighbours,
    x_grid,
    x_test,
    x_train,
    y_test,
    y_train,
    y_true,
):
    # k-NN Regression implementation
    k_value = k_slider.value

    # Fit k-NN model on training data
    knn_model = KNeighborsRegressor(n_neighbors=k_value)
    knn_model.fit(x_train.reshape(-1, 1), y_train)

    # Make predictions
    y_knn_pred_grid = knn_model.predict(x_grid.reshape(-1, 1))
    y_knn_pred_test = knn_model.predict(x_test.reshape(-1, 1))

    # Calculate test performance
    knn_test_mse = np.mean((y_test - y_knn_pred_test)**2)
    knn_test_r2 = 1 - np.sum((y_test - y_knn_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2)

    # Create plot
    knn_fig = go.Figure()

    # Plot training data
    knn_fig.add_trace(go.Scatter(
        x=x_train,
        y=y_train,
        mode='markers',
        marker=dict(color='blue', size=8, opacity=0.7),
        name='Training Data'
    ))

    # Plot test data
    knn_fig.add_trace(go.Scatter(
        x=x_test,
        y=y_test,
        mode='markers',
        marker=dict(color='red', size=10, symbol='diamond', opacity=0.8),
        name='Test Data'
    ))

    # Plot true function
    knn_fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_true,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Function'
    ))

    # Plot k-NN prediction
    knn_fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_knn_pred_grid,
        mode='lines',
        line=dict(color='darkblue', width=3),
        name=f'k-NN Regression (k={k_value})'
    ))

    # Plot test predictions
    knn_fig.add_trace(go.Scatter(
        x=x_test,
        y=y_knn_pred_test,
        mode='markers',
        marker=dict(color='orange', size=8, symbol='x'),
        name='Test Predictions'
    ))

    # Show nearest neighbours for the query point if requested
    if show_neighbours.value:
        # Use the query point from slider
        query_point_knn = query_point_knn_slider.value
        query_idx_knn = abs(x_grid - query_point_knn).argmin()
        query_prediction = knn_model.predict([[query_point_knn]])[0]

        # Find k nearest neighbours
        distances_knn, indices_knn = knn_model.kneighbors([[query_point_knn]], n_neighbors=k_value)
        neighbour_x_knn = x_train[indices_knn[0]]
        neighbour_y_knn = y_train[indices_knn[0]]

        # Highlight the query point
        knn_fig.add_trace(go.Scatter(
            x=[query_point_knn],
            y=[query_prediction],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name=f'Query Point (x={query_point_knn:.1f})'
        ))

        # Highlight nearest neighbours
        knn_fig.add_trace(go.Scatter(
            x=neighbour_x_knn,
            y=neighbour_y_knn,
            mode='markers',
            marker=dict(color='orange', size=12, line=dict(color='red', width=2)),
            name=f'{k_value} Nearest Neighbours'
        ))

        # Draw lines to neighbours
        for nx, ny in zip(neighbour_x_knn, neighbour_y_knn):
            knn_fig.add_trace(go.Scatter(
                x=[query_point_knn, nx],
                y=[query_prediction, ny],
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

    knn_fig.update_layout(
        title=f'k-NN Regression (k={k_value}) - Test MSE: {knn_test_mse:.3f}, R²: {knn_test_r2:.3f}',
        xaxis_title='Predictor (x)',
        yaxis_title='Response (y)',
        width=900,
        height=600,
        showlegend=True
    )

    knn_fig

    return (
        k_value,
        knn_fig,
        knn_model,
        knn_test_mse,
        knn_test_r2,
        y_knn_pred_grid,
        y_knn_pred_test,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Kernel Functions

        Kernel functions provide systematic weighting schemes for observations based on their distance from target points. Common kernel functions include:

        - **Gaussian (Normal)**: $K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{u^2}{2}}$
        - **Epanechnikov**: $K(u) = \frac{3}{4}(1-u^2)$ for $|u| \leq 1$, 0 otherwise
        - **Triangular**: $K(u) = (1-|u|)$ for $|u| \leq 1$, 0 otherwise
        """
    )
    return


@app.cell
def __(Callable, np):
    # Define kernel functions
    def gaussian_kernel(u: np.ndarray) -> np.ndarray:
        """Gaussian (Normal) kernel function."""
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
        """Epanechnikov kernel function."""
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def triangular_kernel(u: np.ndarray) -> np.ndarray:
        """Triangular kernel function."""
        return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)

    def uniform_kernel(u: np.ndarray) -> np.ndarray:
        """Uniform kernel function."""
        return np.where(np.abs(u) <= 1, 0.5, 0)

    # Dictionary of available kernels
    KERNELS = {
        'Gaussian': gaussian_kernel,
        'Epanechnikov': epanechnikov_kernel,
        'Triangular': triangular_kernel,
        'Uniform': uniform_kernel
    }

    return (
        KERNELS,
        epanechnikov_kernel,
        gaussian_kernel,
        triangular_kernel,
        uniform_kernel,
    )


@app.cell
def __(KERNELS, go, np):
    # Visualize kernel functions
    u_vals = np.linspace(-2, 2, 400)

    kernel_fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange']
    for kernel_idx, (kernel_name, kernel_func) in enumerate(KERNELS.items()):
        kernel_vals = kernel_func(u_vals)
        kernel_fig.add_trace(go.Scatter(
            x=u_vals,
            y=kernel_vals,
            mode='lines',
            line=dict(width=3, color=colors[kernel_idx]),
            name=kernel_name
        ))

    kernel_fig.update_layout(
        title='Comparison of Kernel Functions',
        xaxis_title='u (standardized distance)',
        yaxis_title='K(u) (kernel weight)',
        width=700,
        height=400,
        showlegend=True
    )

    kernel_fig

    return kernel_fig, kernel_func, kernel_name, kernel_vals, u_vals


@app.cell
def __(Callable, np):
    # Nadaraya-Watson regression implementation
    def nadaraya_watson_regression(x_train: np.ndarray, y_train: np.ndarray,
                                  x_pred: np.ndarray, bandwidth: float,
                                  kernel: Callable) -> np.ndarray:
        """
        Implement Nadaraya-Watson kernel regression.

        Parameters:
        - x_train: Training predictor values
        - y_train: Training response values
        - x_pred: Prediction points
        - bandwidth: Kernel bandwidth parameter
        - kernel: Kernel function to use

        Returns:
        - predictions: Predicted values at x_pred points
        """
        predictions = np.zeros(len(x_pred))

        for i, x_point in enumerate(x_pred):
            # Calculate standardized distances
            distances = (x_point - x_train) / bandwidth
            weights = kernel(distances)

            # Nadaraya-Watson estimate
            if np.sum(weights) > 0:
                predictions[i] = np.sum(weights * y_train) / np.sum(weights)
            else:
                predictions[i] = np.mean(y_train)  # Fallback for zero weights

        return predictions

    return nadaraya_watson_regression,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Nadaraya-Watson Kernel Regression

        The Nadaraya-Watson estimator provides smooth function estimates through kernel weighting:

        $$\hat{f}(x_0) = \frac{\sum_{i=1}^{n} K\left(\frac{x_0 - x_i}{h}\right) y_i}{\sum_{i=1}^{n} K\left(\frac{x_0 - x_i}{h}\right)}$$

        where $K(\cdot)$ is the kernel function and $h$ is the bandwidth parameter.
        """
    )
    return


@app.cell
def __(KERNELS, mo):
    # UI controls for Nadaraya-Watson regression
    kernel_selector = mo.ui.dropdown(
        options=list(KERNELS.keys()),
        value='Gaussian',
        label='Kernel Function:'
    )

    bandwidth_slider = mo.ui.slider(
        start=0.2, stop=3.0, step=0.2, value=1.0,
        label='Bandwidth (h):'
    )

    query_point_nw_slider = mo.ui.slider(
        start=0.0, stop=10.0, step=0.1, value=5.0,
        label="Query Point (x-value):"
    )

    show_weights = mo.ui.checkbox(
        value=False, label='Show Kernel Weights for Query Point'
    )

    mo.md(f"""
    **Nadaraya-Watson Parameters:**

    {kernel_selector}
    {bandwidth_slider}
    {query_point_nw_slider}
    {show_weights}
    """)
    return bandwidth_slider, kernel_selector, query_point_nw_slider, show_weights


@app.cell
def __(
    KERNELS,
    bandwidth_slider,
    go,
    kernel_selector,
    nadaraya_watson_regression,
    np,
    query_point_nw_slider,
    show_weights,
    x_grid,
    x_test,
    x_train,
    y_test,
    y_train,
    y_true,
):
    # Nadaraya-Watson Regression
    selected_kernel_func = KERNELS[kernel_selector.value]
    bandwidth = bandwidth_slider.value

    # Fit Nadaraya-Watson model
    y_nw_pred_grid = nadaraya_watson_regression(x_train, y_train, x_grid,
                                         bandwidth, selected_kernel_func)
    y_nw_pred_test = nadaraya_watson_regression(x_train, y_train, x_test,
                                         bandwidth, selected_kernel_func)

    # Calculate test performance
    nw_test_mse = np.mean((y_test - y_nw_pred_test)**2)
    nw_test_r2 = 1 - np.sum((y_test - y_nw_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2)

    # Create plot
    nw_fig = go.Figure()

    # Plot training data
    nw_fig.add_trace(go.Scatter(
        x=x_train,
        y=y_train,
        mode='markers',
        marker=dict(color='blue', size=8, opacity=0.7),
        name='Training Data'
    ))

    # Plot test data
    nw_fig.add_trace(go.Scatter(
        x=x_test,
        y=y_test,
        mode='markers',
        marker=dict(color='red', size=10, symbol='diamond', opacity=0.8),
        name='Test Data'
    ))

    # Plot true function
    nw_fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_true,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Function'
    ))

    # Plot Nadaraya-Watson prediction
    nw_fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_nw_pred_grid,
        mode='lines',
        line=dict(color='darkred', width=3),
        name=f'Nadaraya-Watson ({kernel_selector.value})'
    ))

    # Plot test predictions
    nw_fig.add_trace(go.Scatter(
        x=x_test,
        y=y_nw_pred_test,
        mode='markers',
        marker=dict(color='orange', size=8, symbol='x'),
        name='Test Predictions'
    ))

    # Show kernel weights for the query point if requested
    if show_weights.value:
        # Use the query point from slider
        query_point_nw = query_point_nw_slider.value
        query_prediction_nw = nadaraya_watson_regression(
            x_train, y_train, np.array([query_point_nw]), bandwidth, selected_kernel_func
        )[0]

        # Calculate weights for all training points
        distances_nw = (query_point_nw - x_train) / bandwidth
        weights_nw = selected_kernel_func(distances_nw)
        weights_normalized_nw = weights_nw / np.sum(weights_nw) if np.sum(weights_nw) > 0 else weights_nw

        # Highlight the query point
        nw_fig.add_trace(go.Scatter(
            x=[query_point_nw],
            y=[query_prediction_nw],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name=f'Query Point (x={query_point_nw:.1f})'
        ))

        # Show weights as varying size
        for weight_idx, (xi, yi, w) in enumerate(zip(x_train, y_train, weights_normalized_nw)):
            if w > 0.001:  # Only show significant weights
                nw_fig.add_trace(go.Scatter(
                    x=[xi],
                    y=[yi],
                    mode='markers',
                    marker=dict(
                        color='orange',
                        size=max(8, 300*w + 6),
                        opacity=0.8,
                        line=dict(color='red', width=1)
                    ),
                    showlegend=False,
                    hovertemplate=f'Weight: {w:.3f}<br>x: {xi:.2f}<br>y: {yi:.2f}<extra></extra>'
                ))

    nw_fig.update_layout(
        title=f'Nadaraya-Watson ({kernel_selector.value}, h={bandwidth}) - Test MSE: {nw_test_mse:.3f}, R²: {nw_test_r2:.3f}',
        xaxis_title='Predictor (x)',
        yaxis_title='Response (y)',
        width=900,
        height=600,
        showlegend=True
    )

    nw_fig

    return (
        bandwidth,
        nw_fig,
        nw_test_mse,
        nw_test_r2,
        selected_kernel_func,
        y_nw_pred_grid,
        y_nw_pred_test,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Method Comparison

        Let's compare both methods side by side to understand their different characteristics using proper train/test evaluation.
        """
    )
    return


@app.cell
def __(
    KERNELS,
    KNeighborsRegressor,
    bandwidth_slider,
    go,
    k_slider,
    kernel_selector,
    make_subplots,
    nadaraya_watson_regression,
    np,
    x_grid,
    x_test,
    x_train,
    y_test,
    y_train,
    y_true,
):
    # Side-by-side comparison with proper test evaluation
    comparison_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'k-NN Regression (k={k_slider.value})',
                       f'Nadaraya-Watson (h={bandwidth_slider.value})'),
        horizontal_spacing=0.1
    )

    # k-NN plot (left)
    comp_knn_model = KNeighborsRegressor(n_neighbors=k_slider.value)
    comp_knn_model.fit(x_train.reshape(-1, 1), y_train)
    y_knn_comp_grid = comp_knn_model.predict(x_grid.reshape(-1, 1))
    y_knn_comp_test = comp_knn_model.predict(x_test.reshape(-1, 1))

    knn_comp_mse = np.mean((y_test - y_knn_comp_test)**2)

    # Add k-NN traces
    comparison_fig.add_trace(go.Scatter(
        x=x_train, y=y_train, mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Training Data', showlegend=True
    ), row=1, col=1)

    comparison_fig.add_trace(go.Scatter(
        x=x_test, y=y_test, mode='markers',
        marker=dict(color='red', size=8, symbol='diamond', opacity=0.8),
        name='Test Data', showlegend=True
    ), row=1, col=1)

    comparison_fig.add_trace(go.Scatter(
        x=x_grid, y=y_true, mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Function', showlegend=True
    ), row=1, col=1)

    comparison_fig.add_trace(go.Scatter(
        x=x_grid, y=y_knn_comp_grid, mode='lines',
        line=dict(color='darkblue', width=3),
        name=f'k-NN (k={k_slider.value})', showlegend=True
    ), row=1, col=1)

    # Nadaraya-Watson plot (right)
    comp_kernel_func = KERNELS[kernel_selector.value]
    y_nw_comp_grid = nadaraya_watson_regression(x_train, y_train, x_grid,
                                          bandwidth_slider.value, comp_kernel_func)
    y_nw_comp_test = nadaraya_watson_regression(x_train, y_train, x_test,
                                          bandwidth_slider.value, comp_kernel_func)

    nw_comp_mse = np.mean((y_test - y_nw_comp_test)**2)

    # Add Nadaraya-Watson traces
    comparison_fig.add_trace(go.Scatter(
        x=x_train, y=y_train, mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Training Data', showlegend=False
    ), row=1, col=2)

    comparison_fig.add_trace(go.Scatter(
        x=x_test, y=y_test, mode='markers',
        marker=dict(color='red', size=8, symbol='diamond', opacity=0.8),
        name='Test Data', showlegend=False
    ), row=1, col=2)

    comparison_fig.add_trace(go.Scatter(
        x=x_grid, y=y_true, mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='True Function', showlegend=False
    ), row=1, col=2)

    comparison_fig.add_trace(go.Scatter(
        x=x_grid, y=y_nw_comp_grid, mode='lines',
        line=dict(color='darkred', width=3),
        name=f'Nadaraya-Watson ({kernel_selector.value})', showlegend=True
    ), row=1, col=2)

    comparison_fig.update_layout(
        title=f'Method Comparison - k-NN MSE: {knn_comp_mse:.3f}, NW MSE: {nw_comp_mse:.3f}',
        width=1200,
        height=600,
        showlegend=True
    )

    comparison_fig.update_xaxes(title_text="Predictor (x)")
    comparison_fig.update_yaxes(title_text="Response (y)")

    comparison_fig

    return (
        comp_kernel_func,
        comp_knn_model,
        comparison_fig,
        knn_comp_mse,
        nw_comp_mse,
        y_knn_comp_grid,
        y_knn_comp_test,
        y_nw_comp_grid,
        y_nw_comp_test,
    )


@app.cell
def __(
    bandwidth_slider,
    k_slider,
    knn_comp_mse,
    knn_test_r2,
    mo,
    nw_comp_mse,
    nw_test_r2,
):
    mo.md(f"""
    ## Test Set Performance Comparison

    **Evaluation on Hold-out Test Set:**

    | Method | Test MSE | Test R² |
    |--------|----------|---------|
    | k-NN (k={k_slider.value}) | {knn_comp_mse:.4f} | {knn_test_r2:.4f} |
    | Nadaraya-Watson (h={bandwidth_slider.value}) | {nw_comp_mse:.4f} | {nw_test_r2:.4f} |

    **Note:** These metrics are calculated on the test set that was held out from training. This provides an unbiased estimate of model performance on unseen data.

    **Key Insight:** Lower MSE and higher R² indicate better performance. The test set evaluation helps us understand which method generalises better to new data points.
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Key Insights and Best Practices

        ### Parameter Selection

        **k-NN Regression:**
        - **Small k**: More flexible, can capture local patterns but may overfit
        - **Large k**: Smoother predictions but may underfit
        - **Rule of thumb**: Try k = √n as a starting point, then experiment

        **Nadaraya-Watson Regression:**
        - **Small bandwidth (h)**: More flexible, follows data closely but may be noisy
        - **Large bandwidth (h)**: Smoother predictions but may miss important patterns
        - **Bandwidth selection**: Critical for good performance, experiment with different values

        ### Interactive Learning Benefits

        **Query Point Exploration:**
        - Use the query point sliders to see exactly how predictions are made at specific locations
        - Observe how nearest neighbours or kernel weights change across the input space
        - Understanding local prediction behaviour helps build intuition

        ### Method Comparison

        - **k-NN**: Step-wise predictions, focuses on exact neighbours
        - **Nadaraya-Watson**: Smooth continuous predictions, uses weighted averaging
        - **Performance**: Compare test set MSE and R² to evaluate which works better for your data

        ### Statistical Considerations

        - **Test set evaluation** provides unbiased performance estimates
        - **Different data patterns** may favor different methods
        - **Parameter tuning** is essential for optimal performance
        - **Visual inspection** of fits helps understand method behaviour
        """
    )
    return



if __name__ == "__main__":
    app.run()
