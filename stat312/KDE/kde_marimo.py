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
        # Kernel Density Estimation (KDE): Interactive Demonstration
        ## Building Intuition Through Visual "Bump Summing"

        Welcome to this interactive exploration of **Kernel Density Estimation**! This notebook demonstrates how KDE builds smooth probability density functions by placing and summing "bumps" (kernels) at each data point.

        ## What is Kernel Density Estimation?

        KDE is a non-parametric method for estimating probability density functions. Instead of assuming a specific distribution shape, KDE lets the data speak for itself by:

        **🎯 The Core Concept:**

        - Place a small "bump" (kernel function) at each data point
        - Sum all these bumps together to form a smooth density curve
        - Control the width of bumps with a bandwidth parameter

        **📊 Mathematical Foundation:**

        The KDE formula is:
        $$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^n K\left( \frac{x - X_i}{h} \right)$$

        where:

        - $\hat{f}(x)$ is the estimated density at point $x$
        - $n$ is the number of data points
        - $h$ is the bandwidth (controls bump width)
        - $K(\cdot)$ is the kernel function (shape of each bump)
        - $X_i$ are the observed data points

        **🔧 Key Features:**

        - **Bandwidth $h$**: Narrow = wiggly (follows data closely), Wide = smooth (may miss features)
        - **Kernel Choice**: Shape of individual bumps (Gaussian, Epanechnikov, etc.)
        - **Normalisation**: Division by $nh$ ensures total area = 1 (valid probability density)
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
    from scipy import stats
    from scipy.integrate import quad
    import pandas as pd

    # Set random seed for reproducibility
    np.random.seed(2025)
    return go, make_subplots, np, pd, px, quad, stats


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Sample Dataset Generation

        First, let's generate a sample dataset to demonstrate KDE concepts. You can experiment with different data patterns to see how KDE adapts.
        """
    )
    return


@app.cell
def __(mo, np):
    # UI controls for data generation
    sample_size_slider = mo.ui.slider(
        start=20, stop=100, step=5, value=40,
        label="Sample Size (n):"
    )

    data_pattern_selector = mo.ui.dropdown(
        options=["Bimodal Normal", "Skewed Distribution", "Uniform + Outliers", "Single Normal"],
        value="Bimodal Normal",
        label="Data Pattern:"
    )

    regenerate_button = mo.ui.button(
        value=0,
        label="Regenerate Sample Data"
    )

    mo.md(f"""
    **Data Generation Controls:**

    {sample_size_slider}
    {data_pattern_selector}
    {regenerate_button}
    """)
    return data_pattern_selector, regenerate_button, sample_size_slider


@app.cell
def __(data_pattern_selector, go, np, regenerate_button, sample_size_slider):
    # Generate sample data based on user selections
    # Using regenerate_button.value as a trigger for reproducible randomness
    _seed = 2025 + regenerate_button.value * 17  # Change seed when button clicked
    np.random.seed(_seed)

    n_samples = sample_size_slider.value

    # Generate data based on selected pattern
    if data_pattern_selector.value == "Bimodal Normal":
        # Mixture of two normal distributions
        _n1 = n_samples // 2
        _n2 = n_samples - _n1
        sample_data = np.concatenate([
            np.random.normal(-2, 0.8, _n1),
            np.random.normal(2, 1.2, _n2)
        ])
    elif data_pattern_selector.value == "Skewed Distribution":
        # Exponential-like distribution
        sample_data = np.random.gamma(2, 1, n_samples) - 1
    elif data_pattern_selector.value == "Uniform + Outliers":
        # Uniform with some outliers
        _main_data = np.random.uniform(-3, 3, int(0.9 * n_samples))
        _outliers = np.random.choice([-8, 8], int(0.1 * n_samples))
        sample_data = np.concatenate([_main_data, _outliers])
    else:  # Single Normal
        sample_data = np.random.normal(0, 1.5, n_samples)

    # Create initial data visualization
    data_fig = go.Figure()

    # Add sample points as rug plot at y=0
    data_fig.add_trace(go.Scatter(
        x=sample_data,
        y=np.zeros(len(sample_data)),
        mode='markers',
        marker=dict(
            color='darkblue',
            size=8,
            opacity=0.8,
            symbol='line-ns',
            line=dict(width=2)
        ),
        name=f'Sample Data (n={len(sample_data)})',
        hovertemplate='Data Point: %{x:.2f}<extra></extra>'
    ))

    data_fig.update_layout(
        title=f'Sample Dataset: {data_pattern_selector.value} (n={len(sample_data)})',
        xaxis_title='x',
        yaxis_title='',
        yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
        height=300,
        showlegend=True
    )

    return data_fig, sample_data


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Kernel Functions

        Different kernel functions create different "bump" shapes. Let's explore the most common kernels used in practice.
        """
    )
    return


@app.cell
def __(np, stats):
    # Define kernel functions
    def gaussian_kernel(u):
        """Standard Gaussian kernel: K(u) = (1/√(2π)) * exp(-u²/2)"""
        return stats.norm.pdf(u)

    def epanechnikov_kernel(u):
        """Epanechnikov kernel: K(u) = (3/4)(1-u²) for |u| ≤ 1"""
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def triangular_kernel(u):
        """Triangular kernel: K(u) = (1-|u|) for |u| ≤ 1"""
        return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)

    def uniform_kernel(u):
        """Uniform (rectangular) kernel: K(u) = 1/2 for |u| ≤ 1"""
        return np.where(np.abs(u) <= 1, 0.5, 0)

    # Dictionary of available kernels with descriptions
    kernel_functions = {
        'Gaussian': {
            'func': gaussian_kernel,
            'desc': 'Smooth, bell-shaped bumps (most common)'
        },
        'Epanechnikov': {
            'func': epanechnikov_kernel,
            'desc': 'Parabola-shaped, theoretically optimal'
        },
        'Triangular': {
            'func': triangular_kernel,
            'desc': 'Triangle-shaped with finite support'
        },
        'Uniform': {
            'func': uniform_kernel,
            'desc': 'Rectangular bumps (like histogram bins)'
        }
    }

    return (
        epanechnikov_kernel,
        gaussian_kernel,
        kernel_functions,
        triangular_kernel,
        uniform_kernel,
    )


@app.cell
def __(go, kernel_functions, np):
    # Visualise kernel functions
    u_vals = np.linspace(-3, 3, 500)

    kernel_comparison_fig = go.Figure()

    _colors = ['blue', 'red', 'green', 'orange']
    for _i, (_kernel_name, _kernel_info) in enumerate(kernel_functions.items()):
        _kernel_vals = _kernel_info['func'](u_vals)
        kernel_comparison_fig.add_trace(go.Scatter(
            x=u_vals,
            y=_kernel_vals,
            mode='lines',
            line=dict(width=3, color=_colors[_i]),
            name=f'{_kernel_name}',
            hovertemplate=f'{_kernel_name} Kernel<br>u: %{{x:.2f}}<br>K(u): %{{y:.3f}}<extra></extra>'
        ))

    kernel_comparison_fig.update_layout(
        title='Kernel Function Comparison',
        xaxis_title='u (standardised distance)',
        yaxis_title='K(u) (kernel weight)',
        width=800,
        height=500,
        showlegend=True
    )

    return kernel_comparison_fig, u_vals


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interactive KDE Construction

        Now let's build a KDE step-by-step! The plot defaults to the **normalised scale**, which makes it much easier to see how individual shaded kernel "bumps" contribute to the final density curve. The dual y-axis allows comparison between normalised density and the raw unnormalised sum.

        **💡 Why Normalised Scale?** On this scale, individual kernels are clearly visible and their contributions to the final KDE are easy to understand. The unnormalised sum can make individual bumps appear tiny relative to the main curve.

        **🎨 Colorblind-Safe Design:** Each kernel "bump" uses a different colour from a colorblind-friendly palette, making it easy to distinguish individual contributions even when they overlap.
        """
    )
    return


@app.cell
def __(kernel_functions, mo):
    # UI controls for KDE parameters
    kernel_selector = mo.ui.dropdown(
        options=list(kernel_functions.keys()),
        value='Gaussian',
        label='Kernel Function:'
    )

    bandwidth_slider = mo.ui.slider(
        start=0.2, stop=3.0, step=0.1, value=0.8,
        label='Bandwidth (h):'
    )

    show_individual_kernels = mo.ui.checkbox(
        value=True,
        label='Show Individual Kernels (Shaded Bumps)'
    )

    display_mode = mo.ui.dropdown(
        options=["Normalised KDE", "Unnormalised Sum"],
        value="Normalised KDE",
        label='Display Mode:'
    )

    highlight_point_slider = mo.ui.slider(
        start=-5, stop=5, step=0.1, value=0.0,
        label='Highlight Point (x):'
    )

    mo.md(f"""
    **KDE Construction Parameters:**

    {kernel_selector}
    {bandwidth_slider}

    **Visualisation Options:**
    {display_mode}
    {show_individual_kernels}

    **Interactive Exploration:**
    {highlight_point_slider}
    """)
    return (
        bandwidth_slider,
        display_mode,
        highlight_point_slider,
        kernel_selector,
        show_individual_kernels,
    )


@app.cell
def __(
    bandwidth_slider,
    display_mode,
    go,
    highlight_point_slider,
    kernel_functions,
    kernel_selector,
    make_subplots,
    np,
    sample_data,
    show_individual_kernels,
):
    # Build KDE step by step
    selected_kernel = kernel_functions[kernel_selector.value]['func']
    bandwidth = bandwidth_slider.value
    highlight_x = highlight_point_slider.value

    # Create evaluation grid
    x_min, x_max = np.min(sample_data) - 3, np.max(sample_data) + 3
    x_grid = np.linspace(x_min, x_max, 500)

    # Calculate individual kernels and their sum
    individual_kernels = []
    kernel_sum = np.zeros_like(x_grid)

    for _i, _xi in enumerate(sample_data):
        # Calculate kernel centred at data point _xi
        _kernel_vals = selected_kernel((x_grid - _xi) / bandwidth)
        individual_kernels.append(_kernel_vals)
        kernel_sum += _kernel_vals

    # Final KDE (normalised)
    kde_estimate = kernel_sum / (len(sample_data) * bandwidth)

    # Determine which curve to display and calculate y-axis scales
    if display_mode.value == "Unnormalised Sum":
        primary_curve = kernel_sum
        primary_label = 'Sum of Kernels'
        primary_color = 'orange'
        # Secondary axis shows normalised scale
        normalisation_factor = len(sample_data) * bandwidth
    else:  # Normalised KDE
        primary_curve = kde_estimate
        primary_label = 'Normalised KDE'
        primary_color = 'red'
        # Secondary axis shows unnormalised scale
        normalisation_factor = 1.0 / (len(sample_data) * bandwidth)

    # Create subplot with secondary y-axis
    kde_construction_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add sample data points
    kde_construction_fig.add_trace(
        go.Scatter(
            x=sample_data,
            y=np.zeros(len(sample_data)),
            mode='markers',
            marker=dict(color='black', size=10, symbol='line-ns', line=dict(width=3)),
            name='Sample Data Points',
            hovertemplate='Data Point: %{x:.2f}<extra></extra>'
        ),
        secondary_y=False
    )

    # Show individual kernels with shaded areas if requested
    if show_individual_kernels.value:
        # Colorblind-safe palette - each color has both fill (with transparency) and line versions
        _kernel_palette = [
            {'fill': 'rgba(31,119,180,0.4)', 'line': 'rgb(31,119,180)'},      # Blue
            {'fill': 'rgba(255,127,14,0.4)', 'line': 'rgb(255,127,14)'},      # Orange
            {'fill': 'rgba(44,160,44,0.4)', 'line': 'rgb(44,160,44)'},        # Green
            {'fill': 'rgba(214,39,40,0.4)', 'line': 'rgb(214,39,40)'},        # Red
            {'fill': 'rgba(148,103,189,0.4)', 'line': 'rgb(148,103,189)'},    # Purple
            {'fill': 'rgba(140,86,75,0.4)', 'line': 'rgb(140,86,75)'},        # Brown
            {'fill': 'rgba(227,119,194,0.4)', 'line': 'rgb(227,119,194)'},    # Pink
            {'fill': 'rgba(127,127,127,0.4)', 'line': 'rgb(127,127,127)'}     # Grey
        ]

        for _i, (_xi, _kernel_vals) in enumerate(zip(sample_data, individual_kernels)):
            # Scale kernel values to match display mode
            if display_mode.value == "Normalised KDE":
                _display_kernel_vals = _kernel_vals / (len(sample_data) * bandwidth)
            else:
                _display_kernel_vals = _kernel_vals

            # Cycle through colors using modulo
            _color_idx = _i % len(_kernel_palette)
            _fill_color = _kernel_palette[_color_idx]['fill']
            _line_color = _kernel_palette[_color_idx]['line']

            # Add shaded area under each kernel with alternating colors
            kde_construction_fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=_display_kernel_vals,
                    mode='lines',
                    line=dict(width=1.5, color=_line_color),
                    fill='tozeroy',
                    fillcolor=_fill_color,
                    name='Individual Kernels' if _i == 0 else '',
                    showlegend=(_i == 0),
                    hovertemplate=f'Kernel {_i+1} at x={_xi:.2f}<br>Height: %{{y:.3f}}<extra></extra>'
                ),
                secondary_y=False
            )

    # Add the primary curve
    kde_construction_fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=primary_curve,
            mode='lines',
            line=dict(width=4, color=primary_color),
            name=primary_label,
            hovertemplate=f'{primary_label}: %{{y:.3f}}<extra></extra>'
        ),
        secondary_y=False
    )

    # Highlight specific point and show calculation
    _highlight_idx = np.argmin(np.abs(x_grid - highlight_x))
    _highlight_primary_val = primary_curve[_highlight_idx]

    kde_construction_fig.add_trace(
        go.Scatter(
            x=[highlight_x],
            y=[_highlight_primary_val],
            mode='markers',
            marker=dict(color='purple', size=15, symbol='star'),
            name=f'Query Point (x={highlight_x:.1f})',
            hovertemplate=f'x={highlight_x:.1f}<br>{primary_label}={_highlight_primary_val:.3f}<extra></extra>'
        ),
        secondary_y=False
    )

    # Add vertical line at highlight point
    kde_construction_fig.add_vline(
        x=highlight_x,
        line=dict(color="purple", dash="dash", width=2),
        opacity=0.7
    )

    # Set x-axis properties
    kde_construction_fig.update_xaxes(title_text="x")

    # Set y-axes properties
    if display_mode.value == "Unnormalised Sum":
        kde_construction_fig.update_yaxes(title_text="Unnormalised Sum", secondary_y=False)
        kde_construction_fig.update_yaxes(title_text="Normalised Density", secondary_y=True)
        # Set secondary y-axis range to show normalised scale
        _max_primary = np.max(primary_curve)
        kde_construction_fig.update_yaxes(
            range=[0, _max_primary / normalisation_factor],
            secondary_y=True
        )
    else:  # Normalised KDE
        kde_construction_fig.update_yaxes(title_text="Normalised Density", secondary_y=False)
        kde_construction_fig.update_yaxes(title_text="Unnormalised Sum", secondary_y=True)
        # Set secondary y-axis range to show unnormalised scale
        _max_primary = np.max(primary_curve)
        kde_construction_fig.update_yaxes(
            range=[0, _max_primary * len(sample_data) * bandwidth],
            secondary_y=True
        )

    kde_construction_fig.update_layout(
        title=f'KDE Construction: {kernel_selector.value} Kernel (h={bandwidth:.1f}) - {display_mode.value}',
        width=1000,
        height=600,
        showlegend=True
    )

    return (
        bandwidth,
        highlight_x,
        individual_kernels,
        kde_construction_fig,
        kde_estimate,
        kernel_sum,
        primary_curve,
        selected_kernel,
        x_grid,
        x_max,
        x_min,
    )


@app.cell
def __(
    bandwidth,
    highlight_point_slider,
    kernel_sum,
    len,
    mo,
    np,
    sample_data,
    selected_kernel,
    x_grid,
):
    # Calculate and display the mathematical breakdown at the highlight point
    highlight_x_val = highlight_point_slider.value

    # Find the closest grid point for calculation
    _highlight_idx = np.argmin(np.abs(x_grid - highlight_x_val))
    _x_eval = x_grid[_highlight_idx]

    # Calculate individual contributions
    _contributions = []
    _total_kernel_sum = 0

    for _i, _xi in enumerate(sample_data):
        _u = (_x_eval - _xi) / bandwidth
        _kernel_val = selected_kernel(_u)
        _contributions.append({
            'data_point': _xi,
            'standardised_distance': _u,
            'kernel_value': _kernel_val
        })
        _total_kernel_sum += _kernel_val

    # Final KDE value
    _final_kde = _total_kernel_sum / (len(sample_data) * bandwidth)

    # Create summary text
    mo.md(f"""
    ## Mathematical Breakdown at x = {_x_eval:.2f}

    **Step-by-step KDE Calculation:**

    1. **Individual Kernel Contributions:**
       - For each data point $X_i$, calculate standardised distance: $u_i = \\frac{{x - X_i}}{{h}} = \\frac{{{_x_eval:.2f} - X_i}}{{{bandwidth:.1f}}}$
       - Apply kernel function: $K(u_i)$

    2. **Sum of All Kernels:** $\\sum_{{i=1}}^n K(u_i) = {_total_kernel_sum:.3f}$

    3. **Final KDE Value:** $\\hat{{f}}(x) = \\frac{{1}}{{nh}} \\sum_{{i=1}}^n K(u_i) = \\frac{{{_total_kernel_sum:.3f}}}{{{len(sample_data)} \\times {bandwidth:.1f}}} = {_final_kde:.3f}$

    **Parameters:**
    - Sample size (n): {len(sample_data)}
    - Bandwidth (h): {bandwidth:.1f}
    - Normalisation factor (nh): {len(sample_data) * bandwidth:.1f}

    **Note:** The normalisation by $nh$ ensures the total area under the KDE equals 1, making it a valid probability density function.
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Area Verification: Ensuring Valid Probability Density

        A fundamental property of probability density functions is that the total area under the curve must equal 1. Let's verify this for our KDE estimate.
        """
    )
    return


@app.cell
def __(go, kde_estimate, np, quad, x_grid):
    # Verify that the KDE integrates to approximately 1
    from scipy.interpolate import interp1d

    # Create interpolation function for numerical integration
    _kde_interp = interp1d(x_grid, kde_estimate, kind='cubic',
                          bounds_error=False, fill_value=0)

    # Numerical integration
    _x_min_int, _x_max_int = np.min(x_grid), np.max(x_grid)
    _integral_result, _integral_error = quad(_kde_interp, _x_min_int, _x_max_int)

    # Create area verification plot
    area_fig = go.Figure()

    # Plot the KDE with filled area
    area_fig.add_trace(go.Scatter(
        x=x_grid,
        y=kde_estimate,
        mode='lines',
        line=dict(width=3, color='darkred'),
        fill='tozeroy',
        fillcolor='rgba(139,0,0,0.3)',
        name='KDE Estimate',
        hovertemplate='x: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
    ))

    # Add horizontal line at y=0
    area_fig.add_hline(y=0, line=dict(color="black", width=1))

    area_fig.update_layout(
        title=f'KDE Area Verification (∫f(x)dx ≈ {_integral_result:.4f})',
        xaxis_title='x',
        yaxis_title='Density f(x)',
        width=800,
        height=500,
        showlegend=True,
        annotations=[
            dict(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text=f"Total Area = {_integral_result:.4f}<br>(Should ≈ 1.000)",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=14)
            )
        ]
    )

    return area_fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Bandwidth Effects: The Bias-Variance Trade-off

        The bandwidth parameter $h$ controls the smoothness of the KDE. Let's explore how different bandwidth values affect the density estimate.
        """
    )
    return


@app.cell
def __(go, kernel_functions, make_subplots, np, sample_data):
    # Compare different bandwidth values
    bandwidth_values = [0.3, 0.8, 2.0]
    bandwidth_labels = ["Narrow (h=0.3)", "Medium (h=0.8)", "Wide (h=2.0)"]

    # Create subplot figure
    bandwidth_comparison_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=bandwidth_labels,
        horizontal_spacing=0.08
    )

    # Use Gaussian kernel for comparison
    _gaussian_func = kernel_functions['Gaussian']['func']

    # Calculate range
    _x_min, _x_max = np.min(sample_data) - 3, np.max(sample_data) + 3
    _x_grid = np.linspace(_x_min, _x_max, 300)

    _colors = ['blue', 'green', 'red']

    for _i, (_h_val, _h_label) in enumerate(zip(bandwidth_values, bandwidth_labels)):
        # Calculate KDE for this bandwidth
        _kde_vals = np.zeros_like(_x_grid)
        for _xi in sample_data:
            _kde_vals += _gaussian_func((_x_grid - _xi) / _h_val)
        _kde_vals = _kde_vals / (len(sample_data) * _h_val)

        # Add data points (rug plot)
        bandwidth_comparison_fig.add_trace(
            go.Scatter(
                x=sample_data,
                y=np.zeros(len(sample_data)),
                mode='markers',
                marker=dict(color='black', size=6, symbol='line-ns', line=dict(width=2)),
                name='Data' if _i == 0 else '',
                showlegend=(_i == 0)
            ),
            row=1, col=_i+1
        )

        # Add KDE curve
        bandwidth_comparison_fig.add_trace(
            go.Scatter(
                x=_x_grid,
                y=_kde_vals,
                mode='lines',
                line=dict(width=3, color=_colors[_i]),
                name=_h_label,
                hovertemplate=f'h={_h_val}<br>x: %{{x:.2f}}<br>Density: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=_i+1
        )

    bandwidth_comparison_fig.update_layout(
        title='Bandwidth Effect on KDE: Bias-Variance Trade-off',
        height=500,
        width=1200,
        showlegend=True
    )

    # Update axis labels
    for _i in range(3):
        bandwidth_comparison_fig.update_xaxes(title_text="x", row=1, col=_i+1)
        bandwidth_comparison_fig.update_yaxes(title_text="Density", row=1, col=_i+1)

    return bandwidth_comparison_fig, bandwidth_labels, bandwidth_values


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Kernel Comparison on the Same Dataset

        Different kernel functions produce slightly different density estimates. Let's compare them using the same bandwidth to see their effects.
        """
    )
    return


@app.cell
def __(bandwidth_slider, go, kernel_functions, np, sample_data):
    # Compare different kernels with the same bandwidth
    kernel_comparison_data_fig = go.Figure()

    # Use current bandwidth from slider
    _current_bandwidth = bandwidth_slider.value
    _x_range = np.linspace(np.min(sample_data) - 3, np.max(sample_data) + 3, 400)

    # Colors for different kernels
    _kernel_colors = {'Gaussian': 'blue', 'Epanechnikov': 'red',
                     'Triangular': 'green', 'Uniform': 'orange'}

    # Add data points first
    kernel_comparison_data_fig.add_trace(go.Scatter(
        x=sample_data,
        y=np.zeros(len(sample_data)),
        mode='markers',
        marker=dict(color='black', size=8, symbol='line-ns', line=dict(width=3)),
        name='Sample Data',
        hovertemplate='Data: %{x:.2f}<extra></extra>'
    ))

    # Calculate and plot KDE for each kernel
    for _kernel_name, _kernel_info in kernel_functions.items():
        _kernel_func = _kernel_info['func']

        # Calculate KDE
        _kde_vals = np.zeros_like(_x_range)
        for _xi in sample_data:
            _kde_vals += _kernel_func((_x_range - _xi) / _current_bandwidth)
        _kde_vals = _kde_vals / (len(sample_data) * _current_bandwidth)

        # Add to plot
        kernel_comparison_data_fig.add_trace(go.Scatter(
            x=_x_range,
            y=_kde_vals,
            mode='lines',
            line=dict(width=3, color=_kernel_colors[_kernel_name]),
            name=f'{_kernel_name} Kernel',
            hovertemplate=f'{_kernel_name}<br>x: %{{x:.2f}}<br>Density: %{{y:.3f}}<extra></extra>'
        ))

    kernel_comparison_data_fig.update_layout(
        title=f'Kernel Function Comparison (h={_current_bandwidth:.1f})',
        xaxis_title='x',
        yaxis_title='Density',
        width=900,
        height=600,
        showlegend=True
    )

    return kernel_comparison_data_fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interactive Kernel Exploration at a Specific Point

        Select a point to see how each data point contributes to the density estimate at that location. This helps understand the "local averaging" nature of KDE.
        """
    )
    return


@app.cell
def __(mo):
    # Query point selector for detailed analysis
    query_point_slider = mo.ui.slider(
        start=-4, stop=4, step=0.1, value=0.0,
        label='Query Point for Detailed Analysis:'
    )

    show_kernel_weights = mo.ui.checkbox(
        value=True,
        label='Show Individual Kernel Weights'
    )

    mo.md(f"""
    **Detailed Analysis Controls:**

    {query_point_slider}
    {show_kernel_weights}
    """)
    return query_point_slider, show_kernel_weights


@app.cell
def __(
    bandwidth_slider,
    go,
    kernel_functions,
    kernel_selector,
    np,
    query_point_slider,
    sample_data,
    show_kernel_weights,
):
    # Detailed analysis at query point
    query_x = query_point_slider.value
    analysis_bandwidth = bandwidth_slider.value
    analysis_kernel = kernel_functions[kernel_selector.value]['func']

    # Calculate contributions from each data point
    contributions_data = []
    for _i, _xi in enumerate(sample_data):
        _u = (query_x - _xi) / analysis_bandwidth
        _kernel_weight = analysis_kernel(_u)
        contributions_data.append({
            'index': _i,
            'data_point': _xi,
            'distance': abs(query_x - _xi),
            'standardised_distance': _u,
            'kernel_weight': _kernel_weight
        })

    # Sort by kernel weight (descending)
    contributions_data.sort(key=lambda x: x['kernel_weight'], reverse=True)

    # Create detailed analysis figure
    analysis_fig = go.Figure()

    # Add all data points
    analysis_fig.add_trace(go.Scatter(
        x=sample_data,
        y=np.zeros(len(sample_data)),
        mode='markers',
        marker=dict(color='lightgray', size=8, symbol='line-ns', line=dict(width=2)),
        name='All Data Points',
        hovertemplate='Data: %{x:.2f}<extra></extra>'
    ))

    if show_kernel_weights.value:
        # Show top contributors with proportional marker sizes
        _max_weight = max([c['kernel_weight'] for c in contributions_data])

        for contrib in contributions_data:
            if contrib['kernel_weight'] > 0.001:  # Only show significant contributors
                _size = 8 + (contrib['kernel_weight'] / _max_weight) * 20
                _color_intensity = contrib['kernel_weight'] / _max_weight

                analysis_fig.add_trace(go.Scatter(
                    x=[contrib['data_point']],
                    y=[0],
                    mode='markers',
                    marker=dict(
                        color=f'rgba(255,0,0,{_color_intensity})',
                        size=_size,
                        symbol='circle',
                        line=dict(color='red', width=1)
                    ),
                    name=f"Point {contrib['index']}" if contrib == contributions_data[0] else '',
                    showlegend=(contrib == contributions_data[0]),
                    hovertemplate=f'Data: {contrib["data_point"]:.2f}<br>'
                                f'Distance: {contrib["distance"]:.2f}<br>'
                                f'Weight: {contrib["kernel_weight"]:.3f}<extra></extra>'
                ))

    # Highlight query point
    analysis_fig.add_trace(go.Scatter(
        x=[query_x],
        y=[0],
        mode='markers',
        marker=dict(color='blue', size=15, symbol='star'),
        name=f'Query Point (x={query_x:.1f})',
        hovertemplate=f'Query Point: {query_x:.1f}<extra></extra>'
    ))

    # Add vertical line at query point
    analysis_fig.add_vline(
        x=query_x,
        line=dict(color="blue", dash="dash", width=2),
        opacity=0.7
    )

    analysis_fig.update_layout(
        title=f'Kernel Contributions at x = {query_x:.1f}',
        xaxis_title='x',
        yaxis_title='',
        yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
        width=900,
        height=400,
        showlegend=True
    )

    # Calculate final KDE value at query point
    _total_contribution = sum([c['kernel_weight'] for c in contributions_data])
    _final_kde_value = _total_contribution / (len(sample_data) * analysis_bandwidth)

    return analysis_fig, contributions_data, query_x


@app.cell
def __(analysis_bandwidth, contributions_data, len, mo, query_x, sample_data):
    # Display contribution table
    _top_contributors = contributions_data[:8]  # Show top 8 contributors
    _total_weight = sum([c['kernel_weight'] for c in contributions_data])
    _final_value = _total_weight / (len(sample_data) * analysis_bandwidth)

    # Create contribution table as markdown string
    _table_rows = "".join([
        f"| {c['data_point']:.2f} | {c['distance']:.2f} | {c['standardised_distance']:.2f} | {c['kernel_weight']:.4f} |\n"
        for c in _top_contributors
    ])

    mo.md(f"""
    ## Contribution Analysis at x = {query_x:.2f}

    **Top Contributors (by kernel weight):**

    | Data Point | Distance | Standardised u | Kernel Weight |
    |------------|----------|----------------|---------------|
    {_table_rows}

    **Final Calculation:**

    - Sum of all kernel weights: {_total_weight:.4f}
    - Normalisation factor (n×h): {len(sample_data)} × {analysis_bandwidth:.1f} = {len(sample_data) * analysis_bandwidth:.1f}
    - **KDE value at x={query_x:.2f}: {_final_value:.4f}**

    Notice how data points closer to the query point contribute more weight to the final density estimate. This demonstrates the "local averaging" principle of KDE.
    """)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Key Insights and Educational Summary

        ### Understanding KDE Through Interactive Exploration

        **🎯 Core Concept:**
        KDE builds smooth density estimates by:

        1. Placing a "bump" (kernel) at each data point
        2. Summing all bumps together
        3. Normalising by dividing by $nh$ to ensure total area = 1

        **📊 Mathematical Intuition:**

        - **Individual kernels**: Each data point contributes a smooth bump
        - **Summing bumps**: Overlapping kernels create peaks where data clusters
        - **Normalisation**: Division by $nh$ makes it a valid probability density
        - **Local averaging**: Points closer to query location have higher influence

        ### Parameter Effects

        **🔧 Bandwidth (h) - The Smoothness Controller:**

        - **Small h**: Wiggly, follows data closely (low bias, high variance)
        - **Large h**: Smooth, may miss features (high bias, low variance)
        - **Optimal h**: Balances detail and smoothness

        **📐 Kernel Choice - The Bump Shape:**

        - **Gaussian**: Smooth, infinite support, most popular
        - **Epanechnikov**: Finite support, theoretically optimal
        - **Triangular**: Simple, finite support
        - **Uniform**: Rectangular bumps (equivalent to histogram)
        - **Impact**: Kernel choice matters less than bandwidth selection

        ### Practical Applications

        **✅ When to Use KDE:**

        - Exploratory data analysis to understand data shape
        - No assumptions about underlying distribution
        - Need smooth, continuous density estimates
        - Sufficient sample size available

        **⚠️ Limitations to Consider:**

        - Requires more data than parametric methods
        - Computationally intensive for large datasets
        - Boundary effects near data edges
        - Curse of dimensionality in high dimensions

        ### Interactive Learning Benefits

        Through this notebook, you've experienced:

        - **Visual construction** of density estimates from individual components
        - **Parameter sensitivity** through real-time adjustments
        - **Mathematical transparency** with step-by-step calculations
        - **Comparative analysis** across different settings

        This hands-on approach builds intuition that's difficult to achieve through equations alone!
        """
    )
    return


if __name__ == "__main__":
    app.run()
