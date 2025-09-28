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
        # k-Nearest Neighbors (k-NN) Classification Interactive Tutorial

        Welcome to this interactive tutorial on **k-Nearest Neighbors Classification**! This notebook will help you understand the fundamental concepts of k-NN through hands-on experimentation.

        ## What is k-NN Classification?

        The **k-Nearest Neighbors** algorithm is one of the simplest and most intuitive machine learning algorithms. It classifies data points based on the class of their nearest neighbors in the feature space.

        ### Key Concepts:

        **📊 The Algorithm:**
        1. Choose a value for **k** (number of neighbors to consider)
        2. Calculate the distance from the new point to all training points
        3. Find the **k closest** training points
        4. Assign the **majority class** among these k neighbors

        **📏 Distance Metrics:**
        - **Euclidean Distance**: √[(x₁-x₂)² + (y₁-y₂)²] (most common)
        - **Manhattan Distance**: |x₁-x₂| + |y₁-y₂|
        - **Minkowski Distance**: Generalization of the above

        **🎯 Decision Boundary:**
        The decision boundary separates different classes in the feature space. With k-NN, this boundary becomes smoother as k increases.

        **🔧 Choosing k:**
        - **Small k (k=1)**: More sensitive to noise, complex boundaries
        - **Large k**: Smoother boundaries, may oversimplify
        - **Odd k**: Helps avoid ties in binary classification
        """
    )
    return


@app.cell
def __():
    # Import all necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs, make_classification
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    # Set random seed for reproducibility
    np.random.seed(42)
    return (
        KNeighborsClassifier,
        accuracy_score,
        go,
        make_blobs,
        make_classification,
        make_subplots,
        np,
        plt,
        px,
        train_test_split,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Interactive Data Generation""")
    return


@app.cell
def __(mo):
    # UI controls for data generation
    n_samples_slider = mo.ui.slider(
        start=50, stop=300, step=25, value=150, label="Number of data points:"
    )

    cluster_std_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.5, label="Cluster separation:"
    )

    random_state_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=42, label="Random seed:"
    )

    mo.md(
        f"""
        **Adjust the data generation parameters:**

        {n_samples_slider}

        {cluster_std_slider}

        {random_state_slider}
        """
    )
    return cluster_std_slider, n_samples_slider, random_state_slider


@app.cell
def __(cluster_std_slider, go, make_blobs, n_samples_slider, np, random_state_slider):
    # Generate synthetic dataset based on UI controls
    def generate_data():
        X, y = make_blobs(
            n_samples=n_samples_slider.value,
            centers=2,
            cluster_std=cluster_std_slider.value,
            random_state=random_state_slider.value,
            center_box=(-5, 5)
        )
        return X, y

    X, y = generate_data()

    # Store original data for reference
    X_original, y_original = X.copy(), y.copy()

    # Create a simple scatter plot of the generated data
    data_fig = go.Figure()

    _colors = ['red', 'blue']
    _class_names = ['Class 0', 'Class 1']

    for _i in range(2):
        _mask = y == _i
        data_fig.add_trace(go.Scatter(
            x=X[_mask, 0],
            y=X[_mask, 1],
            mode='markers',
            marker=dict(
                color=_colors[_i],
                size=8,
                line=dict(color='black', width=1)
            ),
            name=_class_names[_i],
            showlegend=True
        ))

    data_fig.update_layout(
        title=f'Generated Dataset ({n_samples_slider.value} points)',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        width=600,
        height=400,
        hovermode='closest'
    )

    data_fig

    return X, X_original, data_fig, generate_data, y, y_original


@app.cell
def __(mo):
    mo.md(r"""## 🤖 k-NN Classification with Interactive k""")
    return


@app.cell
def __(mo):
    # UI control for k value
    k_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=5, label="k (number of neighbors):"
    )

    mo.md(f"""
    **Choose the value of k:**

    {k_slider}

    Watch how the decision boundary changes as you adjust k!
    """)
    return k_slider,


@app.cell
def __(KNeighborsClassifier, X, go, k_slider, np, y):
    # Create and train k-NN classifier
    def create_knn_model(k_val):
        _knn_model = KNeighborsClassifier(n_neighbors=k_val)
        _knn_model.fit(X, y)
        return _knn_model

    # Create decision boundary visualization using Plotly
    def create_decision_boundary_plot(X, y, k_val):
        _boundary_knn = create_knn_model(k_val)

        # Create a mesh to plot the decision boundary
        h = 0.1  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = _boundary_knn.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        # Create the plotly figure
        fig = go.Figure()

        # Add decision boundary as contour
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale=[[0, 'lightcoral'], [1, 'lightblue']],
            opacity=0.3,
            showscale=False,
            line=dict(width=0),
            name='Decision Boundary'
        ))

        # Plot data points
        _plot_colors = ['red', 'blue']
        _plot_class_names = ['Class 0', 'Class 1']
        for _plot_i, _color in enumerate(_plot_colors):
            _plot_mask = y == _plot_i
            fig.add_trace(go.Scatter(
                x=X[_plot_mask, 0],
                y=X[_plot_mask, 1],
                mode='markers',
                marker=dict(
                    color=_color,
                    size=8,
                    line=dict(color='black', width=1)
                ),
                name=_plot_class_names[_plot_i],
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title=f'k-NN Classification with k={k_val}<br>Decision Boundary Visualization',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            width=800,
            height=600,
            hovermode='closest'
        )

        return fig

    # Create and display the plot
    decision_plot = create_decision_boundary_plot(X, y, k_slider.value)
    decision_plot
    return create_knn_model, create_decision_boundary_plot, decision_plot


@app.cell
def __(mo):
    mo.md(r"""
    ## 📊 Performance Evaluation: The Elbow Plot

    To properly evaluate our k-NN model, we need to split our data into **training** and **testing** sets. This prevents overfitting and gives us a realistic estimate of how the model will perform on new, unseen data.

    **Train-Test Split Strategy:**
    - **70% Training Data**: Used to fit the k-NN model
    - **30% Testing Data**: Used to evaluate model performance
    - **Stratified Sampling**: Maintains the same class proportions in both splits

    The elbow plot below shows how accuracy changes with different values of k:
    """)
    return


@app.cell
def __(
    KNeighborsClassifier,
    X,
    accuracy_score,
    go,
    k_slider,
    mo,
    np,
    train_test_split,
    y,
):
    # Performance evaluation across different k values
    def evaluate_k_values(X, y, max_k=20):
        # Split data for proper evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        k_values = range(1, max_k + 1)
        train_accuracies = []
        test_accuracies = []

        for k in k_values:
            _eval_knn = KNeighborsClassifier(n_neighbors=k)
            _eval_knn.fit(X_train, y_train)

            # Training accuracy
            train_pred = _eval_knn.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            train_accuracies.append(train_acc)

            # Testing accuracy
            test_pred = _eval_knn.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_accuracies.append(test_acc)

        return k_values, train_accuracies, test_accuracies

    # Generate performance data
    k_vals, train_acc, test_acc = evaluate_k_values(X, y)

    # Create elbow plot using Plotly
    elbow_fig = go.Figure()

    # Add training accuracy line
    elbow_fig.add_trace(go.Scatter(
        x=list(k_vals),
        y=train_acc,
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='blue', width=2),
        marker=dict(size=6, symbol='circle')
    ))

    # Add testing accuracy line
    elbow_fig.add_trace(go.Scatter(
        x=list(k_vals),
        y=test_acc,
        mode='lines+markers',
        name='Testing Accuracy',
        line=dict(color='red', width=2),
        marker=dict(size=6, symbol='square')
    ))

    # Highlight the current k value
    current_k = k_slider.value
    if current_k <= len(k_vals):
        current_train_acc = train_acc[current_k - 1]
        current_test_acc = test_acc[current_k - 1]

        # Add highlighted points for current k
        elbow_fig.add_trace(go.Scatter(
            x=[current_k],
            y=[current_train_acc],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            name=f'Current k={current_k} (Train)',
            showlegend=False
        ))

        elbow_fig.add_trace(go.Scatter(
            x=[current_k],
            y=[current_test_acc],
            mode='markers',
            marker=dict(color='red', size=12, symbol='square'),
            name=f'Current k={current_k} (Test)',
            showlegend=False
        ))

        # Add vertical line for current k
        elbow_fig.add_vline(
            x=current_k,
            line_dash="dot",
            line_color="gray",
            opacity=0.7,
            annotation_text=f"Current k={current_k}"
        )

    # Find and annotate best k
    best_k = k_vals[np.argmax(test_acc)]
    best_acc = max(test_acc)

    # Add annotation for best k
    elbow_fig.add_annotation(
        x=best_k,
        y=best_acc,
        text=f"Best k={best_k}<br>Acc={best_acc:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        bgcolor="lightgreen",
        bordercolor="green",
        font=dict(color="green", size=10)
    )

    # Update layout
    elbow_fig.update_layout(
        title='Model Performance vs. k Value (Elbow Plot)',
        xaxis_title='k (Number of Neighbors)',
        yaxis_title='Accuracy',
        width=800,
        height=500,
        hovermode='x unified',
        yaxis=dict(range=[0.5, 1.05])
    )

    # Calculate dataset info
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create performance summary
    performance_summary = mo.md(f"""
    📈 **Performance Summary:**

    **Dataset Split:**
    - **Training Set**: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)
    - **Testing Set**: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)

    **Model Performance:**
    - **Current k={current_k}**: Test Accuracy = {test_acc[current_k-1]:.3f}
    - **Best k={best_k}**: Test Accuracy = {best_acc:.3f}

    **Understanding the Plot:**
    - **Training Accuracy** (blue): How well the model fits the training data
    - **Test Accuracy** (red): How well the model generalizes to new data
    - **Gap between lines**: Indicates overfitting (larger gap = more overfitting)
    - **Optimal k**: Balances bias and variance for best generalization
    """)

    # Display both the plot and summary
    (elbow_fig, performance_summary)

    return (
        best_acc,
        best_k,
        current_k,
        elbow_fig,
        evaluate_k_values,
        k_vals,
        performance_summary,
        test_acc,
        train_acc,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🎯 Interactive Prediction

        **Instructions:**
        1. Enter x and y coordinates below to add a new point for prediction
        2. The algorithm will predict its class using the current k value
        3. The k nearest neighbors will be highlighted
        4. The prediction will be displayed with the plot

        **Try different points to see how k-NN classification works!**
        """
    )
    return


@app.cell
def __(mo):
    # Create input controls for prediction point
    x_input = mo.ui.number(
        start=-8, stop=8, step=0.1, value=0.0, label="X coordinate:"
    )
    y_input = mo.ui.number(
        start=-8, stop=8, step=0.1, value=0.0, label="Y coordinate:"
    )

    # Use checkbox as trigger instead of button for better reactivity
    trigger_prediction = mo.ui.checkbox(
        label="Show prediction for point above", value=False
    )

    mo.md(f"""
    **Enter coordinates for a new point:**

    {x_input}

    {y_input}

    {trigger_prediction}
    """)
    return trigger_prediction, x_input, y_input


@app.cell
def __(
    X,
    create_knn_model,
    go,
    k_slider,
    mo,
    np,
    trigger_prediction,
    x_input,
    y_input,
    y,
):
    # Create the base plot with training data
    pred_fig = go.Figure()

    # Plot original data points
    _pred_colors = ['red', 'blue']
    _pred_class_names = ['Class 0', 'Class 1']

    for _pred_i in range(2):
        _pred_mask = y == _pred_i
        pred_fig.add_trace(
            go.Scatter(
                x=X[_pred_mask, 0],
                y=X[_pred_mask, 1],
                mode='markers',
                marker=dict(color=_pred_colors[_pred_i], size=8, line=dict(color='black', width=1)),
                name=_pred_class_names[_pred_i],
                showlegend=True
            )
        )

    _prediction_info = ""

    # If checkbox is checked and we have valid inputs, make prediction
    if trigger_prediction.value and x_input.value is not None and y_input.value is not None:
        new_x, new_y = x_input.value, y_input.value
        new_point = np.array([[new_x, new_y]])

        # Create k-NN model and make prediction
        _pred_knn = create_knn_model(k_slider.value)
        prediction = _pred_knn.predict(new_point)[0]
        probabilities = _pred_knn.predict_proba(new_point)[0]

        # Find k nearest neighbors
        distances, indices = _pred_knn.kneighbors(new_point, n_neighbors=k_slider.value)
        neighbor_points = X[indices[0]]

        # Add new point to plot
        pred_fig.add_trace(
            go.Scatter(
                x=[new_x],
                y=[new_y],
                mode='markers',
                marker=dict(
                    color='gold',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='New Point',
                showlegend=True
            )
        )

        # Highlight k nearest neighbors
        pred_fig.add_trace(
            go.Scatter(
                x=neighbor_points[:, 0],
                y=neighbor_points[:, 1],
                mode='markers',
                marker=dict(
                    color='lime',
                    size=12,
                    symbol='circle-open',
                    line=dict(color='green', width=3)
                ),
                name=f'{k_slider.value} Nearest Neighbors',
                showlegend=True
            )
        )

        # Draw lines to nearest neighbors
        for _neighbor in neighbor_points:
            pred_fig.add_trace(
                go.Scatter(
                    x=[new_x, _neighbor[0]],
                    y=[new_y, _neighbor[1]],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        # Create prediction info
        neighbor_classes = y[indices[0]]
        class_counts = np.bincount(neighbor_classes, minlength=2)

        _prediction_info = f"""
🎯 **Prediction Results:**

**New Point:** ({new_x:.2f}, {new_y:.2f})

**Predicted Class:** {prediction} ({_pred_class_names[prediction]})

**Confidence:** {probabilities[prediction]:.3f}

**k={k_slider.value} Nearest Neighbors:**
- Class 0: {class_counts[0]} neighbors
- Class 1: {class_counts[1]} neighbors

**Decision:** Majority vote = Class {prediction}
        """

    # Update layout
    pred_fig.update_layout(
        title=f"k-NN Interactive Prediction (k={k_slider.value})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        width=700,
        height=500,
        hovermode='closest'
    )

    # Display info
    if _prediction_info:
        _pred_info = mo.md(_prediction_info)
    else:
        _pred_info = mo.md("👆 Enter coordinates above and check the box to see k-NN in action!")

    (pred_fig, _pred_info)
    return pred_fig, _pred_info


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🎓 Key Takeaways

        **What you've learned:**

        1. **k-NN Basics**: The algorithm classifies points based on the majority class of their k nearest neighbors

        2. **Impact of k**:
           - **Small k**: More sensitive to noise, complex decision boundaries
           - **Large k**: Smoother boundaries, may oversimplify patterns
           - **Optimal k**: Found using validation/cross-validation (see elbow plot)

        3. **Decision Boundaries**: Visual representation of how the algorithm separates different classes

        4. **Model Evaluation**: Training vs. testing accuracy helps identify overfitting/underfitting

        5. **Interactive Prediction**: Real-time classification demonstrates how the algorithm works

        **Best Practices:**
        - Choose k using cross-validation
        - Use odd values of k to avoid ties (in binary classification)
        - Scale/normalize features when they have different units
        - Consider computational cost for large datasets

        **Next Steps:**
        - Experiment with different datasets
        - Try different distance metrics
        - Explore feature scaling effects
        - Compare with other classification algorithms
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        **📚 Additional Resources:**
        - Scikit-learn k-NN Documentation
        - "Pattern Recognition and Machine Learning" by Bishop
        - "The Elements of Statistical Learning" by Hastie et al.

        *This interactive tutorial was created with Marimo for educational purposes.*
        """
    )
    return


if __name__ == "__main__":
    app.run()