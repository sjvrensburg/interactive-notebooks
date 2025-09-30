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
        # Classification and Regression Trees (CART): Interactive Demonstration
        ## Understanding Tree Growth and Cost-Complexity Pruning

        Welcome to this interactive exploration of **Classification and Regression Trees (CART)**! This notebook demonstrates how decision trees grow in complexity and how **Cost-Complexity Pruning** helps control overfitting.

        ## What are Classification Trees?

        Decision trees are non-parametric supervised learning methods that recursively partition the feature space into regions based on simple decision rules. Unlike linear models that assume global functional forms, trees construct piecewise constant approximations to capture local patterns.

        **🎯 The Core Concept:**

        - Recursively split the feature space based on feature thresholds
        - Each split maximises information gain (minimises impurity)
        - Terminal nodes (leaves) make class predictions
        - Tree depth controls model complexity

        **📊 Mathematical Foundation:**

        **Gini Impurity** (for classification):
        $$\text{Gini}(t) = 1 - \sum_{i=1}^C p_i^2$$

        where $p_i$ is the proportion of class $i$ in node $t$.

        **Cost-Complexity Pruning:**
        $$R_\alpha(T) = R(T) + \alpha |T|$$

        where:
        - $R(T)$ is the misclassification rate
        - $|T|$ is the number of terminal nodes (complexity)
        - $\alpha \geq 0$ is the complexity parameter (pruning strength)

        **🔧 Key Features:**

        - **Max Depth**: Controls how deep the tree can grow (complexity)
        - **Pruning $\alpha$**: Trades off accuracy for simplicity by removing weak branches
        - **Decision Boundary**: Non-linear boundaries that adapt to data structure
        """
    )
    return


@app.cell
def __():
    # Import required libraries
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.datasets import make_moons
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd

    # Set random seed for reproducibility
    np.random.seed(2025)
    return (
        DecisionTreeClassifier,
        accuracy_score,
        export_text,
        go,
        make_moons,
        make_subplots,
        np,
        pd,
        train_test_split,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Sample Dataset Generation

        We'll use a **two-moons** synthetic dataset, which is ideal for demonstrating non-linear decision boundaries. This dataset has two interleaving half-circles, making it a classic test case for non-linear classifiers.
        """
    )
    return


@app.cell
def __(make_moons, np, train_test_split):
    # Generate synthetic two-moons dataset
    _X_full, _y_full = make_moons(n_samples=300, noise=0.25, random_state=2025)

    # Split into train (70%) and test (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        _X_full, _y_full, test_size=0.3, random_state=2025, stratify=_y_full
    )

    return X_test, X_train, y_test, y_train


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interactive Controls

        Adjust the parameters below to explore how tree complexity and pruning affect model behaviour:

        **Max Depth:** Controls how deep the tree can grow. Deeper trees capture more complex patterns but risk overfitting.

        **Pruning Parameter ($\alpha$):** Controls cost-complexity pruning. Higher $\alpha$ values produce simpler trees by removing weak branches that don't sufficiently improve accuracy.
        """
    )
    return


@app.cell
def __(mo):
    # Interactive control for tree growth (horizontal)
    max_depth_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=5,
        label="Maximum Tree Depth:"
    )

    mo.md(f"""
    **Tree Growth Control:**
    {max_depth_slider}
    """)
    return max_depth_slider,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Tree Structure Visualisation

        The diagram shows the current tree structure. When α = 0, you see the full unpruned tree. As you increase α, weak branches are removed.

        **💡 Controls:**

        - Use the vertical α slider (left) to interactively prune the tree
        - Use browser zoom (Ctrl/Cmd + scroll or +/-) to zoom the diagram
        """
    )
    return


@app.cell
def __(mo):
    # Vertical slider for pruning control
    ccp_alpha_slider = mo.ui.slider(
        start=0.0, stop=0.1, step=0.005, value=0.0,
        label="Pruning α:",
        orientation="vertical"
    )
    return ccp_alpha_slider,


@app.cell
def __(
    DecisionTreeClassifier,
    X_train,
    ccp_alpha_slider,
    max_depth_slider,
    np,
    y_train,
):
    # Build the tree with current max_depth and ccp_alpha settings
    current_tree = DecisionTreeClassifier(
        max_depth=max_depth_slider.value,
        ccp_alpha=ccp_alpha_slider.value,
        random_state=2025,
        criterion='gini',
        min_impurity_decrease=0.0001,  # Prevent splits with negligible improvement
        min_samples_split=5,            # Require at least 5 samples to split
        min_samples_leaf=2              # Require at least 2 samples in each leaf
    )
    current_tree.fit(X_train, y_train)

    # Tree to Mermaid conversion function
    def tree_to_mermaid(tree, feature_names=None):
        """
        Convert sklearn DecisionTreeClassifier to Mermaid.js graph syntax.

        Args:
            tree: Fitted DecisionTreeClassifier
            feature_names: Optional list of feature names

        Returns:
            String containing Mermaid.js graph definition
        """
        _tree = tree.tree_
        if feature_names is None:
            feature_names = [f"X[{_i}]" for _i in range(_tree.n_features)]

        _lines = ["graph TD;"]

        def _recurse(node_id, depth=0):
            """Recursively build Mermaid nodes and edges."""
            _indent = "    " * depth

            # Check if terminal node
            if _tree.feature[node_id] == -2:
                # Terminal node - show class prediction
                _class_counts = _tree.value[node_id][0]
                _predicted_class = int(np.argmax(_class_counts))
                _gini = _tree.impurity[node_id]
                _samples = int(_tree.n_node_samples[node_id])
                _lines.append(
                    f'{_indent}node{node_id}["Class {_predicted_class}<br/>Gini: {_gini:.3f}<br/>Samples: {_samples}"]'
                )
                return

            # Internal node - show split condition
            _feature_idx = _tree.feature[node_id]
            _threshold = _tree.threshold[node_id]
            _gini = _tree.impurity[node_id]
            _samples = int(_tree.n_node_samples[node_id])

            _lines.append(
                f'{_indent}node{node_id}["{feature_names[_feature_idx]} ≤ {_threshold:.3f}<br/>Gini: {_gini:.3f}<br/>Samples: {_samples}"]'
            )

            # Get children
            _left_child = _tree.children_left[node_id]
            _right_child = _tree.children_right[node_id]

            # Add edges
            if _left_child != -1:
                _lines.append(f'{_indent}node{node_id} -->|True| node{_left_child}')
                _recurse(_left_child, depth + 1)

            if _right_child != -1:
                _lines.append(f'{_indent}node{node_id} -->|False| node{_right_child}')
                _recurse(_right_child, depth + 1)

        _recurse(0)
        return "\n".join(_lines)

    # Generate Mermaid diagram
    current_tree_mermaid = tree_to_mermaid(current_tree)
    
    # Also generate text representation for WASM compatibility
    current_tree_text = export_text(current_tree, feature_names=["X[0]", "X[1]"])

    return current_tree, current_tree_mermaid, current_tree_text, tree_to_mermaid


@app.cell
def __(
    X_test,
    X_train,
    accuracy_score,
    ccp_alpha_slider,
    current_tree,
    current_tree_mermaid,
    current_tree_text,
    mo,
    y_test,
    y_train,
):
    import sys
    
    # Calculate metrics for current tree
    _train_acc = accuracy_score(y_train, current_tree.predict(X_train))
    _test_acc = accuracy_score(y_test, current_tree.predict(X_test))
    _n_nodes = current_tree.tree_.node_count

    # Detect if running in WASM environment (Pyodide)
    _is_wasm = 'pyodide' in sys.modules or 'emscripten' in sys.modules
    
    if _is_wasm:
        # Use text representation in WASM
        _tree_viz = mo.md(f"""**Text-based Tree (WASM mode):**
```
{current_tree_text}
```""")
    else:
        # Use Mermaid for interactive viewing
        _tree_viz = mo.mermaid(current_tree_mermaid)
    
    # Create the tree diagram with metrics
    _tree_content = mo.vstack([
        _tree_viz,
        mo.md(f"""
        **Current Tree Metrics:**
        Training Accuracy: {_train_acc:.3f} | Test Accuracy: {_test_acc:.3f} | Nodes: {_n_nodes} | α: {ccp_alpha_slider.value:.3f} {'**(Unpruned)**' if ccp_alpha_slider.value == 0.0 else '**(Pruned)**'}
        """)
    ], gap=0.5)

    # Display vertical slider next to tree with minimal gap
    mo.hstack([
        ccp_alpha_slider,
        _tree_content
    ], justify="start", gap=1)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Decision Boundary Visualisation

        The plot below shows how the pruned tree partitions the 2D feature space. Training points are shown with filled markers, whilst test points use hollow markers. The background colour represents the predicted class in each region.
        """
    )
    return


@app.cell
def __(
    X_test,
    X_train,
    current_tree,
    go,
    np,
    y_test,
    y_train,
):
    # Create a mesh for decision boundary visualisation
    _x_min, _x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    _y_min, _y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    _xx, _yy = np.meshgrid(
        np.linspace(_x_min, _x_max, 200),
        np.linspace(_y_min, _y_max, 200)
    )

    # Predict on the mesh using current tree
    _Z = current_tree.predict(np.c_[_xx.ravel(), _yy.ravel()])
    _Z = _Z.reshape(_xx.shape)

    # Create decision boundary plot
    decision_boundary_fig = go.Figure()

    # Add decision boundary as contour
    decision_boundary_fig.add_trace(go.Contour(
        x=np.linspace(_x_min, _x_max, 200),
        y=np.linspace(_y_min, _y_max, 200),
        z=_Z,
        colorscale=[[0, 'rgba(31,119,180,0.3)'], [1, 'rgba(255,127,14,0.3)']],
        showscale=False,
        hoverinfo='skip',
        contours=dict(
            coloring='heatmap',
            showlabels=False
        )
    ))

    # Add training data
    for _class_idx in [0, 1]:
        _mask = y_train == _class_idx
        decision_boundary_fig.add_trace(go.Scatter(
            x=X_train[_mask, 0],
            y=X_train[_mask, 1],
            mode='markers',
            marker=dict(
                size=8,
                color='blue' if _class_idx == 0 else 'orange',
                symbol='circle',
                line=dict(width=1, color='darkblue' if _class_idx == 0 else 'darkorange')
            ),
            name=f'Class {_class_idx} (Train)',
            hovertemplate=f'Class {_class_idx} (Train)<br>X1: %{{x:.2f}}<br>X2: %{{y:.2f}}<extra></extra>'
        ))

    # Add test data
    for _class_idx in [0, 1]:
        _mask = y_test == _class_idx
        decision_boundary_fig.add_trace(go.Scatter(
            x=X_test[_mask, 0],
            y=X_test[_mask, 1],
            mode='markers',
            marker=dict(
                size=8,
                color='blue' if _class_idx == 0 else 'orange',
                symbol='circle-open',
                line=dict(width=2, color='darkblue' if _class_idx == 0 else 'darkorange')
            ),
            name=f'Class {_class_idx} (Test)',
            hovertemplate=f'Class {_class_idx} (Test)<br>X1: %{{x:.2f}}<br>X2: %{{y:.2f}}<extra></extra>'
        ))

    decision_boundary_fig.update_layout(
        title='Decision Boundary (Current Tree)',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        width=800,
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    decision_boundary_fig
    return decision_boundary_fig,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Understanding Cost-Complexity Pruning

        **The Pruning Trade-off:**

        Cost-complexity pruning (also called weakest-link pruning) removes branches that provide minimal improvement in classification accuracy relative to their added complexity.

        **How it works:**

        1. **Grow a full tree** to the specified `max_depth`
        2. **Calculate cost-complexity** for each subtree: $R_\alpha(T) = R(T) + \alpha |T|$
        3. **Prune weakest links** iteratively: Remove subtrees where the complexity cost exceeds the accuracy benefit
        4. **Select optimal $\alpha$** via cross-validation (or manual tuning as demonstrated here)

        **Key Insights:**

        - **$\alpha = 0$**: No pruning (full tree up to max depth)
        - **Small $\alpha$**: Minimal pruning, removes only very weak branches
        - **Large $\alpha$**: Aggressive pruning, may reduce to a stump or very simple tree
        - **Optimal $\alpha$**: Balances test accuracy and model simplicity

        **Practical Considerations:**

        - **Bias-Variance Trade-off**: Pruning increases bias slightly but can dramatically reduce variance
        - **Generalisation**: Pruned trees often achieve better test accuracy despite lower training accuracy
        - **Interpretability**: Simpler trees are easier to explain and visualise
        - **Computational Efficiency**: Fewer nodes means faster predictions

        **💡 Experiment:** Try adjusting both sliders to observe:

        - How max depth controls initial tree complexity
        - How $\alpha$ removes weak branches from the grown tree
        - The effect on training vs. test accuracy (watch for overfitting!)
        - Changes in decision boundary smoothness
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Summary and Key Takeaways

        **Classification Trees (CART):**

        - Partition feature space recursively using simple threshold rules
        - Use impurity measures (Gini index) to determine optimal splits
        - Flexible non-linear decision boundaries

        **Tree Growth (Max Depth):**

        - Controls model complexity directly
        - Deeper trees capture more intricate patterns but risk overfitting
        - Limited depth acts as a form of regularisation

        **Cost-Complexity Pruning ($\alpha$):**

        - Post-hoc regularisation technique
        - Removes branches with poor cost-benefit ratio
        - Often improves generalisation by reducing overfitting
        - Produces more interpretable models

        **Best Practices:**

        - Use proper train/test splits for unbiased evaluation
        - Monitor both training and test accuracy (gap indicates overfitting)
        - Consider computational cost: simpler trees predict faster
        - Tune both max depth and $\alpha$ via cross-validation for optimal performance

        **Extensions and Advanced Topics:**

        - **Random Forests**: Ensemble of trees with feature randomness
        - **Gradient Boosting**: Sequential trees correcting previous errors
        - **Regression Trees**: CART applied to continuous outcomes
        - **Feature Importance**: Quantify predictive power of each feature
        """
    )
    return


if __name__ == "__main__":
    app.run()
