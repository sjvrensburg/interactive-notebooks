# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", app_title="Ridge & LASSO Geometry")


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, mo, np


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Geometry of Ridge Regression and LASSO

        Both methods solve a **penalised** least-squares problem, which is
        equivalent (for some budget $t$) to a **constrained** one:

        $$\min_{\boldsymbol{\beta}}\;(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})' \mathbf{X}'\mathbf{X}\,(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})
        \quad\text{subject to}\quad
        \begin{cases}\|\boldsymbol{\beta}\|_2 \le t & \text{(Ridge)}\\[2pt] \|\boldsymbol{\beta}\|_1 \le t & \text{(LASSO)}\end{cases}$$

        The solution is the **first point** where an expanding RSS ellipse —
        centred at the OLS estimate $\hat{\boldsymbol{\beta}}$ — touches the
        constraint region. Ridge's circle is smooth, so contact happens on an
        **arc** (coefficients shrunk, never zero). LASSO's diamond has
        **corners on the axes**, so contact frequently occurs at a corner —
        producing an **exactly zero** coefficient (sparsity).

        **Controls** — adjust $\rho$, $\hat{\beta}_1$, $\hat{\beta}_2$, and the
        penalty $\lambda$ to watch the contact point move.
        """
    )
    return


# -------------------------------------------------------------------
# Estimator helpers (closed-form ridge, coordinate-descent lasso)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(np):
    def soft_threshold(x, gamma):
        """Soft-thresholding operator S(x, γ) = sign(x) · max(|x| − γ, 0)."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)

    def ridge_solution(A, b, lam):
        """Ridge: β = (A + λI)⁻¹ b, with A = X'X and b = X'X β̂_ols."""
        return np.linalg.solve(A + lam * np.eye(A.shape[0]), b)

    def lasso_coordinate_descent(A, b, lam, n_iter=500, tol=1e-12):
        """LASSO via coordinate descent on ½(β−β̂)'A(β−β̂) + λ‖β‖₁."""
        p = A.shape[0]
        beta = np.zeros(p)
        for _ in range(n_iter):
            beta_old = beta.copy()
            for j in range(p):
                # ρ_j = b_j − Σ_{k≠j} A_jk β_k  (linear term excluding j)
                rho_j = b[j] - A[j] @ beta + A[j, j] * beta[j]
                beta[j] = soft_threshold(rho_j, lam) / A[j, j]
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta

    def ellipse_points(center, A, level, n=160):
        """Points {β : (β−c)' A (β−c) = level} via eigendecomposition of A."""
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 1e-12, None)
        theta = np.linspace(0, 2 * np.pi, n)
        unit = np.vstack([np.cos(theta), np.sin(theta)])          # (2, n)
        scaled = V @ (np.sqrt(level / w)[:, None] * unit)         # (2, n)
        return center[:, None] + scaled                            # (2, n)
    return ellipse_points, lasso_coordinate_descent, ridge_solution, soft_threshold


# -------------------------------------------------------------------
# UI Controls
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    rho = mo.ui.slider(-0.95, 0.95, value=0.70, step=0.05, label="ρ = Corr(X₁, X₂)")
    beta1 = mo.ui.slider(-3.0, 3.0, value=2.00, step=0.1, label="β̂₁ (OLS)")
    beta2 = mo.ui.slider(-3.0, 3.0, value=0.80, step=0.1, label="β̂₂ (OLS)")
    lam = mo.ui.slider(0.0, 5.0, value=1.20, step=0.05, label="λ (penalty)")
    show_ols = mo.ui.checkbox(value=True, label="Show OLS point β̂")
    show_contours = mo.ui.checkbox(value=True, label="Show nested RSS contours")
    show_labels = mo.ui.checkbox(value=True, label="Show point labels")

    controls_grid = mo.md(
        f"""
        | Parameter             | Value           |
        |:----------------------|:----------------|
        | Correlation (ρ)       | {rho}           |
        | OLS β̂₁               | {beta1}         |
        | OLS β̂₂               | {beta2}         |
        | Penalty (λ)           | {lam}           |
        | Show OLS β̂           | {show_ols}      |
        | Show contours         | {show_contours} |
        | Show labels           | {show_labels}   |
        """
    )
    return beta1, beta2, controls_grid, lam, rho, show_contours, show_labels, show_ols


# -------------------------------------------------------------------
# Solve the problem for the current parameters
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    beta1, beta2, ellipse_points, lasso_coordinate_descent, lam, np,
    ridge_solution, rho,
):
    # OLS centre of the RSS ellipses
    beta_ols = np.array([beta1.value, beta2.value])

    # X'X from the correlation structure (standardised regressors)
    A = np.array([[1.0, rho.value], [rho.value, 1.0]])
    b = A @ beta_ols                       # = X'X β̂_ols  (= X'y)

    lam_val = lam.value
    ridge_beta = ridge_solution(A, b, lam_val)
    lasso_beta = lasso_coordinate_descent(A, b, lam_val)

    # Equivalent constraint budgets (size of each region)
    t_ridge = float(np.linalg.norm(ridge_beta))        # ‖β‖₂
    t_lasso = float(np.sum(np.abs(lasso_beta)))        # ‖β‖₁

    # RSS contour level passing through each solution (the "contact" ellipse)
    def rss_level(beta):
        d = beta - beta_ols
        return float(d @ A @ d)
    c_ridge = rss_level(ridge_beta)
    c_lasso = rss_level(lasso_beta)

    # How sparse is the LASSO solution? (corner vs face contact)
    lasso_zeroed = [j for j in range(2) if abs(lasso_beta[j]) < 1e-9]
    n_zero = len(lasso_zeroed)

    return A, b, beta_ols, c_lasso, c_ridge, lasso_beta, lasso_zeroed, n_zero, ridge_beta, t_lasso, t_ridge


# -------------------------------------------------------------------
# Colour palette
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    COLOUR_RIDGE = "#0072BD"          # blue
    COLOUR_LASSO = "#D9531A"          # orange
    COLOUR_OLS = "#006B34"            # green
    COLOUR_CONSTRAINT = "#7E2F8E"     # purple
    COLOUR_CONTOUR = "#9aa0a6"        # grey
    COLOUR_CONTACT = "#a30044"        # deep red — the contact ellipse
    return COLOUR_CONSTRAINT, COLOUR_CONTACT, COLOUR_CONTOUR, COLOUR_LASSO, COLOUR_OLS, COLOUR_RIDGE


# -------------------------------------------------------------------
# Build the geometry figure (LASSO diamond | Ridge circle)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    A, COLOUR_CONSTRAINT, COLOUR_CONTACT, COLOUR_CONTOUR, COLOUR_LASSO,
    COLOUR_OLS, COLOUR_RIDGE, beta_ols, c_lasso, c_ridge, ellipse_points,
    go, lasso_beta, make_subplots, np, ridge_beta, show_contours, show_labels,
    show_ols, t_lasso, t_ridge,
):
    # Shared axis range so both panels share the same geometry
    _max = max(np.linalg.norm(beta_ols), t_ridge, t_lasso, 1.0) * 1.35
    _range = [-_max, _max]

    def _contour_traces(level_star, colour_star):
        """Nested faint RSS contours + one bold 'contact' contour."""
        traces = []
        if show_contours.value and level_star > 1e-12:
            for frac in (0.12, 0.25, 0.42, 0.62, 0.82):
                pts = ellipse_points(beta_ols, A, level_star * frac)
                traces.append(go.Scatter(
                    x=pts[0], y=pts[1], mode="lines",
                    line=dict(color=COLOUR_CONTOUR, width=1, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                ))
            pts = ellipse_points(beta_ols, A, level_star)
            traces.append(go.Scatter(
                x=pts[0], y=pts[1], mode="lines",
                line=dict(color=colour_star, width=2.4),
                name="RSS contour at solution",
                hoverinfo="name", showlegend=False,
            ))
        return traces

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"LASSO  —  ℓ₁ ball  (t = ‖β̂‖₁ = {t_lasso:.3f})",
            f"Ridge  —  ℓ₂ ball  (t = ‖β̂‖₂ = {t_ridge:.3f})",
        ),
        horizontal_spacing=0.12,
    )

    # ---- LASSO panel (col 1): diamond constraint region ----
    tL = t_lasso
    diamond_x = [tL, 0, -tL, 0, tL]
    diamond_y = [0, tL, 0, -tL, 0]
    _fig.add_trace(go.Scatter(
        x=diamond_x, y=diamond_y, fill="toself", mode="lines",
        line=dict(color=COLOUR_CONSTRAINT, width=2.5),
        fillcolor="rgba(126,47,142,0.12)",
        name="ℓ₁ constraint", hoverinfo="name", showlegend=False,
    ), row=1, col=1)
    for _tr in _contour_traces(c_lasso, COLOUR_CONTACT):
        _fig.add_trace(_tr, row=1, col=1)
    _fig.add_trace(go.Scatter(
        x=[lasso_beta[0]], y=[lasso_beta[1]], mode="markers",
        marker=dict(color=COLOUR_LASSO, size=13, line=dict(color="white", width=2)),
        name="LASSO solution", hovertemplate="LASSO<br>β₁=%{x:.3f}<br>β₂=%{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    if show_ols.value:
        _fig.add_trace(go.Scatter(
            x=[beta_ols[0]], y=[beta_ols[1]], mode="markers",
            marker=dict(color=COLOUR_OLS, size=12, symbol="star", line=dict(color="white", width=1.5)),
            name="OLS β̂", hovertemplate="OLS<br>β̂₁=%{x:.3f}<br>β̂₂=%{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
    if show_labels.value:
        _fig.add_trace(go.Scatter(
            x=[lasso_beta[0]], y=[lasso_beta[1]], mode="text",
            text=["β̂_lasso"], textposition="top right",
            textfont=dict(color=COLOUR_LASSO, size=12), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # ---- Ridge panel (col 2): circular constraint region ----
    tR = t_ridge
    _ang = np.linspace(0, 2 * np.pi, 200)
    _fig.add_trace(go.Scatter(
        x=tR * np.cos(_ang), y=tR * np.sin(_ang), fill="toself", mode="lines",
        line=dict(color=COLOUR_CONSTRAINT, width=2.5),
        fillcolor="rgba(126,47,142,0.12)",
        name="ℓ₂ constraint", hoverinfo="name", showlegend=False,
    ), row=1, col=2)
    for _tr in _contour_traces(c_ridge, COLOUR_CONTACT):
        _fig.add_trace(_tr, row=1, col=2)
    _fig.add_trace(go.Scatter(
        x=[ridge_beta[0]], y=[ridge_beta[1]], mode="markers",
        marker=dict(color=COLOUR_RIDGE, size=13, line=dict(color="white", width=2)),
        name="Ridge solution", hovertemplate="Ridge<br>β₁=%{x:.3f}<br>β₂=%{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    if show_ols.value:
        _fig.add_trace(go.Scatter(
            x=[beta_ols[0]], y=[beta_ols[1]], mode="markers",
            marker=dict(color=COLOUR_OLS, size=12, symbol="star", line=dict(color="white", width=1.5)),
            name="OLS β̂", hoverinfo="skip", showlegend=False,
        ), row=1, col=2)
    if show_labels.value:
        _fig.add_trace(go.Scatter(
            x=[ridge_beta[0]], y=[ridge_beta[1]], mode="text",
            text=["β̂_ridge"], textposition="top right",
            textfont=dict(color=COLOUR_RIDGE, size=12), showlegend=False, hoverinfo="skip",
        ), row=1, col=2)

    # Equal-aspect axes (square geometry is the whole point)
    _fig.update_xaxes(range=_range, zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc",
                      scaleanchor="y", scaleratio=1, title="β₁", showgrid=False, row=1, col=1)
    _fig.update_yaxes(range=_range, zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc",
                      title="β₂", showgrid=False, row=1, col=1)
    _fig.update_xaxes(range=_range, zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc",
                      scaleanchor="y", scaleratio=1, title="β₁", showgrid=False, row=1, col=2)
    _fig.update_yaxes(range=_range, zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc",
                      title="β₂", showgrid=False, row=1, col=2)

    _fig.update_layout(
        height=480, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", uirevision="geometry",
    )
    geometry_fig = _fig
    return (geometry_fig,)


# -------------------------------------------------------------------
# Build the coefficient-paths figure (β vs λ)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    A, COLOUR_LASSO, COLOUR_OLS, COLOUR_RIDGE, b, go, lasso_coordinate_descent,
    lam, np, ridge_solution,
):
    _lam_grid = np.linspace(0.01, 5.0, 80)
    ridge_path = np.array([ridge_solution(A, b, lm) for lm in _lam_grid])      # (80,2)
    lasso_path = np.array([lasso_coordinate_descent(A, b, lm) for lm in _lam_grid])

    _fig = go.Figure()
    # Ridge paths (smooth)
    _fig.add_trace(go.Scatter(
        x=_lam_grid, y=ridge_path[:, 0], mode="lines",
        line=dict(color=COLOUR_RIDGE, width=2.5), name="Ridge β̂₁",
    ))
    _fig.add_trace(go.Scatter(
        x=_lam_grid, y=ridge_path[:, 1], mode="lines",
        line=dict(color=COLOUR_RIDGE, width=2.5, dash="dash"), name="Ridge β̂₂",
    ))
    # LASSO paths (piecewise linear — note the flat segments at 0)
    _fig.add_trace(go.Scatter(
        x=_lam_grid, y=lasso_path[:, 0], mode="lines",
        line=dict(color=COLOUR_LASSO, width=2.5), name="LASSO β̂₁",
    ))
    _fig.add_trace(go.Scatter(
        x=_lam_grid, y=lasso_path[:, 1], mode="lines",
        line=dict(color=COLOUR_LASSO, width=2.5, dash="dash"), name="LASSO β̂₂",
    ))
    # OLS asymptotes
    _fig.add_trace(go.Scatter(
        x=[_lam_grid[0], _lam_grid[-1]], y=[b[0], b[0]], mode="lines",
        line=dict(color=COLOUR_OLS, width=1, dash="dot"), name="OLS β̂₁",
    ))
    _fig.add_trace(go.Scatter(
        x=[_lam_grid[0], _lam_grid[-1]], y=[b[1], b[1]], mode="lines",
        line=dict(color=COLOUR_OLS, width=1, dash="dot"), name="OLS β̂₂",
    ))
    # Current-λ marker
    _fig.add_vline(x=lam.value, line=dict(color="#444444", width=1.5, dash="solid"))

    _fig.update_layout(
        title="Coefficient paths — β̂ vs λ  (LASSO hits exactly 0; Ridge only shrinks)",
        xaxis=dict(title="λ (penalty)", gridcolor="#eeeeee"),
        yaxis=dict(title="β̂", gridcolor="#eeeeee", zeroline=True, zerolinewidth=1, zerolinecolor="#cccccc"),
        height=430, margin=dict(l=50, r=20, t=55, b=45),
        legend=dict(orientation="h", y=-0.22, x=0.0),
        uirevision="paths",
    )
    paths_fig = _fig
    return (paths_fig,)


# -------------------------------------------------------------------
# Dynamic contact-geometry explanation (colour-coded)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_LASSO, COLOUR_OLS, COLOUR_RIDGE, beta_ols, lam, lasso_beta, mo, n_zero,
    ridge_beta, rho,
):
    def _c(colour, text):
        return f'<span style="color:{colour};font-weight:bold">{text}</span>'

    _ols = _c(COLOUR_OLS, f"β̂ = ({beta_ols[0]:.2f}, {beta_ols[1]:.2f})")
    _ridge = _c(COLOUR_RIDGE, f"β̂_ridge = ({ridge_beta[0]:.3f}, {ridge_beta[1]:.3f})")
    _lasso = _c(COLOUR_LASSO, f"β̂_lasso = ({lasso_beta[0]:.3f}, {lasso_beta[1]:.3f})")

    if n_zero == 0:
        _lasso_note = (
            f"At λ = {lam.value:.2f} the contact lies on a **face** of the "
            f"diamond, so {_lasso} has **no zero** coefficient — both variables "
            f"survive (but shrunk)."
        )
    else:
        zero_which = "β̂₂" if abs(lasso_beta[1]) < 1e-9 else "β̂₁"
        _lasso_note = (
            f"At λ = {lam.value:.2f} the first ellipse touches a **corner** of "
            f"the diamond, so {_lasso} sets **{zero_which} = 0** — a sparse "
            f"solution. This corner geometry is why LASSO performs variable "
            f"selection."
        )

    collinear = (
        f" ⚠️ High collinearity (ρ = {rho.value:.2f}) stretches the ellipses "
        f"along the 45° diagonal, making LASSO's sparsity much more likely."
        if abs(rho.value) > 0.8 else ""
    )

    explanation = mo.md(
        f"""
        ### Reading the picture

        The nested {_c(COLOUR_OLS, 'ellipses')} are level sets of the RSS,
        all centred at the OLS estimate {_ols}. We expand them outward until
        each one first touches its constraint region.

        - **Ridge** — the ℓ₂ constraint is a **smooth circle**, so contact is
          always on an arc: {_ridge}. Coefficients are **shrunk toward zero
          but never reach it**.
        - **LASSO** — the ℓ₁ constraint is a **diamond with corners on the
          axes**, so contact often lands on a corner: {_lasso}.

        {_lasso_note}{collinear}
        """
    )
    return (explanation,)


# -------------------------------------------------------------------
# Solution / statistics panel
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    beta_ols, lasso_beta, lam, mo, n_zero, ridge_beta, rho, t_lasso, t_ridge,
):
    def _shrink(beta):
        n = np.linalg.norm(beta)
        d = np.linalg.norm(beta_ols)
        return (n / d) if d > 1e-12 else float("nan")

    stats_md = mo.md(
        f"""
        ### Solutions at λ = {lam.value:.2f}   (ρ = {rho.value:.2f})

        | Method | β̂₁ | β̂₂ | Constraint budget | Shrinkage ‖β̂‖₂/‖β̂_ols‖₂ |
        |:--|--:|--:|--:|--:|
        | **OLS**    | {beta_ols[0]:.4f} | {beta_ols[1]:.4f} | — | 1.000 |
        | **Ridge**  | {ridge_beta[0]:.4f} | {ridge_beta[1]:.4f} | t₂ = {t_ridge:.4f} | {_shrink(ridge_beta):.3f} |
        | **LASSO**  | {lasso_beta[0]:.4f} | {lasso_beta[1]:.4f} | t₁ = {t_lasso:.4f} | {_shrink(lasso_beta):.3f} |

        LASSO zeroes **{n_zero}** coefficient(s) at this λ.
        Ridge **never** produces an exact zero — it only shrinks.
        """
    )
    return (stats_md,)


# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    controls_grid, explanation, geometry_fig, mo, paths_fig, stats_md,
):
    mo.vstack([
        mo.hstack(
            [
                mo.vstack([mo.md("### Parameters"), controls_grid]),
                mo.ui.tabs({
                    "📐 Constraint Geometry": mo.vstack([geometry_fig, explanation]),
                    "📈 Coefficient Paths": paths_fig,
                    "📋 Solutions": stats_md,
                }),
            ],
            widths=[1, 4],
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
