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
app = marimo.App(width="full", app_title="OLS Geometry Explorer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return go, mo, np


# -------------------------------------------------------------------
# Core linear-algebra helpers (kept separate for future FWL re-use)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(np):
    def projection_matrix(X):
        """Compute the projection matrix P_X = X (X'X)^{-1} X'."""
        return X @ np.linalg.solve(X.T @ X, X.T)

    def residual_maker(X):
        """Compute the residual-maker matrix M_X = I - P_X."""
        n = X.shape[0]
        return np.eye(n) - projection_matrix(X)

    def ols_fit(X, y):
        """Return (beta_hat, y_hat, residuals) from OLS."""
        beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
        y_hat = X @ beta_hat
        e = y - y_hat
        return beta_hat, y_hat, e

    def calc_r_squared(y, e):
        """Centred R-squared."""
        ss_res = e @ e
        y_bar = y.mean()
        ss_tot = (y - y_bar) @ (y - y_bar)
        if ss_tot == 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def calc_r_squared_uncentred(y, y_hat):
        """Uncentred R-squared."""
        yy = y @ y
        if yy == 0:
            return float("nan")
        return (y_hat @ y_hat) / yy
    return (
        calc_r_squared,
        calc_r_squared_uncentred,
        ols_fit,
        projection_matrix,
        residual_maker,
    )


# -------------------------------------------------------------------
# Colour palette and plotting helpers
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(go, np):
    NMU_GREEN = "#006B34"
    NMU_DARK_BLUE = "#18324B"
    NMU_PURPLE = "#6C284F"

    COLOUR_X = "#0072BD"
    COLOUR_Y = "#D9531A"
    COLOUR_YHAT = NMU_GREEN
    COLOUR_RESIDUAL = NMU_PURPLE
    COLOUR_CUSTOM = "#E6AB02"  # gold — custom/auxiliary vector
    COLOUR_PLANE = "rgba(200,200,200,0.35)"
    COLOUR_RIGHT_ANGLE = "#888888"

    def arrow_cone(tip, direction, colour, size=0.08):
        d = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            d = np.array([0.0, 0.0, 1.0])
        else:
            d = d / norm
        return go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[d[0]], v=[d[1]], w=[d[2]],
            sizemode="absolute", sizeref=size,
            showscale=False,
            colorscale=[[0, colour], [1, colour]],
            hoverinfo="skip",
        )

    def vector_trace(origin, tip, colour, name, width=5, dash=None, show_arrow=True):
        o = np.asarray(origin, dtype=float)
        t = np.asarray(tip, dtype=float)
        line = go.Scatter3d(
            x=[o[0], t[0]], y=[o[1], t[1]], z=[o[2], t[2]],
            mode="lines",
            line=dict(color=colour, width=width, dash=dash),
            name=name,
            hoverinfo="name",
            showlegend=True,
        )
        traces = [line]
        if show_arrow:
            traces.append(arrow_cone(t, t - o, colour))
        return traces

    def plane_mesh(x1, x2, colour=COLOUR_PLANE, n_grid=6):
        origin = np.zeros(3)
        s_vals = np.linspace(-0.3, 1.3, n_grid)
        t_vals = np.linspace(-0.3, 1.3, n_grid)
        ss, tt = np.meshgrid(s_vals, t_vals)
        pts = (
            origin[:, None, None]
            + np.outer(x1, np.ones(n_grid))[..., None] * ss[None, :, :]
            + np.outer(x2, np.ones(n_grid))[..., None] * tt[None, :, :]
        )
        return go.Mesh3d(
            x=pts[0].ravel(), y=pts[1].ravel(), z=pts[2].ravel(),
            color=colour, opacity=0.30,
            name="Column space", showlegend=True,
            hoverinfo="name", alphahull=0,
        )

    def right_angle_traces(vertex, arm_a, arm_b, size=0.18):
        a = np.asarray(arm_a, dtype=float)
        b = np.asarray(arm_b, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return []
        a_hat = a / na * size
        b_hat = b / nb * size
        v = np.asarray(vertex, dtype=float)
        pts = [v + a_hat, v + a_hat + b_hat, v + b_hat]
        return [go.Scatter3d(
            x=[p[0] for p in pts], y=[p[1] for p in pts], z=[p[2] for p in pts],
            mode="lines", line=dict(color=COLOUR_RIGHT_ANGLE, width=3),
            name="Right angle", showlegend=False, hoverinfo="skip",
        )]
    return (
        COLOUR_CUSTOM,
        COLOUR_PLANE,
        COLOUR_RESIDUAL,
        COLOUR_RIGHT_ANGLE,
        COLOUR_X,
        COLOUR_Y,
        COLOUR_YHAT,
        NMU_DARK_BLUE,
        NMU_GREEN,
        NMU_PURPLE,
        arrow_cone,
        plane_mesh,
        right_angle_traces,
        vector_trace,
    )


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Geometry of Ordinary Least Squares

        This interactive tool accompanies **Section 1.4** of the STAT321 lecture
        notes (`04-ols-geometry.pdf`). Change the vector entries below and use the tick boxes to explore
        how $\hat{\mathbf{y}}$, the residuals, and the projection matrices
        respond. **Click and drag** the 3D plot to rotate it; **scroll** to zoom.
        """
    )
    return


# -------------------------------------------------------------------
# Data-entry widgets (number spinners in a grid)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    x1_1 = mo.ui.number(value=2.5, step=0.1, label="", full_width=True)
    x1_2 = mo.ui.number(value=0.5, step=0.1, label="", full_width=True)
    x1_3 = mo.ui.number(value=0.5, step=0.1, label="", full_width=True)

    x2_1 = mo.ui.number(value=-0.5, step=0.1, label="", full_width=True)
    x2_2 = mo.ui.number(value=2.5, step=0.1, label="", full_width=True)
    x2_3 = mo.ui.number(value=0.3, step=0.1, label="", full_width=True)

    y_1 = mo.ui.number(value=2.0, step=0.1, label="", full_width=True)
    y_2 = mo.ui.number(value=1.0, step=0.1, label="", full_width=True)
    y_3 = mo.ui.number(value=2.5, step=0.1, label="", full_width=True)

    # Custom / auxiliary vector (default: ȳ·ι for demonstrating
    # that the mean vector lies in the column space)
    c_1 = mo.ui.number(value=1.83, step=0.1, label="", full_width=True)
    c_2 = mo.ui.number(value=1.83, step=0.1, label="", full_width=True)
    c_3 = mo.ui.number(value=1.83, step=0.1, label="", full_width=True)

    data_grid = mo.md(
        f"""
        | | **x\u2081** | **x\u2082** | **y** |
        |:---|:---:|:---:|:---:|
        | **Obs 1** | {x1_1} | {x2_1} | {y_1} |
        | **Obs 2** | {x1_2} | {x2_2} | {y_2} |
        | **Obs 3** | {x1_3} | {x2_3} | {y_3} |

        | | **Custom vector** |
        |:---|:---:|
        | **Obs 1** | {c_1} |
        | **Obs 2** | {c_2} |
        | **Obs 3** | {c_3} |
        """
    )
    return (
        c_1, c_2, c_3,
        data_grid,
        x1_1, x1_2, x1_3,
        x2_1, x2_2, x2_3,
        y_1, y_2, y_3,
    )


# -------------------------------------------------------------------
# Display-option tick boxes
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    show_column_space = mo.ui.checkbox(value=True, label="Column space (plane)")
    show_x1 = mo.ui.checkbox(value=True, label="x\u2081 vector")
    show_x2 = mo.ui.checkbox(value=True, label="x\u2082 vector")
    show_y = mo.ui.checkbox(value=True, label="y (observed)")
    show_y_hat = mo.ui.checkbox(value=True, label="\u0177 (fitted values)")
    show_residual = mo.ui.checkbox(value=True, label="y \u2212 \u0177 (residuals)")
    show_right_angle = mo.ui.checkbox(value=True, label="Right-angle marker")
    show_axes = mo.ui.checkbox(value=True, label="Observation axes")
    show_projection_lines = mo.ui.checkbox(value=False, label="\u0177 drop-lines to axes")
    show_custom = mo.ui.checkbox(value=False, label="Custom vector")

    tick_boxes = mo.hstack(
        [
            mo.vstack([show_column_space, show_x1, show_x2]),
            mo.vstack([show_y, show_y_hat, show_residual]),
            mo.vstack([show_right_angle, show_axes, show_projection_lines, show_custom]),
        ],
        gap=2,
    )
    return (
        show_axes,
        show_column_space,
        show_custom,
        show_projection_lines,
        show_residual,
        show_right_angle,
        show_x1,
        show_x2,
        show_y,
        show_y_hat,
        tick_boxes,
    )


# -------------------------------------------------------------------
# Compute OLS results (reactive — reruns when any spinner changes)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    np, ols_fit, projection_matrix,
    calc_r_squared, calc_r_squared_uncentred,
    c_1, c_2, c_3,
    x1_1, x1_2, x1_3,
    x2_1, x2_2, x2_3,
    y_1, y_2, y_3,
):
    x1_vec = np.array([x1_1.value, x1_2.value, x1_3.value], dtype=float)
    x2_vec = np.array([x2_1.value, x2_2.value, x2_3.value], dtype=float)
    y_vec = np.array([y_1.value, y_2.value, y_3.value], dtype=float)
    custom_vec = np.array([c_1.value, c_2.value, c_3.value], dtype=float)

    X_mat = np.column_stack([x1_vec, x2_vec])
    beta_hat, y_hat_vec, e_vec = ols_fit(X_mat, y_vec)

    P_mat = projection_matrix(X_mat)
    M_mat = np.eye(3) - P_mat

    ssr_val = float(e_vec @ e_vec)
    dist_val = float(np.sqrt(ssr_val))
    r2_val = calc_r_squared(y_vec, e_vec)
    r2u_val = calc_r_squared_uncentred(y_vec, y_hat_vec)
    return (
        M_mat,
        P_mat,
        X_mat,
        beta_hat,
        custom_vec,
        dist_val,
        e_vec,
        r2_val,
        r2u_val,
        ssr_val,
        x1_vec,
        x2_vec,
        y_hat_vec,
        y_vec,
    )


# -------------------------------------------------------------------
# Build the 3-D figure
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_CUSTOM, COLOUR_RESIDUAL, COLOUR_X, COLOUR_Y, COLOUR_YHAT,
    go, np,
    plane_mesh, right_angle_traces, vector_trace,
    custom_vec, e_vec, x1_vec, x2_vec, y_hat_vec, y_vec,
    show_axes, show_column_space, show_custom, show_projection_lines,
    show_residual, show_right_angle,
    show_x1, show_x2, show_y, show_y_hat,
):
    origin = np.zeros(3)
    traces = []

    if show_column_space.value:
        traces.append(plane_mesh(x1_vec, x2_vec))
    if show_x1.value:
        traces.extend(vector_trace(origin, x1_vec, COLOUR_X, "x\u2081"))
    if show_x2.value:
        traces.extend(vector_trace(origin, x2_vec, COLOUR_X, "x\u2082"))
    if show_y.value:
        traces.extend(vector_trace(origin, y_vec, COLOUR_Y, "y"))
    if show_y_hat.value:
        traces.extend(vector_trace(origin, y_hat_vec, COLOUR_YHAT, "\u0177 (fitted)"))
    if show_residual.value:
        traces.extend(
            vector_trace(y_hat_vec, y_vec, COLOUR_RESIDUAL,
                         "y \u2212 \u0177 (residual)", dash="dash")
        )
    if show_right_angle.value and show_residual.value and show_y_hat.value:
        traces.extend(right_angle_traces(y_hat_vec, e_vec, -y_hat_vec))
    if show_projection_lines.value:
        # Drop-lines from ŷ to each observation axis, showing the
        # individual fitted-value components.
        drop_labels = ["Obs 1", "Obs 2", "Obs 3"]
        for i in range(3):
            foot = np.zeros(3)
            foot[i] = y_hat_vec[i]
            traces.append(go.Scatter3d(
                x=[y_hat_vec[0], foot[0]],
                y=[y_hat_vec[1], foot[1]],
                z=[y_hat_vec[2], foot[2]],
                mode="lines+text",
                line=dict(color="grey", width=2, dash="dot"),
                text=["", f"\u0177{chr(0x2081 + i)} = {y_hat_vec[i]:.2f}"],
                textposition="bottom center",
                textfont=dict(size=9, color="grey"),
                name=f"Drop to {drop_labels[i]}",
                showlegend=(i == 0),
                hoverinfo="text",
            ))
    if show_custom.value:
        traces.extend(
            vector_trace(origin, custom_vec, COLOUR_CUSTOM, "Custom vector")
        )
    if show_axes.value:
        axis_len = float(max(
            np.max(np.abs(x1_vec)),
            np.max(np.abs(x2_vec)),
            np.max(np.abs(y_vec)),
            np.max(np.abs(custom_vec)) if show_custom.value else 0,
        ) * 1.3)
        for i, label in enumerate(["Obs 1", "Obs 2", "Obs 3"]):
            tip = np.zeros(3)
            tip[i] = axis_len
            traces.append(go.Scatter3d(
                x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                mode="lines+text",
                line=dict(color="lightgrey", width=2),
                text=["", label], textposition="top center",
                textfont=dict(size=10, color="grey"),
                showlegend=False, hoverinfo="skip",
            ))

    ols_fig = go.Figure(data=traces).update_layout(
        scene=dict(
            xaxis=dict(title="Observation 1", showspikes=False),
            yaxis=dict(title="Observation 2", showspikes=False),
            zaxis=dict(title="Observation 3", showspikes=False),
            aspectmode="cube",
            dragmode="turntable",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=620,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return (ols_fig,)


# -------------------------------------------------------------------
# Summary statistics (markdown)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    beta_hat, dist_val, e_vec, mo, np,
    r2_val, r2u_val, ssr_val,
    x1_vec, x2_vec, y_hat_vec,
):
    stats_md = mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |:---|:---|
        | $\\hat{{\\boldsymbol{{\\beta}}}}$ | $({beta_hat[0]:.4f},\\; {beta_hat[1]:.4f})$ |
        | $\\hat{{\\mathbf{{y}}}}$ | $({y_hat_vec[0]:.4f},\\; {y_hat_vec[1]:.4f},\\; {y_hat_vec[2]:.4f})$ |
        | Residuals $\\mathbf{{e}}$ | $({e_vec[0]:.4f},\\; {e_vec[1]:.4f},\\; {e_vec[2]:.4f})$ |
        | SSR ($\\mathbf{{e'e}}$) | ${ssr_val:.4f}$ |
        | $\\|\\mathbf{{y}} - \\hat{{\\mathbf{{y}}}}\\|$ | ${dist_val:.4f}$ |
        | Uncentred $R^2_u$ | ${r2u_val:.4f}$ |
        | Centred $R^2$ | ${r2_val:.4f}$ |

        **Orthogonality check** (should be $\\approx 0$):
        $\\mathbf{{e'x}}_1 = {np.dot(e_vec, x1_vec):.2e}$,
        $\\;\\mathbf{{e'x}}_2 = {np.dot(e_vec, x2_vec):.2e}$
        """
    )
    return (stats_md,)


# -------------------------------------------------------------------
# Projection matrices (markdown)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(M_mat, P_mat, X_mat, mo, np):
    def fmt_matrix(M):
        rows = []
        for i in range(M.shape[0]):
            cells = " & ".join(f"{M[i,j]:.4f}" for j in range(M.shape[1]))
            rows.append(cells)
        return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"

    matrices_md = mo.md(
        f"""
        ### Projection Matrices

        **Design matrix:**
        $\\mathbf{{X}} = {fmt_matrix(X_mat)}$

        **Projection matrix** $\\mathbf{{P_X}} = \\mathbf{{X}}(\\mathbf{{X'X}})^{{-1}}\\mathbf{{X'}}$:

        $\\mathbf{{P_X}} = {fmt_matrix(P_mat)}$

        **Residual-maker** $\\mathbf{{M_X}} = \\mathbf{{I}} - \\mathbf{{P_X}}$:

        $\\mathbf{{M_X}} = {fmt_matrix(M_mat)}$

        **Properties:**

        | Property | Max diff |
        |:---|:---|
        | $\\mathbf{{P_X X}} = \\mathbf{{X}}$ | ${np.max(np.abs(P_mat @ X_mat - X_mat)):.2e}$ |
        | $\\mathbf{{P_X}}$ symmetric | ${np.max(np.abs(P_mat - P_mat.T)):.2e}$ |
        | $\\mathbf{{P_X}}$ idempotent | ${np.max(np.abs(P_mat @ P_mat - P_mat)):.2e}$ |
        | $\\mathbf{{M_X X}} = \\mathbf{{0}}$ | ${np.max(np.abs(M_mat @ X_mat)):.2e}$ |
        """
    )
    return (fmt_matrix, matrices_md)


# -------------------------------------------------------------------
# Main layout: side-by-side with tabs for the right panel
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    data_grid, matrices_md, mo,
    ols_fig, stats_md, tick_boxes,
):
    left_panel = mo.vstack([
        mo.md("### Data"),
        data_grid,
        mo.md("### Display"),
        tick_boxes,
    ])

    right_panel = mo.ui.tabs({
        "\U0001f4ca 3D Plot": ols_fig,
        "\U0001f4cb Summary": stats_md,
        "\U0001f9ee Matrices": matrices_md,
    })

    mo.hstack(
        [left_panel, right_panel],
        widths=[1, 3],
        gap=1.5,
        align="start",
    )
    return (left_panel, right_panel)


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        *STAT321 — Linear Models and Time Series Analysis* |
        Nelson Mandela University
        """
    )
    return


if __name__ == "__main__":
    app.run()
