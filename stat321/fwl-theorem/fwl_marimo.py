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
app = marimo.App(width="full", app_title="FWL Theorem in R3")


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return go, mo, np


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Frisch-Waugh-Lovell Theorem in R³

        Regression = **orthogonal projection** onto the span of the regressors.
        FWL shows how partialling out isolates each regressor’s unique effect.

        **Controls**
        • **Left-click + drag** → rotate freely (orbit)
        • **Shift + drag** or right-click + drag → pan
        • **Scroll** → zoom
        • Hover top-right for modebar (reset, etc.)
        """
    )
    return


# -------------------------------------------------------------------
# UI Controls
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    rho = mo.ui.slider(-0.95, 0.95, value=0.6, step=0.05, label="ρ = Corr(X1, X2)")
    beta1 = mo.ui.slider(-3.0, 3.0, value=2.0, step=0.1, label="β1 (true)")
    beta2 = mo.ui.slider(-3.0, 3.0, value=1.5, step=0.1, label="β2 (true)")
    sigma = mo.ui.slider(0.0, 1.5, value=0.3, step=0.05, label="σ (noise)")
    fwl_step = mo.ui.slider(0, 4, value=0, step=1, label="FWL Step")
    hide_spines = mo.ui.checkbox(value=True, label="Hide axis spines")
    show_labels = mo.ui.checkbox(value=True, label="Show vector labels")

    controls_grid = mo.md(
        f"""
        | Parameter       | Value          |
        |:----------------|:---------------|
        | Correlation (ρ) | {rho}          |
        | True β1         | {beta1}        |
        | True β2         | {beta2}        |
        | Noise (σ)       | {sigma}        |
        | FWL Step        | {fwl_step}     |
        | Hide spines     | {hide_spines}  |
        | Show labels     | {show_labels}  |
        """
    )

    return beta1, beta2, controls_grid, fwl_step, hide_spines, rho, show_labels, sigma


# -------------------------------------------------------------------
# Step Description (separate cell — reads fwl_step.value)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(fwl_step, mo):
    _step_descriptions = {
        0: "Base regression: **y**, **X₁**, **X₂**, projection **ŷ**, and residual **e**.",
        1: "Project **y** onto **X₂**: show P_{X₂} y and residual M_{X₂} y.",
        2: "Project **X₁** onto **X₂**: show P_{X₂} X₁ and residual M_{X₂} X₁.",
        3: "FWL projection: regress M_{X₂} y on M_{X₂} X₁ to recover β̂₁.",
        4: "Decompose **ŷ** = P_{X₂} ŷ + β̂₁ X̃₁ — the FWL fitted value is the component of **ŷ** orthogonal to **X₂**.",
    }

    step_description = mo.md(
        f"**Step {fwl_step.value}:** {_step_descriptions[fwl_step.value]}"
    )
    return (step_description,)


# -------------------------------------------------------------------
# Construct Vectors and Compute Projections (FIXED — no centering)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(beta1, beta2, np, rho, sigma):
    e1, e2, e3 = np.eye(3)

    X1 = e1.copy()
    X2 = rho.value * X1 + np.sqrt(max(0, 1 - rho.value**2)) * e2

    n_plane = np.cross(X1, X2)
    n_plane = n_plane / (np.linalg.norm(n_plane) + 1e-10)
    eps = sigma.value * n_plane

    y = beta1.value * X1 + beta2.value * X2 + eps

    # OLS projection onto span(X1, X2) — no centering
    X_full = np.column_stack([X1, X2])
    XtX_inv = np.linalg.inv(X_full.T @ X_full)
    beta_hat_full = XtX_inv @ X_full.T @ y
    beta1_full = beta_hat_full[0]
    beta2_full = beta_hat_full[1]

    P = X_full @ XtX_inv @ X_full.T
    y_hat = P @ y
    e = y - y_hat

    # FWL (no centering)
    proj_scalar_X1 = np.dot(X1, X2) / np.dot(X2, X2)
    X1_proj_X2 = proj_scalar_X1 * X2
    X1_tilde = X1 - X1_proj_X2

    proj_scalar_y = np.dot(y, X2) / np.dot(X2, X2)
    y_proj_X2 = proj_scalar_y * X2
    y_tilde = y - y_proj_X2

    beta1_fwl = np.dot(y_tilde, X1_tilde) / np.dot(X1_tilde, X1_tilde)
    y_tilde_hat = beta1_fwl * X1_tilde

    # Projection of ŷ onto X2 — used in step 4 to show ŷ = P_{X2}ŷ + β̂₁ X̃₁
    y_hat_proj_X2 = (np.dot(y_hat, X2) / np.dot(X2, X2)) * X2

    X1_tilde_norm = np.linalg.norm(X1_tilde)
    beta_diff = abs(beta1_full - beta1_fwl)

    return (
        X1, X1_proj_X2, X1_tilde, X2,
        beta1_full, beta1_fwl, beta2_full,
        beta_diff, e, y, y_hat, y_hat_proj_X2,
        y_proj_X2, y_tilde, y_tilde_hat,
        X1_tilde_norm,
    )


# -------------------------------------------------------------------
# Plot Helpers
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(go, np):
    COLOUR_X1 = "#0072BD"
    COLOUR_X2 = "#7E2F8E"
    COLOUR_Y = "#D9531A"
    COLOUR_YHAT = "#006B34"
    COLOUR_RESIDUAL = "#A2142F"
    COLOUR_PROJ_Y = "#E6AB02"       # P_{X2} y  (gold)
    COLOUR_TILDE_Y = "#17BECF"      # M_{X2} y  (cyan)
    COLOUR_PROJ_X1 = "#BCBD22"      # P_{X2} X1 (olive)
    COLOUR_TILDE_X1 = "#56B4E9"     # M_{X2} X1 (sky blue)
    COLOUR_FWL_PROJ = "#8C564B"
    COLOUR_YHAT_DECOMP = "#FF6EC7"

    def make_vec(origin, tip, colour, name, width=5, dash=None, visible=True):
        return go.Scatter3d(
            x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
            mode="lines",
            line=dict(color=colour, width=width, dash=dash),
            name=name,
            hoverinfo="name",
            visible=visible,
        )

    def make_plane_mesh(x1, x2):
        s = np.linspace(-0.2, 1.5, 6)
        t = np.linspace(-0.2, 1.5, 6)
        ss, tt = np.meshgrid(s, t)
        pts = np.outer(x1, ss.ravel()) + np.outer(x2, tt.ravel())
        return go.Mesh3d(
            x=pts[0], y=pts[1], z=pts[2],
            color="rgba(150,150,200,0.25)", opacity=0.3,
            name="Span(X1,X2)", hoverinfo="name", alphahull=0,
        )

    def make_label(tip, text, colour, textposition="top right", visible=True):
        return go.Scatter3d(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            mode="text", text=[text],
            textposition=textposition,
            textfont=dict(color=colour, size=13, family="Arial"),
            hoverinfo="skip", showlegend=False,
            visible=visible,
        )

    return (
        COLOUR_FWL_PROJ, COLOUR_PROJ_X1, COLOUR_PROJ_Y, COLOUR_RESIDUAL,
        COLOUR_TILDE_X1, COLOUR_TILDE_Y,
        COLOUR_X1, COLOUR_X2, COLOUR_Y, COLOUR_YHAT, COLOUR_YHAT_DECOMP,
        make_label, make_plane_mesh, make_vec,
    )


# -------------------------------------------------------------------
# Build Traces — Displaced Path View
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_FWL_PROJ, COLOUR_PROJ_X1, COLOUR_PROJ_Y, COLOUR_RESIDUAL,
    COLOUR_TILDE_X1, COLOUR_TILDE_Y,
    COLOUR_X1, COLOUR_X2, COLOUR_Y, COLOUR_YHAT, COLOUR_YHAT_DECOMP,
    X1, X1_proj_X2, X1_tilde, X2,
    fwl_step, make_label, make_plane_mesh, make_vec,
    np, show_labels, y, y_hat, y_hat_proj_X2,
    y_proj_X2, y_tilde, y_tilde_hat,
):
    _origin = np.zeros(3)
    _step = fwl_step.value
    _labels = show_labels.value
    displaced_traces = []

    # Base: always visible
    displaced_traces.append(make_plane_mesh(X1, X2))
    displaced_traces.append(make_vec(_origin, X1, COLOUR_X1, "X1", 6))
    displaced_traces.append(make_vec(_origin, X2, COLOUR_X2, "X2", 6))
    displaced_traces.append(make_vec(_origin, y, COLOUR_Y, "y", 6))
    displaced_traces.append(make_vec(_origin, y_hat, COLOUR_YHAT, "ŷ = P_X y", 5))
    displaced_traces.append(make_vec(y_hat, y, COLOUR_RESIDUAL, "e = y − ŷ", 4, "dash"))

    # Step 1: project y onto X2
    displaced_traces.append(make_vec(_origin, y_proj_X2, COLOUR_PROJ_Y, "P_{X2} y", 4, "dot", visible=_step >= 1))
    displaced_traces.append(make_vec(y_proj_X2, y, COLOUR_TILDE_Y, "M_{X2} y", 5, visible=_step >= 1))

    # Step 2: project X1 onto X2
    displaced_traces.append(make_vec(_origin, X1_proj_X2, COLOUR_PROJ_X1, "P_{X2} X1", 4, "dot", visible=_step >= 2))
    displaced_traces.append(make_vec(X1_proj_X2, X1, COLOUR_TILDE_X1, "M_{X2} X1", 5, visible=_step >= 2))

    # Step 3: FWL projection — displaced from P_{X2}y so it's visually
    # parallel to M_{X2}X1 (both start from their projection base)
    _fwl_displaced_tip = y_proj_X2 + y_tilde_hat
    displaced_traces.append(make_vec(y_proj_X2, _fwl_displaced_tip, COLOUR_FWL_PROJ, "β̂₁ X̃₁ (FWL fit)", 4, visible=_step >= 3))

    # Step 4: decompose ŷ = P_{X2}ŷ + β̂₁ X̃₁
    displaced_traces.append(make_vec(_origin, y_hat_proj_X2, COLOUR_YHAT_DECOMP, "P_{X2} ŷ", 4, "dot", visible=_step >= 4))
    displaced_traces.append(make_vec(y_hat_proj_X2, y_hat, COLOUR_FWL_PROJ, "β̂₁ X̃₁ → ŷ", 5, visible=_step >= 4))

    # Labels
    displaced_traces.append(make_label(X1, "X₁", COLOUR_X1, visible=_labels))
    displaced_traces.append(make_label(X2, "X₂", COLOUR_X2, visible=_labels))
    displaced_traces.append(make_label(y, "y", COLOUR_Y, visible=_labels))
    displaced_traces.append(make_label(y_hat, "ŷ", COLOUR_YHAT, "bottom right", visible=_labels))
    displaced_traces.append(make_label(y_proj_X2, "P_{X₂} y", COLOUR_PROJ_Y, "middle right", visible=_labels and _step >= 1))
    displaced_traces.append(make_label(y_tilde, "M_{X₂} y", COLOUR_TILDE_Y, "top left", visible=_labels and _step >= 1))
    displaced_traces.append(make_label(X1_proj_X2, "P_{X₂} X₁", COLOUR_PROJ_X1, visible=_labels and _step >= 2))
    displaced_traces.append(make_label(X1_tilde, "M_{X₂} X₁", COLOUR_TILDE_X1, "top left", visible=_labels and _step >= 2))
    displaced_traces.append(make_label(_fwl_displaced_tip, "β̂₁ X̃₁", COLOUR_FWL_PROJ, visible=_labels and _step >= 3))
    displaced_traces.append(make_label(y_hat_proj_X2, "P_{X₂} ŷ", COLOUR_YHAT_DECOMP, "middle right", visible=_labels and _step >= 4))

    return (displaced_traces,)


# -------------------------------------------------------------------
# Build Traces — Origin Directions View
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_FWL_PROJ, COLOUR_PROJ_X1, COLOUR_PROJ_Y, COLOUR_RESIDUAL,
    COLOUR_TILDE_X1, COLOUR_TILDE_Y,
    COLOUR_X1, COLOUR_X2, COLOUR_Y, COLOUR_YHAT, COLOUR_YHAT_DECOMP,
    X1, X1_proj_X2, X1_tilde, X2,
    fwl_step, make_label, make_plane_mesh, make_vec,
    np, show_labels, y, y_hat, y_hat_proj_X2,
    y_proj_X2, y_tilde, y_tilde_hat,
):
    _origin = np.zeros(3)
    _step = fwl_step.value
    _labels = show_labels.value
    origin_traces = []

    # Base: always visible
    origin_traces.append(make_plane_mesh(X1, X2))
    origin_traces.append(make_vec(_origin, X1, COLOUR_X1, "X1", 6))
    origin_traces.append(make_vec(_origin, X2, COLOUR_X2, "X2", 6))
    origin_traces.append(make_vec(_origin, y, COLOUR_Y, "y", 6))
    origin_traces.append(make_vec(_origin, y_hat, COLOUR_YHAT, "ŷ = P_X y", 5))
    origin_traces.append(make_vec(y_hat, y, COLOUR_RESIDUAL, "e = y − ŷ", 4, "dash"))

    # Step 1: y residualised w.r.t. X2 — from origin
    origin_traces.append(make_vec(_origin, y_proj_X2, COLOUR_PROJ_Y, "P_{X2} y", 4, "dot", visible=_step >= 1))
    origin_traces.append(make_vec(_origin, y_tilde, COLOUR_TILDE_Y, "M_{X2} y", 5, visible=_step >= 1))

    # Step 2: X1 residualised w.r.t. X2 — from origin
    origin_traces.append(make_vec(_origin, X1_proj_X2, COLOUR_PROJ_X1, "P_{X2} X1", 4, "dot", visible=_step >= 2))
    origin_traces.append(make_vec(_origin, X1_tilde, COLOUR_TILDE_X1, "M_{X2} X1", 5, visible=_step >= 2))

    # Step 3: FWL projection — from origin
    origin_traces.append(make_vec(_origin, y_tilde_hat, COLOUR_FWL_PROJ, "β̂₁ X̃₁ (FWL fit)", 4, visible=_step >= 3))

    # Step 4: decompose ŷ = P_{X2}ŷ + β̂₁ X̃₁
    origin_traces.append(make_vec(_origin, y_hat_proj_X2, COLOUR_YHAT_DECOMP, "P_{X2} ŷ", 4, "dot", visible=_step >= 4))
    origin_traces.append(make_vec(y_hat_proj_X2, y_hat, COLOUR_FWL_PROJ, "β̂₁ X̃₁ → ŷ", 5, visible=_step >= 4))

    # Labels
    origin_traces.append(make_label(X1, "X₁", COLOUR_X1, visible=_labels))
    origin_traces.append(make_label(X2, "X₂", COLOUR_X2, visible=_labels))
    origin_traces.append(make_label(y, "y", COLOUR_Y, visible=_labels))
    origin_traces.append(make_label(y_hat, "ŷ", COLOUR_YHAT, "bottom right", visible=_labels))
    origin_traces.append(make_label(y_proj_X2, "P_{X₂} y", COLOUR_PROJ_Y, "middle right", visible=_labels and _step >= 1))
    origin_traces.append(make_label(y_tilde, "M_{X₂} y", COLOUR_TILDE_Y, "top left", visible=_labels and _step >= 1))
    origin_traces.append(make_label(X1_proj_X2, "P_{X₂} X₁", COLOUR_PROJ_X1, visible=_labels and _step >= 2))
    origin_traces.append(make_label(X1_tilde, "M_{X₂} X₁", COLOUR_TILDE_X1, "top left", visible=_labels and _step >= 2))
    origin_traces.append(make_label(y_tilde_hat, "β̂₁ X̃₁", COLOUR_FWL_PROJ, visible=_labels and _step >= 3))
    origin_traces.append(make_label(y_hat_proj_X2, "P_{X₂} ŷ", COLOUR_YHAT_DECOMP, "middle right", visible=_labels and _step >= 4))

    return (origin_traces,)


# -------------------------------------------------------------------
# Step-by-Step Explanation — Displaced Path (colour-coded)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_FWL_PROJ, COLOUR_PROJ_X1, COLOUR_PROJ_Y, COLOUR_RESIDUAL,
    COLOUR_TILDE_X1, COLOUR_TILDE_Y,
    COLOUR_X1, COLOUR_X2, COLOUR_Y, COLOUR_YHAT, COLOUR_YHAT_DECOMP,
    fwl_step, mo,
):
    def _c(colour, text):
        return f'<span style="color:{colour};font-weight:bold">{text}</span>'

    _step = fwl_step.value
    _x1 = _c(COLOUR_X1, "X₁")
    _x2 = _c(COLOUR_X2, "X₂")
    _y = _c(COLOUR_Y, "y")
    _yhat = _c(COLOUR_YHAT, "ŷ")
    _resid = _c(COLOUR_RESIDUAL, "e")
    _fwl = _c(COLOUR_FWL_PROJ, "β̂₁ X̃₁")
    _decomp = _c(COLOUR_YHAT_DECOMP, "P<sub>X₂</sub> ŷ")

    _sections = [
        f"**Step 0 — Full OLS regression.**  \n"
        f"The model is {_y} = β₁ {_x1} + β₂ {_x2} + ε. "
        f"OLS projects {_y} orthogonally onto Span({_x1}, {_x2}) to obtain "
        f"{_yhat}. The {_resid} = {_y} − {_yhat} (dashed) is perpendicular "
        f"to the plane."
    ]

    if _step >= 1:
        _sections.append(
            f"**Step 1 — Partial out {_x2} from {_y}.**  \n"
            f"Project {_y} onto {_x2} to get "
            f"{_c(COLOUR_PROJ_Y, 'P<sub>X₂</sub> y')} (dotted, from origin). "
            f"The residual {_c(COLOUR_TILDE_Y, 'M<sub>X₂</sub> y')} is the "
            f"displaced segment from the tip of "
            f"{_c(COLOUR_PROJ_Y, 'P<sub>X₂</sub> y')} to {_y}, "
            f"showing the subtraction {_y} − P<sub>X₂</sub> y geometrically."
        )

    if _step >= 2:
        _sections.append(
            f"**Step 2 — Partial out {_x2} from {_x1}.**  \n"
            f"Project {_x1} onto {_x2} to get "
            f"{_c(COLOUR_PROJ_X1, 'P<sub>X₂</sub> X₁')} (dotted, from origin). "
            f"The residual {_c(COLOUR_TILDE_X1, 'M<sub>X₂</sub> X₁')} = X̃₁ "
            f"is the displaced segment from the tip of "
            f"{_c(COLOUR_PROJ_X1, 'P<sub>X₂</sub> X₁')} to {_x1} — "
            f"the unique direction of {_x1} not collinear with {_x2}."
        )

    if _step >= 3:
        _sections.append(
            f"**Step 3 — FWL regression.**  \n"
            f"Regress {_c(COLOUR_TILDE_Y, 'M<sub>X₂</sub> y')} on "
            f"{_c(COLOUR_TILDE_X1, 'M<sub>X₂</sub> X₁')} = X̃₁. "
            f"The fitted value {_fwl} is drawn starting from "
            f"{_c(COLOUR_PROJ_Y, 'P<sub>X₂</sub> y')}, parallel to X̃₁ — "
            f"easy to see it shares the direction of the displaced "
            f"{_c(COLOUR_TILDE_X1, 'M<sub>X₂</sub> X₁')}."
        )

    if _step >= 4:
        _sections.append(
            f"**Step 4 — Why it works: decompose {_yhat}.**  \n"
            f"Project {_yhat} onto {_x2} to get {_decomp} (dotted). "
            f"The displaced vector from {_decomp} to {_yhat} is exactly "
            f"{_fwl}. This confirms {_yhat} = {_decomp} + {_fwl}: "
            f"the FWL fitted value is the component of {_yhat} "
            f"orthogonal to {_x2}."
        )

    displaced_explanation = mo.md("\n\n".join(_sections))
    return (displaced_explanation,)


# -------------------------------------------------------------------
# Step-by-Step Explanation — Origin Directions (colour-coded)
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    COLOUR_FWL_PROJ, COLOUR_PROJ_X1, COLOUR_PROJ_Y, COLOUR_RESIDUAL,
    COLOUR_TILDE_X1, COLOUR_TILDE_Y,
    COLOUR_X1, COLOUR_X2, COLOUR_Y, COLOUR_YHAT, COLOUR_YHAT_DECOMP,
    fwl_step, mo,
):
    def _c(colour, text):
        return f'<span style="color:{colour};font-weight:bold">{text}</span>'

    _step = fwl_step.value
    _x1 = _c(COLOUR_X1, "X₁")
    _x2 = _c(COLOUR_X2, "X₂")
    _y = _c(COLOUR_Y, "y")
    _yhat = _c(COLOUR_YHAT, "ŷ")
    _resid = _c(COLOUR_RESIDUAL, "e")
    _fwl = _c(COLOUR_FWL_PROJ, "β̂₁ X̃₁")
    _decomp = _c(COLOUR_YHAT_DECOMP, "P<sub>X₂</sub> ŷ")

    _sections = [
        f"**Step 0 — Full OLS regression.**  \n"
        f"The model is {_y} = β₁ {_x1} + β₂ {_x2} + ε. "
        f"OLS projects {_y} orthogonally onto Span({_x1}, {_x2}) to obtain "
        f"{_yhat}. The {_resid} = {_y} − {_yhat} (dashed) is perpendicular "
        f"to the plane."
    ]

    if _step >= 1:
        _sections.append(
            f"**Step 1 — Partial out {_x2} from {_y}.**  \n"
            f"Project {_y} onto {_x2} to get "
            f"{_c(COLOUR_PROJ_Y, 'P<sub>X₂</sub> y')} (dotted). "
            f"The residual {_c(COLOUR_TILDE_Y, 'M<sub>X₂</sub> y')} is drawn "
            f"from the origin — the component of {_y} orthogonal to {_x2}."
        )

    if _step >= 2:
        _sections.append(
            f"**Step 2 — Partial out {_x2} from {_x1}.**  \n"
            f"Project {_x1} onto {_x2} to get "
            f"{_c(COLOUR_PROJ_X1, 'P<sub>X₂</sub> X₁')} (dotted). "
            f"The residual {_c(COLOUR_TILDE_X1, 'M<sub>X₂</sub> X₁')} = X̃₁ "
            f"is drawn from the origin — the unique direction of {_x1} "
            f"after removing the part collinear with {_x2}. "
            f"Notice X̃₁ is perpendicular to {_x2}."
        )

    if _step >= 3:
        _sections.append(
            f"**Step 3 — FWL regression.**  \n"
            f"Regress {_c(COLOUR_TILDE_Y, 'M<sub>X₂</sub> y')} on "
            f"{_c(COLOUR_TILDE_X1, 'M<sub>X₂</sub> X₁')} = X̃₁. "
            f"The fitted value {_fwl} is drawn from the origin, "
            f"visually confirming it lies along "
            f"{_c(COLOUR_TILDE_X1, 'X̃₁')}. "
            f"This simple regression recovers the same β̂₁ as full OLS."
        )

    if _step >= 4:
        _sections.append(
            f"**Step 4 — Why it works: decompose {_yhat}.**  \n"
            f"Project {_yhat} onto {_x2} to get {_decomp} (dotted). "
            f"From {_decomp}, the vector {_fwl} reaches {_yhat}. "
            f"Both are shown from the origin; their sum equals {_yhat}, "
            f"confirming {_yhat} = {_decomp} + {_fwl}."
        )

    origin_explanation = mo.md("\n\n".join(_sections))
    return (origin_explanation,)


# -------------------------------------------------------------------
# Statistics Panel
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(beta1, beta1_full, beta1_fwl, beta2, beta2_full, beta_diff, mo, rho, X1_tilde_norm):
    eq_symbol = "✓" if beta_diff < 1e-10 else "✗"
    mc_warn = f"⚠️ High collinearity: ‖X1⊥X2‖ = {X1_tilde_norm:.4f}" if abs(rho.value) > 0.9 else f"‖X1⊥X2‖ = {X1_tilde_norm:.4f}"

    stats_md = mo.md(
        f"""
        ### Coefficient Comparison

        | Method     | β̂1      | β̂2      |
        |:-----------|--------:|--------:|
        | True       | {beta1.value:.2f} | {beta2.value:.2f} |
        | Full OLS   | **{beta1_full:.4f}** | {beta2_full:.4f} |
        | FWL        | **{beta1_fwl:.4f}** | — |

        {eq_symbol} |β̂1^full − β̂1^FWL| = {beta_diff:.2e}

        {mc_warn}
        """
    )
    return (stats_md,)


# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    controls_grid, displaced_explanation, displaced_traces, go,
    hide_spines, mo, origin_explanation, origin_traces,
    stats_md, step_description,
):
    _axis_props = {"visible": False} if hide_spines.value else {
        "visible": True, "showgrid": True, "zeroline": False,
        "showline": True, "showticklabels": False, "title": "", "showbackground": False,
    }

    _layout = dict(
        scene=dict(
            aspectmode="cube",
            xaxis=_axis_props,
            yaxis=_axis_props,
            zaxis=_axis_props,
            camera=dict(eye=dict(x=1.65, y=1.45, z=1.25)),
            uirevision="keep",
        ),
        uirevision="keep",
        dragmode="orbit",
        margin=dict(l=0, r=0, t=25, b=0),
        height=580,
        showlegend=False,
    )

    _fig_displaced = go.Figure(data=displaced_traces)
    _fig_displaced.update_layout(**_layout)

    _fig_origin = go.Figure(data=origin_traces)
    _fig_origin.update_layout(**_layout)

    mo.vstack([
        mo.hstack(
            [
                mo.vstack([mo.md("### Parameters"), controls_grid, step_description]),
                mo.ui.tabs({
                    "📐 Displaced Path": mo.vstack([_fig_displaced, displaced_explanation]),
                    "🎯 Origin Directions": mo.vstack([_fig_origin, origin_explanation]),
                    "📋 Statistics": stats_md,
                }),
            ],
            widths=[1, 3],
        ),
    ])

    return


if __name__ == "__main__":
    app.run()
