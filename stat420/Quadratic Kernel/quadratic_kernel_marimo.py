# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", app_title="Quadratic Kernel Ridge Regression")


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Quadratic Kernel: The Trick Made Visible

        Companion to *Nonlinear forecasting with many predictors using kernel
        ridge regression* (Exterkate, Groenen, Heij & van Dijk, 2016),
        §4.2–4.4 of the STAT420 kernel ridge regression notes.

        Ridge regression in a feature space $\varphi:\mathbb{R}^N\to\mathbb{R}^M$ predicts

        $$\hat y_* = \mathbf{z}_*'(\mathbf{Z}'\mathbf{Z}+\lambda\mathbf{I}_M)^{-1}\mathbf{Z}'\mathbf{y}, \qquad \mathbf{z}=\varphi(\mathbf{x}),$$

        which requires forming and inverting an $M\times M$ matrix. The
        **kernel trick** rewrites this identically as

        $$\boxed{\hat y_* = \mathbf{k}_*'(\mathbf{K}+\lambda\mathbf{I}_T)^{-1}\mathbf{y}},\qquad
        K_{st}=\kappa(\mathbf{x}_s,\mathbf{x}_t),\ \ (\mathbf{k}_*)_t=\kappa(\mathbf{x}_t,\mathbf{x}_*),$$

        an equally-valid $T\times T$ problem, *provided* $\kappa(a,b)=\varphi(a)'\varphi(b)$
        can be evaluated **without ever building $\varphi$**. The **quadratic
        kernel**

        $$\kappa(a,b) = (1+a'b)^2$$

        is the cleanest place to see this happen, because its feature map

        $$\varphi(a) = \bigl(1,\ \sqrt2 a_1,\dots,\sqrt2 a_N,\
        a_1^2,\dots,a_N^2,\ \sqrt2 a_1a_2,\dots,\sqrt2 a_{N-1}a_N\bigr)'$$

        is small enough to write down in full, and $\varphi(a)'\varphi(b)$
        collapses to $(1+a'b)^2$ by the identity $(\sum_n u_n)^2 = \sum_n
        u_n^2 + 2\sum_{n<m}u_nu_m$ applied to $u_n=a_nb_n$ — the $\sqrt2$
        factors exist purely to make that collapse exact.

        Four tabs below let you handle each piece of §4.2–4.4 directly:
        verify the trick and its $O(N)$ vs $O(N^2)$ saving, replay the
        fully-worked $N{=}2,T{=}3$ example with your own numbers, watch the
        feature map bend a circle flat in 3D, and finally see the quadratic
        kernel fit a curved boundary that a linear kernel cannot.
        """
    )
    return


# -------------------------------------------------------------------
# Helper functions: feature map, kernels, KRR solve, matrix formatting
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(np):
    def feature_map_quadratic(X):
        """φ(a) = (1, √2 a, a², √2 aᵢaⱼ for i<j) — the explicit Poly(2) map."""
        X = np.atleast_2d(X).astype(float)
        T, N = X.shape
        cols = [np.ones(T)]
        cols += [np.sqrt(2.0) * X[:, n] for n in range(N)]
        cols += [X[:, n] ** 2 for n in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                cols.append(np.sqrt(2.0) * X[:, i] * X[:, j])
        return np.column_stack(cols)

    def kernel_quadratic(A, B):
        """κ(a,b) = (1 + a'b)² — never touches φ."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        return (1.0 + A @ B.T) ** 2

    def kernel_linear(A, B):
        """κ(a,b) = a'b — the linear (Poly(1)-without-intercept) kernel of §4.1."""
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        return A @ B.T

    def feature_dim(N):
        """M = C(N+2, 2), the number of monomials up to degree 2 in N variables."""
        return (N + 1) * (N + 2) // 2

    def krr_fit(K, y, lam):
        """α̂ = (K + λI)⁻¹y, solved rather than inverted (§3.5 remark)."""
        T = K.shape[0]
        return np.linalg.solve(K + lam * np.eye(T), y)

    def mat_to_md(M, row_labels=None, col_labels=None, fmt="{:.3f}"):
        """Render a small numpy matrix/vector as a Markdown table."""
        M = np.atleast_2d(M)
        n_rows, n_cols = M.shape
        col_labels = col_labels or [f"c{j}" for j in range(n_cols)]
        header = "|  | " + " | ".join(col_labels) + " |"
        sep = "|---|" + "---|" * n_cols
        lines = [header, sep]
        for i in range(n_rows):
            label = row_labels[i] if row_labels else f"r{i}"
            vals = " | ".join(fmt.format(v) for v in M[i])
            lines.append(f"| **{label}** | {vals} |")
        return "\n".join(lines)

    return (
        feature_dim,
        feature_map_quadratic,
        kernel_linear,
        kernel_quadratic,
        krr_fit,
        mat_to_md,
    )


# -------------------------------------------------------------------
# Colour palette
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    COLOUR_KERNEL = "#0072BD"      # blue   — the O(N) kernel route
    COLOUR_EXPLICIT = "#D9531A"    # orange — the O(N²) explicit-φ route
    COLOUR_TRAIN = "#0072BD"       # blue   — training points
    COLOUR_TEST = "#D9531A"        # orange — the query / movable point
    COLOUR_ALPHA_POS = "#006B34"   # green  — positive α weight
    COLOUR_ALPHA_NEG = "#7E2F8E"   # purple — negative α weight
    COLOUR_PLANE_RADIUS = "#0072BD"
    COLOUR_PLANE_SIGN = "#7E2F8E"
    return (
        COLOUR_ALPHA_NEG,
        COLOUR_ALPHA_POS,
        COLOUR_EXPLICIT,
        COLOUR_KERNEL,
        COLOUR_PLANE_RADIUS,
        COLOUR_PLANE_SIGN,
        COLOUR_TEST,
        COLOUR_TRAIN,
    )


# =====================================================================
# TAB 1 — The kernel trick and its O(N) vs O(N²) saving  (§4.2)
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(1, 200, value=132, step=1, label="N (number of predictors)")
    seed_number = mo.ui.number(0, 9999, value=0, step=1, label="Random seed (resample a, b)")
    return n_slider, seed_number


@app.cell(hide_code=True)
def _(mo, n_slider, seed_number):
    controls_tab1 = mo.vstack([
        mo.md(
            "## §4.2 — The kernel trick, verified and costed\n\n"
            "Pick $N$ and a seed; two random points $a,b\\in\\mathbb{R}^N$ are drawn "
            "and $\\kappa(a,b)$ is computed **two ways**."
        ),
        n_slider,
        seed_number,
    ])
    return (controls_tab1,)


@app.cell(hide_code=True)
def _(feature_dim, feature_map_quadratic, kernel_quadratic, n_slider, np, seed_number):
    _rng = np.random.RandomState(seed_number.value)
    a_vec = _rng.randn(1, n_slider.value)
    b_vec = _rng.randn(1, n_slider.value)

    phi_a = feature_map_quadratic(a_vec)
    phi_b = feature_map_quadratic(b_vec)
    via_feature_map = (phi_a @ phi_b.T).item()
    via_kernel = kernel_quadratic(a_vec, b_vec).item()
    residual = abs(via_feature_map - via_kernel)

    M_current = feature_dim(n_slider.value)
    ops_kernel_current = 2 * n_slider.value + 1
    ops_explicit_current = 2 * M_current - 1
    speedup_current = ops_explicit_current / ops_kernel_current
    return (
        M_current,
        ops_explicit_current,
        ops_kernel_current,
        residual,
        speedup_current,
        via_feature_map,
        via_kernel,
    )


@app.cell(hide_code=True)
def _(
    COLOUR_EXPLICIT,
    COLOUR_KERNEL,
    go,
    n_slider,
    np,
):
    _N = np.arange(1, max(200, int(n_slider.value * 1.3)) + 1)
    _M = (_N + 1) * (_N + 2) // 2
    _ops_kernel = 2 * _N + 1
    _ops_explicit = 2 * _M - 1

    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(
        x=_N, y=_ops_kernel, mode="lines", name="Via κ(a,b)=(1+a'b)² — O(N)",
        line=dict(color=COLOUR_KERNEL, width=2.5),
    ))
    cost_fig.add_trace(go.Scatter(
        x=_N, y=_ops_explicit, mode="lines", name="Via φ(a)'φ(b) explicitly — O(N²)",
        line=dict(color=COLOUR_EXPLICIT, width=2.5),
    ))
    cost_fig.add_vline(
        x=n_slider.value, line=dict(color="#666666", width=1.5, dash="dot"),
        annotation_text=f"N = {n_slider.value}", annotation_position="top",
    )
    if _N.min() <= 132 <= _N.max():
        cost_fig.add_trace(go.Scatter(
            x=[132], y=[2 * 132 + 1], mode="markers+text",
            marker=dict(color=COLOUR_KERNEL, size=9, symbol="star"),
            text=["paper's N=132"], textposition="top center", showlegend=False,
        ))
    cost_fig = cost_fig.update_layout(
        title="Operations per kernel-matrix entry",
        xaxis=dict(title="N"), yaxis=dict(title="operations", type="log"),
        height=420, margin=dict(l=50, r=20, t=50, b=45),
        legend=dict(orientation="h", y=-0.2, x=0.0),
        uirevision="cost",
    )
    return (cost_fig,)


@app.cell(hide_code=True)
def _(
    M_current,
    mo,
    n_slider,
    ops_explicit_current,
    ops_kernel_current,
    residual,
    speedup_current,
    via_feature_map,
    via_kernel,
):
    tab1_explanation = mo.md(
        f"""
        ### Reading the numbers

        For $N={n_slider.value}$: the explicit feature map has
        $M={M_current}$ features. Computing one entry of $\\mathbf{{K}}$:

        | Route | Operations | This N |
        |:--|:--|--:|
        | Kernel $\\kappa(a,b)=(1+a'b)^2$ | $2N+1$ | **{ops_kernel_current}** |
        | Explicit $\\varphi(a)'\\varphi(b)$ | $2M-1$ | **{ops_explicit_current}** |

        Speed-up this N: **{speedup_current:.1f}×**.

        **Verification** — $\\varphi(a)'\\varphi(b) = {via_feature_map:.6f}$ vs
        $(1+a'b)^2 = {via_kernel:.6f}$ — they agree to
        ${residual:.2e}$, as the algebra guarantees for *any* $a,b$. The kernel
        route never built the {M_current}-vector $\\varphi(a)$ to get there.
        """
    )
    return (tab1_explanation,)


@app.cell(hide_code=True)
def _(cost_fig, mo, tab1_explanation):
    tab1_content = mo.vstack([cost_fig, tab1_explanation])
    return (tab1_content,)


# =====================================================================
# TAB 2 — A fully worked example, made interactive  (§4.3)
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    lam_slider = mo.ui.slider(0.05, 5.0, value=1.0, step=0.05, label="λ (penalty)")
    y1_slider = mo.ui.slider(-2.0, 2.0, value=1.0, step=0.1, label="y₁")
    y2_slider = mo.ui.slider(-2.0, 2.0, value=-1.0, step=0.1, label="y₂")
    y3_slider = mo.ui.slider(-2.0, 2.0, value=0.5, step=0.1, label="y₃")
    xstar1_slider = mo.ui.slider(-2.0, 2.0, value=0.5, step=0.1, label="x*₁")
    xstar2_slider = mo.ui.slider(-2.0, 2.0, value=0.5, step=0.1, label="x*₂")
    return (
        lam_slider,
        xstar1_slider,
        xstar2_slider,
        y1_slider,
        y2_slider,
        y3_slider,
    )


@app.cell(hide_code=True)
def _(
    lam_slider,
    mo,
    xstar1_slider,
    xstar2_slider,
    y1_slider,
    y2_slider,
    y3_slider,
):
    controls_tab2 = mo.vstack([
        mo.md(
            "## §4.3 — The N=2, T=3 worked example, live\n\n"
            "The three training points are fixed at "
            "$\\mathbf{x}_1=(1,1)$, $\\mathbf{x}_2=(-1,1)$, $\\mathbf{x}_3=(0,-1)$, exactly as in "
            "the notes. Move everything else — the defaults reproduce the "
            "notes' answer $\\hat y_*=43/120\\approx0.3583$ exactly."
        ),
        mo.hstack([lam_slider, xstar1_slider, xstar2_slider], justify="start"),
        mo.hstack([y1_slider, y2_slider, y3_slider], justify="start"),
    ])
    return (controls_tab2,)


@app.cell(hide_code=True)
def _(
    kernel_quadratic,
    krr_fit,
    lam_slider,
    np,
    xstar1_slider,
    xstar2_slider,
    y1_slider,
    y2_slider,
    y3_slider,
):
    X_fixed = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]])
    y_vec = np.array([y1_slider.value, y2_slider.value, y3_slider.value])
    xstar_vec = np.array([[xstar1_slider.value, xstar2_slider.value]])

    K_mat = kernel_quadratic(X_fixed, X_fixed)
    K_reg = K_mat + lam_slider.value * np.eye(3)
    alpha_hat = krr_fit(K_mat, y_vec, lam_slider.value)
    kstar_vec = kernel_quadratic(X_fixed, xstar_vec).ravel()
    yhat_star = float(kstar_vec @ alpha_hat)
    return K_mat, K_reg, X_fixed, alpha_hat, kstar_vec, xstar_vec, y_vec, yhat_star


@app.cell(hide_code=True)
def _(K_mat, K_reg, alpha_hat, kstar_vec, lam_slider, mat_to_md, mo, yhat_star):
    _labels = ["x₁", "x₂", "x₃"]
    _raw = f"""
    **K** (depends only on $\\mathbf{{x}}_1,\\mathbf{{x}}_2,\\mathbf{{x}}_3$, never on $\\lambda$ or $\\mathbf{{y}}$):

    {mat_to_md(K_mat, _labels, _labels)}

    **K + λI** (λ = {lam_slider.value:.2f}):

    {mat_to_md(K_reg, _labels, _labels)}

    **α̂ = (K+λI)⁻¹y**:

    {mat_to_md(alpha_hat.reshape(1, -1), ["α̂"], _labels)}

    **k\\*** = (κ(x₁,x\\*), κ(x₂,x\\*), κ(x₃,x\\*)):

    {mat_to_md(kstar_vec.reshape(1, -1), ["k*"], _labels)}

    **ŷ\\* = k\\*'α̂ = {yhat_star:.4f}**
    """
    # Embedded table strings from mat_to_md carry no leading indentation, which
    # breaks markdown's common-indent dedent and makes the whole block render
    # as a literal code block — strip every line so nothing hits the 4-space
    # "indented code block" threshold.
    tab2_matrices_md = mo.md("\n".join(line.lstrip() for line in _raw.split("\n")))
    return (tab2_matrices_md,)


@app.cell(hide_code=True)
def _(
    COLOUR_ALPHA_NEG,
    COLOUR_ALPHA_POS,
    COLOUR_TEST,
    X_fixed,
    alpha_hat,
    go,
    kstar_vec,
    np,
    xstar_vec,
):
    _kmax = max(kstar_vec.max(), 1e-9)
    tab2_fig = go.Figure()

    # Similarity edges from x* to each training point, weight ∝ κ(xᵢ,x*)
    for _i in range(3):
        tab2_fig.add_trace(go.Scatter(
            x=[X_fixed[_i, 0], xstar_vec[0, 0]],
            y=[X_fixed[_i, 1], xstar_vec[0, 1]],
            mode="lines",
            line=dict(color="#999999", width=1 + 5 * kstar_vec[_i] / _kmax),
            opacity=0.3 + 0.6 * kstar_vec[_i] / _kmax,
            hoverinfo="skip", showlegend=False,
        ))

    tab2_fig.add_trace(go.Scatter(
        x=X_fixed[:, 0], y=X_fixed[:, 1], mode="markers+text",
        marker=dict(
            size=18,
            color=[COLOUR_ALPHA_POS if a >= 0 else COLOUR_ALPHA_NEG for a in alpha_hat],
            line=dict(color="white", width=2),
        ),
        text=[f"x₁<br>α̂={alpha_hat[0]:.3f}", f"x₂<br>α̂={alpha_hat[1]:.3f}", f"x₃<br>α̂={alpha_hat[2]:.3f}"],
        textposition="bottom center", name="training points",
        hovertemplate="%{text}<extra></extra>",
    ))
    tab2_fig.add_trace(go.Scatter(
        x=[xstar_vec[0, 0]], y=[xstar_vec[0, 1]], mode="markers+text",
        marker=dict(size=16, color=COLOUR_TEST, symbol="diamond", line=dict(color="white", width=2)),
        text=["x*"], textposition="top center", name="query point",
    ))
    tab2_fig = tab2_fig.update_layout(
        title="Edge width/opacity = κ(xᵢ, x*) — how much each training point votes",
        xaxis=dict(title="a₁", range=[-2.2, 2.2], zeroline=True),
        yaxis=dict(title="a₂", range=[-2.2, 2.2], zeroline=True, scaleanchor="x", scaleratio=1),
        height=420, margin=dict(l=50, r=20, t=50, b=40), showlegend=False,
        uirevision="worked-example",
    )
    return (tab2_fig,)


@app.cell(hide_code=True)
def _(mo, tab2_fig, tab2_matrices_md):
    tab2_content = mo.hstack([tab2_fig, tab2_matrices_md], widths=[3, 2])
    return (tab2_content,)


# =====================================================================
# TAB 3 — Watching the feature map act, 2D → 3D  (§4.4)
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    r_slider = mo.ui.slider(0.1, 2.0, value=1.4142, step=0.05, label="radius r of your point")
    theta_slider = mo.ui.slider(0, 360, value=45, step=5, label="angle θ (degrees)")
    return r_slider, theta_slider


@app.cell(hide_code=True)
def _(mo, r_slider, theta_slider):
    controls_tab3 = mo.vstack([
        mo.md(
            r"""
            ## §4.4 — From 2D to 3D: watching the map act

            The pure-quadratic block $\varphi_3(a)=(a_1^2,\,a_2^2,\,\sqrt2\,a_1a_2)$
            already shows the effect: equal-radius points land on the same
            **plane** $q_1+q_2=\|a\|^2$, and same-sign-pattern points land on
            the same side of the plane $q_3=0$. Move your own point $x_4$ by
            radius and angle and watch both facts hold.
            """
        ),
        r_slider,
        theta_slider,
    ])
    return (controls_tab3,)


@app.cell(hide_code=True)
def _(np, r_slider, theta_slider):
    X3_fixed = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]])
    theta_rad = np.deg2rad(theta_slider.value)
    x4_point = np.array([r_slider.value * np.cos(theta_rad), r_slider.value * np.sin(theta_rad)])

    def phi3(a):
        a = np.atleast_2d(a)
        return np.column_stack([a[:, 0] ** 2, a[:, 1] ** 2, np.sqrt(2.0) * a[:, 0] * a[:, 1]])

    Q3_fixed = phi3(X3_fixed)
    q4_point = phi3(x4_point[None, :])[0]
    return Q3_fixed, X3_fixed, phi3, q4_point, x4_point


@app.cell(hide_code=True)
def _(np):
    def constraint_plane(C, qmax=4.4, zmax=4.4, n=2):
        """The plane q1+q2=C, clipped to q1,q2 ≥ 0 — a ruled surface in (q1,q3)."""
        q1_lo, q1_hi = max(0.0, C - qmax), min(qmax, C)
        if q1_hi <= q1_lo:
            q1_hi = q1_lo + 1e-6
        q1 = np.linspace(q1_lo, q1_hi, n)
        z = np.linspace(-zmax, zmax, n)
        Q1, Z = np.meshgrid(q1, z)
        Q2 = C - Q1
        return Q1, Q2, Z

    def sign_plane(qmax=4.4, n=2):
        """The plane q3=0."""
        q1 = np.linspace(0.0, qmax, n)
        q2 = np.linspace(0.0, qmax, n)
        Q1, Q2 = np.meshgrid(q1, q2)
        return Q1, Q2, np.zeros_like(Q1)

    return constraint_plane, sign_plane


@app.cell(hide_code=True)
def _(
    COLOUR_PLANE_RADIUS,
    COLOUR_PLANE_SIGN,
    COLOUR_TEST,
    COLOUR_TRAIN,
    Q3_fixed,
    X3_fixed,
    constraint_plane,
    go,
    make_subplots,
    np,
    q4_point,
    r_slider,
    sign_plane,
    theta_slider,
    x4_point,
):
    def _flat_surface(Q1, Q2, Z, colour, opacity):
        return go.Surface(
            x=Q1, y=Q2, z=Z, showscale=False, opacity=opacity,
            colorscale=[[0, colour], [1, colour]], surfacecolor=np.zeros_like(Q1),
            hoverinfo="skip", showlegend=False,
        )

    tab3_fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Input space R² (N=2)", "Feature space R³ (pure-quadratic block)"),
        horizontal_spacing=0.06,
    )

    # ---- left: input space ----
    _ang = np.linspace(0, 2 * np.pi, 100)
    tab3_fig.add_trace(go.Scatter(
        x=r_slider.value * np.cos(_ang), y=r_slider.value * np.sin(_ang),
        mode="lines", line=dict(color=COLOUR_PLANE_RADIUS, width=1.5, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    tab3_fig.add_trace(go.Scatter(
        x=X3_fixed[:, 0], y=X3_fixed[:, 1], mode="markers+text",
        marker=dict(color=COLOUR_TRAIN, size=11, line=dict(color="white", width=1.5)),
        text=["x₁", "x₂", "x₃"], textposition="top center", showlegend=False,
    ), row=1, col=1)
    tab3_fig.add_trace(go.Scatter(
        x=[x4_point[0]], y=[x4_point[1]], mode="markers+text",
        marker=dict(color=COLOUR_TEST, size=13, symbol="diamond", line=dict(color="white", width=1.5)),
        text=["x₄"], textposition="top center", showlegend=False,
    ), row=1, col=1)
    tab3_fig.update_xaxes(range=[-2.2, 2.2], title="a₁", zeroline=True, row=1, col=1)
    tab3_fig.update_yaxes(range=[-2.2, 2.2], title="a₂", zeroline=True,
                          scaleanchor="x", scaleratio=1, row=1, col=1)

    # ---- right: feature space ----
    _Q1, _Q2, _Z = constraint_plane(r_slider.value ** 2)
    tab3_fig.add_trace(_flat_surface(_Q1, _Q2, _Z, COLOUR_PLANE_RADIUS, 0.30), row=1, col=2)
    _Q1r, _Q2r, _Zr = constraint_plane(2.0)  # reference plane through x1, x2 (‖·‖²=2)
    tab3_fig.add_trace(_flat_surface(_Q1r, _Q2r, _Zr, "#888888", 0.12), row=1, col=2)
    _Q1s, _Q2s, _Zs = sign_plane()
    tab3_fig.add_trace(_flat_surface(_Q1s, _Q2s, _Zs, COLOUR_PLANE_SIGN, 0.15), row=1, col=2)

    tab3_fig.add_trace(go.Scatter3d(
        x=Q3_fixed[:, 0], y=Q3_fixed[:, 1], z=Q3_fixed[:, 2], mode="markers+text",
        marker=dict(color=COLOUR_TRAIN, size=5), text=["φ₃(x₁)", "φ₃(x₂)", "φ₃(x₃)"],
        showlegend=False,
    ), row=1, col=2)
    tab3_fig.add_trace(go.Scatter3d(
        x=[q4_point[0]], y=[q4_point[1]], z=[q4_point[2]], mode="markers+text",
        marker=dict(color=COLOUR_TEST, size=6, symbol="diamond"), text=["φ₃(x₄)"],
        showlegend=False,
    ), row=1, col=2)

    tab3_fig.update_scenes(
        xaxis=dict(title="q₁=a₁²", range=[0, 4.4]),
        yaxis=dict(title="q₂=a₂²", range=[0, 4.4]),
        zaxis=dict(title="q₃=√2 a₁a₂", range=[-4.4, 4.4]),
        aspectmode="cube",
    )
    tab3_fig = tab3_fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), uirevision="phi3")
    return (tab3_fig,)


@app.cell(hide_code=True)
def _(mo, q4_point, r_slider, theta_slider, x4_point):
    _same_radius = "x₁ and x₂ (both ‖·‖²=2)" if abs(r_slider.value ** 2 - 2.0) < 1e-6 else "no fixed point"
    _sign = "positive" if x4_point[0] * x4_point[1] > 0 else ("negative" if x4_point[0] * x4_point[1] < 0 else "zero")
    tab3_explanation = mo.md(
        f"""
        ### Reading the picture

        Your point $x_4$ has radius **{r_slider.value:.2f}**, angle
        **{theta_slider.value:.0f}°**, and maps to
        $\\varphi_3(x_4)=({q4_point[0]:.2f},\\,{q4_point[1]:.2f},\\,{q4_point[2]:.2f})$.

        - The **blue plane** is $q_1+q_2=r^2={r_slider.value**2:.2f}$ — *every*
          point of that radius, in *any* direction, lands somewhere on it. The
          faint grey plane is the fixed reference $q_1+q_2=2$, which $x_1$ and
          $x_2$ always sit on regardless of what you move.
        - The **purple plane** is $q_3=0$. Your point's coordinates
          $(a_1,a_2)=({x4_point[0]:.2f},{x4_point[1]:.2f})$ have a
          **{_sign}** product, matching the side of the plane $\\varphi_3(x_4)$
          falls on.
        - Circles in the left panel (not expressible by any line) become
          **flat planes** on the right; a sign/quadrant pattern (not linearly
          separable on the left) becomes **separable by $q_3=0$** on the
          right. That is the entire value of the feature map, drawn.
        """
    )
    return (tab3_explanation,)


@app.cell(hide_code=True)
def _(mo, tab3_fig, tab3_explanation):
    tab3_content = mo.vstack([tab3_fig, tab3_explanation])
    return (tab3_content,)


# =====================================================================
# TAB 4 — Putting it to work: a boundary a linear kernel cannot fit
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    R_slider = mo.ui.slider(0.5, 2.0, value=1.2, step=0.05, label="true boundary radius R")
    n_points_slider = mo.ui.slider(20, 200, value=80, step=10, label="number of training points")
    noise_slider = mo.ui.slider(0.0, 1.0, value=0.3, step=0.05, label="label noise (SD)")
    lam_payoff_slider = mo.ui.slider(0.01, 5.0, value=0.5, step=0.01, label="λ (both kernels)")
    return R_slider, lam_payoff_slider, n_points_slider, noise_slider


@app.cell(hide_code=True)
def _(R_slider, lam_payoff_slider, mo, n_points_slider, noise_slider):
    controls_tab4 = mo.vstack([
        mo.md(
            "## Putting it to work: linear vs quadratic kernel on a circular pattern\n\n"
            "Target: $y=R^2-\\|x\\|^2$ (positive inside the circle, negative "
            "outside), corrupted by noise. Both models are ridge regression "
            "— one with kernel $\\kappa(a,b)=a'b$ (linear), one with "
            "$\\kappa(a,b)=(1+a'b)^2$ (quadratic) — fit with the **same** λ."
        ),
        mo.hstack([R_slider, n_points_slider], justify="start"),
        mo.hstack([noise_slider, lam_payoff_slider], justify="start"),
    ])
    return (controls_tab4,)


@app.cell(hide_code=True)
def _(
    R_slider,
    kernel_linear,
    kernel_quadratic,
    krr_fit,
    lam_payoff_slider,
    n_points_slider,
    noise_slider,
    np,
):
    np.random.seed(2026)
    X_payoff = np.random.uniform(-2.0, 2.0, size=(n_points_slider.value, 2))
    r2_payoff = np.sum(X_payoff ** 2, axis=1)
    y_true_payoff = R_slider.value ** 2 - r2_payoff
    y_noisy_payoff = y_true_payoff + noise_slider.value * np.random.randn(n_points_slider.value)

    K_lin_payoff = kernel_linear(X_payoff, X_payoff)
    K_quad_payoff = kernel_quadratic(X_payoff, X_payoff)
    alpha_lin_payoff = krr_fit(K_lin_payoff, y_noisy_payoff, lam_payoff_slider.value)
    alpha_quad_payoff = krr_fit(K_quad_payoff, y_noisy_payoff, lam_payoff_slider.value)

    _grid = np.linspace(-2.0, 2.0, 90)
    GX_payoff, GY_payoff = np.meshgrid(_grid, _grid)
    Xg_payoff = np.column_stack([GX_payoff.ravel(), GY_payoff.ravel()])

    pred_lin_payoff = (kernel_linear(Xg_payoff, X_payoff) @ alpha_lin_payoff).reshape(GX_payoff.shape)
    pred_quad_payoff = (kernel_quadratic(Xg_payoff, X_payoff) @ alpha_quad_payoff).reshape(GX_payoff.shape)

    # In-sample sign agreement — a quick proxy for how well each model recovers the circle
    acc_lin = float(np.mean(np.sign(kernel_linear(X_payoff, X_payoff) @ alpha_lin_payoff) == np.sign(y_true_payoff)))
    acc_quad = float(np.mean(np.sign(kernel_quadratic(X_payoff, X_payoff) @ alpha_quad_payoff) == np.sign(y_true_payoff)))
    return (
        GX_payoff,
        GY_payoff,
        X_payoff,
        acc_lin,
        acc_quad,
        pred_lin_payoff,
        pred_quad_payoff,
        y_true_payoff,
    )


@app.cell(hide_code=True)
def _(
    GX_payoff,
    GY_payoff,
    X_payoff,
    go,
    make_subplots,
    pred_lin_payoff,
    pred_quad_payoff,
    y_true_payoff,
):
    def _panel(fig, pred, col):
        fig.add_trace(go.Contour(
            x=GX_payoff[0], y=GY_payoff[:, 0], z=pred,
            colorscale="RdBu", zmid=0, showscale=False,
            contours=dict(coloring="heatmap"), hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Contour(
            x=GX_payoff[0], y=GY_payoff[:, 0], z=pred,
            contours=dict(start=0, end=0, size=1, coloring="lines", showlabels=False),
            line=dict(color="black", width=3), showscale=False, hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=X_payoff[:, 0], y=X_payoff[:, 1], mode="markers",
            marker=dict(
                size=6,
                color=["#1a1a1a" if yy >= 0 else "#f2f2f2" for yy in y_true_payoff],
                line=dict(color="white", width=0.5),
            ),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

    tab4_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Linear kernel κ(a,b)=a'b — a straight boundary",
                         "Quadratic kernel κ(a,b)=(1+a'b)² — a curved boundary"),
        horizontal_spacing=0.08,
    )
    _panel(tab4_fig, pred_lin_payoff, 1)
    _panel(tab4_fig, pred_quad_payoff, 2)

    for _c in (1, 2):
        tab4_fig.update_xaxes(range=[-2, 2], title="x₁", row=1, col=_c)
        tab4_fig.update_yaxes(range=[-2, 2], title="x₂", scaleanchor=f"x{'' if _c == 1 else 2}",
                              scaleratio=1, row=1, col=_c)
    tab4_fig = tab4_fig.update_layout(height=460, margin=dict(l=45, r=20, t=45, b=40), uirevision="payoff")
    return (tab4_fig,)


@app.cell(hide_code=True)
def _(R_slider, acc_lin, acc_quad, mo):
    tab4_explanation = mo.md(
        f"""
        ### Reading the picture

        Black points sit inside the true circle of radius {R_slider.value:.2f}
        (target $y>0$), white points outside. The bold black contour is each
        model's zero-level set — its learned decision boundary.

        The **linear** kernel can only bend feature space by a hyperplane
        through the origin, so its boundary is forced to be a **straight
        line** through $(0,0)$ no matter how the data curve — in-sample sign
        agreement **{acc_lin:.0%}**.

        The **quadratic** kernel has $q_1+q_2=\\|x\\|^2$ sitting inside its
        feature space (§4.4's flat plane, exactly), so a linear fit *in that
        space* can trace out a genuine circle back in $x$-space — in-sample
        sign agreement **{acc_quad:.0%}**. Nothing about the ridge solver
        changed between the two panels: only which nonlinear features it was
        allowed to use.
        """
    )
    return (tab4_explanation,)


@app.cell(hide_code=True)
def _(mo, tab4_explanation, tab4_fig):
    tab4_content = mo.vstack([tab4_fig, tab4_explanation])
    return (tab4_content,)


# -------------------------------------------------------------------
# Main layout
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(
    controls_tab1,
    controls_tab2,
    controls_tab3,
    controls_tab4,
    mo,
    tab1_content,
    tab2_content,
    tab3_content,
    tab4_content,
):
    mo.ui.tabs({
        "🧮 Kernel Trick (§4.2)": mo.vstack([controls_tab1, tab1_content]),
        "📐 Worked Example (§4.3)": mo.vstack([controls_tab2, tab2_content]),
        "📦 Feature Geometry (§4.4)": mo.vstack([controls_tab3, tab3_content]),
        "🎯 Putting It to Work": mo.vstack([controls_tab4, tab4_content]),
    })
    return


if __name__ == "__main__":
    app.run()
