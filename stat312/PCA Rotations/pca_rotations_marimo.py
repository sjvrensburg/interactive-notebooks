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
app = marimo.App(width="full", app_title="PCA as Rotations")


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    import html

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, html, make_subplots, mo, np


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # PCA as a Series of Rotations

        Principal component analysis finds an orthonormal basis
        $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ — the eigenvectors of the
        sample covariance matrix $\mathbf{S} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}'$
        — and re-expresses each centred observation in that basis:

        $$\mathbf{z}_i = \mathbf{V}'\mathbf{x}_i .$$

        Because $\mathbf{V}$ is orthogonal (and we may choose the signs of the
        eigenvectors so that $\det\mathbf{V} = +1$), the map
        $\mathbf{x}\mapsto\mathbf{V}'\mathbf{x}$ is a **pure rotation**: no
        stretching, no reflection, distances and total variance unchanged. Any
        3D rotation can be built from three **plane (Givens) rotations**, so

        $$\mathbf{V}' = \mathbf{G}_3\,\mathbf{G}_2\,\mathbf{G}_1 ,$$

        | Step | Plane | Rotates about | Purpose |
        |:---:|:---:|:---:|:---|
        | $\mathbf{G}_1$ | $x$–$y$ | $z$-axis | swing $\mathbf{v}_1$ into the $x$–$z$ plane |
        | $\mathbf{G}_2$ | $x$–$z$ | $y$-axis | tip $\mathbf{v}_1$ down onto the $x$-axis |
        | $\mathbf{G}_3$ | $y$–$z$ | $x$-axis | spin about PC1 until $\mathbf{v}_2$ lies on the $y$-axis |

        After the third rotation, $\mathbf{v}_3$ has nowhere left to go but the
        $z$-axis. Watch the **covariance matrix** on the right: $\mathbf{G}_2$
        clears the off-diagonals of the first row and column, $\mathbf{G}_3$
        clears the last one, and what remains is
        $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1,\lambda_2,\lambda_3)$ —
        the variances of the principal component scores.

        Press **▶ Play** under the 3D plot (drag to change the camera at any
        time — it is kept between frames).
        """
    )
    return


# -------------------------------------------------------------------
# Linear-algebra helpers
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(np):
    def givens(i, j, theta):
        """3×3 rotation by θ in the (i, j) coordinate plane."""
        G = np.eye(3)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i], G[i, j] = c, s
        G[j, i], G[j, j] = -s, c
        return G

    # The three planes, in the order they are applied: (x,y), (x,z), (y,z).
    PLANES = [(0, 1), (0, 2), (1, 2)]

    def givens_angles(V):
        """Angles (θ₁, θ₂, θ₃) with G₃G₂G₁V = I, for V a rotation matrix.

        This is QR by Givens rotations: each rotation zeros one entry below the
        diagonal of V. Using atan2 keeps the diagonal positive, and since an
        orthogonal upper-triangular matrix with det +1 and positive diagonal is
        the identity, the three rotations together equal V'.
        """
        W = V.copy()
        angles = []
        for (i, j), col in zip(PLANES, [0, 0, 1]):
            theta = np.arctan2(W[j, col], W[i, col])
            W = givens(i, j, theta) @ W
            angles.append(theta)
        return np.array(angles)

    def oriented_eigenbasis(S):
        """Eigen-decomposition of S with the sign choice that rotates least.

        Eigenvectors are only defined up to sign. We try each sign for v₁ and
        v₂, set v₃ = v₁ × v₂ (so det V = +1, a proper rotation), and keep the
        choice whose three Givens angles have the smallest total magnitude.
        """
        lam, V = np.linalg.eigh(S)
        order = np.argsort(lam)[::-1]
        lam, V = lam[order], V[:, order]
        best = None
        for s1 in (1, -1):
            for s2 in (1, -1):
                v1, v2 = s1 * V[:, 0], s2 * V[:, 1]
                Vs = np.column_stack([v1, v2, np.cross(v1, v2)])
                ang = givens_angles(Vs)
                cost = np.abs(ang).sum()
                if best is None or cost < best[0]:
                    best = (cost, Vs, ang)
        return lam, best[1], best[2]

    def smoothstep(t):
        """Ease-in/ease-out so each rotation starts and stops gently."""
        return t * t * (3 - 2 * t)

    return PLANES, givens, oriented_eigenbasis, smoothstep


# -------------------------------------------------------------------
# UI Controls
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    sd1 = mo.ui.slider(0.2, 3.0, value=2.0, step=0.1, label="SD of x")
    sd2 = mo.ui.slider(0.2, 3.0, value=1.2, step=0.1, label="SD of y")
    sd3 = mo.ui.slider(0.2, 3.0, value=1.0, step=0.1, label="SD of z")
    r12 = mo.ui.slider(-0.95, 0.95, value=0.40, step=0.05, label="ρ(x, y)")
    r13 = mo.ui.slider(-0.95, 0.95, value=-0.60, step=0.05, label="ρ(x, z)")
    r23 = mo.ui.slider(-0.95, 0.95, value=-0.70, step=0.05, label="ρ(y, z)")
    n_obs = mo.ui.slider(50, 500, value=250, step=50, label="n (observations)")

    view = mo.ui.radio(
        options=["Rotate the data", "Rotate the axes"],
        value="Rotate the data",
        label="Point of view",
    )
    project = mo.ui.checkbox(value=True, label="Finish by dropping PC3 (project onto PC1–PC2)")
    n_frames = mo.ui.slider(10, 40, value=24, step=2, label="Frames per rotation")
    speed = mo.ui.slider(20, 200, value=60, step=10, label="Frame duration (ms)")
    return n_frames, n_obs, project, r12, r13, r23, sd1, sd2, sd3, speed, view


@app.cell(hide_code=True)
def _(mo, n_frames, n_obs, project, r12, r13, r23, sd1, sd2, sd3, speed, view):
    controls = mo.vstack(
        [
            mo.md("**Population covariance**"),
            sd1, sd2, sd3, r12, r13, r23, n_obs,
            mo.md("**Animation**"),
            view, project, n_frames, speed,
        ],
        gap=0.5,
    )
    return (controls,)


# -------------------------------------------------------------------
# Data and PCA
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(n_obs, np, oriented_eigenbasis, r12, r13, r23, sd1, sd2, sd3):
    sds = np.array([sd1.value, sd2.value, sd3.value])
    R = np.array(
        [
            [1.0, r12.value, r13.value],
            [r12.value, 1.0, r23.value],
            [r13.value, r23.value, 1.0],
        ]
    )

    # Not every triple of correlations is valid. If R is not positive
    # definite, lift its smallest eigenvalues and rescale to unit diagonal.
    r_eig, r_vec = np.linalg.eigh(R)
    corr_adjusted = bool(r_eig.min() < 0.02)
    if corr_adjusted:
        R = r_vec @ np.diag(np.clip(r_eig, 0.02, None)) @ r_vec.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    Sigma = np.outer(sds, sds) * R

    rng = np.random.default_rng(2026)
    X = rng.multivariate_normal(np.zeros(3), Sigma, size=n_obs.value)
    X = X - X.mean(axis=0)
    S = np.cov(X, rowvar=False)

    lam, V, angles = oriented_eigenbasis(S)
    scores = X @ V
    return R, S, V, X, angles, corr_adjusted, lam, scores


# -------------------------------------------------------------------
# Rotation schedule: one cumulative rotation matrix per frame
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(PLANES, angles, givens, lam, n_frames, np, project, smoothstep, view):
    AXIS_NAME = {(0, 1): "z", (0, 2): "y", (1, 2): "x"}
    PLANE_NAME = {(0, 1): "x–y", (0, 2): "x–z", (1, 2): "y–z"}
    HOLD = 8  # frames to pause on at each milestone

    # Each schedule entry: (M, squash, title). M is the cumulative rotation
    # applied so far; squash ∈ [0, 1] shrinks the PC3 coordinate (projection).
    schedule = []
    _M = np.eye(3)
    schedule += [(_M, 0.0, "Start: the centred data in the original x, y, z coordinates")] * HOLD

    step_names = [
        "swing PC1 into the x–z plane",
        "tip PC1 down onto the x-axis",
        "spin about PC1 until PC2 lies on the y-axis",
    ]
    for _k, ((_i, _j), _theta) in enumerate(zip(PLANES, angles)):
        _head = (
            f"Step {_k + 1}: G{_k + 1} rotates in the {PLANE_NAME[(_i, _j)]} plane "
            f"(about the {AXIS_NAME[(_i, _j)]}-axis) by {np.degrees(_theta):+.1f}° — "
            f"{step_names[_k]}"
        )
        for _t in np.linspace(0, 1, n_frames.value + 1)[1:]:
            schedule.append((givens(_i, _j, smoothstep(_t) * _theta) @ _M, 0.0, _head))
        _M = givens(_i, _j, _theta) @ _M
        schedule += [(_M, 0.0, _head)] * HOLD

    done = "Done: G3·G2·G1 = V′ — the coordinates are now the principal component scores"
    schedule += [(_M, 0.0, done)] * HOLD

    do_project = project.value and view.value == "Rotate the data"
    if do_project:
        kept = 100 * lam[:2].sum() / lam.sum()
        proj_head = (
            f"Step 4: drop PC3 — project onto the PC1–PC2 plane "
            f"({kept:.1f}% of the variance retained)"
        )
        for _t in np.linspace(0, 1, n_frames.value + 1)[1:]:
            schedule.append((_M, smoothstep(_t), proj_head))
        schedule += [(_M, 1.0, proj_head)] * HOLD

    milestones = {
        "Start": 0,
        "G₁": HOLD + n_frames.value,
        "G₂": HOLD + 2 * n_frames.value + HOLD,
        "G₃": HOLD + 3 * n_frames.value + 2 * HOLD,
    }
    if do_project:
        milestones["Project"] = len(schedule) - 1
    return milestones, schedule


# -------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(S, V, X, go, html, lam, make_subplots, milestones, mo, np, schedule, scores, speed, view):
    rotate_data = view.value == "Rotate the data"
    PC_COLOURS = ["#d62728", "#2ca02c", "#1f77b4"]
    AXIS_COLOUR = "#555555"
    # Rotations preserve each point's distance from the origin, so a cube
    # enclosing the largest norm keeps every frame inside the same box.
    reach = 1.05 * np.linalg.norm(X, axis=1).max()
    arrow_len = [2.2 * np.sqrt(l) for l in lam]
    cmax = np.abs(S).max()
    labels = ["x", "y", "z"]

    def segment(vec, length):
        p = np.round(vec * length, 4)
        return [0, p[0]], [0, p[1]], [0, p[2]]

    def frame_traces(M, squash):
        """Traces that change from frame to frame, in a fixed order:
        points, three moving arrows, covariance heatmap."""
        P = np.diag([1.0, 1.0, 1.0 - squash])
        if rotate_data:
            pts = X @ M.T @ P               # active: move every point
            moving = [M @ V[:, _k] for _k in range(3)]   # PC directions travel with the data
            lens = arrow_len
        else:
            pts = X                         # passive: the data stay put
            moving = [M.T[:, _k] for _k in range(3)]     # the coordinate axes travel instead
            lens = [reach / 1.05] * 3
        pts = np.round(pts, 4)
        cov = P @ M @ S @ M.T @ P
        out = [go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])]
        for _k in range(3):
            xs, ys, zs = segment(moving[_k], lens[_k])
            out.append(go.Scatter3d(x=xs, y=ys, z=zs))
        out.append(
            go.Heatmap(
                z=cov[::-1],
                text=np.vectorize(lambda v: f"{v:.2f}")(np.where(np.abs(cov) < 5e-3, 0.0, cov)[::-1]),
            )
        )
        return out

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.70, 0.30],
        specs=[[{"type": "scene"}, {"type": "heatmap"}]],
        subplot_titles=("", "Covariance of the current coordinates"),
        horizontal_spacing=0.04,
    )

    M0, sq0, title0 = schedule[0]
    first = frame_traces(M0, sq0)

    # Points, coloured by PC1 score so individual points can be tracked.
    fig.add_trace(
        go.Scatter3d(
            x=first[0].x, y=first[0].y, z=first[0].z,
            mode="markers",
            marker=dict(
                size=3,
                color=scores[:, 0],
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="PC1<br>score", x=0.0, len=0.6, thickness=12),
            ),
            name="observations",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Moving arrows: PCs (data view) or the coordinate axes (axes view).
    for _k in range(3):
        name = f"PC{_k + 1}" if rotate_data else f"{labels[_k]}′ axis"
        fig.add_trace(
            go.Scatter3d(
                x=first[_k + 1].x, y=first[_k + 1].y, z=first[_k + 1].z,
                mode="lines+text",
                line=dict(color=PC_COLOURS[_k] if rotate_data else AXIS_COLOUR, width=8),
                text=["", name.replace(" axis", "")],
                textfont=dict(size=14, color=PC_COLOURS[_k] if rotate_data else AXIS_COLOUR),
                name=name,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Static targets: the fixed axes (data view) or the fixed PCs (axes view).
    for _k in range(3):
        if rotate_data:
            vec, length, colour, name = np.eye(3)[_k], reach / 1.05, AXIS_COLOUR, f"{labels[_k]}-axis"
        else:
            vec, length, colour, name = V[:, _k], arrow_len[_k], PC_COLOURS[_k], f"PC{_k + 1}"
        xs, ys, zs = segment(vec, length)
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines+text",
                line=dict(color=colour, width=4, dash="dash"),
                text=["", name.replace("-axis", "")],
                textfont=dict(size=12, color=colour),
                name=f"{name} (target)",
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=first[4].z,
            text=first[4].text,
            texttemplate="%{text}",
            textfont=dict(size=15),
            x=labels,
            y=labels[::-1],
            colorscale="RdBu",
            reversescale=True,
            zmin=-cmax,
            zmax=cmax,
            showscale=False,
            hovertemplate="Cov(%{y}, %{x}) = %{z:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # Trace indices updated by frames: points (0), moving arrows (1–3), heatmap (7).
    MOVING = [0, 1, 2, 3, 7]
    frames = []
    for _idx, (_M, _sq, _title) in enumerate(schedule):
        tr = frame_traces(_M, _sq)
        frames.append(
            go.Frame(
                name=str(_idx),
                data=tr,
                traces=MOVING,
                layout=go.Layout(title_text=_title),
            )
        )
    fig.frames = frames

    play_args = dict(
        frame=dict(duration=speed.value, redraw=True),
        transition=dict(duration=0),
        fromcurrent=True,
        mode="immediate",
    )
    milestone_at = {idx: label for label, idx in milestones.items()}
    fig.update_layout(
        title=dict(text=title0, x=0.02, font=dict(size=16)),
        height=680,
        margin=dict(l=10, r=10, t=70, b=70),
        uirevision="pca-rotations",
        legend=dict(orientation="h", x=0.72, y=-0.02, xanchor="left", yanchor="top", font=dict(size=11)),
        scene=dict(
            xaxis=dict(range=[-reach, reach], title="x"),
            yaxis=dict(range=[-reach, reach], title="y"),
            zaxis=dict(range=[-reach, reach], title="z"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.25, z=0.85)),
            domain=dict(x=[0.07, 0.68], y=[0.0, 1.0]),
        ),
        xaxis=dict(side="top", scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                showactive=False,
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, play_args]),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.14,
                y=-0.02,
                len=0.52,
                yanchor="top",
                pad=dict(t=0),
                currentvalue=dict(visible=False),
                ticklen=0,
                font=dict(size=11),
                steps=[
                    dict(
                        method="animate",
                        label=milestone_at.get(i, ""),
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                    )
                    for i in range(len(schedule))
                ],
            )
        ],
    )
    # Marimo's Plotly renderer updates traces in place but keeps the old
    # animation frames, so render into an iframe to get a fresh figure (and
    # fresh frames) whenever a control changes. A fixed-height srcdoc iframe
    # is used because mo.iframe auto-resizes to the page and overshoots.
    page = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        config={"displaylogo": False},
        default_width="100%",
        default_height="680px",
    )
    pca_fig = mo.Html(
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        'style="width:100%;height:700px;border:0;"></iframe>'
    )
    return (pca_fig,)


# -------------------------------------------------------------------
# Numerical summary
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(R, S, V, angles, corr_adjusted, lam, mo, np):
    def mat(A):
        rows = r" \\ ".join(" & ".join(f"{v:.3f}" for v in row) for row in A)
        return r"\begin{pmatrix}" + rows + r"\end{pmatrix}"

    pve = 100 * lam / lam.sum()
    warn = (
        mo.callout(
            mo.md(
                "The chosen correlations do not form a valid correlation matrix "
                "(it is not positive definite), so the nearest valid one was used:\n\n"
                + f"$$\\mathbf{{R}} = {mat(R)}$$"
            ),
            kind="warn",
        )
        if corr_adjusted
        else mo.md("")
    )
    summary = mo.vstack(
        [
            warn,
            mo.md(
                rf"""
                ### The numbers behind the animation

                $$\mathbf{{S}} = {mat(S)}
                \qquad
                \mathbf{{V}} = {mat(V)}$$

                with $\det\mathbf{{V}} = {np.linalg.det(V):+.3f}$ (a proper rotation).

                | | PC1 | PC2 | PC3 |
                |---|---:|---:|---:|
                | Eigenvalue $\lambda_k$ | {lam[0]:.3f} | {lam[1]:.3f} | {lam[2]:.3f} |
                | % of variance | {pve[0]:.1f}% | {pve[1]:.1f}% | {pve[2]:.1f}% |

                **Rotation angles:**
                $\theta_1 = {np.degrees(angles[0]):+.1f}^\circ$ (about $z$),
                $\theta_2 = {np.degrees(angles[1]):+.1f}^\circ$ (about $y$),
                $\theta_3 = {np.degrees(angles[2]):+.1f}^\circ$ (about $x$).

                The trace is preserved by every rotation:
                $\operatorname{{tr}}\mathbf{{S}} = {np.trace(S):.3f}
                = \lambda_1+\lambda_2+\lambda_3$.
                """
            ),
        ]
    )
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    notes = mo.accordion(
        {
            "Why can the eigenvector signs be flipped?": mo.md(
                r"""
                If $\mathbf{S}\mathbf{v} = \lambda\mathbf{v}$ then also
                $\mathbf{S}(-\mathbf{v}) = \lambda(-\mathbf{v})$, so each principal
                direction is only defined up to sign. Flipping a sign just mirrors
                that component's scores. The notebook tries the four sign choices
                for $\mathbf{v}_1,\mathbf{v}_2$, sets
                $\mathbf{v}_3 = \mathbf{v}_1\times\mathbf{v}_2$ so that
                $\det\mathbf{V} = +1$ (a rotation rather than a reflection), and
                keeps whichever needs the **smallest total turning**.
                """
            ),
            "How are the three angles found?": mo.md(
                r"""
                Apply Givens rotations to $\mathbf{V}$ to zero its entries below
                the diagonal, one at a time — the same idea as a QR decomposition:

                1. $\theta_1 = \operatorname{atan2}(V_{21}, V_{11})$ zeros $V_{21}$ — PC1 now has no $y$-component.
                2. $\theta_2 = \operatorname{atan2}(V_{31}, V_{11})$ zeros $V_{31}$ — PC1 now points along $x$.
                3. $\theta_3 = \operatorname{atan2}(V_{32}, V_{22})$ zeros $V_{32}$ — PC2 now points along $y$.

                The result $\mathbf{G}_3\mathbf{G}_2\mathbf{G}_1\mathbf{V}$ is
                orthogonal, upper triangular, with positive diagonal and
                determinant $+1$ — which forces it to be $\mathbf{I}$. Hence
                $\mathbf{G}_3\mathbf{G}_2\mathbf{G}_1 = \mathbf{V}'$.
                These are a form of **Euler angles**.
                """
            ),
            "Rotating the data vs rotating the axes": mo.md(
                r"""
                *Rotate the data* (active view): the axes stay fixed and every
                point moves, $\mathbf{x}_i \mapsto \mathbf{M}\mathbf{x}_i$, until
                the cloud lines up with $x$, $y$, $z$.

                *Rotate the axes* (passive view): the cloud stays still and the
                coordinate axes turn, $\mathbf{e}_k \mapsto \mathbf{M}'\mathbf{e}_k$,
                until they land on $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$.

                Both describe the same coordinates $\mathbf{M}\mathbf{x}_i$, which
                is why the covariance panel is identical in the two views. PCA is
                usually thought of passively — choosing better axes — but the
                active view makes the "diagonalising" visible.
                """
            ),
            "Why does the covariance become diagonal?": mo.md(
                r"""
                If $\mathbf{z} = \mathbf{M}\mathbf{x}$ then
                $\operatorname{Cov}(\mathbf{z}) = \mathbf{M}\mathbf{S}\mathbf{M}'$.
                With $\mathbf{M} = \mathbf{V}'$ this is
                $\mathbf{V}'\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}'\mathbf{V} = \boldsymbol{\Lambda}$:
                the principal component scores are **uncorrelated**, with variances
                $\lambda_1 \ge \lambda_2 \ge \lambda_3$. Dropping PC3 discards the
                direction with the least variance, keeping a fraction
                $(\lambda_1+\lambda_2)/(\lambda_1+\lambda_2+\lambda_3)$ of the total.
                """
            ),
        }
    )
    return (notes,)


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(controls, mo, notes, pca_fig, summary):
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([mo.md("### Parameters"), controls]), pca_fig],
                widths=[1, 4],
                align="start",
            ),
            mo.hstack([summary, notes], widths=[1, 1], align="start", gap=2),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
