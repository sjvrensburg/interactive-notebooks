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
    import json

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, html, json, make_subplots, mo, np


# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # PCA as a Series of Rotations

        Picture the standardised data as a cloud of points shaped like a
        **rugby ball**. PCA finds the direction in which the ball is longest
        (PC1), then the longest direction perpendicular to that (PC2), and
        finally the remaining direction (PC3).

        Finding the principal components is the same as **turning the cloud**
        until its longest axis lies along the first coordinate axis, its next
        longest along the second, and so on. After the turning, each point's
        new coordinates are its **principal component scores**. The turning
        does not stretch or squash the cloud, so no information is lost
        until we choose to drop a component.

        The animation does the turning in **three simple turns**, each about
        one of the coordinate axes. Use **⏭ Next step** to go one turn at a
        time, or **▶ Play all** to watch it straight through. Drag the plot
        to look at the cloud from any angle.
        """
    )
    return


# -------------------------------------------------------------------
# Rotation helpers (hidden from students)
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

    # The three planes, in the order they are applied: (1,2), (1,3), (2,3),
    # i.e. turns about the third, second and first axes.
    PLANES = [(0, 1), (0, 2), (1, 2)]

    def givens_angles(V):
        """Angles (θ₁, θ₂, θ₃) with G₃G₂G₁V = I, for V a rotation matrix.

        Each turn zeros one entry below the diagonal of V (QR by Givens
        rotations). An orthogonal upper-triangular matrix with positive
        diagonal and det +1 is the identity, so the turns together equal V'.
        """
        W = V.copy()
        angles = []
        for (i, j), col in zip(PLANES, [0, 0, 1]):
            theta = np.arctan2(W[j, col], W[i, col])
            W = givens(i, j, theta) @ W
            angles.append(theta)
        return np.array(angles)

    def oriented_eigenbasis(S):
        """Eigen-decomposition of S with the sign choice that turns least.

        Eigenvectors are only defined up to sign. Try each sign for v₁ and
        v₂, set v₃ = v₁ × v₂ (so V is a rotation, not a reflection), and keep
        the choice whose three turning angles are smallest in total.
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
        """Ease-in/ease-out so each turn starts and stops gently."""
        return t * t * (3 - 2 * t)

    return PLANES, givens, oriented_eigenbasis, smoothstep


# -------------------------------------------------------------------
# UI Controls
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    r12 = mo.ui.slider(-0.9, 0.9, value=0.5, step=0.1, label="Corr(Z₁, Z₂)")
    r13 = mo.ui.slider(-0.9, 0.9, value=0.1, step=0.1, label="Corr(Z₁, Z₃)")
    r23 = mo.ui.slider(-0.9, 0.9, value=0.6, step=0.1, label="Corr(Z₂, Z₃)")
    n_obs = mo.ui.slider(50, 400, value=200, step=50, label="n")

    view = mo.ui.radio(
        options=["Rotate the points", "Rotate the axes"],
        value="Rotate the points",
        label="What turns?",
    )
    project = mo.ui.checkbox(value=True, label="Finish by dropping PC3")
    speed = mo.ui.dropdown(
        options={"Slow": 110, "Normal": 60, "Fast": 30},
        value="Normal",
        label="Speed",
    )
    return n_obs, project, r12, r13, r23, speed, view


@app.cell(hide_code=True)
def _(mo, n_obs, project, r12, r13, r23, speed, view):
    controls = mo.vstack(
        [
            mo.md("**Correlations**"),
            r12, r13, r23, n_obs,
            mo.md("**Animation**"),
            view, project, speed,
        ],
        gap=0.5,
    )
    return (controls,)


# -------------------------------------------------------------------
# Data and PCA
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(n_obs, np, oriented_eigenbasis, r12, r13, r23):
    R_pop = np.array(
        [
            [1.0, r12.value, r13.value],
            [r12.value, 1.0, r23.value],
            [r13.value, r23.value, 1.0],
        ]
    )

    # Not every triple of correlations is possible. If this one is not, lift
    # the smallest eigenvalues and rescale to the nearest valid matrix.
    r_eig, r_vec = np.linalg.eigh(R_pop)
    corr_adjusted = bool(r_eig.min() < 0.02)
    if corr_adjusted:
        R_pop = r_vec @ np.diag(np.clip(r_eig, 0.02, None)) @ r_vec.T
        d = np.sqrt(np.diag(R_pop))
        R_pop = R_pop / np.outer(d, d)

    rng = np.random.default_rng(2026)
    raw = rng.multivariate_normal(np.zeros(3), R_pop, size=n_obs.value)

    # Standardise, as in the notes: every column has mean 0 and SD 1, so the
    # covariance matrix of Z is the sample correlation matrix R.
    Z = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    R = np.cov(Z, rowvar=False)

    lam, V, angles = oriented_eigenbasis(R)
    scores = Z @ V
    return R, V, Z, angles, corr_adjusted, lam, scores


# -------------------------------------------------------------------
# Animation schedule: one cumulative rotation matrix per frame
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(PLANES, angles, givens, lam, np, project, smoothstep, view):
    N_FRAMES = 24  # frames per turn
    HOLD = 8       # frames spent on each milestone when playing straight through
    AXIS = {(0, 1): "Z₃", (0, 2): "Z₂", (1, 2): "Z₁"}
    _rotate_points = view.value == "Rotate the points"
    _pve = 100 * lam / lam.sum()

    if _rotate_points:
        start = "Start: the standardised data on the original axes Z₁, Z₂, Z₃"
        heads = [
            f"Turn {_k + 1} of 3: turning the points about the {AXIS[_p]}-axis "
            f"by {abs(np.degrees(_a)):.0f}°"
            for _k, (_p, _a) in enumerate(zip(PLANES, angles))
        ]
        achieved = [
            "After turn 1: PC1 now points the same way as Z₁, "
            "just tilted up or down",
            f"After turn 2: PC1 found — the longest axis of the cloud lies "
            f"along Z₁ ({_pve[0]:.0f}% of the variance)",
            f"After turn 3: PC2 and PC3 found — they lie along Z₂ and Z₃. "
            f"The coordinates are now the PC scores",
        ]
    else:
        start = "Start: the standardised data and the original axes Z₁, Z₂, Z₃"
        heads = [
            f"Turn {_k + 1} of 3: turning the axes by {abs(np.degrees(_a)):.0f}°"
            for _k, _a in enumerate(angles)
        ]
        achieved = [
            "After turn 1: the first axis now points the same way as PC1, "
            "just tilted up or down",
            f"After turn 2: PC1 found — the first axis lies along the longest "
            f"axis of the cloud ({_pve[0]:.0f}% of the variance)",
            "After turn 3: PC2 and PC3 found — the second and third axes lie "
            "along them",
        ]

    # Each schedule entry: (M, squash, title). M is the turning applied so
    # far; squash ∈ [0, 1] shrinks the PC3 coordinate (dropping PC3).
    # `stops` holds the frames where "Next step" pauses.
    schedule = []
    stops = []
    _M = np.eye(3)
    schedule += [(_M, 0.0, start)] * HOLD
    stops.append(len(schedule) - 1)

    for _k, ((_i, _j), _theta) in enumerate(zip(PLANES, angles)):
        for _t in np.linspace(0, 1, N_FRAMES + 1)[1:]:
            schedule.append((givens(_i, _j, smoothstep(_t) * _theta) @ _M, 0.0, heads[_k]))
        _M = givens(_i, _j, _theta) @ _M
        schedule += [(_M, 0.0, achieved[_k])] * HOLD
        stops.append(len(schedule) - 1)

    if project.value and _rotate_points:
        kept = _pve[:2].sum()
        for _t in np.linspace(0, 1, N_FRAMES + 1)[1:]:
            schedule.append((_M, smoothstep(_t), "Dropping PC3: flattening the cloud onto the PC1–PC2 plane"))
        schedule += [
            (_M, 1.0, f"PC3 dropped: PC1 and PC2 keep {kept:.0f}% of the variance")
        ] * HOLD
        stops.append(len(schedule) - 1)
    return schedule, stops


# -------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(R, V, Z, go, html, json, lam, make_subplots, mo, np, schedule, scores, speed, stops, view):
    rotate_points = view.value == "Rotate the points"
    PC_COLOURS = ["#d62728", "#2ca02c", "#1f77b4"]
    AXIS_COLOUR = "#555555"
    labels = ["Z₁", "Z₂", "Z₃"]
    # Turning preserves each point's distance from the origin, so a cube
    # enclosing the furthest point keeps every frame inside the same box.
    reach = 1.05 * np.linalg.norm(Z, axis=1).max()
    arrow_len = [2.2 * np.sqrt(l) for l in lam]
    cmax = np.abs(R).max()

    def segment(vec, length):
        p = np.round(vec * length, 4)
        return [0, p[0]], [0, p[1]], [0, p[2]]

    def frame_traces(M, squash):
        """Traces that change from frame to frame, in a fixed order:
        points, three moving arrows, covariance heatmap."""
        P = np.diag([1.0, 1.0, 1.0 - squash])
        if rotate_points:
            pts = Z @ M.T @ P                             # every point turns
            moving = [M @ V[:, _k] for _k in range(3)]    # PCs turn with the points
            lens = arrow_len
        else:
            pts = Z                                       # the points stay put
            moving = [M.T[:, _k] for _k in range(3)]      # the axes turn instead
            lens = [reach / 1.05] * 3
        pts = np.round(pts, 4)
        cov = P @ M @ R @ M.T @ P
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
        column_widths=[0.75, 0.25],
        specs=[[{"type": "scene"}, {"type": "heatmap"}]],
        horizontal_spacing=0.03,
    )

    # Heatmap title placed inside the plot area so long captions above the
    # figure never run into it.
    fig.add_annotation(
        text="Covariance matrix of the<br>current coordinates",
        x=0.885, y=0.93, xref="paper", yref="paper",
        xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14),
    )

    M0, sq0, title0 = schedule[0]
    first = frame_traces(M0, sq0)

    # Points, coloured by PC1 score so individual points can be followed.
    fig.add_trace(
        go.Scatter3d(
            x=first[0].x, y=first[0].y, z=first[0].z,
            mode="markers",
            marker=dict(
                size=3.5,
                color=scores[:, 0],
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(title="PC1<br>score", x=0.0, len=0.55, thickness=12),
            ),
            name="observations",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Moving arrows: the PCs (points view) or the coordinate axes (axes view).
    for _k in range(3):
        name = f"PC{_k + 1}" if rotate_points else labels[_k]
        colour = PC_COLOURS[_k] if rotate_points else AXIS_COLOUR
        fig.add_trace(
            go.Scatter3d(
                x=first[_k + 1].x, y=first[_k + 1].y, z=first[_k + 1].z,
                mode="lines+text",
                line=dict(color=colour, width=8),
                text=["", name],
                textfont=dict(size=15, color=colour),
                name=name if rotate_points else f"{name} axis (turns)",
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Dashed targets: where the moving arrows will end up.
    for _k in range(3):
        if rotate_points:
            vec, length, colour, name = np.eye(3)[_k], reach / 1.05, AXIS_COLOUR, labels[_k]
        else:
            vec, length, colour, name = V[:, _k], 0.85 * reach, PC_COLOURS[_k], f"PC{_k + 1}"
        xs, ys, zs = segment(vec, length)
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines+text",
                line=dict(color=colour, width=4, dash="dash"),
                text=["", name],
                textfont=dict(size=13, color=colour),
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
            hovertemplate="Cov(%{y}, %{x}) = %{z:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # Trace indices updated by frames: points (0), moving arrows (1–3), heatmap (7).
    MOVING = [0, 1, 2, 3, 7]
    frames = []
    for _idx, (_M, _sq, _title) in enumerate(schedule):
        frames.append(
            go.Frame(
                name=str(_idx),
                data=frame_traces(_M, _sq),
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
    fig.update_layout(
        title=dict(text=title0, x=0.02, font=dict(size=17)),
        height=760,
        margin=dict(l=10, r=10, t=60, b=110),
        uirevision="pca-rotations",
        legend=dict(x=0.80, y=0.28, xanchor="left", yanchor="top", font=dict(size=12)),
        scene=dict(
            xaxis=dict(range=[-reach, reach], title="Z₁"),
            yaxis=dict(range=[-reach, reach], title="Z₂"),
            zaxis=dict(range=[-reach, reach], title="Z₃"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.3, y=1.1, z=0.75)),
            domain=dict(x=[0.06, 0.74], y=[0.0, 1.0]),
        ),
        xaxis=dict(side="top", scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain", domain=[0.35, 0.9]),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.06,
                y=-0.01,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                showactive=False,
                buttons=[
                    # "skip" does nothing in Plotly itself; the script added
                    # below catches the click and plays up to the next stop.
                    dict(label="⏭ Next step", method="skip", args=[None]),
                    dict(label="▶ Play all", method="animate", args=[None, play_args]),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                    dict(
                        label="⏮ Reset",
                        method="animate",
                        args=[["0"], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.06,
                y=-0.08,
                len=0.68,
                yanchor="top",
                pad=dict(t=0),
                currentvalue=dict(visible=False),
                ticklen=0,
                font=dict(size=11),
                steps=[
                    dict(
                        method="animate",
                        # Plotly shows only every n-th label on a long slider,
                        # so milestones are announced in the title instead.
                        label="",
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                    )
                    for i in range(len(schedule))
                ],
            )
        ],
    )

    # "Next step": play from the current frame to the next milestone, then
    # stop. Plotly reports each frame it shows, so the position stays in sync
    # with the slider, Play all, Pause and Reset.
    next_step_js = (
        """
        const gd = document.getElementById('{plot_id}');
        const stops = STOPS;
        const nFrames = N_FRAMES;
        let current = 0;
        gd.on('plotly_animatingframe', (e) => { current = parseInt(e.name, 10); });
        gd.on('plotly_buttonclicked', (e) => {
            if (!e.button.label.startsWith('⏭')) return;
            const target = stops.find((s) => s > current);
            if (target === undefined) {
                Plotly.animate(gd, ['0'], {frame: {duration: 0, redraw: true}, mode: 'immediate'});
                return;
            }
            const names = [];
            for (let i = current + 1; i <= target && i < nFrames; i++) names.push(String(i));
            Plotly.animate(gd, names, {
                frame: {duration: DURATION, redraw: true},
                transition: {duration: 0},
                mode: 'immediate',
            });
        });
        """
        .replace("STOPS", json.dumps(stops[1:]))
        .replace("N_FRAMES", str(len(schedule)))
        .replace("DURATION", str(speed.value))
    )

    # Marimo's Plotly renderer updates traces in place but keeps the old
    # animation frames, so render into an iframe to get a fresh figure (and
    # fresh frames) whenever a control changes. A fixed-height srcdoc iframe
    # is used because mo.iframe auto-resizes to the page and overshoots.
    page = fig.to_html(
        post_script=next_step_js,
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        config={"displayModeBar": False},
        default_width="100%",
        default_height="760px",
    )
    pca_fig = mo.Html(
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        'style="width:100%;height:780px;border:0;"></iframe>'
    )
    return (pca_fig,)


# -------------------------------------------------------------------
# What to notice
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(V, corr_adjusted, lam, mo, np):
    _pve = 100 * lam / lam.sum()
    _cum = np.cumsum(_pve)
    warn = (
        mo.callout(
            mo.md(
                "These three correlations cannot all occur together, so the "
                "closest possible set of correlations was used instead."
            ),
            kind="warn",
        )
        if corr_adjusted
        else mo.md("")
    )
    notice = mo.md(
        rf"""
        ### What to notice

        - **Turning only.** The cloud is never stretched or squashed, so the
          total variance stays the same throughout:
          $\lambda_1 + \lambda_2 + \lambda_3 = {lam.sum():.2f} = p$.
        - **The heatmap starts as the correlation matrix $\mathbf{{R}}$.** As
          the turns happen, the off-diagonal entries shrink to zero: the
          principal components are **uncorrelated**.
        - **The diagonal ends as the eigenvalues.** The variance along each
          principal component is its eigenvalue, $\operatorname{{Var}}(\text{{PC}}_j) = \lambda_j$.
        - **Dropping PC3** removes only the smallest variance, so PC1 and PC2
          still keep {_cum[1]:.0f}% of it.
        """
    )
    table = mo.md(
        rf"""
        ### The principal components

        | | PC1 | PC2 | PC3 |
        |---|---:|---:|---:|
        | Eigenvalue $\lambda_j$ | {lam[0]:.2f} | {lam[1]:.2f} | {lam[2]:.2f} |
        | Proportion of variance | {_pve[0]:.1f}% | {_pve[1]:.1f}% | {_pve[2]:.1f}% |
        | Cumulative proportion | {_cum[0]:.1f}% | {_cum[1]:.1f}% | {_cum[2]:.1f}% |
        | Weight on $Z_1$ | {V[0, 0]:.2f} | {V[0, 1]:.2f} | {V[0, 2]:.2f} |
        | Weight on $Z_2$ | {V[1, 0]:.2f} | {V[1, 1]:.2f} | {V[1, 2]:.2f} |
        | Weight on $Z_3$ | {V[2, 0]:.2f} | {V[2, 1]:.2f} | {V[2, 2]:.2f} |

        The weights are the eigenvectors $\mathbf{{v}}_j$: the direction of
        each coloured arrow in the plot.
        """
    )
    summary = mo.vstack([warn, mo.hstack([notice, table], widths=[1, 1], align="start", gap=2)])
    return (summary,)


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
@app.cell(hide_code=True)
def _(controls, mo, pca_fig, summary):
    mo.vstack(
        [
            mo.hstack([controls, pca_fig], widths=[1.3, 6], align="start"),
            summary,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
