# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="Gaussian Classifiers")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from scipy.stats import multivariate_normal
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.metrics import accuracy_score
    import plotly.graph_objects as go

    return (
        GaussianNB,
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        accuracy_score,
        go,
        mo,
        multivariate_normal,
        np,
    )


@app.cell(hide_code=True)
def _():
    COLOUR_C0 = "#0072BD"     # blue — Class 0
    COLOUR_C1 = "#D9531A"     # orange — Class 1
    COLOUR_BAYES = "#000000"  # black — true Bayes-optimal boundary
    COLOUR_NB = "#7E2F8E"     # purple — Gaussian Naive Bayes
    COLOUR_LDA = "#17BECF"    # cyan — LDA
    COLOUR_QDA = "#EDB120"    # gold — QDA
    return COLOUR_BAYES, COLOUR_C0, COLOUR_C1, COLOUR_LDA, COLOUR_NB, COLOUR_QDA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Classifiers: Naive Bayes, LDA & QDA

    Three classifiers built on the same idea — **Bayes' theorem** — but with
    progressively less restrictive assumptions about the class-conditional
    covariance structure of two Gaussian features:

    | Classifier | Covariance assumption | Boundary shape |
    |:---|:---|:---|
    | **Gaussian Naive Bayes** | Diagonal, *possibly different per class* (features independent given $Y$) | Quadratic, axis-aligned bias |
    | **LDA** | Full, but **shared** across classes ($\Sigma_0 = \Sigma_1 = \Sigma$) | Linear |
    | **QDA** *(not in the notes — a natural extension)* | Full, **separate** per class | Quadratic |

    **Use the controls to:**

    1. Set the true population mean vectors and covariance matrices for two
       classes (subject to a positive semi-definite constraint, enforced
       automatically) and inspect the exact **theoretical Bayes decision
       boundary**.
    2. Simulate a labelled sample from that population and compare the
       **estimated** decision boundaries produced by Gaussian Naive Bayes,
       LDA and QDA against the truth.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theoretical Foundation

    For two classes $k \in \{0, 1\}$ with class-conditional densities
    $f_{\mathbf{X}|Y}(\mathbf{x}|k)$ and priors $P(Y=k)$, Bayes' theorem gives

    $$
    P(Y = k \mid \mathbf{X} = \mathbf{x}) \propto f_{\mathbf{X}|Y}(\mathbf{x}|k)\, P(Y=k),
    $$

    and the **Bayes classifier** assigns $\mathbf{x}$ to the class with the
    larger posterior, $\hat{y} = \arg\max_k P(Y=k\mid\mathbf{X}=\mathbf{x})$.
    When class densities are multivariate normal,
    $\mathbf{X}\mid Y=k \sim N(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, the
    decision boundary is the set of points where the two posteriors are
    equal:

    $$
    \log P(Y{=}1) + \log f(\mathbf{x}\mid 1) \;=\; \log P(Y{=}0) + \log f(\mathbf{x}\mid 0).
    $$

    - If $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1$, the quadratic terms
      $\mathbf{x}^\top\boldsymbol{\Sigma}_k^{-1}\mathbf{x}$ cancel and the
      boundary is **linear** — this is exactly the LDA discriminant
      $\delta_k(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k - \tfrac{1}{2}\boldsymbol{\mu}_k^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \log P(Y{=}k)$.
    - If $\boldsymbol{\Sigma}_0 \neq \boldsymbol{\Sigma}_1$, the quadratic
      terms do **not** cancel and the true boundary is a conic section
      (quadratic) — this is what **QDA** models by fitting a separate
      $\boldsymbol{\Sigma}_k$ per class.
    - **Gaussian Naive Bayes** goes one step further than QDA: it also
      forces each $\boldsymbol{\Sigma}_k$ to be **diagonal**, i.e. it ignores
      correlation between the two features within a class.

    Below, means and (co)variances are population **parameters** you set
    directly; the fitted models instead **estimate** $\hat{\mu}_{jk}$ and
    $\hat{\sigma}^2_{jk}$ (or $\hat{\boldsymbol{\Sigma}}_k$) from a simulated
    sample, exactly as in the medical diagnosis example in the notes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mu0_1 = mo.ui.slider(-6.0, 8.0, value=0.0, step=0.5, label="μ₀₁")
    mu0_2 = mo.ui.slider(-6.0, 8.0, value=0.0, step=0.5, label="μ₀₂")
    sigma0_1 = mo.ui.slider(0.3, 3.0, value=1.0, step=0.1, label="σ₀₁")
    sigma0_2 = mo.ui.slider(0.3, 3.0, value=1.0, step=0.1, label="σ₀₂")
    rho0 = mo.ui.slider(-0.9, 0.9, value=0.0, step=0.05, label="ρ₀")

    mu1_1 = mo.ui.slider(-6.0, 8.0, value=3.0, step=0.5, label="μ₁₁")
    mu1_2 = mo.ui.slider(-6.0, 8.0, value=2.5, step=0.5, label="μ₁₂")
    sigma1_1 = mo.ui.slider(0.3, 3.0, value=1.6, step=0.1, label="σ₁₁")
    sigma1_2 = mo.ui.slider(0.3, 3.0, value=0.8, step=0.1, label="σ₁₂")
    rho1 = mo.ui.slider(-0.9, 0.9, value=0.55, step=0.05, label="ρ₁")

    prior1 = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="P(Y=1)")

    pop_controls = mo.md(
        f"""
        | Parameter | Class 0 | Class 1 |
        |:---|:---|:---|
        | Mean $\\mu_1$ | {mu0_1} | {mu1_1} |
        | Mean $\\mu_2$ | {mu0_2} | {mu1_2} |
        | Std dev $\\sigma_1$ | {sigma0_1} | {sigma1_1} |
        | Std dev $\\sigma_2$ | {sigma0_2} | {sigma1_2} |
        | Correlation $\\rho$ | {rho0} | {rho1} |

        **Prior** $P(Y=1)$: {prior1}

        *Tip: set Class 1's σ and ρ equal to Class 0's to satisfy the LDA
        equal-covariance assumption exactly; change them to see the true
        boundary curve away from a straight line.*
        """
    )
    return (
        mu0_1,
        mu0_2,
        mu1_1,
        mu1_2,
        pop_controls,
        prior1,
        rho0,
        rho1,
        sigma0_1,
        sigma0_2,
        sigma1_1,
        sigma1_2,
    )


@app.cell(hide_code=True)
def _(
    mu0_1,
    mu0_2,
    mu1_1,
    mu1_2,
    np,
    prior1,
    rho0,
    rho1,
    sigma0_1,
    sigma0_2,
    sigma1_1,
    sigma1_2,
):
    def _make_cov(s1, s2, rho):
        return np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])

    mu0 = np.array([mu0_1.value, mu0_2.value])
    mu1 = np.array([mu1_1.value, mu1_2.value])
    Sigma0 = _make_cov(sigma0_1.value, sigma0_2.value, rho0.value)
    Sigma1 = _make_cov(sigma1_1.value, sigma1_2.value, rho1.value)
    prior0 = 1.0 - prior1.value
    prior1_val = prior1.value
    return Sigma0, Sigma1, mu0, mu1, prior0, prior1_val


@app.cell(hide_code=True)
def _(Sigma0, Sigma1, mo, np):
    def _fmt_mat(S):
        return (
            rf"\begin{{pmatrix}} {S[0, 0]:.2f} & {S[0, 1]:.2f} \\ "
            rf"{S[1, 0]:.2f} & {S[1, 1]:.2f} \end{{pmatrix}}"
        )

    _eig0 = np.linalg.eigvalsh(Sigma0)
    _eig1 = np.linalg.eigvalsh(Sigma1)
    equal_cov = bool(np.allclose(Sigma0, Sigma1, atol=1e-8))

    cov_md = mo.md(
        rf"""
        **Population covariance matrices** — positive semi-definite by
        construction, since $\sigma_1,\sigma_2 > 0$ and $|\rho| < 1$:

        $$
        \Sigma_0 = {_fmt_mat(Sigma0)}, \qquad
        \Sigma_1 = {_fmt_mat(Sigma1)}
        $$

        Eigenvalues: $\Sigma_0 \to$ ({_eig0[0]:.2f}, {_eig0[1]:.2f}),
        $\Sigma_1 \to$ ({_eig1[0]:.2f}, {_eig1[1]:.2f}) — both strictly
        positive confirms positive-definiteness.

        {"**Equal covariances** ⇒ the true Bayes boundary is *linear*, matching the LDA assumption." if equal_cov else "**Unequal covariances** ⇒ the true Bayes boundary is *quadratic* — only QDA (or the theoretical rule) can represent it exactly."}
        """
    )
    return cov_md, equal_cov


@app.cell(hide_code=True)
def _(Sigma0, Sigma1, mu0, mu1, np):
    _pad = 3.0
    _means = np.vstack([mu0, mu1])
    _max_std = max(
        float(np.sqrt(np.max(np.diag(Sigma0)))),
        float(np.sqrt(np.max(np.diag(Sigma1)))),
    )
    _x_min = _means[:, 0].min() - 3.5 * _max_std - _pad
    _x_max = _means[:, 0].max() + 3.5 * _max_std + _pad
    _y_min = _means[:, 1].min() - 3.5 * _max_std - _pad
    _y_max = _means[:, 1].max() + 3.5 * _max_std + _pad

    _xs = np.linspace(_x_min, _x_max, 150)
    _ys = np.linspace(_y_min, _y_max, 150)
    xx, yy = np.meshgrid(_xs, _ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    return grid_points, xx, yy


@app.cell(hide_code=True)
def _(Sigma0, Sigma1, grid_points, mu0, mu1, multivariate_normal, np, prior0, prior1_val, xx):
    _rv0 = multivariate_normal(mean=mu0, cov=Sigma0)
    _rv1 = multivariate_normal(mean=mu1, cov=Sigma1)

    dens0 = _rv0.pdf(grid_points).reshape(xx.shape)
    dens1 = _rv1.pdf(grid_points).reshape(xx.shape)

    _log_post0 = np.log(prior0) + _rv0.logpdf(grid_points)
    _log_post1 = np.log(prior1_val) + _rv1.logpdf(grid_points)
    bayes_score = (_log_post1 - _log_post0).reshape(xx.shape)
    return bayes_score, dens0, dens1


@app.cell(hide_code=True)
def _(mo):
    show_density_theory = mo.ui.checkbox(value=True, label="Class density contours")
    show_bayes_boundary_theory = mo.ui.checkbox(
        value=True, label="True Bayes decision boundary"
    )

    theory_toggle_controls = mo.md(
        f"""
        **Display options**

        {show_density_theory}

        {show_bayes_boundary_theory}
        """
    )
    return show_bayes_boundary_theory, show_density_theory, theory_toggle_controls


@app.cell(hide_code=True)
def _(
    COLOUR_BAYES,
    COLOUR_C0,
    COLOUR_C1,
    bayes_score,
    dens0,
    dens1,
    go,
    mu0,
    mu1,
    show_bayes_boundary_theory,
    show_density_theory,
    xx,
    yy,
):
    _fig = go.Figure()

    if show_density_theory.value:
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=dens0,
            showscale=False, ncontours=8,
            colorscale=[[0, COLOUR_C0], [1, COLOUR_C0]],
            contours=dict(coloring="lines"), line=dict(width=1),
            opacity=0.6, name="f(x | Class 0)", showlegend=True,
        ))
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=dens1,
            showscale=False, ncontours=8,
            colorscale=[[0, COLOUR_C1], [1, COLOUR_C1]],
            contours=dict(coloring="lines"), line=dict(width=1),
            opacity=0.6, name="f(x | Class 1)", showlegend=True,
        ))

    if show_bayes_boundary_theory.value:
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=bayes_score,
            contours=dict(start=0, end=0, size=2, coloring="lines"),
            line=dict(color=COLOUR_BAYES, width=3),
            showscale=False, name="True Bayes boundary", showlegend=True,
        ))

    _fig.add_trace(go.Scatter(
        x=[mu0[0]], y=[mu0[1]], mode="markers",
        marker=dict(color=COLOUR_C0, size=13, symbol="x", line=dict(width=2)),
        name="μ₀",
    ))
    _fig.add_trace(go.Scatter(
        x=[mu1[0]], y=[mu1[1]], mode="markers",
        marker=dict(color=COLOUR_C1, size=13, symbol="x", line=dict(width=2)),
        name="μ₁",
    ))

    _fig.update_layout(
        title="Theoretical Bayes-Optimal Decision Boundary",
        xaxis_title="X₁", yaxis_title="X₂",
        width=650, height=550,
        legend=dict(orientation="h", y=-0.15),
    )

    fig_theory = _fig
    return (fig_theory,)


@app.cell(hide_code=True)
def _(mo):
    n_total = mo.ui.slider(50, 600, value=200, step=10, label="Total sample size n")
    seed_btn = mo.ui.button(value=0, on_click=lambda v: v + 1, label="\U0001F3B2 New sample")

    sim_controls = mo.md(
        f"""
        **Simulation** — class labels are drawn according to the prior
        $P(Y{{=}}1)$ set above, then features are drawn from the
        corresponding population Gaussian.

        | Parameter | Value |
        |:---|:---|
        | Total sample size (n) | {n_total} |
        | Resample | {seed_btn} |
        """
    )
    return n_total, seed_btn, sim_controls


@app.cell(hide_code=True)
def _(Sigma0, Sigma1, mu0, mu1, n_total, np, prior1_val, seed_btn):
    _rng = np.random.default_rng(2026 + seed_btn.value)
    _n = n_total.value
    y_train = (_rng.random(_n) < prior1_val).astype(int)
    _n1 = int(y_train.sum())
    _n0 = _n - _n1

    X_train = np.empty((_n, 2))
    X_train[y_train == 0] = _rng.multivariate_normal(mu0, Sigma0, size=_n0)
    X_train[y_train == 1] = _rng.multivariate_normal(mu1, Sigma1, size=_n1)
    return X_train, y_train


@app.cell(hide_code=True)
def _(
    GaussianNB,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    X_train,
    grid_points,
    xx,
    y_train,
):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)

    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)

    nb_score = (nb_model.predict_proba(grid_points)[:, 1] - 0.5).reshape(xx.shape)
    lda_score = lda_model.decision_function(grid_points).reshape(xx.shape)
    qda_score = qda_model.decision_function(grid_points).reshape(xx.shape)

    return lda_model, lda_score, nb_model, nb_score, qda_model, qda_score


@app.cell(hide_code=True)
def _(mo):
    show_points = mo.ui.checkbox(value=True, label="Simulated training points")
    show_density_cmp = mo.ui.checkbox(value=False, label="True density contours")
    show_bayes_cmp = mo.ui.checkbox(value=True, label="True Bayes boundary")
    show_nb = mo.ui.checkbox(value=True, label="Gaussian Naive Bayes boundary")
    show_lda = mo.ui.checkbox(value=True, label="LDA boundary")
    show_qda = mo.ui.checkbox(value=True, label="QDA boundary")
    shade_select = mo.ui.dropdown(
        options=["None", "True Bayes", "Naive Bayes", "LDA", "QDA"],
        value="None", label="Shade decision regions for",
    )

    cmp_toggle_controls = mo.md(
        f"""
        **Display options**

        {show_points}

        {show_density_cmp}

        {show_bayes_cmp}

        {show_nb}

        {show_lda}

        {show_qda}

        {shade_select}
        """
    )
    return (
        cmp_toggle_controls,
        shade_select,
        show_bayes_cmp,
        show_density_cmp,
        show_lda,
        show_nb,
        show_points,
        show_qda,
    )


@app.cell(hide_code=True)
def _(
    COLOUR_BAYES,
    COLOUR_C0,
    COLOUR_C1,
    COLOUR_LDA,
    COLOUR_NB,
    COLOUR_QDA,
    X_train,
    bayes_score,
    dens0,
    dens1,
    go,
    lda_score,
    nb_score,
    qda_score,
    shade_select,
    show_bayes_cmp,
    show_density_cmp,
    show_lda,
    show_nb,
    show_points,
    show_qda,
    xx,
    y_train,
    yy,
):
    _fig = go.Figure()

    _shade_map = {
        "True Bayes": bayes_score,
        "Naive Bayes": nb_score,
        "LDA": lda_score,
        "QDA": qda_score,
    }

    if shade_select.value != "None":
        _z = _shade_map[shade_select.value]
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=(_z > 0).astype(int),
            showscale=False,
            colorscale=[[0, "lightcoral"], [1, "lightblue"]],
            opacity=0.25, line=dict(width=0),
            contours=dict(coloring="fill"),
            hoverinfo="skip", showlegend=False,
        ))

    if show_density_cmp.value:
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=dens0,
            showscale=False, ncontours=6,
            colorscale=[[0, COLOUR_C0], [1, COLOUR_C0]],
            contours=dict(coloring="lines"), line=dict(width=1),
            opacity=0.5, name="f(x | Class 0)", showlegend=True,
        ))
        _fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=dens1,
            showscale=False, ncontours=6,
            colorscale=[[0, COLOUR_C1], [1, COLOUR_C1]],
            contours=dict(coloring="lines"), line=dict(width=1),
            opacity=0.5, name="f(x | Class 1)", showlegend=True,
        ))

    _boundary_specs = [
        (show_bayes_cmp, bayes_score, COLOUR_BAYES, "True Bayes", "solid"),
        (show_nb, nb_score, COLOUR_NB, "Gaussian NB", "dot"),
        (show_lda, lda_score, COLOUR_LDA, "LDA", "dash"),
        (show_qda, qda_score, COLOUR_QDA, "QDA", "dashdot"),
    ]
    for _toggle, _z, _colour, _label, _dash in _boundary_specs:
        if _toggle.value:
            _fig.add_trace(go.Contour(
                x=xx[0], y=yy[:, 0], z=_z,
                contours=dict(start=0, end=0, size=2, coloring="lines"),
                line=dict(color=_colour, width=3, dash=_dash),
                showscale=False, name=f"{_label} boundary", showlegend=True,
            ))

    if show_points.value:
        for _cls, _colour, _label in [
            (0, COLOUR_C0, "Class 0 (train)"),
            (1, COLOUR_C1, "Class 1 (train)"),
        ]:
            _mask = y_train == _cls
            _fig.add_trace(go.Scatter(
                x=X_train[_mask, 0], y=X_train[_mask, 1], mode="markers",
                marker=dict(color=_colour, size=7, line=dict(color="black", width=0.5)),
                name=_label,
            ))

    _fig.update_layout(
        title="Estimated Decision Boundaries vs. True Bayes Boundary",
        xaxis_title="X₁", yaxis_title="X₂",
        width=700, height=580,
        legend=dict(orientation="h", y=-0.15),
    )

    fig_cmp = _fig
    return (fig_cmp,)


@app.cell(hide_code=True)
def _(
    Sigma0,
    Sigma1,
    accuracy_score,
    equal_cov,
    lda_model,
    mo,
    mu0,
    mu1,
    multivariate_normal,
    nb_model,
    np,
    prior0,
    prior1_val,
    qda_model,
    seed_btn,
):
    _rng = np.random.default_rng(99999 + seed_btn.value)
    _n_test = 1000
    _y_test = (_rng.random(_n_test) < prior1_val).astype(int)
    _n1 = int(_y_test.sum())
    _n0 = _n_test - _n1

    _X_test = np.empty((_n_test, 2))
    _X_test[_y_test == 0] = _rng.multivariate_normal(mu0, Sigma0, size=_n0)
    _X_test[_y_test == 1] = _rng.multivariate_normal(mu1, Sigma1, size=_n1)

    _rv0 = multivariate_normal(mean=mu0, cov=Sigma0)
    _rv1 = multivariate_normal(mean=mu1, cov=Sigma1)
    _bayes_score_test = (
        (np.log(prior1_val) + _rv1.logpdf(_X_test))
        - (np.log(prior0) + _rv0.logpdf(_X_test))
    )
    _bayes_pred_test = (_bayes_score_test > 0).astype(int)

    _results = {
        "Bayes-optimal (theoretical ceiling)": accuracy_score(_y_test, _bayes_pred_test),
        "Gaussian Naive Bayes": accuracy_score(_y_test, nb_model.predict(_X_test)),
        "LDA": accuracy_score(_y_test, lda_model.predict(_X_test)),
        "QDA": accuracy_score(_y_test, qda_model.predict(_X_test)),
    }
    _rows = "\n        ".join(f"| {_name} | {_acc:.3f} |" for _name, _acc in _results.items())

    _interpretation = (
        "Since the classes share a covariance matrix, LDA's accuracy should "
        "track the Bayes-optimal accuracy closely, while QDA — with more "
        "parameters to estimate — may lag slightly at small *n*."
        if equal_cov
        else "Since the classes have different covariance matrices, QDA (and "
        "the theoretical rule) can represent the true boundary exactly, "
        "while LDA is restricted to a linear approximation and Naive Bayes "
        "further restricts each class's covariance to be diagonal."
    )

    acc_md = mo.md(
        f"""
        Accuracy on an independent test set of {_n_test} points drawn from
        the **true** population model (not used for fitting):

        | Method | Accuracy |
        |:---|---:|
        {_rows}

        No estimated classifier can exceed the Bayes-optimal accuracy on
        average — it is the theoretical ceiling implied by class overlap.
        {_interpretation}
        """
    )
    return (acc_md,)


@app.cell(hide_code=True)
def _(
    acc_md,
    cmp_toggle_controls,
    cov_md,
    fig_cmp,
    fig_theory,
    mo,
    pop_controls,
    sim_controls,
    theory_toggle_controls,
):
    mo.hstack(
        [
            mo.vstack([
                mo.md("### Population Parameters"),
                pop_controls,
                cov_md,
            ]),
            mo.ui.tabs({
                "\U0001F4D0 Theoretical Boundary": mo.vstack([theory_toggle_controls, fig_theory]),
                "\U0001F3B2 Simulated Comparison": mo.vstack([sim_controls, cmp_toggle_controls, fig_cmp]),
                "\U0001F4CA Accuracy": acc_md,
            }),
        ],
        widths=[1, 3],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Selection Considerations

    - **Start simple.** Gaussian Naive Bayes has the fewest parameters (a
      mean and variance per feature per class) and remains reasonable even
      with limited data — but it ignores correlation between features.
    - **LDA** models the full correlation structure while assuming it is
      identical across classes, which yields a linear boundary and only one
      covariance matrix to estimate. It is a good default when that
      assumption is plausible and training data is not abundant.
    - **QDA** relaxes the equal-covariance assumption at the cost of
      estimating a full covariance matrix *per class* — more flexible, but
      it needs more data to estimate reliably, especially as the number of
      features grows.
    - Toggle the density contours and boundary shading above to see exactly
      where each method's assumptions cause it to diverge from the true
      Bayes-optimal boundary, and use the *Accuracy* tab to quantify the
      cost of a wrong assumption versus the cost of estimating more
      parameters from a small sample.

    ---

    STAT312: Advanced Data Analytics — Nelson Mandela University.
    Grounded in Chapter 6 (*Classification Methods: Naive Bayes and Linear
    Discriminant Analysis*) of the STAT312 class notes.
    """)
    return


if __name__ == "__main__":
    app.run()
