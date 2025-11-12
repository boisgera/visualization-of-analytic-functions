import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Potentials
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **complex potential** is a holomorphic function $f : \Omega \to \mathbb{C}$ that we usually decompose  into a **velocity potential** $\phi :\Omega \to \mathbb{R}$ and **stream function** $\psi :\Omega \to \mathbb{R}$ such that for any $z = x + i y \in \Omega$,

    $$
    f(z) = \phi(x, y) + i \psi(x, y).
    $$

    Since $f$ is holomorphic, for every $(x, y) \in \Omega$, the Cauchy-Riemman equations hold:

    $$
    \frac{\partial \phi}{\partial x}(x, y) = \frac{\partial \psi}{\partial y}(x, y), \;\;\;
    \frac{\partial \phi}{\partial y}(x, y) = -\frac{\partial \psi}{\partial x}(x, y).
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **velocity** $v : \Omega \to \R^2$ associated to a velocity potential is its gradient:

    $$
    v= \nabla \phi = \begin{bmatrix} \partial_x \phi \\ \partial_y \phi \end{bmatrix}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **divergence** $\nabla \cdot v$ of a differentiable vector field $v = (v_x, v_y)$ is

    $$
    \nabla \cdot v = \partial_x v_x  + \partial_y v_y.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **rotational** $\nabla \wedge v$ of a differentiable vector field $v = (v_x, v_y)$  is

    $$
    \nabla \wedge v = \partial_x v_y  - \partial_y v_x.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Theorem

    When a velocity field $v$ is derived from a complex potential, its divergence and rotational fields are zero:

    $$
    \nabla \cdot v = 0, \;\;\;
    \nabla \wedge v = 0.
    $$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Proof

    Let $f = \phi + i \psi$ be the standard decomposition of the complex potential
    $f$ associated to $v$. Since $v$ is smooth, the Schwarz theorem applies. Thus, by the Cauchy conditions, we get:

    $$
    \nabla \cdot v
        = \partial_x v_x + \partial_y v_y
        = \partial_x (\partial_x \phi) + \partial_y (\partial_y \phi)
        = \partial_x (\partial_y \psi) + \partial_y (-\partial_x \psi)
        = \partial^2_{xy} \psi - \partial^2_{xy} \psi
        = 0
    $$

    and

    $$
    \nabla \wedge v
        = \partial_x v_y - \partial_y v_x
        = \partial_x (\partial_y \phi) - \partial_y (\partial_x \phi)
        = 0.
    $$


    ///
    """)
    return


@app.cell
def _():
    # TODO: contour, streamline, etc. 
    return


if __name__ == "__main__":
    app.run()
