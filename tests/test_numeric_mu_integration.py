"""Filon quadrature against direct integration, including the small-phase limit."""

import numpy as np
import pytest
from scipy.integrate import quad

from cosmo_wap.numeric_mu import accel, int_bk, integration


def _direct(func, lo, hi, omega):
    """Real and imaginary parts through scipy's oscillatory quadrature."""
    real_cos = quad(lambda x: func(x).real, lo, hi, weight="cos", wvar=omega)[0]
    imag_sin = quad(lambda x: func(x).imag, lo, hi, weight="sin", wvar=omega)[0]
    real_sin = quad(lambda x: func(x).real, lo, hi, weight="sin", wvar=omega)[0]
    imag_cos = quad(lambda x: func(x).imag, lo, hi, weight="cos", wvar=omega)[0]
    return real_cos - imag_sin + 1j * (real_sin + imag_cos)


@pytest.mark.parametrize("implementation", ["pk", "bk", "numba"])
def test_filon_linear_amplitude(implementation):
    """A complex linear amplitude is exact, on nonuniform nodes at either sign of frequency."""
    nodes = np.geomspace(0.1, 5.0, 31)
    omega = np.array([[0.0, 1e-10, -1e-10, 0.3], [-0.3, 8.0, 80.0, -80.0]])
    a = np.array([[1.2 + 0.7j], [-0.4 + 1.1j]])
    b = np.array([0.3 - 0.2j, -0.6j, 0.5, -0.7 + 0.4j])
    values = a[..., None] + b[..., None] * nodes

    if implementation == "bk":
        got = int_bk._filon(nodes, omega, values)
    else:
        if implementation == "numba":
            pytest.importorskip("numba", reason="numba is optional - accel falls back to numpy")
        integrate = accel._filon_nb if implementation == "numba" else integration.filon_integrate
        got = integrate(nodes, omega[..., None], 1.0, values, 1.0)

    want = np.empty(omega.shape, dtype=complex)
    for i, j in np.ndindex(omega.shape):
        want[i, j] = _direct(lambda x: a[i, 0] + b[j] * x, nodes[0], nodes[-1], omega[i, j])
    np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-10)


def test_filon_broadcast_grids():
    """Each frequency sees its own nonuniform grid and piecewise-linear complex amplitude."""
    nodes = np.array([0.0, 0.02, 0.15, 0.6, 1.4, 2.0]) * np.array([0.7, 1.3])[:, None, None]
    omega = np.array([[0.0, 1e-8, -25.0]])
    rng = np.random.default_rng(27)
    values = rng.normal(size=(2, 3, 6)) + 1j * rng.normal(size=(2, 3, 6))
    got = int_bk._filon(nodes, omega, values)

    want = np.empty((2, 3), dtype=complex)
    for i, j in np.ndindex(want.shape):
        grid, vals = nodes[i, 0], values[i, j]

        def amplitude(x):
            return np.interp(x, grid, vals.real) + 1j * np.interp(x, grid, vals.imag)

        want[i, j] = sum(_direct(amplitude, lo, hi, omega[0, j]) for lo, hi in zip(grid[:-1], grid[1:]))
    np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-10)
