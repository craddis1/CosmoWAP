"""Optional numba kernels for the per-bin bias rebuild - cubic splines and d/dln(d).

A sampler rebuilds ~20 of these per likelihood call (the beta coefficients and every
lnd_derivatives set, once per redshift bin), always on the same ~100-point survey grid.
At that size scipy's CubicSpline is dominated by its own input handling rather than by the
tridiagonal solve: 70-90 us to build, against ~5-12 us for the loop below, so the fit is
7-15x. Only the *coefficients* are computed here - evaluation still goes through scipy's
compiled PPoly, so nothing downstream changes.

``gradient`` is the same story one step earlier: ClassWAP.lnd_derivatives differentiates on
the (non-uniform) ln(d) grid before splining, and numpy's generic np.gradient costs 39 us
against 2 us jitted. That one reproduces numpy bit for bit, so it is a pure substitution.
(lib.betas.dy_dz is a different case - a uniform grid with edge_order=2 - and stays numpy.)

numba is optional (``pip install cosmowap[fast]``); lib.jit decides whether it is used, and
each kernel sits next to the wrapper that falls back for it. What the kernels do not cover
exactly - too short a grid, or a dtype other than float64 - goes back to scipy/numpy rather
than being approximated (_jittable). tests/test_lib_accel.py checks both paths.
"""

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from cosmo_wap.lib.jit import have_numba, njit

HAVE_NUMBA = have_numba()


def _jittable(x, arr, min_points):
    """Whether the kernels below reproduce scipy/numpy exactly for this input.

    Both take the same thing: a float64 stack of curves (n_curve, len(x)) over a 1D float64
    grid of at least min_points. Anything else falls back rather than being approximated
    here - see the module docstring.
    """
    return (
        HAVE_NUMBA
        and x.ndim == 1
        and arr.ndim == 2
        and arr.shape[-1] == x.size
        and x.size >= min_points
        and x.dtype == np.float64
        and arr.dtype == np.float64
    )


# --- cubic spline construction -------------------------------------------------------------


@njit(cache=True)
def _notaknot_coeffs(x, y):
    """Not-a-knot cubic spline coefficients in scipy's PPoly layout.

    y is (n, n_curve); the result is (4, n-1, n_curve) with
    s(v) = sum_k c[k, i, j] * (v - x[i])**(3-k) on interval i, matching
    ``CubicSpline(x, y, axis=0).c`` term for term.

    The system for the first derivatives s[i] is scipy's exactly (same rows, same
    not-a-knot boundary rows); it is solved by the Thomas algorithm rather than LAPACK's
    banded solve. The interior rows are diagonally dominant and only the two boundary rows
    are not; the first elimination step leaves di[1] = dx[0] + dx[1] exactly, so no pivot is
    lost there either. Dropping the pivoting still costs round-off, and how much depends on
    the grid: bit-identical to scipy on the near-uniform ones this is called on (z_survey and
    ln(d) over it), ~1e-9 relative on randomly clustered ones and ~1e-6 once neighbouring
    spacings differ by orders of magnitude. fastmath is off here because that margin is the
    whole argument for substituting the solver.
    """
    n, m = y.shape

    dx = np.empty(n - 1)
    for i in range(n - 1):
        dx[i] = x[i + 1] - x[i]
        if dx[i] <= 0:  # scipy validates this; silently fitting unsorted x would be worse
            raise ValueError("x must be strictly increasing")

    slope = np.empty((n - 1, m))
    for i in range(n - 1):
        for j in range(m):
            slope[i, j] = (y[i + 1, j] - y[i, j]) / dx[i]

    # tridiagonal system, one row per node: lo[i]*s[i-1] + di[i]*s[i] + up[i]*s[i+1] = b[i]
    lo = np.empty(n)
    di = np.empty(n)
    up = np.empty(n)
    b = np.empty((n, m))

    for i in range(1, n - 1):
        lo[i] = dx[i]
        di[i] = 2.0 * (dx[i - 1] + dx[i])
        up[i] = dx[i - 1]
        for j in range(m):
            b[i, j] = 3.0 * (dx[i] * slope[i - 1, j] + dx[i - 1] * slope[i, j])

    # not-a-knot: the third derivative is continuous across the first and last interior knot
    d_lo = x[2] - x[0]
    di[0] = dx[1]
    up[0] = d_lo
    for j in range(m):
        b[0, j] = ((dx[0] + 2.0 * d_lo) * dx[1] * slope[0, j] + dx[0] ** 2 * slope[1, j]) / d_lo

    d_hi = x[n - 1] - x[n - 3]
    di[n - 1] = dx[n - 3]
    lo[n - 1] = d_hi
    for j in range(m):
        b[n - 1, j] = (dx[n - 2] ** 2 * slope[n - 3, j] + (2.0 * d_hi + dx[n - 2]) * dx[n - 3] * slope[n - 2, j]) / d_hi

    for i in range(1, n):  # forward elimination
        w = lo[i] / di[i - 1]
        di[i] -= w * up[i - 1]
        for j in range(m):
            b[i, j] -= w * b[i - 1, j]

    s = np.empty((n, m))  # back substitution - s[i] is the spline's first derivative at x[i]
    for j in range(m):
        s[n - 1, j] = b[n - 1, j] / di[n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(m):
            s[i, j] = (b[i, j] - up[i] * s[i + 1, j]) / di[i]

    c = np.empty((4, n - 1, m))
    for i in range(n - 1):
        for j in range(m):
            t = (s[i, j] + s[i + 1, j] - 2.0 * slope[i, j]) / dx[i]
            c[0, i, j] = t / dx[i]
            c[1, i, j] = (slope[i, j] - s[i, j]) / dx[i] - t
            c[2, i, j] = s[i, j]
            c[3, i, j] = y[i, j]
    return c


def spline_stack(x, arr):
    """CubicSpline(x, arr.T, axis=0) for a stack of curves arr of shape (n_curve, len(x))."""
    x, arr = np.ascontiguousarray(x), np.asarray(arr)
    if not _jittable(x, arr, min_points=4):  # not-a-knot needs 4 points; below that scipy differs
        return CubicSpline(x, arr.T, axis=0)
    # (n, n_curve): the transpose is a copy either way, and this way the inner loop over
    # curves is contiguous
    y = np.ascontiguousarray(arr.T)
    return PPoly.construct_fast(_notaknot_coeffs(x, y), x, extrapolate=True)


# --- first derivative on a non-uniform grid -------------------------------------------------


@njit(cache=True)
def _gradient_loop(y, x, out):
    """np.gradient(y, x, axis=-1) with numpy's default edge_order=1, for y of shape (n_curve, n).

    Written in numpy's own operation order so the result is bit-identical rather than merely
    close - the interior is its second-order non-uniform stencil, the two edges its
    first-order one-sided difference.
    """
    n_c, n = y.shape
    for j in range(n_c):
        out[j, 0] = (y[j, 1] - y[j, 0]) / (x[1] - x[0])
        out[j, n - 1] = (y[j, n - 1] - y[j, n - 2]) / (x[n - 1] - x[n - 2])
        for i in range(1, n - 1):
            d1 = x[i] - x[i - 1]
            d2 = x[i + 1] - x[i]
            out[j, i] = (
                -(d2 / (d1 * (d1 + d2))) * y[j, i - 1]
                + ((d2 - d1) / (d1 * d2)) * y[j, i]
                + (d1 / (d2 * (d1 + d2))) * y[j, i + 1]
            )


def gradient(y, x):
    """np.gradient(y, x, axis=-1) for a stack of curves y of shape (n_curve, len(x))."""
    y, x = np.asarray(y), np.ascontiguousarray(x)
    if not _jittable(x, y, min_points=2):  # n=2 is all edge and no interior, which numpy allows
        return np.gradient(y, x, axis=-1)
    y = np.ascontiguousarray(y)
    out = np.empty_like(y)
    _gradient_loop(y, x, out)
    return out
