"""Hankel transforms from power spectrum multipoles to correlation function multipoles,

    xi_l(r) = Re[ i^l / (2 pi^2) int dk k^2 P_l(k) j_l(kr) ],

so P_l(k) as CosmoWAP returns it (real for even l, imaginary for odd l) gives a real xi_l(r) in the same
basis - the same line of sight t and pair ordering. Plain quadrature in ln k: it is not the bottleneck (P_l is).

At large r the integral is a cancellation over hundreds of oscillations of j_l(kr), far larger than the result,
that carry no information - power at kr >> 1 averages out. So each r gets its own window (taper), exactly 1 below
k = kr_taper/r and smoothly 0 by three times that. With kr_taper = 200 a log grid of 2^14 points to k_max = 15 h/Mpc
is converged (unchanged at 2^15), and the windows move the Kaiser + WS xi_l by < 2e-5 out to r = 400 Mpc/h (< 3e-6
for 10 Mpc/h bins) against a brute-force grid of 8e4 points. A single Gaussian damping exp(-(k/k_d)^2) is not used -
at k_d = 2 h/Mpc it already takes ~1% off P at k = 0.2 h/Mpc, shifting xi_0(100 Mpc/h) by ~0.4%.

xi_to_pk is the inverse, for a windowed xi that vanishes beyond the survey (lib/survey_window.py).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson
from scipy.special import spherical_jn


def window(k, k1, k2):
    """1 below k1, 0 above k2 and smooth between (zero slope at both ends, in ln k) - as FAST-PT's window.
    k1, k2 can be arrays (one window per row), broadcast against k on the last axis"""
    k1, k2 = np.asarray(k1, dtype=float)[..., None], np.asarray(k2, dtype=float)[..., None]
    x = np.clip(np.log(np.asarray(k, dtype=float) / k1) / np.log(k2 / k1), 0, 1)
    return 1 - (x - np.sin(2 * np.pi * x) / (2 * np.pi))


def taper(k, r, kr_taper=200.0):
    """window for each separation r, shape (len(r), len(k)): from kr_taper/r to 3 kr_taper/r, but never past the
    top of the grid - there it is [k[-1]/3, k[-1]], and xi at r < 3 kr_taper/k[-1] is limited by k[-1] itself"""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore"):
        k1 = np.minimum(kr_taper / np.atleast_1d(np.asarray(r, dtype=float)), k[-1] / 3)
    return window(k, k1, np.minimum(3 * k1, k[-1]))


def bessel_kernel(ell, k, r=None, r_edges=None, n_gl=16):
    """j_l(kr) on (r, k) - or with r_edges its r^2-weighted average over each bin, shape (len(r_edges)-1, len(k)),
    from n_gl Gauss-Legendre nodes per bin (16 is converged to ~1e-13 for bins of 5-10 Mpc/h)"""
    k = np.asarray(k, dtype=float)
    if (r is None) == (r_edges is None):
        raise ValueError("Pass one of r (separations) or r_edges (bin edges).")
    if r_edges is None:
        return spherical_jn(ell, np.outer(np.atleast_1d(np.asarray(r, dtype=float)), k))

    a, b = _bin_edges(r_edges)
    nodes, weights = leggauss(n_gl)
    out = np.zeros((len(a), len(k)))
    for x, w in zip(nodes, weights):  # one node at a time - (bins, k) arrays rather than (bins, n_gl, k)
        rn = (b - a) / 2 * x + (a + b) / 2
        out += (w * (b - a) / 2 * rn**2)[:, None] * spherical_jn(ell, np.outer(rn, k))
    return out / ((b**3 - a**3) / 3)[:, None]  # int r^2 dr - exact for the GL rule too


def pk_to_xi(k, pk_l, ell, r=None, r_edges=None, kr_taper=200.0, k_ir=1e-4, n_gl=16, kernel=None):
    """xi_l(r), or its bin averages with r_edges, from P_l on the grid k (increasing, log-spaced - e.g. 2^14
    points from 1e-5 h/Mpc to k_max).

    kr_taper: where each r's window starts, in units of 1/r (see taper); bins use their centres. None to switch
              the windows off - then the cut at k[-1] rings into small r and large r needs a much finer grid.
    k_ir:     for l = 0, use j_0 - S, S = window(k, k_ir/3, k_ir) (1 below k_ir/3, 0 above k_ir). That removes
              only an r-independent constant, int S k^3 P_0 dlnk, the part of xi_0 the k^-4 (potential) terms
              make diverge as k[0] -> 0 - so xi_0 does not depend on k[0]. S is smooth so the quadrature does
              not leave a grid-dependent constant either. Keep k_ir well below 1/r: for the Kaiser monopole the
              constant is < 4e-7 of xi_0(400 Mpc/h) at k_ir = 1e-4 h/Mpc, but ~0.4% at 1e-3. None or 0 to
              switch it off.
    kernel:   a precomputed bessel_kernel(ell, k, r, r_edges, n_gl), to reuse across calls."""
    k = np.asarray(k, dtype=float)
    lnk = np.log(k)
    f = k**3 * np.asarray(pk_l)
    if kernel is None:
        kernel = bessel_kernel(ell, k, r, r_edges, n_gl)
    if kr_taper:
        if r_edges is not None:
            a, b = _bin_edges(r_edges)
            r = (a + b) / 2
        kernel = kernel * taper(k, r, kr_taper)

    val = simpson(kernel * f, x=lnk, axis=-1)
    if ell == 0 and k_ir:
        val = val - simpson(window(k, k_ir / 3, k_ir) * f, x=lnk, axis=-1)
    return np.real(1j**ell * val) / (2 * np.pi**2)


def xi_to_pk(s, xi_l, ell, k=None, k_edges=None, n_gl=16, kernel=None):
    """P_l(k) = 4 pi (-i)^l int ds s^2 xi_l(s) j_l(ks), or its k^2-weighted averages over the bins k_edges - pk_to_xi's
    inverse, complex as P_l. Trapezoidal on s (increasing, from ~0), so xi_l should vanish by s[-1], as a windowed xi
    does, and s resolve j_l(ks) at the largest k. j_l(ks) is symmetric in k and s, so bessel_kernel(ell, s, k, k_edges,
    n_gl) is the kernel, bins and all - pass it as kernel to reuse."""
    s = np.asarray(s, dtype=float)
    if kernel is None:
        kernel = bessel_kernel(ell, s, k, k_edges, n_gl)
    return (-1j) ** ell * 4 * np.pi * np.trapezoid(kernel * (s**2 * np.asarray(xi_l)), x=s, axis=-1)


def _bin_edges(r_edges):
    edges = np.asarray(r_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("r_edges must be increasing, non-negative bin edges (at least two).")
    return edges[:-1], edges[1:]
