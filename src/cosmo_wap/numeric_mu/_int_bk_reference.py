"""Direct line-of-sight quadrature and the reference angular projection."""

import itertools

import numpy as np
from scipy.special import sph_harm_y

from cosmo_wap.lib import utils

from ._int_bk_common import (
    U_MAX,
    _det_vars,
    _fields,
    _gram,
    _integrand,
    _kvec,
    _nodes,
    _positions,
    _set_y,
    _sphere,
    _terms,
    solve_modes,
)
from .bk import los_cosines
from .integration import filon_weights as _filon_weights


def _outer(name, term, d, los_leg, n_y):
    """Gauss-Legendre nodes y and weights in r of an outer node variable - in y, or in ln u for those in det M, whose
    integrand falls only as a power of u out to u ~ K_MAX / k"""
    x, w = utils.leggauss(n_y)
    if name in _det_vars(los_leg, term.c):
        y = np.exp(-(x + 1) / 2 * np.log(U_MAX))
        return y, d * y * w / 2 * np.log(U_MAX)  # dr = d dy = d e^{-s} ds, s = ln u
    return (x + 1) / 2, d * w / 2


def _setup(term, d, los_leg, los_mode, n_y, fixed=None):
    """The node variables not in fixed ({name: y}), each on its own trailing axis: returns their names, y, the
    Gauss-Legendre weights of all but the last (_outer), and the last one's Filon variable t with dr/dt (None if no
    nodes) - u on a log-spaced grid, or y on Gauss-Legendre nodes"""
    fixed = fixed or {}
    names = [name for name in _nodes(term, los_leg, los_mode) if name not in fixed]
    nv = len(names)

    y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
    for name, val in fixed.items():
        _set_y(y, name, val)
    weight, filon = 1.0, None
    for v, name in enumerate(names):
        shape = (n_y,) + (1,) * (nv - 1 - v)
        if v < nv - 1:
            yv, wv = _outer(name, term, d, los_leg, n_y)
            weight = weight * wv.reshape(shape)
        elif name in _det_vars(los_leg, term.c):  # dr = d du / u**2
            t = np.geomspace(1, U_MAX, n_y)
            yv, filon = 1 / t, (t, d / t**2)
        else:
            yv = (utils.leggauss(n_y)[0] + 1) / 2
            filon = (yv, np.full(n_y, d))
        _set_y(y, name, yv.reshape(shape))
    return names, y, weight, filon


def _filon(t, omega, f):
    """int f(t) e^{i omega t} dt over the last axis - see _filon_weights, omega broadcasting against f[..., 0]"""
    return np.sum(_filon_weights(t, np.broadcast_to(omega, f.shape[:-1])) * f, axis=-1)


def _term(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, n_y, deg, n_p, fixed=None):
    """One term at the line of sight cosines mus = (mu1, mu2, mu3), summed over its node variables not in fixed - the
    last by Filon, the others by Gauss-Legendre in y, which only converges while their phase is resolved"""
    d = cosmo_funcs.comoving_dist(zz)
    names, y, weight, filon = _setup(term, d, los_leg, los_mode, n_y, fixed)
    nv = len(names)

    def ex(arr):
        return np.asarray(arr)[(...,) + (None,) * nv]

    ks, mus = [ex(k) for k in ks], [ex(mu) for mu in mus]
    gram = {key: ex(val) for key, val in gram.items()}
    val, phase = _integrand(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, y, deg, n_p)
    val = val * weight
    if not nv:
        return val * np.exp(1j * phase)

    # Filon on the last axis: the phase is linear along it, so take out its slope and keep the rest in the amplitude
    t, dr_dt = filon
    phase = np.broadcast_to(phase, val.shape)
    omega = (phase[..., -1] - phase[..., 0]) / (t[-1] - t[0])
    val = _filon(t, omega, val * dr_dt * np.exp(1j * (phase - omega[..., None] * t)))
    return val.sum(axis=tuple(range(-(nv - 1), 0))) if nv > 1 else val


def get_mu_phi_int(
    mu,
    phi,
    kernels1,
    kernels2,
    kernels3,
    cosmo_funcs,
    k1,
    k2,
    k3,
    theta,
    zz,
    los_leg=2,
    orders=(1,),
    perms=(0, 1, 2),
    n_y=256,
    los_mode=True,
    allow_partial=False,
    deg=8,
    n_p=2000,
    max_size=2**22,
):
    """B(k1,k2,k3,mu,phi) from the terms with integrated parts, on the grid of numeric_mu.bk.get_mu_phi (same
    conventions: field i at k_i with kernelsi and tracer ti = i-1, mu and phi 1D on the last two axes). Pointwise only -
    its multipoles need get_multipoles_int's grids, see the module docstring.

    orders: which numbers of integrated parts to keep - 1 is ISS (with SS|I, kappa^(2) on the second order field),
        2 IIS, 3 III, and 0 the local tree (for checks).
    perms: which fields to take as the second order one.
    n_y: nodes per integral off the line of sight - see _setup.
    los_mode: False puts the line of sight field's integrals on nodes too rather than J_f(p) splines (for checks).
    max_size: elements per chunk - the mu axis is split so a term's (mu, phi, nodes) arrays stay below it."""
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    gram = _gram(k1, k2, k3, theta)

    tri_size = np.broadcast(k1, k2, k3, theta).size
    out = np.zeros(np.broadcast_shapes(*(np.shape(x) for x in (k1, k2, k3, theta)), (len(mu), len(phi))), dtype=complex)
    for term in _terms(fields, los_leg, orders, perms, los_mode):
        rows = max(1, max_size // (tri_size * len(phi) * n_y ** len(_nodes(term, los_leg, los_mode))))
        for start in range(0, len(mu), rows):
            sl = slice(start, start + rows)
            mus = los_cosines(mu[sl, None], phi[None, :], k1, k2, k3, theta)
            out[..., sl, :] += _term(
                term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, los_mode, n_y, deg, n_p
            )
    return out


def _phase_vector(term, y, d, los_leg, los_mode, kvec):
    """W with the phase of _integrand W.x, at scalar node positions y: sum_j (R_j - R_cj) q_j"""
    (r1, r2), coef, _ = solve_modes(los_leg, term.c, y)
    R = _positions(term, y, d, los_leg, los_mode)
    A, B = (sum(coef[j][m] * (R[j] - R["c" + j]) for j in "ab") for m in (0, 1))
    return A * kvec[r1] + B * kvec[r2]


def _pole(term, d, k1, k2, k3, theta, los_leg, los_mode, n_y, fixed):
    """In-plane angle from k1 of the vector the term's phase x.W varies along as its Filon variable does - the spike
    sits on the great circle perpendicular to it. With no node it is the line of sight field's own side"""
    kvec = _kvec(k1, k2, theta)
    names, _, _, _ = _setup(term, d, los_leg, los_mode, n_y, fixed)
    if not names:
        vec = kvec[los_leg]
    else:
        W = []
        for val in (1.0, 0.5):  # W is linear along the variable, so two points give its direction
            y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
            for name, fval in (fixed or {}).items():
                _set_y(y, name, fval)
            _set_y(y, names[-1], val)
            W.append(_phase_vector(term, y, d, los_leg, los_mode, kvec))
        vec = W[0] - W[1]
    return np.arctan2(vec[1], vec[0])


def _pole_grid(n_mu, edges):
    """Composite Gauss-Legendre on [-1, 1], the same n_mu // (2 * (len(edges) + 1)) nodes on every panel between 0, the
    edges and 1 (and their mirror images) - so graded edges cluster the nodes at 0 over decades"""
    edges = np.concatenate(([0.0], np.asarray(edges, dtype=float), [1.0]))
    x, w = utils.leggauss(max(n_mu // (2 * (len(edges) - 1)), 2))
    half = [(b - a) / 2 * x + (a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    half_w = [(b - a) / 2 * w for a, b in zip(edges[:-1], edges[1:])]
    nodes, wts = np.concatenate(half), np.concatenate(half_w)
    return np.concatenate((-nodes[::-1], nodes)), np.concatenate((wts[::-1], wts))


def get_multipoles_int(
    kernels1,
    kernels2,
    kernels3,
    lm,
    cosmo_funcs,
    k1,
    k2,
    k3=None,
    theta=None,
    zz=0,
    sigma=None,
    n_mu=160,
    n_phi=16,
    edges=(1e-4, 1e-3, 1e-2, 0.1),
    los_leg=2,
    orders=(1,),
    perms=(0, 1, 2),
    n_y=256,
    los_mode=True,
    allow_partial=False,
    deg=8,
    n_p=2000,
):
    """(l, m) multipoles of the integrated terms, as numeric_mu.bk.get_multipoles: int dOmega conj(Ylm) B, each term on
    its own sphere grid (_sphere: n_mu x n_phi nodes, the cosine panels between edges). A term with two node variables
    is a loop over the outer one, its pole moving with it. Returns array of shape (len(lm), *triangle shape) - see
    get_mu_phi_int for the rest."""
    k3, theta = utils.get_theta_k3(k1, k2, k3, theta)
    k1, k2, k3, theta = utils.enable_broadcasting(k1, k2, k3, theta)
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    gram = _gram(k1, k2, k3, theta)
    d = cosmo_funcs.comoving_dist(zz)

    out = 0
    for term in _terms(fields, los_leg, orders, perms, los_mode):
        outer = _nodes(term, los_leg, los_mode)[:-1]
        grids = [_outer(name, term, d, los_leg, n_y) for name in outer]
        for idx in itertools.product(range(n_y), repeat=len(outer)):
            fixed = {name: grid[0][i] for name, grid, i in zip(outer, grids, idx)}
            wt = np.prod([grid[1][i] for grid, i in zip(grids, idx)])
            alpha = _pole(term, d, k1, k2, k3, theta, los_leg, los_mode, n_y, fixed)
            mus, (mu, phi), weights = _sphere(alpha, k1, k2, k3, theta, *_pole_grid(n_mu, edges), n_phi)
            arr = wt * _term(
                term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, los_mode, n_y, deg, n_p, fixed
            )
            if sigma is not None:
                arr = arr * np.exp(-(1 / 2) * ((k1 * mus[0]) ** 2 + (k2 * mus[1]) ** 2 + (k3 * mus[2]) ** 2) * sigma**2)
            out = out + np.array(
                [
                    2 * np.sum(weights * np.real(sph_harm_y(l, m, np.arccos(mu), phi)) * arr, axis=(-2, -1))
                    for l, m in lm
                ]
            )
    return out
