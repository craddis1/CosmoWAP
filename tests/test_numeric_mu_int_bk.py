"""numeric_mu.int_bk - the integrated bispectrum (brute-force reference).

The conventions are pinned to code that is already checked: the local tree to numeric_mu.bk, an integrated first order
field on the line of sight to numeric_mu.pk's IS spline, a rescaled one off it to pk's SI (the same Filon), and the
quadrature nodes standing in for the J_f(p) splines (los_mode=False) to IntK1 and IntK2 themselves. With a unit second
order kernel and a unit first order one on the third field, a term is exactly a power spectrum times P(k):
    LOS on the integrated field (IS):         B = P(k2) P^IS(k3, mu3)
    LOS on the second order field (SI):       B = P(k2) P^SI(k1, -mu1)
"""

import numpy as np
import pytest
from scipy.special import spherical_jn

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import _int_bk_radial as ib_radial
from cosmo_wap.numeric_mu import _int_bk_reference as ib_reference
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import int_bk as ib
from cosmo_wap.numeric_mu import pk as npk
from cosmo_wap.numeric_mu.kernels import K1, K2

ZZ = 1.0
MU = np.array([-0.83, -0.2, 0.31, 0.77])
PHI = np.array([0.3, 1.9, 2.8])
SMALL = (0.004, 0.005, 0.006)  # kd ~ 14: the J_f(p) fits are good to ~1e-5 here (see test_numeric_mu_intk2)


def _tri(k1, k2, k3):
    return k1, k2, k3, utils.get_theta(k1, k2, k3)


@pytest.fixture
def unit(monkeypatch):
    """Z1 = 1 and Z2 = 1/2, so 2 Z1 Z2 = 1 - turns a term into a power spectrum"""

    def one(cosmo_funcs, zz, mu, kk, ti=0, **kwargs):
        return np.ones(np.broadcast_shapes(np.shape(mu), np.shape(kk)))

    def half(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
        return 0.5 * np.ones(np.broadcast_shapes(*map(np.shape, (mu1, mu2, q1, q2, cos12))))

    monkeypatch.setattr(K1, "One", staticmethod(one), raising=False)
    monkeypatch.setattr(K2, "One", staticmethod(half), raising=False)


@pytest.mark.parametrize("kern", [["N"], ["N", "LP"]])
def test_local_limit(cosmo_funcs, kern):
    """orders=(0,) is the local tree, whatever the line of sight"""
    tri = _tri(0.01, 0.012, 0.015)
    want = nbk.get_mu_phi(MU, PHI, kern, kern, kern, cosmo_funcs, *tri, ZZ)
    for los_leg in (0, 2):
        got = ib.get_mu_phi_int(MU, PHI, kern, kern, kern, cosmo_funcs, *tri, ZZ, orders=(0,), los_leg=los_leg)
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_local_multipoles(cosmo_funcs):
    """The per-term sphere grids are exact on the polynomial local tree - degree ~10 with Ylm, so >= 6 nodes a panel"""
    kern, lm = ["N", "LP"], [(0, 0), (1, 0), (2, 0), (2, 2)]
    k1, k2, k3, _ = _tri(0.01, 0.012, 0.015)
    want = nbk.get_multipoles(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ)
    got = ib.get_multipoles_int(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, orders=(0,), n_mu=100, n_phi=16)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * np.max(np.abs(want)))


@pytest.mark.parametrize("los_leg", [0, 1, 2])
@pytest.mark.parametrize("c", [0, 1, 2])
def test_solve_modes(los_leg, c):
    """q = M^-1 k satisfies the transverse delta functions k_i = sum_j (y_j [field of j is i] - y_cj [c = i]) q_j"""
    rng = np.random.default_rng(1)
    a, b = (i for i in range(3) if i != c)
    y = dict(zip(("a", "b", "ca", "cb"), rng.uniform(0.1, 1, 4)))
    kv = rng.normal(size=(2, 3))
    kv = np.vstack((kv, -kv.sum(axis=0)))
    (r1, r2), coef, _ = ib.solve_modes(los_leg, c, y)
    q = {j: coef[j][0] * kv[r1] + coef[j][1] * kv[r2] for j in "ab"}
    for i in (r1, r2):
        M = {"a": y["a"] * (a == i) - y["ca"] * (c == i), "b": y["b"] * (b == i) - y["cb"] * (c == i)}
        np.testing.assert_allclose(M["a"] * q["a"] + M["b"] * q["b"], kv[i], atol=1e-12)


def test_is_matches_pk(cosmo_funcs, unit):
    """Lensing on the line of sight field, second order field 1: P(k2) P^IS(k3, mu3), the same spline"""
    tri = _tri(0.01, 0.012, 0.015)
    got = ib.get_mu_phi_int(MU, PHI, ["One"], ["One"], ["L"], cosmo_funcs, *tri, ZZ, perms=(0,))
    mu3 = nbk.los_cosines(MU[:, None], PHI[None, :], *tri)[2]
    want = BaseInt(cosmo_funcs).pk(tri[1], ZZ) * npk.get_mu(mu3, ["L"], ["One"], cosmo_funcs, np.array(tri[2]), ZZ)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_si_matches_pk(cosmo_funcs, unit):
    """Lensing on field 1 with the line of sight on the second order field: P(k2) P^SI(k1, -mu1) - a rescaling.
    pk's SI puts its Filon on Gauss-Legendre nodes in y, here they are log-spaced in u, so this is at their
    convergence"""
    tri = _tri(0.01, 0.012, 0.015)
    mu = np.array([-0.2, 0.31])
    got = ib.get_mu_phi_int(mu, PHI[:1], ["L"], ["One"], ["One"], cosmo_funcs, *tri, ZZ, perms=(2,), n_y=512)[:, 0]
    want = (
        BaseInt(cosmo_funcs).pk(tri[1], ZZ)
        * npk.get_mu(-mu[:, None], ["One"], ["L"], cosmo_funcs, np.array([[tri[0]]]), ZZ, n=2048).ravel()
    )
    np.testing.assert_allclose(got, want, rtol=1e-2)


@pytest.mark.parametrize("perms", [(0,), (2,)])
def test_nodes_reproduce_splines(cosmo_funcs, unit, perms):
    """los_mode=False puts the line of sight field's integrals on Filon nodes: the IS spline (perm 0) and IntK2 with all
    its layouts, the two-point one 2D (perm 2), come back"""
    tri = _tri(*SMALL)
    want = ib.get_mu_phi_int(MU, PHI, ["One"], ["One"], ["L"], cosmo_funcs, *tri, ZZ, perms=perms)
    got = ib.get_mu_phi_int(
        MU, PHI, ["One"], ["One"], ["L"], cosmo_funcs, *tri, ZZ, perms=perms, los_mode=False, n_y=256
    )
    np.testing.assert_allclose(got, want, atol=2e-4 * np.max(np.abs(want)))


def test_conjugate_mirror(cosmo_funcs):
    """B(-mu, pi - phi) = B(mu, phi)* term by term - flipping x conjugates every kernel and every e^{ipr}"""
    kern = ["N", "LP", "L"]
    tri = _tri(*SMALL)
    B = ib.get_mu_phi_int(MU, PHI, kern, kern, kern, cosmo_funcs, *tri, ZZ, n_y=64)
    B_flip = ib.get_mu_phi_int(-MU, np.pi - PHI, kern, kern, kern, cosmo_funcs, *tri, ZZ, n_y=64)
    np.testing.assert_allclose(B_flip, np.conj(B), rtol=1e-10, atol=1e-12 * np.max(np.abs(B)))


@pytest.mark.parametrize("c", [0, 1, 2])
def test_relabel(cosmo_funcs, c):
    """Field 3 on the line of sight at (k1, k2, k3) is field 1 on it at (k3, k1, k2), with second order field c + 1 mod
    3. Every node variable, Filon axis and pole is the same physical one either way, so the monopole (the only multipole
    that does not refer to k1) agrees to rounding - the solver, phases and poles in each labelling of the perm"""
    kern, lm = ["N", "L"], [(0, 0)]
    k1, k2, k3 = SMALL
    kw = dict(zz=ZZ, n_mu=40, n_phi=8, n_y=24)
    A = ib.get_multipoles_int(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, los_leg=2, perms=(c,), **kw)
    B = ib.get_multipoles_int(kern, kern, kern, lm, cosmo_funcs, k3, k1, k2, los_leg=0, perms=((c + 1) % 3,), **kw)
    np.testing.assert_allclose(A, B, rtol=1e-9)


def test_partial_and_png_raise(cosmo_funcs):
    tri = _tri(*SMALL)
    with pytest.raises(NotImplementedError):
        ib.get_mu_phi_int(MU, PHI, ["N", "I"], ["N"], ["N"], cosmo_funcs, *tri, ZZ)
    with pytest.raises(NotImplementedError):
        ib.get_mu_phi_int(MU, PHI, ["N", "Loc", "L"], ["N"], ["N"], cosmo_funcs, *tri, ZZ)


# the fast path, get_multipoles_int_fast: angles exact at each node, the rest radial


def test_fast_local_limit(cosmo_funcs):
    """orders=(0,) is the local tree - the exact angular rule on a polynomial of degree ~8 + l"""
    kern, lm = ["N", "LP"], [(0, 0), (1, 0), (2, 0), (2, 2), (4, 0)]
    k1, k2, k3, _ = _tri(0.01, 0.012, 0.015)
    want = nbk.get_multipoles(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ)
    got = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, orders=(0,))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10 * np.max(np.abs(want)))


def _g(y):
    """Smooth in the node positions, ~y**2 at the observer so the rescalings' u integrals converge"""
    g = 1.0
    for j in ("a", "b", "ca", "cb"):
        g = g * y[j] ** 2 * (2 - y[j])
    return g


def _brute(term, los_leg, lm, cosmo_funcs, k1, k2, k3, n=500):
    """d^n int dy _g(y) int dOmega Re Y_lm e^{iW.x} = d^n int dy _g 4 pi i^l j_l(|W|) Re Y_lm(W), Gauss-Legendre in
    sqrt(y) for each node variable - the reference's mode solve and phase, nothing of the fast path's geometry. Past
    two variables it loops over the first"""
    d = cosmo_funcs.comoving_dist(ZZ)
    kvec = ib._kvec(np.asarray(k1), np.asarray(k2), np.asarray(utils.get_theta(k1, k2, k3)))
    names = ib._nodes(term, los_leg, False)
    t, w = utils.leggauss(n)
    t, w = (t + 1) / 2, w / 2
    lead = [(t[i] ** 2, 2 * t[i] * w[i] * d) for i in range(n)] if len(names) > 2 else [(None, 1.0)]
    out = 0
    for y0, w0 in lead:
        k = len(names) - (y0 is not None)
        ys = [tv**2 for tv in np.meshgrid(*([t] * k), indexing="ij")]
        wts = w0 * np.prod(np.meshgrid(*([2 * t * w * d] * k), indexing="ij"), axis=0)
        y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
        for name, yv in zip(names, ([y0] if y0 is not None else []) + ys):
            ib._set_y(y, name, yv)
        (r1, r2), coef, _ = ib.solve_modes(los_leg, term[0], y)
        R = ib._positions(term, y, d, los_leg, False)
        A, B = (sum(coef[j][m] * (R[j] - R["c" + j]) for j in "ab") for m in (0, 1))
        W = [A * kvec[r1][i] + B * kvec[r2][i] for i in (0, 1)]
        X = np.hypot(*W)
        mu, phi = W[0] / np.where(X > 0, X, 1) + (X == 0), np.where(W[1] >= 0, 0.0, np.pi)  # W = 0: only l = 0 is left
        out = out + np.array(
            [np.sum(_g(y) * wts * 4 * np.pi * 1j**l * spherical_jn(l, X) * ib._re_ylm(l, m, mu, phi)) for l, m in lm]
        )
    return out


@pytest.mark.parametrize("los_leg", [0, 2])
@pytest.mark.parametrize("shape", [(1, 1.2, 1.5), (1, 1, 1.8), (1, 1, 0.15)])
def test_fast_grids(cosmo_funcs, monkeypatch, shape, los_leg):
    """Every grid against _brute, the term swapped for _g: pins each node map, Jacobian and pole - the rays along a
    side, the parallelogram, the misaligned cone and strip - for both lines of sight, an obtuse triangle (stationary
    points on the strip's lines) and a squeezed one (short-side rescalings)"""
    kern, lm = ["N", "L"], [(0, 0), (1, 0), (2, 0), (2, 2)]
    fields = ib._fields((kern,) * 3)
    k1, k2, k3 = (4 / cosmo_funcs.comoving_dist(ZZ) * s for s in shape)  # kd ~ 4: _brute needs no Filon
    monkeypatch.setattr(ib, "_integrand", lambda *args: (_g(args[9]) + 0 * args[5][0], None))
    for term in list(ib._terms(fields, los_leg, (1,), (0, 1, 2), False)):
        monkeypatch.setattr(ib, "_terms", lambda *args, term=term: iter([term]))
        got = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, los_leg=los_leg)
        want = _brute(term, los_leg, lm, cosmo_funcs, k1, k2, k3)
        np.testing.assert_allclose(got, want, atol=2e-6 * np.max(np.abs(want)), err_msg=str(term))


@pytest.mark.parametrize("los_leg, shape", [(2, (1, 1.2, 1.5)), (0, (1, 1, 0.15))])
def test_fast_grids_orders23(cosmo_funcs, monkeypatch, los_leg, shape):
    """test_fast_grids for IIS and III: the polygons and their kinks, caps and corners, and the spectators (on their own
    nodes, split=False). _brute gets fewer nodes the more node variables there are, and its rescalings limit it to
    ~1e-5 at three and ~1e-4 at four"""
    kern, lm = ["N", "L"], [(0, 0), (1, 0), (2, 0), (2, 2)]
    fields = ib._fields((kern,) * 3)
    k1, k2, k3 = (4 / cosmo_funcs.comoving_dist(ZZ) * s for s in shape)
    monkeypatch.setattr(ib, "_integrand", lambda *args: (_g(args[9]) + 0 * args[5][0], None))
    for term in list(ib._terms(fields, los_leg, (2, 3), (0, 1, 2), False)):
        nv = len(ib._nodes(term, los_leg, False))
        monkeypatch.setattr(ib, "_terms", lambda *args, term=term: iter([term]))
        got = ib.get_multipoles_int_fast(
            kern,
            kern,
            kern,
            lm,
            cosmo_funcs,
            k1,
            k2,
            k3,
            zz=ZZ,
            los_leg=los_leg,
            n_s=64 if nv == 3 else 32,
            split=False,
        )
        want = _brute(term, los_leg, lm, cosmo_funcs, k1, k2, k3, n={2: 400, 3: 90, 4: 40}[nv])
        tol = {2: 3e-6, 3: 1e-4, 4: 2e-4}[nv]
        np.testing.assert_allclose(got, want, atol=tol * np.max(np.abs(want)), err_msg=str(term))


def test_fast_split(cosmo_funcs, monkeypatch):
    """_integrand_split against the whole integrand summed over the same spectator nodes (split=False), for every IIS
    and III term with a spectator - IntK1 and IntK2 split term by term, Q_POWERS, the powers of lambda in the Jacobian
    and det M. On a coarse grid, which both share"""
    kern, lm = ["N", "LP", "L"], [(0, 0), (1, 0), (2, 0)]
    fields = ib._fields((kern,) * 3)
    kw = dict(zz=ZZ, n_psi=4, n_d=8, n_f=8, n_s=16)
    terms = ib._terms
    for los_leg in (0, 2):
        for term in terms(fields, los_leg, (2, 3), (0, 1, 2), False):
            _, _, _, I, C = ib._sides(term, los_leg)
            if not any(I[j] and C[j] for j in "ab"):
                continue
            monkeypatch.setattr(ib, "_terms", lambda *args, term=term: iter([term]))
            got = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, *SMALL, los_leg=los_leg, **kw)
            want = ib.get_multipoles_int_fast(
                kern, kern, kern, lm, cosmo_funcs, *SMALL, los_leg=los_leg, split=False, **kw
            )
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12 * np.max(np.abs(want)), err_msg=str(term))


def test_fast_vs_reference(cosmo_funcs, monkeypatch):
    """Term by term against get_multipoles_int, where it is converged: its y integrals (the mixed layouts, the line of
    sight field's own) settle fast, to ~1e-4 at its defaults. Its rescalings do not - the near-observer spike leaves
    them ~15% off at its defaults even here, ~3% between finer grids, while the fast path holds to ~1e-5 under its
    own. They share _integrand with these, and test_fast_grids pins their geometry"""
    kern, lm = ["N", "LP", "L"], [(0, 0), (1, 0), (2, 0)]
    fields = ib._fields((kern,) * 3)
    k1, k2, k3, _ = _tri(*SMALL)
    terms = ib._terms
    for term in terms(fields, 2, (1,), (0, 1, 2), True):
        if term[5] == "two" or set(ib._nodes(term, 2, True)) & ib._det_vars(2, term[0]):
            continue
        monkeypatch.setattr(ib, "_terms", lambda *args, term=term: iter([term]))
        monkeypatch.setattr(ib_reference, "_terms", lambda *args, term=term: iter([term]))
        want = ib.get_multipoles_int(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, perms=(term[0],))
        # the reference's integrals on the line of sight field are the fast path's layouts
        fast = [t for t in terms(fields, 2, (1,), (term[0],), False) if t[3:5] == term[3:5]]
        fast = [t for t in fast if t[5] == term[5] or (term[5] == "I" and t[5] in ib.LAYOUTS)]
        monkeypatch.setattr(ib, "_terms", lambda *args, fast=fast: iter(fast))
        got = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, perms=(term[0],))
        np.testing.assert_allclose(got, want, atol=3e-4 * np.max(np.abs(want)), err_msg=str(term))


@pytest.mark.parametrize("c", [0, 1, 2])
def test_fast_relabel(cosmo_funcs, c):
    """As test_relabel - the monopole of every perm agrees across labellings, here to rounding as the grids are the
    same physical ones either way"""
    kern, lm = ["N", "L"], [(0, 0)]
    k1, k2, k3 = SMALL
    A = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, los_leg=2, perms=(c,))
    B = ib.get_multipoles_int_fast(
        kern, kern, kern, lm, cosmo_funcs, k3, k1, k2, zz=ZZ, los_leg=0, perms=((c + 1) % 3,)
    )
    np.testing.assert_allclose(A, B, rtol=1e-10)


@pytest.mark.parametrize("c", [0, 1, 2])
def test_fast_relabel_orders23(cosmo_funcs, c):
    """test_fast_relabel for IIS and III, on a coarse grid - the regions, kinks, caps, corners and spectators are the same
    physical ones in either labelling, though a and b can swap"""
    kern, lm = ["N", "L"], [(0, 0)]
    k1, k2, k3 = SMALL
    kw = dict(zz=ZZ, orders=(2, 3), n_psi=8, n_d=16, n_f=32, n_s=32)
    A = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, los_leg=2, perms=(c,), **kw)
    B = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k3, k1, k2, los_leg=0, perms=((c + 1) % 3,), **kw)
    np.testing.assert_allclose(A, B, rtol=1e-10)


@pytest.mark.parametrize("orders, kw", [((1,), dict(n_psi=16)), ((2, 3), dict(n_psi=8, n_d=16, n_f=32, n_s=32))])
def test_fast_batch(cosmo_funcs, orders, kw):
    """Arrays of triangles give what each does alone - with n_psi fixed, as the default follows the largest kd"""
    kern, lm = ["N", "L"], [(0, 0), (1, 0)]
    k1, k2, k3 = np.array([0.004, 0.01]), np.array([0.005, 0.011]), np.array([0.006, 0.019])
    got = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, k1, k2, k3, zz=ZZ, orders=orders, **kw)
    for i in range(2):
        want = ib.get_multipoles_int_fast(
            kern, kern, kern, lm, cosmo_funcs, k1[i], k2[i], k3[i], zz=ZZ, orders=orders, **kw
        )
        np.testing.assert_allclose(got[:, i], want, rtol=1e-12)


@pytest.mark.parametrize("orders", [(0,), (1,), (2,), (3,)])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("max_size", [256, 2**17])
def test_fast_chunks_match_whole(forecast_mt, orders, split, max_size):
    """Small chunks keep broadcasting, tracer placements, FoG and every integrated order."""
    kernels = (["N", "LP", "L"], ["N", "L"], ["N", "LP", "kappa_g"])
    lm = [(0, 0), (1, 0), (2, 2)]
    k1, k2 = np.array([[0.004], [0.011]]), np.array([[0.005, 0.009, 0.013]])
    theta = np.array([[1.0, 1.7, 2.1], [1.3, 1.9, 2.2]])
    kw = dict(zz=ZZ, theta=theta, sigma=6.0, orders=orders, los_leg=orders[0] % 3, split=split)
    kw.update(n_psi=3, n_d=6, n_f=8, n_s=16)
    args = (*kernels, lm, forecast_mt.cosmo_funcs, k1, k2)

    whole = ib.get_multipoles_int_fast(*args, max_size=2**30, **kw)
    chunks = ib.get_multipoles_int_fast(*args, max_size=max_size, **kw)
    assert chunks.shape == (len(lm), 2, 3)
    np.testing.assert_allclose(chunks, whole, rtol=1e-12, atol=1e-12 * np.max(np.abs(whole)))


def test_fast_chunks_keep_default_quadrature(cosmo_funcs, monkeypatch):
    """The largest triangle sets n_psi once, including for smaller triangles in their own chunks."""
    grids = ib._grids
    settings = []

    def record(*args, **kwargs):
        settings.append(tuple(args[8]))
        return grids(*args, **kwargs)

    monkeypatch.setattr(ib, "_grids", record)
    kern, lm = ["N", "L"], [(0, 0), (2, 0)]
    sides = (np.array([0.001, 0.025]), np.array([0.0012, 0.028]), np.array([0.0014, 0.03]))
    kw = dict(zz=ZZ, n_d=8, n_f=8)
    whole = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, *sides, max_size=2**30, **kw)
    chunks = ib.get_multipoles_int_fast(kern, kern, kern, lm, cosmo_funcs, *sides, max_size=1024, **kw)

    assert len(set(settings)) == 1
    assert max(settings[0]) > 16
    np.testing.assert_allclose(chunks, whole, rtol=1e-12, atol=1e-12 * np.max(np.abs(whole)))


def test_fast_chunks_before_grid_allocation(cosmo_funcs, monkeypatch):
    """A small budget must split triangles before any radial grid is allocated."""
    grids = ib._grids
    sizes = []

    def record(*args, **kwargs):
        sizes.append(np.size(args[3][0]) // 2)  # kvec has its two components on the first axis
        return grids(*args, **kwargs)

    monkeypatch.setattr(ib, "_grids", record)
    kern = ["N", "L"]
    k1, k2 = np.array([[0.004], [0.007]]), np.array([[0.005, 0.008]])
    got = ib.get_multipoles_int_fast(
        kern,
        kern,
        kern,
        [(0, 0)],
        cosmo_funcs,
        k1,
        k2,
        theta=1.5,
        zz=ZZ,
        n_psi=3,
        n_d=6,
        n_f=8,
        max_size=256,
    )
    assert got.shape == (1, 2, 2)
    assert sizes and max(sizes) == 1


def test_fast_streams_grid_pieces(cosmo_funcs, monkeypatch):
    """A region's next edge or strip is built after the preceding piece was projected."""
    events = []

    def track(func, event):
        def record(*args, **kwargs):
            events.append(event)
            return func(*args, **kwargs)

        return record

    monkeypatch.setattr(ib, "_terms", lambda *args: iter([ib._Term(2, 0, 1, "I", "I", "two")]))
    monkeypatch.setattr(ib_radial, "_grid_edge", track(ib_radial._grid_edge, "build"))
    monkeypatch.setattr(ib_radial, "_grid_lines", track(ib_radial._grid_lines, "build"))
    monkeypatch.setattr(ib, "_project", track(ib._project, "project"))
    ib.get_multipoles_int_fast(
        ["L"],
        ["L"],
        ["L"],
        [(0, 0)],
        cosmo_funcs,
        *SMALL,
        zz=ZZ,
        orders=(3,),
        n_psi=3,
        n_d=6,
        n_f=8,
        n_s=16,
    )
    assert events.count("build") > 1
    assert all(pair != ("build", "build") for pair in zip(events[:-1], events[1:]))


def test_fast_spectators_expand_in_projection(cosmo_funcs, monkeypatch):
    """The unsplit check expands spectators only after selecting a projection chunk."""
    project, spectate = ib._project, ib._spectate
    projecting = False
    expanded = []

    def in_project(*args, **kwargs):
        nonlocal projecting
        projecting = True
        try:
            return project(*args, **kwargs)
        finally:
            projecting = False

    def expand(y, jac, spec, n_s):
        assert projecting
        expanded.append(jac.shape)
        assert jac.shape[-2] == 1  # one ray at a time with this budget
        return spectate(y, jac, spec, n_s)

    monkeypatch.setattr(ib, "_terms", lambda *args: iter([ib._Term(2, 0, 1, "I", "I", "two")]))
    monkeypatch.setattr(ib, "_project", in_project)
    monkeypatch.setattr(ib, "_spectate", expand)
    ib.get_multipoles_int_fast(
        ["L"],
        ["L"],
        ["L"],
        [(0, 0)],
        cosmo_funcs,
        *SMALL,
        zz=ZZ,
        orders=(3,),
        split=False,
        n_psi=3,
        n_d=6,
        n_f=8,
        n_s=16,
        max_size=256,
    )
    assert expanded
