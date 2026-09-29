"""The numeric-mu bispectrum against the analytic expressions.

The kernels keep every order in (H/k) while the analytic GR1/GR2 stop at (H/k)**2. To compare them,
scale gr1 and beta14-19 (~H) by eps and gr2 and beta6-13 (~H**2) by eps**2: the product
Z1*Z1*Z2 is then a degree 6 polynomial in eps, and solving for its coefficients exactly gives
the Newtonian (eps**0), GR1 (eps**1) and GR2 (eps**2) parts separately.

MathWAP evaluates Z2 at (k1, k2) where the bispectrum needs (-k1, -k2) - see numeric_mu.bk.get_mu_phi -
which flips the beta14-beta18 terms, the ones odd in the pair that sources Z2. The analytic GR1/GR2 (bk and
bk_mt) negate those betas after unpacking them, so they are compared as they are.
"""

import numpy as np
import pytest

import cosmo_wap as cw
import cosmo_wap.bk as bk
import cosmo_wap.bk_mt as bk_mt
import cosmo_wap.pk as pk
from cosmo_wap.lib.angular_integrate import legendre, ylm
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import pk as npk

ZZ = 1.2

# Triangle configurations: (k1, k2, theta)
TRIANGLES = {
    "equilateral": (0.05, 0.05, 2 * np.pi / 3),
    "squeezed": (0.1, 0.1, 3.0),
    "generic": (0.02, 0.035, 2.0),
}

BETA_ORDER = [1, 2] + [2] * 8 + [1] * 6  # gr1, gr2 | beta6-13 | beta14-19
EPS = np.linspace(-1.5, 1.5, 7)

MU = np.array([-0.83, -0.2, 0.31, 0.77])
PHI = np.array([0.3, 1.9, 4.4])


def _triangle(name):
    k1, k2, theta = (np.array([x]) for x in TRIANGLES[name])
    return k1, k2, theta


def _eps_orders(cf, monkeypatch, func):
    """Coefficients of eps**n in func() once the betas carry their (H/k) order - see module docstring."""
    orig = type(cf).get_beta_funcs
    scale = {"eps": 1.0}

    def scaled(self, zz, ti=0):
        return [b * scale["eps"] ** o for b, o in zip(orig(self, zz, ti=ti), BETA_ORDER)]

    vals = []
    # Patching the class avoids leaving a bound method in the session fixture's __dict__:
    # copies used for bias derivatives must bind get_beta_funcs to the shifted object.
    with monkeypatch.context() as patch:
        patch.setattr(type(cf), "get_beta_funcs", scaled)
        for e in EPS:
            scale["eps"] = e
            vals.append(func())

    vals = np.array(vals)
    coef = np.linalg.solve(np.vander(EPS, len(EPS), increasing=True), vals.reshape(len(EPS), -1))
    return coef.reshape(vals.shape)


def _B(cf, kernels, k1, k2, theta):
    """Numeric B on the (MU, PHI) test grid, triangle axis first."""
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))
    k1, k2, k3, theta = (x[:, None, None] for x in (k1, k2, k3, theta))
    return nbk.get_mu_phi(MU, PHI, kernels, kernels, kernels, cf, k1, k2, k3, theta, ZZ)


def _analytic_B(func, cf, k1, k2, theta):
    return func(MU[:, None], PHI[None, :], cf, k1[:, None, None], k2[:, None, None], theta=theta[:, None, None], zz=ZZ)


# ---------------------------------------------------------------------------
# B(mu, phi) point-wise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tri", TRIANGLES)
def test_newtonian_matches_Bk_0(cosmo_funcs, tri):
    k1, k2, theta = _triangle(tri)
    np.testing.assert_allclose(
        _B(cosmo_funcs, ["N"], k1, k2, theta), _analytic_B(bk.Bk_0, cosmo_funcs, k1, k2, theta), rtol=1e-12
    )


@pytest.mark.parametrize("tri", TRIANGLES)
def test_local_GR_matches_GR_1_and_GR_2(cosmo_funcs, monkeypatch, tri):
    """eps**1 is GR_1 and eps**2 is GR_2 - term by term in the betas, so also a check of each K2.LP term."""
    k1, k2, theta = _triangle(tri)
    coef = _eps_orders(cosmo_funcs, monkeypatch, lambda: _B(cosmo_funcs, ["N", "LP"], k1, k2, theta))

    for order, func in enumerate([bk.Bk_0, bk.GR_1, bk.GR_2]):
        ref = _analytic_B(func, cosmo_funcs, k1, k2, theta)
        np.testing.assert_allclose(coef[order], ref, rtol=0, atol=1e-10 * np.max(np.abs(ref)))


def test_mirrored_grid_matches_full_grid(cosmo_funcs):
    k1, k2, theta = _triangle("generic")
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))
    args = (["N", "LP"],) * 3 + (cosmo_funcs,) + tuple(x[:, None, None] for x in (k1, k2, k3, theta)) + (ZZ,)
    for n_mu in (8, 9):  # with and without a node at mu=0
        mu, phi, _ = nbk.get_mu_phi_grid(n_mu, 8)
        np.testing.assert_allclose(nbk.get_mu_phi_sym(mu, phi, *args), nbk.get_mu_phi(mu, phi, *args), rtol=1e-13)


# ---------------------------------------------------------------------------
# multipoles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tri", TRIANGLES)
def test_newtonian_multipoles(cosmo_funcs, tri):
    k1, k2, theta = _triangle(tri)
    num = nbk.get_multipoles(["N"], ["N"], ["N"], [(0, 0), (2, 0), (1, 0)], cosmo_funcs, k1, k2, theta=theta, zz=ZZ)
    for (l, _), val in zip([(0, 0), (2, 0)], num):
        ref = getattr(bk.NPP, f"l{l}")(cosmo_funcs, k1, k2, theta=theta, zz=ZZ)
        np.testing.assert_allclose(val, ref, rtol=1e-10)
    np.testing.assert_allclose(num[2], 0, atol=1e-10 * np.abs(num[0]).max())  # no odd multipoles


def _assert_multipoles(coef, orders, refs):
    """Multipoles that vanish for a triangle (e.g. l1 for equilateral) are left as cancellation noise by
    any quadrature, so the tolerance is set by the largest multipole at the same eps order."""
    for i, (order, ref) in enumerate(zip(orders, refs)):
        scale = max(np.abs(r).max() for o, r in zip(orders, refs) if o == order)
        np.testing.assert_allclose(coef[order][i], ref, rtol=1e-8, atol=1e-10 * scale)


GR_LM = [(1, 0, bk.GR1, "l1"), (3, 0, bk.GR1, "l3"), (1, 1, bk.GR1, "l1m1"), (3, 2, bk.GR1, "l3m2")]
GR_LM += [(0, 0, bk.GR2, "l0"), (2, 0, bk.GR2, "l2")]


@pytest.mark.parametrize("tri", TRIANGLES)
@pytest.mark.parametrize("sigma", [None, 8.0])
def test_local_GR_multipoles(cosmo_funcs, monkeypatch, tri, sigma):
    k1, k2, theta = _triangle(tri)
    lm = [(l, m) for l, m, _, _ in GR_LM]
    coef = _eps_orders(
        cosmo_funcs,
        monkeypatch,
        lambda: nbk.get_multipoles(
            ["N", "LP"], ["N", "LP"], ["N", "LP"], lm, cosmo_funcs, k1, k2, theta=theta, zz=ZZ, sigma=sigma
        ),
    )
    orders = [1 if cls is bk.GR1 else 2 for _, _, cls, _ in GR_LM]
    refs = [getattr(cls, meth)(cosmo_funcs, k1, k2, theta=theta, zz=ZZ) for _, _, cls, meth in GR_LM]
    if sigma is not None:
        refs = [
            ylm(bk.GR_1 if cls is bk.GR1 else bk.GR_2, l, m, cosmo_funcs, k1, k2, theta=theta, zz=ZZ, sigma=sigma, n=64)
            for l, m, cls, _ in GR_LM
        ]
    _assert_multipoles(coef, orders, refs)


@pytest.mark.parametrize("sigma", [None, 8.0])
def test_pk_LP_leading_orders_match_GR(cosmo_funcs_mt, monkeypatch, sigma):
    kk = np.geomspace(0.01, 0.15, 8)
    ln = [0, 1, 2, 3]
    cf = cosmo_funcs_mt
    coef = _eps_orders(
        cf, monkeypatch, lambda: npk.get_multipoles(["N", "LP"], ["N", "LP"], ln, cf, kk, ZZ, sigma=sigma)
    )
    # An independent high-order quadrature of the analytic mu expressions, also stable at small k*sigma.
    for order, cls in [(1, pk.GR1), (2, pk.GR2)]:
        refs = np.array([legendre(cls.mu, l, cf, kk, ZZ, sigma=sigma, n_mu=64) for l in ln])
        np.testing.assert_allclose(coef[order], refs, rtol=1e-9, atol=1e-10 * np.max(np.abs(refs)))


@pytest.fixture(scope="module")
def cosmo_funcs_mt(cosmo):
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(6e-16), verbose=False)


def test_multi_tracer_matches_bk_mt(cosmo_funcs_mt, monkeypatch):
    """Per-field tracers as bk_mt: field i at k_i carries survey[i-1]."""
    cf = cosmo_funcs_mt
    k1, k2, theta = _triangle("generic")
    MT = [(0, bk_mt.NPP, "l0", 0), (2, bk_mt.NPP, "l2", 0), (1, bk_mt.GR1, "l1", 1), (3, bk_mt.GR1, "l3", 1)]
    MT += [(0, bk_mt.GR2, "l0", 2), (2, bk_mt.GR2, "l2", 2)]
    lm = [(l, 0) for l, _, _, _ in MT]
    coef = _eps_orders(
        cf,
        monkeypatch,
        lambda: nbk.get_multipoles(["N", "LP"], ["N", "LP"], ["N", "LP"], lm, cf, k1, k2, theta=theta, zz=ZZ),
    )
    refs = [getattr(cls, meth)(cf, k1, k2, theta=theta, zz=ZZ) for _, cls, meth, _ in MT]
    _assert_multipoles(coef, [order for _, _, _, order in MT], refs)


# ---------------------------------------------------------------------------
# bk_func and FoG
# ---------------------------------------------------------------------------


def test_bk_func_kernels(cosmo_funcs):
    k1, k2, theta = _triangle("generic")
    args = (0, cosmo_funcs, k1, k2)
    numeric = bk.bk_func(None, *args, theta=theta, zz=ZZ, kernels=["N"])
    np.testing.assert_allclose(numeric, bk.NPP.l0(cosmo_funcs, k1, k2, theta=theta, zz=ZZ), rtol=1e-10)

    both = bk.bk_func("NPP", *args, theta=theta, zz=ZZ, kernels=["N"])  # analytic + numeric
    np.testing.assert_allclose(both, 2 * numeric, rtol=1e-10)


def test_bk_func_multipole_list(cosmo_funcs):
    """A list of l gives what one call per l does - analytic and numeric parts alike."""
    k1, k2, theta = _triangle("generic")
    ln, terms, kw = [0, 1, 2, 3], ["NPP", "GR1", "GR2"], dict(theta=theta, zz=ZZ, kernels=["N", "LP"])
    vals = bk.bk_func(terms, ln, cosmo_funcs, k1, k2, **kw)
    for l, val in zip(ln, vals):
        np.testing.assert_allclose(val, bk.bk_func(terms, l, cosmo_funcs, k1, k2, **kw), rtol=1e-12)


def test_fog_matches_ylm(cosmo_funcs):
    """Damping inside the angular integral, as the analytic ylm route."""
    k1, k2, theta = _triangle("generic")
    for l in (0, 2):
        num = nbk.get_multipoles(["N"], ["N"], ["N"], [(l, 0)], cosmo_funcs, k1, k2, theta=theta, zz=ZZ, sigma=8.0)
        ref = bk.NPP.ylm(l, 0, cosmo_funcs, k1, k2, theta=theta, zz=ZZ, sigma=8.0)
        np.testing.assert_allclose(num[0], ref, rtol=1e-6)


def test_kernel_without_second_order_raises(cosmo_funcs):
    k1, k2, theta = _triangle("generic")
    with pytest.raises(NotImplementedError, match="second order"):
        nbk.get_multipoles(["I"], ["I"], ["I"], [(0, 0)], cosmo_funcs, k1, k2, theta=theta, zz=ZZ)


# ---------------------------------------------------------------------------
# PNG: scale-dependent bias at first and second order + the primordial bispectrum, linear in fNL
# ---------------------------------------------------------------------------

SHAPES = ["Loc", "Eq", "Orth"]


@pytest.fixture(scope="module")
def cosmo_funcs_png(cosmo):
    """compute_bias for the eq/orth scale-dependent biases - see ClassWAP.get_PNG_bias."""
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), compute_bias=True, verbose=False)


def _png_B(cf, kern, k1, k2, theta, **kwargs):
    """PNG part of the numeric B: with the kernel minus without (no fNL^0 part survives)"""
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))
    args = (cf, *(x[:, None, None] for x in (k1, k2, k3, theta)), ZZ)
    return nbk.get_mu_phi(MU, PHI, ["N", kern], ["N", kern], ["N", kern], *args, **kwargs) - nbk.get_mu_phi(
        MU, PHI, ["N"], ["N"], ["N"], *args
    )


def _png_multipoles(cf, kern, lm, k1, k2, theta, **kwargs):
    kerns = (["N", kern],) * 3
    return nbk.get_multipoles(*kerns, lm, cf, k1, k2, theta=theta, zz=ZZ, **kwargs) - nbk.get_multipoles(
        ["N"], ["N"], ["N"], lm, cf, k1, k2, theta=theta, zz=ZZ
    )


@pytest.mark.parametrize("tri", TRIANGLES)
@pytest.mark.parametrize("shape", SHAPES)
def test_png_matches_analytic_mu_phi(cosmo_funcs_png, shape, tri):
    k1, k2, theta = _triangle(tri)
    ref = _analytic_B(getattr(bk, f"{shape}_"), cosmo_funcs_png, k1, k2, theta)
    np.testing.assert_allclose(
        _png_B(cosmo_funcs_png, shape, k1, k2, theta), ref, rtol=0, atol=1e-10 * np.abs(ref).max()
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_png_multipoles(cosmo_funcs_png, shape):
    k1, k2, theta = _triangle("generic")
    num = _png_multipoles(cosmo_funcs_png, shape, [(0, 0), (2, 0)], k1, k2, theta)
    for i, l in enumerate((0, 2)):
        ref = getattr(getattr(bk, shape), f"l{l}")(cosmo_funcs_png, k1, k2, theta=theta, zz=ZZ)
        np.testing.assert_allclose(num[i], ref, rtol=1e-10)


def test_png_is_all_three_shapes(cosmo_funcs_png):
    """PNG brings every shape's kernels and primordial bispectrum, each with its own fNL."""
    k1, k2, theta = _triangle("generic")
    fNLs = {"fNL_loc": 1.5, "fNL_eq": -20.0, "fNL_orth": 7.0}
    total = _png_B(cosmo_funcs_png, "PNG", k1, k2, theta, **fNLs)
    parts = sum(_png_B(cosmo_funcs_png, shape, k1, k2, theta, **fNLs) for shape in SHAPES)
    np.testing.assert_allclose(total, parts, rtol=1e-12)


def test_png_linear_in_fNL(cosmo_funcs_png):
    k1, k2, theta = _triangle("squeezed")
    one = _png_B(cosmo_funcs_png, "Loc", k1, k2, theta)
    np.testing.assert_allclose(_png_B(cosmo_funcs_png, "Loc", k1, k2, theta, fNL=-4.0), -4 * one, rtol=1e-12)
    np.testing.assert_allclose(_png_B(cosmo_funcs_png, "Loc", k1, k2, theta, fNL_loc=3.0), 3 * one, rtol=1e-12)


def test_png_multi_tracer_matches_bk_mt(cosmo_funcs_mt):
    """bk_mt.Loc keeps small fNL^2 and fNL^3 terms (single-tracer bk.Loc does not): compare with its part
    linear in fNL, which the stencil over fNL = +-1, +-2 isolates up to fNL^4."""
    k1, k2, theta = _triangle("generic")
    num = _png_multipoles(cosmo_funcs_mt, "Loc", [(0, 0), (2, 0)], k1, k2, theta)
    for i, l in enumerate((0, 2)):
        B = {F: getattr(bk_mt.Loc, f"l{l}")(cosmo_funcs_mt, k1, k2, theta=theta, zz=ZZ, fNL=F) for F in (-2, -1, 1, 2)}
        ref = (8 * (B[1] - B[-1]) - (B[2] - B[-2])) / 12
        np.testing.assert_allclose(num[i], ref, rtol=1e-10)
