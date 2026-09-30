"""cosmo_funcs.full_betas - the (H/k)^3 and (H/k)^4 GR terms of 2011.13660 in the numeric bispectrum.

beta1-beta5 of (3.24) go into K2.LP, and Upsilon3-Upsilon5 of (3.26) (b_01 phi_p x GR projection) into a
field's PNG Z2 when that field also has LP. Upsilon1, Upsilon2 are the full Z1 on the primordial bispectrum,
which numeric_mu.bk has already - see lib.betas.interpolate_upsilon.

Cross-checks against the paper's Tables 1-3 (TERM kernel x COEFFICIENT): beta3 and Upsilon3-5 rebuild from
them exactly (Upsilon3 with the -R of Table 3, not the +R of (B.3)); beta4, beta5 need the factor 2f the
V_GR^(2) rows also miss against beta9, beta10. beta1, beta2 do not rebuild from the Phi/Psi/V_GR^(2) rows
as printed, so they are taken from (A.1), (A.2) as the other betas are and only checked for their order.
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.lib.betas import dy_dz, survey_terms
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu.kernels import K2

ZZ = 1.2
MU = np.array([-0.83, -0.2, 0.31, 0.77])
PHI = np.array([0.3, 1.9, 4.4])
TRI = (np.array([0.004, 0.02]), np.array([0.006, 0.03]), np.array([2.0, 2.6]))  # (k1, k2, theta) - large scales

BETA_ORDER = [1, 2] + [2] * 8 + [1] * 6  # gr1, gr2 | beta6-13 | beta14-19 - as test_numeric_mu_bk
HO_ORDER = [4, 4, 3, 3, 3]  # beta1, beta2 | beta3-5


@pytest.fixture
def cf_full(cosmo_funcs, monkeypatch):
    monkeypatch.setattr(cosmo_funcs, "full_betas", True)
    return cosmo_funcs


@pytest.fixture(scope="module")
def cosmo_funcs_png(cosmo):
    """compute_bias for the eq/orth scale-dependent biases - see ClassWAP.get_PNG_bias."""
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), compute_bias=True, verbose=False)


def _B(cf, kernels, **kwargs):
    k1, k2, theta = TRI
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))
    k1, k2, k3, theta = (x[:, None, None] for x in (k1, k2, k3, theta))
    return nbk.get_mu_phi(MU, PHI, kernels, kernels, kernels, cf, k1, k2, k3, theta, ZZ, **kwargs)


def _node_terms(cf, ti=0):
    """Survey and cosmology terms at the survey grid node nearest ZZ, primes over H - as the paper prints them"""
    zz, Q, b_e, b_1, dQL, db1L, dlnLc, dQ_dt, dbe_dt, _, cosmo = survey_terms(cf, ti)
    H, f, Om, xi, dH_dt, dH_dt2 = cosmo
    i = np.argmin(abs(zz - ZZ))
    fp = -(1 + zz) * dy_dz(f, zz)
    terms = dict(
        z=zz[i], H=H[i], f=f[i], Om=Om[i], Q=Q[i], be=b_e[i], b1=b_1[i], x=1 / (xi[i] * H[i]),
        dH=dH_dt[i] / H[i] ** 2, ddH=dH_dt2[i] / H[i] ** 3, fp=fp[i], Qp=dQ_dt[i] / H[i], bep=dbe_dt[i] / H[i],
        dQL=dQL[i], db1L=db1L[i], dlnLc=dlnLc[i],
    )  # fmt: skip
    terms["R"] = 2 * (1 - terms["Q"]) * terms["x"] + terms["dH"]  # (A.28)
    return terms


# ---------------------------------------------------------------------------
# default path
# ---------------------------------------------------------------------------


def test_off_builds_nothing(cosmo_funcs):
    """Flag off: neither the beta1-5 stack nor any Upsilon is built"""
    cf = utils.copy(cosmo_funcs)
    for s in set(cf.survey):
        s.reset_cache()
    assert cf.full_betas is False
    _B(cf, ["N", "LP", "Loc"])
    assert cf.survey[0].betas_HO is None
    assert not any(key.startswith("ups_") for key in cf.survey[0].deriv)


def test_reset_cache_clears(cf_full):
    cf = utils.copy(cf_full)
    cf.get_beta_HO(ZZ)
    cf.get_upsilon(ZZ, 0, "Loc")
    tracer = cf.survey[0]
    assert tracer.betas_HO is not None and "ups_Loc" in tracer.deriv
    tracer.reset_cache()
    assert tracer.betas_HO is None and "ups_Loc" not in tracer.deriv


# ---------------------------------------------------------------------------
# beta1-beta5
# ---------------------------------------------------------------------------


def test_beta4_beta5_are_gamma1(cf_full):
    """(A.4), (A.5): beta4 = -9/2 Om H^2 gamma1, beta5 = 3 Om H^2 gamma1 - as beta9, beta10 against f H^2"""
    zz = cf_full.survey[0].z_survey
    gr1 = cf_full.get_beta_funcs(zz)[0]
    _, _, _, beta4, beta5 = cf_full.get_beta_HO(zz)
    Om, H = cf_full.Om_m(zz), cf_full.H_c(zz)
    np.testing.assert_allclose(beta4, -4.5 * Om * H**2 * gr1, rtol=1e-6)
    np.testing.assert_allclose(beta5, 3 * Om * H**2 * gr1, rtol=1e-6)


def test_beta3_from_table1(cf_full):
    """beta3 as the sum of its Table 1 rows (V d V, Psi d V, Psi d Phi) with A3, A4 of (A.25), (A.26)"""
    t = _node_terms(cf_full)
    f, Om, Q, be, x, dH, ddH = t["f"], t["Om"], t["Q"], t["be"], t["x"], t["dH"], t["ddH"]
    fp, Qp, bep, dQL = t["fp"], t["Qp"], t["bep"], t["dQL"]

    A3 = 2 * (
        -3 + 4 * be + 2 * be * (1 - Q) * x - be**2 + 2 * be * Q - 6 * Q - bep - 6 * (1 - Q) * x + 2 * (1 - x) * Qp
    )
    A4 = (
        4 + 2*f*(-3 + f + 2*be - 3*Q - 4*(1 - Q)*x - 2*dH) + 2*fp - 6*be - 2*be**2 + 12*be*Q - 8*Q
        - 16*Q**2 + 16*dQL + 12*Qp - 2*bep - 4*x**2*(1 - Q + 2*Q**2 - 2*dQL)
        - 4*x*(-1 - 2*be + 2*be*Q + Q - 6*Q**2 + 3*dH*(1 - Q) + 6*dQL + 2*Qp)
        + 2*dH*(3 + 2*be - 6*Q - 3*dH) + 2*ddH
    )  # fmt: skip
    beta3 = t["H"] ** 3 * (f**2 / 2 * A3 - 3 * f * Om / 4 * A4 + 9 * Om**2 / 8 * 2 * (f - 2 + 2 * Q))

    np.testing.assert_allclose(cf_full.get_beta_HO(t["z"])[2], beta3, rtol=1e-8)


def test_LP_adds_beta_HO_term(cosmo_funcs, cf_full):
    """K2.LP with the flag is K2.LP without plus (3.24)'s beta1-5 line over 2 q1^2 q2^2 - E2 of (3.25)"""
    q1, q2, cos12 = 0.004, 0.007, -0.3
    mu1, mu2 = MU[:, None], MU[None, :]
    on = K2.LP(cf_full, ZZ, mu1, mu2, q1, q2, cos12)
    cf_full.full_betas = False
    off = K2.LP(cf_full, ZZ, mu1, mu2, q1, q2, cos12)

    beta1, beta2, beta3, beta4, beta5 = cf_full.get_beta_HO(ZZ)
    E2 = (q1 * q2) ** 2 / (q1**2 + q2**2 + 2 * q1 * q2 * cos12) ** 2 * (3 + 2 * (q1 / q2 + q2 / q1) * cos12 + cos12**2)
    p3 = q1 * mu1 + q2 * mu2
    new = cf_full.D(ZZ) ** 2 / (2 * q1**2 * q2**2) * (beta1 + E2 * beta2 + 1j * p3 * (beta3 + beta4 + E2 * beta5))
    np.testing.assert_allclose(on - off, new, rtol=1e-10, atol=1e-12 * np.max(np.abs(off)))


def test_beta_HO_orders(cosmo_funcs, monkeypatch):
    """Every beta scaled by eps^(its H order): B with the flag minus B without starts at eps^3, and its
    eps^3 part is odd (imaginary coefficients) and eps^4 part has a real piece - beta1-5 are the only change"""
    cls = type(cosmo_funcs)
    orig, orig_HO = cls.get_beta_funcs, cls.get_beta_HO
    scale = {"eps": 1.0}
    eps_grid = np.linspace(-1.5, 1.5, 9)  # Z1 Z1 Z2 is degree 8 in eps

    def diff():
        vals = []
        for e in eps_grid:
            scale["eps"] = e
            cosmo_funcs.full_betas = True
            on = _B(cosmo_funcs, ["N", "LP"])
            cosmo_funcs.full_betas = False
            vals.append(on - _B(cosmo_funcs, ["N", "LP"]))
        vals = np.array(vals)
        coef = np.linalg.solve(np.vander(eps_grid, len(eps_grid), increasing=True), vals.reshape(len(eps_grid), -1))
        return coef.reshape(vals.shape)

    with monkeypatch.context() as patch:
        patch.setattr(
            cls,
            "get_beta_funcs",
            lambda self, zz, ti=0: [b * scale["eps"] ** o for b, o in zip(orig(self, zz, ti=ti), BETA_ORDER)],
        )
        patch.setattr(
            cls,
            "get_beta_HO",
            lambda self, zz, ti=0: [b * scale["eps"] ** o for b, o in zip(orig_HO(self, zz, ti=ti), HO_ORDER)],
        )
        coef = diff()
    cosmo_funcs.full_betas = False

    size = np.max(np.abs(coef))
    np.testing.assert_allclose(coef[:3], 0, atol=1e-9 * size)
    assert np.max(np.abs(coef[3])) > 1e-3 * size
    np.testing.assert_allclose(coef[3].real, 0, atol=1e-9 * size)  # N N x i(beta3-5)
    assert np.max(np.abs(coef[4].real)) > 1e-3 * size  # N N x (beta1, beta2) among others


# ---------------------------------------------------------------------------
# Upsilons
# ---------------------------------------------------------------------------


def test_upsilon1_is_twice_gamma2(cf_full):
    """(B.1) as printed is 2 fNL gamma2 for LCDM growth - to the growth equation's residual in CLASS - so the
    Upsilon1 term is Z1_GR(k3) on the primordial bispectrum, as Upsilon2 = 2 fNL gamma1 is by (B.2)"""
    zz, Q, b_e, *_, cosmo = survey_terms(cf_full, 0)
    H, f, Om, xi, dH_dt, _ = cosmo
    R = 2 * (1 - Q) / (xi * H) + dH_dt / H**2
    fp = -(1 + zz) * dy_dz(f, zz)
    U1 = H**2 * (2 * (3 - b_e) * f + 3 * Om * (1 + b_e - 4 * Q - R)
                 + 6 * Om / (3 * Om + 2 * f) * (fp + (1 + 2 * dH_dt / H**2) * f))  # fmt: skip
    gr2 = cf_full.get_beta_funcs(zz)[1]
    np.testing.assert_allclose(U1, 2 * gr2, rtol=2e-3)


def test_upsilon_from_table3(cf_full):
    """Upsilon3-5 as the Table 3 rows (Psi phi_p, V phi_p | Psi d phi_p | phi_p d V) - universality, b01 = 2 delta_c (b1 - 1)"""
    tracer = cf_full.survey[0]
    assert tracer.p == 1  # Table 3 writes d ln b01 = d b10/(b10 - 1)
    t = _node_terms(cf_full)
    H, f, Om, Q, be, x, R = t["H"], t["f"], t["Om"], t["Q"], t["be"], t["x"], t["R"]
    b10 = t["b1"]
    S = 4 * (2 - x) * t["db1L"]  # (A.29)
    zz = tracer.z_survey
    db1_lna = -(1 + zz) * (dy_dz(tracer.b_1(zz), zz) - tracer.db1_dlnL(zz) * tracer.dlnLc_dz(zz))  # b10'/H at fixed L
    b10p = db1_lna[np.argmin(abs(zz - ZZ))] * H
    b01 = tracer.loc.b_01(t["z"])

    U3 = (-3 * Om * H**2 / 4 * b01 * (8 * Q + 2 * R - 2 * be - 4 - S / (b10 - 1))
          + f * H / 2 * b01 * (2 * (3 - be - f) * H + b10p / (b10 - 1)))  # fmt: skip
    U4 = -3 * Om * H**2 / 4 * 2 * b01 / H
    U5 = f * H / 2 * b01 * (2 * f + 2 * be - 4 * Q - 2 * R + S / (b10 - 1))

    np.testing.assert_allclose(cf_full.get_upsilon(t["z"], 0, "Loc"), [U3, U4, U5], rtol=1e-8)


def test_upsilon_needs_LP(cosmo_funcs, cf_full):
    """Without LP on the field the flag leaves the PNG part alone; with it the change is linear in fNL"""
    on = _B(cf_full, ["N", "Loc"])
    cf_full.full_betas = False
    np.testing.assert_array_equal(on, _B(cf_full, ["N", "Loc"]))

    def png_change(fNL):
        cf_full.full_betas = True
        on = _B(cf_full, ["N", "LP", "Loc"], fNL=fNL) - _B(cf_full, ["N", "LP"])
        cf_full.full_betas = False
        off = _B(cf_full, ["N", "LP", "Loc"], fNL=fNL) - _B(cf_full, ["N", "LP"])
        return on - off

    one, three = png_change(1), png_change(3)
    assert np.max(np.abs(one)) > 0
    np.testing.assert_allclose(three, 3 * one, rtol=1e-8, atol=1e-10 * np.max(np.abs(three)))


@pytest.mark.parametrize("shape", ["Eq", "Orth"])
def test_upsilon_nonlocal_shapes(cosmo_funcs_png, monkeypatch, shape):
    """Eq/Orth take the same couplings with their own b_01 and k^alpha (and no db_01_dlnL on the HMF holders)"""
    cf = cosmo_funcs_png
    monkeypatch.setattr(cf, "full_betas", True)
    U3, U4, U5 = cf.get_upsilon(ZZ, 0, shape)
    b01, _ = cf.get_PNG_bias(ZZ, 0, shape)
    np.testing.assert_allclose(U4, -1.5 * cf.Om_m(ZZ) * cf.H_c(ZZ) * b01, rtol=1e-6)
    assert np.all(np.isfinite([U3, U5]))

    on = _B(cf, ["N", "LP", shape])
    monkeypatch.setattr(cf, "full_betas", False)
    off = _B(cf, ["N", "LP", shape])
    assert np.all(np.isfinite(on)) and np.max(np.abs(on - off)) > 0
