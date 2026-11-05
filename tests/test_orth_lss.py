"""'Orth' is the orthogonal-LSS template (cosmo_wap.lib.orth_lss): template, biases, and the analytic Orth terms
built from Eq and Loc. Single-tracer analytic vs numeric-mu is in test_numeric_mu_bk (SHAPES); the multi-tracer
check is here, since its fixture there has no Eq/Orth biases."""

import gc

import numpy as np
import pytest

import cosmo_wap as cw
import cosmo_wap.bk as bk
import cosmo_wap.bk_mt as bk_mt
import cosmo_wap.pk as pk
from cosmo_wap.bk.table import TABLE_MODULES
from cosmo_wap.lib import orth_lss, utils
from cosmo_wap.numeric_mu import bk as nbk

ZZ = 1.0
K1, K2, THETA = np.array([0.02, 0.05, 0.1]), np.array([0.035, 0.05, 0.1]), np.array([2.0, 2 * np.pi / 3, 3.0])


def _linear(func):
    """d/dt func(t) at t = 0 for a polynomial of degree <= 4 in t - its part linear in fNL"""
    return (8 * (func(1) - func(-1)) - (func(2) - func(-2))) / 12


def _standard(k1, k2, k3):
    """the standard orthogonal template, B_Phi/fNL, for P_Phi = k^-3"""
    P = [k**-3.0 for k in (k1, k2, k3)]
    pairs = P[0] * P[1] + P[0] * P[2] + P[1] * P[2]
    cube = (P[0] * P[1] * P[2]) ** (2 / 3)
    mixed = sum(P[a] ** (1 / 3) * P[b] ** (2 / 3) * P[3 - a - b] for a in range(3) for b in range(3) if a != b)
    return 6 * (-3 * pairs - 8 * cube + 3 * mixed)


def _lss(k1, k2, k3):
    return orth_lss.B_phi(*(k**-3.0 for k in (k1, k2, k3)), k1, k2, k3)


def test_template_limits():
    """Equal to the standard template at equilateral triangles; squeezed (k1 = k2 = 1, k3 = eps), B -> 4 A eps^(alpha-3):
    alpha = 2 with A = 3 - 5Q/9 here, where the standard template has alpha = 1, A = -3"""
    one = np.array(1.0)
    np.testing.assert_allclose(_lss(one, one, one), _standard(one, one, one), rtol=1e-12)
    eps = 1e-4
    np.testing.assert_allclose(_lss(one, one, eps) * eps / 4, orth_lss.A, rtol=1e-3)
    np.testing.assert_allclose(_standard(one, one, eps) * eps**2 / 4, -3, rtol=1e-3)
    np.testing.assert_allclose(orth_lss.A, -1.737, atol=1e-3)


def test_orthogonal_to_equilateral():
    """Q is what makes it orthogonal to the equilateral template (uniform weight over triangles, scale invariant)"""
    k = np.linspace(0.02, 1.0, 120)
    k1, k2, k3 = np.meshgrid(k, k, k, indexing="ij")
    m = (k1 <= k2 + k3) & (k2 <= k1 + k3) & (k3 <= k1 + k2)
    k1, k2, k3 = k1[m], k2[m], k3[m]
    P = [x**-3.0 for x in (k1, k2, k3)]
    pairs = P[0] * P[1] + P[0] * P[2] + P[1] * P[2]
    cube = (P[0] * P[1] * P[2]) ** (2 / 3)
    mixed = sum(P[a] ** (1 / 3) * P[b] ** (2 / 3) * P[3 - a - b] for a in range(3) for b in range(3) if a != b)
    S = (k1 * k2 * k3) ** 2  # shape functions, B * (k1 k2 k3)^2
    eq, lss = S * (-pairs - 2 * cube + mixed), S * _lss(k1, k2, k3) / 6
    assert abs(np.sum(eq * lss)) < 0.01 * np.sqrt(np.sum(eq**2) * np.sum(lss**2))


def test_bias_holder(cosmo_funcs_png):
    orth = cosmo_funcs_png.survey[0].orth
    assert (orth.A, orth.alpha) == (orth_lss.A, 2)


def test_pk_goes_as_k2(cosmo_funcs_png, monkeypatch):
    """P(k)'s PNG term has k^alpha/M(k) with alpha = 2, as Eq - with the same b_01 the two agree"""
    monkeypatch.setattr(cosmo_funcs_png, "get_PNG_bias", lambda zz, ti, shape: (np.array(2.0), np.array(0.0)))
    k = np.array([0.01, 0.05])
    for l in (0, 2):
        np.testing.assert_allclose(
            getattr(pk.Orth, f"l{l}")(cosmo_funcs_png, k, ZZ),
            getattr(pk.Eq, f"l{l}")(cosmo_funcs_png, k, ZZ),
            rtol=1e-12,
        )


def test_exports_replaced():
    for f in (bk.Orth.l0, bk.Orth.l2, bk.Orth_, bk_mt.Orth.l0, bk_mt.Orth.l2):
        assert getattr(f, "orth_lss", False)
    assert "Orth" not in TABLE_MODULES  # the standard-template table would undo it


def test_fNL_keywords(cosmo_funcs_png):
    """fNL_orth wins over fNL, and the other shapes' fNL_loc/fNL_eq (which bk_func passes every shape) are ignored"""
    one = bk.Orth.l0(cosmo_funcs_png, K1, K2, theta=THETA, zz=ZZ)
    got = bk.Orth.l0(cosmo_funcs_png, K1, K2, theta=THETA, zz=ZZ, fNL=5.0, fNL_orth=2.0, fNL_eq=50.0, fNL_loc=7.0)
    np.testing.assert_allclose(got, 2 * one, rtol=1e-12)


def test_views_reused_and_released(cosmo_funcs_png):
    """one pair of views per cosmo_funcs, so the bk tables' cache hits - and they do not keep it alive"""
    assert orth_lss._views(cosmo_funcs_png) is orth_lss._views(cosmo_funcs_png)
    cf = utils.copy(cosmo_funcs_png)
    assert orth_lss._views(cf)[0] is not orth_lss._views(cosmo_funcs_png)[0]
    n = len(orth_lss._VIEWS)
    del cf
    gc.collect()
    assert len(orth_lss._VIEWS) == n - 1


@pytest.fixture(scope="module")
def cf_mt_png(cosmo):
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(6e-16), compute_bias=True, verbose=False)


@pytest.mark.parametrize("shape", ["Eq", "Orth"])
def test_multi_tracer_matches_numeric(cf_mt_png, shape):
    """bk_mt.Eq and the orthogonal-LSS bk_mt.Orth against the numeric-mu bispectrum, field i carrying survey[i-1] -
    through the part linear in fNL (bk_mt keeps fNL^2 terms, in which Orth's primordial cross terms are Eq's)"""
    lm = [(0, 0), (2, 0)]

    def numeric(F):
        kerns = (["N", shape],) * 3
        return nbk.get_multipoles(*kerns, lm, cf_mt_png, K1, K2, theta=THETA, zz=ZZ, fNL=F) - nbk.get_multipoles(
            ["N"], ["N"], ["N"], lm, cf_mt_png, K1, K2, theta=THETA, zz=ZZ
        )

    num = _linear(numeric)
    for i, l in enumerate((0, 2)):
        ref = _linear(
            lambda F, l=l: getattr(getattr(bk_mt, shape), f"l{l}")(cf_mt_png, K1, K2, theta=THETA, zz=ZZ, fNL=F)
        )
        np.testing.assert_allclose(num[i], ref, rtol=1e-8)


def test_full_betas_reaches_shifted_cosmology(cosmo, cosmo_funcs_png, monkeypatch):
    """cosmology derivatives rebuild ClassWAP and adopt the fiducial survey - the model options must come too"""
    monkeypatch.setattr(cosmo_funcs_png, "full_betas", True)
    cf_h = cw.ClassWAP(cosmo, fast=True, verbose=False).adopt_survey(cosmo_funcs_png)
    assert cf_h.full_betas is True
