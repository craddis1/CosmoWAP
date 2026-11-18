"""bk_func(sigma=...) - FoG inside the (mu, phi) integral of each term's ylm - against the analytic multipoles as
sigma -> 0, single and multi-tracer.

The analytic multipoles themselves are checked against the (mu, phi) forms in test_numeric_mu_bk (the numeric
kernels are the forms point by point, and their multipoles the analytic ones), test_pp_terms and test_ws_terms.
"""

import numpy as np
import pytest

import cosmo_wap as cw
import cosmo_wap.bk as bk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zz():
    return 1.0


RTOL = 1e-4  # 2D integration converges slower than 1D


# ---------------------------------------------------------------------------
# bk_func(sigma=...) — FoG damping, which routes through the term's ylm method
# so the damping sits inside the (mu,phi) integral.
# ---------------------------------------------------------------------------

# leading (non-zero) multipole of each term that has a (mu,phi) expression
FOG_TERMS = {"NPP": 0, "GR1": 1, "GR2": 0, "Loc": 0, "Eq": 0, "Orth": 0}
# the WS terms' (mu,phi) forms come from mathwap (bk/WS_.py), their multipoles from MathWAP - an independent check
WS_FOG_TERMS = {"WA1": 1, "RR2": 0, "RRGR": 2, "WS": 0}

TRI = (np.array([0.08]), np.array([0.12]), np.array([np.pi / 4]))


class TestBkFuncFoG:
    @pytest.mark.parametrize("ell", [0, 2])
    @pytest.mark.parametrize("term", FOG_TERMS)
    def test_sigma_zero_matches_analytic(self, cosmo_funcs_png, zz, term, ell):
        """Undamped, the numerical route must return the analytic multipole."""
        k1, k2, theta = TRI
        args = (term, ell, cosmo_funcs_png, k1, k2)
        damped = bk.bk_func(*args, theta=theta, zz=zz, sigma=0.0)
        analytic = bk.bk_func(*args, theta=theta, zz=zz)
        np.testing.assert_allclose(np.real(damped), analytic, rtol=RTOL)

    @pytest.mark.parametrize("term,ell", FOG_TERMS.items())
    def test_sigma_damps(self, cosmo_funcs_png, zz, term, ell):
        k1, k2, theta = TRI
        args = (term, ell, cosmo_funcs_png, k1, k2)
        undamped = bk.bk_func(*args, theta=theta, zz=zz, sigma=0.0)
        damped = bk.bk_func(*args, theta=theta, zz=zz, sigma=8.0)
        assert np.all(np.abs(damped) < np.abs(undamped))

    @pytest.mark.parametrize("fNL_kwarg", ["fNL", "fNL_loc"])
    def test_fNL_reaches_the_expression(self, cosmo_funcs_png, zz, fNL_kwarg):
        """Both the generic and the shape-specific name, as for the analytic multipoles."""
        k1, k2, theta = TRI
        args = ("Loc", 0, cosmo_funcs_png, k1, k2)
        one = np.real(bk.bk_func(*args, theta=theta, zz=zz, sigma=0.0))
        five = np.real(bk.bk_func(*args, theta=theta, zz=zz, sigma=0.0, **{fNL_kwarg: 5}))
        np.testing.assert_allclose(five / one, 5.0, rtol=RTOL)

    @pytest.mark.parametrize("term,ell", WS_FOG_TERMS.items())
    def test_ws_terms_sigma_zero_matches_analytic(self, cosmo_funcs, zz, term, ell):
        k1, k2, theta = TRI
        args = (term, ell, cosmo_funcs, k1, k2)
        damped = bk.bk_func(*args, theta=theta, zz=zz, r=0.3, s=0.2, sigma=0.0)
        analytic = bk.bk_func(*args, theta=theta, zz=zz, r=0.3, s=0.2)
        np.testing.assert_allclose(damped, analytic, rtol=1e-9)

    def test_multi_tracer(self, cosmo, zz):
        """every bk_mt term has a ylm now (NPP, GR1, GR2 and PNG from bk/PP_.py), so FoG -> undamped as sigma -> 0"""
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(6e-16), verbose=False)
        k1, k2, theta = TRI
        for term in ("Full", "Loc"):
            damped = bk.bk_func(term, 2, cf, k1, k2, theta=theta, zz=zz, sigma=0.0)
            np.testing.assert_allclose(damped, bk.bk_func(term, 2, cf, k1, k2, theta=theta, zz=zz), rtol=1e-9)
