"""Test that numerical mu integration agrees with analytic multipoles for Pk terms.

Each term class has:
- mu(mu,...): the full mu-dependent expression
- l(l,...):   numerical Legendre integration via integrate.legendre
- l0, l1, l2, ...: precomputed analytic multipoles

We test three things:
1. Direct numerical integration of mu() vs analytic l{ell}()
2. The l() method (via integrate.legendre) vs analytic l{ell}()
3. Kernel-based P(k,mu) = Pk * K1(tracer=0) * K1(tracer=1) vs mu() for NPP/GR
"""
import pytest
import numpy as np
import scipy

import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrate import int_mu
from cosmo_wap.pk import NPP, GR1, GR2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cosmo():
    return utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=4.0)


@pytest.fixture(scope="module")
def single_tracer(cosmo):
    sp = cw.SurveyParams.Euclid(cosmo)
    return cw.ClassWAP(cosmo, sp, verbose=False)


@pytest.fixture(scope="module")
def multi_tracer(cosmo):
    sp = cw.SurveyParams.Euclid(cosmo)
    sp.BF_split(5e-16)
    return cw.ClassWAP(cosmo, [sp.bright, sp.faint], verbose=False)


@pytest.fixture(scope="module")
def k1():
    return np.linspace(0.01, 0.2, 10)


@pytest.fixture(scope="module")
def zz():
    return 1.0


N_MU = 64  # high enough for good convergence
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numerical_multipole(term_cls, ell, cf, k1, zz, t=0):
    """Numerically integrate (2l+1)/2 * P_l(mu) * mu_func(mu,...) over mu in [-1,1]."""
    def integrand(mu, cosmo_funcs, k1, zz, t):
        leg = scipy.special.eval_legendre(ell, mu)
        expression = term_cls.mu(mu, cosmo_funcs, k1, zz, t)
        return ((2*ell+1)/2) * leg * expression

    return int_mu(integrand, N_MU, cf, k1, zz, t=t)


def _check_multipole(term_cls, ell, cf, k1, zz, t=0):
    """Assert numerical integration of mu() ≈ analytic l{ell}() for a given term."""
    numerical = _numerical_multipole(term_cls, ell, cf, k1, zz, t=t)
    analytic = getattr(term_cls, f"l{ell}")(cf, k1, zz, t=t)
    np.testing.assert_allclose(np.real(numerical), np.real(analytic), rtol=RTOL,
                               err_msg=f"{term_cls.__name__} l{ell} real part mismatch")
    np.testing.assert_allclose(np.imag(numerical), np.imag(analytic), atol=1e-15, rtol=RTOL,
                               err_msg=f"{term_cls.__name__} l{ell} imag part mismatch")


def _check_legendre_method(term_cls, ell, cf, k1, zz, t=0):
    """Assert that the l() method (integrate.legendre) ≈ analytic l{ell}()."""
    numerical = term_cls.l(ell, cf, k1, zz, t=t, n_mu=N_MU)
    analytic = getattr(term_cls, f"l{ell}")(cf, k1, zz, t=t)
    np.testing.assert_allclose(np.real(numerical), np.real(analytic), rtol=RTOL,
                               err_msg=f"{term_cls.__name__} l({ell}) real part mismatch")
    np.testing.assert_allclose(np.imag(numerical), np.imag(analytic), atol=1e-15, rtol=RTOL,
                               err_msg=f"{term_cls.__name__} l({ell}) imag part mismatch")


# ---------------------------------------------------------------------------
# NPP — Newtonian plane-parallel (even multipoles: l0, l2, l4)
# ---------------------------------------------------------------------------

class TestNPP:
    @pytest.mark.parametrize("ell", [0, 2, 4])
    def test_single_tracer(self, single_tracer, k1, zz, ell):
        _check_multipole(NPP, ell, single_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [0, 2, 4])
    def test_multi_tracer(self, multi_tracer, k1, zz, ell):
        _check_multipole(NPP, ell, multi_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [0, 2, 4])
    def test_legendre_method(self, single_tracer, k1, zz, ell):
        _check_legendre_method(NPP, ell, single_tracer, k1, zz)

    def test_kernel_consistency(self, single_tracer, k1, zz):
        """NPP.mu should equal Pk * K1_N(tracer=0) * K1_N(tracer=1)."""
        mu_vals = np.array([0.3, 0.7])
        cf = single_tracer
        Pk = cf.Pk(k1)
        D1 = cf.D(zz)
        f = cf.f(zz)
        b1 = cf.survey[0].b_1(zz)
        xb1 = cf.survey[1].b_1(zz)
        mu_b = mu_vals[np.newaxis, :]
        k_b = k1[:, np.newaxis]
        # Kaiser kernels: K = D*(b + f*mu^2)
        K_0 = D1 * (b1 + f * mu_b**2)
        K_1 = D1 * (xb1 + f * mu_b**2)
        from_kernels = Pk[:, np.newaxis] * K_0 * K_1
        from_mu = NPP.mu(mu_b, cf, k_b, zz)
        np.testing.assert_allclose(from_kernels, from_mu, rtol=1e-12)


# ---------------------------------------------------------------------------
# GR1 — first-order relativistic (odd multipoles: l1, l3)
# ---------------------------------------------------------------------------

class TestGR1:
    @pytest.mark.parametrize("ell", [1, 3])
    def test_single_tracer(self, single_tracer, k1, zz, ell):
        _check_multipole(GR1, ell, single_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [1, 3])
    def test_multi_tracer(self, multi_tracer, k1, zz, ell):
        _check_multipole(GR1, ell, multi_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [1, 3])
    def test_legendre_method(self, single_tracer, k1, zz, ell):
        _check_legendre_method(GR1, ell, single_tracer, k1, zz)

    def test_kernel_consistency(self, multi_tracer, k1, zz):
        """GR1.mu is the O(1/k) part of the cross spectrum: i*mu*Pk*D^2*[...]/k."""
        mu_vals = np.array([0.3, 0.7])
        cf = multi_tracer
        Pk = cf.Pk(k1)
        mu_b = mu_vals[np.newaxis, :]
        k_b = k1[:, np.newaxis]
        D1 = cf.D(zz)
        f = cf.f(zz)
        b1 = cf.survey[0].b_1(zz)
        xb1 = cf.survey[1].b_1(zz)
        gr1, _ = cf.get_beta_funcs(zz, ti=0)[:2]
        xgr1, _ = cf.get_beta_funcs(zz, ti=1)[:2]
        # GR1.mu = i*D^2*Pk*mu*(-b1*xgr1 + f*mu^2*(gr1-xgr1) + gr1*xb1)/k
        expected = 1j * D1**2 * Pk[:, np.newaxis] * mu_b * (
            -b1*xgr1 + f*mu_b**2*(gr1 - xgr1) + gr1*xb1
        ) / k_b
        from_mu = GR1.mu(mu_b, cf, k_b, zz)
        np.testing.assert_allclose(from_mu, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# GR2 — second-order relativistic (even multipoles: l0, l2)
# ---------------------------------------------------------------------------

class TestGR2:
    @pytest.mark.parametrize("ell", [0, 2])
    def test_single_tracer(self, single_tracer, k1, zz, ell):
        _check_multipole(GR2, ell, single_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [0, 2])
    def test_multi_tracer(self, multi_tracer, k1, zz, ell):
        _check_multipole(GR2, ell, multi_tracer, k1, zz)

    @pytest.mark.parametrize("ell", [0, 2])
    def test_legendre_method(self, single_tracer, k1, zz, ell):
        _check_legendre_method(GR2, ell, single_tracer, k1, zz)

