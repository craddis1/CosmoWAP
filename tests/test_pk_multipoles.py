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

import numpy as np
import pytest
import scipy

import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.lib.angular_integrate import int_mu
from cosmo_wap.pk import GR1, GR2, NPP

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
        return ((2 * ell + 1) / 2) * leg * expression

    return int_mu(integrand, N_MU, cf, k1, zz, t=t)


def _check_multipole(term_cls, ell, cf, k1, zz, t=0):
    """Assert numerical integration of mu() ≈ analytic l{ell}() for a given term."""
    numerical = _numerical_multipole(term_cls, ell, cf, k1, zz, t=t)
    analytic = getattr(term_cls, f"l{ell}")(cf, k1, zz, t=t)
    np.testing.assert_allclose(
        np.real(numerical), np.real(analytic), rtol=RTOL, err_msg=f"{term_cls.__name__} l{ell} real part mismatch"
    )
    # atol covers the near-zero auto-tracer odd multipoles, which only cancel to float
    # dust (~1e-13) in the analytic expressions
    np.testing.assert_allclose(
        np.imag(numerical),
        np.imag(analytic),
        atol=1e-10,
        rtol=RTOL,
        err_msg=f"{term_cls.__name__} l{ell} imag part mismatch",
    )


def _check_legendre_method(term_cls, ell, cf, k1, zz, t=0):
    """Assert that the l() method (integrate.legendre) ≈ analytic l{ell}()."""
    numerical = term_cls.l(ell, cf, k1, zz, t=t, n_mu=N_MU)
    analytic = getattr(term_cls, f"l{ell}")(cf, k1, zz, t=t)
    np.testing.assert_allclose(
        np.real(numerical), np.real(analytic), rtol=RTOL, err_msg=f"{term_cls.__name__} l({ell}) real part mismatch"
    )
    # atol covers the near-zero auto-tracer odd multipoles, which only cancel to float
    # dust (~1e-13) in the analytic expressions
    np.testing.assert_allclose(
        np.imag(numerical),
        np.imag(analytic),
        atol=1e-10,
        rtol=RTOL,
        err_msg=f"{term_cls.__name__} l({ell}) imag part mismatch",
    )


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
        expected = 1j * D1**2 * Pk[:, np.newaxis] * mu_b * (-b1 * xgr1 + f * mu_b**2 * (gr1 - xgr1) + gr1 * xb1) / k_b
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


# ---------------------------------------------------------------------------
# PNG kernel — scale-dependent bias as a numeric-mu kernel
# ---------------------------------------------------------------------------


class TestPNGKernel:
    """K1.PNG must reproduce the analytic Loc term when paired with the Kaiser kernel.

    ['N','PNG'] squares to NPP + Loc: the analytic Loc holds the N x PNG cross and the
    PNG x PNG piece, so the sum of the two analytic terms is the exact counterpart.
    """

    # source-only kernels are polynomial in mu, so Gauss-Legendre is exact (each panel of the
    # composite rule integrates the polynomial exactly too) - pinned rather than left to the
    # default so the rtol=1e-10 below does not ride on whatever n_mu the default happens to be.
    # The GL=False trapezoid grid only gives ~1e-3 on l2.
    GL_GRID = [64, True, 8, 8]  # n_mu, GL, los_n, deg

    @pytest.mark.parametrize("fNL", [1.0, 10.0, -5.0])
    @pytest.mark.parametrize("ell", [0, 2])
    def test_matches_analytic_loc(self, single_tracer, k1, zz, ell, fNL):
        from cosmo_wap.pk import pk_func

        kernel = pk_func(None, ell, single_tracer, k1, zz, kernels=["N", "PNG"], mu_grid=self.GL_GRID, fNL=fNL)
        analytic = pk_func(["NPP", "Loc"], ell, single_tracer, k1, zz, fNL=fNL)
        np.testing.assert_allclose(np.real(kernel), np.real(analytic), rtol=1e-10)

    @pytest.mark.parametrize("ell", [0, 2])
    def test_multi_tracer(self, multi_tracer, k1, zz, ell):
        from cosmo_wap.pk import pk_func

        kernel = pk_func(None, ell, multi_tracer, k1, zz, kernels=["N", "PNG"], mu_grid=self.GL_GRID, fNL=10.0)
        analytic = pk_func(["NPP", "Loc"], ell, multi_tracer, k1, zz, fNL=10.0)
        np.testing.assert_allclose(np.real(kernel), np.real(analytic), rtol=1e-10)

    def test_fnl_zero_leaves_kaiser(self, single_tracer, k1, zz):
        """At fNL=0 the PNG kernel vanishes and only the Kaiser signal is left."""
        from cosmo_wap.pk import pk_func

        with_png = pk_func(None, 0, single_tracer, k1, zz, kernels=["N", "PNG"], fNL=0)
        kaiser = pk_func(None, 0, single_tracer, k1, zz, kernels=["N"])
        np.testing.assert_allclose(np.real(with_png), np.real(kaiser), rtol=1e-12)

    def test_shape_sets_k_scaling(self, single_tracer, k1, zz, monkeypatch):
        """Eq/Orth differ from Loc by k**alpha, and alpha comes from shape alone.

        The PNG bias is stubbed out so the ratio isolates the k-scaling - the real Eq/Orth
        biases need compute_bias=True on ClassWAP."""
        from cosmo_wap.numeric_mu.kernels import K1

        monkeypatch.setattr(single_tracer, "get_PNG_bias", lambda zz, ti, shape: (2.0, 0.0))
        mu = np.array([0.3])[np.newaxis, :]
        k_b = k1[:, np.newaxis]

        loc = K1.PNG(single_tracer, zz, mu, k_b, fNL=1.0, shape="Loc")
        for shape, alpha in [("Orth", 1), ("Eq", 2)]:
            got = K1.PNG(single_tracer, zz, mu, k_b, fNL=1.0, shape=shape)
            np.testing.assert_allclose(got, loc * k_b**alpha, rtol=1e-12)

    def test_finite_with_integrated_kernels(self, single_tracer, k1, zz):
        """The LOS branches evaluate the source kernel at q = k*d/r, well past K_MAX where the
        raw Pk spline extrapolates negative - M_tail keeps 1/M finite instead of nan."""
        from cosmo_wap.pk import pk_func

        out = pk_func(None, 0, single_tracer, k1, zz, kernels=["N", "I", "PNG"], fNL=10.0)
        assert np.all(np.isfinite(out))

    def test_M_tail_matches_classwap_below_kmax(self, single_tracer, zz):
        """The k**-3 tail must leave the normal k range untouched."""
        from cosmo_wap.numeric_mu.kernels import M_tail

        k = np.linspace(0.01, 0.5, 20)
        np.testing.assert_allclose(M_tail(single_tracer, k, zz), single_tracer.M(k, zz), rtol=1e-12)

    def test_per_shape_fnl_override(self, single_tracer, k1, zz):
        """fNL_loc overrides fNL for the local shape, as in the analytic classes."""
        from cosmo_wap.numeric_mu.kernels import K1

        mu = np.array([0.3])[np.newaxis, :]
        k_b = k1[:, np.newaxis]
        base = K1.PNG(single_tracer, zz, mu, k_b, fNL=7.0)
        override = K1.PNG(single_tracer, zz, mu, k_b, fNL=1.0, fNL_loc=7.0)
        np.testing.assert_allclose(override, base, rtol=1e-12)
