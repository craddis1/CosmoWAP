"""Pk terms: the l() method (numerical Legendre integration via integrate.legendre, which the composites project
missing multipoles with - see utils._project_mu) against the analytic l{ell}(), and the PNG numeric-mu kernel.

The analytic multipoles against each class's mu() are in test_pk_pp_terms (and test_ws_terms for the WS terms).
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.pk import GR1, GR2, NPP

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_tracer(cosmo):
    # The method stubs below need their own wrapper; share only the expensive CLASS setup.
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
# l() - NPP, GR1 and GR2 on two tracers, so GR1 is not zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term_cls, ell", [(NPP, 0), (NPP, 2), (NPP, 4), (GR1, 1), (GR1, 3), (GR2, 0), (GR2, 2)])
def test_legendre_method(multi_tracer, k1, zz, term_cls, ell):
    _check_legendre_method(term_cls, ell, multi_tracer, k1, zz)


# ---------------------------------------------------------------------------
# PNG kernel — scale-dependent bias as a numeric-mu kernel
# ---------------------------------------------------------------------------


class TestPNGKernel:
    """K1.Loc must reproduce the analytic Loc term when paired with the Kaiser kernel.

    ['N','Loc'] squares to NPP + Loc: the analytic Loc holds the N x PNG cross and the
    PNG x PNG piece, so the sum of the two analytic terms is the exact counterpart.
    """

    # source-only kernels are polynomial in mu, so Gauss-Legendre is exact (each panel of the
    # composite rule integrates the polynomial exactly too) - pinned rather than left to the
    # default so the rtol=1e-10 below does not ride on whatever n_mu the default happens to be.
    # The GL=False trapezoid grid only gives ~1e-3 on l2.
    GL_GRID = [64, True, 8, 8]  # n_mu, GL, los_n, deg

    @pytest.mark.parametrize("ell", [0, 2])
    def test_multi_tracer(self, multi_tracer, k1, zz, ell):
        from cosmo_wap.pk import pk_func

        kernel = pk_func(None, ell, multi_tracer, k1, zz, kernels=["N", "Loc"], mu_grid=self.GL_GRID, fNL=10.0)
        analytic = pk_func(["NPP", "Loc"], ell, multi_tracer, k1, zz, fNL=10.0)
        np.testing.assert_allclose(np.real(kernel), np.real(analytic), rtol=1e-10)

    def test_fnl_zero_leaves_kaiser(self, single_tracer, k1, zz):
        """At fNL=0 the PNG kernel vanishes and only the Kaiser signal is left."""
        from cosmo_wap.pk import pk_func

        with_png = pk_func(None, 0, single_tracer, k1, zz, kernels=["N", "Loc"], fNL=0)
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

        loc = K1.Loc(single_tracer, zz, mu, k_b, fNL=1.0)
        for shape, alpha in [("Orth", 2), ("Eq", 2)]:  # orthogonal-LSS goes as k^2, as Eq
            got = getattr(K1, shape)(single_tracer, zz, mu, k_b, fNL=1.0)
            np.testing.assert_allclose(got, loc * k_b**alpha, rtol=1e-12)

    def test_finite_with_integrated_kernels(self, single_tracer, k1, zz):
        """The LOS branches evaluate the source kernel at q = k*d/r, well past K_MAX where the
        raw Pk spline extrapolates negative - M_tail keeps 1/M finite instead of nan."""
        from cosmo_wap.pk import pk_func

        out = pk_func(None, 0, single_tracer, k1, zz, kernels=["N", "I", "Loc"], fNL=10.0)
        assert np.all(np.isfinite(out))

    def test_M_tail_matches_classwap_below_kmax(self, single_tracer, zz):
        """The k**-3 tail must leave the normal k range untouched."""
        from cosmo_wap.numeric_mu.kernels import M_tail

        k = np.linspace(0.01, 0.5, 20)
        np.testing.assert_allclose(M_tail(single_tracer, k, zz), single_tracer.M(k, zz), rtol=1e-12)

    def test_shapes_combine_with_cross_term(self, single_tracer, k1, zz, monkeypatch):
        """Several shapes in one kernel list sum into K before it is squared.

        So ['N','Eq','Orth'] is |N+Eq+Orth|**2, which carries the Eq x Orth cross term that
        two separate calls (one per shape) would miss while double counting |N|**2.
        Biases are stubbed as the real Eq/Orth ones need compute_bias=True on ClassWAP."""
        from cosmo_wap.numeric_mu import pk as numeric_mu_pk

        b01 = {"Loc": 2.0, "Eq": 3.0, "Orth": -1.5}
        monkeypatch.setattr(single_tracer, "get_PNG_bias", lambda zz, ti, shape: (b01[shape], 0.0))

        def P(kern1, kern2=None):
            return numeric_mu_pk.get_multipoles(kern1, kern2 or kern1, [0, 2], single_tracer, k1, zz, n_mu=64, fNL=1.0)

        both = P(["N", "Eq", "Orth"])
        expected = P(["N", "Eq"]) + P(["N", "Orth"]) - P(["N"]) + P(["Eq"], ["Orth"]) + P(["Orth"], ["Eq"])
        np.testing.assert_allclose(np.real(both), np.real(expected), rtol=1e-10)

        separate = P(["N", "Eq"]) + P(["N", "Orth"])  # the wrong way round - keep them distinguishable
        assert not np.allclose(np.real(both), np.real(separate), rtol=1e-3)

    def test_per_shape_fnl_override(self, single_tracer, k1, zz):
        """fNL_loc overrides fNL for the local shape, as in the analytic classes."""
        from cosmo_wap.numeric_mu.kernels import K1

        mu = np.array([0.3])[np.newaxis, :]
        k_b = k1[:, np.newaxis]
        base = K1.Loc(single_tracer, zz, mu, k_b, fNL=7.0)
        override = K1.Loc(single_tracer, zz, mu, k_b, fNL=1.0, fNL_loc=7.0)
        np.testing.assert_allclose(override, base, rtol=1e-12)
