"""Tests for cosmo_wap.lib.luminosity_funcs — positivity, monotonicity, finiteness."""

import numpy as np
import pytest

from cosmo_wap.lib.luminosity_funcs import (
    BGSLuminosityFunction,
    LBGLuminosityFunction,
    Model1LuminosityFunction,
    Model3LuminosityFunction,
    WISELuminosityFunction,
    lum_derivs,
)


@pytest.fixture(scope="module")
def cosmo_lf():
    """Dedicated CLASS instance for luminosity-function tests."""
    from cosmo_wap.lib import utils

    return utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=6.0)


# ── H-alpha luminosity functions (Model 1 & 3) ──────────────────────────────


class TestHaLuminosityFunctions:
    @pytest.fixture(scope="class", params=["model1", "model3"])
    @classmethod
    def lf(cls, request, cosmo_lf):
        if request.param == "model1":
            return Model1LuminosityFunction(cosmo_lf)
        return Model3LuminosityFunction(cosmo_lf)

    def test_phi_non_negative(self, lf):
        """Luminosity function Φ(L,z) >= 0."""
        L = np.logspace(40, 44, 50)
        z_arr = np.array([1.0, 1.5])
        phi = lf.luminosity_function(L, z_arr)
        assert np.all(phi >= 0)

    def test_be_finite(self, lf):
        z = np.linspace(0.9, 1.8, 20)
        be = lf.get_be(2e-16, z)
        assert np.all(np.isfinite(be))

    def test_b1_number_weighted(self, lf):
        """b_1 = <b(L)> weighted by phi dL (1909.12069 eq 15) iff db_1/dlnL_c = Q (b_1 - b(L_c))."""
        z = np.linspace(1.0, 1.8, 9)
        F_c = 3e-16
        _, db1, _ = lum_derivs(lf, F_c, z)
        b_at_cut = lf.b_1(np.log10(lf.L_c(F_c, z)), z)
        np.testing.assert_allclose(db1, lf.get_Q(F_c, z) * (lf.get_b_1(F_c, z) - b_at_cut), rtol=1e-3)


# ── LBG luminosity function (MegaMapper) ────────────────────────────────────


class TestLBGLuminosityFunction:
    @pytest.fixture(scope="class")
    @classmethod
    def lf(cls, cosmo_lf):
        return LBGLuminosityFunction(cosmo_lf)

    def test_b1_is_limit_bias(self, lf):
        """1904.13378 eq 2.7 is already the bias of the m < m_c sample - not integrated again."""
        z = lf.z_values
        np.testing.assert_allclose(lf.get_b_1(24.5, z), 0.60 * (1 + z) + 0.11 * (1 + z) ** 2)  # A(24.5), B(24.5)

    def test_off_grid_redshift_raises(self, lf):
        """The Schechter parameters exist only at z_values - not silently another redshift's."""
        with pytest.raises(ValueError, match="only defined"):
            lf.number_density(24.5, np.array([2.5]))


# ── Every luminosity function ───────────────────────────────────────────────

# (LF, cut, a brighter cut, redshifts) - BGS and LBG cut in magnitude, so brighter is smaller
EVERY_LF = [
    (Model1LuminosityFunction, 2e-16, 5e-16, np.linspace(0.9, 1.8, 4)),
    (Model3LuminosityFunction, 2e-16, 5e-16, np.linspace(0.9, 1.8, 4)),
    (WISELuminosityFunction, 2e-16, 2e-15, np.linspace(0.1, 2.0, 4)),
    (BGSLuminosityFunction, 20.0, 18.0, np.linspace(0.1, 0.5, 4)),
    (LBGLuminosityFunction, 24.5, 23.0, np.array([2.0, 3.0, 3.8, 4.9, 5.9])),
]


@pytest.mark.parametrize("LF, cut, brighter, zz", EVERY_LF)
def test_counts_and_Q(cosmo_lf, LF, cut, brighter, zz):
    """n_g > 0 and falls past a brighter cut, Q finite."""
    lf = LF(cosmo_lf)
    ng = lf.number_density(cut, zz)
    assert np.all(ng > 0)
    assert np.all(ng >= lf.number_density(brighter, zz))
    assert np.all(np.isfinite(lf.get_Q(cut, zz)))


@pytest.mark.parametrize("LF, cut, brighter, zz", EVERY_LF)
def test_single_redshift_matches_grid(cosmo_lf, LF, cut, brighter, zz):
    """SurveyBase._get_fraction solves the bright cut one redshift at a time - each call must give that
    redshift's value (the LBG LF used to return all of z_values, so [0] was always z = 2.0)."""
    lf = LF(cosmo_lf)
    for func in (lf.number_density, lf.get_Q):
        one = np.array([func(cut, np.array([z]))[0] for z in zz])
        np.testing.assert_allclose(one, func(cut, zz), rtol=1e-6)


# ── WISE 2.4 micron luminosity function (SPHEREx) ───────────────────────────


def gamma_upper(a, y):
    """Upper incomplete gamma Gamma(a, y) for the a < 0 that alpha = -1.05 gives, which
    scipy's gammaincc does not accept. Integrated in ln y - quad to infinity in y misses
    the sharp rise just above the lower limit and silently returns the wrong answer."""
    from scipy.integrate import quad

    return quad(lambda u: np.exp(a * u - np.exp(u)), np.log(y), np.log(200.0))[0]


class TestWISELuminosityFunction:
    """Anchored on numbers Lake et al. (2018) [arXiv:1702.07829] quote independently of
    their fit tables, so the whole unit chain (nu F_nu -> F_nu -> 4 pi d_L^2 F_nu in solar
    units) is checked, not just the parameter values."""

    @pytest.fixture(scope="class")
    @classmethod
    def lf(cls):
        """Built on the WMAP9 cosmology of the original fit, so the abstract's numbers apply."""
        from cosmo_wap.lib import utils

        return WISELuminosityFunction(utils.get_cosmo(h=0.7, Omega_m=0.2793, k_max=1.0, z_max=6.0))

    def test_lake_abstract_number_densities(self, lf):
        """n(>1e6 L_sun) = 0.08 Mpc^-3 and n(>L*) = 1e-3 Mpc^-3 at z=0, to their one digit."""
        h3 = lf.cosmo.h() ** 3
        a = 1 + lf.alpha
        assert lf.phi_0 * gamma_upper(a, 1e6 / 1e10 / lf.L_0) * h3 == pytest.approx(0.08, rel=0.1)
        assert lf.phi_0 * gamma_upper(a, 1.0) * h3 == pytest.approx(1e-3, rel=0.3)

    def test_G_matches_incomplete_gamma(self, lf):
        """The simpson integral over luminosity reproduces Gamma(1+alpha, y_c)."""
        zz = np.linspace(0.1, 4.3, 15)
        y_c = lf.get_y(lf.L_c(2e-16, zz), zz)
        exact = np.array([gamma_upper(1 + lf.alpha, y) for y in y_c])
        np.testing.assert_allclose(lf.get_G(2e-16, zz), exact, rtol=1e-5)

    def test_schechter_parameters_at_reference_redshifts(self, lf):
        """phi*(z) and L*(z) against Table 2 of arXiv:2608.18334, which samples the same fit."""
        for z, phi_star, L_star in [(0.0, 0.0169, 3.12), (0.38, 0.016, 5.6), (1.5, 0.015, 9.9)]:
            zz = np.array([z])
            assert lf.get_phi_star(zz)[0] == pytest.approx(phi_star, rel=0.02)
            assert 1 / lf.get_y(1.0, zz)[0] == pytest.approx(L_star, rel=0.02)  # y = L/L*, so L*=1/y(L=1)

    def test_Q_finite_and_increasing(self, lf):
        """Q rises with z as the fixed flux cut probes further up the Schechter cutoff."""
        Q = lf.get_Q(2e-16, np.linspace(0.1, 4.3, 20))
        assert np.all(np.isfinite(Q))
        assert np.all(np.diff(Q) > 0)

    def test_no_linear_bias_model(self, lf):
        """The parent's H-alpha b_1 fit must not be inherited - SPHEREx supplies its own."""
        assert getattr(lf, "get_b_1", None) is None
