"""Tests for cosmo_wap.lib.luminosity_funcs — positivity, monotonicity, finiteness."""

import numpy as np
import pytest

from cosmo_wap.lib.luminosity_funcs import (
    BGSLuminosityFunction,
    LBGLuminosityFunction,
    Model1LuminosityFunction,
    Model3LuminosityFunction,
    WISELuminosityFunction,
)


@pytest.fixture(scope="module")
def cosmo_lf():
    """Dedicated CLASS instance for luminosity-function tests."""
    from cosmo_wap.lib import utils

    return utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=6.0)


# ── H-alpha luminosity functions (Model 1 & 3) ──────────────────────────────


class TestHaLuminosityFunctions:
    @pytest.fixture(scope="class", params=["model1", "model3"])
    def lf(self, request, cosmo_lf):
        if request.param == "model1":
            return Model1LuminosityFunction(cosmo_lf)
        return Model3LuminosityFunction(cosmo_lf)

    def test_phi_non_negative(self, lf):
        """Luminosity function Φ(L,z) >= 0."""
        L = np.logspace(40, 44, 50)
        z_arr = np.array([1.0, 1.5])
        phi = lf.luminosity_function(L, z_arr)
        assert np.all(phi >= 0)

    def test_number_density_positive(self, lf):
        """n_g > 0 for a reasonable flux cut."""
        z = np.linspace(0.9, 1.8, 20)
        ng = lf.number_density(2e-16, z)
        assert np.all(ng > 0)

    def test_brighter_cut_fewer_galaxies(self, lf):
        """Raising the flux cut (brighter) should reduce n_g."""
        z = np.linspace(0.9, 1.8, 20)
        ng_faint = lf.number_density(1e-16, z)
        ng_bright = lf.number_density(5e-16, z)
        assert np.all(ng_faint >= ng_bright)

    def test_Q_finite(self, lf):
        z = np.linspace(0.9, 1.8, 20)
        Q = lf.get_Q(2e-16, z)
        assert np.all(np.isfinite(Q))

    def test_be_finite(self, lf):
        z = np.linspace(0.9, 1.8, 20)
        be = lf.get_be(2e-16, z)
        assert np.all(np.isfinite(be))


# ── BGS luminosity function ─────────────────────────────────────────────────


class TestBGSLuminosityFunction:
    @pytest.fixture(scope="class")
    def lf(self, cosmo_lf):
        return BGSLuminosityFunction(cosmo_lf)

    def test_number_density_positive(self, lf):
        z = np.linspace(0.05, 0.5, 20)
        ng = lf.number_density(20, z)
        assert np.all(ng > 0)

    def test_brighter_cut_fewer_galaxies(self, lf):
        z = np.linspace(0.05, 0.5, 20)
        ng_faint = lf.number_density(20, z)
        ng_bright = lf.number_density(18, z)
        assert np.all(ng_faint >= ng_bright)

    def test_Q_finite(self, lf):
        z = np.linspace(0.05, 0.5, 20)
        Q = lf.get_Q(20, z)
        assert np.all(np.isfinite(Q))


# ── LBG luminosity function (MegaMapper) ────────────────────────────────────


class TestLBGLuminosityFunction:
    @pytest.fixture(scope="class")
    def lf(self, cosmo_lf):
        return LBGLuminosityFunction(cosmo_lf)

    def test_number_density_positive(self, lf):
        ng = lf.number_density(24.5)
        assert np.all(ng > 0)

    def test_brighter_cut_fewer_galaxies(self, lf):
        ng_faint = lf.number_density(25)
        ng_bright = lf.number_density(23)
        assert np.all(ng_faint >= ng_bright)

    def test_Q_finite(self, lf):
        Q = lf.get_Q(24.5)
        assert np.all(np.isfinite(Q))


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
    def lf(self):
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

    def test_number_density_positive(self, lf):
        assert np.all(lf.number_density(2e-16, np.linspace(0.1, 4.3, 20)) > 0)

    def test_brighter_cut_fewer_galaxies(self, lf):
        z = np.linspace(0.1, 4.3, 20)
        assert np.all(lf.number_density(2e-16, z) >= lf.number_density(2e-15, z))

    def test_Q_finite_and_increasing(self, lf):
        """Q rises with z as the fixed flux cut probes further up the Schechter cutoff."""
        Q = lf.get_Q(2e-16, np.linspace(0.1, 4.3, 20))
        assert np.all(np.isfinite(Q))
        assert np.all(np.diff(Q) > 0)

    def test_no_linear_bias_model(self, lf):
        """The parent's H-alpha b_1 fit must not be inherited - SPHEREx supplies its own."""
        assert getattr(lf, "get_b_1", None) is None
