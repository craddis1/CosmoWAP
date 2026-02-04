"""Tests for cosmo_wap.lib.luminosity_funcs — positivity, monotonicity, finiteness."""
import numpy as np
import pytest

from cosmo_wap.lib.luminosity_funcs import (
    Model1LuminosityFunction,
    Model3LuminosityFunction,
    BGSLuminosityFunction,
    LBGLuminosityFunction,
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
