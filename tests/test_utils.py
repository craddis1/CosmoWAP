"""Tests for cosmo_wap.lib.utils — geometry, broadcasting, decorators, get_cosmo."""
import numpy as np
import pytest

from cosmo_wap.lib import utils


# ── Triangle geometry ─────────────────────────────────────────────────────────

class TestGetTheta:
    def test_equilateral(self):
        """Equilateral triangle → θ = 2π/3 (angle opposite k3 via cosine rule)."""
        theta = utils.get_theta(0.1, 0.1, 0.1)
        assert theta == pytest.approx(2 * np.pi / 3, abs=1e-10)

    def test_degenerate_folded(self):
        """k3 = k1 + k2 → θ = 0 (folded / collinear)."""
        theta = utils.get_theta(0.1, 0.1, 0.2)
        assert theta == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_squeezed(self):
        """k3 ≈ |k1 − k2| → θ ≈ π (squeezed / anti-collinear)."""
        theta = utils.get_theta(0.1, 0.05, 0.05)
        assert theta == pytest.approx(np.pi, abs=1e-10)

    def test_vectorised(self):
        """Should broadcast over arrays."""
        k1 = np.array([0.1, 0.2])
        k2 = np.array([0.1, 0.2])
        k3 = np.array([0.1, 0.2])
        theta = utils.get_theta(k1, k2, k3)
        np.testing.assert_allclose(theta, 2 * np.pi / 3, atol=1e-10)


class TestGetK3:
    def test_round_trip(self):
        """get_k3(get_theta(k1,k2,k3), k1, k2) ≈ k3."""
        k1, k2, k3_in = 0.1, 0.15, 0.12
        theta = utils.get_theta(k1, k2, k3_in)
        k3_out = utils.get_k3(theta, k1, k2)
        assert k3_out == pytest.approx(k3_in, rel=1e-10)

    def test_zero_k3_clamped(self):
        """When θ = π and k1 = k2, k3 → 0 is clamped to 1e-4."""
        k3 = utils.get_k3(np.pi, 0.1, 0.1)
        assert k3 == pytest.approx(1e-4)


class TestGetThetaK3:
    def test_theta_from_k3(self):
        theta_expected = utils.get_theta(0.1, 0.1, 0.1)
        k3, theta = utils.get_theta_k3(0.1, 0.1, 0.1, None)
        assert theta == pytest.approx(theta_expected)
        assert k3 == pytest.approx(0.1)

    def test_k3_from_theta(self):
        k3, theta = utils.get_theta_k3(0.1, 0.1, None, 2 * np.pi / 3)
        assert k3 == pytest.approx(0.1, rel=1e-5)

    def test_raises_without_either(self):
        with pytest.raises(ValueError):
            utils.get_theta_k3(0.1, 0.1, None, None)


# ── Broadcasting helper ──────────────────────────────────────────────────────

class TestEnableBroadcasting:
    def test_adds_axes(self):
        a = np.array([1.0, 2.0])
        (b,) = utils.enable_broadcasting(a, n=2)
        assert b.shape == (2, 1, 1)

    def test_scalars_unchanged(self):
        (s,) = utils.enable_broadcasting(3.14, n=2)
        assert s == 3.14

    def test_multiple_args(self):
        a = np.ones(3)
        b, c = utils.enable_broadcasting(a, 5.0, n=3)
        assert b.shape == (3, 1, 1, 1)
        assert c == 5.0


# ── Decorators ────────────────────────────────────────────────────────────────

class TestAddEmptyMethods:
    def test_pk_decorator(self):
        @utils.add_empty_methods_pk("l3", "l4")
        class DummyPk:
            pass

        k = np.ones(5)
        assert np.all(DummyPk.l3(None, k) == 0)
        assert np.all(DummyPk.l4(None, k) == 0)

    def test_bk_decorator(self):
        @utils.add_empty_methods_bk("l3", "l4")
        class DummyBk:
            pass

        k = np.ones(5)
        result = DummyBk.l3(None, k, k, k)
        assert result.shape == (5,)
        assert np.all(result == 0)

    def test_does_not_overwrite(self):
        @utils.add_empty_methods_pk("l0")
        class HasL0:
            @staticmethod
            def l0(cosmo_funcs, k1, zz=0):
                return np.ones_like(k1)

        assert np.all(HasL0.l0(None, np.ones(3)) == 1)


# ── get_cosmo ─────────────────────────────────────────────────────────────────

class TestGetCosmo:
    def test_returns_class_object(self, cosmo):
        from classy import Class
        assert isinstance(cosmo, Class)

    def test_h_value(self, cosmo):
        assert cosmo.h() == pytest.approx(0.67, rel=1e-5)

    def test_emulator_returns_tuple(self):
        c, params = utils.get_cosmo(emulator=True)
        assert isinstance(params, dict)
        assert "h" in params
