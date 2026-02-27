"""Test that numerical 2D integration agrees with analytic multipoles for Bk terms.

Each bispectrum term class has:
- A mu,phi-dependent function (Bk_0, GR_1, GR_2 in cosmo_wap.bk)
- l0, l2, ...:  precomputed analytic multipoles (m=0)

For GR1 there are also m!=0 analytic multipoles (l1m1, l3m1, l3m2, l3m3).

We test by performing direct numerical integration of
    int conj(Y_lm) * B(mu,phi) dmu dphi
and comparing against the analytic l{ell} or l{ell}m{m} methods.
"""

import numpy as np
import pytest
import scipy

import cosmo_wap as cw
import cosmo_wap.bk as bk
from cosmo_wap.bk import GR1, GR2, NPP, RRGR
from cosmo_wap.lib import utils

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cosmo():
    return utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=4.0)


@pytest.fixture(scope="module")
def cosmo_funcs(cosmo):
    sp = cw.SurveyParams.Euclid(cosmo)
    return cw.ClassWAP(cosmo, sp, verbose=False)


@pytest.fixture(scope="module")
def zz():
    return 1.0


# Triangle configurations: (k1, k2, theta)
TRIANGLES = {
    "equilateral": (0.1, 0.1, np.pi / 3),
    "squeezed": (0.1, 0.1, 0.1),
    "generic": (0.08, 0.12, np.pi / 4),
}

N_GL = 32  # Gauss-Legendre quadrature order per dimension
RTOL = 1e-4  # 2D integration converges slower than 1D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numerical_ylm(bk_func, ell, m, cf, k1, k2, theta, zz):
    """Numerically integrate conj(Y_lm) * B(mu,phi) over mu in [-1,1] and phi in [0,2pi].

    Uses 2D Gauss-Legendre quadrature, independently of integrate.ylm.
    """
    nodes, weights = np.polynomial.legendre.leggauss(N_GL)
    nodes = np.real(nodes)
    mesh_x, mesh_y = np.meshgrid(nodes, nodes, indexing="ij")
    mesh_wx, mesh_wy = np.meshgrid(weights, weights, indexing="ij")

    # Transform nodes: phi in [0,2pi], mu in [-1,1]
    phi_nodes = (2 * np.pi) * (mesh_x + 1) / 2  # [-1,1] -> [0,2pi]
    mu_nodes = mesh_y  # [-1,1] -> [-1,1]

    # Add trailing dims for broadcasting with quadrature grid
    k1a = k1[:, np.newaxis, np.newaxis]
    k2a = k2[:, np.newaxis, np.newaxis]
    theta_a = theta[:, np.newaxis, np.newaxis]

    # Evaluate bispectrum on the grid
    bk_vals = bk_func(mu_nodes, phi_nodes, cf, k1a, k2a, theta=theta_a, zz=zz)

    # Spherical harmonic: sph_harm_y(n, m, theta_polar, phi_azimuthal)
    ylm_vals = scipy.special.sph_harm_y(ell, m, np.arccos(mu_nodes), phi_nodes)

    # Integrate: Jacobian for phi transformation = pi
    integrand = np.conj(ylm_vals) * bk_vals
    return np.pi * np.sum(mesh_wx * mesh_wy * integrand, axis=(-2, -1))


def _check_multipole(bk_func, term_cls, ell, m, cf, k1, k2, theta, zz, rtol=RTOL):
    """Assert numerical integration matches the analytic multipole."""
    k1a, k2a, theta_a = np.array([k1]), np.array([k2]), np.array([theta])

    numerical = _numerical_ylm(bk_func, ell, m, cf, k1a, k2a, theta_a, zz)

    method_name = f"l{ell}" if m == 0 else f"l{ell}m{m}"
    analytic = getattr(term_cls, method_name)(cf, k1a, k2a, theta=theta_a, zz=zz)

    np.testing.assert_allclose(
        np.real(numerical),
        np.real(analytic),
        atol=1e-8,
        rtol=rtol,
        err_msg=f"{term_cls.__name__} l={ell} m={m} real part mismatch (k1={k1}, k2={k2}, theta={theta:.3f})",
    )
    np.testing.assert_allclose(
        np.imag(numerical),
        np.imag(analytic),
        atol=1e-8,
        rtol=rtol,
        err_msg=f"{term_cls.__name__} l={ell} m={m} imag part mismatch (k1={k1}, k2={k2}, theta={theta:.3f})",
    )


# ---------------------------------------------------------------------------
# NPP — Newtonian plane-parallel (even multipoles: l0, l2)
# ---------------------------------------------------------------------------


class TestBkNPP:
    @pytest.mark.parametrize("ell", [0, 2])
    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    def test_multipole(self, cosmo_funcs, zz, ell, tri):
        k1, k2, theta = TRIANGLES[tri]
        _check_multipole(bk.Bk_0, NPP, ell, 0, cosmo_funcs, k1, k2, theta, zz)


# ---------------------------------------------------------------------------
# GR1 — first-order relativistic (odd multipoles: l1, l3)
# ---------------------------------------------------------------------------


class TestBkGR1:
    @pytest.mark.parametrize("ell", [1, 3])
    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    def test_multipole_m0(self, cosmo_funcs, zz, ell, tri):
        k1, k2, theta = TRIANGLES[tri]
        _check_multipole(bk.GR_1, GR1, ell, 0, cosmo_funcs, k1, k2, theta, zz)

    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    def test_l1m1(self, cosmo_funcs, zz, tri):
        k1, k2, theta = TRIANGLES[tri]
        _check_multipole(bk.GR_1, GR1, 1, 1, cosmo_funcs, k1, k2, theta, zz)

    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_l3m(self, cosmo_funcs, zz, tri, m):
        k1, k2, theta = TRIANGLES[tri]
        _check_multipole(bk.GR_1, GR1, 3, m, cosmo_funcs, k1, k2, theta, zz)


# ---------------------------------------------------------------------------
# GR2 — second-order relativistic (even multipoles: l0, l2)
# ---------------------------------------------------------------------------


class TestBkGR2:
    @pytest.mark.parametrize("ell", [0, 2])
    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    def test_multipole(self, cosmo_funcs, zz, ell, tri):
        k1, k2, theta = TRIANGLES[tri]
        _check_multipole(bk.GR_2, GR2, ell, 0, cosmo_funcs, k1, k2, theta, zz)


# ---------------------------------------------------------------------------
# RRGR — radial-radial × GR cross term (even multipoles: l0, l2)
# No mu,phi-dependent function available, so we smoke-test the analytic
# multipoles directly (exercises get_derivs + get_beta_derivs).
# ---------------------------------------------------------------------------


class TestBkRRGR:
    @pytest.mark.parametrize("ell", [0, 2])
    @pytest.mark.parametrize("tri", TRIANGLES.keys())
    def test_multipole_finite(self, cosmo_funcs, zz, ell, tri):
        k1, k2, theta = TRIANGLES[tri]
        k1a, k2a, theta_a = np.array([k1]), np.array([k2]), np.array([theta])
        result = getattr(RRGR, f"l{ell}")(cosmo_funcs, k1a, k2a, theta=theta_a, zz=zz)
        assert np.all(np.isfinite(result)), f"RRGR l{ell} returned non-finite for {tri}"

    @pytest.mark.parametrize("ell", [0, 2])
    def test_multipole_shape(self, cosmo_funcs, zz, ell):
        k = np.linspace(0.05, 0.15, 5)
        theta = np.full_like(k, np.pi / 3)
        result = getattr(RRGR, f"l{ell}")(cosmo_funcs, k, k, theta=theta, zz=zz)
        assert result.shape == k.shape, f"RRGR l{ell} shape mismatch"
