"""lib/hankel.py: P_l(k) -> xi_l(r) against closed forms (no CLASS needed)."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson
from scipy.special import spherical_jn

from cosmo_wap.lib import hankel

K = np.geomspace(1e-5, 15.0, 2**14)


def gaussian_xi(ell, r, s):
    """int dk k^2/(2 pi^2) k^l e^(-s^2 k^2) j_l(kr) = sqrt(pi) r^l e^(-r^2/4s^2) / (2^(l+2) s^(2l+3)) / (2 pi^2)"""
    return (
        np.sqrt(np.pi) * r**ell * np.exp(-(r**2) / (4 * s**2)) / (2 ** (ell + 2) * s ** (2 * ell + 3)) / (2 * np.pi**2)
    )


@pytest.mark.parametrize("ell", range(5))
@pytest.mark.parametrize("kr_taper", [200.0, None])
def test_gaussian_closed_form(ell, kr_taper):
    """P_l = (-i)^l k^l e^(-s^2 k^2) - imaginary for odd l, as CosmoWAP's - so i^l P_l is real"""
    r = np.array([5.0, 20.0, 40.0])
    pk_l = (-1j) ** ell * K**ell * np.exp(-((10.0 * K) ** 2))
    xi = hankel.pk_to_xi(K, pk_l, ell, r, kr_taper=kr_taper, k_ir=None)
    np.testing.assert_allclose(xi, gaussian_xi(ell, r, 10.0), rtol=1e-9)


def test_bin_average_l0_closed_form():
    """int r^2 j_0(kr) dr = r^2 j_1(kr)/k"""
    edges = np.arange(10.0, 110.0, 10.0)
    a, b = edges[:-1, None], edges[1:, None]
    k = np.geomspace(1e-3, 1.0, 200)
    exact = 3 * (b**2 * spherical_jn(1, k * b) - a**2 * spherical_jn(1, k * a)) / (k * (b**3 - a**3))
    np.testing.assert_allclose(hankel.bessel_kernel(0, k, r_edges=edges), exact, atol=1e-12)


@pytest.mark.parametrize("ell", [0, 1, 2, 4])
def test_narrow_bins_are_point_values(ell):
    r = np.array([20.0, 60.0, 150.0])
    edges = np.sort(np.concatenate([r - 0.005, r + 0.005]))
    k = np.geomspace(1e-3, 1.0, 200)
    np.testing.assert_allclose(
        hankel.bessel_kernel(ell, k, r_edges=edges)[::2], hankel.bessel_kernel(ell, k, r=r), atol=1e-6
    )


def test_ir_fix_removes_k_min_dependence():
    """P_0 ~ k^(n_s - 4) at low k makes xi_0 diverge (slowly) as k_min -> 0; j_0 - S below k_ir fixes it"""
    r = np.array([20.0, 100.0, 300.0])

    def xi0(k_min, k_ir):
        k = np.geomspace(k_min, 15.0, 2**15)
        return hankel.pk_to_xi(k, k ** (-3.035) * np.exp(-((k / 0.5) ** 2)), 0, r, k_ir=k_ir)

    np.testing.assert_allclose(xi0(1e-5, 1e-4), xi0(1e-6, 1e-4), rtol=1e-6)
    assert np.all(np.abs(xi0(1e-5, None) / xi0(1e-6, None) - 1) > 1e-2)  # without it xi_0 moves with k_min


def test_ir_fix_negligible_for_convergent_terms():
    """Kaiser-like P ~ k at low k: the constant removed is ~k_ir^4"""
    r = np.array([20.0, 100.0, 300.0])
    pk = K * np.exp(-((K / 0.05) ** 2)) * 1e5
    on, off = hankel.pk_to_xi(K, pk, 0, r), hankel.pk_to_xi(K, pk, 0, r, k_ir=None)
    np.testing.assert_allclose(on, off, rtol=1e-6)


def test_bins_against_averaged_points():
    """bin average of the point xi on a fine r grid (r^2 weights) = the bin-averaged transform"""
    pk = 1e4 * K * np.exp(-((K / 0.1) ** 2)) / (1 + (K / 0.02) ** 3)
    edges = np.array([30.0, 40.0, 60.0, 100.0])
    binned = hankel.pk_to_xi(K, pk, 2, r_edges=edges)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        rf = np.linspace(a, b, 401)
        avg = simpson(rf**2 * hankel.pk_to_xi(K, pk, 2, rf), x=rf) / ((b**3 - a**3) / 3)
        np.testing.assert_allclose(binned[i], avg, rtol=1e-6)


@pytest.mark.parametrize("ell", range(5))
def test_xi_to_pk_inverts_closed_form(ell):
    """the Gaussian pair the other way, points and k^2-weighted bins"""
    r = np.linspace(0, 400, 4001)
    k = np.array([0.01, 0.05, 0.1, 0.2])
    exact = lambda k: (-1j) ** ell * k**ell * np.exp(-((10.0 * k) ** 2))
    np.testing.assert_allclose(hankel.xi_to_pk(r, gaussian_xi(ell, r, 10.0), ell, k), exact(k), rtol=1e-10)
    edges = np.array([0.02, 0.05, 0.1, 0.15])
    x, w = leggauss(16)
    a, b = edges[:-1, None], edges[1:, None]
    kn = (b - a) / 2 * x + (a + b) / 2
    binned = (w * kn**2 * exact(kn)).sum(-1) / (w * kn**2).sum(-1)
    np.testing.assert_allclose(hankel.xi_to_pk(r, gaussian_xi(ell, r, 10.0), ell, k_edges=edges), binned, rtol=1e-10)


def test_window():
    k = np.geomspace(0.01, 10.0, 1000)
    w = hankel.window(k, 1.0, 3.0)
    assert np.all(w[k <= 1.0] == 1) and np.all(w[k >= 3.0] == 0) and np.all(np.diff(w) <= 0)


def test_bad_arguments():
    with pytest.raises(ValueError, match="one of r"):
        hankel.bessel_kernel(0, K)
    with pytest.raises(ValueError, match="one of r"):
        hankel.bessel_kernel(0, K, r=[10.0], r_edges=[10.0, 20.0])
    with pytest.raises(ValueError, match="increasing"):
        hankel.bessel_kernel(0, K, r_edges=[20.0, 10.0])
