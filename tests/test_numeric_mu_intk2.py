"""IntK2 - kappa^(2) as a Z2, its line-of-sight integrals evaluated with the field on the line of sight.

The evaluation folds the 1/r of grad_perp^2 into the radial functions, splits the post-Born r and r' integrals and
integrates polynomial fits (compute_robust_integral). These check it against the integrands summed directly, against
IntK1.kappa_g for the lensing of Phi^(2), and the symmetries every Z2 of a real field has.

The quadrature checks sit at |p| d <~ 12. Past p d ~ 20 the deg-8 fit itself is off by ~1%: Om_m comes from dH_c, a
spline derivative, so phi_r is only piecewise smooth - the same for IntK1 in numeric_mu.pk, not a property of IntK2.
"""

import numpy as np
import pytest

from cosmo_wap.numeric_mu.kernels import F2, IntK1, IntK2, Unpack

TERMS = ["_I1", "_I2", "_I3", "_I4", "_I5"]


@pytest.fixture(scope="module")
def tri():
    """One triangle seen along a few lines of sight - large scales, so |p| d <~ 12 (see above)"""
    zz, q1, q2, cos12 = 1.0, 0.004, 0.006, -0.3
    mu1 = np.array([-0.7, 0.05, 0.4, 0.9])
    phi = np.array([0.3, 2.0, 4.0, 5.5])
    mu2 = mu1 * cos12 + np.sqrt(1 - mu1**2) * np.sqrt(1 - cos12**2) * np.cos(phi)
    return zz, mu1, mu2, q1, q2, cos12


def _brute(term, cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, n=200001):
    """The (4.8)-(4.12) integrands as written in the IntK2 docstrings, 1/r unfolded - trapezoid in r and r'"""
    d, H, _, _, _ = Unpack.get_int_params(cosmo_funcs, zz)
    r = np.linspace(1e-3, d, n)
    _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)
    phi_r = 1.5 * OM_r * H_r**2 * D1_r

    def J(f, p):
        return np.trapezoid(np.exp(1j * p[:, None] * (r - d)) * f, r, axis=-1)

    def L(q, mu):  # grad_perp^2 -> -q**2 L, on the r grid
        return 1 - mu[:, None] ** 2 + 2j * mu[:, None] / (q * r)

    phi_x = 1.5 * cosmo_funcs.Om_m(zz) * H**2 * cosmo_funcs.D(zz)
    fHD = cosmo_funcs.f(zz) * H * cosmo_funcs.D(zz)
    qs, mus = (q1, q2), (mu1, mu2)
    dlna = [phi_x / q**2 + 1j * fHD * mu / q for q, mu in zip(qs, mus)]
    A = -q1 * q2 * (cos12 - mu1 * mu2)

    if term == "_I1":
        q12 = np.sqrt(q1**2 + q2**2 + 2 * q1 * q2 * cos12)
        mu12 = (q1 * mu1 + q2 * mu2) / q12
        return F2(q1, q2, cos12, 3 / 7) * J((d - r) * r / d * phi_r * D1_r * L(q12, mu12), q1 * mu1 + q2 * mu2)
    tot = 0
    for a, b in ((0, 1), (1, 0)):
        pa, pb = qs[a] * mus[a], qs[b] * mus[b]
        if term == "_I2":
            tot = tot - dlna[a] / (2 * H) * J(r / d * phi_r * L(qs[b], mus[b]), pb)
        elif term == "_I3":
            tot = tot + A / qs[a] ** 2 * J(phi_r, pa) * dlna[b] / (2 * H)
        elif term == "_I4":
            Ja = J((d - r) * r**2 / d * phi_r * L(qs[a], mus[a]), pa)
            tot = tot + A / qs[b] ** 2 * Ja * J((1 - r / d) * phi_r, pb)
        else:
            Ja = J((d - r) * r**2 / d * phi_r, pa)
            tot = tot + A**2 / (q1 * q2) ** 2 * Ja * J(r / d * (r / d - 1) * phi_r, pb)
    return tot


@pytest.mark.parametrize("term", TERMS)
def test_matches_direct_quadrature(cosmo_funcs, tri, term):
    """Polynomial fit + folded 1/r + factorised r, r' against the integrands summed as written"""
    got = getattr(IntK2, term)(cosmo_funcs, *tri)
    want = _brute(term, cosmo_funcs, *tri)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.max(np.abs(want)))


def test_phi2_lensed_as_kappa_g(cosmo_funcs, tri):
    """I1 is Phi^(2) through the first order convergence: IntK1.kappa_g's terms with D1_r*F2 on the second order
    potential - pins the 3/2, the (d - r) r/d and the sign of the 2i term to the first order kernel"""
    zz, mu1, mu2, q1, q2, cos12 = tri
    d = cosmo_funcs.comoving_dist(zz)
    r = np.linspace(1e-3, d, 200001)
    D1_r = cosmo_funcs.D(cosmo_funcs.d_to_z(r))
    q12 = np.sqrt(q1**2 + q2**2 + 2 * q1 * q2 * cos12)
    p12 = q1 * mu1 + q2 * mu2
    mu12 = p12 / q12

    want = 0
    for i, j, arr, wt in IntK1.kappa_g(r, cosmo_funcs, zz=zz):
        assert wt == {1: 1.0}
        want = want + mu12**i * q12**j * np.trapezoid(np.exp(1j * p12[:, None] * (r - d)) * arr * D1_r, r, axis=-1)
    want = F2(q1, q2, cos12, 3 / 7) * want
    np.testing.assert_allclose(IntK2._I1(cosmo_funcs, *tri), want, rtol=1e-4, atol=1e-4 * np.max(np.abs(want)))


@pytest.mark.parametrize("term", TERMS)
def test_real_field(cosmo_funcs, tri, term):
    """K(-q1, -q2) = K(q1, q2)* - the 2i terms, the Doppler in Dlna and the e^{ip(r-d)} must flip together"""
    zz, mu1, mu2, q1, q2, cos12 = tri
    kern = getattr(IntK2, term)
    np.testing.assert_allclose(
        kern(cosmo_funcs, zz, -mu1, -mu2, q1, q2, cos12),
        np.conj(kern(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12)),
        rtol=1e-10,
    )


@pytest.mark.parametrize("term", TERMS)
def test_symmetric_in_the_two_modes(cosmo_funcs, tri, term):
    zz, mu1, mu2, q1, q2, cos12 = tri
    kern = getattr(IntK2, term)
    np.testing.assert_allclose(
        kern(cosmo_funcs, zz, mu2, mu1, q2, q1, cos12), kern(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12), rtol=1e-10
    )


def test_kappa2_and_L(cosmo_funcs, tri):
    """kappa2 shares one _los across the five terms; L weights it by 2(Q - 1) of the field's own tracer"""
    zz = tri[0]
    parts = sum(getattr(IntK2, term)(cosmo_funcs, *tri) for term in TERMS)
    kappa2 = IntK2.kappa2(cosmo_funcs, *tri)
    np.testing.assert_allclose(kappa2, parts, rtol=1e-12)
    Qm = cosmo_funcs.survey[0].Q(zz)
    np.testing.assert_allclose(IntK2.L(cosmo_funcs, *tri), 2 * (Qm - 1) * kappa2, rtol=1e-12)
