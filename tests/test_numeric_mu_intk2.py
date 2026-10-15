"""IntK2 - kappa^(2) as a Z2, its line-of-sight integrals evaluated with the field on the line of sight.

The evaluation folds the 1/r of grad_perp^2 into the radial functions, splits the post-Born r and r' integrals and
integrates polynomial fits (compute_robust_integral). These check it against the integrands summed directly, against
IntK1.kappa_g for the lensing of Phi^(2), the products I6 - I8 against the counts expanded, and the symmetries every
Z2 of a real field has.

The quadrature checks sit at |p| d <~ 12. Further out the deg-8 fit is good to ~1e-4 up to p d ~ 400 - it was ~1e-3
while Om_m came from the spline derivative of H_c, which left phi_r only piecewise smooth (same for IntK1 in
numeric_mu.pk).
"""

import numpy as np
import pytest

from cosmo_wap.numeric_mu.kernels import F2, IntK1, IntK2, Unpack

TERMS = ["_I1", "_I2", "_I3", "_I4", "_I5", "_I6", "_I7", "_I8"]


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

    tracer = cosmo_funcs.survey[0]
    Qm, Qp, b1p = tracer.Q(zz), tracer.dQ_dlnL(zz), tracer.db1_dlnL(zz)
    D, f, b1 = cosmo_funcs.D(zz), cosmo_funcs.f(zz), tracer.b_1(zz)
    ZN = [D * (b1 + f * mu**2) for mu in mus]
    if term == "_I8":  # [(1 - Q)(1 - 2Q) - 2Q'] kappa^2 + (Q - 1)|gamma|^2, gamma off the flat Hessian
        kappa = [J((d - r) * r / d * phi_r * L(q, mu), q * mu) for q, mu in zip(qs, mus)]
        G = [J((d - r) * r / d * phi_r, q * mu) for q, mu in zip(qs, mus)]
        gamma2 = G[0] * G[1] * (2 * A**2 / (q1 * q2) ** 2 - (1 - mu1**2) * (1 - mu2**2))
        return ((1 - Qm) * (1 - 2 * Qm) - 2 * Qp) * kappa[0] * kappa[1] + (Qm - 1) * gamma2
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
        elif term == "_I6":  # Dx_perp.grad_perp Delta_N
            tot = tot + A / qs[a] ** 2 * J((d - r) * phi_r, pa) * ZN[b]
        elif term == "_I7":  # 2 kappa [(Q - 1) Delta_N - b1' D delta]
            tot = tot + J((d - r) * r / d * phi_r * L(qs[a], mus[a]), pa) * ((Qm - 1) * ZN[b] - D * b1p)
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
    """kappa2 shares one _los across its five terms; L weights it by 2(Q - 1) of the field's own tracer and adds the
    products I6 - I8"""
    zz = tri[0]
    kappa2 = IntK2.kappa2(cosmo_funcs, *tri)
    np.testing.assert_allclose(kappa2, sum(getattr(IntK2, term)(cosmo_funcs, *tri) for term in TERMS[:5]), rtol=1e-12)
    products = sum(getattr(IntK2, term)(cosmo_funcs, *tri) for term in TERMS[5:])
    Qm = cosmo_funcs.survey[0].Q(zz)
    np.testing.assert_allclose(IntK2.L(cosmo_funcs, *tri), 2 * (Qm - 1) * kappa2 + products, rtol=1e-12)


@pytest.mark.parametrize("deg", [3, 8])
def test_los_fits_radial_functions_once(cosmo_funcs, tri, monkeypatch, deg):
    """All three frequency groups share one fit, while keeping their own broadcast shapes."""
    polyfit = np.polyfit
    calls = []

    def fit(*args, **kwargs):
        calls.append(kwargs["deg"])
        return polyfit(*args, **kwargs)

    monkeypatch.setattr(np, "polyfit", fit)
    zz, mu1, mu2, q1, q2, _ = tri
    los = IntK2._los(cosmo_funcs, zz, mu1[:, None], mu2[None, :], q1, q2, deg=deg)
    assert calls == [deg]
    assert los["J"][0]["1a"].shape == (len(mu1), 1)
    assert los["J"][1]["1a"].shape == (1, len(mu2))
    assert los["J12"]["1a"].shape == (len(mu1), len(mu2))


def test_products_from_the_counts(cosmo_funcs):
    """I6 - I8 against the counts they come from, expanded numerically: 1 + Delta = [1 + Delta_N(x + Dx_perp)] det A
    exp(Q ln mu - Q' (ln mu)**2 / 2), b1 -> b1 - b1' ln mu, mu = 1/det A, with the sky's Hessian of the lensing
    potential psi_ij = 2 G q_i q_j / q**2 + (2i mu J_4c / q) delta_ij - the curvature term a pure trace - and
    Dx_perp = 2i d J_4c q_perp / q**2. Mode 1 lenses and mode 2 is the source (I6, I7), or both lens (I8): the
    epsilon_1 epsilon_2 part of the counts is twice the symmetric Z2. Made-up radial integrals, Q' and b1', q ~ 1/d so
    every piece is O(1)"""
    zz = 1.0
    d = cosmo_funcs.comoving_dist(zz)
    q1, q2, mu1, mu2, cos12 = 2 / d, 3 / d, 0.35, -0.6, -0.2
    perp = [q * np.sqrt(1 - mu**2) for q, mu in ((q1, mu1), (q2, mu2))]
    c = (cos12 - mu1 * mu2) * q1 * q2 / (perp[0] * perp[1])  # cosine between the transverse parts
    qp = [np.array([perp[0], 0.0]), perp[1] * np.array([c, np.sqrt(1 - c**2)])]
    J = [{"4b": 0.7 - 0.2j, "4c": (0.3 + 0.1j) / d}, {"4b": -0.4 + 0.3j, "4c": (-0.2 + 0.25j) / d}]
    zero = dict.fromkeys(J[0], 0)
    los = IntK2.source(cosmo_funcs, zz, mu1, mu2, q1, q2)
    los.update(Qp=0.8, Db1p=0.45, ZN=[1.3, 0.9])
    Qm, Qp, Db1p, ZN = los["Q"], los["Qp"], los["Db1p"], los["ZN"]

    def psi(i):
        return 2 * J[i]["4b"] * np.outer(qp[i], qp[i]) / (q1, q2)[i] ** 2 + 2j * (mu1, mu2)[i] * J[i]["4c"] / (q1, q2)[
            i
        ] * np.eye(2)

    def counts(e1, e2, both):
        P = e1 * psi(0) + (e2 * psi(1) if both else 0)
        detA = np.linalg.det(np.eye(2) - P)
        lnmu = -np.log(detA)
        source = (
            0 if both else e2 * (ZN[1] * np.exp(1j * qp[1] @ (e1 * 2j * d * J[0]["4c"] * qp[0] / q1**2)) - Db1p * lnmu)
        )
        return (1 + source) * detA * np.exp(Qm * lnmu - Qp * lnmu**2 / 2)

    h = 1e-4
    args = (cosmo_funcs, zz, mu1, mu2, q1, q2, cos12)
    for both, terms in ((False, ("_I6", "_I7")), (True, ("_I8",))):
        mixed = sum(s1 * s2 * counts(s1 * h, s2 * h, both) for s1 in (1, -1) for s2 in (1, -1)) / (4 * h**2)
        kern = sum(getattr(IntK2, t)(*args, los=dict(los, J=[J[0], J[1] if both else zero])) for t in terms)
        np.testing.assert_allclose(mixed, 2 * kern, rtol=1e-6)
