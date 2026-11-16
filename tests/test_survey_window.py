"""lib/survey_window.py: window matrix and integral constraints against Gaussian closed forms (no CLASS needed)."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre, spherical_in

from cosmo_wap.lib import hankel
from cosmo_wap.lib.survey_window import SurveyWindow, coupling, window_multipoles, windowed_xi

R, SIG = 100.0, 5.0  # Gaussian window W(x) = exp(-x^2/2R^2) and P = exp(-k^2 SIG^2)
S = np.linspace(0, 12 * R, 2401)
Q_GAUSS = {0: np.exp(-(S**2) / (4 * R**2))}
K = np.linspace(0.002, 0.3, 40)


def gaussian_pk_hat(ell, k, sig=SIG):
    """P_l = (-i)^l k^l e^(-sig^2 k^2) through Q_0 = e^(-s^2/4R^2): xi_l ~ r^l e^(-r^2/4sig^2) (test_hankel), so the
    product is the same form with 1/sig'^2 = 1/sig^2 + 1/R^2"""
    sp2 = sig**2 * R**2 / (sig**2 + R**2)
    return (np.sqrt(sp2) / sig) ** (2 * ell + 3) * (-1j) ** ell * k**ell * np.exp(-sp2 * k**2)


def gaussian_pk(ells, k, sig=SIG):
    return np.array([(-1j) ** l * k**l * np.exp(-((sig * k) ** 2)) for l in ells])


def test_coupling():
    """Wilson+17's xî_2 = xi_0 Q_2 + xi_2 (Q_0 + 2/7 Q_2 + 2/7 Q_4) + xi_4 (2/7 Q_2 + 100/693 Q_4 + 25/143 Q_6), and
    L_l' L_L = sum_l coupling L_l in general"""
    for (lp, L), c in {(0, 2): 1, (2, 0): 1, (2, 2): 2 / 7, (2, 4): 2 / 7, (4, 4): 100 / 693, (4, 6): 25 / 143}.items():
        assert coupling(2, lp, L) == pytest.approx(c, rel=1e-14)
    mu = np.linspace(-1, 1, 7)
    for lp in range(6):
        for L in range(6):
            rhs = sum(coupling(l, lp, L) * eval_legendre(l, mu) for l in range(lp + L + 1))
            np.testing.assert_allclose(eval_legendre(lp, mu) * eval_legendre(L, mu), rhs, atol=1e-13)


@pytest.mark.parametrize("folded", [False, True])
def test_window_multipoles(folded):
    """pair counts of a known Q(s, mu) = g(s) h(mu) give back its multipoles"""
    s_edges = np.linspace(0, 400, 81)
    mu_edges = np.linspace(0 if folded else -1, 1, 201)
    h = lambda mu: 1 + 0.4 * mu**2 - 0.3 * mu**4 + (0 if folded else 0.2 * mu)
    norm = 3.7
    mu_f = np.linspace(mu_edges[0], 1, 200 * 64 + 1)
    h_bins = np.add.reduceat(h(mu_f[:-1]) * np.diff(mu_f), np.arange(0, 200 * 64, 64))  # int h over each mu bin
    shell = 4 * np.pi / 3 * np.diff(s_edges**3)  # with g = 1, RR = norm 2pi int s^2 ds int h dmu
    RR = norm * np.outer(shell / 2, h_bins) * (2 if folded else 1)
    Ls = (0, 2, 4) if folded else (0, 1, 2, 4)
    _, Q = window_multipoles(s_edges, mu_edges, RR, norm, Ls)
    x, w = leggauss(32)
    for L in Ls:
        exact = (2 * L + 1) / 2 * np.sum(w * h(x) * eval_legendre(L, x))
        np.testing.assert_allclose(Q[L], exact, atol=1e-4)
    if folded:
        with pytest.raises(ValueError, match="odd L"):
            window_multipoles(s_edges, mu_edges, RR, norm, (0, 1))


@pytest.mark.parametrize("binned", [False, True])
def test_isotropic_gaussian_closed_form(binned):
    ells = range(5)
    if binned:
        edges = np.linspace(0.0, 0.3, 31)
        W = SurveyWindow(S, Q_GAUSS, ells=ells, k_edges=edges, ells_in=ells, ic=None)
        x, w = leggauss(16)
        a, b = edges[:-1, None], edges[1:, None]
        kn = (b - a) / 2 * x + (a + b) / 2  # k^2-weighted average over each bin by Gauss-Legendre
        exact = np.array([(w * kn**2 * gaussian_pk_hat(l, kn)).sum(-1) / (w * kn**2).sum(-1) for l in ells])
    else:
        W = SurveyWindow(S, Q_GAUSS, ells=ells, k=K, ells_in=ells, ic=None)
        exact = np.array([gaussian_pk_hat(l, K) for l in ells])
    P_hat = W(gaussian_pk(ells, W.k_in))
    for l in ells:
        np.testing.assert_allclose(P_hat[l], exact[l], atol=(2e-4 if binned else 2e-5) * np.abs(exact[l]).max())


def test_anisotropic_gaussian_closed_form():
    """global line of sight: W = exp(-x_perp^2/2R_perp^2 - z^2/2R_par^2) and P = exp(-k_perp^2 s_perp^2 - k_par^2 s_par^2)
    convolve axis by axis, P̂ = prod_i (R_i^2/(s_i^2+R_i^2))^(1/2) exp(-k_i^2 s_i^2 R_i^2/(s_i^2+R_i^2)) - every Q_L
    and P_l' couple"""
    Rp, Rz, sp, sz = 60.0, 50.0, 8.0, 10.0
    s = np.linspace(0, 720, 2001)
    x, w = leggauss(64)
    leg = lambda l: (2 * l + 1) / 2 * w * eval_legendre(l, x)
    Q = {
        L: np.exp(-(s[:, None] ** 2) * ((1 - x**2) / (4 * Rp**2) + x**2 / (4 * Rz**2))) @ leg(L)
        for L in range(0, 13, 2)
    }
    ells = [0, 2, 4]
    W = SurveyWindow(s, Q, ells=ells, k=K, ic=None)
    assert W.ells_in == list(range(0, 17, 2))
    P = np.array([np.exp(-(W.k_in[:, None] ** 2) * ((1 - x**2) * sp**2 + x**2 * sz**2)) @ leg(l) for l in W.ells_in])
    a = lambda s_, R_: s_**2 * R_**2 / (s_**2 + R_**2)
    amp = Rp**2 / (sp**2 + Rp**2) * np.sqrt(Rz**2 / (sz**2 + Rz**2))
    P_hat_mu = amp * np.exp(-(K[:, None] ** 2) * ((1 - x**2) * a(sp, Rp) + x**2 * a(sz, Rz)))
    for i, l in enumerate(ells):
        exact = P_hat_mu @ leg(l)
        np.testing.assert_allclose(W(P)[i].real, exact, atol=1e-5 * np.abs(exact).max())


def gaussian_ic_windows(s_ic):
    """dMR19's W^{d,glo}_00(s, D) for the Gaussian window: int W(x)W(x+s)W(x+D) = (2pi R^2/3)^(3/2)
    exp(-(s^2+D^2-s.D)/3R^2), whose L = 0 part is exp(-(s^2+D^2)/3R^2) i_0(s D/3R^2); over int W A"""
    s, D = np.meshgrid(s_ic, s_ic, indexing="ij")
    z = s * D / (3 * R**2)
    t0 = (2 * np.pi * R**2 / 3) ** 1.5 * np.exp(-((s - D) ** 2) / (3 * R**2) - 2 * z) * spherical_in(0, z)
    return {(0, 0): t0 / ((2 * np.pi * R**2) ** 1.5 * (np.pi * R**2) ** 1.5)}


def test_global_ic_closed_form():
    """Fourier space, F = W (d - int W d / int W): A P^cic = int P |W(k-q)|^2 - 2 W(k)/W(0) int P W(k-q) W(q)
    + |W(k)|^2/W(0)^2 int P |W|^2 = T1 - 2 T2 + T3, all Gaussian. 'global' is T1 - T3 (dMR19's approximation, Beutler+17)"""
    sig = 20.0
    k = np.linspace(0.0005, 0.1, 40)
    a = sig**2 + R**2
    T1 = (R**2 / a) ** 1.5 * np.exp(-(k**2) * sig**2 * R**2 / a)
    T2 = (R**2 / a) ** 1.5 * np.exp(k**2 * R**4 / (4 * a) - k**2 * R**2)
    T3 = (R**2 / a) ** 1.5 * np.exp(-(k**2) * R**2)
    s_ic = np.linspace(0, 10 * R, 1001)
    for ic, exact in [("global", T1 - T3), (dict(s=s_ic, d_ic=gaussian_ic_windows(s_ic)), T1 - 2 * T2 + T3)]:
        W = SurveyWindow(S, Q_GAUSS, ells=[0], k=k, ells_in=[0], ic=ic)
        P_hat = W(gaussian_pk([0], W.k_in, sig))[0].real
        np.testing.assert_allclose(P_hat, exact, atol=3e-5 * np.abs(T1).max())  # -> 0 as k -> 0
        np.testing.assert_allclose(W.shot_noise[0].real, 1 - np.exp(-(k**2) * R**2), atol=1e-6)


def test_global_ic_as_kernel():
    """W^{glo,glo}_lp(s, D) = Q_l(s) Q_p(D) / int 4pi D^2 Q_0 (dMR19 eq 2.22) passed as ic_ic is 'global'"""
    Q = {0: Q_GAUSS[0], 2: 0.1 * (S / R) * Q_GAUSS[0]}
    s_ic = S[::2]
    V = 4 * np.pi * np.trapezoid(S**2 * Q[0], S)
    q = {L: np.interp(s_ic, S, QL) for L, QL in Q.items()}
    kernel = {(l, p): np.outer(q[l], q[p]) / V for l in Q for p in Q}
    P_hat = [
        SurveyWindow(S, Q, ells=[0, 2], k=K, ells_in=[0, 2], ic=ic)(gaussian_pk([0, 2], np.geomspace(1e-4, 0.55, 256)))
        for ic in ("global", dict(s=s_ic, ic_ic=kernel))
    ]
    np.testing.assert_allclose(P_hat[1], P_hat[0], atol=1e-5 * np.abs(P_hat[0]).max())


def test_ic_missing_output_multipoles_are_zero():
    s_ic = S[::8]
    kernel = gaussian_ic_windows(s_ic)
    kw = dict(ells=[0, 2], k=K, ells_in=[0, 2])
    partial = SurveyWindow(S, Q_GAUSS, ic=dict(s=s_ic, ic_ic=kernel), **kw)
    padded = SurveyWindow(S, Q_GAUSS, ic=dict(s=s_ic, ic_ic={**kernel, (2, 0): np.zeros_like(kernel[(0, 0)])}), **kw)
    P = gaussian_pk(partial.ells_in, partial.k_in)
    np.testing.assert_allclose(partial(P), padded(P), rtol=1e-13, atol=1e-15)
    no_ic = SurveyWindow(S, Q_GAUSS, ic=None, **kw)
    np.testing.assert_allclose(partial(P)[1], no_ic(P)[1], rtol=1e-13, atol=1e-15)


def test_exact_ic_needs_every_output_multipole():
    """with d_ic the global ic_ic changes sign for every l, so an l with no cross terms would get +IC^{ic,ic} alone"""
    s_ic = S[::8]
    kernel = gaussian_ic_windows(s_ic)
    padded = {**kernel, (2, 0): np.zeros_like(kernel[(0, 0)])}
    kw = dict(ells=[0, 2], k=K, ells_in=[0, 2])
    with pytest.raises(ValueError, match="d_ic windows for every output l - none for l = \\[2\\]"):
        SurveyWindow(S, Q_GAUSS, ic=dict(s=s_ic, d_ic=kernel), **kw)
    with pytest.raises(ValueError, match="ic_d windows"):
        SurveyWindow(S, Q_GAUSS, ic=dict(s=s_ic, d_ic=padded, ic_d=kernel), **kw)
    SurveyWindow(S, Q_GAUSS, ic=dict(s=s_ic, d_ic=padded), **kw)


@pytest.mark.parametrize("custom", [False, True])
def test_inferred_inputs_include_ic_multipoles(custom):
    Q = {0: Q_GAUSS[0], 1: 0.1 * S / R * Q_GAUSS[0], 4: 0.01 * (S / R) ** 4 * Q_GAUSS[0]}
    ic = "global"
    if custom:
        s_ic = S[::8]
        V = 4 * np.pi * np.trapezoid(S**2 * Q[0], S)
        ic = dict(s=s_ic, ic_ic={(1, 4): np.outer(Q[1][::8], Q[4][::8]) / V})
    dipole = SurveyWindow(S, Q, ells=[1], k=K, ic=ic)
    both = SurveyWindow(S, Q, ells=[0, 1], k=K, ic=ic)
    assert 4 in dipole.ells_in

    def P(W):
        return np.array([1e4 * W.k_in**4 * np.exp(-((SIG * W.k_in) ** 2)) if l == 4 else 0 * W.k_in for l in W.ells_in])

    np.testing.assert_allclose(dipole(P(dipole))[0], both(P(both))[1], rtol=1e-12, atol=1e-15)
    assert np.max(np.abs(dipole(P(dipole)))) > 1e-10


def test_ic_removes_ir_sensitivity():
    """P_0 ~ k^-3 (the potential terms) windows to a log in k_in[0]; the IC takes it out (a constant in xi_0 windows
    to const x Q_l, which it removes)"""
    s = np.linspace(0, 1000, 2001)
    Q = {0: np.clip(1 - 1.5 * s / 1000 + 0.5 * (s / 1000) ** 3, 0, None)}  # a sphere of radius 500
    k = np.linspace(0.005, 0.1, 20)

    def P_hat(k_min, ic):
        W = SurveyWindow(s, Q, ells=[0], k=k, ells_in=[0], k_in=np.geomspace(k_min, 0.35, 300), ic=ic)
        return W((W.k_in**-3.0 * np.exp(-((W.k_in / 0.05) ** 2)))[None])[0].real

    np.testing.assert_allclose(P_hat(1e-5, "global"), P_hat(1e-6, "global"), rtol=1e-4)
    assert np.all(np.abs(P_hat(1e-5, None) / P_hat(1e-6, None) - 1) > 1e-3)


def test_wide_angle_orders():
    """an order without its own Q^(n) uses Q^(0) - the same as Q^(1) = (s/d) Q^(0) given, or adding it to order 0;
    and twice that window is twice its response"""
    d = 2000.0
    Q1 = {(0, 0): Q_GAUSS[0], (0, 1): S / d * Q_GAUSS[0]}
    kw = dict(ells=[0], k=K, ells_in=[0], ic="global", d=d)
    P0, P1 = gaussian_pk([0], np.geomspace(1e-4, 0.55, 256)), 0.3 * gaussian_pk([0], np.geomspace(1e-4, 0.55, 256), 8.0)
    plain = SurveyWindow(S, Q_GAUSS, **kw)
    own = SurveyWindow(S, Q1, **kw)
    double = SurveyWindow(S, {**Q1, (0, 1): 2 * Q1[(0, 1)]}, **kw)
    np.testing.assert_allclose(plain({0: P0, 1: P1}), plain(P0 + P1), rtol=1e-12)
    np.testing.assert_allclose(own({0: P0, 1: P1}), plain(P0 + P1), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        double({0: P0, 1: P1}) - own({0: P0, 1: P1}), own({1: P1}) - own({0: 0 * P1}), rtol=1e-12
    )


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("start", [0.0, 5.0])
@pytest.mark.parametrize("ic", [None, "global"])
def test_wide_angle_xi_between_window_nodes(n, start, ic):
    s = np.arange(start, 610.0, 10.0)
    q = np.exp(-((s / R) ** 2))
    d = 2000.0
    r = np.array([0.0, 1.0, 5.0, 15.0, 55.0, 700.0])
    kw = dict(ells=[0], ells_in=[0], k=K, d=d, ic=ic)
    plain = SurveyWindow(s, {0: q}, **kw)
    own = SurveyWindow(s, {(0, 0): q, (0, n): (s / d) ** n * q}, **kw)
    P = {0: gaussian_pk([0], own.k_in), n: 0.3 * gaussian_pk([0], own.k_in, 8.0)}
    xi = {0: np.exp(-((r / 40.0) ** 2))[None], n: 0.3 * np.exp(-((r / 60.0) ** 2))[None]}
    np.testing.assert_allclose(own(P), plain(P), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(own.xi(xi, r, P), plain.xi(xi, r, P), rtol=1e-12, atol=1e-14)


def gaussian_xi(ells, r):
    """xi_l of P_l = (-i)^l k^l e^(-sig^2 k^2), as test_hankel"""
    return np.array(
        [
            np.sqrt(np.pi) * r**l * np.exp(-(r**2) / (4 * SIG**2)) / (2 ** (l + 2) * SIG ** (2 * l + 3) * 2 * np.pi**2)
            for l in ells
        ]
    )


def test_xi_is_the_hankel_pair():
    """window.xi, IC and all, transforms to the window's P̂ - xi of P_l = (-i)^l k^l e^(-sig^2 k^2) as test_hankel"""
    Q = {0: Q_GAUSS[0], 2: -0.2 * (S / R) * Q_GAUSS[0]}
    W = SurveyWindow(S, Q, ells=[0, 1, 2], k=K, ells_in=[0, 1, 2, 3, 4], k_in=np.geomspace(1e-5, 0.55, 300))
    xi = gaussian_xi(W.ells_in, S)
    P = gaussian_pk(W.ells_in, W.k_in)
    xi_hat = W.xi(xi, P=P)
    no_ic = SurveyWindow(S, Q, ells=[0, 1, 2], k=K, ells_in=[0, 1, 2, 3, 4], ic=None)
    np.testing.assert_allclose(no_ic.xi(xi), windowed_xi(dict(zip(W.ells_in, xi)), Q, W.ells), rtol=1e-14)
    P_hat = W(P)
    for i, l in enumerate(W.ells):
        np.testing.assert_allclose(hankel.xi_to_pk(S, xi_hat[i], l, K), P_hat[i], atol=1e-5 * np.abs(P_hat[i]).max())
    with pytest.raises(ValueError, match="needs P"):
        W.xi(xi)


@pytest.mark.parametrize("k_split", [0.01, 0.05])
def test_xi_split_adds_up(k_split):
    """xi's part above k_split from a converged transform, and P's below through the window's quadrature, is all of xi"""
    Q = {0: Q_GAUSS[0], 2: -0.2 * (S / R) * Q_GAUSS[0]}
    W = SurveyWindow(S, Q, ells=[0, 1, 2], k=K, ells_in=[0, 1, 2, 3, 4], k_in=np.geomspace(1e-5, 0.55, 300))
    r = np.array([1.0, 10.0, 30.0, 100.0, 300.0])
    k = np.geomspace(1e-5, 2.0, 2**14)
    high = 1 - hankel.window(k, k_split, 2 * k_split)
    xi_high = np.array(
        [hankel.pk_to_xi(k, p * high, l, r, k_ir=None) for l, p in zip(W.ells_in, gaussian_pk(W.ells_in, k))]
    )
    P = gaussian_pk(W.ells_in, W.k_in)
    whole = W.xi(gaussian_xi(W.ells_in, r), r, P)
    np.testing.assert_allclose(W.xi(xi_high, r, P, k_split), whole, atol=1e-7 * np.abs(whole).max())
    with pytest.raises(ValueError, match="k_split"):
        W.xi(xi_high, r, P, 0.3)


def test_from_matrix():
    """a matrix acting on P directly - odd multipoles imaginary, as CosmoWAP's - gives the same P̂"""
    W = SurveyWindow(S, {0: Q_GAUSS[0], 2: 0.1 * Q_GAUSS[0]}, ells=[0, 1, 2], k=K, ells_in=[0, 1, 2, 3], ic=None)
    phase = np.outer(np.repeat((-1j) ** np.array(W.ells), len(K)), np.repeat(1j ** np.array(W.ells_in), len(W.k_in)))
    ext = SurveyWindow.from_matrix(phase * W.matrix, W.k_in, W.ells_in, K, W.ells)
    P = gaussian_pk(W.ells_in, W.k_in)
    np.testing.assert_allclose(ext(P), W(P), rtol=1e-12)
    np.testing.assert_allclose(ext(P).ravel(), (phase * W.matrix) @ P.ravel(), rtol=1e-12)


def test_bad_set_up():
    with pytest.raises(ValueError, match="one of k"):
        SurveyWindow(S, Q_GAUSS)
    with pytest.raises(ValueError, match="Q_0"):
        SurveyWindow(S, {2: Q_GAUSS[0]}, k=K)
    with pytest.raises(ValueError, match="pass s and d"):
        SurveyWindow(S, {(0, 0): Q_GAUSS[0], (0, 1): Q_GAUSS[0]}, k=K)
    with pytest.raises(ValueError, match="ells_in"):
        SurveyWindow(S, Q_GAUSS, ells=[0], k=K, ells_in=[0], ic=dict(s=S[::4], ic_ic={(0, 2): np.eye(len(S[::4]))}))
