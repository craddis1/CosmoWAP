"""cosmo_wap.Model: kernels per field, the wide-separation terms on top, pk / pk_mu / xi and bk / bk_mu."""

from types import SimpleNamespace

import numpy as np
import pytest

import cosmo_wap as cw
import cosmo_wap.bk as bk
import cosmo_wap.pk as pk
from cosmo_wap.lib import hankel, utils
from cosmo_wap.lib.survey_window import SurveyWindow, windowed_xi
from cosmo_wap.lib.utils import CachedSpline
from cosmo_wap.model import WS_GROUPS, WS_ORDER
from cosmo_wap.numeric_mu import bk as numeric_bk
from cosmo_wap.numeric_mu import pk as numeric_pk

K = np.geomspace(1e-3, 0.3, 40)
ZZ = 1.0


@pytest.fixture(scope="module")
def multi_tracer(cosmo):
    """bright x faint and faint x bright - the same pair in both orders"""
    sp = cw.SurveyParams.Euclid(cosmo)
    sp.BF_split(5e-16)
    return (
        cw.ClassWAP(cosmo, [sp.bright, sp.faint], verbose=False),
        cw.ClassWAP(cosmo, [sp.faint, sp.bright], verbose=False),
    )


# ---------------------------------------------------------------------------
# set-up: kernels and wide_separation
# ---------------------------------------------------------------------------


def test_defaults(cosmo_funcs):
    m = cw.Model(cosmo_funcs)
    assert m.kernels == m.kernels2 == m.kernels3 == ["N"] and m.wide_separation == ()
    assert m.t == 0 and (m.r, m.s) == (0, 0)  # end points: x1 for pk, x3 for bk


def test_wide_separation_resolution(cosmo_funcs):
    ws = lambda kernels, value: cw.Model(cosmo_funcs, kernels=kernels, wide_separation=value).wide_separation
    assert ws("N", True) == WS_GROUPS["WS"]
    assert ws(["N", "LP"], True) == WS_GROUPS["WSGR"]
    assert ws("N", "WA") == ("WA1", "WA2")
    assert ws("N", "WS2") == ("WA2", "RR2", "WARR")
    assert ws("N", ["WS", "WA1"]) == WS_GROUPS["WS"]  # each block once
    assert ws("N", ["RR1", "WA1"]) == ("WA1", "RR1")  # canonical order
    assert ws("N", False) == ws("N", None) == ws("N", []) == ()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(wide_separation="WAX"), "Unknown wide_separation"),
        (dict(kernels="N", kernels2=["N", "LP"], wide_separation=True), "same kernels on every field"),
        (dict(kernels="N", kernels3=["N", "LP"], wide_separation=True), "same kernels on every field"),
        (dict(kernels="LP", wide_separation="WS"), "needs 'N'"),
        (dict(kernels="N", wide_separation="WAGR"), "need 'LP'"),
        (dict(kernels="Q"), "Unknown kernels"),
        (dict(kernels=[]), "kernels is empty"),
        (dict(kernels2=[]), "kernels2 is empty"),
        (dict(kernels3=[]), "kernels3 is empty"),
    ],
)
def test_bad_set_up(cosmo_funcs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cw.Model(cosmo_funcs, **kwargs)


# ---------------------------------------------------------------------------
# pk and pk_mu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [None, 4.0])
def test_pk_is_pk_func(cosmo_funcs, sigma):
    """the kernels path is pk_func's (both fields the same) and the WS terms - their full angular polynomials, so
    every l up to 6 - are summed on top, exactly; t = 0.2 so the odd WS multipoles are not zero for one tracer"""
    ells = [0, 1, 2, 3, 4]
    ref = pk.pk_func(None, ells, cosmo_funcs, K, zz=ZZ, kernels=["N", "LP"], sigma=sigma)
    P = cw.Model(cosmo_funcs, kernels=["N", "LP"], sigma=sigma).pk(K, ells, ZZ)
    np.testing.assert_allclose(P, ref, rtol=1e-12, atol=1e-14 * np.max(np.abs(ref)))  # odd l: zero up to rounding
    ws_ells = list(range(7))
    ws = pk.pk_func(list(WS_GROUPS["WSGR"]), ws_ells, cosmo_funcs, K, zz=ZZ, t=0.2, sigma=sigma)
    m, m_ws = (cw.Model(cosmo_funcs, kernels=["N", "LP"], wide_separation=w, t=0.2, sigma=sigma) for w in (False, True))
    P = m_ws.pk(K, ws_ells, ZZ) - m.pk(K, ws_ells, ZZ)
    np.testing.assert_allclose(P, ws, rtol=1e-9, atol=1e-12 * np.max(np.abs(ws)))
    assert cw.Model(cosmo_funcs).pk(K, 2, ZZ).shape == K.shape


def test_pieces_add_up(cosmo_funcs):
    """P is bilinear in the fields' kernels - the kernel-pair pieces sum to the total"""
    ells = [0, 1, 2]
    total = cw.Model(cosmo_funcs, kernels=["N", "LP"]).pk(K, ells, ZZ)
    pieces = sum(cw.Model(cosmo_funcs, kernels=a, kernels2=b).pk(K, ells, ZZ) for a in ("N", "LP") for b in ("N", "LP"))
    np.testing.assert_allclose(pieces, total, rtol=1e-10, atol=1e-12 * np.max(np.abs(total)))


@pytest.mark.parametrize("ell", range(4))
def test_swapping_fields_flips_odd_multipoles(cosmo_funcs, ell):
    NL = cw.Model(cosmo_funcs, kernels="N", kernels2="LP").pk(K, ell, ZZ)
    LN = cw.Model(cosmo_funcs, kernels="LP", kernels2="N").pk(K, ell, ZZ)
    np.testing.assert_allclose(NL, (-1) ** ell * LN, rtol=1e-10, atol=1e-14 * np.max(np.abs(NL)))


def test_single_tracer_midpoint_has_no_odd_multipoles(cosmo_funcs):
    kw = dict(kernels=["N", "LP"], wide_separation=True)
    P0 = np.max(np.abs(cw.Model(cosmo_funcs, t=0.5, **kw).pk(K, 0, ZZ)))
    assert np.max(np.abs(cw.Model(cosmo_funcs, t=0.5, **kw).pk(K, [1, 3], ZZ))) < 1e-10 * P0
    assert np.max(np.abs(cw.Model(cosmo_funcs, t=0, **kw).pk(K, 1, ZZ))) > 1e-6 * P0  # but there are at x1


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_tracer_swap_is_t_to_one_minus_t(multi_tracer, ell):
    """AB at t is (-1)^l BA at 1 - t: the same pair seen from the other end"""
    ab, ba = multi_tracer
    kw = dict(kernels=["N", "LP"], wide_separation="WS")
    P_ab = cw.Model(ab, t=0.3, **kw).pk(K, ell, ZZ)
    P_ba = cw.Model(ba, t=0.7, **kw).pk(K, ell, ZZ)
    np.testing.assert_allclose(P_ab, (-1) ** ell * P_ba, rtol=1e-9, atol=1e-12 * np.max(np.abs(P_ab)))


def test_pk_mu_projects_onto_pk(cosmo_funcs):
    """on get_multipoles' own mu nodes the projection of pk_mu is pk exactly"""
    m = cw.Model(cosmo_funcs, kernels=["N", "LP"], wide_separation=True)
    mu, w = numeric_pk.get_mu_grid(48)
    Pmu = m.pk_mu(K, mu, ZZ)
    assert Pmu.shape == (len(K), len(mu))
    ells = [0, 1, 2, 3, 4]
    proj = np.array([numeric_pk.project_multipole(Pmu, mu, w, l, K) for l in ells])
    P = m.pk(K, ells, ZZ)
    np.testing.assert_allclose(proj, P, rtol=1e-9, atol=1e-12 * np.max(np.abs(P)))


# ---------------------------------------------------------------------------
# bk and bk_mu
# ---------------------------------------------------------------------------

K1, K2, TH = np.array([0.02, 0.05, 0.08]), np.array([0.015, 0.04, 0.05]), np.array([2.0, 1.2, 2.6])
LM = [(l, m) for l in range(5) for m in range(l + 1)]


def _bk(model, lm=LM):
    return model.bk(K1, K2, l=lm, zz=ZZ, theta=TH)


@pytest.mark.parametrize("sigma", [None, 4.0])
def test_bk_kernels_are_get_multipoles(cosmo_funcs, sigma):
    """each field's kernels go to its place in numeric_mu.bk, for every (l, m); a single l takes m, and k3 does as
    well as theta"""
    kernels = (["N"], ["N", "LP"], ["LP"])
    ref = numeric_bk.get_multipoles(*kernels, LM, cosmo_funcs, K1, K2, theta=TH, zz=ZZ, sigma=sigma)
    m = cw.Model(cosmo_funcs, kernels=kernels[0], kernels2=kernels[1], kernels3=kernels[2], sigma=sigma)
    np.testing.assert_allclose(_bk(m), ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())
    k3 = utils.get_k3(TH, K1, K2)
    B21 = m.bk(K1, K2, k3, l=2, zz=ZZ, m=1)
    assert B21.shape == K1.shape
    np.testing.assert_allclose(B21, ref[LM.index((2, 1))], rtol=1e-10)


def test_bk_pieces_add_up(cosmo_funcs):
    """B is linear in each field's kernels - each field carries one Z1 or Z2 in every term - so the 8 pieces sum to
    the total"""
    lm = [(0, 0), (1, 0), (2, 1), (3, 0)]
    total = _bk(cw.Model(cosmo_funcs, kernels=["N", "LP"]), lm)
    kerns = ("N", "LP")
    pieces = sum(
        _bk(cw.Model(cosmo_funcs, kernels=a, kernels2=b, kernels3=c), lm) for a in kerns for b in kerns for c in kerns
    )
    np.testing.assert_allclose(pieces, total, rtol=1e-10, atol=1e-12 * np.abs(total).max())


@pytest.mark.parametrize("r, s", [(0, 0), (0.3, 0.2)])
def test_bk_wide_separation_is_exact(cosmo_funcs, r, s):
    """the WS blocks' angle-dependent forms on the (mu, phi) grid - degree 8 in the LOS direction, as the kernels, so
    [8, 8] projects every (l, m) to l = 4 exactly - against their exact ylm; r, s move only them"""
    kw = dict(kernels=["N", "LP"], r=r, s=s, bk_mu_grid=(8, 8))
    B = _bk(cw.Model(cosmo_funcs, wide_separation=True, **kw)) - _bk(cw.Model(cosmo_funcs, **kw))
    ref = np.array(
        [
            sum(getattr(bk, b).ylm(l, m, cosmo_funcs, K1, K2, None, TH, ZZ, r, s) for b in WS_GROUPS["WSGR"])
            for l, m in LM
        ]
    )
    np.testing.assert_allclose(B, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())


def test_bk_fog_damps_wide_separation_too(cosmo_funcs):
    """with FoG the WS terms are damped as the kernels, on the same grid - the default [16, 16] against
    angular_integrate's n = 32 (its Gauss-Legendre phi needs more nodes than the midpoint rule here)"""
    lm = [(0, 0), (2, 0), (4, 0), (2, 1), (4, 4)]
    kw = dict(kernels=["N", "LP"], sigma=4.0)
    B = _bk(cw.Model(cosmo_funcs, wide_separation=True, **kw), lm) - _bk(cw.Model(cosmo_funcs, **kw), lm)
    ref = np.array(
        [
            sum(getattr(bk, b).ylm(l, m, cosmo_funcs, K1, K2, None, TH, ZZ, sigma=4.0, n=32) for b in WS_GROUPS["WSGR"])
            for l, m in lm
        ]
    )
    np.testing.assert_allclose(B, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())


def test_bk_mu_projects_onto_bk(cosmo_funcs):
    """on get_multipoles' own (mu, phi) nodes the projection of bk_mu - FoG already in it - is bk exactly"""
    m = cw.Model(cosmo_funcs, kernels=["N", "LP"], wide_separation=True, sigma=4.0)
    mu, phi, w = numeric_bk.get_mu_phi_grid(16, 16)
    Bmu = m.bk_mu(K1, K2, None, mu, phi, ZZ, theta=TH)
    assert Bmu.shape == (len(K1), len(mu), len(phi))
    tri = utils.enable_broadcasting(K1, K2, utils.get_k3(TH, K1, K2), TH)
    proj = np.array([numeric_bk.project_multipole(Bmu, mu, phi, w, l, mm, *tri) for l, mm in LM])
    B = _bk(m)
    np.testing.assert_allclose(proj, B, rtol=1e-10, atol=1e-12 * np.abs(B).max())


def test_bk_needs_second_order_kernels(cosmo_funcs):
    """field 3 is the bispectrum's alone - an integrated kernel there leaves pk alone and bk raises"""
    m = cw.Model(cosmo_funcs, kernels="N", kernels3="L")
    np.testing.assert_array_equal(m.pk(K, [0, 2], ZZ), cw.Model(cosmo_funcs).pk(K, [0, 2], ZZ))
    with pytest.raises(NotImplementedError, match="second order kernel"):
        _bk(m)


# ---------------------------------------------------------------------------
# xi
# ---------------------------------------------------------------------------

R = np.array([20.0, 50.0, 100.0, 200.0])


def test_xi_coarse_grid_matches_fine(cosmo_funcs):
    m = cw.Model(cosmo_funcs, wide_separation=True)
    np.testing.assert_allclose(m.xi(R, [0, 2], ZZ), m.xi(R, [0, 2], ZZ, n_kc=None), rtol=1e-5)


def test_xi_odd_is_real_and_follows_pk(multi_tracer):
    """xi_l = Re[i^l int k^2 P_l j_l] - a real dipole that flips when the fields swap ends, like P_1. Swapping
    fields means swapping the tracer with its kernels: N_A x LP_B from either end"""
    ab, ba = multi_tracer
    x1 = cw.Model(ab, kernels="N", kernels2="LP").xi(R, 1, ZZ)
    x1_swap = cw.Model(ba, kernels="LP", kernels2="N").xi(R, 1, ZZ)
    assert np.isrealobj(x1) and np.all(np.abs(x1) > 0)
    np.testing.assert_allclose(x1, -x1_swap, rtol=1e-8)


def test_xi_bins(cosmo_funcs):
    m = cw.Model(cosmo_funcs)
    edges = np.sort(np.concatenate([R - 0.01, R + 0.01]))
    np.testing.assert_allclose(m.xi(r_edges=edges, l=[0, 2], zz=ZZ)[:, ::2], m.xi(R, [0, 2], ZZ), rtol=1e-6)
    assert m.xi(r_edges=np.arange(10.0, 110.0, 10.0), l=0, zz=ZZ).shape == (9,)


# COFFE (v3.0.1) flat-sky Kaiser xi_l at zbar = 0.5, r = 20, 50, 100, 200 Mpc/h, for tracer A below in the default
# get_cosmo() cosmology with P(k) tapered by exp(-(k/2 h/Mpc)^2) in both codes - from the COFFE comparison of the
# code paper (wap_paper/validation/coffe). The codes' backgrounds differ by ~1e-4, hence 0.05%.
COFFE_FLAT = {
    0: [0.1063686, 0.0090131933, 0.0019861371, -0.00017383753],
    2: [-0.11199454, -0.023614304, -0.0037312237, -0.00069708144],
    4: [0.0090466863, 0.0025600203, 0.00076333611, 0.00014649829],
}


@pytest.fixture(scope="module")
def coffe_tracer():
    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.InitNew(cosmo).update(
        z_range=[0.05, 3.0],
        b_1=lambda z: 0.9 + 0.4 * z,
        Q=lambda z: 0.583 + 2.02 * z - 0.568 * z**2 + 0.0411 * z**3,
        be=lambda z: -7.29 + 0.470 * z + 1.17 * z**2 - 0.290 * z**3,
    )
    cf = cw.ClassWAP(cosmo, survey, verbose=False, fast=True)
    k = np.logspace(-5, np.log10(cf.K_MAX), 4000)
    cf.Pk = CachedSpline(k, cf.Pk(k) * np.exp(-((k / 2.0) ** 2)))
    cf.Pk_d, cf.Pk_dd = cf.Pk.derivative(nu=1), cf.Pk.derivative(nu=2)
    return cf


def test_xi_against_coffe(coffe_tracer):
    xi = cw.Model(coffe_tracer).xi(R, [0, 2, 4], 0.5)
    for i, ell in enumerate((0, 2, 4)):
        np.testing.assert_allclose(xi[i], COFFE_FLAT[ell], rtol=5e-4)


# ---------------------------------------------------------------------------
# wide-angle orders and the survey window (lib/survey_window.py)
# ---------------------------------------------------------------------------

S_W = np.linspace(0, 3000, 1501)
Q_W = {0: np.exp(-(S_W**2) / (4 * 250.0**2)), 2: 0.05 * S_W / 250.0 * np.exp(-(S_W**2) / (4 * 250.0**2))}


def test_by_order_adds_up(cosmo_funcs):
    """the orders sum to pk and xi, and order n is the blocks with 1/d^n"""
    ells = [0, 1, 2]
    kw = dict(kernels=["N", "LP"], t=0.2)
    m = cw.Model(cosmo_funcs, wide_separation=True, **kw)
    parts = m.pk(K, ells, ZZ, by_order=True)
    assert sorted(parts) == [0, 1, 2]
    np.testing.assert_allclose(sum(parts.values()), m.pk(K, ells, ZZ), rtol=1e-12)
    first = [b for b in m.wide_separation if WS_ORDER[b] == 1]
    P1 = cw.Model(cosmo_funcs, wide_separation=first, **kw).pk(K, ells, ZZ) - cw.Model(cosmo_funcs, **kw).pk(
        K, ells, ZZ
    )
    np.testing.assert_allclose(parts[1], P1, rtol=1e-9, atol=1e-12 * np.max(np.abs(P1)))
    xi = m.xi(R, ells, ZZ, by_order=True)
    np.testing.assert_allclose(sum(xi.values()), m.xi(R, ells, ZZ), rtol=1e-12)


def test_pk_windowed(cosmo_funcs):
    """pk through a window is the window on its P, and by order adds up to it; an order with Q^(n) = (s/d)^n Q^(0) is
    the same as none; k and l must be the window's"""
    d = cosmo_funcs.comoving_dist(ZZ)
    m = cw.Model(cosmo_funcs, wide_separation=True, t=0.0)
    ells = [0, 1, 2]
    W = SurveyWindow(S_W, Q_W, ells=ells, k=K)
    P_hat = m.pk(K, ells, ZZ, window=W)
    np.testing.assert_allclose(P_hat, W(m.pk(W.k_in, W.ells_in, ZZ)), rtol=1e-12)
    parts = m.pk(K, ells, ZZ, by_order=True, window=W)
    assert sorted(parts) == [0, 1, 2]
    np.testing.assert_allclose(sum(parts.values()), P_hat, rtol=1e-12, atol=1e-14 * np.abs(P_hat).max())
    W1 = SurveyWindow(
        S_W,
        {**{(L, 0): q for L, q in Q_W.items()}, **{(L, 1): S_W / d * q for L, q in Q_W.items()}},
        d=d,
        ells=ells,
        k=K,
    )
    np.testing.assert_allclose(m.pk(K, ells, ZZ, window=W1), P_hat, rtol=1e-10)
    with pytest.raises(ValueError, match="window.d"):
        m.pk(K, ells, 0.5, window=W1)
    with pytest.raises(ValueError, match="ells"):
        m.pk(K, [0, 2], ZZ, window=W)
    with pytest.raises(ValueError, match="window.k"):
        m.pk(K[:-1], ells, ZZ, window=W)


def test_radial_window_not_twice(cosmo_funcs):
    """a survey window with the bin's radial window (dr, k_f) needs box - the triangle again would count the bin's
    pair counts twice. Local kernels never see (dr, k_f), so they pass"""
    W = SurveyWindow(S_W, Q_W, ells=[0, 2], k=K, ic=None)
    bin_window = (124.0, 5e-4)
    m = cw.Model(cosmo_funcs, kernels=["N", "L"], window=bin_window)
    with pytest.raises(ValueError, match="box=1"):
        m.pk(K, [0, 2], ZZ, window=W)
    with pytest.raises(ValueError, match="box=1"):
        m.xi(R, [0, 2], ZZ, window=W)
    with pytest.raises(ValueError, match="box=1"):  # the integrated field on either side
        cw.Model(cosmo_funcs, kernels=["N"], kernels2=["L"], window=bin_window).pk(K, [0, 2], ZZ, window=W)
    assert np.all(
        np.isfinite(cw.Model(cosmo_funcs, kernels=["N", "L"], window=bin_window, box=1).pk(K, [0, 2], ZZ, window=W))
    )
    local = cw.Model(cosmo_funcs, window=bin_window).pk(K, [0, 2], ZZ, window=W)
    np.testing.assert_array_equal(local, cw.Model(cosmo_funcs).pk(K, [0, 2], ZZ, window=W))


def test_xi_windowed(cosmo_funcs):
    """windowed_xi of the converged xi - only its IR end goes through the window's quadrature, not the cut at
    k_in[-1] - and the global IC takes off (P̂_0(0)/V) Q_l(r)"""
    m = cw.Model(cosmo_funcs)
    kw = dict(ells=[0, 2], k=K)
    W = SurveyWindow(S_W, Q_W, ic=None, **kw)
    plain = m.xi(R, [0, 2], ZZ, window=W)
    Q_r = {L: np.interp(R, S_W, q) for L, q in Q_W.items()}
    ref = windowed_xi(dict(zip(W.ells_in, m.xi(R, W.ells_in, ZZ))), Q_r, [0, 2])
    np.testing.assert_allclose(plain, ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())
    W_ic = SurveyWindow(S_W, Q_W, **kw)
    shift = (plain - m.xi(R, [0, 2], ZZ, window=W_ic)) / np.array([Q_r[0], Q_r[2]])
    np.testing.assert_allclose(shift, shift[0, 0], rtol=1e-10)
    ws = cw.Model(cosmo_funcs, wide_separation=True)
    parts = ws.xi(R, [0, 2], ZZ, by_order=True, window=W_ic)
    total = ws.xi(R, [0, 2], ZZ, window=W_ic)
    np.testing.assert_allclose(sum(parts.values()), total, rtol=1e-12, atol=1e-14 * np.abs(total).max())
    with pytest.raises(ValueError, match="points"):
        m.xi(None, [0, 2], ZZ, r_edges=[20.0, 30.0], window=W)


@pytest.mark.parametrize("custom_ic", [False, True])
def test_xi_windowed_ir_is_finite_and_pairs_with_pk(monkeypatch, custom_ic):
    m = cw.Model(SimpleNamespace(K_MAX=1.0))

    def pk(k, l=0, zz=0, by_order=False):
        ells = [l] if np.isscalar(l) else l
        P = np.array([k**-3 * np.exp(-((k / 0.05) ** 2)) if ell == 0 else 0 * k for ell in ells])
        return {0: P} if by_order else P

    monkeypatch.setattr(m, "pk", pk)
    s = np.linspace(0, 800, 801)
    Q = {0: np.exp(-((s / 150.0) ** 2)), 2: 0.1 * s / 150.0 * np.exp(-((s / 150.0) ** 2))}
    ic = "global"
    if custom_ic:
        V = 4 * np.pi * np.trapezoid(s**2 * Q[0], s)
        ic = dict(s=s, ic_ic={(l, 0): np.outer(q, Q[0]) / V for l, q in Q.items()})
    values = []
    for k_min in (1e-4, 1e-6):
        W = SurveyWindow(s, Q, ells=[0, 2], k=[0.005, 0.01, 0.03], k_in=np.geomspace(k_min, 0.8, 400), ds=1.0, ic=ic)
        xi = m.xi(s, [0, 2], window=W, n_k=2**12)
        values.append(xi[:, [10, 100, 300]])
        P = W(m.pk(W.k_in, W.ells_in))
        for i, ell in enumerate(W.ells):
            np.testing.assert_allclose(hankel.xi_to_pk(s, xi[i], ell, W.k), P[i], atol=2e-5 * np.max(np.abs(P[i])))
        assert abs(np.trapezoid(s**2 * xi[0], s)) < 2e-5 * np.trapezoid(s**2 * np.abs(xi[0]), s)
    np.testing.assert_allclose(values[0], values[1], rtol=2e-4, atol=1e-7)
