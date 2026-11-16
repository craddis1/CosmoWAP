"""Composite terms (pk/bk/bk_mt WS, WSGR, Full, ...) built by lib.utils.composite: each multipole is the sum of the
parts of its parity, a part's missing multipole is projected from its full angular form, and where that can't be done
the composite has no such multipole rather than a silent zero."""

import numpy as np
from scipy.special import eval_legendre

from cosmo_wap import bk, bk_mt, pk
from cosmo_wap.lib.angular_integrate import ylm

K = np.array([0.003, 0.02, 0.1])
TRI = dict(k1=np.array([0.02, 0.05, 0.08]), k2=np.array([0.015, 0.04, 0.05]), theta=np.array([2.0, 1.2, 2.6]))
PK_PARTS = ["WA1", "RR1", "WA2", "WARR", "RR2", "WAGR", "RRGR", "GR1", "GR2"]


def test_pk_full_is_its_mu_polynomial(cosmo_funcs):
    """every l to 6, with and without FoG (sigma in the closed forms' convention, exp(-(k mu sigma)^2/2)) - GR2's l4 and
    GR1/GR2's l5/l6 come from projecting their mu, so the damped GR2 l4 is in (its add_empty_methods l4 is 0)"""
    x, w = np.polynomial.legendre.leggauss(64)
    P = np.array([sum(getattr(pk, n).mu(xi, cosmo_funcs, K, 1.0, 0.3) for n in PK_PARTS) for xi in x])
    for sigma in (None, 4.0):
        damp = 1 if sigma is None else np.exp(-((K[None, :] * x[:, None] * sigma) ** 2) / 2)
        for l in range(7):
            want = (2 * l + 1) / 2 * np.sum(w[:, None] * P * damp * eval_legendre(l, x)[:, None], axis=0)
            got = getattr(pk.Full, f"l{l}")(cosmo_funcs, K, 1.0, 0.3, sigma)
            # 1e-6: GR2.l2's erf closed form with FoG is good to ~3e-7 here
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6 * np.abs(P).max())


def test_pk_parity_and_sums(cosmo_funcs):
    np.testing.assert_array_equal(pk.WS_GR.l3(cosmo_funcs, K, 1.0, 0.3), np.zeros_like(K))
    want = sum(getattr(pk, n).l4(cosmo_funcs, K, 1.0, 0.3) for n in ("WA2", "WARR", "RR2", "WAGR", "RRGR"))
    np.testing.assert_allclose(pk.WSGR.l4(cosmo_funcs, K, 1.0, 0.3), want, rtol=1e-14)


def test_bk_full_l4_has_gr2(cosmo_funcs):
    """bk.GR2 has no analytic l4 - Full takes it from GR2's ylm (16 nodes, ~1e-7 at m = 0)"""
    args = (cosmo_funcs, TRI["k1"], TRI["k2"], None, TRI["theta"], 1.0, 0.3, 0.2)
    gr2 = np.squeeze(ylm(bk.GR_2, 4, 0, *args, n=32))
    np.testing.assert_allclose(bk.Full.l4(*args), bk.WSGR.l4(*args) + gr2, rtol=1e-6)


def test_bk_m_multipoles(cosmo_funcs):
    """m > 0 where every part has one exactly - the WS terms (bk.WS_) and GR1's analytic l1m1, l3m*; GR2 has none and
    its 16-node ylm is a few % off at l4m4, so Full has no even m > 0"""
    args = (cosmo_funcs, TRI["k1"], TRI["k2"], None, TRI["theta"], 1.0, 0.3, 0.2)
    want = sum(getattr(bk, n).l2m1(*args) for n in ("WA2", "WARR", "RR2"))
    np.testing.assert_allclose(bk.WS.l2m1(*args), want, rtol=1e-14)
    want = sum(getattr(bk, n).l3m2(*args) for n in ("WA1", "RR1", "GR1"))
    np.testing.assert_allclose(bk.Full.l3m2(*args), want, rtol=1e-14)
    assert hasattr(bk.WSGR, "l4m4") and not hasattr(bk.Full, "l2m1")


def test_bk_mt_full_has_no_l4():
    """bk_mt.GR2's l4 is only a stand-in (the export never had it) and there is no multi-tracer GR2 ylm - so Full stops
    at l3, m = 0, and has no ylm (no FoG)"""
    assert all(hasattr(bk_mt.Full, f"l{l}") for l in range(4))
    assert not hasattr(bk_mt.Full, "l4") and not hasattr(bk_mt.Full, "l1m1") and not hasattr(bk_mt.Full, "ylm")
    assert hasattr(bk_mt.WSGR, "l4m4") and hasattr(bk_mt.WSGR, "ylm")
