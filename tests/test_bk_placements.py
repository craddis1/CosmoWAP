"""all_tracer=8 - every tracer placement on the bispectrum legs, see covariances.bk_placements.

The views put tracer i on leg i, so a placement is a relabelling of another's legs; on equal sides placements
coincide and the covariance must be exactly degenerate there for the pseudoinverse to drop the copies.
"""

import numpy as np
import pytest

import cosmo_wap.bk as bk
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import contract
from cosmo_wap.forecast.covariances import bk_placements
from cosmo_wap.lib import utils

K = ["N", "LP"]
P8 = bk_placements(8)


@pytest.fixture(scope="module")
def mt_forecasts(forecast_mt):
    cf = forecast_mt.cosmo_funcs
    return {a: FullForecast(cf, kmax_func=0.05, s_k=2, N_bins=2, all_m=a) for a in (False, True)}


@pytest.fixture(scope="module")
def bins(mt_forecasts):
    return {(n, a): F.get_bk_bin(0, all_tracer=n) for a, F in mt_forecasts.items() for n in (True, 8)}


def test_placements():
    assert bk_placements(False) == [(0, 0, 0)]
    assert bk_placements(True) == bk_placements(4) == [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
    assert sorted(P8) == sorted(set(P8)) and len(P8) == 8 and set(bk_placements(4)) <= set(P8)
    with pytest.raises(ValueError):
        bk_placements(3)


@pytest.mark.parametrize("term", [None, "NPP", "GR1"])
def test_views_put_tracer_on_its_leg(bins, term):
    """B^{a c b}(k1, k2, k3) = B^{a b c}(k1, k3, k2) for m = 0 - swapping legs 2 and 3 with their tracers.
    Numeric kernels and the analytic bk_mt terms"""
    fc = bins[(8, False)]
    _, k1, k2, k3, theta, zz = fc.args
    theta_swap = utils.get_theta(k1, k3, k2)
    ln = [1] if term == "GR1" else [0, 1, 2]
    kw = dict(kernels=K) if term is None else {}
    for a, b, c in [(0, 0, 1), (1, 0, 1), (0, 1, 1)]:
        got = bk.bk_func(term, ln, fc.cf_mat_bk[a][c][b], k1, k2, k3, theta, zz, **kw)
        ref = bk.bk_func(term, ln, fc.cf_mat_bk[a][b][c], k1, k3, k2, theta_swap, zz, **kw)
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())


def test_m_views_carry_sign(bins):
    """with m > 0 the k2 <-> k3 relabelling also flips the frame's x - (-1)^m, as FullCovBk.re_ylm"""
    fc = bins[(8, True)]
    _, k1, k2, k3, theta, zz = fc.args
    lm = [(1, 1), (2, 1), (2, 2)]
    got = bk.bk_func(None, lm, fc.cf_mat_bk[0][1][0], k1, k2, k3, theta, zz, kernels=K)
    ref = bk.bk_func(None, lm, fc.cf_mat_bk[0][0][1], k1, k3, k2, utils.get_theta(k1, k3, k2), zz, kernels=K)
    sign = np.array([(-1) ** m for _, m in lm])[:, np.newaxis]
    np.testing.assert_allclose(got, sign * ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("all_m", [False, True])
def test_equal_sides_are_degenerate(bins, all_m):
    """k2 = k3: FFB = FBF and BFB = BBF, x (-1)^m - the k2 <-> k3 relabelling keeps k1, the multipoles' reference.
    k1 = k2: FBF = BFF and FBB = BFB for the monopole only - that relabelling moves the reference leg, so for l > 0
    they are different observables (equilateral too). The copies' covariance rows and data are equal, so the
    covariance is singular there - to machine precision at the default quadrature (16 leaves ~1e-6 on k1 = k2)"""
    fc = bins[(8, all_m)]
    ln = [0, 1, 2]
    lm = fc.multipoles(ln) if all_m else [(l, 0) for l in ln]
    d = fc.get_data_vector(None, ln, kernels=K)
    C = fc.get_cov_mat(ln, n_mu=32, n_phi=32)
    _, k1, k2, k3, _, _ = fc.args
    nt = len(P8)
    for sel, pairs, only_l0 in [
        (k2 == k3, [((0, 0, 1), (0, 1, 0)), ((1, 0, 1), (1, 1, 0))], False),
        (k1 == k2, [((0, 1, 0), (1, 0, 0)), ((0, 1, 1), (1, 0, 1))], True),
    ]:
        assert sel.any()
        for p, q in pairs:
            for i, (l, m) in enumerate(lm):
                if only_l0 and l > 0:
                    continue
                r1, r2 = i * nt + P8.index(p), i * nt + P8.index(q)
                # flat (k2 parallel to k3): the swap is no reflection - +1, and no m != 0 signal (re_ylm's fallback)
                sign = np.where(np.isclose(k1, k2 + k3), 1, (-1) ** m)[sel]
                scale = np.abs(C[r1, r1][sel]).max()
                np.testing.assert_allclose(C[r2, :, sel], sign[:, np.newaxis] * C[r1, :, sel], atol=1e-12 * scale)
                np.testing.assert_allclose(d[r2, sel], sign * d[r1, sel], atol=1e-8 * np.abs(d[r1]).max())


def snr(fc, ln, sel=None):
    d = fc.get_data_vector(None, ln, kernels=K)
    inv = fc.get_inv_cov(ln)
    if sel is not None:
        d, inv = d[:, sel], inv[..., sel]
    return contract(d, inv, d).real


def test_equilateral_monopole_adds_nothing(bins):
    """on equilateral triangles the monopoles of the 8 placements are the 4 plus copies - the same S/N, as the
    pseudoinverse drops the copies. The l > 0 ones are not copies (the tracer on k1 differs), so they do add"""
    fc4, fc8 = bins[(True, False)], bins[(8, False)]
    _, k1, k2, k3, _, _ = fc4.args
    equi = (k1 == k2) & (k2 == k3)
    assert snr(fc8, [0], equi) == pytest.approx(snr(fc4, [0], equi), rel=1e-10)
    assert snr(fc8, [1, 2], equi) > snr(fc4, [1, 2], equi)


@pytest.mark.parametrize("all_m", [False, True])
def test_more_placements_more_information(bins, all_m):
    fc4, fc8 = bins[(True, all_m)], bins[(8, all_m)]
    assert snr(fc8, [1]) > snr(fc4, [1])


@pytest.mark.parametrize("cov_ng, pkln", [(False, None), (True, None), (True, [0, 2])])
def test_fisher_gains_information(forecast_mt, cov_ng, pkln):
    """F(8) - F(4) positive semi-definite - bk alone, with the non-Gaussian covariance and joint with pk (PBCov)"""
    F = FullForecast(forecast_mt.cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, cov_ng=cov_ng)
    kw = dict(terms=None, bkln=[0, 1, 2], pkln=pkln, bk_kernels=K, kernels=K, cov_terms=K, verbose=False)
    F4, F8 = (F.get_fish(["LP", "A_b_1"], all_tracer=n, **kw).fisher_matrix for n in (True, 8))
    gain = np.linalg.eigvalsh(F8 - F4)
    assert gain.min() > -1e-8 * np.abs(F4).max()
    assert gain.max() > 1e-4 * np.abs(F4).max()


def test_sampler(forecast_mt, tmp_path):
    from cosmo_wap.forecast import Sampler

    F = FullForecast(forecast_mt.cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2)
    s = Sampler(
        F, ["LP"], terms=None, bk_kernels=K, pkln=None, bkln=[0, 1, 2], all_tracer=8, fisher_covmat=False, drag=False
    )
    assert s.data[0][0]["bk"].shape[0] == 3 * 8
    fid = {p: s.fiducial[p] for p in s.param_list}
    assert abs(s.get_likelihood(**fid)) < 1e-10
    s.samples_df = None  # save() expects a run
    s.save(tmp_path / "s.pkl")
    np.testing.assert_array_equal(Sampler.load(tmp_path / "s.pkl", F).data[0][0]["bk"], s.data[0][0]["bk"])
