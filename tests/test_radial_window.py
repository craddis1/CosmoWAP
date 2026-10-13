"""Integrated terms through a redshift bin's radial window - numeric_mu.pk.get_int_window, in the Gaussian covariances
(always) and the pk signal (FullForecast(radial_window=True)).

get_mu has no window, so its lensing spike at mu = 0 is ~d deep and the covariance, which squares it, comes out ~d/dr
too big at low k. The window keeps the spike's area, so the multipoles agree once k dr >> 2pi.
"""

import numpy as np
import pytest
from scipy.special import eval_legendre

import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.covariances import BBCovBk
from cosmo_wap.lib import utils
from cosmo_wap.numeric_mu import pk as npk

TERMS = ["N", "LP", "I"]
LOCAL = ["N", "LP"]
LN = [0, 2]


@pytest.fixture(scope="module")
def small(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.04, s_k=2, N_bins=2)


@pytest.fixture(scope="module")
def small_mt(cosmo):
    cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(6e-16), verbose=False)
    return FullForecast(cf, kmax_func=0.035, s_k=2, N_bins=2)


def sinh_quad(n=2000, mu_min=1e-6):
    """mu nodes and weights dense at mu = 0 - resolves get_mu's unwindowed spike"""
    T = np.arcsinh(1 / mu_min)
    x, w = utils.leggauss(n)
    t = T * x
    return np.sinh(t) / np.sinh(T), w * T * np.cosh(t) / np.sinh(T)


def raw_int(cosmo_funcs, mu, kk, zz):
    """P_I of get_mu - no window"""
    full = npk.get_mu_sym(mu, TERMS, TERMS, cosmo_funcs, kk, zz, n=64)
    return full - npk.get_mu_sym(mu, LOCAL, LOCAL, cosmo_funcs, kk, zz)


def test_local_kernels_untouched(cosmo_funcs):
    """no integrated kernels - get_mu_window is get_mu_sym, so local covariances and signals are as before"""
    mu, _ = utils.leggauss(16)
    kk = np.array([[0.01], [0.05]])
    window = npk.get_mu_window(mu, LOCAL, LOCAL, cosmo_funcs, kk, 1.2, window=(300.0, 2e-3))
    np.testing.assert_array_equal(window, npk.get_mu_sym(mu, LOCAL, LOCAL, cosmo_funcs, kk, 1.2))
    assert npk.get_int_window(LOCAL, LOCAL, cosmo_funcs, 1.2, 300.0, 2e-3, 0.05) is None


@pytest.mark.parametrize("l", LN)
def test_multipoles_keep_the_spike_area(cosmo_funcs, l):
    """k dr >> 2pi: the window only spreads the spike in mu, so the multipoles of P_I are get_mu's. l = 2 converges
    as 1/(k dr) - the sinc^2 tails reach where L_2 moves: -3% at dr = 3000, -0.9% at 12000"""
    zz, dr = 1.5, 12000.0
    kk = np.array([0.03, 0.05])
    mu, w = sinh_quad()
    leg = (2 * l + 1) / 2 * w * eval_legendre(l, mu)
    raw = np.sum(leg * raw_int(cosmo_funcs, mu, kk[:, None], zz), axis=-1)
    P_W = npk.get_int_window(TERMS, TERMS, cosmo_funcs, zz, dr, 2e-3, kk.max(), n_par=512)  # dr >> any bin's
    np.testing.assert_allclose(np.sum(leg * P_W(kk[:, None], mu), axis=-1), raw, rtol=2e-2)


def test_thin_bin_cuts_the_covariance(cosmo_funcs):
    """dz = 0.1 at z = 1.55, k = 0.005: get_mu's spike puts P_I^2 at several C_N, the window at a few percent"""
    zz, k = 1.55, np.array([[0.005]])
    dr = cosmo_funcs.comoving_dist(zz + 0.05) - cosmo_funcs.comoving_dist(zz - 0.05)
    mu, w = sinh_quad()
    P_N = npk.get_mu_sym(mu, LOCAL, LOCAL, cosmo_funcs, k, zz) + 1 / cosmo_funcs.n_g(zz)

    def dC(P_I):
        return np.sum(w * np.abs(P_N + P_I) ** 2) / np.sum(w * np.abs(P_N) ** 2) - 1

    P_W = npk.get_int_window(TERMS, TERMS, cosmo_funcs, zz, dr, 3.6e-3, 0.01)
    raw, windowed = dC(raw_int(cosmo_funcs, mu, k, zz)), dC(P_W(k, mu))
    assert raw > 1
    assert 0 < windowed < 0.2


def test_bk_cov_converged(small):
    """windowed, P_I is smooth enough for FullCovBk's (mu, phi) grid - unwindowed it was 40-90% short at 32 x 32"""
    fc = small.get_bk_bin(1, cov_terms=TERMS)
    coarse = fc.get_cov_mat(LN, n_mu=24, n_phi=24)
    fine = fc.get_cov_mat(LN, n_mu=40, n_phi=40)
    local = small.get_bk_bin(1, cov_terms=LOCAL).get_cov_mat(LN, n_mu=40, n_phi=40)
    scale = np.abs(np.diagonal(fine, axis1=0, axis2=1)).max(axis=1)  # per triangle
    assert np.max(np.abs(fine - local) / scale) > 1e-3  # the integrated terms are there
    assert np.max(np.abs(coarse - fine) / scale) < 1e-4


def test_pk_cov_converged(small):
    fc = small.get_pk_bin(1, cov_terms=TERMS)
    coarse, fine = fc.get_cov_mat(LN, n_mu=64), fc.get_cov_mat(LN, n_mu=128)
    local = small.get_pk_bin(1, cov_terms=LOCAL).get_cov_mat(LN, n_mu=128)
    scale = np.abs(np.diagonal(fine, axis1=0, axis2=1)).max(axis=1)
    assert np.max(np.abs(fine - local) / scale) > 1e-3
    assert np.max(np.abs(coarse - fine) / scale) < 1e-4


def test_mt_pk_cov_hermitian(small_mt):
    """get_mu's LOS sits on the first field, so with integrated kernels IS != SI* and P^XX is not real - the caches
    keep the hermitian part, (P^ij + P^ji*)/2. Before, the off-parity XY blocks were 36% of the largest element off
    (2e-3 windowed). FullCovPk builds every element itself, so this is a real check"""
    cov = small_mt.get_pk_bin(0, cov_terms=TERMS, all_tracer=True).get_cov_mat([0, 1, 2])
    cov = np.moveaxis(cov, -1, 0)
    np.testing.assert_allclose(cov, np.conj(np.swapaxes(cov, 1, 2)), rtol=0, atol=1e-14 * np.abs(cov).max())
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_mt_bk_cov_hermitian(small_mt):
    """as the pk - FullCovBk.get_multi_tracer fills the lower triangle by conjugation, so ask get_tracer directly.
    Scalene triangles are the same integral either way round (3.6% off before); on equal sides the two orders add
    their delta-function permutations about different legs, the same up to the (mu, phi) quadrature"""
    fc = small_mt.get_bk_bin(0, cov_terms=TERMS, all_tracer=True)
    fc.get_cov_mat([0], n_mu=32, n_phi=32)
    # on equal sides each order's permutation term is the other's on the swapped relabelling - so per bin
    upper = fc.cov_tri.mean(fc.cov.get_tracer(0, 0, 0, 0, 0, 1, TERMS, 0, 1))
    lower = fc.cov_tri.mean(fc.cov.get_tracer(0, 0, 1, 0, 0, 0, TERMS, 1, 0))
    k1, k2, k3 = fc.k_centre
    scalene = (k1 != k2) & (k2 != k3)
    diff = np.abs(upper - np.conj(lower)) / np.abs(upper).max()
    assert diff[scalene].max() < 1e-13
    assert diff[~scalene].max() < 1e-3


def test_ng_cov_drops_integrated(small):
    """BB and PT stay local - integrated effects are in the Gaussian part only"""
    bb = BBCovBk(small.get_bk_bin(0, cov_terms=TERMS), TERMS, [0])
    bb_local = BBCovBk(small.get_bk_bin(0, cov_terms=LOCAL), LOCAL, [0])
    np.testing.assert_array_equal(bb.U, bb_local.U)


def test_signal_flag(cosmo_funcs):
    """radial_window=True puts the pk signal's integrated kernels through the bin's window - derivatives too"""
    off = FullForecast(cosmo_funcs, kmax_func=0.04, s_k=2, N_bins=2).get_pk_bin(1)
    on = FullForecast(cosmo_funcs, kmax_func=0.04, s_k=2, N_bins=2, radial_window=True).get_pk_bin(1)

    d_on = on.get_data_vector(None, LN, kernels=TERMS)
    d_off = off.get_data_vector(None, LN, kernels=TERMS)
    np.testing.assert_array_equal(
        d_on, npk.get_multipoles(TERMS, TERMS, LN, cosmo_funcs, *on.args[1:], window=on.bin_window())
    )
    np.testing.assert_array_equal(d_off, npk.get_multipoles(TERMS, TERMS, LN, cosmo_funcs, *off.args[1:]))
    assert not np.allclose(d_on, d_off, rtol=1e-3)

    # derivatives see it through five_point_stencil
    assert not np.allclose(
        on.get_data_vector(None, LN, param="b_1", kernels=TERMS),
        off.get_data_vector(None, LN, param="b_1", kernels=TERMS),
        rtol=1e-3,
    )
    # nothing integrated, nothing to window
    np.testing.assert_array_equal(
        on.get_data_vector(None, LN, kernels=LOCAL), off.get_data_vector(None, LN, kernels=LOCAL)
    )


def test_sampler_theory_matches_data(cosmo_funcs):
    """the sampler's theory (get_pk_d1) takes the window as its data (get_data_vector) does - else the fiducial chi2
    is not 0"""
    pytest.importorskip("cobaya")
    from cosmo_wap.forecast.sampler import Sampler

    F = FullForecast(cosmo_funcs, kmax_func=0.04, s_k=2, N_bins=2, radial_window=True)
    s = Sampler(F, ["fNL"], terms=None, kernels=TERMS, pkln=LN, bkln=None, fisher_covmat=False, drag=False)
    assert abs(s.get_likelihood(**{p: s.fiducial[p] for p in s.param_list})) < 1e-10
