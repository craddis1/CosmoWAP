"""The bispectrum on each bin's effective triangles - forecast.core._effective_triangles and BkForecast.bin_mean.

The effective sides against a brute-force mean over a fine grid of the bin's closed triangles, the relabellings of
equal-shell sides, and the averaging back to the bins.
"""

import itertools

import numpy as np
import pytest

import cosmo_wap.bk as bk
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import _effective_triangles

TERMS = ["N", "LP"]


@pytest.fixture(scope="module")
def fc_avg(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2).get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def fc_centre(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=False).get_bk_bin(0, cov_terms=TERMS)


@pytest.mark.parametrize("T", [(8, 7, 1), (8, 8, 1), (8, 4, 4), (6, 5, 4), (5, 5, 5), (3, 2, 1)])
def test_effective_triangle_matches_brute_force(T):
    """mean longest, middle and shortest side over a midpoint grid of the bin, weighted q1 q2 q3 where it closes"""
    n = 60
    x = (np.arange(n) + 0.5) / n - 0.5
    q = np.stack(np.meshgrid(*(Ti + x for Ti in T), indexing="ij"))
    closed = (q[0] <= q[1] + q[2]) & (q[1] <= q[0] + q[2]) & (q[2] <= q[0] + q[1])
    w = np.prod(q, axis=0) * closed
    ref = np.sum(np.sort(q, axis=0)[::-1] * w, axis=(1, 2, 3)) / w.sum()

    sides, bins, weights = _effective_triangles(*(np.array([float(Ti)]) for Ti in T), 1.0)
    np.testing.assert_allclose(sides[:, 0], ref, atol=2e-3)
    perms = [p for p in itertools.permutations(range(3)) if all(T[p[i]] == T[i] for i in range(3))]  # equal shells
    assert len(bins) == len(perms) and np.allclose(weights, 1 / len(perms))
    assert sorted(map(tuple, sides.T)) == sorted(tuple(sides[list(p), 0]) for p in perms)


def test_eval_triangles(fc_avg, fc_centre):
    """every bin closed and not flattened on its effective triangles, its relabellings consecutive with weights summing
    to 1 - bin_avg=False is the bin centres"""
    k1, k2, k3 = fc_avg.eval_args[1:4]
    assert np.all(k1 + k2 - k3 > 0) and np.all(k1 + k3 - k2 > 0) and np.all(k2 + k3 - k1 > 0)
    assert np.all(np.diff(fc_avg.eval_bin) >= 0)
    np.testing.assert_allclose(np.bincount(fc_avg.eval_bin, weights=fc_avg.eval_w), 1)
    assert len(fc_avg.eval_bin) > len(fc_avg.k_centre[0])  # isosceles and equilateral bins are relabelled
    np.testing.assert_array_equal(np.array(fc_centre.eval_args[1:4]), fc_centre.k_centre)
    x = np.arange(len(fc_avg.eval_bin), dtype=float)
    ref = [np.mean(x[fc_avg.eval_bin == t]) for t in range(len(fc_avg.k_centre[0]))]
    np.testing.assert_allclose(fc_avg.bin_mean(np.stack([x, 2 * x])), [ref, 2 * np.array(ref)])


def test_data_vector_on_effective_triangles(fc_avg, fc_centre):
    """the monopole is the same on every relabelling, so the signal is B_0 on each bin's effective triangle; with
    bin_avg=False on its centre. Far from the bin centre on (i, i-1, 1)"""
    cf = fc_avg.cosmo_funcs
    first = np.flatnonzero(np.diff(fc_avg.eval_bin, prepend=-1))
    args = [arr[first] for arr in fc_avg.eval_args[1:5]]
    np.testing.assert_allclose(fc_avg.get_data_vector("NPP", [0])[0], bk.bk_func("NPP", 0, cf, *args, fc_avg.z_mid))
    centre = fc_centre.get_data_vector("NPP", [0])[0]
    np.testing.assert_allclose(centre, bk.bk_func("NPP", 0, cf, *fc_centre.args[1:5], fc_centre.z_mid))

    t = np.flatnonzero(np.all(np.isclose(fc_avg.k_centre / (2 * fc_avg.k_f), [[7], [6], [1]]), axis=0))[0]
    avg = fc_avg.get_data_vector("NPP", [0])[0, t]
    assert abs(centre[t] / avg - 1) > 0.3


def test_quadrupole_symmetric_on_isosceles(fc_avg):
    """on k1 = k2 the estimator has the Legendre on either side - the relabelled mean is the same if the sides swap"""
    cf = fc_avg.cosmo_funcs
    k1, k2, k3, theta, zz = fc_avg.eval_args[1:]
    iso = np.flatnonzero(np.isclose(fc_avg.k_centre[0], fc_avg.k_centre[1]) & ~np.isclose(*fc_avg.k_centre[1:]))
    sel = np.isin(fc_avg.eval_bin, iso)
    B = bk.bk_func("NPP", 2, cf, k1[sel], k2[sel], k3[sel], theta[sel], zz)
    B_swap = bk.bk_func("NPP", 2, cf, k2[sel], k1[sel], k3[sel], theta[sel], zz)
    assert np.abs(B - B_swap).max() > 1e-3 * np.abs(B).max()  # the effective triangle is not isosceles
    np.testing.assert_allclose(fc_avg.get_data_vector("NPP", [2])[0, iso], B.reshape(-1, 2).mean(1), rtol=1e-12)


def test_gaussian_cov_on_effective_triangles(fc_avg, fc_centre):
    """the Gaussian covariance is the bin mean over the relabellings - each evaluated with the bin's equal-side
    permutation terms - and close to the bin centre's on closed bins with every side >= 4 dk"""
    C = fc_avg.get_cov_mat([0, 2], n_mu=16, n_phi=16)
    C0 = fc_centre.get_cov_mat([0, 2], n_mu=16, n_phi=16)
    assert np.all(np.real(np.einsum("iit->ti", C)) > 0)
    k = fc_avg.k_centre / (2 * fc_avg.k_f)
    far = (k[2] >= 4) & (k[1] + k[2] - k[0] >= 4)
    np.testing.assert_allclose(C[0, 0, far].real, C0[0, 0, far].real, rtol=0.05)


@pytest.mark.parametrize("bin_avg", [False, True])
def test_sampler_bk_theory_matches_data(cosmo_funcs, bin_avg):
    """the likelihood's bk theory (Sampler.get_bk_d1, averaged back to the bins) is the precomputed data vector at the
    fiducial, and moves off it with fNL"""
    pytest.importorskip("cobaya")
    from cosmo_wap.forecast.sampler import Sampler

    F = FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=bin_avg)
    s = Sampler(F, ["fNL"], terms=["NPP", "Loc"], bkln=[0, 2], fisher_covmat=False, drag=False)
    fid = {p: s.fiducial[p] for p in s.param_list}
    assert abs(s.get_likelihood(**fid)) < 1e-10
    assert s.get_likelihood(**{**fid, "fNL": fid["fNL"] + 5}) < -1e-6
