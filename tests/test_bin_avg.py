"""The bispectrum on each bin's triangles - forecast.triangle_bins and BkForecast.bin_mean.

The effective sides against a brute-force mean over a fine grid of the bin's closed triangles, the relabellings of
equal-shell sides, and the averaging back to the bins. The bin average of 'exact' against the same grid. And the power
spectrum over each k-shell's modes, PkForecast.bin_mean, which follows the same option.
"""

import itertools
import pickle

import numpy as np
import pytest

import cosmo_wap.bk as bk
import cosmo_wap.pk as pk
from cosmo_wap.forecast import FullForecast, triangle_bins
from cosmo_wap.forecast.covariances import BBCovBk
from cosmo_wap.forecast.triangle_bins import N_AVG, N_DELTA, effective, exact

TERMS = ["N", "LP"]


@pytest.fixture(scope="module")
def fc_avg(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2).get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def fc_centre(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=False).get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def fc_exact(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg="exact").get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def pk_avg(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2).get_pk_bin(0)


@pytest.fixture(scope="module")
def pk_centre(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=False).get_pk_bin(0)


def _grid(T, n):
    """midpoint grid of the bin (sides in units of dk) and its weights, q1 q2 q3 where it closes"""
    x = (np.arange(n) + 0.5) / n - 0.5
    q = np.stack(np.meshgrid(*(Ti + x for Ti in T), indexing="ij"))
    closed = (q[0] <= q[1] + q[2]) & (q[1] <= q[0] + q[2]) & (q[2] <= q[0] + q[1])
    return q, np.prod(q, axis=0) * closed


def _f(q):
    """smooth, and symmetric in no pair of sides"""
    return q[0] * q[1] ** 2 + 3 * q[2] ** 3 + q[0] * q[2]


@pytest.mark.parametrize("T", [(8, 7, 1), (8, 8, 1), (8, 4, 4), (6, 5, 4), (5, 5, 5), (3, 2, 1)])
def test_effective_triangle_matches_brute_force(T):
    """mean longest, middle and shortest side over a midpoint grid of the bin, weighted q1 q2 q3 where it closes"""
    q, w = _grid(T, 60)
    ref = np.sum(np.sort(q, axis=0)[::-1] * w, axis=(1, 2, 3)) / w.sum()

    tri = effective(*(np.array([float(Ti)]) for Ti in T), 1.0)
    sides, bins, weights = tri.sides, tri.bins, tri.weights
    np.testing.assert_allclose(sides[:, 0], ref, atol=2e-3)
    perms = [p for p in itertools.permutations(range(3)) if all(T[p[i]] == T[i] for i in range(3))]  # equal shells
    assert len(bins) == len(perms) and np.allclose(weights, 1 / len(perms))
    assert sorted(map(tuple, sides.T)) == sorted(tuple(sides[list(p), 0]) for p in perms)


def test_eval_triangles(fc_avg, fc_centre):
    """every bin closed and not flattened on its effective triangles, its relabellings consecutive with weights summing
    to 1 - bin_avg=False is the bin centres"""
    k1, k2, k3 = fc_avg.eval_args[1:4]
    assert np.all(k1 + k2 - k3 > 0) and np.all(k1 + k3 - k2 > 0) and np.all(k2 + k3 - k1 > 0)
    assert np.all(np.diff(fc_avg.eval_tri.bins) >= 0)
    np.testing.assert_allclose(np.bincount(fc_avg.eval_tri.bins, weights=fc_avg.eval_tri.weights), 1)
    assert len(fc_avg.eval_tri.bins) > len(fc_avg.k_centre[0])  # isosceles and equilateral bins are relabelled
    np.testing.assert_array_equal(np.array(fc_centre.eval_args[1:4]), fc_centre.k_centre)
    x = np.arange(len(fc_avg.eval_tri.bins), dtype=float)
    ref = [np.mean(x[fc_avg.eval_tri.bins == t]) for t in range(len(fc_avg.k_centre[0]))]
    np.testing.assert_allclose(fc_avg.bin_mean(np.stack([x, 2 * x])), [ref, 2 * np.array(ref)])


def test_data_vector_on_effective_triangles(fc_avg, fc_centre):
    """the monopole is the same on every relabelling, so the signal is B_0 on each bin's effective triangle; with
    bin_avg=False on its centre. Far from the bin centre on (i, i-1, 1)"""
    cf = fc_avg.cosmo_funcs
    first = np.flatnonzero(np.diff(fc_avg.eval_tri.bins, prepend=-1))
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
    sel = np.isin(fc_avg.eval_tri.bins, iso)
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


def test_option_names():
    """True is 'effective', False 'centre' - and only the names in OPTIONS otherwise"""
    names = [triangle_bins.option(b) for b in (True, False, np.bool_(True), "exact")]
    assert names == ["effective", "centre", "effective", "exact"]
    with pytest.raises(ValueError, match="bin_avg"):
        triangle_bins.option("mean")


@pytest.mark.parametrize("T", [(8, 7, 1), (8, 8, 1), (8, 4, 4), (6, 5, 4), (5, 5, 5), (3, 2, 1), (2, 1, 1), (1, 1, 1)])
def test_exact_matches_brute_force(T):
    """the mean over exact's nodes is the bin average - ~1e-3 at n = N_AVG, closer at 8 - on closed triangles, with
    weights summing to 1"""
    q, w = _grid(T, 160)
    ref = np.sum(_f(q) * w) / w.sum()
    for n, rtol in ((N_AVG, 2e-3), (8, 6e-4)):
        tri = exact(*(np.array([float(Ti)]) for Ti in T), 1.0, n=n)
        k1, k2, k3 = tri.sides
        assert np.all(k1 <= k2 + k3 + 1e-12) and np.all(k2 <= k1 + k3 + 1e-12) and np.all(k3 <= k1 + k2 + 1e-12)
        np.testing.assert_allclose(tri.weights.sum(), 1, rtol=1e-12)
        np.testing.assert_allclose(tri.mean(_f(tri.sides)), [ref], rtol=rtol)


@pytest.mark.parametrize("T", [(8, 8, 1), (8, 4, 4), (5, 5, 5), (1, 1, 1)])
def test_exact_symmetric_in_equal_shells(T):
    """exact's nodes already hold the mean over the relabellings the estimator sums - (1, 1, 1) by relabelling them"""
    tri = exact(*(np.array([float(Ti)]) for Ti in T), 1.0)
    for p in itertools.permutations(range(3)):
        if all(T[p[i]] == T[i] for i in range(3)):
            np.testing.assert_allclose(tri.mean(_f(tri.sides[list(p)])), tri.mean(_f(tri.sides)), rtol=1e-12)


def test_exact_forecast(fc_avg, fc_centre, fc_exact):
    """bin_avg='exact' evaluates the signal on the bin average - effective's B_0 within 10% of it, the centre's not on
    (8, 7, 1) - by the analytic terms and the numeric-mu kernels (in chunks) alike. The Gaussian covariance stays on
    the effective triangles"""
    assert fc_exact.bin_avg == "exact"
    np.testing.assert_array_equal(fc_exact.cov_tri.sides, fc_avg.cov_tri.sides)
    B = fc_exact.get_data_vector("NPP", [0])[0]
    np.testing.assert_allclose(fc_avg.get_data_vector("NPP", [0])[0], B, rtol=0.1)
    t = np.flatnonzero(np.all(np.isclose(fc_avg.k_centre / (2 * fc_avg.k_f), [[8], [7], [1]]), axis=0))[0]
    assert abs(fc_centre.get_data_vector("NPP", [0])[0, t] / B[t] - 1) > 0.3
    assert len(fc_exact.eval_tri) * 16 * 16 > 2**20  # more than one chunk of numeric_mu.bk.get_multipoles
    np.testing.assert_allclose(fc_exact.get_data_vector(None, [0], kernels=["N"])[0], B, rtol=1e-10)


def test_bb_n_avg_follows_bin_avg(fc_avg, fc_centre, fc_exact):
    """BBCovBk's default n_avg is the bin_avg option's - see triangle_bins.OPTIONS"""
    kw = dict(n_mu=6, n_psi=4, n_delta=2)
    for fc, n_avg in ((fc_centre, 0), (fc_avg, N_AVG), (fc_exact, N_AVG)):
        assert triangle_bins.OPTIONS[fc.bin_avg][2] == n_avg
        np.testing.assert_array_equal(BBCovBk(fc, TERMS, [0], **kw).U, BBCovBk(fc, TERMS, [0], n_avg=n_avg, **kw).U)


def test_save_load_needs_matching_bin_avg(cosmo_funcs, tmp_path):
    """load() needs the bin_avg a chain was saved with - True and 'effective' alike - and refuses a pickle without it"""
    pytest.importorskip("cobaya")
    from cosmo_wap.forecast.sampler import Sampler

    F = {b: FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=b) for b in (True, "effective", False)}
    s = Sampler(F[True], ["fNL"], terms=["NPP", "Loc"], bkln=[0], fisher_covmat=False, drag=False)
    s.samples_df = None  # save() expects a run
    path = tmp_path / "s.pkl"
    s.save(path)
    with pytest.raises(ValueError, match="bin_avg"):
        Sampler.load(path, F[False])
    Sampler.load(path, F["effective"])

    with open(path, "rb") as f:
        saved = pickle.load(f)
    del saved["bin_avg"]
    with open(path, "wb") as f:
        pickle.dump(saved, f)
    with pytest.raises(ValueError, match="rerun the chain"):
        Sampler.load(path, F[True])


@pytest.mark.parametrize("p", [-2, 0, 1, 3])
def test_pk_bin_mean_exact(pk_avg, p):
    """N_DELTA nodes across each shell, weighted K^2 as its modes: the mean of K^p is the shell average, exactly to p=3"""
    assert pk_avg.eval_w.shape == (len(pk_avg.args[1]), N_DELTA)
    _, kk, _ = pk_avg.args
    dk = pk_avg.forecast.s_k * pk_avg.k_f
    lo, hi = kk - dk / 2, kk + dk / 2
    ref = 3 / (p + 3) * (hi ** (p + 3) - lo ** (p + 3)) / (hi**3 - lo**3)
    np.testing.assert_allclose(pk_avg.bin_mean(pk_avg.eval_args[1] ** p), ref, rtol=1e-12)


def test_pk_follows_bin_avg(cosmo_funcs, pk_avg, pk_centre):
    """With bin_avg on the pk signal and its derivatives are the mean over each shell's modes - converged at N_DELTA
    nodes against 16 (ng_kwargs, as BBCovBk), and ~9% above the centre on the first shell. With bin_avg=False the
    centre"""
    np.testing.assert_array_equal(pk_centre.eval_args[1], pk_centre.args[1])
    np.testing.assert_array_equal(
        pk_centre.get_data_vector(["NPP"], [0, 2]), pk.pk_func(["NPP"], [0, 2], cosmo_funcs, *pk_centre.args[1:])
    )

    fine = FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, ng_kwargs=dict(n_delta=16)).get_pk_bin(0)
    assert len(fine.delta_K) == 16
    for kw in (dict(), dict(param="fNL", fNL=5.0)):
        d = pk_avg.get_data_vector(["NPP", "Loc"], [0, 2], **kw)
        np.testing.assert_allclose(d, fine.get_data_vector(["NPP", "Loc"], [0, 2], **kw), rtol=2e-3)
    P0 = pk_avg.get_data_vector(["NPP"], [0])[0]
    assert P0[0].real / pk_centre.get_data_vector(["NPP"], [0])[0, 0].real > 1.05


@pytest.mark.parametrize("bin_avg", [False, True])
def test_sampler_pk_theory_matches_data(cosmo_funcs, bin_avg):
    """the likelihood's pk theory (Sampler.get_pk_d1, averaged over each shell's modes) is the precomputed data vector
    at the fiducial, and moves off it with fNL"""
    pytest.importorskip("cobaya")
    from cosmo_wap.forecast.sampler import Sampler

    F = FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, bin_avg=bin_avg)
    s = Sampler(F, ["fNL"], terms=["NPP", "Loc"], pkln=[0, 2], fisher_covmat=False, drag=False)
    fid = {p: s.fiducial[p] for p in s.param_list}
    assert abs(s.get_likelihood(**fid)) < 1e-10
    assert s.get_likelihood(**{**fid, "fNL": fid["fNL"] + 5}) < -1e-6
