"""m != 0 bispectrum covariance - FullCovBk.get_tracer against explicit triangle geometry.

The reference builds the three wavevectors, every relabelling the estimator counts on equal sides, each labelled
triangle's own frame (z along its k1, k2 in the xz-plane at positive x) and the Wick pairing by matching vectors -
FullCovBk uses the closed-form cos(phi') of re_ylm and the letter permutations of get_tracer instead. Same (mu, phi)
quadrature, so they agree to rounding. Toy two-tracer P(k, mu) with an odd imaginary cross part, no shot noise.
"""

import itertools

import numpy as np
import pytest
from scipy.special import sph_harm_y

from cosmo_wap.forecast.covariances import FullCovBk
from cosmo_wap.lib import utils

# scalene, k2=k3, k1=k2, equilateral - get_tracer adds the equal-side permutations with ==
TRIANGLES = [(3.0, 2.5, 2.0), (3.0, 2.0, 2.0), (3.0, 3.0, 2.0), (2.0, 2.0, 2.0)]
B, GAMMA, F = (1.3, 2.1), (0.4, -0.7), 0.8


def P(i, j, k, mu):
    """<delta_i delta_j^*> for delta_i = Z_i sqrt(P_m), Z_i = b_i + f mu^2 + i gamma_i mu / k"""
    Z = lambda t: B[t] + F * mu**2 + 1j * GAMMA[t] * mu / k
    return Z(i) * np.conj(Z(j)) / k


class NoShot:
    def n_g(self, zz):
        return np.inf


def toy_cov(n_mu=10, n_phi=16):
    cov = FullCovBk.__new__(FullCovBk)  # just get_tracer, on the toy P
    nodes, w_mu = (np.real(x) for x in utils.leggauss(n_mu))
    nodes_phi, w_phi = (np.real(x) for x in utils.leggauss(n_phi))
    cov.weights = w_mu[:, np.newaxis] * w_phi
    cov.phi = 2 * np.pi * (nodes_phi + 1) / 2
    mu = nodes[:, np.newaxis]
    k1, k2, k3 = (np.array(k)[:, np.newaxis, np.newaxis] for k in zip(*TRIANGLES))
    theta = utils.get_theta(k1, k2, k3)
    mu2 = mu * np.cos(theta) + np.sqrt(1 - mu**2) * np.sin(theta) * np.cos(cov.phi)
    cov.mus = mu, mu2, -(mu * k1 + mu2 * k2) / k3
    cov.ks = np.array([k1, k2, k3])
    cov.sigma, cov.zz, cov._ylm_cache = None, 1.0, {}
    cov.cf_mat = [[NoShot()] * 2] * 2
    cov.pk_cache = [[[P(i, j, cov.ks[ki], cov.mus[ki]) for j in range(2)] for i in range(2)] for ki in range(3)]
    return cov


def ylm(l, m, mu, phi, real):
    y = (
        sph_harm_y(l, abs(m), np.arccos(mu), phi)
        if m >= 0
        else (-1) ** m * np.conj(sph_harm_y(l, -m, np.arccos(mu), phi))
    )
    return np.real(y) if real else y


def reference(cov, t, tr1, tr2, lm1, lm2, real=True):
    """C[B^tr1_lm1, B^tr2_lm2] of triangle t by explicit vectors - see module docstring"""
    k = np.array(TRIANGLES[t])
    theta = utils.get_theta(*k)
    v = np.array([[0, 0, k[0]], [k[1] * np.sin(theta), 0, k[1] * np.cos(theta)], [0, 0, 0]])
    v[2] = -v[0] - v[1]
    mu = cov.mus[0][:, 0][:, np.newaxis]
    s = np.sqrt(1 - mu**2)
    n = np.stack(np.broadcast_arrays(s * np.cos(cov.phi), s * np.sin(cov.phi), mu), -1)  # LOS (n_mu, n_phi, 3)
    mus = n @ (v / k[:, np.newaxis]).T  # LOS cosine of each leg

    tot = 0
    for p in itertools.permutations(range(3)):
        if not np.allclose(k[list(p)], k):
            continue  # the second triangle's leg j is this one's leg p[j] - only on equal sides
        z = v[p[0]] / k[0]
        x = v[p[1]] - (v[p[1]] @ z) * z
        x /= np.linalg.norm(x)
        y2 = ylm(*lm2, n @ z, np.arctan2(n @ np.cross(z, x), n @ x), real)
        PPP = np.prod([P(tr1[i], tr2[p.index(i)], k[i], mus[..., i]) for i in range(3)], axis=0)
        tot = tot + np.sum(cov.weights * 4 * np.pi * np.conj(ylm(*lm1, n[..., 2], cov.phi, real)) * y2 * PPP)
    return (2 * np.pi) / 2.0 * tot


LM = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 1), (3, 3)]
TRACERS = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]


@pytest.fixture(scope="module")
def cov():
    return toy_cov()


@pytest.mark.parametrize("lm1, lm2", [(a, b) for a in LM for b in LM if a <= b])
def test_get_tracer_matches_geometry(cov, lm1, lm2):
    for tr1, tr2 in [(TRACERS[0], TRACERS[0]), (TRACERS[1], TRACERS[2]), (TRACERS[4], TRACERS[1])]:
        got = cov.get_tracer(*tr1, *tr2, None, lm1, lm2)
        ref = np.array([reference(cov, t, tr1, tr2, lm1, lm2) for t in range(len(TRIANGLES))])
        scale = max(abs(reference(cov, t, tr1, tr1, (0, 0), (0, 0))) for t in range(len(TRIANGLES)))  # vanishing blocks
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * scale)


def test_swap_k2_k3_flips_odd_m(cov):
    """the k2=k3 relabelling sends phi -> pi - phi, so its term carries (-1)^m - dropping the sign changes C"""
    t = 1  # (3, 2, 2)
    same = [reference(cov, t, (0, 0, 0), (0, 0, 0), (1, 1), (1, 1))]
    legs = cov.integrate_mu(0, 0, 0, 0, 0, 0, None, (1, 1), (1, 1), (0, 2))[t]
    naive = cov.integrate_mu(0, 0, 0, 0, 0, 0, None, (1, 1), (1, 1), (0, 1))[t]
    ident = cov.integrate_mu(0, 0, 0, 0, 0, 0, None, (1, 1), (1, 1))[t]
    assert ident + legs == pytest.approx(same[0], rel=1e-10)
    assert abs((ident + naive) - same[0]) > 1e-3 * abs(same[0])


@pytest.mark.parametrize("tr", [TRACERS[0], TRACERS[1]])
def test_positive_m_is_complete(cov, tr):
    """Re Y_lm rows are (B_lm + (-1)^m B_l-m)/2 of the complex estimators: their covariance is get_tracer's, and the
    other half (B_lm - (-1)^m B_l-m)/2 is uncorrelated with it and has no signal (B is even in phi) - so m >= 0 in
    the Re Y basis carries all the information of every m"""
    lms = [(l, m) for l in (1, 2) for m in range(-l, l + 1)]
    for t in range(len(TRIANGLES)):
        C = np.array([[reference(cov, t, tr, tr, a, b, real=False) for b in lms] for a in lms])
        S = np.zeros((0, len(lms)))
        A = np.zeros((0, len(lms)))
        for l, m in [(l, m) for l, m in lms if m >= 0]:
            s, a = np.zeros(len(lms)), np.zeros(len(lms))
            s[lms.index((l, m))] += 0.5
            s[lms.index((l, -m))] += 0.5 * (-1) ** m
            if m > 0:
                a[lms.index((l, m))], a[lms.index((l, -m))] = 0.5, -0.5 * (-1) ** m
                A = np.vstack([A, a])
            S = np.vstack([S, s])
        pos = [(l, m) for l, m in lms if m >= 0]
        got = np.array([[cov.get_tracer(*tr, *tr, None, a, b)[t] for b in pos] for a in pos])
        scale = np.max(np.abs(C))
        np.testing.assert_allclose(S @ C @ S.T, got, atol=1e-10 * scale)
        np.testing.assert_allclose(A @ C @ S.T, 0, atol=1e-10 * scale)


def test_flat_triangle_is_finite():
    """k1 = k2 + k3 has no plane - re_ylm falls back to the identity frame, as there is no m != 0 signal"""
    cov = toy_cov()
    k1, k2, k3 = (
        np.array([4.0, 3.0])[:, None, None],
        np.array([2.0, 2.0])[:, None, None],
        np.array([2.0, 1.5])[:, None, None],
    )
    theta = utils.get_theta(k1, k2, k3)
    mu = cov.mus[0]
    mu2 = mu * np.cos(theta) + np.sqrt(1 - mu**2) * np.sin(theta) * np.cos(cov.phi)
    cov.mus = mu, mu2, -(mu * k1 + mu2 * k2) / k3
    cov.ks, cov._ylm_cache = np.array([k1, k2, k3]), {}
    cov.pk_cache = [[[P(i, j, cov.ks[ki], cov.mus[ki]) for j in range(2)] for i in range(2)] for ki in range(3)]
    assert np.all(np.isfinite(cov.get_tracer(0, 0, 0, 0, 0, 0, None, (1, 1), (1, 1))))


# ---- through BkForecast and the Fisher: FullForecast(all_m=True)
K = ["N", "LP"]


@pytest.fixture(scope="module")
def forecasts(cosmo_funcs):
    from cosmo_wap.forecast import FullForecast

    return [FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, all_m=a) for a in (False, True)]


def test_multipoles(forecasts):
    F0, F1 = forecasts
    assert F0.get_bk_bin(0).multipoles([0, 2]) == [0, 2]
    assert F1.get_bk_bin(0).multipoles([0, 1, 2]) == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]


def test_m0_rows_unchanged(forecasts):
    """the m=0 rows and their covariance block are exactly the all_m=False ones"""
    fc0, fc1 = (F.get_bk_bin(0) for F in forecasts)
    ln = [0, 1, 2]
    rows = [i for i, (_, m) in enumerate(fc1.multipoles(ln)) if m == 0]
    np.testing.assert_array_equal(
        fc1.get_data_vector(None, ln, kernels=K)[rows], fc0.get_data_vector(None, ln, kernels=K)
    )
    C1, C0 = fc1.get_cov_mat(ln, n_mu=16, n_phi=16), fc0.get_cov_mat(ln, n_mu=16, n_phi=16)
    np.testing.assert_array_equal(C1[np.ix_(rows, rows)], C0)


def test_fisher_gains_information(forecasts):
    """adding the m > 0 rows can only add information - F(all_m) - F(m=0) is positive semi-definite"""
    F0, F1 = (
        F.get_fish(["LP", "A_b_1"], terms=None, bkln=[0, 1, 2], bk_kernels=K, verbose=False).fisher_matrix
        for F in forecasts
    )
    gain = np.linalg.eigvalsh(F1 - F0)
    assert gain.min() > -1e-8 * np.abs(F0).max()
    assert gain.max() > 1e-4 * np.abs(F0).max()


def test_m_needs_kernels(forecasts):
    fc = forecasts[1].get_bk_bin(0)
    with pytest.raises(NotImplementedError):
        fc.get_data_vector("GR1", [1])
    with pytest.raises(NotImplementedError):
        fc.SNR(None, [1], m=1, kernels=K)


def test_cov_ng_fisher_gains_information(cosmo_funcs):
    """cov_ng with all_m, bk alone and joint with pk (PBCov): the m > 0 rows add information here too"""
    from cosmo_wap.forecast import FullForecast

    kw = dict(terms=None, bkln=[0, 1, 2], bk_kernels=K, kernels=K, cov_terms=K, verbose=False)
    for pkln in (None, [0, 2]):
        F0, F1 = (
            FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, cov_ng=True, all_m=a)
            .get_fish(["LP", "A_b_1"], pkln=pkln, **kw)
            .fisher_matrix
            for a in (False, True)
        )
        gain = np.linalg.eigvalsh(F1 - F0)
        assert gain.min() > -1e-8 * np.abs(F0).max()
        assert gain.max() > 1e-4 * np.abs(F0).max()


class TestSamplerAllM:
    """the sampler's bk data vector and theory carry the m > 0 rows, and the fiducial is the minimum"""

    ARGS = dict(pkln=None, bkln=[0, 1, 2], fisher_covmat=False, drag=False, terms=None, bk_kernels=K)

    @pytest.mark.parametrize("mt", [False, True])
    def test_fiducial_and_shape(self, cosmo_funcs, forecast_mt, mt):
        from cosmo_wap.forecast import FullForecast, Sampler

        cf = forecast_mt.cosmo_funcs if mt else cosmo_funcs
        F = FullForecast(cf, kmax_func=0.05, s_k=2, N_bins=2, all_m=True)
        s = Sampler(F, ["LP"], all_tracer=mt, **self.ARGS)
        nt = 4 if mt else 1
        for i in range(F.N_bins):
            assert s.data[0][i]["bk"].shape[0] == 6 * nt == s.inv_covs[i]["bk"].shape[0]
        fid = {p: s.fiducial[p] for p in s.param_list}
        assert abs(s.get_likelihood(**fid)) < 1e-10
        assert s.get_likelihood(**{p: v + 0.5 for p, v in fid.items()}) < -1e-6

    def test_save_load_needs_matching_all_m(self, cosmo_funcs, tmp_path):
        from cosmo_wap.forecast import FullForecast, Sampler

        F0, F1 = (FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, all_m=a) for a in (False, True))
        s = Sampler(F1, ["LP"], **self.ARGS)
        s.samples_df = None  # save() expects a run
        path = tmp_path / "s.pkl"
        s.save(path)
        with pytest.raises(ValueError):
            Sampler.load(path, F0)
        loaded = Sampler.load(path, F1)
        np.testing.assert_array_equal(loaded.data[0][0]["bk"], s.data[0][0]["bk"])
