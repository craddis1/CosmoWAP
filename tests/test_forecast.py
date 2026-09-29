"""Tests for cosmo_wap.forecast — FullForecast, PkForecast, BkForecast data vectors, covariances, SNR."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import Forecast, _triangle_beta, contract
from cosmo_wap.forecast.covariances import FullCovBk, FullCovPk
from cosmo_wap.lib import utils
from cosmo_wap.numeric_mu import pk as numeric_mu_pk
from cosmo_wap.numeric_mu.kernels import K1

# ── FullForecast initialisation ──────────────────────────────────────────────


class TestFullForecastInit:
    def test_z_bins_shape(self, forecast):
        assert forecast.z_bins.shape == (2, 2)

    def test_z_bins_ordered(self, forecast):
        for lo, hi in forecast.z_bins:
            assert lo < hi

    def test_kmax_list_length(self, forecast):
        assert len(forecast.k_max_list) == forecast.N_bins

    def test_kmax_positive(self, forecast):
        assert np.all(forecast.k_max_list > 0)

    def test_callable_kmax(self, cosmo_funcs):
        ff = FullForecast(cosmo_funcs, kmax_func=lambda z: 0.05 + 0.02 * z, N_bins=2)
        assert ff.k_max_list[0] != ff.k_max_list[1]


# ── PkForecast data vector ───────────────────────────────────────────────────


class TestPkDataVector:
    def test_npp_monopole_shape(self, pk_bin):
        """Data vector shape = (N_l, N_k)."""
        dv = pk_bin.get_data_vector("NPP", [0])
        assert dv.shape == (1, len(pk_bin.k_bin))

    def test_npp_monopole_positive(self, pk_bin):
        """NPP monopole should be everywhere positive."""
        dv = pk_bin.get_data_vector("NPP", [0])
        assert np.all(dv > 0)

    def test_odd_multipole_zero(self, pk_bin):
        """Odd multipoles of NPP should vanish (parity symmetry)."""
        dv = pk_bin.get_data_vector("NPP", [1])
        np.testing.assert_allclose(dv, 0, atol=1e-15)

    def test_multi_multipole_shape(self, pk_bin):
        dv = pk_bin.get_data_vector("NPP", [0, 2])
        assert dv.shape == (2, len(pk_bin.k_bin))


# ── PkForecast covariance ────────────────────────────────────────────────────


class TestPkCovariance:
    @pytest.fixture(scope="class")
    def pk_cov(self, pk_bin):
        return pk_bin.get_cov_mat([0])

    def test_shape(self, pk_cov, pk_bin):
        assert pk_cov.shape == (1, 1, len(pk_bin.k_bin))

    def test_positive_diagonal(self, pk_cov):
        diag = pk_cov[0, 0, :]
        assert np.all(diag > 0)

    def test_symmetric_multi(self, pk_bin):
        cov = pk_bin.get_cov_mat([0, 2])
        np.testing.assert_allclose(cov[0, 1, :], cov[1, 0, :], rtol=1e-10)


# ── BkForecast data vector ───────────────────────────────────────────────────


class TestBkDataVector:
    def test_shape(self, bk_bin):
        dv = bk_bin.get_data_vector("NPP", [0])
        n_tri = len(bk_bin.args[1])  # k1 array length
        assert dv.shape == (1, n_tri)

    def test_triangle_inequality(self, bk_bin):
        """All stored triangles must satisfy |k1-k2| <= k3 <= k1+k2."""
        _, k1, k2, k3, _, _ = bk_bin.args
        assert np.all(k3 <= k1 + k2 + 1e-12)
        assert np.all(k3 >= np.abs(k1 - k2) - 1e-12)

    def test_ordering(self, bk_bin):
        """k1 >= k2 >= k3 for each stored triangle."""
        _, k1, k2, k3, _, _ = bk_bin.args
        assert np.all(k1 >= k2 - 1e-12)
        assert np.all(k2 >= k3 - 1e-12)


# ── BkForecast covariance ────────────────────────────────────────────────────


class TestBkCovariance:
    @pytest.fixture(scope="class")
    def bk_cov(self, bk_bin):
        return bk_bin.get_cov_mat([0])

    def test_positive_diagonal(self, bk_cov):
        diag = bk_cov[0, 0, :]
        assert np.all(diag > 0)

    def test_symmetric_multi(self, bk_bin):
        cov = bk_bin.get_cov_mat([0, 2])
        np.testing.assert_allclose(cov[0, 1, :], cov[1, 0, :], rtol=1e-10)


class TestCovKernelCrossTerms:
    """With several cov_terms kernels the covariance's P is <(Z_N + Z_LP)(Z_N + Z_LP)*> P, cross terms
    included - not <Z_N Z_N*> P + <Z_LP Z_LP*> P"""

    TERMS = ["N", "LP"]

    def test_pk_monopole(self, pk_bin):
        cov = FullCovPk(pk_bin, pk_bin.cf_mat, self.TERMS)
        cf, kk, zz = cov.args
        P = numeric_mu_pk.get_mu(cov.mu, self.TERMS, self.TERMS, cf, kk, zz) + 1 / cf.n_g(zz)
        np.testing.assert_allclose(cov.get_cov([0])[0, 0], np.sum(cov.weights * np.abs(P) ** 2, axis=-1), rtol=1e-10)

    def test_bk_monopole(self, bk_bin):
        cov = FullCovBk(bk_bin, bk_bin.cf_mat, self.TERMS, n_mu=16, n_phi=16)
        cf = bk_bin.cf_mat[0][0]
        P = [
            numeric_mu_pk.get_mu(cov.mus[i], self.TERMS, self.TERMS, cf, cov.ks[i], cov.zz) + 1 / cf.n_g(cov.zz)
            for i in range(3)
        ]
        ref = bk_bin.s123 * np.pi * np.sum(cov.weights * P[0] * P[1] * P[2], axis=(-2, -1))  # 4pi|Y00|^2 = 1
        np.testing.assert_allclose(cov.get_cov([0])[0, 0], ref, rtol=1e-10)

    def test_bk_single_tracer_is_all_tracer_XXX(self, forecast_mt):
        """single tracer is the XXX block of all_tracer, including the extra pairings of degenerate triangles -
        s123 times the identity pairing is wrong when l1, l2 > 0 (the swapped pairing has Y_l2 on mu2, not mu1;
        with l1 = 0 the orientation average makes the two equal)"""
        ln = [0, 2]
        bk_mt = forecast_mt.get_bk_bin(0, all_tracer=True)
        cov_mt = FullCovBk(bk_mt, bk_mt.cf_mat, self.TERMS, n_mu=16, n_phi=16).get_cov(ln)

        bk_st = forecast_mt.get_bk_bin(0)
        cov_st = FullCovBk(bk_st, [[bk_st.cf_mat[0][0]]], self.TERMS, n_mu=16, n_phi=16)
        st = cov_st.get_cov(ln)
        np.testing.assert_allclose(st, cov_mt[::4, ::4], rtol=1e-10)  # rows l*4 + combo, XXX is combo 0

        # and the fix matters: s123 x identity is off for l1 = l2 = 2 on k1=k2 bins
        k1, k2, _ = cov_st.ks.squeeze()
        naive = bk_st.s123 * cov_st.integrate_mu(0, 0, 0, 0, 0, 0, self.TERMS, 2, 2)
        np.testing.assert_allclose(st[1, 1][k1 != k2], naive[k1 != k2], rtol=1e-10)
        assert not np.allclose(st[1, 1][k1 == k2], naive[k1 == k2], rtol=1e-2)

    def test_pk_cross_tracer_keeps_odd_part(self, forecast_mt):
        """the XY dipole is all N x LP"""
        pk_mt = forecast_mt.get_pk_bin(0, all_tracer=True)
        cov = FullCovPk(pk_mt, pk_mt.cf_mat, self.TERMS)
        _, kk, zz = cov.args
        ref = numeric_mu_pk.get_mu(cov.mu, self.TERMS, self.TERMS, pk_mt.cf_mat[0][1], kk, zz)
        np.testing.assert_allclose(cov.pk_cache[0][1], ref, rtol=1e-10)
        assert np.max(np.abs(ref.imag)) > 1e-3 * np.max(np.abs(ref.real))


class TestInvertMatrix:
    """Forecast.invert_matrix - pseudoinverse on the unit-diagonal rescaled covariance (batch last, as the forecasts).
    Rows are labelled by tracer as row_tracers - [0, 1, 0, 1] is two tracers at two multipoles"""

    class Rows:
        def __init__(self, labels):
            self.labels = np.asarray(labels)

        def row_tracers(self, n):
            return self.labels

    def inv(self, A, labels, rtol=1e-10):
        return Forecast.invert_matrix(self.Rows(labels), A, rtol)

    @staticmethod
    def random_cov(n, N, scales, seed=0):
        """N hermitian positive definite n x n matrices, correlation ~0.5-0.9, rows scaled by `scales`"""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, n, 2 * n)) + 1j * rng.normal(size=(N, n, 2 * n))
        C = X @ X.conj().swapaxes(-1, -2) + 0.2 * np.eye(n)
        C = C * np.outer(scales, scales)
        return np.moveaxis(C, 0, -1)

    @staticmethod
    def exact_inv(A):
        return np.moveaxis(np.linalg.inv(np.moveaxis(A, -1, 0)), 0, -1)

    def test_well_scaled_is_exact_inverse(self):
        A = self.random_cov(4, 50, np.ones(4))
        ref = self.exact_inv(A)
        for labels in ([0, 1, 0, 1], [0, 0, 0, 0]):
            np.testing.assert_allclose(self.inv(A, labels), ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())

    def test_sparse_tracer_keeps_faint_directions(self):
        """variances 1e30 apart (the BGS bright tail at high z) - the cut on the raw eigenvalues dropped the faint ones"""
        scales = np.array([1.0, 1e15, 1.0, 1e15])
        A = self.random_cov(4, 50, scales)
        d = np.random.default_rng(1).normal(size=(4, 50)) * scales[:, np.newaxis]
        ref = sum(np.real(d[:, t] @ np.linalg.solve(A[:, :, t], d[:, t])) for t in range(50))
        assert contract(d, self.inv(A, [0, 1, 0, 1]), d).real == pytest.approx(ref, rel=1e-8)
        assert contract(d, self.inv(A, [0, 1, 0, 1], None), d).real == pytest.approx(ref, rel=1e-8)

    @pytest.mark.parametrize("null_diag", [-1e-3, 0.0, 1e-3])
    def test_null_rows_get_zero_inverse(self, null_diag):
        """tracer 0's second multipole with a roundoff variance (<1e-11 of its first - either sign), off-diagonals
        roundoff too: dropped, and the rest is the inverse of the remaining block. Kept and rescaled, the positive
        one is a unit-variance direction of pure roundoff"""
        A = self.random_cov(4, 20, np.array([1e8, 1.0, 1e8, 1.0]))
        A[2] *= 1e-8
        A[:, 2] *= 1e-8
        A[2, 2] = null_diag
        inv = self.inv(A, [0, 1, 0, 1])
        assert np.all(inv[2] == 0) and np.all(inv[:, 2] == 0)
        keep = [0, 1, 3]
        np.testing.assert_allclose(inv[np.ix_(keep, keep)], self.exact_inv(A[np.ix_(keep, keep)]), rtol=1e-8)

    def test_degenerate_direction_is_dropped(self):
        """two identical rows - singular, one direction kept, and the pseudoinverse contracts a vector in the span"""
        A = self.random_cov(3, 10, np.array([1e6, 1.0, 1.0]))
        A[2], A[:, 2] = A[1], A[:, 1]
        x = np.array([1.0, 2.0, 0.0])
        d = np.moveaxis(np.moveaxis(A, -1, 0) @ x, 0, -1)  # d = A x, so d^dagger A^+ d = x^dagger A x
        ref = np.einsum("i,ijt,j->t", x, A, x).real.sum()
        assert contract(d, self.inv(A, [0, 1, 2]), d).real == pytest.approx(ref, rel=1e-8)

    def test_bk_row_tracers(self, bk_bin, forecast_mt):
        """all_tracer bk rows are XXX,XXY,XYY,YYY per l - single tracer one label"""
        np.testing.assert_array_equal(bk_bin.row_tracers(3), [0, 0, 0])
        np.testing.assert_array_equal(forecast_mt.get_bk_bin(0, all_tracer=True).row_tracers(8), [0, 1, 2, 3] * 2)


def test_pk_cov_is_d_dagger():
    """FullCovPk's C is <d d^dagger>, not <d* d^T> - which fixes the order in core.contract. Toy two-tracer shell
    with an odd imaginary P^XY, so the even-odd entries are complex and the two differ in sign. N half-shell
    modes, each with its -k partner (delta(-k) = delta(k)*), give <d d^dagger> = C / 2N."""
    P = {(0, 0): lambda m: 1.0 + 0.5 * m**2 + 0j, (1, 1): lambda m: 2.0 + 0.3 * m**2 + 0j}
    P[(0, 1)] = lambda m: 1.2 + 0.4 * m**2 + 0.6j * m
    P[(1, 0)] = lambda m: np.conj(P[(0, 1)](m))
    rows = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (0, 0, 2), (0, 1, 2), (1, 1, 2)]  # (a, b, l) as all_tracer

    class NoShot:
        def n_g(self, zz):
            return np.inf

    cov = FullCovPk.__new__(FullCovPk)  # just get_tracer, on the toy P
    cov.sigma = None
    mu, cov.weights = utils.leggauss(32)
    cov.mu, cov.zz, cov.cf_mat = np.real(mu), 1.0, [[NoShot()] * 2] * 2
    cov.pk_cache = [[P[(i, j)](cov.mu) for j in range(2)] for i in range(2)]
    C = np.array([[cov.get_tracer(a, b, c, d, None, l1, l2) for c, d, l2 in rows] for a, b, l1 in rows])

    R, N = 20000, 100
    rng = np.random.default_rng(0)
    mu = -1 + (np.arange(N) + 0.5) * 2 / N
    L = np.linalg.cholesky(np.array([[P[(i, j)](mu) for j in range(2)] for i in range(2)]).transpose(2, 0, 1))
    dl = np.einsum(
        "nij,rnj->rni", L, (rng.standard_normal((R, N, 2)) + 1j * rng.standard_normal((R, N, 2))) / np.sqrt(2)
    )
    d = np.stack(
        [
            (2 * l + 1)
            / (2 * N)
            * np.sum(
                eval_legendre(l, mu) * dl[..., a] * dl[..., b].conj()
                + eval_legendre(l, -mu) * dl[..., a].conj() * dl[..., b],
                -1,
            )
            for a, b, l in rows
        ],
        -1,
    )
    d -= d.mean(0)
    S = 2 * N * d.T @ d.conj() / R
    assert np.abs(S - C).max() < 0.05 * np.abs(C).max()  # ~0.005
    assert np.abs(S.conj() - C).max() > 0.15 * np.abs(C).max()  # <d* d^T>: ~0.23


# ── triangle bin closure fraction (beta) ─────────────────────────────────────


def _beta_reference(k1, k2, k3, dk):
    """beta by direct 2D quadrature - independent of the masking/einsum in _triangle_beta."""
    from scipy import integrate

    def q1_int(d3, d2):
        hi = min(0.5, (k2 + k3 - k1) / dk + d2 + d3)
        return 0.0 if hi <= -0.5 else k1 * (hi + 0.5) + dk * (hi**2 - 0.25) / 2

    val, _ = integrate.dblquad(
        lambda d3, d2: q1_int(d3, d2) * (k2 + dk * d2) * (k3 + dk * d3), -0.5, 0.5, -0.5, 0.5, epsrel=1e-10
    )
    return val / (k1 * k2 * k3)


class TestTriangleBeta:
    DK = 0.0125  # a realistic s_k=4 bin width

    @pytest.fixture(scope="class")
    def grid(self):
        """Bin-index triples spanning u = -1 .. u_max, as the triangle loop would order them."""
        n = 8
        tri = [(i, j, k) for i in range(1, n + 1) for j in range(1, i + 1) for k in range(max(i - j - 1, 1), j + 1)]
        tri = np.array(tri, dtype=float)
        return tri[:, 0] * self.DK, tri[:, 1] * self.DK, tri[:, 2] * self.DK

    def test_matches_reference(self, grid):
        k1, k2, k3 = grid
        got = _triangle_beta(k1, k2, k3, self.DK)
        expected = np.array([_beta_reference(a, b, c, self.DK) for a, b, c in zip(k1, k2, k3)])
        assert np.abs(got - expected).max() < 1e-3

    def test_exactly_one_well_inside(self, grid):
        """u >= 3/2 means the whole bin satisfies closure, so beta is 1 with no quadrature error."""
        k1, k2, k3 = grid
        u = (k2 + k3 - k1) / self.DK
        np.testing.assert_array_equal(_triangle_beta(k1, k2, k3, self.DK)[u >= 1.5], 1.0)

    def test_zero_well_outside(self):
        k1, k2, k3 = np.array([10.0]), np.array([3.0]), np.array([3.0])
        assert _triangle_beta(k1, k2, k3, 1.0)[0] == 0.0

    def test_thin_bin_limit_is_one_half(self):
        """The 1/2 this replaced is the k/dk -> infinity limit on the folded triangles."""
        dk = 1.0
        k1, k2, k3 = np.array([2000.0]), np.array([1000.0]), np.array([1000.0])
        assert abs(_triangle_beta(k1, k2, k3, dk)[0] - 0.5) < 1e-3

    def test_folded_exceeds_one_half_at_thick_bins(self, grid):
        """At s_k=4 the folded triangles sit well above 1/2 - the reason for the change."""
        k1, k2, k3 = grid
        u = (k2 + k3 - k1) / self.DK
        folded = _triangle_beta(k1, k2, k3, self.DK)[np.isclose(u, 0)]
        assert folded.size > 0
        assert np.all(folded > 0.5)
        assert np.all(folded < 0.65)

    def test_bk_forecast_beta_applied(self, bk_bin):
        """V123 on the real grid carries beta, not the binary 1/2."""
        k1, k2, k3 = bk_bin.args[1:4]
        thin = 8 * np.pi**2 * k1 * k2 * k3 * bk_bin.forecast.s_k**3
        np.testing.assert_allclose(
            bk_bin.V123 / thin, _triangle_beta(k1, k2, k3, bk_bin.forecast.s_k * bk_bin.k_f), rtol=1e-12
        )


# ── SNR ──────────────────────────────────────────────────────────────────────


class TestSNR:
    def test_pk_snr_positive(self, forecast):
        snr = forecast.pk_SNR("NPP", [0], verbose=False)
        assert np.all(snr.real > 0)

    def test_bk_snr_positive(self, forecast):
        snr = forecast.bk_SNR("NPP", [0], verbose=False)
        assert np.all(snr.real > 0)


# ── Kaiser kernel K1.N ───────────────────────────────────────────────────────


class TestKaiserKernel:
    def test_mu0_gives_D_b1(self, cosmo_funcs):
        z = 1.0
        k = np.array([0.1])
        D1 = cosmo_funcs.D(z)
        b1 = cosmo_funcs.survey[0].b_1(z)
        result = K1.N(cosmo_funcs, z, mu=0, k1=k)
        assert result == pytest.approx(D1 * b1, rel=1e-10)

    def test_mu1_gives_D_b1_plus_f(self, cosmo_funcs):
        z = 1.0
        k = np.array([0.1])
        D1 = cosmo_funcs.D(z)
        b1 = cosmo_funcs.survey[0].b_1(z)
        f = cosmo_funcs.f(z)
        result = K1.N(cosmo_funcs, z, mu=1, k1=k)
        assert result == pytest.approx(D1 * (b1 + f), rel=1e-10)


# ── Full pipeline integration ────────────────────────────────────────────────


class TestFullPipeline:
    def test_pk_bk_fisher_end_to_end(self, cosmo_funcs):
        """End-to-end: cosmology → forecast → Pk+Bk → Fisher → errors."""
        ff = FullForecast(cosmo_funcs, kmax_func=0.1, s_k=2, N_bins=2)
        fish = ff.get_fish(["A_s", "n_s"], terms="NPP", pkln=[0], bkln=[0], verbose=False)
        assert fish.fisher_matrix.shape == (2, 2)
        assert np.all(fish.errors > 0)
        assert np.all(np.isfinite(fish.errors))


# ── Cosmology-derivative cache ───────────────────────────────────────────────


class TestCosmoDerivCache:
    def test_skips_halofit_when_linear(self, forecast):
        """nonlin=False: nothing on the data-vector path reads Pk_NL, so the shifted
        cosmologies skip halofit entirely (fast=True) and adopt the fiducial biases."""
        cache = forecast._precompute_cache(["A_s"])
        cf_h = cache[0]["A_s"]
        assert not hasattr(cf_h, "Pk_NL")
        assert cf_h.survey[0].b_1 is forecast.cosmo_funcs.survey[0].b_1

    def test_keeps_halofit_when_nonlin(self, cosmo_funcs):
        """nonlin=True: the shifted cosmologies build Pk_NL and carry the nonlin flag,
        so derivative theory matches the signal theory."""
        ff = FullForecast(cosmo_funcs, kmax_func=0.1, s_k=2, N_bins=2, nonlin=True)
        cache = ff._precompute_cache(["A_s"])
        cf_h = cache[0]["A_s"]
        assert cf_h.nonlin
        assert hasattr(cf_h, "Pk_NL")


# ── Multi-tracer derivative data vectors ─────────────────────────────────────


class TestMultiTracerDerivatives:
    """With all_tracer=True the XX/XY/YY rows must be per-combo derivatives, not
    three copies of the full-object one (see five_point_stencil)."""

    @pytest.fixture(scope="class")
    def pkb(self, forecast_mt):
        return forecast_mt.get_pk_bin(0, all_tracer=True)

    @staticmethod
    def rows(dv):
        """XX, XY, YY rows of an ln=[0] all_tracer data vector."""
        return dv[0], dv[1], dv[2]

    def test_signal_rows_differ(self, pkb):
        xx, xy, yy = self.rows(pkb.get_data_vector(["NPP"], [0]))
        assert not np.allclose(xx, xy) and not np.allclose(xy, yy)

    @pytest.mark.parametrize("param", ["Xb_1", "X_b_1"])  # per-bin bias and amplitude bias paths
    def test_tracer_bias_deriv_per_combo(self, pkb, param):
        xx, xy, yy = self.rows(pkb.get_data_vector(["NPP"], [0], param=param))
        # YY doesn't depend on an X bias - zero up to finite-difference cancellation noise
        assert np.max(np.abs(yy)) < 1e-8 * np.max(np.abs(xx))
        assert not np.allclose(xx, xy)

    def test_cosmo_deriv_per_combo_cached(self, forecast_mt):
        cache = forecast_mt._precompute_cache(["Omega_m"])
        pkb = forecast_mt.get_pk_bin(0, all_tracer=True, cache=cache)
        xx, xy, yy = self.rows(pkb.get_data_vector(["NPP"], [0], param="Omega_m"))
        assert not np.allclose(xx, xy) and not np.allclose(xy, yy)

        # cosmology derivatives hold biases fixed: the shifted objects adopt copies of the
        # fiducial tracers (shared bias splines, fresh cache) with their own cosmology
        cf_h, fid = cache[0]["Omega_m"], forecast_mt.cosmo_funcs
        assert cf_h.survey[0].b_1 is fid.survey[0].b_1  # bias functions shared with fiducial
        assert cf_h.survey[0] is not fid.survey[0]  # but tracers are copies - caches don't leak
        assert cf_h.Omega_m != fid.Omega_m  # while the cosmology is genuinely shifted

    def test_bk_tracer_bias_deriv_per_combo(self, forecast_mt):
        bkb = forecast_mt.get_bk_bin(0, all_tracer=True)
        dv = bkb.get_data_vector("NPP", [0], param="X_b_1")  # rows: XXX, XXY, XYY, YYY
        assert np.max(np.abs(dv[3])) < 1e-8 * np.max(np.abs(dv[0]))  # YYY independent of X bias
        assert not np.allclose(dv[0], dv[1])

    def test_mt_fisher_finite(self, forecast_mt):
        fish = forecast_mt.get_fish(["Omega_m"], terms="NPP", pkln=[0, 2], all_tracer=True, verbose=False)
        assert np.all(np.isfinite(fish.fisher_matrix))
        assert fish.fisher_matrix[0, 0] > 0
