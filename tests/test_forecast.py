"""Tests for cosmo_wap.forecast — FullForecast, PkForecast, BkForecast data vectors, covariances, SNR."""
import numpy as np
import pytest

import cosmo_wap.pk as pk
from cosmo_wap.lib.kernels import K1
from cosmo_wap.forecast import FullForecast


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
        fish = ff.get_fish(
            ["A_s", "n_s"], terms="NPP", pkln=[0], bkln=[0], verbose=False
        )
        assert fish.fisher_matrix.shape == (2, 2)
        assert np.all(fish.errors > 0)
        assert np.all(np.isfinite(fish.errors))
