"""Tests for cosmo_wap.forecast.fisher — FisherMat symmetry, PD, covariance, errors, correlation."""
import numpy as np
import pytest

from cosmo_wap.forecast import FisherMat


@pytest.fixture(scope="module")
def fisher_pk(forecast):
    """Fisher matrix from Pk monopole only, 2 parameters (fast)."""
    return forecast.get_fish(
        ["A_s", "n_s"], terms="NPP", pkln=[0], bkln=None, verbose=False
    )


@pytest.fixture(scope="module")
def fisher_pk_quad(forecast):
    """Fisher matrix from Pk monopole + quadrupole."""
    return forecast.get_fish(
        ["A_s", "n_s"], terms="NPP", pkln=[0, 2], bkln=None, verbose=False
    )


@pytest.fixture(scope="module")
def fisher_bk(forecast):
    """Fisher matrix from Bk monopole only."""
    return forecast.get_fish(
        ["A_s", "n_s"], terms="NPP", pkln=None, bkln=[0], verbose=False
    )


# ── Matrix properties ────────────────────────────────────────────────────────

class TestFisherMatrixProperties:
    def test_shape(self, fisher_pk):
        assert fisher_pk.fisher_matrix.shape == (2, 2)

    def test_symmetric(self, fisher_pk):
        F = fisher_pk.fisher_matrix
        np.testing.assert_allclose(F, F.T, rtol=1e-10)

    def test_positive_definite(self, fisher_pk):
        eigvals = np.linalg.eigvalsh(fisher_pk.fisher_matrix)
        assert np.all(eigvals > 0)

    def test_diagonal_positive(self, fisher_pk):
        assert np.all(np.diag(fisher_pk.fisher_matrix) > 0)


# ── Covariance = F^{-1} ─────────────────────────────────────────────────────

class TestCovariance:
    def test_cov_times_fisher_is_identity(self, fisher_pk):
        product = fisher_pk.covariance @ fisher_pk.fisher_matrix
        np.testing.assert_allclose(product, np.eye(2), atol=1e-8)

    def test_covariance_symmetric(self, fisher_pk):
        C = fisher_pk.covariance
        np.testing.assert_allclose(C, C.T, rtol=1e-10)

    def test_covariance_positive_diagonal(self, fisher_pk):
        assert np.all(np.diag(fisher_pk.covariance) > 0)


# ── Errors ────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_errors_positive(self, fisher_pk):
        assert np.all(fisher_pk.errors > 0)

    def test_errors_match_sqrt_diag_cov(self, fisher_pk):
        expected = np.sqrt(np.diag(fisher_pk.covariance))
        np.testing.assert_allclose(fisher_pk.errors, expected, rtol=1e-12)

    def test_marginalized_geq_unmarginalized(self, fisher_pk):
        """Marginalised errors >= unmarginalized (Cramér–Rao)."""
        unmarg = 1.0 / np.sqrt(np.diag(fisher_pk.fisher_matrix))
        assert np.all(fisher_pk.errors >= unmarg - 1e-15)

    def test_get_error_by_name(self, fisher_pk):
        e = fisher_pk.get_error("A_s")
        assert e > 0

    def test_get_error_raises(self, fisher_pk):
        with pytest.raises(ValueError):
            fisher_pk.get_error("nonexistent_param")


# ── Correlation ──────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_diagonal_is_one(self, fisher_pk):
        diag = np.diag(fisher_pk.correlation)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_off_diagonal_bounded(self, fisher_pk):
        C = fisher_pk.correlation
        mask = ~np.eye(C.shape[0], dtype=bool)
        assert np.all(np.abs(C[mask]) <= 1.0 + 1e-10)

    def test_get_correlation_method(self, fisher_pk):
        r = fisher_pk.get_correlation("A_s", "n_s")
        assert -1.0 <= r <= 1.0


# ── Adding information tightens errors ───────────────────────────────────────

class TestMultipoleTightening:
    def test_quadrupole_tightens(self, fisher_pk, fisher_pk_quad):
        """Adding l=2 should not increase errors (more info → tighter)."""
        for i in range(len(fisher_pk.errors)):
            assert fisher_pk_quad.errors[i] <= fisher_pk.errors[i] + 1e-15

    def test_bk_fisher_positive_definite(self, fisher_bk):
        eigvals = np.linalg.eigvalsh(fisher_bk.fisher_matrix)
        assert np.all(eigvals > 0)

    def test_bk_errors_positive(self, fisher_bk):
        assert np.all(fisher_bk.errors > 0)
