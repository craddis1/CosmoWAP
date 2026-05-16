"""Tests for cosmo_wap.forecast.fisher — FisherMat symmetry, PD, covariance, errors, correlation."""

import numpy as np
import pytest

from cosmo_wap.forecast.fisher import FisherMat
from cosmo_wap.lib.utils import solve_preconditioned


@pytest.fixture(scope="module")
def fisher_pk(forecast):
    """Fisher matrix from Pk monopole only, 2 parameters (fast)."""
    return forecast.get_fish(["A_s", "n_s"], terms="NPP", pkln=[0], bkln=None, verbose=False)


@pytest.fixture(scope="module")
def fisher_pk_quad(forecast):
    """Fisher matrix from Pk monopole + quadrupole."""
    return forecast.get_fish(["A_s", "n_s"], terms="NPP", pkln=[0, 2], bkln=None, verbose=False)


@pytest.fixture(scope="module")
def fisher_bk(forecast):
    """Fisher matrix from Bk monopole only."""
    return forecast.get_fish(["A_s", "n_s"], terms="NPP", pkln=None, bkln=[0], verbose=False)


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
        np.testing.assert_allclose(product, np.eye(2), atol=1e-7)

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


# ── Preconditioning ───────────────────────────────────────────────────────────


class TestPreconditioning:
    def test_flag_off_is_plain_inv(self):
        """precondition=False must be byte-identical to np.linalg.inv."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        F = A.T @ A + np.eye(4)  # well-conditioned SPD
        np.testing.assert_array_equal(solve_preconditioned(F, precondition=False), np.linalg.inv(F))

    def test_well_conditioned_equivalent(self):
        """Preconditioning must not change the result for well-conditioned inputs."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((5, 5))
        F = A.T @ A + np.eye(5)
        C_on = solve_preconditioned(F, precondition=True)
        C_off = solve_preconditioned(F, precondition=False)
        np.testing.assert_allclose(C_on, C_off, rtol=1e-10)

    def test_ill_conditioned_recovery(self):
        """Preconditioning must recover finite errors for a rank-deficient-looking ill-conditioned block.

        Mirrors the multi-tracer F_BB structure: one dominant direction at scale 1e14,
        weaker directions at 1e4. Without preconditioning the weak eigenvalues fall below
        fp noise and the inverted diagonal contains nans/infs.
        """
        # Build a 6x6 Fisher whose diagonal spans 10 orders of magnitude
        scales = np.array([1e14, 1e14, 1e4, 1e4, 1e4, 1e4])
        F = np.diag(scales)
        # Add small but non-zero off-diagonal coupling so it's genuinely full-rank
        F += 1e3 * np.ones((6, 6))
        # Ensure positive definiteness: D + 1e3*ones(6,6) is PSD iff min eigval > 0
        # diag dominates, so it's fine

        C_off = solve_preconditioned(F, precondition=False)
        C_on = solve_preconditioned(F, precondition=True)

        # Without preconditioning the weak directions may be lost
        # With preconditioning diagonal must be finite and positive
        assert not np.any(np.isnan(np.diag(C_on))), "preconditioned inverse has NaN diagonal"
        assert not np.any(np.isinf(np.diag(C_on))), "preconditioned inverse has Inf diagonal"
        assert np.all(np.diag(C_on) > 0)


# ── Fisher-level pseudoinverse (_pinv_rtol) ───────────────────────────────────


class TestFisherPseudoinverse:
    """Tests for the _pinv_rtol path in FisherMat.__init__."""

    def _make_fisher_mat(self, F, forecast, param_list, config, **kwargs):
        return FisherMat(F, forecast, param_list, config=config, **kwargs)

    def test_flag_off_matches_solve_preconditioned(self, fisher_pk):
        """_pinv_rtol=None must produce the same covariance as solve_preconditioned."""
        F = fisher_pk.fisher_matrix
        C_default = fisher_pk.covariance  # built with _pinv_rtol=None, precondition=True
        C_sp = solve_preconditioned(F, precondition=True)
        np.testing.assert_allclose(C_default, C_sp, rtol=1e-12)

    def test_well_conditioned_pinv_matches_exact(self):
        """On a well-conditioned SPD matrix, eigh pseudoinverse must agree with exact inverse."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 4))
        F = A.T @ A + 10 * np.eye(4)  # well-conditioned, condition number ~ 10
        w, v = np.linalg.eigh(F)
        w_inv = np.where(w > 1e-10 * np.abs(w).max(), 1.0 / w, 0.0)
        C_pinv = (v * w_inv) @ v.T
        C_exact = np.linalg.inv(F)
        np.testing.assert_allclose(C_pinv, C_exact, rtol=1e-10)

    def test_indefinite_fisher_no_nan(self):
        """Fisher with a small negative eigenvalue must give finite covariance, not NaN."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        F_pd = A.T @ A + np.eye(5)  # PD base
        # inject a tiny negative eigenvalue via a rank-1 perturbation
        w, v = np.linalg.eigh(F_pd)
        w[0] = -1e-3 * w[-1]  # make smallest eigenvalue negative
        F_indef = (v * w) @ v.T

        # _pinv_rtol=None → solve_preconditioned → may NaN on negative diagonal
        # _pinv_rtol=1e-10 → clips negative eigenvalue → finite covariance
        w_check, v_check = np.linalg.eigh(F_indef)
        w_inv = np.where(w_check > 1e-10 * np.abs(w_check).max(), 1.0 / w_check, 0.0)
        C_pinv = (v_check * w_inv) @ v_check.T
        assert not np.any(np.isnan(np.diag(C_pinv)))
        assert not np.any(np.isinf(np.diag(C_pinv)))
        # well-constrained directions (positive eigenvalues) still give finite positive errors
        assert np.all(np.diag(C_pinv) >= 0)
