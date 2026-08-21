"""Tests for cosmo_wap.forecast.fisher — FisherMat symmetry, PD, covariance, errors, correlation."""

import numpy as np
import pytest

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


# ── Linked biases: b_phi and b_phi_e ─────────────────────────────────────────


class TestLinkedBias:
    """b_phi is survey.loc.b_01, b_phi_e shifts b_phi additively and b_e by f(z)/2 of that.

    fNL is non-zero throughout - b_01 only ever appears multiplied by fNL in the PNG terms,
    so the b_phi half of the coupling would not enter at all at the default fNL=0.
    """

    TERMS = ["NPP", "GR2", "Loc"]

    def derivs(self, forecast, params):
        pk_bin = forecast.get_pk_bin(0)
        return {p: np.ravel(pk_bin.get_data_vector(self.TERMS, [0, 2], param=p, fNL=5.0)) for p in params}

    def test_b_phi_e_is_b_phi_plus_b_e(self, forecast):
        """The b_phi_e derivative minus the b_phi derivative must be exactly along d/db_e."""
        d = self.derivs(forecast, ["b_phi", "b_phi_e", "be"])
        extra = d["b_phi_e"] - d["b_phi"]
        cos = np.dot(extra, d["be"]) / np.linalg.norm(extra) / np.linalg.norm(d["be"])
        assert cos == pytest.approx(1.0, abs=1e-6)

    def test_global_amplitude_matches_per_bin_direction(self, forecast):
        """A_b_phi_e is the same direction in data space as the absolute b_phi_e, just rescaled."""
        d = self.derivs(forecast, ["A_b_phi_e", "b_phi_e"])
        cos = np.dot(d["A_b_phi_e"], d["b_phi_e"]) / np.linalg.norm(d["A_b_phi_e"]) / np.linalg.norm(d["b_phi_e"])
        assert cos == pytest.approx(1.0, abs=1e-6)

    def test_b_phi_vanishes_at_zero_fnl(self, forecast):
        """b_01 only enters multiplied by fNL, so at fNL=0 the b_phi derivative is identically zero."""
        pk_bin = forecast.get_pk_bin(0)
        d = np.ravel(pk_bin.get_data_vector(self.TERMS, [0, 2], param="b_phi", fNL=0.0))
        # scale set by b_phi_e, which still moves through b_e at fNL=0 - b_phi is pure round-off
        scale = np.linalg.norm(pk_bin.get_data_vector(self.TERMS, [0, 2], param="b_phi_e", fNL=0.0))
        assert np.linalg.norm(d) < 1e-8 * scale

    def test_per_bin_marginalisation_widens_fnl(self, forecast):
        """Marginalising a per-bin b_phi_e must degrade sigma(fNL), never tighten it."""
        base = forecast.get_fish(["fNL"], terms=self.TERMS, pkln=[0, 2], fNL=5.0, verbose=False)
        marg = forecast.get_fish(
            ["fNL"], terms=self.TERMS, pkln=[0, 2], per_bin_params=["b_phi_e"], fNL=5.0, verbose=False
        )
        assert marg.errors[0] > base.errors[0]

    def test_fiducial_is_b_phi(self, forecast):
        """The absolute-units fiducial for both linked params is b_phi at the mid-redshift."""
        fish = forecast.get_fish(["b_phi", "b_phi_e"], terms=self.TERMS, pkln=[0], fNL=5.0, verbose=False)
        mid_z = (forecast.cosmo_funcs.z_min + forecast.cosmo_funcs.z_max) / 2
        expected = forecast.cosmo_funcs.survey[0].loc.b_01(mid_z)
        assert fish.fiducial["b_phi"] == pytest.approx(expected)
        assert fish.fiducial["b_phi_e"] == pytest.approx(expected)

    def test_unknown_per_bin_param_still_raises(self, forecast):
        with pytest.raises(NotImplementedError):
            forecast.get_fish(["fNL"], terms=["NPP"], pkln=[0], per_bin_params=["b_phi_nonsense"], verbose=False)


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


# ── PNG bias amplitudes: {X,Y,A}_{loc,eq,orth}_{b_01,b_11} ───────────────────


class TestPNGAmplitudeBias:
    """Global multiplicative amplitudes on the scale-dependent biases.

    b_11 (= b_phi_delta) is a second-order bias, so it only enters the bispectrum - these need
    a bk term, and like b_01 it always appears multiplied by fNL.
    """

    ARGS = dict(terms="NPP", pkln=[0], bkln=[0], bk_terms=["NPP", "Loc"], verbose=False)

    def test_names_cover_every_shape_and_order(self, forecast):
        expected = {f"{t}_{s}_{b}" for t in ("X", "Y", "A") for s in ("loc", "eq", "orth") for b in ("b_01", "b_11")}
        assert set(forecast.png_amp_bias) == expected

    def test_amplitude_fiducials_are_one(self, forecast):
        """They multiply the bias, so the fiducial is 1 - a 0 would put the truth marker (and
        the drag-tuning base point) at a bias of zero."""
        fish = forecast.get_fish(["fNL_loc", "A_loc_b_01", "A_loc_b_11"], fNL=5.0, **self.ARGS)
        for p in ["A_loc_b_01", "A_loc_b_11"]:
            assert fish.fiducial[p] == 1.0

    def test_b11_amplitude_is_constrained(self, forecast):
        fish = forecast.get_fish(["fNL_loc", "A_loc_b_11"], fNL=5.0, **self.ARGS)
        assert np.all(np.linalg.eigvalsh(fish.fisher_matrix) > 0)
        assert np.all(np.isfinite(fish.errors)) and np.all(fish.errors > 0)

    def test_marginalising_b11_widens_fnl(self, forecast):
        base = forecast.get_fish(["fNL_loc"], fNL=5.0, **self.ARGS)
        marg = forecast.get_fish(["fNL_loc", "A_loc_b_11"], fNL=5.0, **self.ARGS)
        assert marg.errors[0] > base.errors[0]

    def test_b11_is_a_distinct_direction_from_b01(self, forecast):
        """Both are PNG biases on the same term - if b_11 were being routed onto b_01 (the
        name parsing is positional) the two derivatives would be exactly parallel."""
        bk_bin = forecast.get_bk_bin(0)
        d = {
            p: np.ravel(bk_bin.get_data_vector(["NPP", "Loc"], [0], param=p, fNL=5.0))
            for p in ("A_loc_b_01", "A_loc_b_11")
        }
        cos = (
            np.dot(d["A_loc_b_01"], d["A_loc_b_11"]) / np.linalg.norm(d["A_loc_b_01"]) / np.linalg.norm(d["A_loc_b_11"])
        )
        assert abs(cos) < 0.99

    def test_b11_vanishes_at_zero_fnl(self, forecast):
        bk_bin = forecast.get_bk_bin(0)
        d0 = np.ravel(bk_bin.get_data_vector(["NPP", "Loc"], [0], param="A_loc_b_11", fNL=0.0))
        d5 = np.ravel(bk_bin.get_data_vector(["NPP", "Loc"], [0], param="A_loc_b_11", fNL=5.0))
        assert np.linalg.norm(d0) < 1e-8 * np.linalg.norm(d5)
