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


# ── Selection functions: n_evol stored, b_e derived ──────────────────────────


def _gamma(cf, ti=0):
    """Doppler coefficient gamma = gr1/(H f) of tracer ti, from the betas."""
    from cosmo_wap.lib.betas import interpolate_beta_funcs

    zz = cf.survey[ti].z_survey[2:-2]
    beta = interpolate_beta_funcs(cf, ti)
    return zz, beta[0](zz) / (cf.H_c(zz) * cf.f(zz))


class TestSelectionFunctions:
    """A tracer with a K-correction stores n_evol = dlnN/dln(1+z) and Q, and derives
    b_e = -n_evol - dlnH/dln(1+z) + 2/(chi H) - c_Q Q at the current cosmology - which makes the
    Doppler coefficient gamma = -(1 + n_evol) - (4 + K')Q free of cosmology."""

    TERMS = ["NPP", "GR2", "Loc"]

    def derivs(self, forecast, params, basis):
        pk_bin = forecast.get_pk_bin(0)
        forecast.set_selection_basis(basis)
        try:
            return {p: np.ravel(pk_bin.get_data_vector(self.TERMS, [0, 2], param=p, fNL=5.0)) for p in params}
        finally:
            forecast.set_selection_basis([])  # session fixture - leave it in the (n_evol, Q) basis

    def test_derived_b_e_is_the_surveys(self, cosmo_funcs, survey_params):
        tr = cosmo_funcs.survey[0]
        assert "n_evol" in tr.__dict__
        zz = tr.z_survey[1:-1]
        np.testing.assert_allclose(tr.be(zz), survey_params.be(zz), atol=1e-6)

    def test_gamma_from_n_evol_and_Q(self, cosmo_funcs):
        tr = cosmo_funcs.survey[0]
        zz, gamma = _gamma(cosmo_funcs)
        Kp = tr.c_Q(zz) - 2 * (1 + 1 / (cosmo_funcs.comoving_dist(zz) * cosmo_funcs.H_c(zz)))
        np.testing.assert_allclose(gamma, -(1 + tr.n_evol(zz)) - (4 + Kp) * tr.Q(zz), atol=1e-6)

    def test_cosmology_change_holds_n_evol(self, cosmo_funcs):
        """An adopted survey keeps n_evol and Q, so b_e moves with the cosmology but gamma does not.
        Omega_m, as chi H is in units of h and barely moves with h alone."""
        import cosmo_wap as cw
        from cosmo_wap.lib.utils import get_cosmo

        cf_h = cw.ClassWAP(get_cosmo(Omega_m=0.35), fast=True, verbose=False).adopt_survey(cosmo_funcs)
        tr, tr_h = cosmo_funcs.survey[0], cf_h.survey[0]
        zz = tr.z_survey[2:-2]
        assert tr_h.n_evol is tr.n_evol and tr_h.Q is tr.Q
        assert np.abs(tr_h.be(zz) - tr.be(zz)).max() > 1e-3
        np.testing.assert_allclose(_gamma(cf_h)[1], _gamma(cosmo_funcs)[1], atol=1e-6)

    def test_b_e_param_is_minus_n_evol(self, forecast):
        d_be = self.derivs(forecast, ["be"], ["be", "Q"])["be"]
        d_n = self.derivs(forecast, ["n_evol"], ["n_evol", "Q"])["n_evol"]
        np.testing.assert_allclose(d_be, -d_n, rtol=1e-6, atol=1e-8 * np.abs(d_n).max())

    def test_Q_at_fixed_b_e_is_jacobian(self, forecast):
        """d/dQ|_be = d/dQ|_n_evol - c_Q d/dn_evol"""
        d_Qbe = self.derivs(forecast, ["Q"], ["be", "Q"])["Q"]
        d = self.derivs(forecast, ["Q", "n_evol"], ["n_evol", "Q"])
        cQ = forecast.cosmo_funcs.survey[0].c_Q(forecast.get_pk_bin(0).z_mid)
        expected = d["Q"] - cQ * d["n_evol"]
        np.testing.assert_allclose(d_Qbe, expected, rtol=1e-6, atol=1e-8 * np.abs(expected).max())

    def test_marginalised_fisher_is_basis_independent(self, forecast):
        """Without a prior (n_evol, Q) is only a reparametrisation of the per-bin (b_e, Q) block."""
        kw = dict(terms=self.TERMS, pkln=[0, 2], fNL=5.0, verbose=False)
        f_be = forecast.get_fish(["fNL"], per_bin_params=["be", "Q"], **kw)
        f_n = forecast.get_fish(["fNL"], per_bin_params=["n_evol", "Q"], **kw)
        forecast.set_selection_basis([])
        assert f_n.errors[0] == pytest.approx(f_be.errors[0], rel=1e-5)

    def test_be_and_n_evol_together_raise(self, forecast):
        with pytest.raises(ValueError):
            forecast.get_fish(["fNL"], terms=["NPP"], pkln=[0], per_bin_params=["n_evol", "be"], verbose=False)
        forecast.set_selection_basis([])

    def test_assigning_b_e_raises(self, cosmo_funcs):
        from cosmo_wap.lib.utils import copy

        with pytest.raises(AttributeError):
            copy(cosmo_funcs).survey[0].be = lambda zz: 0 * zz

    def test_unselected_tracer_keeps_b_e(self, cosmo):
        """Without a K-correction there is no c_Q, so b_e stays the survey's own and n_evol is refused."""
        import cosmo_wap as cw
        from cosmo_wap.lib.utils import selection_key

        tr = cw.ClassWAP(cosmo, cw.SurveyParams.DM_part(cosmo), verbose=False).survey[0]
        assert "n_evol" not in tr.__dict__ and "be" in tr.__dict__
        assert selection_key(tr, "be") is None
        with pytest.raises(ValueError):
            selection_key(tr, "n_evol")


# ── Preconditioning ───────────────────────────────────────────────────────────


class TestBkKernels:
    """bk_kernels=['N'] is the analytic NPP computed numerically - the Fisher must not tell them apart."""

    ARGS = dict(pkln=None, bkln=[0, 1, 2], verbose=False)

    @pytest.mark.parametrize("fc_name", ["forecast", "forecast_mt"])
    def test_matches_analytic(self, fc_name, request):
        fc = request.getfixturevalue(fc_name)
        all_tracer = fc_name == "forecast_mt"
        params = ["A_b_1", "A_b_2"]
        analytic = fc.get_fish(params, terms="NPP", all_tracer=all_tracer, **self.ARGS)
        numeric = fc.get_fish(params, terms=None, bk_kernels=["N"], all_tracer=all_tracer, **self.ARGS)
        np.testing.assert_allclose(numeric.fisher_matrix, analytic.fisher_matrix, rtol=1e-8)


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


# ── Planck prior ─────────────────────────────────────────────────────────────


def _dummy_fisher(forecast, params):
    """Identity FisherMat - just a carrier for param_list/cosmo_funcs to test the prior on."""
    from cosmo_wap.forecast.fisher import FisherMat

    return FisherMat(np.eye(len(params)), forecast, params, config={"bias": None})


class TestPlanckPrior:
    # published Planck 2018 (Table 2, arXiv:1807.06209) 1 sigma: TTTEEE+lowE+lensing and +BAO
    PUBLISHED = {
        False: {"Omega_m": 0.0073, "h": 0.0054, "n_s": 0.0042, "ln_A_s": 0.014, "sigma8": 0.0060},
        True: {"Omega_m": 0.0056, "h": 0.0042, "n_s": 0.0038, "ln_A_s": 0.014, "sigma8": 0.0060},
    }

    @pytest.mark.parametrize("bao", [False, True])
    def test_matches_published_errors(self, forecast, bao):
        params = ["Omega_m", "h", "ln_A_s", "n_s", "sigma8"]
        cov, got = _dummy_fisher(forecast, params).planck_cov(bao=bao)
        assert got == params
        sig = dict(zip(params, np.sqrt(np.diag(cov))))
        for p, s in self.PUBLISHED[bao].items():
            assert sig[p] == pytest.approx(s, rel=0.05), p

    def test_omega_m_h_degeneracy(self, forecast):
        cov, _ = _dummy_fisher(forecast, ["Omega_m", "h"]).planck_cov()
        assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) < -0.9

    def test_bao_tightens(self, forecast):
        fish = _dummy_fisher(forecast, ["Omega_m", "h"])
        assert np.all(np.diag(fish.planck_cov(bao=True)[0]) < np.diag(fish.planck_cov()[0]))

    def test_params_in_param_list_order_and_unconstrained_skipped(self, forecast):
        cov, got = _dummy_fisher(forecast, ["fNL_loc", "n_s", "Omega_b"]).planck_cov()
        assert got == ["n_s", "Omega_b"]
        assert cov.shape == (2, 2)

    def test_A_s_is_ln_A_s_rescaled(self, forecast, cosmo_funcs):
        cov_ln, _ = _dummy_fisher(forecast, ["ln_A_s", "n_s"]).planck_cov()
        cov_A, _ = _dummy_fisher(forecast, ["A_s", "n_s"]).planck_cov()
        J = np.diag([cosmo_funcs.A_s, 1.0])
        np.testing.assert_allclose(cov_A, J @ cov_ln @ J, rtol=1e-12)

    def test_add_planck_prior_only_touches_planck_block(self, forecast):
        params = ["fNL_loc", "Omega_m", "h"]
        fish = _dummy_fisher(forecast, params)
        cov, _ = fish.planck_cov()
        F = fish.add_planck_prior().fisher_matrix
        np.testing.assert_allclose(F[1:, 1:], np.eye(2) + np.linalg.inv(cov))
        np.testing.assert_allclose(F[0], [1, 0, 0])

    def test_add_planck_prior_noop(self, forecast):
        fish = _dummy_fisher(forecast, ["fNL_loc"])
        assert fish.add_planck_prior() is fish

    def test_add_chain_cov_with_ln_A_s(self, forecast):
        fish = _dummy_fisher(forecast, ["ln_A_s", "n_s", "fNL_loc"])
        c = fish.add_chain_cov()
        assert set(c.get_chain(c.get_names()[0]).samples.columns) >= {"ln_A_s", "n_s"}


# ── Best-fit bias: marginalised F^-1 B vs conditional B_i/F_ii ───────────────


class TestBestFitBias:
    ARGS = dict(terms=["NPP", "Loc"], pkln=[0, 2], bias_list="GR1", verbose=False)

    def test_bias_is_inverse_fisher_times_B(self, forecast):
        """Marginalised shift is F^-1 B, where B_i = F_ii * conditional shift."""
        fish = forecast.get_fish(["fNL", "n_s", "A_s"], **self.ARGS)
        cond = np.array([fish.conditional_bias[-1][p] for p in fish.param_list])
        B = np.diag(fish.fisher_matrix) * cond
        marg = np.array([fish.bias[-1][p] for p in fish.param_list])
        np.testing.assert_allclose(marg, fish.covariance @ B, rtol=1e-10)

    def test_single_param_marginal_equals_conditional(self, forecast):
        fish = forecast.get_fish(["fNL"], **self.ARGS)
        assert fish.bias[-1]["fNL"] == pytest.approx(fish.conditional_bias[-1]["fNL"], rel=1e-10)

    def test_prior_enters_marginal_shift(self, forecast):
        """A prior added after get_fish changes F, so bias is recomputed as (F + F_prior)^-1 B."""
        fish = forecast.get_fish(["fNL", "n_s", "A_s"], **self.ARGS)
        B = np.atleast_2d(fish.config["B"])[-1]
        wide = fish.add_gaussian_priors({"n_s": 0.01})
        np.testing.assert_allclose([wide.bias[-1][p] for p in wide.param_list], wide.covariance @ B, rtol=1e-10)
        assert wide.conditional_bias == fish.conditional_bias

    def test_multiple_terms_total_is_sum(self, forecast):
        args = dict(self.ARGS, bias_list=["GR1", "WS"])
        fish = forecast.get_fish(["fNL", "n_s"], **args)
        assert len(fish.bias) == 3
        for p in fish.param_list:
            assert fish.bias[-1][p] == pytest.approx(fish.bias[0][p] + fish.bias[1][p], rel=1e-10)

    def test_schur_matches_full_block(self, forecast):
        """Marginalising per-bin nuisance via the Schur complement gives the same global shift
        as inverting the full block Fisher matrix."""
        kw = dict(self.ARGS, per_bin_params=["b_1"])
        schur = forecast.get_fish(["fNL", "n_s"], marginalize_per_bin=True, **kw)
        full = forecast.get_fish(["fNL", "n_s"], marginalize_per_bin=False, **kw)
        for p in ["fNL", "n_s"]:
            assert schur.bias[-1][p] == pytest.approx(full.bias[-1][p], rel=1e-6)

    def test_to_S8_transforms_B(self, forecast):
        """S8 is a reparameterisation, so the Omega_m shift is unchanged and the S8 shift follows the chain rule."""
        fish = forecast.get_fish(["sigma8", "Omega_m"], **self.ARGS)
        s8 = fish.to_S8()
        cf = forecast.cosmo_funcs
        b = fish.bias[-1]
        assert s8.bias[-1]["Omega_m"] == pytest.approx(b["Omega_m"], rel=1e-8)
        dS8 = np.sqrt(cf.Omega_m / 0.3) * b["sigma8"] + cf.sigma8 / (2 * np.sqrt(0.3 * cf.Omega_m)) * b["Omega_m"]
        assert s8.bias[-1]["S8"] == pytest.approx(dS8, rel=1e-8)


# ── Save / load ──────────────────────────────────────────────────────────────


class TestSaveLoad:
    def _roundtrip(self, fish, forecast, path):
        fish.save(path)
        return type(fish).load(path, forecast)  # without the .npz np.savez adds

    def test_roundtrip(self, forecast, fisher_pk, tmp_path):
        loaded = self._roundtrip(fisher_pk, forecast, tmp_path / "fish")
        np.testing.assert_array_equal(loaded.fisher_matrix, fisher_pk.fisher_matrix)
        np.testing.assert_array_equal(loaded.errors, fisher_pk.errors)
        assert loaded.param_list == fisher_pk.param_list
        assert loaded.term == fisher_pk.term
        assert loaded.name == fisher_pk.name
        assert loaded.fiducial == fisher_pk.fiducial

    def test_roundtrip_per_bin(self, forecast, tmp_path):
        fish = forecast.get_fish(["A_b_1"], terms="NPP", pkln=[0], per_bin_params=["Q"], verbose=False)
        assert fish.per_bin_cov is not None  # Schur path - the array has to survive
        loaded = self._roundtrip(fish, forecast, tmp_path / "fish_pb")
        assert loaded.per_bin_param_list == fish.per_bin_param_list
        np.testing.assert_array_equal(loaded.per_bin_cov, fish.per_bin_cov)
        np.testing.assert_array_equal(loaded.get_per_bin_error("Q"), fish.get_per_bin_error("Q"))
