"""Tests for cosmo_wap.forecast.Sampler — per-bin nuisance params, multi-tracer and the LF prior.

GR2 is included in `terms` for the multi-tracer cases so the per-bin Q/be amplitudes
actually enter the theory (NPP depends only on b_1).
"""

import numpy as np
import pytest

pytest.importorskip("cobaya")

from cosmo_wap.forecast.sampler import Sampler
from cosmo_wap.lib.lf_priors import LFBiasPrior


def fid_vals(s):
    return {p: s.fiducial[p] for p in s.param_list}


@pytest.fixture(scope="module")
def sampler_st(forecast):
    return Sampler(forecast, ["fNL"], terms=["NPP"], pkln=[0], per_bin_params=["b_1"], fisher_covmat=False, drag=False)


@pytest.fixture(scope="module")
def sampler_mt(forecast_mt):
    return Sampler(
        forecast_mt,
        ["fNL"],
        terms=["NPP", "GR2"],
        pkln=[0],
        all_tracer=True,
        per_bin_params=["Xb_1", "YQ"],
        fisher_covmat=False,
        drag=False,
    )


# ── single tracer (regression) ───────────────────────────────────────────────


class TestSingleTracerPerBin:
    def test_fiducial_likelihood_zero(self, sampler_st):
        assert abs(sampler_st.get_likelihood(**fid_vals(sampler_st))) < 1e-10

    def test_per_bin_perturbation_moves_likelihood(self, sampler_st):
        v = fid_vals(sampler_st)
        v["b_1_0"] = 1.05
        assert sampler_st.get_likelihood(**v) < -1e-3
        # bias scaling must be fully restored after the perturbed call
        assert abs(sampler_st.get_likelihood(**fid_vals(sampler_st))) < 1e-10

    def test_tracer_prefix_requires_multi_tracer(self, forecast):
        with pytest.raises(ValueError):
            Sampler(forecast, ["fNL"], terms=["NPP"], pkln=[0], per_bin_params=["Xb_1"], fisher_covmat=False)

    def test_unknown_per_bin_param_raises(self, forecast):
        with pytest.raises(NotImplementedError):
            Sampler(forecast, ["fNL"], terms=["NPP"], pkln=[0], per_bin_params=["nonsense"], fisher_covmat=False)


# ── term amplitudes & fisher proposal ────────────────────────────────────────


class TestTermAmplitude:
    def test_fiducial_likelihood_zero(self, forecast):
        """GR2 sits after fNL in param_list - its theory must be indexed by bin, not parameter position."""
        s = Sampler(forecast, ["fNL", "GR2"], terms=["NPP"], pkln=[0], fisher_covmat=False, drag=False)
        assert abs(s.get_likelihood(**fid_vals(s))) < 1e-10

    def test_amplitude_moves_likelihood(self, forecast):
        s = Sampler(forecast, ["fNL", "GR2"], terms=["NPP"], pkln=[0], fisher_covmat=False, drag=False)
        v = fid_vals(s)
        v["GR2"] = 1.5
        assert s.get_likelihood(**v) < -1e-6


class TestFisherCovmat:
    def test_unconstrained_param_raises(self, forecast, monkeypatch):
        """An unconstrained param must fail loudly, not poison the proposal covmat."""
        from types import SimpleNamespace

        s = Sampler(forecast, ["fNL"], terms=["NPP"], pkln=[0], fisher_covmat=False, drag=False)
        fake = SimpleNamespace(covariance=np.diag([1e25]), param_list=["fNL"])
        monkeypatch.setattr(s.forecast, "get_fish", lambda *a, **k: fake)
        with pytest.raises(ValueError, match="no constraint"):
            s.get_fisher_covmat()


# ── linked biases: b_phi and b_phi_e ─────────────────────────────────────────


@pytest.fixture(scope="module")
def sampler_linked(forecast):
    return Sampler(
        forecast,
        ["fNL", "A_b_phi_e"],
        terms=["NPP", "GR2", "Loc"],
        pkln=[0, 2],
        per_bin_params=["b_phi", "b_phi_e"],
        fisher_covmat=False,
        drag=False,
    )


class TestLinkedBias:
    def test_fiducial_likelihood_zero(self, sampler_linked):
        assert abs(sampler_linked.get_likelihood(**fid_vals(sampler_linked))) < 1e-10

    def test_amplitude_fiducials_are_one(self, sampler_linked):
        for p in ["A_b_phi_e", "b_phi_0", "b_phi_e_0"]:
            assert sampler_linked.fiducial[p] == 1.0

    def test_tighter_prior_than_lum_default(self, sampler_linked):
        assert sampler_linked.prior_dict["b_phi_e_0"]["prior"] == {"min": -10, "max": 10}
        assert sampler_linked.prior_dict["A_b_phi_e"]["prior"] == {"min": -10, "max": 10}

    @pytest.mark.parametrize("param", ["b_phi_0", "b_phi_e_0", "A_b_phi_e"])
    def test_perturbation_moves_likelihood_and_restores(self, sampler_linked, forecast, param):
        """b_phi only enters multiplied by fNL, so perturb around a non-zero fNL for all three."""
        survey = forecast.cosmo_funcs.survey[0]
        before = (survey.be(1.0), survey.loc.b_01(1.0))

        base = fid_vals(sampler_linked) | {"fNL": 5.0}
        logl_base = sampler_linked.get_likelihood(**base)
        assert sampler_linked.get_likelihood(**(base | {param: 1.1})) < logl_base - 1e-8

        # the sampler edits the cached (shared) survey in place - it must undo every edit
        assert (survey.be(1.0), survey.loc.b_01(1.0)) == before
        assert abs(sampler_linked.get_likelihood(**fid_vals(sampler_linked))) < 1e-10

    def test_b_phi_needs_nonzero_fnl(self, sampler_linked):
        """At fNL=0 a per-bin b_phi is invisible - the whole PNG contribution is proportional to fNL."""
        v = fid_vals(sampler_linked)
        assert sampler_linked.get_likelihood(**(v | {"b_phi_0": 1.1})) == pytest.approx(
            sampler_linked.get_likelihood(**v), abs=1e-10
        )

    def test_b_phi_e_also_moves_b_e(self, sampler_linked, forecast):
        """Inside the per-bin context b_phi scales and b_e picks up the same additive shift."""
        survey = forecast.cosmo_funcs.survey[0]
        be0, b_phi0 = survey.be(1.0), survey.loc.b_01(1.0)

        vals = [1.1 if p == "b_phi_e_0" else sampler_linked.fiducial[p] for p in sampler_linked.param_list]
        with sampler_linked._per_bin_bias(forecast.cosmo_funcs, vals, 0):
            assert survey.loc.b_01(1.0) == pytest.approx(1.1 * b_phi0)
            assert survey.be(1.0) == pytest.approx(be0 + 0.1 * b_phi0)
        assert (survey.be(1.0), survey.loc.b_01(1.0)) == (be0, b_phi0)

    def test_b_phi_leaves_b_e_alone(self, sampler_linked, forecast):
        survey = forecast.cosmo_funcs.survey[0]
        be0, b_phi0 = survey.be(1.0), survey.loc.b_01(1.0)

        vals = [1.1 if p == "b_phi_0" else sampler_linked.fiducial[p] for p in sampler_linked.param_list]
        with sampler_linked._per_bin_bias(forecast.cosmo_funcs, vals, 0):
            assert survey.loc.b_01(1.0) == pytest.approx(1.1 * b_phi0)
            assert survey.be(1.0) == pytest.approx(be0)
        assert (survey.be(1.0), survey.loc.b_01(1.0)) == (be0, b_phi0)


# ── multi-tracer per-bin params ──────────────────────────────────────────────


class TestMultiTracerPerBin:
    def test_prefixed_params_inherit_base_prior(self, sampler_mt):
        assert sampler_mt.prior_dict["Xb_1_0"]["prior"] == {"min": 0.8, "max": 1.2}  # tight b_1 prior
        assert sampler_mt.prior_dict["YQ_0"]["prior"] == {"min": -50, "max": 50}  # wide selection-function prior

    def test_fiducial_likelihood_zero(self, sampler_mt):
        assert abs(sampler_mt.get_likelihood(**fid_vals(sampler_mt))) < 1e-10

    def test_tracer_specific_perturbations(self, sampler_mt):
        v = fid_vals(sampler_mt)
        v["Xb_1_0"] = 1.05
        l_x = sampler_mt.get_likelihood(**v)
        v = fid_vals(sampler_mt)
        v["YQ_0"] = 1.5
        l_y = sampler_mt.get_likelihood(**v)
        assert l_x < -1e-3
        assert l_y < -1e-6
        assert not np.isclose(l_x, l_y)
        # bias scaling must be fully restored after the perturbed calls
        assert abs(sampler_mt.get_likelihood(**fid_vals(sampler_mt))) < 1e-10

    def test_tracer_views_are_reused_across_calls(self, sampler_mt):
        """The per-combination views must be the same objects at an unchanged cosmology.

        They are cached on the cosmology object rather than rebuilt per likelihood call,
        because each rebuild is a fresh utils.copy and id(cosmo_funcs) is part of the bk
        coefficient-table cache key - rebuilding them makes every table lookup miss, so
        under all_tracer the tables get built, used once and discarded on every call.
        Nothing about the answer changes when that happens, only the cost, so this needs
        asserting rather than leaving to a timing to notice.
        """
        vals = [sampler_mt.fiducial[p] for p in sampler_mt.param_list]
        cf_a = sampler_mt.update_cosmo_funcs(vals)
        cf_b = sampler_mt.update_cosmo_funcs(vals)
        assert cf_a is cf_b
        assert cf_a.cf_mat_bk is cf_b.cf_mat_bk
        assert cf_a.cf_mat_bk[0][0][1] is cf_b.cf_mat_bk[0][0][1]
        assert cf_a.cf_mat[0][1] is cf_b.cf_mat[0][1]

        # moving a non-cosmology parameter is exactly the fast step the cache exists for,
        # so it too must land on the views already built
        moved = list(vals)
        moved[sampler_mt.param_list.index("fNL")] = vals[sampler_mt.param_list.index("fNL")] + 1.0
        assert sampler_mt.update_cosmo_funcs(moved).cf_mat_bk is cf_a.cf_mat_bk

    def test_fisher_proposal_covmat_with_per_bin(self, forecast_mt):
        """get_fisher_covmat Schur-marginalises the per-bin params out of the global block."""
        s = Sampler(
            forecast_mt,
            ["Y_b_1"],  # constrained: only X's per-bin b_1 amplitudes are marginalised
            terms=["NPP", "GR2"],
            pkln=[0],
            all_tracer=True,
            per_bin_params=["Xb_1", "YQ"],
            fisher_covmat=False,
            drag=False,
        )
        covmat, params = s.get_fisher_covmat()
        assert params == ["Y_b_1"]
        assert covmat.shape == (1, 1)
        assert np.all(np.isfinite(covmat)) and covmat[0, 0] > 0

    def test_fisher_proposal_unconstrained_raises(self, sampler_mt):
        """fNL carries no information in this tiny forecast - must raise, not return a garbage covmat."""
        with pytest.raises(ValueError, match="no constraint"):
            sampler_mt.get_fisher_covmat()

    def test_bk_multipole_ordering(self, forecast_mt):
        """Multi-tracer bk theory vector must be l-major to match the data/covariance
        ordering — a tracer-major theory vector gives a nonzero fiducial likelihood."""
        s = Sampler(
            forecast_mt,
            ["fNL"],
            terms=["NPP"],
            pkln=None,
            bkln=[0, 1],
            all_tracer=True,
            fisher_covmat=False,
            drag=False,
        )
        assert abs(s.get_likelihood(**fid_vals(s))) < 1e-10


# ── multi-tracer LF prior ────────────────────────────────────────────────────


class TestMultiTracerLFPrior:
    @pytest.fixture(scope="class")
    def sampler_lf(self, forecast_mt):
        bias_prior = LFBiasPrior.from_survey(forecast_mt.cosmo_funcs.survey_params[0], n_samples=200, seed=0)
        return Sampler(
            forecast_mt,
            ["fNL"],
            terms=["NPP", "GR2"],
            pkln=[0],
            all_tracer=True,
            per_bin_params=["Xbe", "XQ", "Ybe", "YQ"],
            lf_prior=bias_prior,
            fisher_covmat=False,
            drag=False,
        )

    def test_prior_registered_over_per_bin_names(self, sampler_lf):
        lf = sampler_lf.info["likelihood"]["lf_prior"]
        assert sorted(lf["input_params"]) == sorted(sampler_lf.per_bin_names)

    def test_prior_zero_at_fiducial_and_penalises(self, sampler_lf):
        lf = sampler_lf.info["likelihood"]["lf_prior"]
        fid = {p: 1.0 for p in lf["input_params"]}
        assert abs(lf["external"](**fid)) < 1e-12
        pert = dict(fid, Xbe_0=1.5)
        assert lf["external"](**pert) < 0

    def test_fiducial_likelihood_zero(self, sampler_lf):
        assert abs(sampler_lf.get_likelihood(**fid_vals(sampler_lf))) < 1e-10


# ── numeric-mu kernels ───────────────────────────────────────────────────────


class TestKernels:
    """The kernel signal must enter the data vector exactly once, whatever `terms` holds.

    It used to be added once per analytic term - each one reached the term branch of
    five_point_stencil with kernels still in kwargs - so the fiducial likelihood was only
    zero for a single term.
    """

    KERNELS = ["N", "LP", "I"]
    ARGS = dict(pkln=[0, 1], bkln=None, fisher_covmat=False, drag=False)

    @pytest.mark.parametrize("terms", [None, ["WAGR"], ["WAGR", "RRGR"], ["WAGR", "RRGR", "Loc"]])
    def test_fiducial_likelihood_zero(self, forecast, terms):
        """terms=None is the kernels-only model - the signal is the kernels alone."""
        s = Sampler(forecast, ["fNL"], terms=terms, kernels=self.KERNELS, **self.ARGS)
        assert abs(s.get_likelihood(**fid_vals(s))) < 1e-10

    def test_analytic_and_kernel_parts_add(self, forecast):
        """data(terms+kernels) - data(kernels only) == data(terms only) - i.e. one kernel copy."""
        terms = ["WAGR", "RRGR"]
        d_both = Sampler(forecast, ["fNL"], terms=terms, kernels=self.KERNELS, **self.ARGS).data[0]
        d_kern = Sampler(forecast, ["fNL"], terms=None, kernels=self.KERNELS, **self.ARGS).data[0]
        d_terms = Sampler(forecast, ["fNL"], terms=terms, **self.ARGS).data[0]

        for i in range(forecast.N_bins):
            # WAGR/RRGR have no dipole, so l=1 subtracts two equal kernel rows - atol floors that noise
            atol = 1e-12 * np.abs(d_both[i]["pk"]).max()
            np.testing.assert_allclose(d_both[i]["pk"] - d_kern[i]["pk"], d_terms[i]["pk"], rtol=1e-10, atol=atol)
