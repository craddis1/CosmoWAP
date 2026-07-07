"""Tests for cosmo_wap.forecast.Sampler — per-bin nuisance params, multi-tracer and the LF prior.

GR2 is included in `terms` for the multi-tracer cases so the per-bin Q/be amplitudes
actually enter the theory (NPP depends only on b_1).
"""

import numpy as np
import pytest

pytest.importorskip("cobaya")

import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.sampler import Sampler
from cosmo_wap.lib.lf_priors import LFBiasPrior


def fid_vals(s):
    return {p: s.fiducial[p] for p in s.param_list}


@pytest.fixture(scope="module")
def forecast_mt(cosmo):
    """Small multi-tracer (bright/faint split) FullForecast."""
    survey = cw.SurveyParams.Euclid(cosmo).BF_split(6e-16)
    cf = cw.ClassWAP(cosmo, survey, verbose=False)
    return FullForecast(cf, kmax_func=0.1, s_k=2, N_bins=2)


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

    def test_fisher_proposal_covmat_with_per_bin(self, sampler_mt):
        """get_fisher_covmat Schur-marginalises the per-bin params out of the global block."""
        covmat, params = sampler_mt.get_fisher_covmat()
        assert params == ["fNL"]
        assert covmat.shape == (1, 1)
        assert np.all(np.isfinite(covmat)) and covmat[0, 0] > 0

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
