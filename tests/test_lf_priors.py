"""Tests for cosmo_wap.lib.lf_priors - the (n_evol, Q) components of LFBiasPrior and SelectionPrior.

Fisher and sampler use the small session fixtures (2 bins, kmax=0.1); the split is the fixed-cut
Euclid BF_split of conftest.forecast_mt, plus a fixed-fraction one here.
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.lib.lf_priors import LFBiasPrior, SelectionPrior, count_slope

TERMS = dict(terms=["NPP", "GR2", "Loc"], pkln=[0, 2], fNL=5.0, verbose=False)


def tracers(cf):
    return {["X", "Y"][s.t]: s for s in set(cf.survey)}


@pytest.fixture(scope="module")
def cf_fraction(cosmo):
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(fraction=0.3), verbose=False)


def test_count_slope_two_redshifts(cosmo):
    zz = np.array([1.0, 1.5])
    n_g = (1 + zz) ** 3 * cosmo.Hubble(zz) / cosmo.comoving_distance(zz) ** 2
    np.testing.assert_allclose(count_slope(cosmo, n_g, zz), 3.0, atol=1e-12)


# ── LFBiasPrior: components at the fiducial luminosity function ──────────────


class TestLFComponents:
    """At the fiducial LF parameters every component is the survey's own selection function - n_evol from
    count_slope(n_g) agrees with the n_evol a tracer inverts from its b_e (SetSurveyFunctions.set_selection)."""

    def check(self, prior, LF, cf):
        fid = prior.evaluate(LF)
        tr = tracers(cf)
        for i, name in enumerate(prior.components):
            base, label = (name[1:], name[0]) if name[0] in "XY" else (name, "X")
            ref = getattr(tr[label], base)(prior.z_grid)
            # np.gradient end points differ from the tracer's spline - compare the interior
            np.testing.assert_allclose(fid[i][2:-2], ref[2:-2], atol=1e-3, err_msg=name)

    def test_single(self, survey_params, cosmo_funcs):
        prior = LFBiasPrior.from_survey(survey_params, n_samples=20, seed=0)
        assert prior.components == ["n_evol", "be", "Q"]
        self.check(prior, survey_params.LF, cosmo_funcs)

    def test_split(self, forecast_mt):
        cf = forecast_mt.cosmo_funcs
        prior = LFBiasPrior.from_survey(cf.survey_params, n_samples=20, seed=0)
        assert prior.components == ["Xn_evol", "Xbe", "XQ", "Yn_evol", "Ybe", "YQ"]
        self.check(prior, cf.survey_params[0].LF, cf)

    def test_fraction_split(self, cf_fraction):
        """The bright cut re-solved per draw - at the fiducial it is the survey's own, n_evol shared."""
        prior = LFBiasPrior.from_survey(cf_fraction.survey_params, n_samples=20, seed=0)
        assert prior.components == ["n_evol", "Xbe", "XQ", "Ybe", "YQ"]
        self.check(prior, cf_fraction.survey_params[0].LF, cf_fraction)  # 'n_evol' against the bright
        n_evol, faint = prior.evaluate(prior.LF)[0], tracers(cf_fraction)["Y"]
        np.testing.assert_allclose(n_evol[2:-2], faint.n_evol(prior.z_grid)[2:-2], atol=1e-3)


class TestLFFisher:
    @pytest.fixture(scope="class")
    @classmethod
    def prior(cls, survey_params):
        return LFBiasPrior.from_survey(survey_params, n_samples=50, seed=0)

    @pytest.mark.parametrize("basis", [["n_evol", "Q"], ["be", "Q"]])
    def test_prior_tightens(self, forecast, prior, basis):
        """Either selection basis finds its components in the prior."""
        try:
            free = forecast.get_fish(["fNL"], per_bin_params=basis, **TERMS).errors[0]
            prior_err = forecast.get_fish(["fNL"], per_bin_params=basis, lf_prior=prior, **TERMS).errors[0]
        finally:
            forecast.set_selection_basis([])  # session fixture - leave it in the (n_evol, Q) basis
        assert prior_err < free


# ── SelectionPrior ────────────────────────────────────────────────────────────


class TestSelectionPrior:
    def test_single(self, cosmo_funcs):
        prior = SelectionPrior(cosmo_funcs, sigma_n=0.4, sigma_Q=0.1)
        zz = np.array([1.0, 1.5])
        cov = prior.covariance(zz)
        assert prior.components == ["n_evol", "Q"]
        np.testing.assert_allclose(cov[:, 0, 0], 0.4**2)
        np.testing.assert_allclose(cov[:, 1, 1], (0.1 * cosmo_funcs.survey[0].Q(zz)) ** 2)
        np.testing.assert_allclose(cov[:, 0, 1], 0.0)

    def test_split_propagates_measured_pair(self, forecast_mt):
        """(Q_B, Q_F) built from independent Q_T, Q_B: Q_T = r Q_B + (1 - r) Q_F recovers its own sigma, and
        is uncorrelated with Q_B."""
        cf = forecast_mt.cosmo_funcs
        prior = SelectionPrior(cf, sigma_Q=0.05)
        zz = np.array([1.0, 1.4])
        cov = prior.covariance(zz, ["XQ", "YQ"])

        tr = tracers(cf)
        n_B, n_F = tr["X"].n_g(zz), tr["Y"].n_g(zz)
        r = n_B / (n_B + n_F)
        Q_T = r * tr["X"].Q(zz) + (1 - r) * tr["Y"].Q(zz)
        w = np.stack([r, 1 - r], axis=-1)  # Q_T = w . (Q_B, Q_F)
        np.testing.assert_allclose(np.einsum("ki,kij,kj->k", w, cov, w), (0.05 * Q_T) ** 2, rtol=1e-10)
        np.testing.assert_allclose(np.einsum("kj,kj->k", cov[:, 0, :], w), 0.0, atol=1e-12)
        np.testing.assert_allclose(cov[:, 0, 0], (0.05 * tr["X"].Q(zz)) ** 2)

    def test_split_n_evol(self, forecast_mt):
        prior = SelectionPrior(forecast_mt.cosmo_funcs, sigma_n=0.3)
        cov = prior.covariance(np.array([1.2]), ["n_evol", "Xn_evol", "Yn_evol", "XQ"])[0]
        np.testing.assert_allclose(np.diag(cov)[:3], 0.3**2)
        np.testing.assert_allclose(cov[:3, :3] - np.diag(np.diag(cov)[:3]), 0.0)
        np.testing.assert_allclose(cov[:3, 3], 0.0)

    def test_needs_a_split(self, cosmo_funcs):
        """Two tracers that are not a BF_split of one sample have no (Q_T, Q_B) to propagate."""

        class Tracer:
            def __init__(self, t):
                self.t, self.Q, self.n_g = t, cosmo_funcs.survey[0].Q, cosmo_funcs.survey[0].n_g

        class CF:
            multi_tracer = True
            survey = [Tracer(0), Tracer(1)]
            survey_params = [object(), object()]

        with pytest.raises(ValueError):
            SelectionPrior(CF())

    def test_fisher(self, forecast, cosmo_funcs):
        try:
            free = forecast.get_fish(["fNL"], per_bin_params=["n_evol", "Q"], **TERMS).errors[0]
            prior = forecast.get_fish(
                ["fNL"], per_bin_params=["n_evol", "Q"], lf_prior=SelectionPrior(cosmo_funcs), **TERMS
            ).errors[0]
        finally:
            forecast.set_selection_basis([])
        assert prior < free


class TestSamplerOffset:
    """n_evol is sampled as an offset (fiducial 0) - the prior penalty must not divide by its fiducial."""

    @pytest.fixture(scope="class")
    @classmethod
    def lf(cls, forecast, cosmo_funcs):
        pytest.importorskip("cobaya")
        from cosmo_wap.forecast.sampler import Sampler

        sampler = Sampler(
            forecast,
            ["fNL"],
            terms=["NPP", "GR2"],
            pkln=[0],
            per_bin_params=["n_evol", "Q"],
            lf_prior=SelectionPrior(cosmo_funcs, sigma_n=0.5, sigma_Q=0.05),
            fisher_covmat=False,
            drag=False,
        )
        yield sampler.info["likelihood"]["lf_prior"], sampler.fiducial
        forecast.set_selection_basis([])

    def test_zero_at_fiducial(self, lf):
        prior, fid = lf
        assert prior["external"](**{p: fid[p] for p in prior["input_params"]}) == pytest.approx(0.0, abs=1e-12)

    def test_one_sigma(self, lf):
        """SelectionPrior is diagonal in (n_evol, Q): a 1-sigma step on either costs exactly 1/2."""
        prior, fid = lf
        fid = {p: fid[p] for p in prior["input_params"]}
        assert fid["n_evol_0"] == 0.0
        assert prior["external"](**dict(fid, n_evol_0=0.5)) == pytest.approx(-0.5)
        assert prior["external"](**dict(fid, Q_0=1.05)) == pytest.approx(-0.5)
