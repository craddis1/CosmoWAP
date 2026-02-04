"""Tests for cosmo_wap.survey_params — init, attributes, update, BF_split, other surveys."""
import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.survey_params import SurveyParams


# ── Euclid survey basics ─────────────────────────────────────────────────────

class TestEuclidInit:
    def test_has_required_attributes(self, survey_params):
        for attr in ("b_1", "n_g", "f_sky", "z_range", "Q", "be"):
            assert hasattr(survey_params, attr), f"missing {attr}"

    def test_z_range_is_two_element(self, survey_params):
        assert len(survey_params.z_range) == 2
        assert survey_params.z_range[0] < survey_params.z_range[1]

    def test_fsky_in_unit_interval(self, survey_params):
        assert 0 < survey_params.f_sky <= 1

    def test_bias_positive(self, survey_params):
        z = np.linspace(*survey_params.z_range, 20)
        b1 = survey_params.b_1(z)
        assert np.all(b1 > 0)

    def test_number_density_positive(self, survey_params):
        z = np.linspace(*survey_params.z_range, 20)
        ng = survey_params.n_g(z)
        assert np.all(ng > 0)


# ── update() ─────────────────────────────────────────────────────────────────

class TestSurveyUpdate:
    def test_returns_copy(self, survey_params):
        updated = survey_params.update(f_sky=0.123)
        assert updated is not survey_params

    def test_changes_target(self, survey_params):
        updated = survey_params.update(f_sky=0.123)
        assert updated.f_sky == pytest.approx(0.123)

    def test_does_not_mutate_original(self, survey_params):
        original_fsky = survey_params.f_sky
        survey_params.update(f_sky=0.999)
        assert survey_params.f_sky == pytest.approx(original_fsky)


# ── BF_split ─────────────────────────────────────────────────────────────────

class TestBFSplit:
    @pytest.fixture(scope="class")
    def split_tracers(self, survey_params):
        """Split Euclid at a flux cut brighter than the main cut."""
        return survey_params.BF_split(5e-16)

    def test_returns_two_tracers(self, split_tracers):
        assert len(split_tracers) == 2

    def test_density_sum(self, survey_params, split_tracers):
        """n_bright + n_faint ≈ n_total at the survey redshifts."""
        bright, faint = split_tracers
        z = np.linspace(*survey_params.z_range, 30)
        n_total = survey_params.n_g(z)
        n_sum = bright.n_g(z) + faint.n_g(z)
        np.testing.assert_allclose(n_sum, n_total, rtol=0.02)

    def test_bright_density_positive(self, split_tracers):
        bright = split_tracers[0]
        z = 1.2
        assert bright.n_g(z) > 0

    def test_faint_density_positive(self, split_tracers):
        faint = split_tracers[1]
        z = 1.2
        assert faint.n_g(z) > 0


# ── Other surveys ────────────────────────────────────────────────────────────

class TestOtherSurveys:
    @pytest.fixture(scope="class", params=["SKAO1", "SKAO2", "DM_part"])
    def survey(self, request, cosmo):
        cls = getattr(SurveyParams, request.param)
        return cls(cosmo)

    def test_fsky_valid(self, survey):
        assert 0 < survey.f_sky <= 1

    def test_bias_positive(self, survey):
        z = np.mean(survey.z_range)
        assert survey.b_1(z) > 0

    def test_ng_positive(self, survey):
        z = np.mean(survey.z_range)
        assert survey.n_g(z) > 0
