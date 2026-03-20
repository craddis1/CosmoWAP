"""Smoke test for compute_bias=True pathway (HOD/HMF bias computation)."""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.lib import utils


@pytest.fixture(scope="module")
def cosmo():
    return utils.get_cosmo()


@pytest.fixture(scope="module")
def cosmo_funcs_bias(cosmo):
    sp = cw.SurveyParams.Euclid(cosmo)
    return cw.ClassWAP(cosmo, sp, compute_bias=True, verbose=False)


class TestComputeBias:
    def test_bias_attributes_exist(self, cosmo_funcs_bias):
        survey = cosmo_funcs_bias.survey[0]
        assert callable(survey.b_1)
        assert callable(survey.b_2)
        assert callable(survey.g_2)
        assert callable(survey.n_g)

    def test_b1_positive(self, cosmo_funcs_bias):
        survey = cosmo_funcs_bias.survey[0]
        z_arr = np.linspace(survey.z_range[0], survey.z_range[1], 5)
        for z in z_arr:
            assert survey.b_1(z) > 0

    def test_b1_reasonable(self, cosmo_funcs_bias):
        """Linear bias should be O(1) for typical galaxy surveys."""
        survey = cosmo_funcs_bias.survey[0]
        z_mid = np.mean(survey.z_range)
        b1 = survey.b_1(z_mid)
        assert 0.5 < b1 < 5.0

    def test_png_biases_exist(self, cosmo_funcs_bias):
        survey = cosmo_funcs_bias.survey[0]
        for png_type in ["loc", "eq", "orth"]:
            obj = getattr(survey, png_type)
            assert callable(obj.b_01)
            assert callable(obj.b_11)

    def test_number_density_positive(self, cosmo_funcs_bias):
        survey = cosmo_funcs_bias.survey[0]
        z_mid = np.mean(survey.z_range)
        assert survey.n_g(z_mid) > 0


@pytest.fixture(scope="module")
def cosmo_funcs_bgs_hod(cosmo):
    sp = cw.SurveyParams.BGS(cosmo, m_c=20, type="HOD")
    return cw.ClassWAP(cosmo, sp, compute_bias=True, hod="Smith_BGS", verbose=False)


class TestSmithBGSHOD:
    def test_bias_attributes_exist(self, cosmo_funcs_bgs_hod):
        survey = cosmo_funcs_bgs_hod.survey[0]
        assert callable(survey.b_1)
        assert callable(survey.b_2)
        assert callable(survey.g_2)
        assert callable(survey.n_g)

    def test_Q_and_be_computed(self, cosmo_funcs_bgs_hod):
        survey = cosmo_funcs_bgs_hod.survey[0]
        assert callable(survey.Q)
        assert callable(survey.be)

    def test_b1_positive(self, cosmo_funcs_bgs_hod):
        survey = cosmo_funcs_bgs_hod.survey[0]
        z_arr = np.linspace(survey.z_range[0], survey.z_range[1], 5)
        for z in z_arr:
            assert survey.b_1(z) > 0

    def test_Q_reasonable(self, cosmo_funcs_bgs_hod):
        """Magnification bias Q should be positive and O(1)."""
        survey = cosmo_funcs_bgs_hod.survey[0]
        z_mid = np.mean(survey.z_range)
        assert 0 < survey.Q(z_mid) < 10

    def test_number_density_positive(self, cosmo_funcs_bgs_hod):
        survey = cosmo_funcs_bgs_hod.survey[0]
        z_mid = np.mean(survey.z_range)
        assert survey.n_g(z_mid) > 0

    def test_png_biases_exist(self, cosmo_funcs_bgs_hod):
        survey = cosmo_funcs_bgs_hod.survey[0]
        for png_type in ["loc", "eq", "orth"]:
            obj = getattr(survey, png_type)
            assert callable(obj.b_01)
            assert callable(obj.b_11)
