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
    sp = cw.SurveyParams.BGS(cosmo, cut=20, flag="HOD")
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


class TestBiasMemoisation:
    """utils.array_memo on HMF.n_h and YP.HOD - it caches the arrays PBBias asks for ~159
    times per build. A wrong hit would shift every galaxy bias silently, so check both that
    a hit equals a fresh computation and that different arguments do not get one.

    The cache attribute name is array_memo's: _<method>_memo on the instance.
    """

    @pytest.fixture(scope="class")
    def pbb(self, cosmo, cosmo_funcs_bias):
        # ClassWAP keeps only the splines PBBias produces, not the object, so build one
        from cosmo_wap.HOD.peak_background_bias import PBBias

        return PBBias(cosmo_funcs_bias, cw.SurveyParams.Euclid(cosmo))

    def test_n_h_cached_value_matches_a_fresh_computation(self, pbb):
        zz = np.linspace(pbb.z_samps[0], pbb.z_samps[-1], 12)
        first = pbb.hmf.n_h(zz)
        assert pbb.hmf.n_h(zz) is first

        pbb.hmf._n_h_memo.clear()
        fresh = pbb.hmf.n_h(zz)
        assert fresh is not first
        np.testing.assert_array_equal(fresh, first)

    def test_n_h_hits_on_an_equal_but_distinct_array(self, pbb):
        """Keyed by value, not identity - a rebuilt but equal array must still hit."""
        zz = np.linspace(pbb.z_samps[0], pbb.z_samps[-1], 7)
        assert pbb.hmf.n_h(zz.copy()) is pbb.hmf.n_h(zz)

    def test_n_h_does_not_serve_a_hit_for_other_redshifts(self, pbb):
        lo = pbb.hmf.n_h(np.linspace(0.9, 1.2, 6))
        hi = pbb.hmf.n_h(np.linspace(1.4, 1.7, 6))
        assert lo is not hi
        assert not np.allclose(lo, hi)

    def test_hod_cached_value_matches_a_fresh_computation(self, pbb):
        args = (pbb.z_samps, *pbb.hod.get_hod_params(pbb.z_samps, pbb.cut))
        first = pbb.hod.HOD(*args)
        assert pbb.hod.HOD(*args) is first

        pbb.hod._HOD_memo.clear()
        np.testing.assert_array_equal(pbb.hod.HOD(*args), first)

    def test_hod_tracks_its_parameters(self, pbb):
        """fit_params moves M0/NO every iteration - those calls must not be served a hit."""
        zz = pbb.z_samps
        M0, NO = pbb.hod.get_hod_params(zz, pbb.cut)
        base = pbb.hod.HOD(zz, M0, NO)
        moved = pbb.hod.HOD(zz, M0 * 1.5, NO)
        assert moved is not base
        assert not np.allclose(moved, base)

    def test_cache_stays_bounded(self, pbb):
        """Distinct redshift grids must not accumulate - each entry is ~0.6 MB."""
        for i in range(20):
            pbb.hmf.n_h(np.linspace(0.9, 1.0 + 0.01 * i, 5))
        assert len(pbb.hmf._n_h_memo) <= 4  # array_memo's default limit


class TestSigmaMoments:
    """sigma_R_n shares the k grid, P(k) and the window across moments - the sequence form
    must give exactly what the scalar form does, since _setup_hmf now makes one call."""

    def test_sequence_matches_scalar_exactly(self, cosmo_funcs_bias):
        from cosmo_wap.HOD.hmf import sigma_R_n

        R = cosmo_funcs_bias.R
        moments = sigma_R_n(cosmo_funcs_bias, R, (0, -1, -2))
        assert moments.shape == (3, R.size)
        for row, n in zip(moments, (0, -1, -2)):
            np.testing.assert_array_equal(row, sigma_R_n(cosmo_funcs_bias, R, n))

    def test_scalar_n_still_returns_one_row(self, cosmo_funcs_bias):
        from cosmo_wap.HOD.hmf import sigma_R_n

        R = cosmo_funcs_bias.R
        assert sigma_R_n(cosmo_funcs_bias, R, 0).shape == R.shape

    def test_setup_stored_the_three_moments(self, cosmo_funcs_bias):
        """sigmaR0 is what sig_R['0'] - and so the whole mass function - is built from."""
        cf = cosmo_funcs_bias
        for arr in (cf.sigmaR0, cf.sigmaR1, cf.sigmaR2):
            assert arr.shape == cf.R.shape
            assert np.all(arr > 0)
        # sigma^2 falls with radius, and higher n weights larger scales more
        assert np.all(np.diff(cf.sigmaR0) < 0)
        assert np.all(cf.sigmaR2 > cf.sigmaR0)


class TestMassIntegralWeights:
    """PBBias integrates over the halo mass grid by dot product against precomputed
    Simpson weights. The weights come from scipy, so this pins them to scipy's answer -
    if the two ever diverge, every galaxy bias is quietly wrong."""

    @pytest.fixture(scope="class")
    def pbb(self, cosmo, cosmo_funcs_bias):
        from cosmo_wap.HOD.peak_background_bias import PBBias

        return PBBias(cosmo_funcs_bias, cw.SurveyParams.Euclid(cosmo))

    def test_weights_reproduce_scipy_simpson(self, pbb):
        from scipy.integrate import simpson

        rng = np.random.default_rng(0)
        y = np.abs(rng.standard_normal((pbb.M.size, 12))) * 1e-20
        np.testing.assert_allclose(pbb._mass_weights @ y, simpson(y, pbb.M, axis=0), rtol=1e-12)

    def test_number_density_matches_a_direct_simpson(self, pbb):
        from scipy.integrate import simpson

        zz = pbb.z_samps
        integrand = pbb.hod.HOD(zz, *pbb.params) * pbb.hmf.n_h(zz)
        np.testing.assert_allclose(
            pbb.number_density(zz, *pbb.params), simpson(integrand, pbb.M, axis=0).squeeze(), rtol=1e-12
        )


class TestLocalPNGBias:
    """The HOD path and the analytic path (survey_params.SetSurveyFunctions.Loc) must give the
    same local-type PNG biases - they are the same universality relations, written differently:

        b_01 = 2 dc (b_1 - p)                        Karagiannis+ 2018 Eq. (10) at A=1, alpha=0
        b_11 = b_01 + 2 (dc bL_20 - bL_10)           their Eq. (11), = 2407.00168 Eq. (D.3b)

    At alpha=0 both halo-bias expressions are linear in b_1 and b_2, so the HOD mass-weighting
    commutes with them and the agreement is exact, not approximate. The 2A prefactor of Eq. (11)
    was once dropped, which halved every b_11 without changing b_01 - hence the check on both.
    """

    @pytest.fixture(scope="class")
    def pbb(self, cosmo, cosmo_funcs_bias):
        from cosmo_wap.HOD.peak_background_bias import PBBias

        return PBBias(cosmo_funcs_bias, cw.SurveyParams.Euclid(cosmo))

    def test_b01_matches_universality(self, cosmo_funcs_bias):
        survey = cosmo_funcs_bias.survey[0]
        zz = np.linspace(survey.z_range[0], survey.z_range[1], 9)
        expected = 2 * cosmo_funcs_bias.delta_c * (survey.b_1(zz) - survey.p)
        np.testing.assert_allclose(survey.loc.b_01(zz), expected, rtol=1e-10)

    def test_b11_matches_universality(self, cosmo_funcs_bias):
        """Built from the HOD's own b_1, b_2 - so this isolates the b_11 relation itself,
        not the (genuinely different) b_2 models of the two paths."""
        survey = cosmo_funcs_bias.survey[0]
        zz = np.linspace(survey.z_range[0], survey.z_range[1], 9)
        delta_c = cosmo_funcs_bias.delta_c

        bL10 = survey.b_1(zz) - 1
        bL20 = survey.b_2(zz) - (8 / 21) * bL10
        expected = 2 * delta_c * (survey.b_1(zz) - survey.p) + 2 * (delta_c * bL20 - bL10)
        np.testing.assert_allclose(survey.loc.b_11(zz), expected, rtol=1e-10)

    def test_halo_b11_prefactor(self, pbb):
        """Directly on the halo bias, before the HOD average: pins the 2A of Eq. (11)."""
        zz = np.linspace(pbb.z_samps[0], pbb.z_samps[-1], 4)
        b1, b2 = pbb.eulbias.b1(zz), pbb.eulbias.b2(zz)
        expected = 2 * (pbb.delta_c * (b2 + (13 / 21) * (b1 - 1)) - b1 + 1)  # alpha=0: sigma terms drop out
        np.testing.assert_allclose(pbb.eulbias.b_11(zz), expected, rtol=1e-10)

    def test_halo_b11_scales_with_A(self, pbb):
        """A is the shape amplitude (3 for equilateral, -3 for orthogonal) - it must factor out."""
        zz = np.linspace(pbb.z_samps[0], pbb.z_samps[-1], 4)
        np.testing.assert_allclose(pbb.eulbias.b_11(zz, A=3), 3 * pbb.eulbias.b_11(zz), rtol=1e-12)
