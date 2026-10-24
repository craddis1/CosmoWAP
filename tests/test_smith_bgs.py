"""Smith BGS HOD against hodpy, the code it is ported from (https://github.com/amjsmith/hodpy).

The reference values were generated once by calling hodpy itself: LuminosityFunctionSchechter (Phi_cumulative and
magnitude), spline.cumulative_spline_kernel, and HOD_BGS.__Mmin_z0/__M1_z0/M0 from the abacus branch (c0259e4), given
the parameters of AbacusSummit_base_c000_ph000_best_params.txt in hodpy's order; GAMA_KCorrection.k and Colour from
master (d303bef). hodpy is not a dependency (it isn't packaged and predates NumPy 2), so they are hard-coded here.
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.HOD.hods import absolute_magnitude, gama_colours, gama_k, gama_k_mean, threshold_K
from cosmo_wap.HOD.peak_background_bias import PBBias

M_REF = np.array([-22.5, -21.0, -20.0, -19.0, -18.0])
Z_REF = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
PHI_REF = [6.299202588577e-06, 1.709433552781e-03, 9.722791742732e-03, 2.961326078722e-02, 6.702834526353e-02]
M_Z0_REF = [-22.598148406757, -20.999991375135, -19.774217253913, -18.357454828886, -16.716247500025]
X_REF = np.array([-1.3, -0.4, -0.1, 0.0, 0.2, 0.55, 1.0])
CSK_REF = [0.0, 0.006954416828, 0.179899785547, 0.324590580762, 0.675409419238, 0.984948930885, 1.0]
MAG_REF = np.array([-21.0, -18.0, -17.0, -16.5, -15.0, -12.0])
LOG_MMIN_REF = [12.622368942827, 11.154024455594, 10.934076788462, 10.825930660283, 10.566743243742, 10.252450368252]
LOG_M1_REF = [13.697477891044, 12.453032888291, 12.223537451221, 12.115422501329, 11.856292380594, 11.542022348625]
M0_REF = [1.338667854233e12, 1.276396762375e07, 2.706595239750e05, 0.0, 0.0, 0.0]
# K at (Z_K, C_K) pairwise - clipped colours at both ends, and z past 0.5 where hodpy extrapolates linearly
Z_K = np.array([0.02, 0.1, 0.25, 0.4, 0.5, 0.7])
C_K = np.array([-0.3, 0.2, 0.55, 0.8, 1.0, 1.4])
K_REF = [-0.127224705637, -0.10348171, -0.0437330070038, 0.279364500179, 0.789560216611, 1.61399048745]
M_COL = np.array([-22.5, -21.0, -19.5, -18.7, -17.0, -15.0])
Z_COL = np.array([0.05, 0.1, 0.2, 0.3, 0.45, 0.6])
F_BLUE_REF = [0.285, 0.39, 0.625, 0.92, 1.0, 1.0]
BLUE_REF = (
    [0.895079667878, 0.730991002293, 0.551173546778, 0.460557902672, 0.356255446143, 0.296493437512],
    [0.07, 0.1, 0.15, 0.186, 0.22, 0.25],
)
RED_REF = ([1.012, 0.964, 0.898, 0.8544, 0.782, 0.718], [0.045, 0.06, 0.081, 0.097, 0.12975, 0.17])


@pytest.fixture(scope="module")
def pb(cosmo):
    sp = cw.SurveyParams.BGS(cosmo, cut=20.175, flag="HOD")
    cf = cw.ClassWAP(cosmo, sp, compute_bias=True, hod="Smith_BGS", verbose=False)
    return PBBias(cf, sp, "Tinker10", "Smith_BGS", cut=sp.cut)


class TestAgainstHodpy:
    def test_target_lf(self, pb):
        np.testing.assert_allclose(pb.hod.target_lf(M_REF, Z_REF), PHI_REF, rtol=1e-10)

    def test_threshold_at_z_fit(self, pb):
        """hodpy interpolates on a 0.01 dex grid in number density, so it is the looser of the two"""
        np.testing.assert_allclose(pb.hod.match_z_fit(M_REF, Z_REF), M_Z0_REF, atol=1e-4)

    def test_central_step(self, pb):
        np.testing.assert_allclose(pb.hod.number_centrals(X_REF, 0.1, 0.3), CSK_REF, atol=1e-10)

    def test_faint_end(self, pb):
        """the cubics swapped for a flattening curve, and M0 = 0, fainter than -17"""
        hod = pb.hod
        np.testing.assert_allclose(hod.M_function(MAG_REF, *hod.params[:4]), LOG_MMIN_REF, atol=1e-10)
        np.testing.assert_allclose(hod.M_function(MAG_REF, *hod.params[10:14]), LOG_M1_REF, atol=1e-10)
        np.testing.assert_allclose(10 ** hod.M0_function(MAG_REF, *hod.params[8:10]), M0_REF, rtol=1e-10)

    def test_k_correction(self):
        np.testing.assert_allclose(gama_k(Z_K, C_K), K_REF, atol=1e-10)

    def test_colours(self):
        f_blue, blue, red = gama_colours(M_COL, Z_COL)
        np.testing.assert_allclose(f_blue, F_BLUE_REF, atol=1e-10)
        np.testing.assert_allclose(blue, BLUE_REF, atol=1e-10)
        np.testing.assert_allclose(red, RED_REF, atol=1e-10)


class TestKCorrection:
    def test_callable_cut(self, cosmo):
        cut = lambda zz: 20 + zz
        sp = cw.SurveyParams.BGS(cosmo, cut=cut)
        np.testing.assert_allclose(sp.K(sp.zz), threshold_K(cosmo, cut(sp.zz), sp.zz))

        cf = cw.ClassWAP(cosmo, sp, verbose=False)
        assert np.all(np.isfinite(cf.survey[0].b_1(sp.zz)))
        assert np.all(cf.survey[0].n_g(sp.zz) > 0)

    def test_reference_band(self, pb):
        """every colour has K = -2.5 log10(1.1) at z = 0.1, so the threshold's does too"""
        np.testing.assert_allclose(pb.survey_params.K(0.1), -2.5 * np.log10(1.1), atol=1e-7)

    @pytest.mark.parametrize("zz", [0.3, 0.5, 0.6])
    def test_colour_average(self, zz):
        """the closed form against brute force on a fine colour grid"""
        mean, rms = np.array([0.1, 0.45, 0.8, 1.1]), np.array([0.05, 0.15, 0.1, 0.2])
        c = np.linspace(-1.5, 2.5, 400001)[:, None]
        w = np.exp(-0.5 * ((c - mean) / rms) ** 2)
        brute = np.trapezoid(w * gama_k(zz, c), c[:, 0], axis=0) / np.trapezoid(w, c[:, 0], axis=0)
        np.testing.assert_allclose(gama_k_mean(zz, mean, rms), brute, atol=1e-8)

    def test_threshold_consistent(self, pb, cosmo):
        """K is the colour average at the threshold it sets"""
        zz = pb.z_samps
        K = pb.survey_params.K(zz)
        f_blue, blue, red = gama_colours(absolute_magnitude(cosmo, pb.cut, zz, K), zz)
        np.testing.assert_allclose(f_blue * gama_k_mean(zz, *blue) + (1 - f_blue) * gama_k_mean(zz, *red), K, atol=1e-6)

    def test_one_K(self, pb):
        """the threshold, b_e, lum_derivs and set_selection all take the survey's K"""
        K = pb.survey_params.K
        assert pb.hod.K is K and pb.hod.lf.K is K and pb.cosmo_funcs.survey[0].K is K


class TestEvolution:
    @pytest.mark.parametrize(
        "zz, cut",
        [(0.5, 17.5), (np.array([0.1, 0.3, 0.5]), 17.5), (0.5, np.array([17.5, 18.5, 20.0]))],
    )
    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_bright_cut_with_zero_initial_occupation(self, pb, zz, cut):
        """A reachable bright cut starts above the mass grid at z = 0.5 - log n is -inf there."""
        np.testing.assert_allclose(pb.number_density(zz, cut), pb.hod.target_number_density(cut, zz), rtol=1e-8, atol=0)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_no_halos_cannot_match_the_target(self, pb, monkeypatch):
        empty = np.zeros_like(pb.hmf.n_h(0.3))
        monkeypatch.setattr(pb.hmf, "n_h", lambda zz: empty)
        with pytest.raises(ValueError, match="target number density out of the HOD's reach"):
            pb.hod.get_params(0.3, pb.cut)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_zero_target_is_rejected(self, pb):
        with pytest.raises(ValueError, match="target number density must be finite and positive"):
            pb.hod.slide_factor(0.5, -100.0, *pb.hod.get_params(0.3, pb.cut))

    def test_identity_at_z_fit(self, pb):
        zz = np.array([pb.hod.z_fit])
        np.testing.assert_allclose(pb.hod.get_M_c(20.175, zz), pb.hod.get_M_obs(20.175, zz), atol=1e-6)

    def test_n_g_is_target_lf(self, pb):
        """the slide factor makes the HOD reproduce the target luminosity function"""
        zz = pb.z_samps
        np.testing.assert_allclose(pb.n_g(zz), pb.hod.target_number_density(pb.cut, zz), rtol=1e-6)

    @pytest.mark.parametrize("shift", [-0.5, 0.3])
    def test_shifted_cut_needs_no_hod(self, pb, shift):
        """Q, lum_derivs and the fraction split read n_g at shifted cuts from the target luminosity function - the
        same as integrating the HOD there with its own slide factor"""
        zz = pb.z_samps
        cut = pb.cut + shift
        np.testing.assert_allclose(pb.number_density(zz, cut), pb.hod.target_number_density(cut, zz), rtol=1e-6)
