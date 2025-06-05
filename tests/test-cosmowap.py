import unittest
import numpy as np
from scipy.interpolate import CubicSpline
import warnings
import os
import sys

# Add the parent directory to the path so we can import cosmo_wap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    import cosmo_wap as cw
    from cosmo_wap.lib.utils import get_cosmo, get_theta, get_k3
    COSMOWAP_AVAILABLE = True
except ImportError as e:
    print(f"cosmo_wap not found: {e}")
    COSMOWAP_AVAILABLE = False

try:
    from classy import Class
    CLASS_AVAILABLE = True
except ImportError:
    CLASS_AVAILABLE = False
    print("Warning: CLASS not installed, skipping tests that require it.")

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestUtils(unittest.TestCase):
    """Test utility functions in cosmo_wap.lib.utils."""

    @unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
    def test_get_theta(self):
        """Test the get_theta function for triangle calculation."""
        # For an equilateral triangle k1=k2=k3
        k1, k2, k3 = 1.0, 1.0, 1.0
        theta = get_theta(k1, k2, k3)
        # Expected angle is 60 degrees or pi/3 radians
        self.assertAlmostEqual(theta, np.pi/3, places=6)

        # For a right-angled triangle with k1=k2 and k3=k1*sqrt(2)
        k3 = np.sqrt(2)
        theta = get_theta(k1, k2, k3)
        # Expected angle is 90 degrees or pi/2 radians
        self.assertAlmostEqual(theta, np.pi/2, places=6)

        # For arrays
        k1_arr = np.array([1.0, 1.0])
        k2_arr = np.array([1.0, 1.0])
        k3_arr = np.array([1.0, np.sqrt(2)])
        theta_arr = get_theta(k1_arr, k2_arr, k3_arr)
        expected = np.array([np.pi/3, np.pi/2])
        np.testing.assert_allclose(theta_arr, expected, rtol=1e-6)

    @unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
    def test_get_k3(self):
        """Test the get_k3 function."""
        k1, k2 = 1.0, 1.0
        theta = np.pi/3  # 60 degrees
        k3 = get_k3(theta, k1, k2)
        # For equilateral triangle, k3 should equal k1 and k2
        self.assertAlmostEqual(k3, 1.0, places=6)
        
        # Test array inputs
        theta_arr = np.array([np.pi/3, np.pi/2])
        k3_arr = get_k3(theta_arr, k1, k2)
        expected = np.array([1.0, np.sqrt(2)])
        np.testing.assert_allclose(k3_arr, expected, rtol=1e-6)

    @unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
    def test_get_cosmo(self):
        """Test the get_cosmo function."""
        cosmo = get_cosmo()
        # Check that cosmo is a Class instance
        self.assertIsInstance(cosmo, Class)
        
        # Check some derived parameters
        params = cosmo.get_current_derived_parameters(['Omega_m', 'sigma8', 'h'])
        self.assertGreater(params['Omega_m'], 0.2)  # Reasonable range for Omega_m
        self.assertLess(params['Omega_m'], 0.4)
        self.assertGreater(params['sigma8'], 0.7)  # Reasonable range for sigma8
        self.assertLess(params['sigma8'], 0.9)
        self.assertGreater(params['h'], 0.6)  # Reasonable range for h
        self.assertLess(params['h'], 0.8)

        # Test custom parameters
        custom_cosmo = get_cosmo(Omega_m=0.25, sigma8=0.8)
        custom_params = custom_cosmo.get_current_derived_parameters(['Omega_m', 'sigma8'])
        self.assertAlmostEqual(custom_params['Omega_m'], 0.25, places=3)
        self.assertAlmostEqual(custom_params['sigma8'], 0.8, places=3)


@unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
class TestSurveyParams(unittest.TestCase):
    """Test the SurveyParams class and survey definitions."""

    def setUp(self):
        """Set up for tests."""
        if CLASS_AVAILABLE:
            self.cosmo = get_cosmo()
            self.survey_params = cw.SurveyParams(self.cosmo)
        else:
            self.survey_params = None

    @unittest.skipIf(not CLASS_AVAILABLE, "CLASS not available")
    def test_survey_initialization(self):
        """Test that survey classes initialize correctly."""
        surveys = {
            'Euclid': self.survey_params.Euclid,
            'BGS': self.survey_params.BGS,
            'MegaMapper': self.survey_params.MegaMapper,
            'SKAO1': self.survey_params.SKAO1,
            'SKAO2': self.survey_params.SKAO2,
            'DM_part': self.survey_params.DM_part,
            'CV_limit': self.survey_params.CV_limit
        }
        
        for name, survey in surveys.items():
            with self.subTest(survey=name):
                # Check essential attributes
                self.assertTrue(hasattr(survey, 'b_1'), f"{name} missing b_1")
                self.assertTrue(hasattr(survey, 'z_range'), f"{name} missing z_range")
                self.assertTrue(hasattr(survey, 'be_survey'), f"{name} missing be_survey")
                self.assertTrue(hasattr(survey, 'Q_survey'), f"{name} missing Q_survey")
                self.assertTrue(hasattr(survey, 'n_g'), f"{name} missing n_g")
                self.assertTrue(hasattr(survey, 'f_sky'), f"{name} missing f_sky")
                
                # Check attribute types
                self.assertIsInstance(survey.z_range, list, f"{name} z_range not list")
                self.assertEqual(len(survey.z_range), 2, f"{name} z_range wrong length")
                self.assertIsInstance(survey.f_sky, (float, int), f"{name} f_sky wrong type")
                
                # Check bias function at sample redshift
                z_sample = (survey.z_range[0] + survey.z_range[1]) / 2
                try:
                    b1_val = survey.b_1(z_sample)
                    self.assertIsInstance(b1_val, (float, np.floating), f"{name} b_1 wrong type")
                    self.assertTrue(np.isfinite(b1_val), f"{name} b_1 not finite")
                    
                    ng_val = survey.n_g(z_sample)
                    self.assertIsInstance(ng_val, (float, np.floating), f"{name} n_g wrong type")
                    self.assertTrue(np.isfinite(ng_val), f"{name} n_g not finite")
                    self.assertGreater(ng_val, 0, f"{name} n_g not positive")
                    
                except Exception as e:
                    self.fail(f"Error evaluating {name} functions: {e}")

    @unittest.skipIf(not CLASS_AVAILABLE, "CLASS not available")
    def test_survey_update_methods(self):
        """Test survey update and modification methods."""
        euclid = self.survey_params.Euclid
        
        # Test update method
        updated_euclid = euclid.update(f_sky=0.5)
        self.assertEqual(updated_euclid.f_sky, 0.5)
        self.assertNotEqual(euclid.f_sky, 0.5)  # Original should be unchanged
        
        # Test modify_func method
        modified_euclid = euclid.modify_func('b_1', lambda x: x + 0.1)
        z_test = 1.0
        original_b1 = euclid.b_1(z_test)
        modified_b1 = modified_euclid.b_1(z_test)
        self.assertAlmostEqual(modified_b1, original_b1 + 0.1, places=5)

    def test_set_survey_functions(self):
        """Test SetSurveyFunctions class."""
        # Create a minimal survey param object
        class MinimalSurvey:
            z_range = [0.5, 2.0]
            f_sky = 0.3
            b_1 = lambda self, z: 1.0 + 0.5 * z
            n_g = lambda self, z: 1e-3
            
        minimal = MinimalSurvey()
        survey_funcs = cw.SetSurveyFunctions(minimal, compute_bias=False)
        
        # Check default values are set
        self.assertTrue(hasattr(survey_funcs, 'be_survey'))
        self.assertTrue(hasattr(survey_funcs, 'Q_survey'))
        self.assertTrue(hasattr(survey_funcs, 'b_2'))
        self.assertTrue(hasattr(survey_funcs, 'g_2'))
        
        # Test function calls
        z_test = 1.0
        self.assertEqual(survey_funcs.b_1(z_test), 1.5)
        self.assertEqual(survey_funcs.n_g(z_test), 1e-3)
        self.assertTrue(np.isfinite(survey_funcs.b_2(z_test)))
        self.assertTrue(np.isfinite(survey_funcs.g_2(z_test)))


@unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
class TestClassWAP(unittest.TestCase):
    """Test the main ClassWAP class."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmo()
        self.survey_params = cw.SurveyParams(self.cosmo)
        self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)
    
    def test_initialization(self):
        """Test ClassWAP initialization and basic attributes."""
        # Check basic cosmological parameters
        self.assertIsInstance(self.class_wap.Omega_m, float)
        self.assertIsInstance(self.class_wap.h, float)
        self.assertGreater(self.class_wap.Omega_m, 0)
        self.assertGreater(self.class_wap.h, 0)
        
        # Check redshift range
        self.assertEqual(self.class_wap.z_min, self.survey_params.Euclid.z_range[0])
        self.assertEqual(self.class_wap.z_max, self.survey_params.Euclid.z_range[1])
        
        # Check survey attributes
        self.assertTrue(hasattr(self.class_wap, 'survey'))
        self.assertTrue(hasattr(self.class_wap, 'survey1'))
        
        # Check interpolation functions
        self.assertIsInstance(self.class_wap.H_c, CubicSpline)
        self.assertIsInstance(self.class_wap.f_intp, CubicSpline)
        self.assertIsInstance(self.class_wap.D_intp, CubicSpline)
        self.assertIsInstance(self.class_wap.comoving_dist, CubicSpline)
        
    def test_power_spectrum(self):
        """Test power spectrum computations."""
        # Test at specific k values
        k_test = np.array([0.01, 0.1, 1.0])  # h/Mpc
        pk_values = self.class_wap.Pk(k_test)
        
        # Power spectrum should be positive
        self.assertTrue(np.all(pk_values > 0))
        
        # Power spectrum should decrease with k (roughly)
        self.assertGreater(pk_values[0], pk_values[-1])
        
        # Test derivatives
        pkd_values = self.class_wap.Pk_d(k_test)
        pkdd_values = self.class_wap.Pk_dd(k_test)
        
        # Derivatives should be finite
        self.assertTrue(np.all(np.isfinite(pkd_values)))
        self.assertTrue(np.all(np.isfinite(pkdd_values)))
        
        # Test non-linear power spectrum if available
        if hasattr(self.class_wap, 'Pk_NL'):
            pk_nl_values = self.class_wap.Pk_NL(k_test)
            self.assertTrue(np.all(pk_nl_values > 0))
            
            # At small k, linear and non-linear should be close
            small_k = np.array([0.01])
            ratio = self.class_wap.Pk(small_k)[0] / self.class_wap.Pk_NL(small_k)[0]
            self.assertAlmostEqual(ratio, 1.0, places=1)

    def test_cosmological_functions(self):
        """Test cosmological function evaluations."""
        z_test = np.array([0.5, 1.0, 2.0])
        
        # Test Hubble parameter
        H_vals = self.class_wap.H_c(z_test)
        self.assertTrue(np.all(H_vals > 0))
        self.assertTrue(np.all(np.isfinite(H_vals)))
        
        # Test growth factor
        D_vals = self.class_wap.D_intp(z_test)
        self.assertTrue(np.all(D_vals > 0))
        self.assertTrue(np.all(np.isfinite(D_vals)))
        # Growth factor should decrease with redshift
        self.assertTrue(np.all(np.diff(D_vals) < 0))
        
        # Test growth rate
        f_vals = self.class_wap.f_intp(z_test)
        self.assertTrue(np.all(f_vals > 0))
        self.assertTrue(np.all(f_vals < 2))  # Reasonable upper bound
        self.assertTrue(np.all(np.isfinite(f_vals)))
        
        # Test comoving distance
        chi_vals = self.class_wap.comoving_dist(z_test)
        self.assertTrue(np.all(chi_vals > 0))
        self.assertTrue(np.all(np.isfinite(chi_vals)))
        # Comoving distance should increase with redshift
        self.assertTrue(np.all(np.diff(chi_vals) > 0))

    def test_get_params_pk(self):
        """Test power spectrum parameter getter."""
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        
        params_pk = self.class_wap.get_params_pk(k1, zz)
        self.assertEqual(len(params_pk), 7)  # Check we get expected number of parameters
        
        k1_out, Pk1, Pkd1, Pkdd1, d, f, D1 = params_pk
        
        # Check shapes match
        np.testing.assert_array_equal(k1_out, k1)
        self.assertEqual(Pk1.shape, k1.shape)
        self.assertEqual(Pkd1.shape, k1.shape)
        self.assertEqual(Pkdd1.shape, k1.shape)
        
        # Check values are reasonable
        self.assertTrue(np.all(Pk1 > 0))
        self.assertTrue(np.all(np.isfinite(Pkd1)))
        self.assertTrue(np.all(np.isfinite(Pkdd1)))
        self.assertGreater(d, 0)
        self.assertGreater(f, 0)
        self.assertGreater(D1, 0)

    def test_get_params_bispectrum(self):
        """Test bispectrum parameter getter."""
        k1 = np.array([0.1])
        k2 = np.array([0.15])
        k3 = np.array([0.2])
        zz = 1.0
        
        # Test with k3
        params = self.class_wap.get_params(k1, k2, k3=k3, zz=zz)
        self.assertEqual(len(params), 21)  # Check we get expected number of parameters
        
        # Test with theta
        theta = get_theta(k1, k2, k3)
        params_theta = self.class_wap.get_params(k1, k2, theta=theta, zz=zz)
        self.assertEqual(len(params_theta), 21)
        
        # Results should be consistent
        for i, (p1, p2) in enumerate(zip(params, params_theta)):
            if isinstance(p1, np.ndarray):
                np.testing.assert_allclose(p1, p2, rtol=1e-10, 
                                         err_msg=f"Parameter {i} mismatch")
            else:
                self.assertAlmostEqual(p1, p2, places=10, 
                                     msg=f"Parameter {i} mismatch")

    def test_get_beta_funcs(self):
        """Test beta function calculations."""
        zz = 1.0
        betas = self.class_wap.get_beta_funcs(zz)
        self.assertIsInstance(betas, list)
        
        # Should return a list of 22 beta values
        self.assertEqual(len(betas), 22)
        
        # All beta values should be finite
        for i, beta in enumerate(betas):
            self.assertTrue(np.isfinite(beta), f"Beta {i} is not finite: {beta}")

    def test_png_parameters(self):
        """Test PNG parameter calculations."""
        k1, k2, k3 = 0.1, 0.15, 0.2
        zz = 1.0
        
        # Test local PNG
        bE01_loc, bE11_loc, Mk1, Mk2, Mk3 = self.class_wap.get_PNGparams(
            zz, k1, k2, k3, shape='Loc')
        
        self.assertTrue(np.isfinite(bE01_loc))
        self.assertTrue(np.isfinite(bE11_loc))
        self.assertTrue(np.all(np.isfinite([Mk1, Mk2, Mk3])))
        self.assertTrue(np.all(np.array([Mk1, Mk2, Mk3]) > 0))
        
        # Test power spectrum PNG
        bE01_pk, Mk1_pk = self.class_wap.get_PNGparams_pk(zz, k1, shape='Loc')
        self.assertTrue(np.isfinite(bE01_pk))
        self.assertTrue(np.isfinite(Mk1_pk))
        self.assertGreater(Mk1_pk, 0)

    def test_multi_tracer_setup(self):
        """Test multi-tracer functionality."""
        # Test with two different surveys
        survey_list = [self.survey_params.Euclid, self.survey_params.SKAO1]
        mt_wap = cw.ClassWAP(self.cosmo, survey_list)
        
        # Check that both surveys are set up
        self.assertTrue(hasattr(mt_wap, 'survey'))
        self.assertTrue(hasattr(mt_wap, 'survey1'))
        self.assertNotEqual(mt_wap.survey, mt_wap.survey1)
        
        # Check redshift range is intersection
        expected_zmin = max(survey.z_range[0] for survey in survey_list)
        expected_zmax = min(survey.z_range[1] for survey in survey_list)
        self.assertEqual(mt_wap.z_min, expected_zmin)
        self.assertEqual(mt_wap.z_max, expected_zmax)

    def test_survey_update(self):
        """Test survey parameter updates."""
        # Create modified survey
        modified_survey = self.survey_params.Euclid.update(f_sky=0.5)
        
        # Update ClassWAP with new survey
        updated_wap = self.class_wap.update_survey(modified_survey, verbose=False)
        
        # Check that survey was updated
        self.assertEqual(updated_wap.survey.f_sky, 0.5)
        self.assertNotEqual(self.class_wap.survey.f_sky, 0.5)  # Original unchanged


@unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
class TestPeakBackgroundBias(unittest.TestCase):
    """Test the Peak-Background Split bias modeling."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmo()
        self.survey_params = cw.SurveyParams(self.cosmo)
        self.euclid = self.survey_params.Euclid
    
    def test_pb_bias_initialization(self):
        """Test PBBias class initialization."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        # Check basic attributes
        self.assertTrue(hasattr(pb_bias, 'M'))
        self.assertTrue(hasattr(pb_bias, 'R'))
        self.assertTrue(hasattr(pb_bias, 'M0_func'))
        self.assertTrue(hasattr(pb_bias, 'NO_func'))
        
        # Check mass and radius arrays
        self.assertGreater(len(pb_bias.M), 10)
        self.assertGreater(len(pb_bias.R), 10)
        self.assertEqual(len(pb_bias.M), len(pb_bias.R))
        self.assertTrue(np.all(pb_bias.M > 0))
        self.assertTrue(np.all(pb_bias.R > 0))
        
        # Check bias functions exist
        self.assertTrue(hasattr(pb_bias, 'b_1'))
        self.assertTrue(hasattr(pb_bias, 'b_2'))
        self.assertTrue(hasattr(pb_bias, 'g_2'))
        
        # Check PNG biases
        self.assertTrue(hasattr(pb_bias, 'loc'))
        self.assertTrue(hasattr(pb_bias, 'eq'))
        self.assertTrue(hasattr(pb_bias, 'orth'))

    def test_bias_function_evaluation(self):
        """Test bias function evaluations."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        z_test = 1.0
        b1_val = pb_bias.b_1(z_test)
        b2_val = pb_bias.b_2(z_test)
        g2_val = pb_bias.g_2(z_test)
        
        # Check types and finite values
        self.assertIsInstance(b1_val, (float, np.floating))
        self.assertIsInstance(b2_val, (float, np.floating))
        self.assertIsInstance(g2_val, (float, np.floating))
        
        self.assertTrue(np.isfinite(b1_val))
        self.assertTrue(np.isfinite(b2_val))
        self.assertTrue(np.isfinite(g2_val))
        
        # Check reasonable ranges
        self.assertGreater(b1_val, 0.5)  # Linear bias should be > 0.5 for galaxies
        self.assertLess(b1_val, 5.0)     # But not too large
        self.assertGreater(abs(b2_val), 0.1)  # Second-order bias should be non-negligible

    def test_png_bias_evaluation(self):
        """Test PNG bias evaluations."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        z_test = 1.0
        
        # Test local PNG
        b01_loc = pb_bias.loc.b_01(z_test)
        b11_loc = pb_bias.loc.b_11(z_test)
        
        self.assertTrue(np.isfinite(b01_loc))
        self.assertTrue(np.isfinite(b11_loc))
        self.assertNotEqual(b01_loc, 0.0)
        self.assertNotEqual(b11_loc, 0.0)
        
        # Test equilateral PNG
        b01_eq = pb_bias.eq.b_01(z_test)
        b11_eq = pb_bias.eq.b_11(z_test)
        
        self.assertTrue(np.isfinite(b01_eq))
        self.assertTrue(np.isfinite(b11_eq))
        
        # Test orthogonal PNG
        b01_orth = pb_bias.orth.b_01(z_test)
        b11_orth = pb_bias.orth.b_11(z_test)
        
        self.assertTrue(np.isfinite(b01_orth))
        self.assertTrue(np.isfinite(b11_orth))

    def test_hod_parameters(self):
        """Test HOD parameter fitting."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        z_test = 1.0
        M0_val = pb_bias.M0_func(z_test)
        NO_val = pb_bias.NO_func(z_test)
        
        # Check reasonable ranges
        self.assertGreater(M0_val, 1e10)  # M0 should be reasonable galaxy mass
        self.assertLess(M0_val, 1e15)     # But not too large
        self.assertGreater(NO_val, 0.1)   # NO should be positive
        self.assertLess(NO_val, 10.0)     # But not too large

    def test_number_density_consistency(self):
        """Test that computed number density matches input."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        z_test = 1.0
        input_ng = self.euclid.n_g(z_test)
        computed_ng = pb_bias.n_g(z_test)
        
        # Should match within reasonable tolerance
        self.assertAlmostEqual(computed_ng / input_ng, 1.0, places=2)


@unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
class TestForecastModule(unittest.TestCase):
    """Test the forecast module functionality."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmo()
        self.survey_params = cw.SurveyParams(self.cosmo)
        self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)
        
    def test_forecast_base_class(self):
        """Test the base Forecast class."""
        z_bin = [1.0, 1.1]
        forecast = cw.forecast.Forecast(z_bin, self.class_wap, k_max=0.1)
        
        # Check basic attributes
        self.assertTrue(hasattr(forecast, 'k_f'))
        self.assertTrue(hasattr(forecast, 'k_bin'))
        self.assertTrue(hasattr(forecast, 'k_cut_bool'))
        self.assertTrue(hasattr(forecast, 'z_mid'))
        
        # Check k_bin is reasonable
        self.assertGreater(len(forecast.k_bin), 0)
        self.assertTrue(np.all(forecast.k_bin <= 0.1))  # All k values should be <= k_max
        self.assertTrue(np.all(forecast.k_bin > 0))     # All k values should be positive
        
        # Check z_mid
        expected_z_mid = (z_bin[0] + z_bin[1]) / 2 + 1e-6
        self.assertAlmostEqual(forecast.z_mid, expected_z_mid, places=5)

    def test_pk_forecast(self):
        """Test power spectrum forecast class."""
        z_bin = [1.0, 1.1]
        pk_forecast = cw.forecast.PkForecast(z_bin, self.class_wap, k_max=0.1)
        
        # Check PkForecast-specific attributes
        self.assertTrue(hasattr(pk_forecast, 'N_k'))
        self.assertTrue(hasattr(pk_forecast, 'args'))
        
        # Check N_k is positive
        self.assertTrue(np.all(pk_forecast.N_k > 0))
        
        # Check args structure
        self.assertEqual(len(pk_forecast.args), 3)

    def test_bk_forecast(self):
        """Test bispectrum forecast class."""
        z_bin = [1.0, 1.1]
        bk_forecast = cw.forecast.BkForecast(z_bin, self.class_wap, k_max=0.1)
        
        # Check BkForecast-specific attributes
        self.assertTrue(hasattr(bk_forecast, 'V123'))
        self.assertTrue(hasattr(bk_forecast, 'is_triangle'))
        self.assertTrue(hasattr(bk_forecast, 'beta'))
        self.assertTrue(hasattr(bk_forecast, 's123'))
        self.assertTrue(hasattr(bk_forecast, 'args'))
        
        # Check triangle constraints
        self.assertTrue(np.any(bk_forecast.is_triangle))  # Some triangles should be valid
        
        # Check V123 is positive for valid triangles
        self.assertTrue(np.all(bk_forecast.V123 > 0))
        
        # Check args structure
        self.assertEqual(len(bk_forecast.args), 6)

    def test_bin_volume(self):
        """Test volume calculation."""
        z_bin = [1.0, 1.1]
        forecast = cw.forecast.Forecast(z_bin, self.class_wap, k_max=0.1)
        
        # Calculate volume
        vol = forecast.bin_volume(forecast.z_mid, 0.05, f_sky=1.0)
        
        # Volume should be positive
        self.assertGreater(vol, 0)
        
        # Volume should scale with f_sky
        vol_half = forecast.bin_volume(forecast.z_mid, 0.05, f_sky=0.5)
        self.assertAlmostEqual(vol_half, vol * 0.5, places=5)
        
        # Volume should increase with redshift (roughly)
        vol_low_z = forecast.bin_volume(0.5, 0.05, f_sky=1.0)
        vol_high_z = forecast.bin_volume(2.0, 0.05, f_sky=1.0)
        self.assertGreater(vol_high_z, vol_low_z)

    def test_full_forecast(self):
        """Test the FullForecast class."""
        # Create a simple k_max function
        kmax_func = lambda z: 0.1 + 0.01 * z
        
        full_forecast = cw.forecast.FullForecast(
            self.class_wap, kmax_func=kmax_func, s_k=1, nonlin=False)
        
        # Check initialization
        self.assertTrue(hasattr(full_forecast, 'z_bins'))
        self.assertTrue(hasattr(full_forecast, 'k_max_list'))
        self.assertTrue(hasattr(full_forecast, 'cosmo_funcs'))
        
        # Check z_bins structure
        self.assertGreater(len(full_forecast.z_bins), 0)
        for z_bin in full_forecast.z_bins:
            self.assertEqual(len(z_bin), 2)
            self.assertLess(z_bin[0], z_bin[1])
        
        # Check k_max_list
        self.assertEqual(len(full_forecast.k_max_list), len(full_forecast.z_bins))
        self.assertTrue(np.all(np.array(full_forecast.k_max_list) > 0))


@unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
class TestIntegrationModule(unittest.TestCase):
    """Test the integration module."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmo()
        self.survey_params = cw.SurveyParams(self.cosmo)
        self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)
        
    def test_legendre_integration(self):
        """Test Legendre-Gauss integration for multipoles."""
        from cosmo_wap.lib.integrate import legendre
        
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        l = 0  # monopole
        
        # Simple test function that should integrate to a known value
        def test_func(mu, cosmo_funcs, k, z, t):
            return np.ones_like(k)  # Constant function
        
        # For l=0 (monopole), integrating constant function should give 2
        # because monopole coefficient is (2*0+1)/2 = 1/2 and integral over [-1,1] is 2
        result = legendre(test_func, l, self.class_wap, k1, zz)
        
        # Check result shape and approximate value
        self.assertEqual(result.shape, k1.shape)
        np.testing.assert_allclose(result, 1.0, rtol=0.1)  # Should be close to 1

    def test_single_integration(self):
        """Test single integral function."""
        from cosmo_wap.lib.integrate import single_int
        
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        
        # Mock function for integration
        def mock_func(dx, cosmo_funcs, k, zz, t, sigma):
            # Return a simple function of distance that integrates to finite value
            return np.ones_like(k) * np.exp(-dx / 1000.0)  # Exponential decay
        
        # Perform integration
        result = single_int(mock_func, self.class_wap, k1, zz)
        
        # Check result shape matches input k1 shape
        self.assertEqual(result.shape, k1.shape)
        
        # Result should be positive and finite
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_double_integration(self):
        """Test double integral function."""
        from cosmo_wap.lib.integrate import double_int
        
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        
        # Mock function for integration
        def mock_func(dx1, dx2, cosmo_funcs, k, zz, t, sigma):
            # Return a simple function that depends on both integration variables
            return np.ones_like(k) * np.exp(-(dx1 + dx2) / 1000.0)
        
        # Perform integration
        result = double_int(mock_func, self.class_wap, k1, zz)
        
        # Check result shape matches input k1 shape
        self.assertEqual(result.shape, k1.shape)
        
        # Result should be positive and finite
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_ylm_integration(self):
        """Test spherical harmonic integration."""
        from cosmo_wap.lib.integrate import ylm
        
        k1 = np.array([0.1])
        k2 = np.array([0.15])
        k3 = np.array([0.2])
        theta = get_theta(k1, k2, k3)
        zz = 1.0
        l, m = 0, 0  # monopole
        
        # Simple test function
        def test_func(mu, phi, cosmo_funcs, k1, k2, k3, theta, zz, r, s):
            return np.ones_like(k1)  # Constant function
        
        result = ylm(test_func, l, m, self.class_wap, k1, k2, k3, theta, zz)
        
        # Check result is finite and has correct shape
        self.assertTrue(np.isfinite(result))
        self.assertIsInstance(result, (complex, np.complexfloating))


@unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")  
class TestLuminosityFunctions(unittest.TestCase):
    """Test luminosity function implementations."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmo()
    
    def test_ha_luminosity_functions(self):
        """Test H-alpha luminosity functions."""
        from cosmo_wap.lib.luminosity_funcs import Model1LuminosityFunction, Model3LuminosityFunction
        
        # Test Model1
        lf1 = Model1LuminosityFunction(self.cosmo)
        z_test = np.array([1.0, 1.5])
        L_test = np.array([1e42, 1e43])  # erg/s
        
        phi_vals = lf1.luminosity_function(L_test, z_test)
        self.assertTrue(np.all(phi_vals > 0))
        self.assertTrue(np.all(np.isfinite(phi_vals)))
        
        # Test number density calculation
        F_c = 2e-16  # erg cm^-2 s^-1
        ng_vals = lf1.number_density(F_c, z_test)
        self.assertTrue(np.all(ng_vals > 0))
        self.assertTrue(np.all(np.isfinite(ng_vals)))
        
        # Test Model3
        lf3 = Model3LuminosityFunction(self.cosmo)
        phi_vals_3 = lf3.luminosity_function(L_test, z_test)
        self.assertTrue(np.all(phi_vals_3 > 0))
        self.assertTrue(np.all(np.isfinite(phi_vals_3)))

    def test_kcorrection_luminosity_functions(self):
        """Test K-correction luminosity functions."""
        from cosmo_wap.lib.luminosity_funcs import BGSLuminosityFunction, LBGLuminosityFunction
        
        # Test BGS
        bgs_lf = BGSLuminosityFunction(self.cosmo)
        z_test = np.array([0.2, 0.4])
        m_test = np.array([19.0, 20.0])  # apparent magnitude
        
        phi_vals = bgs_lf.luminosity_function(m_test, z_test)
        self.assertTrue(np.all(phi_vals > 0))
        self.assertTrue(np.all(np.isfinite(phi_vals)))
        
        # Test number density
        m_cut = 20.0
        ng_vals = bgs_lf.number_density(m_cut, z_test)
        self.assertTrue(np.all(ng_vals > 0))
        self.assertTrue(np.all(np.isfinite(ng_vals)))
        
        # Test LBG
        lbg_lf = LBGLuminosityFunction(self.cosmo)
        z_lbg = lbg_lf.z_values[:3]  # Use first few redshift values
        m_lbg = np.array([24.0, 25.0])
        
        phi_lbg = lbg_lf.luminosity_function(m_lbg, z_lbg)
        self.assertTrue(np.all(phi_lbg > 0))
        self.assertTrue(np.all(np.isfinite(phi_lbg)))


@unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
class TestBetaFunctions(unittest.TestCase):
    """Test beta function calculations."""
    
    def setUp(self):
        """Set up for tests."""
        if CLASS_AVAILABLE:
            self.cosmo = get_cosmo()
            self.survey_params = cw.SurveyParams(self.cosmo)
            self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)

    @unittest.skipIf(not CLASS_AVAILABLE, "CLASS not available")
    def test_beta_interpolation(self):
        """Test beta function interpolation."""
        from cosmo_wap.lib.betas import interpolate_beta_funcs
        
        # Test beta function calculation
        betas = interpolate_beta_funcs(self.class_wap)
        
        # Should return array of function objects
        self.assertIsInstance(betas, np.ndarray)
        self.assertEqual(len(betas), 22)  # Expected number of beta functions
        
        # Test evaluation at specific redshift
        z_test = 1.0
        beta_vals = [beta(z_test) for beta in betas]
        
        # All should be finite
        for i, val in enumerate(beta_vals):
            self.assertTrue(np.isfinite(val), f"Beta {i} not finite: {val}")

    @unittest.skipIf(not CLASS_AVAILABLE, "CLASS not available")
    def test_beta_consistency(self):
        """Test consistency of beta functions."""
        # Test that beta functions are consistent between calls
        betas1 = self.class_wap.get_beta_funcs(1.0)
        betas2 = self.class_wap.get_beta_funcs(1.0)
        
        np.testing.assert_allclose(betas1, betas2, rtol=1e-12)
        
        # Test at different redshifts
        z_array = np.linspace(self.class_wap.z_min, self.class_wap.z_max, 5)
        for z in z_array:
            betas_z = self.class_wap.get_beta_funcs(z)
            self.assertEqual(len(betas_z), 22)
            self.assertTrue(np.all(np.isfinite(betas_z)))


class TestModuleImports(unittest.TestCase):
    """Test that modules import correctly."""
    
    @unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
    def test_main_imports(self):
        """Test main module imports."""
        # Test main components are importable
        self.assertTrue(hasattr(cw, 'ClassWAP'))
        self.assertTrue(hasattr(cw, 'PBBias'))
        self.assertTrue(hasattr(cw, 'SurveyParams'))
        self.assertTrue(hasattr(cw, 'forecast'))
        self.assertTrue(hasattr(cw, 'lib'))
        self.assertTrue(hasattr(cw, 'integrated'))

    @unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
    def test_submodule_imports(self):
        """Test submodule imports."""
        # Test lib submodules
        self.assertTrue(hasattr(cw.lib, 'utils'))
        self.assertTrue(hasattr(cw.lib, 'integrate'))
        self.assertTrue(hasattr(cw.lib, 'betas'))
        self.assertTrue(hasattr(cw.lib, 'luminosity_funcs'))
        
        # Test that pk and bk modules are dynamically imported
        # (these are imported via __init__.py dynamic imports)
        self.assertTrue(hasattr(cw, 'pk'))
        self.assertTrue(hasattr(cw, 'bk'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    @unittest.skipIf(not CLASS_AVAILABLE or not COSMOWAP_AVAILABLE, "CLASS or CosmoWAP not available")
    def test_invalid_redshift_ranges(self):
        """Test handling of invalid redshift ranges."""
        cosmo = get_cosmo()
        survey_params = cw.SurveyParams(cosmo)
        
        # Create surveys with non-overlapping redshift ranges
        survey1 = survey_params.Euclid.update(z_range=[0.5, 1.0])
        survey2 = survey_params.SKAO1.update(z_range=[2.0, 3.0])
        
        # This should raise an error due to incompatible redshift ranges
        with self.assertRaises(ValueError):
            cw.ClassWAP(cosmo, [survey1, survey2])

    @unittest.skipIf(not COSMOWAP_AVAILABLE, "CosmoWAP not available")
    def test_triangle_conditions(self):
        """Test triangle condition handling in get_theta."""
        # Test valid triangle
        k1, k2, k3 = 1.0, 1.0, 1.0
        theta = get_theta(k1, k2, k3)
        self.assertTrue(np.isfinite(theta))
        
        # Test degenerate triangle (should handle gracefully)
        k1, k2, k3 = 1.0, 1.0, 2.0  # Forms a line
        theta = get_theta(k1, k2, k3)
        self.assertTrue(np.isfinite(theta))
        self.assertAlmostEqual(theta, 0.0, places=6)
        
        # Test invalid triangle (should be handled by np.isclose in get_theta)
        k1, k2, k3 = 1.0, 1.0, 3.0  # Violates triangle inequality
        theta = get_theta(k1, k2, k3)
        self.assertTrue(np.isfinite(theta))  # Should still return finite value


def create_test_suite():
    """Create a test suite with all test cases."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUtils,
        TestSurveyParams,
        TestClassWAP,
        TestPeakBackgroundBias,
        TestForecastModule,
        TestIntegrationModule,
        TestLuminosityFunctions,
        TestBetaFunctions,
        TestModuleImports,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Run tests with different verbosity levels based on environment
    verbosity = int(os.environ.get('TEST_VERBOSITY', '2'))
    
    if len(sys.argv) > 1 and sys.argv[1] == 'discover':
        # Run specific test discovery
        unittest.main(verbosity=verbosity)
    else:
        # Run the full test suite
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
        result = runner.run(suite)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print(f"\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")
        
        print(f"{'='*50}")
        
        # Exit with error code if tests failed
        sys.exit(not result.wasSuccessful())
