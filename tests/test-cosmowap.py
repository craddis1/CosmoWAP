import unittest
import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

# Add the parent directory to the path so we can import cosmo_wap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cosmo_wap as cw
    from cosmo_wap.utils import get_cosmology, get_theta
except ImportError:
    print("cosmo_wap not found. Make sure it's installed or the path is correct.")
    sys.exit(1)

try:
    from classy import Class
    HAS_CLASS = True
except ImportError:
    HAS_CLASS = False
    print("Warning: CLASS not installed, skipping tests that require it.")


class TestUtils(unittest.TestCase):
    """Test utility functions in cosmo_wap.utils."""

    def test_get_theta(self):
        """Test the get_theta function for triangle calculation."""
        # For an equilateral triangle k1=k2=k3
        k1 = np.array([1.0])
        k2 = np.array([1.0])
        k3 = np.array([1.0])
        theta = get_theta(k1, k2, k3)
        # Expected angle is 60 degrees or pi/3 radians
        self.assertAlmostEqual(theta[0], np.pi/3, places=6)

        # For a right-angled triangle with k1=k2 and k3=k1*sqrt(2)
        k3 = np.array([np.sqrt(2)])
        theta = get_theta(k1, k2, k3)
        # Expected angle is 90 degrees or pi/2 radians
        self.assertAlmostEqual(theta[0], np.pi/2, places=6)

        # For a line where k3 = k1 + k2
        k3 = np.array([2.0])
        theta = get_theta(k1, k2, k3)
        # Expected angle is 0 degrees or 0 radians
        self.assertAlmostEqual(theta[0], 0.0, places=6)

    @unittest.skipIf(not HAS_CLASS, "CLASS not installed")
    def test_get_cosmology(self):
        """Test the get_cosmology function."""
        cosmo = get_cosmology()
        # Check that cosmo is a Class instance
        self.assertIsInstance(cosmo, Class)
        
        # Check some derived parameters
        params = cosmo.get_current_derived_parameters(['Omega_m', 'sigma8'])
        self.assertGreater(params['Omega_m'], 0.2)  # Reasonable range for Omega_m
        self.assertLess(params['Omega_m'], 0.4)
        self.assertGreater(params['sigma8'], 0.7)  # Reasonable range for sigma8
        self.assertLess(params['sigma8'], 0.9)


class TestSurveyParams(unittest.TestCase):
    """Test the SurveyParams class."""

    def setUp(self):
        """Set up for tests."""
        if HAS_CLASS:
            self.cosmo = get_cosmology()
            self.survey_params = cw.survey_params.SurveyParams(self.cosmo)
        else:
            self.survey_params = cw.survey_params.SurveyParams()

    def test_survey_attributes(self):
        """Test that survey classes have the expected attributes."""
        surveys = [
            self.survey_params.Euclid,
            self.survey_params.SKAO1,
            self.survey_params.SKAO2,
            self.survey_params.DM_part,
            self.survey_params.CV_limit
        ]
        
        # Test BGS only if CLASS is available
        if HAS_CLASS:
            surveys.append(self.survey_params.BGS)
            
        for survey in surveys:
            # Check essential attributes
            self.assertTrue(hasattr(survey, 'b_1'))
            self.assertTrue(hasattr(survey, 'z_range'))
            self.assertTrue(hasattr(survey, 'be_survey'))
            self.assertTrue(hasattr(survey, 'Q_survey'))
            self.assertTrue(hasattr(survey, 'n_g'))
            self.assertTrue(hasattr(survey, 'f_sky'))
            
            # Check attribute types
            self.assertIsInstance(survey.z_range, list)
            self.assertEqual(len(survey.z_range), 2)
            self.assertIsInstance(survey.f_sky, float)
            
            # Check bias function at sample redshift
            z_sample = (survey.z_range[0] + survey.z_range[1]) / 2
            self.assertIsInstance(survey.b_1(z_sample), float)
            self.assertIsInstance(survey.n_g(z_sample), float)
    
    def test_init_new_survey(self):
        """Test creating a new survey."""
        new_survey = self.survey_params.InitNew()
        self.assertTrue(hasattr(new_survey, 'b_1'))
        self.assertTrue(hasattr(new_survey, 'z_range'))
        
        # Check we can call the functions
        z_test = 1.0
        self.assertIsInstance(new_survey.b_1(z_test), float)


@unittest.skipIf(not HAS_CLASS, "CLASS not installed")
class TestClassWAP(unittest.TestCase):
    """Test the main ClassWAP class."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmology()
        self.survey_params = cw.survey_params.SurveyParams(self.cosmo)
        self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)
    
    def test_initialization(self):
        """Test initialization and basic attributes."""
        # Check basic cosmological parameters
        self.assertIsInstance(self.class_wap.Om_0, float)
        self.assertIsInstance(self.class_wap.h, float)
        
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
        
    def test_power_spectrum(self):
        """Test power spectrum computations."""
        # Test at a specific k value
        k_test = np.array([0.1])  # h/Mpc
        pk_value = self.class_wap.Pk(k_test)
        
        # Power spectrum should be positive
        self.assertGreater(pk_value[0], 0)
        
        # Test linear vs non-linear
        if hasattr(self.class_wap, 'Pk_NL'):
            pk_nl_value = self.class_wap.Pk_NL(k_test)
            self.assertGreater(pk_nl_value[0], 0)
            
            # At small k, linear and non-linear should be close
            small_k = np.array([0.01])
            self.assertAlmostEqual(
                self.class_wap.Pk(small_k)[0] / self.class_wap.Pk_NL(small_k)[0],
                1.0,
                places=2
            )
    
    def test_get_params(self):
        """Test parameter getter functions."""
        k1 = np.array([0.1])
        k2 = np.array([0.1])
        zz = 1.0
        
        # Test get_params_pk
        params_pk = self.class_wap.get_params_pk(k1, zz)
        self.assertEqual(len(params_pk), 7)  # Check we get expected number of parameters
        
        # Test get_params with k3
        k3 = np.array([0.1])
        params = self.class_wap.get_params(k1, k2, k3=k3, zz=zz)
        self.assertEqual(len(params), 21)  # Check we get expected number of parameters
        
        # Test get_params with theta
        theta = np.array([np.pi/3])  # 60 degrees
        params = self.class_wap.get_params(k1, k2, theta=theta, zz=zz)
        self.assertEqual(len(params), 21)
    
    def test_get_beta_funcs(self):
        """Test beta function calculations."""
        zz = 1.0
        betas = self.class_wap.get_beta_funcs(zz)
        self.assertIsInstance(betas, list)
        
        # Should return a list of 22 beta values
        self.assertEqual(len(betas), 22)
        
        # All beta values should be finite
        for beta in betas:
            self.assertTrue(np.isfinite(beta))


@unittest.skipIf(not HAS_CLASS, "CLASS not installed")
class TestBiasModeling(unittest.TestCase):
    """Test bias modeling functionality."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmology()
        self.survey_params = cw.survey_params.SurveyParams(self.cosmo)
        self.euclid = self.survey_params.Euclid
    
    def test_pb_bias_initialization(self):
        """Test PBBias class initialization."""
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        # Check basic attributes
        self.assertTrue(hasattr(pb_bias, 'M'))
        self.assertTrue(hasattr(pb_bias, 'R'))
        self.assertTrue(hasattr(pb_bias, 'M0_func'))
        self.assertTrue(hasattr(pb_bias, 'NO_func'))
        
        # Check bias functions
        self.assertTrue(hasattr(pb_bias, 'b_1'))
        self.assertTrue(hasattr(pb_bias, 'b_2'))
        self.assertTrue(hasattr(pb_bias, 'g_2'))
        
        # Check PNG biases
        self.assertTrue(hasattr(pb_bias, 'loc'))
        self.assertTrue(hasattr(pb_bias, 'eq'))
        self.assertTrue(hasattr(pb_bias, 'orth'))
        
        # Test at a specific redshift
        z_test = 1.0
        self.assertIsInstance(pb_bias.b_1(z_test), float)
        self.assertIsInstance(pb_bias.b_2(z_test), float)
        self.assertIsInstance(pb_bias.g_2(z_test), float)
        self.assertIsInstance(pb_bias.loc.b_01(z_test), float)
        self.assertIsInstance(pb_bias.loc.b_11(z_test), float)
    
    def test_bias_consistency(self):
        """Test consistency between different bias parameters."""
        z_values = np.linspace(self.euclid.z_range[0], self.euclid.z_range[1], 5)
        
        # Create PBBias instance
        pb_bias = cw.PBBias(self.cosmo, self.euclid)
        
        for z in z_values:
            # Check tidal bias relation (approximate): g_2 ≈ -(2/7)*(b_1-1)
            g2_expected = -(2/7) * (pb_bias.b_1(z) - 1)
            self.assertAlmostEqual(pb_bias.g_2(z) / g2_expected, 1.0, places=1)
            
            # Check that b_2 is reasonable (typically larger than b_1 for galaxy samples)
            self.assertGreater(abs(pb_bias.b_2(z)), 0.1)
            
            # Check PNG bias parameters are nonzero
            self.assertNotEqual(pb_bias.loc.b_01(z), 0.0)
            self.assertNotEqual(pb_bias.loc.b_11(z), 0.0)


@unittest.skipIf(not HAS_CLASS, "CLASS not installed")
class TestIntegrationModule(unittest.TestCase):
    """Test the integration module."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmology()
        self.survey_params = cw.survey_params.SurveyParams(self.cosmo)
        self.class_wap = cw.ClassWAP(self.cosmo, self.survey_params.Euclid)
        
    def test_single_int(self):
        """Test single integral function."""
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        
        # Mock function for integration
        def mock_func(dx, cosmo_funcs, k, zz, t, sigma=None):
            return np.ones_like(k) * dx
        
        # Perform integration
        result = cw.integrate.single_int(mock_func, self.class_wap, k1, zz)
        
        # Check result shape matches input k1 shape
        self.assertEqual(result.shape, k1.shape)
        
        # Result should be positive
        self.assertTrue(np.all(result > 0))
    
    def test_double_int(self):
        """Test double integral function."""
        k1 = np.array([0.1, 0.2])
        zz = 1.0
        
        # Mock function for integration
        def mock_func(dx1, dx2, cosmo_funcs, k, zz, t, sigma=None):
            return np.ones_like(k) * dx1 * dx2
        
        # Perform integration
        result = cw.integrate.double_int(mock_func, self.class_wap, k1, zz)
        
        # Check result shape matches input k1 shape
        self.assertEqual(result.shape, k1.shape)
        
        # Result should be positive
        self.assertTrue(np.all(result > 0))


@unittest.skipIf(not HAS_CLASS, "CLASS not installed")
class TestForecastModule(unittest.TestCase):
    """Test the forecast module."""
    
    def setUp(self):
        """Set up for tests."""
        self.cosmo = get_cosmology()
        self.survey_params = cw.survey_params.SurveyParams(self.cosmo)
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
        self.assertTrue(len(forecast.k_bin) > 0)
        self.assertTrue(np.all(forecast.k_bin < 0.1))  # All k values should be less than k_max
        
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


if __name__ == '__main__':
    unittest.main()
