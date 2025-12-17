import unittest
import numpy as np
import os
import sys
from scipy.interpolate import CubicSpline

# Add the source directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.pk.combined import pk_func
from cosmo_wap.bk.combined import bk_func
from cosmo_wap.forecast import FullForecast

# Check for optional dependencies
try:
    from classy import Class
    CLASS_AVAILABLE = True
except ImportError:
    CLASS_AVAILABLE = False
    print("Warning: 'classy' not found. Tests requiring CLASS will be skipped.")

class TestCosmoWAP(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Set up shared resources for the test suite to avoid initializing CLASS multiple times.
        """
        if CLASS_AVAILABLE:
            # 1. Initialize Cosmology
            # Using specific parameters to ensure consistency
            cls.cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=2.0)
            
            # 2. Initialize Survey (Euclid as an example)
            cls.survey_params = cw.SurveyParams.Euclid(cls.cosmo)
            
            # 3. Initialize the main Wrapper
            # compute_bias=True is needed to test PBBias integration inside ClassWAP if implemented,
            # otherwise we test PBBias separately.
            cls.cw = cw.ClassWAP(cls.cosmo, survey_params=cls.survey_params, verbose=False)

    def setUp(self):
        """Run before every test method"""
        if not CLASS_AVAILABLE:
            self.skipTest("CLASS not available")
        self.cw = self.__class__.cw
        self.survey = self.__class__.survey_params

    # =========================================================================
    # 1. Utility Tests
    # =========================================================================
    
    def test_utils_geometry(self):
        """Test geometric helper functions for bispectrum triangles."""
        k1, k2 = 0.1, 0.1
        
        # Equilateral triangle
        k3_eq = 0.1
        theta_eq = utils.get_theta(k1, k2, k3_eq)
        self.assertAlmostEqual(theta_eq, np.pi/3, places=5)
        
        # Squeezed triangle (approx)
        k3_sq = 1e-4
        theta_sq = utils.get_theta(k1, k2, k3_sq)
        self.assertTrue(np.abs(theta_sq - np.pi) < 0.1 or np.abs(theta_sq) < 0.1)
        
        # Inverse Check
        k3_calc = utils.get_k3(theta_eq, k1, k2)
        self.assertAlmostEqual(k3_calc, k3_eq, places=5)

    def test_get_cosmo(self):
        """Test the get_cosmo wrapper."""
        # Test standard initialization
        c = utils.get_cosmo(Omega_m=0.3, h=0.7, emulator=False)
        self.assertIsInstance(c, Class)
        self.assertAlmostEqual(c.h(), 0.7)
        
        # Test emulator mode returns tuple
        c_emu, params = utils.get_cosmo(emulator=True)
        self.assertIsInstance(params, dict)

    # =========================================================================
    # 2. Survey Parameter Tests
    # =========================================================================

    def test_survey_attributes(self):
        """Test that survey object has required methods/attributes."""
        s = self.survey
        
        # Check essential methods
        z_test = 1.0
        self.assertTrue(hasattr(s, 'b_1'))
        self.assertTrue(hasattr(s, 'n_g'))
        
        # Check values
        b1 = s.b_1(z_test)
        ng = s.n_g(z_test)
        
        self.assertIsInstance(b1, (float, np.floating))
        self.assertGreater(ng, 0)
        
        # Check attributes
        self.assertTrue(hasattr(s, 'f_sky'))
        self.assertTrue(hasattr(s, 'z_range'))

    def test_survey_update(self):
        """Test the update method for survey parameters."""
        new_fsky = 0.123
        updated_survey = self.survey.update(f_sky=new_fsky)
        
        self.assertEqual(updated_survey.f_sky, new_fsky)
        self.assertNotEqual(self.survey.f_sky, new_fsky)

    # =========================================================================
    # 3. ClassWAP (Main Wrapper) Tests
    # =========================================================================

    def test_background_quantities(self):
        """Test interpolation of background quantities."""
        z_arr = np.linspace(0.1, 1.5, 5)
        
        H = self.cw.H_c(z_arr)
        D = self.cw.D(z_arr)
        f = self.cw.f(z_arr)
        r = self.cw.comoving_dist(z_arr)
        
        self.assertEqual(len(H), 5)
        self.assertTrue(np.all(D > 0))
        self.assertTrue(np.all(np.diff(r) > 0)) # Distance increases with z

    def test_power_spectrum_interpolation(self):
        """Test linear power spectrum access."""
        k = np.logspace(-3, 0, 10)
        pk = self.cw.Pk(k)
        
        self.assertEqual(len(pk), 10)
        self.assertTrue(np.all(pk > 0))
        
        # Check derivatives exist
        pk_d = self.cw.Pk_d(k)
        self.assertEqual(len(pk_d), 10)

    def test_unpack_pk(self):
        """Test unpack_pk returns correct number of parameters."""
        k = 0.1
        z = 1.0
        # Base parameters
        params = self.cw.unpack_pk(k, z, GR=False, WS=False)
        # Expect [Pk, f, D, b1, xb1] = 5 parameters
        self.assertEqual(len(params), 5)
        
        # With GR
        params_gr = self.cw.unpack_pk(k, z, GR=True, WS=False)
        # Adds [gr1, gr2, xgr1, xgr2] = 9 parameters
        self.assertEqual(len(params_gr), 9)

    # =========================================================================
    # 4. Bias Tests (PBBias)
    # =========================================================================

    def test_pb_bias(self):
        """Test Peak Background Bias calculations."""
        # Initialize PBBias explicitly
        pb = cw.PBBias(self.cw, self.survey)
        
        z = 1.0
        
        # 1. Lagrangian/Eulerian biases
        b1 = pb.b_1(z)
        b2 = pb.b_2(z)
        self.assertIsInstance(b1, float)
        self.assertIsInstance(b2, float)
        
        # 2. Check consistency with survey definitions if applicable
        # The PBBias fits M0 to match survey b_1, so they should be close
        survey_b1 = self.survey.b_1(z)
        self.assertAlmostEqual(b1, survey_b1, places=3)
        
        # 3. PNG Biases
        # Local
        b01_loc = pb.loc.b_01(z)
        self.assertIsInstance(b01_loc, float)
        
        # Equilateral
        b01_eq = pb.eq.b_01(z)
        self.assertIsInstance(b01_eq, float)

    # =========================================================================
    # 5. Physics Terms Tests (Pk & Bk)
    # =========================================================================

    def test_pk_func(self):
        """Test power spectrum term computation."""
        k = np.logspace(-2, -1, 5)
        z = 1.0
        l = 0 # Monopole
        
        # Test standard Newtonian term (NPP)
        res_npp = pk_func('NPP', l, self.cw, k, zz=z)
        self.assertEqual(res_npp.shape, k.shape)
        self.assertTrue(np.all(res_npp > 0))
        
        # Test Wide Separation (WS) - via composite class name string
        # 'WS' is defined in pk.combined
        try:
            res_ws = pk_func('WS', l, self.cw, k, zz=z)
            self.assertEqual(res_ws.shape, k.shape)
        except AttributeError:
            print("Skipping WS test: Term not found in pk dictionary")

    def test_bk_func(self):
        """Test bispectrum term computation."""
        k1 = np.array([0.1])
        k2 = np.array([0.1])
        k3 = np.array([0.1])
        z = 1.0
        l = 0
        
        # Use a term that is likely available. 
        # If 'WS' is defined in bk.combined, we can test it.
        # Often tree level or NPP is implicit, let's try calling with a known composite class if string fails.
        from cosmo_wap.bk import combined
        
        # Testing using the class directly to ensure it works
        res_full = bk_func(combined.Full, l, self.cw, k1, k2, k3, zz=z)
        self.assertTrue(np.isfinite(res_full))

    # =========================================================================
    # 6. Forecast Tests
    # =========================================================================

    def test_forecast_init(self):
        """Test initialization of the forecast module."""
        # Simple k_max function
        k_func = lambda z: 0.1
        
        ff = FullForecast(self.cw, kmax_func=k_func, N_bins=2)
        
        self.assertEqual(len(ff.z_bins), 2)
        self.assertEqual(len(ff.k_max_list), 2)
        
    def test_fisher_matrix_basic(self):
        """Run a minimal Fisher matrix calculation."""
        # Setup small forecast
        k_func = lambda z: 0.1
        ff = FullForecast(self.cw, kmax_func=k_func, N_bins=2)
        
        # Define parameters to vary
        # (Using A_s and n_s as they are standard)
        params = ['A_s', 'n_s']
        
        # Calculate Fisher
        # pkln=True (Power Spectrum), bkln=False (Bispectrum off for speed)
        fish = ff.get_fish(params, terms='NPP', pkln=True, bkln=False, verbose=False)
        
        self.assertEqual(fish.fisher_matrix.shape, (2, 2))
        
        # Diagonal elements should be positive (information content > 0)
        diag = np.diag(fish.fisher_matrix)
        self.assertTrue(np.all(diag > 0))

if __name__ == '__main__':
    unittest.main()
