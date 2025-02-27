ClassWAP API Reference
=====================

The `ClassWAP` class is the central interface for the CosmoWAP package. It combines cosmological calculations, survey parameters, and bias models to provide a comprehensive framework for computing power spectra and bispectra.

Core Class
---------

.. py:class:: ClassWAP(cosmo, survey_params, compute_bias=False, HMF='Tinker2010', nonlin=False, growth2=False)
   :module: main

   **Parameters**:
   
   - **cosmo**: Class instance from the CLASS Python wrapper (classy)
   - **survey_params**: Either a single SurveyParams instance or a list of two such instances for multi-tracer analysis
   - **compute_bias**: Boolean flag indicating whether to compute bias parameters using the Peak Background Split approach
   - **HMF**: Halo Mass Function to use if compute_bias is True. Options are 'Tinker2010' (default) or 'ST' (Sheth-Tormen)
   - **nonlin**: Boolean flag to use nonlinear HALOFIT power spectra
   - **growth2**: Boolean flag to include second-order growth corrections to F2 and G2 kernels

   **Attributes**:
   
   - **cosmo**: The CLASS cosmology instance
   - **survey**: First survey (or only survey if single tracer)
   - **survey1**: Second survey (same as survey for single tracer)
   - **Om_0**: Matter density parameter today
   - **h**: Hubble parameter in units of 100 km/s/Mpc
   - **z_min**, **z_max**: Redshift range of the survey(s)
   - **z_survey**: Array of redshift values within the survey range
   - **f_sky**: Sky fraction of the survey(s)
   - **Pk**, **Pk_d**, **Pk_dd**: Linear power spectrum and its derivatives
   - **Pk_NL**: Nonlinear power spectrum (if nonlin=True)

   **Methods**:

   .. method:: get_class_powerspectrum(kk, zz=0)

      Return the linear power spectrum from CLASS.

      :param array-like kk: Wavevectors in h/Mpc
      :param float zz: Redshift
      :return: Linear power spectrum in (Mpc/h)^3

   .. method:: get_Pk_NL(k, z=0)

      Return the nonlinear power spectrum using HALOFIT.

      :param array-like k: Wavevectors in h/Mpc
      :param float z: Redshift
      :return: Nonlinear power spectrum in (Mpc/h)^3

   .. method:: get_params(k1, k2, k3=None, theta=None, zz=0, tracer=None, nonlin=False, growth2=False)

      Get all necessary parameters for bispectrum calculations.

      :param array-like k1: First wavevector magnitude in h/Mpc
      :param array-like k2: Second wavevector magnitude in h/Mpc
      :param array-like k3: Third wavevector magnitude in h/Mpc (optional)
      :param array-like theta: Outside angle θ (optional)
      :param float zz: Redshift
      :param object tracer: Survey tracer to use (defaults to self.survey)
      :param bool nonlin: Whether to use nonlinear power spectra
      :param bool growth2: Whether to use second-order growth corrections
      :return: Tuple of parameters needed for bispectrum calculations

   .. method:: get_params_pk(k1, zz)

      Get all necessary parameters for power spectrum calculations.

      :param array-like k1: Wavevector magnitude in h/Mpc
      :param float zz: Redshift
      :return: Tuple of parameters needed for power spectrum calculations

   .. method:: get_PNGparams(zz, k1, k2, k3, tracer=None, shape='Loc')

      Get parameters needed for primordial non-Gaussianity calculations.

      :param float zz: Redshift
      :param array-like k1: First wavevector magnitude in h/Mpc
      :param array-like k2: Second wavevector magnitude in h/Mpc
      :param array-like k3: Third wavevector magnitude in h/Mpc
      :param object tracer: Survey tracer to use (defaults to self.survey)
      :param str shape: Type of PNG ('Loc', 'Eq', or 'Orth')
      :return: Tuple containing (bE01, bE11, Mk1, Mk2, Mk3) for PNG calculations

   .. method:: get_PNGparams_pk(zz, k1, tracer=None, shape='Loc')

      Get parameters needed for PNG in power spectrum calculations.

      :param float zz: Redshift
      :param array-like k1: Wavevector magnitude in h/Mpc
      :param object tracer: Survey tracer to use (defaults to self.survey)
      :param str shape: Type of PNG ('Loc', 'Eq', or 'Orth')
      :return: Tuple containing (bE01, Mk1) for PNG calculations

   .. method:: get_derivs(zz, tracer=None)

      Get derivatives of redshift-dependent parameters for radial evolution terms.

      :param float zz: Redshift
      :param object tracer: Survey tracer to use (defaults to self.survey)
      :return: Tuple of first and second derivatives of bias parameters and growth factors

   .. method:: get_beta_funcs(zz, tracer=None)

      Get beta coefficients for relativistic contributions.

      :param float zz: Redshift
      :param object tracer: Survey tracer to use (defaults to self.survey)
      :return: List of beta coefficients for relativistic terms

   .. method:: solve_second_order_KC()

      Compute second-order growth factors for F2 and G2 kernels.

   .. method:: lnd_derivatives(functions_to_differentiate)

      Calculate derivatives of functions with respect to log comoving distance.

      :param list functions_to_differentiate: List of functions to differentiate
      :return: List of derivative functions

   .. method:: Pk_phi(k, k0=0.05, units=True)

      Compute the power spectrum of the Bardeen potential Φ in the matter-dominated era.

      :param array-like k: Wavevector magnitude in h/Mpc
      :param float k0: Pivot scale in h/Mpc
      :param bool units: Whether to include dimensionful units
      :return: Power spectrum of the Bardeen potential

   .. method:: M(k, z)

      Compute the scaling factor between primordial and late-time power spectra.

      :param array-like k: Wavevector magnitude in h/Mpc
      :param float z: Redshift
      :return: Transfer function M(k, z)

Helper Functions
---------------

The `ClassWAP` class uses several cosmological functions that are provided through attributes:

- **H_c**: Conformal Hubble parameter in h/Mpc
- **dH_c**, **ddH_c**: First and second derivatives of H_c with respect to redshift
- **comoving_dist**: Comoving distance in Mpc/h
- **d_to_z**: Inverse function mapping comoving distance to redshift
- **f_intp**: Linear growth rate
- **D_intp**: Linear growth factor
- **dD_dz**: Derivative of the growth factor with respect to redshift
- **conf_time**: Conformal time in Mpc/h
- **Om**: Matter density parameter as a function of redshift
- **rho_crit**, **rho_m**: Critical density and matter density in h³M⊙/Mpc³

Survey and Bias Parameters
------------------------

The `ClassWAP` instance provides access to survey parameters through the `survey` and `survey1` attributes. These include:

- **b_1**: Linear bias
- **b_2**: Second-order bias
- **g_2**: Tidal bias
- **be_survey**: Evolution bias
- **Q_survey**: Magnification bias
- **n_g**: Number density
- **f_sky**: Sky fraction
- **z_range**: Redshift range

For primordial non-Gaussianity, each tracer has attributes for the different PNG types:

- **tracer.loc**: Scale-dependent bias parameters for local PNG
- **tracer.eq**: Scale-dependent bias parameters for equilateral PNG
- **tracer.orth**: Scale-dependent bias parameters for orthogonal PNG

Each of these contains:

- **b_01**: First-order scale-dependent bias parameter
- **b_11**: Second-order scale-dependent bias parameter

Example Usage
------------

Here's a basic example of using `ClassWAP`:

```python
import cosmo_wap as cw
from classy import Class

# Initialize cosmology
cosmo = cw.utils.get_cosmology()

# Get survey parameters
survey_params = cw.survey_params.SurveyParams(cosmo)

# Initialize ClassWAP
cosmo_funcs = cw.ClassWAP(cosmo, survey_params.Euclid)

# Calculate power spectrum parameters at k=0.1 h/Mpc and z=1
k = 0.1
z = 1.0
params_pk = cosmo_funcs.get_params_pk(k, z)

# Calculate bispectrum parameters for an equilateral triangle
params_bk = cosmo_funcs.get_params(k, k, k, zz=z)

# Get PNG parameters for local type
png_params = cosmo_funcs.get_PNGparams(z, k, k, k, shape='Loc')
```

For more detailed examples, see the [Getting Started](getting_started.html) page.
