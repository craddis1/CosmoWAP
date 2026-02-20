ClassWAP API Reference
======================

The `ClassWAP` class is the central interface for the CosmoWAP package. It combines cosmology and survey biases and processes and stores them to provide a framework for computing power spectra and bispectra.

Core Class
----------

.. py:class:: ClassWAP(cosmo, survey_params=None, compute_bias=False, HMF='Tinker2010', emulator=False, verbose=True, params=None, fast=False, nonlin=False)
   :module: main

   Initialise CosmoWAP from a CLASS cosmology and (optionally) survey parameters.

   Interpolates cosmological background quantities (e.g. D, f, H etc) from CLASS,
   computes linear P(k) and optionally non-linear P(k) using halofit or an emulator.
   If survey parameters are passed, computes bias functions for given survey(s) and
   optionally derives higher order bias functions from HMF/HOD relations.

   **Parameters**:

   - **cosmo**: Class instance from the CLASS Python wrapper (classy)
   - **survey_params**: Either a single SurveyParams instance or a list of two such instances for multi-tracer analysis
   - **compute_bias**: Boolean flag. If True, derive higher order bias functions and scale-dependent PNG biases from the HMF/HOD
   - **HMF**: Halo Mass Function to use if compute_bias is True. Options are 'Tinker2010' (default) or 'ST' (Sheth-Tormen)
   - **emulator**: If True, initialise a CosmoPower emulator internally. Pass a pre-loaded ``Emulator`` instance to reuse it across multiple ``ClassWAP`` objects
   - **verbose**: Print progress messages when computing bias functions
   - **params**: Pre-computed cosmological parameters dict (h, Omega_m, ...) to load directly instead of querying CLASS — useful for speeding up MCMC sampling
   - **fast**: If True (and ``nonlin`` is False), skip building the non-linear P(k,z) grid
   - **nonlin**: If True, always build the non-linear halofit/emulator P(k,z) grid

   **Attributes**:

   - **cosmo**: The CLASS cosmology instance
   - **survey**: List of survey objects (up to 3 for bispectrum multi-tracer). Access via ``survey[0]``, ``survey[1]``, etc.
   - **multi_tracer**: Boolean flag indicating if multiple distinct tracers are in use
   - **h**: Hubble parameter in units of 100 km/s/Mpc
   - **Omega_m**, **Omega_b**, **Omega_cdm**: Density parameters
   - **A_s**, **n_s**: Primordial power spectrum amplitude and spectral index
   - **z_min**, **z_max**: Redshift range of the survey(s) (intersection if multi-tracer)
   - **z_survey**: Array of redshift values within the survey range
   - **f_sky**: Sky fraction of the survey(s)
   - **Pk**, **Pk_d**, **Pk_dd**: Linear power spectrum and its first two k-derivatives (CubicSpline)
   - **Pk_NL**: 2D interpolated nonlinear power spectrum P(k,z) (if built)
   - **D**: Linear growth factor (CubicSpline in z)
   - **f**: Linear growth rate (CubicSpline in z)
   - **H_c**: Conformal Hubble parameter in h/Mpc (CubicSpline in z)
   - **dH_c**: First derivative of H_c with respect to redshift
   - **comoving_dist**: Comoving distance in Mpc/h (CubicSpline in z)
   - **d_to_z**: Inverse mapping from comoving distance to redshift
   - **Om_m**: Matter density parameter as a function of redshift

   **Methods**:

   .. method:: get_class_powerspectrum(kk, zz=0)

      Return the linear power spectrum from CLASS.

      :param array-like kk: Wavevectors in h/Mpc
      :param float zz: Redshift
      :return: Linear power spectrum in (Mpc/h)^3

   .. method:: get_Pk_NL(kk, zz)

      Build 2D (k,z) interpolated nonlinear power spectrum. Uses halofit via CLASS
      or HMCode via the CosmoPower emulator. Factors out D(z)^2 dependence and falls
      back to linear P(k) on large scales.

      :param array-like kk: Wavevectors in h/Mpc
      :param array-like zz: Array of redshifts
      :return: Callable f(k, z) returning nonlinear P(k) at given (k,z)

   .. method:: pk(k)

      Linear power spectrum with a k^{-3} power-law extrapolation beyond K_MAX.

      :param array-like k: Wavevectors in h/Mpc
      :return: P(k) in (Mpc/h)^3

   .. method:: update_survey(survey_params, verbose=True)

      Set up (or update) survey bias functions for one or two tracers, including
      computing derivatives and shared survey properties.

      :param survey_params: SurveyBase or list[SurveyBase]
      :param bool verbose: Print progress messages
      :return: self

   .. method:: get_PNGparams(zz, k1, k2, k3, ti=0, shape='Loc')

      Get parameters needed for primordial non-Gaussianity bispectrum calculations.

      :param float zz: Redshift
      :param array-like k1: First wavevector magnitude in h/Mpc
      :param array-like k2: Second wavevector magnitude in h/Mpc
      :param array-like k3: Third wavevector magnitude in h/Mpc
      :param int ti: Tracer index (0 or 1)
      :param str shape: Type of PNG ('Loc', 'Eq', or 'Orth')
      :return: Tuple containing (bE01, bE11, Mk1, Mk2, Mk3)

   .. method:: get_PNGparams_pk(zz, k1, ti=0, shape='Loc')

      Get parameters needed for PNG in power spectrum calculations.

      :param float zz: Redshift
      :param array-like k1: Wavevector magnitude in h/Mpc
      :param int ti: Tracer index (0 or 1)
      :param str shape: Type of PNG ('Loc', 'Eq', or 'Orth')
      :return: Tuple containing (bE01, Mk1)

   .. method:: compute_derivs(ti=None)

      Compute derivatives with respect to comoving distance of redshift-dependent parameters
      for radial evolution terms. If ``ti`` is given, computes survey-dependent derivatives
      for that tracer. If None, computes cosmology-dependent derivatives.

      :param int ti: Tracer index, or None for cosmology derivatives
      :return: The tracer or self

   .. method:: get_beta_funcs(zz, ti=0)

      Get beta coefficients for relativistic contributions. Computes and caches them
      on first call.

      :param float zz: Redshift
      :param int ti: Tracer index (0 or 1)
      :return: List of beta coefficient arrays

   .. method:: get_beta_derivs(zz, ti=0)

      Get derivatives of beta coefficients with respect to comoving distance.

      :param float zz: Redshift
      :param int ti: Tracer index (0 or 1)
      :return: List of beta derivative arrays

   .. method:: solve_second_order_KC()

      Compute second-order growth factors — redshift-dependent corrections to F2 and G2 kernels.
      Sets ``self.K_intp`` and ``self.C_intp`` attributes.

   .. method:: lnd_derivatives(functions_to_differentiate, ti=0)

      Calculate derivatives of a list of functions with respect to log comoving distance.

      :param list functions_to_differentiate: List of callables f(z)
      :param int ti: Tracer index (determines redshift sampling)
      :return: List of CubicSpline derivative functions

   .. method:: Pk_phi(k, k0=0.05)

      Power spectrum of the Bardeen potential Phi in the matter-dominated era.

      :param array-like k: Wavevector magnitude in h/Mpc
      :param float k0: Pivot scale in 1/Mpc (default 0.05)
      :return: P_phi(k) in (Mpc/h)^3

   .. method:: M(k, z)

      Scaling factor between the primordial scalar power spectrum and the late-time
      matter power spectrum in linear theory.

      :param array-like k: Wavevector magnitude in h/Mpc
      :param float z: Redshift
      :return: M(k, z)

Cosmological Functions
----------------------

The ``ClassWAP`` class provides the following interpolated cosmological functions as attributes.
All are ``CubicSpline`` objects callable with redshift ``z`` unless stated otherwise.

- **H_c**: Conformal Hubble parameter in h/Mpc
- **dH_c**: First derivative of H_c with respect to redshift, in h/Mpc
- **comoving_dist**: Comoving distance in Mpc/h
- **d_to_z**: Inverse function mapping comoving distance (Mpc/h) to redshift
- **f**: Linear growth rate (dimensionless)
- **D**: Linear growth factor (normalised to unity at z=0)
- **Om_m**: Matter density parameter as a function of redshift (dimensionless)
- **Pk**: Linear power spectrum P(k) at z=0 in (Mpc/h)^3, callable with k in h/Mpc
- **Pk_d**, **Pk_dd**: First and second k-derivatives of Pk
- **Pk_NL**: 2D interpolated nonlinear power spectrum, callable as ``Pk_NL(k, z)`` with k in h/Mpc. D(z)^2 dependence is factored out
- **c**: Speed of light in km/s

Survey and Bias Parameters
--------------------------

The ``ClassWAP`` instance provides access to survey parameters through the ``survey`` list. Each element includes:

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
-------------

Here's a basic example of using ``ClassWAP``:

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    # Initialize cosmology with CLASS
    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=4.0)

    # Get survey parameters
    survey = cw.SurveyParams.Euclid(cosmo)

    # Initialize ClassWAP
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    # Access cosmological functions
    z = 1.0
    print(f"Growth rate f(z=1) = {cosmo_funcs.f(z):.3f}")
    print(f"Growth factor D(z=1) = {cosmo_funcs.D(z):.3f}")

    # Access survey parameters
    print(f"Linear bias b1(z=1) = {cosmo_funcs.survey[0].b_1(z):.3f}")

For more detailed examples, see the :doc:`Getting Started <getting_started>` page.
