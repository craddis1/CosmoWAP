Survey Parameters
=================

The ``SurveyParams`` class defines survey-specific parameters for use in CosmoWAP. This module contains predefined parameters for several surveys, as well as functionality to customise specifications and create multi-tracer samples.

Available Surveys
-----------------

.. py:class:: SurveyParams.Euclid(cosmo, fitting=False, model3=True, F_c=None)

   Euclid Hα galaxy survey (0.9 < z < 1.8).

   :param cosmo: CLASS cosmology instance
   :param bool fitting: Use polynomial fits instead of luminosity function (default: False)
   :param bool model3: Use Model 3 luminosity function (default: True)
   :param float F_c: Flux cut [erg/cm²/s] (default: 2e-16 for model3, 3e-16 for model1)

.. py:class:: SurveyParams.Roman(cosmo, model3=False, F_c=None)

   Roman Space Telescope Hα survey (0.5 < z < 2.0).

.. py:class:: SurveyParams.BGS(cosmo, m_c=20, fitting=False)

   DESI Bright Galaxy Sample (0.05 < z < 0.6).

   :param float m_c: Apparent magnitude cut (default: 20)

.. py:class:: SurveyParams.MegaMapper(cosmo, m_c=24.5)

   MegaMapper LBG survey (2.1 < z < 5.0).

   :param float m_c: Apparent magnitude cut (default: 24.5)

.. py:class:: SurveyParams.SKAO1(cosmo)

   SKA Observatory Phase 1 HI galaxy survey.

.. py:class:: SurveyParams.SKAO2(cosmo)

   SKA Observatory Phase 2 HI galaxy survey.

.. py:class:: SurveyParams.DM_part(cosmo)

   Dark matter particles (b₁ = 1, for testing).

Survey Attributes
-----------------

Each survey class provides:

- **b_1**: Linear bias b₁(z)
- **z_range**: Redshift range [z_min, z_max]
- **be**: Evolution bias bₑ(z)
- **Q**: Magnification bias Q(z)
- **n_g**: Number density n_g(z) [h³/Mpc³]
- **f_sky**: Sky fraction

Optional (if defined or computed via ``PBBias``):

- **b_2**: Second-order bias
- **g_2**: Tidal bias
- **loc.b_01**, **loc.b_11**: Local PNG scale-dependent bias

Basic Usage
-----------

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31)

    # Single tracer
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    # Access survey attributes
    z = 1.2
    print(f"b_1(z=1.2) = {survey.b_1(z):.2f}")
    print(f"n_g(z=1.2) = {survey.n_g(z):.4f} h³/Mpc³")

    # Modify parameters
    survey_half_sky = survey.update(f_sky=0.5)

Multi-Tracer Analysis
---------------------

Pass a list of survey objects to ``ClassWAP`` for multi-tracer analysis:

.. code-block:: python

    # Two different surveys
    survey_euclid = cw.SurveyParams.Euclid(cosmo)
    survey_ska = cw.SurveyParams.SKAO2(cosmo)
    cosmo_funcs_mt = cw.ClassWAP(cosmo, [survey_euclid, survey_ska])

    # Or bright/faint split of same survey (see below)
    bright, faint = cw.SurveyParams.Euclid(cosmo).BF_split(5e-16)
    cosmo_funcs_bf = cw.ClassWAP(cosmo, [bright, faint])

Multi-tracer forecasting then accounts for cross-correlations between tracers.

Bright-Faint Split
------------------

Surveys with luminosity functions can be split into bright and faint subsamples at a given flux/magnitude cut. This enables multi-tracer analysis from a single survey.

.. py:method:: SurveyBase.BF_split(split)

   Split survey into bright and faint subsamples.

   :param float split: Flux cut [erg/cm²/s] for Hα surveys, or magnitude cut for magnitude-limited surveys. Must be brighter than the survey's detection limit.
   :return: List of [bright, faint] survey objects
   :raises ValueError: If survey has no luminosity function defined

The faint sample parameters are derived from:

- n_faint = n_total - n_bright
- b₁_faint = (n_total × b₁_total - n_bright × b₁_bright) / n_faint

.. code-block:: python

    # Euclid with default flux cut 2e-16 erg/cm²/s
    survey = cw.SurveyParams.Euclid(cosmo)

    # Split at brighter flux cut (5e-16)
    bright, faint = survey.BF_split(5e-16)

    # Check number densities
    z = 1.2
    print(f"n_bright = {bright.n_g(z):.4f}")
    print(f"n_faint = {faint.n_g(z):.4f}")
    print(f"n_total = {survey.n_g(z):.4f}")

    # Use in multi-tracer forecast
    cosmo_funcs = cw.ClassWAP(cosmo, [bright, faint])

Luminosity Functions
--------------------

CosmoWAP includes luminosity function classes that compute number densities, magnification bias Q, and evolution bias bₑ from physical models.

Hα Luminosity Functions
~~~~~~~~~~~~~~~~~~~~~~~

For flux-limited Hα surveys (Euclid, Roman):

.. py:class:: lib.luminosity_funcs.Model1LuminosityFunction(cosmo)
.. py:class:: lib.luminosity_funcs.Model3LuminosityFunction(cosmo)

   Schechter-type Hα luminosity functions. Model 3 includes a modified faint-end slope.

   **Methods:**

   .. method:: luminosity_function(L, zz)

      Compute Φ(L, z) [h³/Mpc³].

   .. method:: number_density(F_c, zz)

      Compute n_g above flux cut F_c [erg/cm²/s].

   .. method:: get_Q(F_c, zz)

      Compute magnification bias Q(z).

   .. method:: get_be(F_c, zz)

      Compute evolution bias bₑ(z).

   .. method:: get_b_1(F_c, zz)

      Compute flux-averaged linear bias b₁(z) using semi-analytic model.

Magnitude-Limited Surveys
~~~~~~~~~~~~~~~~~~~~~~~~~

For apparent magnitude-limited surveys (BGS, MegaMapper):

.. py:class:: lib.luminosity_funcs.BGSLuminosityFunction(cosmo)

   DESI BGS r-band luminosity function.

.. py:class:: lib.luminosity_funcs.LBGLuminosityFunction(cosmo)

   Lyman Break Galaxy UV luminosity function for MegaMapper.

   **Methods:** Same as above, but with magnitude cut ``m_c`` instead of flux cut.

Direct Luminosity Function Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cosmo_wap.lib.luminosity_funcs import Model3LuminosityFunction
    from cosmo_wap.lib import utils
    import numpy as np

    cosmo = utils.get_cosmo()
    LF = Model3LuminosityFunction(cosmo)

    z = np.linspace(0.9, 1.8, 50)
    F_c = 2e-16  # erg/cm²/s

    # Number density vs redshift
    n_g = LF.number_density(F_c, z)

    # Magnification and evolution bias
    Q = LF.get_Q(F_c, z)
    be = LF.get_be(F_c, z)

    # Linear bias (semi-analytic)
    b1 = LF.get_b_1(F_c, z)

Custom Surveys
--------------

Create custom survey parameters:

.. code-block:: python

    import numpy as np

    # Start from template
    custom = cw.SurveyParams.InitNew(cosmo)

    # Set parameters
    custom.b_1 = lambda z: 1.0 + 0.5 * z
    custom.z_range = [0.5, 2.0]
    custom.n_g = lambda z: 0.01 * np.exp(-z)
    custom.Q = lambda z: 0.4 + 0 * z
    custom.be = lambda z: 0.0 + 0 * z
    custom.f_sky = 0.3

    cosmo_funcs = cw.ClassWAP(cosmo, custom)
