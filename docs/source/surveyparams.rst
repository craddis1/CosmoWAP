Survey Parameters
=================

The ``SurveyParams`` class defines survey-specific parameters for use in CosmoWAP. It allows for all survey-specific information to be fed into ``ClassWAP``. This module contains predefined parameters for several surveys, as well as functionality to customise specifications and create multi-tracer samples.
The preset surveys are taken from the literature and are the ones used in the forecasts of `arXiv:2407.00168 <https://arxiv.org/abs/2407.00168>`_ and `arXiv:2511.09466 <https://arxiv.org/abs/2511.09466>`_.

Defining a new survey is simple: give a linear bias model and a luminosity function (or adopt an existing one with a given flux or magnitude cut), and the evolution and magnification biases follow. Alternatively, every bias can be given directly as a function of redshift.
Second-order and PNG scale-dependent biases can then be computed from a given HOD and HMF, or again be given as functions of redshift.

Preset Surveys
--------------

.. py:class:: SurveyParams.Euclid(cosmo, fitting=False, model3=True, cut=None)

   Euclid Hα galaxy survey (0.9 < z < 1.8).

   :param cosmo: CLASS cosmology instance
   :param bool fitting: Use polynomial fits instead of luminosity function (default: False)
   :param bool model3: Use Model 3 luminosity function (default: True)
   :param float cut: Flux cut [erg/cm²/s] (default: 2e-16 for model3, 3e-16 for model1)

.. py:class:: SurveyParams.Roman(cosmo, model3=False, cut=None)

   Roman Space Telescope Hα survey (0.5 < z < 2.0).

.. py:class:: SurveyParams.BGS(cosmo, cut=20.175, flag='HOD')

   DESI Bright Galaxy Sample (0.05 < z < 0.6).

   :param float cut: Apparent magnitude cut (default: 20.175)
   :param str flag: ``'HOD'`` (default) computes biases via HOD; ``'LF'`` computes directly from the luminosity function

.. py:class:: SurveyParams.MegaMapper(cosmo, cut=24.5)

   MegaMapper LBG survey (2.1 < z < 5.0).

   :param float cut: Apparent magnitude cut (default: 24.5)

.. py:class:: SurveyParams.SKAO1(cosmo)

   SKA Observatory Phase 1 HI galaxy survey.

.. py:class:: SurveyParams.SKAO2(cosmo)

   SKA Observatory Phase 2 HI galaxy survey.

.. py:class:: SurveyParams.SPHEREx(cosmo, sample=0, cut=2e-16)

   SPHEREx all-sky spectral survey (0.1 < z < 4.3), `Doré et al. (2014) <https://arxiv.org/abs/1412.4872>`_.

   :param int sample: Redshift-accuracy subsample 0-4, i.e. σ_z/(1+z) < 0.003, 0.01, 0.03, 0.1, 0.2 (their Eq. 24)
   :param float cut: Effective flux cut [erg/cm²/s at 2.4 µm] used for Q and bₑ (default: 2e-16)

   Number density and linear bias come from the `SPHEREx public products
   <https://github.com/SPHEREx/Public-products>`_ (vendored as ``data_library/SPHERExData.txt``);
   the bias is the fit b_g(z) = A(1 + βz)^γ to the tabulated values. Q and bₑ come from the WISE
   2.4 µm luminosity function, following `arXiv:2608.18334 <https://arxiv.org/abs/2608.18334>`_.
   ``f_sky`` is 0.75, the fraction left after galactic masking.

   The five subsamples are disjoint, so any pair can be used as a multi-tracer set:

   .. code-block:: python

       bright_z, deep = cw.SurveyParams.SPHEREx(cosmo, sample=0), cw.SurveyParams.SPHEREx(cosmo, sample=1)
       cosmo_funcs = cw.ClassWAP(cosmo, [bright_z, deep])

.. note::

   SPHEREx selects on template-fitted photometric redshifts across many bands, not on 2.4 µm
   flux, so it has no true flux limit and Doré et al. quote no magnification bias. The default
   ``cut`` is the effective value used in arXiv:2608.18334, roughly 320x deeper than SPHEREx's
   actual 5σ point-source depth at 2.4 µm (19.63 AB, i.e. 51 µJy). At the real depth the WISE
   luminosity function - calibrated at z ≲ 1 - puts essentially no galaxies above z ~ 1, so
   ``cut`` is best read as the knob that places Q in a plausible range, and is exposed for that
   reason.

.. py:class:: SurveyParams.DM_part(cosmo)

   Dark matter particles (b₁ = 1, for testing).

Survey Attributes
-----------------

Each survey class provides (either as redshift dependent functions or scalars):

- **b_1**: Linear bias b₁(z)
- **z_range**: Redshift range [z_min, z_max]
- **be**: Evolution bias bₑ(z)
- **Q**: Magnification bias Q(z)
- **n_g**: Number density n_g(z) [h³/Mpc³]
- **f_sky**: Sky fraction
- **p**: Merger exponent in the UMF prediction :math:`b_\phi = 2\delta_c(b_1 - p)`, defaults to 1
- **LF**: Luminosity function object (if defined)
- **b_2**, **g_2**, **loc.b_01**, **eq.b_11**: Optional second-order and PNG biases (if defined or computed via ``PBBias``)

Basic Usage
-----------

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()

    # Single tracer LBG MegaMapper-like survey
    survey = cw.SurveyParams.MegaMapper(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    # Access survey attributes
    z = 3
    print(f"b_1(z=3) = {survey.b_1(z):.2f}")
    print(f"n_g(z=3) = {survey.n_g(z):.4f} h³/Mpc³")

    # It is simple to modify parameters
    survey_half_sky = survey.update(f_sky=0.5)

Multi-Tracer Analysis
---------------------

Pass a list of survey objects to ``ClassWAP`` for multi-tracer analysis:

.. code-block:: python

    # Two different surveys (crucially with overlapping redshifts!)
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

.. note::

   ``BF_split`` is only valid for surveys with a defined luminosity function (e.g. Euclid, Roman, BGS, MegaMapper). Surveys without one (e.g. SKAO, DM_part, custom surveys) will raise a ``ValueError``. SPHEREx also raises: it has a luminosity function, but takes n_g from the survey rather than from it, so the faint sample could not be derived consistently - use two of its five subsamples instead.

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

See Also
--------

- :doc:`Luminosity functions <luminosityfuncs>` for computing n_g, Q, and be from physical models
- :doc:`Bias modelling <biasmodel>` for computing second-order and PNG biases via Peak-Background Split
