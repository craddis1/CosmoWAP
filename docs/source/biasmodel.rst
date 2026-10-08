
Bias Modelling
==============

We can compute higher order biases and scale-dependent biases from PNG from a given Halo Occupation Distribution (HOD) and a Halo Mass Function (HMF). This is implemented in the ``HOD`` subpackage, which contains three modules:

- ``hods`` - HOD models. All HOD models inherit from the abstract base class ``BaseHOD``, which defines the interface (``get_hod_params``, ``HOD``). Currently provides the ``YP`` (Yankelevich & Porciani 2018) and ``Smith_BGS`` (Smith et al. 2024) models.
- ``hmf`` - Halo mass function multiplicity functions, Lagrangian biases, and halo number density.
- ``peak_background_bias`` - Peak-background split galaxy bias computation using the HOD.

Cosmological quantities needed by the HOD/HMF pipeline (``sig_R``, ``rho_m``, ``M_halo``, etc.) are computed lazily by ``HMF`` and cached on ``ClassWAP`` so they are only computed once regardless of the number of tracers.

Except for equilateral and orthogonal PNG, the biases can instead be given directly as redshift-dependent functions on the survey object (see :doc:`surveyparams`).

HOD Models
----------

All HOD models inherit from ``BaseHOD``, which defines the interface:

.. py:class:: BaseHOD
   :module: cosmo_wap.HOD.hods

   Abstract base class for Halo Occupation Distribution models.

   .. method:: get_hod_params(zz, m_c=None)
      :abstractmethod:

      Return HOD parameters as a tuple for the given redshift(s).

   .. method:: HOD(zz, *args)
      :abstractmethod:

      Return the mean number of galaxies per halo N(M) at redshift ``zz``.

**YP** - Yankelevich & Porciani (2018) [`arXiv:1807.07076 <https://arxiv.org/abs/1807.07076>`_]. Two free parameters (M0, NO) fitted to the survey's linear bias and number density. If the survey's b_1 is below the lowest bias the HOD can reach at some z, it warns and uses that minimum instead. Used by default for spectroscopic surveys (Euclid, Roman, SKA, etc.).

**Smith_BGS** - Smith et al. (2024). Five-parameter HOD (Mmin, sigma, M0, M1, alpha) with best-fit parameters from AbacusSummit. Used for the DESI BGS survey, where the HOD parameters are functions of a threshold apparent magnitude ``m_c``.

To add a new HOD model, subclass ``BaseHOD`` and implement ``get_hod_params`` and ``HOD`` (and optionally ``fit_params`` if the model has free parameters that need fitting).

HMF
---

The ``HMF`` class provides halo mass function multiplicity functions, Lagrangian bias models, and the halo number density ``n_h``.

.. py:class:: HMF(cosmo_funcs, hmf='Tinker10')
   :module: cosmo_wap.HOD.hmf

   **Parameters**:

   - **cosmo_funcs**: An instance of ``ClassWAP``. HMF initialises its cached cosmological quantities (``sig_R``, ``delta_c``, ``rho_m``, ``M_halo``) if needed.
   - **hmf**: Choice of HMF. ``'Tinker10'`` (default) uses the Tinker 2010 multiplicity function with analytic Lagrangian biases. ``'ST'`` uses the Sheth-Tormen multiplicity function. Any other value falls back to Tinker 2010 multiplicity with numeric Lagrangian biases.

   **Key attributes**:

   - **nu_func** - peak height :math:`\nu(z,R) = \delta_c / \sigma(z,R)`
   - **multiplicity** - the selected multiplicity function :math:`f(\nu)`
   - **lagbias** - Lagrangian bias model (``LagBias_Tinker10``, ``LagBias_ST``, or numeric ``LagBias``)

   **Methods**:

   .. method:: n_h(zz)

      Halo number density per unit mass dn/dM at redshift ``zz``. Returns an array over the radius grid R.

.. py:function:: cosmo_wap.HOD.hmf.sigma_R_n(cosmo_funcs, R, n, K_MIN=5e-5, N_k=200)

   Compute :math:`\sigma_n^2(R)` at z=0 using Simpson integration on a log-spaced k grid. ``n`` can be a single moment (0, -1 or -2), or a sequence of moments sharing the same grid. HMF caches the results; ``sig_R`` adds the :math:`D(z)^2` dependence.

   :param cosmo_funcs: ``ClassWAP`` instance
   :param R: Radius array in Mpc/h
   :param n: Moment or sequence of moments
   :param float K_MIN: Minimum k in h/Mpc
   :param int N_k: Number of k-grid points
   :return: Array over R, or one row per requested moment

PBBias
------

The ``PBBias`` class computes non-Gaussian biases using the Peak Background Split (PBS) approach. It uses an ``HMF`` instance for the halo mass function and Lagrangian biases, and an HOD model (selected via the ``hod`` parameter) to relate halo masses to galaxy occupation numbers.

For the default ``YP`` HOD, the free parameters are fit to the survey's linear bias and number density, so the only inputs needed are b_1(z), n_g(z), and cosmology. For ``Smith_BGS``, the HOD parameters are derived from the apparent magnitude cut ``m_c``.

.. py:class:: PBBias(cosmo_funcs, survey_params, hmf='Tinker10', hod='YP', cut=None, p=1.0)
   :module: cosmo_wap.HOD.peak_background_bias

   This class computes second-order bias and non-Gaussian biases from the HMF and HOD for a given survey and cosmology. These are then transferred onto the ``SetSurveyFunctions`` object via the ``add_bias_attr`` method, making them available for use in power spectrum and bispectrum calculations.

   **Parameters**:

   - **cosmo_funcs**: An instance of ``ClassWAP`` with cosmological information. HMF initialises the cached HOD/HMF quantities (``sig_R``, ``R``, ``M_halo``, ``rho_m``, ``delta_c``) if needed.
   - **survey_params**: An instance of ``SurveyParams`` containing survey parameters, where the relevant parameters are the linear bias (``b_1``) and the number density (``n_g``).
   - **hmf**: Choice of HMF passed to ``HMF``. ``'Tinker10'`` (default) or ``'ST'`` (Sheth-Tormen).
   - **hod**: Choice of HOD model. ``'YP'`` (default) or ``'Smith_BGS'``.
   - **cut**: Apparent magnitude cut, only used by ``Smith_BGS``.
   - **p**: Merger exponent in the local PNG bias :math:`b_\phi = 2\delta_c(b_1-p)` (default: 1.0).

   **Computed biases** (passed to ``SetSurveyFunctions`` via ``add_bias_attr``):

   - **b_2** - second-order Eulerian bias
   - **g_2** - tidal bias (from local Lagrangian approximation)

   Non-Gaussian bias parameters for each PNG type:

   - **loc** (local-type PNG, :math:`A=1, \alpha=0`)
   - **eq** (equilateral-type PNG, :math:`A=3, \alpha=2`)
   - **orth** (orthogonal-type PNG, :math:`A=-3, \alpha=1`)

   Each of these contains:

   - **b_01** (:math:`b_{\psi}`)
   - **b_11** (:math:`b_{\psi \delta}`)


Usage Example
-------------

Here's how you can use the bias modelling with ``compute_bias=True``:

.. code-block:: python

    import matplotlib.pyplot as plt
    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)

    # compute_bias=True triggers HOD/HMF pipeline
    cosmo_funcs = cw.ClassWAP(cosmo, survey, compute_bias=True)

    # Access biases via the survey tracer
    zz = cosmo_funcs.z_survey
    tracer = cosmo_funcs.survey[0]

    # Compare non-Gaussian biases for local, equilateral, orthogonal PNG
    plt.plot(zz, tracer.loc.b_11(zz), label='local')
    plt.plot(zz, tracer.eq.b_11(zz), label='equilateral')
    plt.plot(zz, tracer.orth.b_11(zz), label='orthogonal')
    plt.legend()
    plt.show()

For the BGS survey with the Smith HOD:

.. code-block:: python

    survey_bgs = cw.SurveyParams.BGS(cosmo, cut=20)
    cosmo_funcs = cw.ClassWAP(cosmo, survey_bgs, compute_bias=True)


Extending
---------

**Adding a new HOD**: Subclass ``BaseHOD`` from ``cosmo_wap.HOD.hods`` and implement ``get_hod_params`` and ``HOD``. Override ``fit_params`` if the model has free parameters to calibrate against survey data.

The ``HMF`` class can be easily extended to support additional halo mass functions using the `hmf <https://hmf.readthedocs.io/en/latest/examples/plugins_and_extending.html#Built-in-Models>`_ libraries.
