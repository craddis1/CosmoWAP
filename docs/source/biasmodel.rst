
Bias Modelling
==============

We can compute higher order biases and scale-dependent biases from PNG from a given Halo Occupation Distribution (HOD) and a Halo Mass Function (HMF). This is implemented in the ``HOD`` subpackage, which contains three modules:

- ``hmf`` — Halo mass function multiplicity functions, Lagrangian biases, and halo number density.
- ``peak_background_bias`` — Peak-background split galaxy bias computation using the HOD.
- ``Smith_HOD`` — Alternative HOD model.

Cosmological quantities needed by the HOD/HMF pipeline (``sigma_R``, ``rho_m``, ``M(z,R)``, etc.) are precomputed and cached on ``ClassWAP`` via ``setup_hod_hmf`` so they are only computed once regardless of the number of tracers.

Barring equilateral and orthogonal PNG — the biases can also be given from polynomial fits — see SetSurveyFunctions.

HMF
---

The ``HMF`` class provides halo mass function multiplicity functions, Lagrangian bias models, and the halo number density ``n_h``.

.. py:class:: HMF(cosmo_funcs, hmf=’Tinker10’)
   :module: cosmo_wap.HOD.hmf

   **Parameters**:

   - **cosmo_funcs**: An instance of ``ClassWAP`` providing cached cosmological quantities (``sig_R``, ``delta_c``, ``rho_m``, ``M``).
   - **hmf**: Choice of HMF. ``’Tinker10’`` (default) uses the Tinker 2010 multiplicity function with analytic Lagrangian biases. ``’ST’`` uses the Sheth-Tormen multiplicity function. Any other value falls back to Tinker 2010 multiplicity with numeric Lagrangian biases.

   **Key attributes**:

   - **nu_func** — peak height :math:`\nu(z,R) = \delta_c / \sigma(z,R)`
   - **multiplicity** — the selected multiplicity function :math:`f(\nu)`
   - **lagbias** — Lagrangian bias model (``LagBias_Tinker10``, ``LagBias_ST``, or numeric ``LagBias``)

   **Methods**:

   .. method:: n_h(zz)

      Halo number density per unit mass dn/dM at redshift ``zz``. Returns an array over the radius grid R.

PBBias
------

The ``PBBias`` class computes non-Gaussian biases using the Peak Background Split (PBS) approach. It uses an ``HMF`` instance for the halo mass function and Lagrangian biases, and the HOD from Yankelevich and Porciani (2018) whereby the free parameters in the HOD are fit to the linear bias and number density of the survey.
So the only things that need to be defined are the linear bias and the number density of the survey (and cosmology).

.. py:class:: PBBias(cosmo_funcs, survey_params, hmf=’Tinker10’)
   :module: cosmo_wap.HOD.peak_background_bias

   This class computes second-order bias and non-Gaussian biases from the HMF and HOD for a given survey and cosmology. These are then transferred onto the ``SetSurveyFunctions`` object via the ``add_bias_attr`` method, making them available for use in power spectrum and bispectrum calculations.

   **Parameters**:

   - **cosmo_funcs**: An instance of ``ClassWAP`` that contains cosmological information and cached HOD/HMF quantities (``sig_R``, ``R``, ``M``, ``rho_m``, ``delta_c``).
   - **survey_params**: An instance of ``SurveyParams`` containing survey parameters, where the relevant parameters are the linear bias (``b_1``) and the number density (``n_g``).
   - **hmf**: Choice of HMF passed to ``HMF``. ``’Tinker10’`` (default) or ``’ST’`` (Sheth-Tormen).

   **Computed biases** (passed to ``SetSurveyFunctions`` via ``add_bias_attr``):

   - **b_2** — second-order Eulerian bias
   - **g_2** — tidal bias (from local Lagrangian approximation)

   Non-Gaussian bias parameters for each PNG type:

   - **loc** (local-type PNG, :math:`A=1, \alpha=0`)
   - **eq** (equilateral-type PNG, :math:`A=3, \alpha=2`)
   - **orth** (orthogonal-type PNG, :math:`A=-3, \alpha=1`)

   Each of these contains:

   - **b_01** (:math:`b_{\psi}`)
   - **b_11** (:math:`b_{\psi \delta}`)


Usage Example
-------------

Here’s how you can use the bias modelling with ``compute_bias=True``:

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
    plt.plot(zz, tracer.loc.b_11(zz), label=’local’)
    plt.plot(zz, tracer.eq.b_11(zz), label=’equilateral’)
    plt.plot(zz, tracer.orth.b_11(zz), label=’orthogonal’)
    plt.legend()
    plt.show()

    # Access HOD free parameters from the fit
    plt.plot(zz, tracer.M0_func(zz))
    plt.ylabel(‘M_0’)
    plt.xlabel(‘z’)
    plt.show()

    plt.plot(zz, tracer.NO_func(zz))
    plt.ylabel(‘N_O’)
    plt.xlabel(‘z’)
    plt.show()


Extending
---------

The ``HMF`` class can be easily extended to support additional halo mass functions using the `hmf <https://hmf.readthedocs.io/en/latest/examples/plugins_and_extending.html#Built-in-Models>`_ libraries.
