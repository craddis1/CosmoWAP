
Bias Modelling
==============

In order to calculate higher order bias parameters and scale-dependent biases from PNG as functions of redshift we need to assume a Halo Occupation Distribution (HOD) and a Halo Mass Function (HMF). This is implemented in the ``PBBias`` class.

PBBias
------

The `PBBias` class computes non-Gaussian biases using the Peak Background Split (PB) approach. It assumes the Halo Mass Function (HMF) from Tinker (2010) as default and the Halo Occupation Distribution (HOD) from Yankelevich and Porciani (2018) whereby the free parameters in the HOD are fit to the linear bias and number density of the survey.

.. py:class:: PBBias(cosmo_funcs, survey_params, HMF='Tinker2010')
   :module: cosmo_wap.peak_background_bias

   This class computes second-order Eulerian biases (``b_2``, ``g_2``) and non-Gaussian biases from the HMF and HOD for a given survey and cosmology. These are then transferred onto the ``SetSurveyFunctions`` object via the ``add_bias_attr`` method, making them available for use in power spectrum and bispectrum calculations.

   **Parameters**:

   - **cosmo_funcs**: An instance of `ClassWAP` that contains cosmological information.
   - **survey_params**: An instance of `SurveyParams` containing survey parameters, where the relevant parameters are the linear bias (`b_1`) and the number density (`n_g`).
   - **HMF**: Choice of HMF. ``'Tinker2010'`` (default) uses the Tinker 2010 multiplicity function with numeric Lagrangian biases. ``'Tinker10'`` uses the same multiplicity but with analytic Lagrangian biases. ``'ST'`` uses the Sheth-Tormen multiplicity function.

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

Here’s how you can instantiate the `PBBias` class:


.. code-block:: python

    import matplotlib.pyplot as plt
    import cosmo_wap as cw
    from cosmo_wap import peak_background_bias as pb_bias
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    # Compute second-order and PNG biases
    survey_bias = pb_bias.PBBias(cosmo_funcs, survey)

    # Compare non-Gaussian biases for local, equilateral, orthogonal PNG
    zz = cosmo_funcs.z_survey
    plt.plot(zz, survey_bias.loc.b_11(zz), label='local')
    plt.plot(zz, survey_bias.eq.b_11(zz), label='equilateral')
    plt.plot(zz, survey_bias.orth.b_11(zz), label='orthogonal')
    plt.legend()
    plt.show()

    # Access HOD free parameters from the fit
    plt.plot(zz, survey_bias.M0_func(zz))
    plt.ylabel('M_0')
    plt.xlabel('z')
    plt.show()

    plt.plot(zz, survey_bias.NO_func(zz))
    plt.ylabel('N_O')
    plt.xlabel('z')
    plt.show()


HMF
---

`PBBias` currently allows for Sheth-Mo-Tormen (ST) or Tinker 2010 (Tinker2010) HMFs but this could be easily extended to a variety of HMFs using the `hmf <https://hmf.readthedocs.io/en/latest/examples/plugins_and_extending.html#Built-in-Models>`_ libraries.
