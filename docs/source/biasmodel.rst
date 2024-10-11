
Bias Modelling
==============

In order to calculate higher order bias parameters and scale-depnedent biases from PNG as functions of redshift we need to assume a Halo-occupation-distribution (HOD) and a Halo-mass-function (HMF). This is implemented in the PBBias class. See ... ref

PBBias
------

The `PBBias` class computes non-Gaussian biases using the Peak Background Split (PB) approach. It assumes the Halo Mass Function (HMF) from Tinker (2010) and the Halo Occupation Distribution (HOD) from Yankelevich and Porciani (2018) whereby the free parameters in the HOD are fit to the linear bias and number density of the survey.

.. py:class:: PBBias(cosmo_funcs, survey_params)
   :module: peak_background_bias
   
   This class computes second-order and non-Gaussian biases for a given survey and cosmology. It stores the computed bias functions as attributes.

   **Parameters**:
   
   - **cosmo_funcs**: An instance of `ClassWAP` that contains cosmological information.
   - **survey_params**: An instance of `SurveyParams` containing survey parameters, where the relevant parameters are the linear bias (`b_1`) and the number density (`n_g`).
   
   **Attributes**:
   
   `n_g`, `b_1`, `b_2`, `g_2`

   Non-Gaussian bias parameters are stored in:

   - **loc**
   - **equil**
   - **orth**

   Each of these attributes contains the following parameters:

   - **b_01** (:math:`b_{\psi}`)
   - **b_11** (:math:`b_{\psi \delta}`)
   

Usage Example
-------------

Here’s how you can instantiate the `PBBias` class:

.. code-block:: python

    from cosmo_wap import peak_background_bias as pb_bias

    survey_bias = pb_bias.PBBias(cosmo_funcs, survey_params)
    
    #compare non-Guassian biases
    zz = cosmo_funcs.z_survey
    plt.plot(zz,survey_bias.loc.b_11(zz))
    plt.plot(zz,survey_bias.equil.b_11(zz))
    plt.plot(zz,survey_bias.orth.b_11(zz))
    

    
    
    
