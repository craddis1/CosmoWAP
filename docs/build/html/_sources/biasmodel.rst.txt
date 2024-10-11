
Bias Modelling
==============

In order to calculate higher order bias parameters and scale-depnedent biases from PNG we need to assume a Halo-occupation-distribution (HOD) and a Halo-mass-function (HMF). This is implemented in the PBBias class.

PBBias
------

The `PBBias` class computes non-Gaussian biases using the Peak Background Split (PB) approach. It assumes the Halo Mass Function (HMF) from Tinker (2010) and the Halo Occupation Distribution (HOD) from Yankelevich and Porciani (2018) whereby the free parameters in the HOD are fit to the linear bias and number density of the survey.

.. py:class:: PBBias(cosmo_functions, survey_params)
   :module: peak_background_bias
   
   This class computes the non-Gaussian biases.

   **Parameters**:
   
   - **cosmo_functions**: ClassWAP instance that conatains cosmology information.
   - **survey_params**: SurveyParams instance containing survey parameters.

Usage Example
-------------

Here’s how you can instantiate the `PBBias` class:

.. code-block:: python

    from cosmo_wap import peak_background_bias as pb_bias

    survey_bias = pb_bias.PBBias(cosmo_functions, survey_params)
