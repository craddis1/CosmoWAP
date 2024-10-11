
Bias Modelling
==============

In order to calculate higher order bias parameters and scale-depnedent biases from PNG we need to assume a Halo-occupation-distribution (HOD) and a Halo-mass-function (HMF). This is implemented in the PBBias class.

PBBias
------

The `PBBias` class computes non-Gaussian biases using the Peak Background Split (PB) approach. It assumes the Halo Mass Function (HMF) from Tinker (2010) and the Halo Occupation Distribution (HOD) from Yankelevich and Porciani (2018). This class is used to compute biases based on cosmological functions and survey parameters.

.. py:class:: PBBias(cosmo_functions, survey_params)

   This class computes the non-Gaussian biases.

   **Parameters**:
   
   - **cosmo_functions**: A module or class that provides cosmological functions like growth functions, power spectra, etc.
   - **survey_params**: A dictionary or object containing survey parameters such as redshift, volume, and bias models.

Methods
-------

.. py:method:: __init__(cosmo_functions, survey_params)
   :module: your_module_name

   Initializes the `PBBias` class with cosmological functions and survey parameters.

   :param cosmo_functions: Cosmological functions needed for bias calculations.
   :param survey_params: Survey-specific parameters for the bias calculations.

Usage Example
-------------

Here’s how you can instantiate the `PBBias` class:

.. code-block:: python

    from your_module import PBBias

    pb_bias = PBBias(cosmo_functions, survey_params)
