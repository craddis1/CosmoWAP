Survey Parameters
================

The `SurveyParams` class provides a convenient way to define and access survey specific parameters for use in CosmoWAP. This module contains predefined parameters for several surveys, as well as functionality customise specifications.

SurveyParams Class
-----------------

.. py:class:: SurveyParams(cosmo=None)
   :module: survey_params

   Initialize and retrieve survey parameters for different surveys. 
   
   **Parameters:**
   
   - **cosmo**: (Optional) Input cosmology from CLASS. Required for DESI BGS since its linear bias depends on the linear growth rate.

   **Attributes:**
   
   Preset survey configurations:

   - **Euclid**: Hα galaxy survey (0.9 < z < 1.8)
   - **BGS**: DESI Bright Galaxy Sample (0.05 < z < 0.6)
   - **SKAO1**: Square Kilometre Array Phase 1 HI galaxy survey
   - **SKAO2**: Conceptual Phase 2 survey
   - **DM_Part**: Dark matter particles (bias = 1)
   - **CV_limit**: Cosmic variance limited survey
   - **InitNew**: Method to create a new empty survey configuration
   
   Example survey class
   --------------------
   
   .. py:class:: SurveyParams.<surveyname>
   
      **Attributes:**
         
      - `b_1`: Linear bias as function of redshift.
      - `z_range`: Redshift range [z_min,z_max].
      - `be_survey`: Evolution bias as function of redshift.
      - `Q_survey`: Magnification bias as function of redshift.
      - `n_g`: Number density as function of redshift [:math:`\text{h}^{3} \text{Mpc}^{3}`].
      - `f_sky`: Sky fraction.
      
      **Optional Attributes:**
   
      - **b_2**: Second order bias.
      - **g_2**: Tidal bias.
      - **loc**, **eq**, **orth**: Primordial non-Gaussianity scale-dependent bias parameters.
 

Usage Examples
--------------

.. code-block:: python

    import cosmo_wap as cw
    from classy import Class

    # Initialize cosmology
    cosmo = cw.utils.get_cosmology()

    # Get survey parameters
    survey_params = cw.survey_params.SurveyParams(cosmo)

    # Initialize with a single survey
    cosmo_funcs_euclid = cw.ClassWAP(cosmo, survey_params.Euclid)

    # Initialize with two surveys for multi-tracer analysis
    cosmo_funcs_mt = cw.ClassWAP(cosmo, [survey_params.Euclid, survey_params.SKAO2])

    # Create a custom survey
    custom_survey = survey_params.InitNew()
    custom_survey.b_1 = lambda z: 1.2 + 0.3*z
    custom_survey.z_range = [0.5, 1.5]
    custom_survey.be_survey = lambda z: 0 + 0*z
    custom_survey.Q_survey = lambda z: 2/5 + 0*z
    custom_survey.n_g = lambda z: 0.02 * np.exp(-z)
    custom_survey.f_sky = 0.5

    # Initialize with custom survey
    cosmo_funcs_custom = cw.ClassWAP(cosmo, custom_survey)
    
Note: multitracer only implemeted currently for the power spectrum but could be extended straightforwardly for the bispectrum.
    


