Survey parameters
==================

CosmoWAP takes in information about the survey (e.g. bias parameters, number density) from the SurveyParams class.

SurveyParams
------------

.. py:class:: SurveyParams(cosmo=None)

   Initialize and retrieve set survey parameters for different surveys. These survey are stored in a class in the SurveyParams class.

   **Attributes:**
   
   Preset surveys configs - see Sec link!!! for details.

   - :py:class:`SurveyParams.Euclid`
   - :py:class:`SurveyParams.BGS`
   - :py:class:`SurveyParams.SKAO1`
   - :py:class:`SurveyParams.SKAO2`
   - :py:class:`SurveyParams.DM_part`
   
   Also initiate new survey 
   
   - :py:class:`SurveyParams.init_new`
   
   Example survey class
   --------------------
   
   .. py:class:: SurveyParams.<surveyname>
   
      **Attributes:**
         
      - `b_1`: Linear bias as function of redshift.
      - `z_range`: Redshift range [z_min,z_max].
      - `be_survey`: Evolution bias as function of redshift.
      - `Q_survey`: Magnification bias as function of redshift.
      - `n_g`: Number density as function of redshift [units].
      - `f_sky`: Sky fraction.
      
      Optional:
      
      - `b_2`: Second order bias.
      - `g_2`: Tidal bias.

Usage Example
-------------

.. code-block:: bash

    import cosmo_wap as cw
    
    survey_params = cw.survey_params.SurveyParams(cosmo)
    
SurveyParams contains some preset survey descriptions
