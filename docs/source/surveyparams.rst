Survey parameters
==================

CosmoWAP takes in information about the survey (e.g. bias parameters, number density) from the SurveyParams class.

SurveyParams
------------

.. py:class:: SurveyParams(cosmo=None)

   Initialize and retrieve set survey parameters for different surveys.

   **Attributes:**
   
   Preset surveys configs - see Sec link!!! for details.

   - :py:class:`SurveyParams.Euclid`
   - :py:class:`SurveyParams.BGS`
   - :py:class:`SurveyParams.SKAO1`
   - :py:class:`SurveyParams.SKAO2`
   - :py:class:`SurveyParams.DM_part`
   
   Also initiate new survey 
   
   - :py:class:`SurveyParams.init_new`
   
   Example survey
   -------------- 
   
   .. py:class:: SurveyParams.<surveyname>
   
      **Attributes:**
         
      - `b_1`
      - `z_range`
      - `be_survey`
      - `Q_survey`
      - `n_g`
      - `f_sky`
      
      Optional:
      
      - `b_2`
      - `g_2`

Usage Example
-------------

.. code-block:: bash

    import cosmo_wap as cw
    
    survey_params = cw.survey_params.SurveyParams(cosmo)
    
SurveyParams contains some preset survey descriptions
