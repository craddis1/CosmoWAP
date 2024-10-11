Survey parameters
==================

CosmoWAP takes in information about the survey (e.g. bias parameters, number density) from the SurveyParams class.

SurveyParams
------------

.. py:class:: SurveyParams(cosmo=None)

   Initialize and retrieve survey parameters for different surveys. Supports optional cosmological input.

   **Attributes:**

   - :py:class:`SurveyParams.Euclid`
   - :py:class:`SurveyParams.BGS`
   - :py:class:`SurveyParams.SKAO1`
   - :py:class:`SurveyParams.SKAO2`
   - :py:class:`SurveyParams.DM_part`
   - :py:class:`SurveyParams.init_new`


Usage Example
-------------

.. code-block:: bash

    import cosmo_wap as cw
    
    survey_params = cw.survey_params.SurveyParams(cosmo)
    
SurveyParams contains some preset survey descriptions
