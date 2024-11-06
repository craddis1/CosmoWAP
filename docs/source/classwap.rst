
ClassWAP
========

.. py:class:: ClassWAP(cosmo,survey_params,compute_bias=True)
   :module: main


    **parameters**:
    - **cosmo**: cosmology computed from classy
    - **survey_params**: either list of two or just one instance of survey params class for survey specific params
    - **compute_bias**: boolean whether to use PBS approach to compute bias or just use relations in Sec... . PBS can be used where implenteed HOD and HMF are suitable.