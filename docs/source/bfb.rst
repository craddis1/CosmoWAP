Best-Fit Bias
=============

Compute systematic bias on parameters from neglecting contributions in the model.

Method
------

Best-fit bias is computed via ``get_fish`` by passing a ``bias_list`` argument specifying the neglected term(s). The resulting ``FisherMat`` object stores the bias values in its ``.bias`` attribute.

.. code-block:: python

    fisher = forecast.get_fish(
        param_list,
        terms='NPP',
        pkln=[0, 2],
        bias_list='WS'   # neglected term(s)
    )

    # Access bias values
    fisher.bias

Usage
-----

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap.lib import utils
    from cosmo_wap.forecast import FullForecast

    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31)
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)
    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Bias on fNL from neglecting wide-separation
    fisher = forecast.get_fish(
        ["fNL", "A_s", "n_s"],
        terms=["NPP", "Loc"],
        pkln=[0, 2],
        bias_list="WS"
    )

    for param in ["fNL", "A_s", "n_s"]:
        print(f"Bias on {param}: {fisher.bias[param]:.4e}")

    # Bias from neglecting GR effects
    fisher_gr = forecast.get_fish(
        ["fNL", "A_s", "n_s"],
        terms=["NPP", "Loc"],
        pkln=[0, 2],
        bias_list="GR1"
    )

Alternative: best_fit_bias
--------------------------

Bias can also be computed directly via the standalone method:

.. py:method:: FullForecast.best_fit_bias(param, bias_term, terms='NPP', pkln=None, bkln=None, verbose=True, sigma=None)

   Compute parameter bias from neglecting a contribution.

   :param param: Parameter(s) to compute bias for (string or list)
   :param bias_term: Neglected term(s) (see :ref:`available-terms`)
   :param str terms: Base terms (see :ref:`available-terms`, default: ``'NPP'``)
   :param list pkln: Pk multipoles
   :param list bkln: Bk multipoles
   :param bool verbose: Show progress
   :param float sigma: FoG damping
   :return: Tuple of (bias_dict, fisher_diagonal)

.. code-block:: python

    bias, fisher_diag = forecast.best_fit_bias(
        ["fNL", "A_s", "n_s"],
        bias_term="WS",
        terms=["NPP", "Loc"],
        pkln=[0, 2]
    )

    for param in ["fNL", "A_s", "n_s"]:
        print(f"Bias on {param}: {bias[param]:.4e}")
