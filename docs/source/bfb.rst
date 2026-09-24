Best-Fit Bias
=============

Compute systematic bias on parameters from neglecting contributions in the model.

Method
------

Best-fit bias is computed via ``get_fish`` by passing a ``bias_list`` argument specifying the neglected term(s). If the data contain a contribution :math:`\Delta D` that the model neglects, the linearised best fit shifts by

.. math::

    \delta\theta_i = (F^{-1})_{ij} B_j, \qquad B_j = \left(\frac{\partial D}{\partial \theta_j}\right)^\dagger \mathsf{C}^{-1} \Delta D .

The resulting ``FisherMat`` stores two versions, each a list with one dict per bias term (the last entry is the total when there is more than one):

- ``.bias`` - the marginalised shift :math:`F^{-1}B`, where all parameters move together (what an MCMC fit to the same data finds). It is recomputed from the stored :math:`B` (``config["B"]``) whenever a new ``FisherMat`` is made, so priors added afterwards (``add_planck_prior``, ``add_gaussian_priors``) and ``to_S8`` are accounted for. Per-bin nuisance parameters are marginalised in the same way as the Fisher matrix. This is the default centre used by ``add_chain`` and ``plot_1D``.
- ``.conditional_bias`` - :math:`B_i/F_{ii}`, the shift in :math:`\theta_i` with every other parameter held fixed (the convention of Addis et al. 2025). It is fixed at ``get_fish`` time and is not updated by priors.

.. code-block:: python

    fisher = forecast.get_fish(
        param_list,
        terms='NPP',
        pkln=[0, 2],
        bias_list='WS'   # neglected term(s)
    )

    # Access bias values (total over all neglected terms)
    fisher.bias[-1]              # marginalised
    fisher.conditional_bias[-1]  # conditional

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
        print(f"Bias on {param}: {fisher.bias[-1][param]:.4e}")

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

   Compute the conditional parameter bias from neglecting a contribution (``conditional_bias`` above).

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
