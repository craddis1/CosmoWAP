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
        kernels=['N'],
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
        kernels=["N", "Loc"],
        pkln=[0, 2],
        bias_list="WS"
    )

    for param in ["fNL", "A_s", "n_s"]:
        print(f"Bias on {param}: {fisher.bias[-1][param]:.4e}")

    # Bias from neglecting the local GR effects (GR1 + GR2)
    fisher_gr = forecast.get_fish(
        ["fNL", "A_s", "n_s"],
        kernels=["N", "Loc"],
        pkln=[0, 2],
        bias_list="GRL"
    )

``kernels`` is the fitted model and ``bias_list`` the contribution it leaves out of the data. The neglected contributions are the analytic terms (see :ref:`available-terms`): each is a complete contribution, its cross terms with the Newtonian signal already in, so the shift does not depend on which kernels the model has.

Alternative: best_fit_bias
--------------------------

Bias can also be computed directly via the standalone method:

.. py:method:: FullForecast.best_fit_bias(param, bias_term, kernels=None, pkln=None, bkln=None, verbose=True, sigma=None, bk_kernels=None, mu_grid=None, bk_mu_grid=None, terms=None, bk_terms=None, t=0, r=0, s=0)

   Compute the conditional parameter bias from neglecting a contribution (``conditional_bias`` above).

   :param param: Parameter(s) to compute bias for (string or list)
   :param bias_term: Neglected term(s) (see :ref:`available-terms`)
   :param list kernels: The signal, as in ``get_fish`` (default: ``['N']``) - with ``bk_kernels``, ``mu_grid``, ``bk_mu_grid`` and the analytic ``terms``/``bk_terms`` (see :doc:`signal`)
   :param list pkln: Pk multipoles
   :param list bkln: Bk multipoles
   :param bool verbose: Show progress
   :param float sigma: FoG damping
   :return: Tuple of (bias_dict, fisher_diagonal)

.. code-block:: python

    bias, fisher_diag = forecast.best_fit_bias(
        ["fNL", "A_s", "n_s"],
        bias_term="WS",
        kernels=["N", "Loc"],
        pkln=[0, 2]
    )

    for param in ["fNL", "A_s", "n_s"]:
        print(f"Bias on {param}: {bias[param]:.4e}")
