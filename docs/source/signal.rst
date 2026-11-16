Signal Model
============

CosmoWAP builds the power spectrum and bispectrum from the redshift-space kernels of each field: the full :math:`P(k,\mu)` or :math:`B(k_1,k_2,k_3,\mu,\phi)` is computed numerically and projected onto the multipoles. This is the default throughout - the signal in ``get_fish`` and the ``Sampler``, and the covariances.

The analytic multipoles exported from MathWAP are kept for the wide-separation corrections, which the kernels do not have, and as a check on the kernels. In a forecast they are summed on top of the kernels - see `In Forecasts`_. Computing either directly is with the :doc:`power spectrum <pk>` and the :doc:`bispectrum <bk>`.

Kernels
-------

Each kernel is one contribution to the observed galaxy overdensity, defined in ``numeric_mu/kernels.py``: ``K1`` holds the first-order part, ``K2`` the second-order part the bispectrum needs, and ``IntK1`` the integrated (line-of-sight) kernels. The kernels of a field are summed before the spectrum is formed, so the cross terms between them are included.

.. list-table::
   :header-rows: 1
   :widths: 15 55 15 15

   * - Kernel
     - Contribution
     - :math:`P`
     - :math:`B`
   * - ``'N'``
     - Newtonian (Kaiser), with :math:`b_2` and :math:`\gamma_2` at second order
     - yes
     - yes
   * - ``'LP'``
     - Local projection (relativistic) effects
     - yes
     - yes
   * - ``'Loc'``, ``'Eq'``, ``'Orth'``, ``'PNG'``
     - Scale-dependent PNG bias
     - yes
     - yes
   * - ``'I'``
     - All integrated effects (L + TD + ISW)
     - yes
     - not yet
   * - ``'L'``, ``'TD'``, ``'ISW'``, ``'kappa_g'``
     - The integrated effects one at a time
     - yes
     - not yet

Their definitions are with the :ref:`power spectrum <pk-kernels>` (first order) and the :ref:`bispectrum <bk-kernels>` (second order), and the line-of-sight integrals in :doc:`integrated`.

In Forecasts
------------

``get_fish``, ``get_cumulative_fish``, ``best_fit_bias`` and ``sampler`` take the signal as:

- ``kernels`` - the signal of both spectra, default ``['N']``. ``[]`` for none.
- ``bk_kernels`` - the bispectrum's, defaulting to those of ``kernels`` with a second-order part (so ``'I'`` is left out, with a log message). ``[]`` for none.
- ``terms`` - analytic contributions summed onto the kernels, mainly the wide-separation ones the kernels do not have: ``WS`` with the Newtonian kernels, and ``WSGR`` (``WS + WAGR + RRGR``) once ``'LP'`` is in, which adds their mixing with the relativistic effects. ``bk_terms`` defaults to ``terms``.

An analytic term that duplicates an active kernel - ``NPP`` with ``'N'``, ``GR1``/``GR2`` with ``'LP'``, an integrated term with an integrated kernel, ``Loc`` with ``'Loc'`` - raises an error rather than counting it twice. For an analytic-only signal pass ``kernels=[]`` (or ``bk_kernels=[]`` for the bispectrum alone).

.. code-block:: python

    from cosmo_wap.forecast import FullForecast

    forecast = FullForecast(cosmo_funcs, kmax_func=0.1, N_bins=4)

    # Kaiser + local PNG in both spectra
    fisher = forecast.get_fish(["fNL", "A_b_1"], kernels=["N", "Loc"], pkln=[0, 2], bkln=[0])

    # relativistic effects too - the bispectrum takes ['N', 'LP', 'Loc'], as 'I' has no
    # second-order kernel yet - and the shift in fNL from neglecting the wide-separation terms
    fisher = forecast.get_fish(
        ["fNL"],
        kernels=["N", "LP", "I", "Loc"],
        bias_list="WSGR",
        pkln=[0, 2],
        bkln=[0, 1, 2],
    )
    print(fisher.bias[-1]["fNL"])

    # in an MCMC model, with the analytic bispectrum (e.g. with the compiled kernels and tables -
    # see Performance)
    sampler = forecast.sampler(
        ["fNL"],
        kernels=["N", "LP", "I", "Loc"],
        terms="WSGR",
        bk_kernels=[],
        bk_terms=["NPP", "GR1", "GR2", "Loc", "WSGR"],
        pkln=[0, 2],
        bkln=[0, 1, 2],
    )

In a Fisher matrix the analytic terms only enter through the parameter derivatives, and the wide-separation terms do not depend on :math:`f_{\rm NL}`, so for :math:`\sigma(f_{\rm NL})` they barely matter - what they matter for is the systematic shift from leaving them out (``bias_list``, see :doc:`bfb`), and the model in an MCMC.

The covariance does not follow ``kernels``: it takes its own ``cov_terms`` (default ``['N']`` plus shot noise) - see :doc:`covariance`. Kernel names can also be forecast parameters, as spectrum amplitudes - see :ref:`kernel-amplitudes`.

Kernels and Analytic Terms
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Analytic term
     - Kernel
     - Agreement
   * - ``NPP``
     - ``'N'``
     - Exact
   * - ``GR1 + GR2`` (``GRL``)
     - ``'LP'``
     - Through second order in :math:`\mathcal{H}/k`; the kernels keep the higher orders
   * - ``Loc``, ``Eq``, ``Orth``
     - the same names
     - At linear order in :math:`f_{\rm NL}` (see :ref:`bk-kernels`)
   * - ``IntNPP + IntInt`` (``GRI``), ``LxNPP``, ...
     - ``'I'``, ``'L'``, ...
     - The same effects; the kernels have an endpoint LOS on the first field, and also the cross terms with ``'LP'``
   * - ``WA1``, ``WA2``, ``RR1``, ``RR2``, ``WARR``, ``WAGR``, ``RRGR`` (``WS``, ``WSGR``)
     - none
     - Analytic only

Only the kernels have: the :math:`m > 0` bispectrum multipoles in forecasts (``all_m``), FoG on the multi-tracer bispectrum (bar the wide-separation terms), the bin's radial window on the integrated terms (``radial_window``), the higher-order relativistic bispectrum terms (``full_betas``), every power of :math:`f_{\rm NL}` in the bispectrum, kernel amplitudes, and the covariances. Only the analytic terms have the wide-separation corrections and the LOS choices ``t``, ``r`` and ``s`` that go with them.
