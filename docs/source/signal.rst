Signal Model
============

CosmoWAP builds the power spectrum and bispectrum from the redshift-space kernels of each field: the full :math:`P(k,\mu)` or :math:`B(k_1,k_2,k_3,\mu,\phi)` is computed numerically and projected onto the multipoles. This is the default throughout - the signal in ``get_fish`` and the ``Sampler``, and the covariances.

The analytic multipoles exported from MathWAP (see :doc:`pk` and :doc:`bk`) are kept for the wide-separation corrections, which the kernels do not have, and as a check on the kernels. In a forecast they are summed on top of the kernels - see `In Forecasts`_.

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

**Standard (evaluated at source):**

- ``'N'`` - Newtonian (Kaiser RSD): :math:`D(z)[b_1 + f\mu^2]`
- ``'LP'`` - Local projection effects (relativistic): :math:`D(z)[i\mu\,\beta_1/k + \beta_2/k^2]`
- ``'Loc'``, ``'Eq'``, ``'Orth'`` - PNG scale-dependent bias for each shape: :math:`D(z) f_{\rm NL} k^{\alpha} b_{01}/M(k)` with :math:`\alpha = 0, 2, 2` (``'Orth'`` is the orthogonal-LSS template - see the note in :doc:`bk`). Named as the analytic classes in ``pk/PNG.py``, and read the same ``fNL`` (or per-shape ``fNL_loc``/``fNL_eq``/``fNL_orth``) keyword. Listing more than one shape sums them into the single kernel, so the square retains their cross term. ``'Eq'``/``'Orth'`` need ``compute_bias=True`` on ``ClassWAP``.
- ``'PNG'`` - all three shapes in one kernel. Identical to listing ``'Loc'``, ``'Eq'``, ``'Orth'``, but :math:`M(k)` is evaluated once rather than per shape.

**Integrated (line-of-sight):**

- ``'I'`` - All integrated effects combined (L + TD + ISW)
- ``'L'`` - Lensing magnification, :math:`2(Q-1)\kappa`
- ``'TD'`` - Time delay
- ``'ISW'`` - Integrated Sachs-Wolfe
- ``'kappa_g'`` - Bare lensing convergence :math:`\kappa` (without the magnification-bias prefactor)

How the line-of-sight integrals are done is in :doc:`integrated`.

Bispectrum kernels
~~~~~~~~~~~~~~~~~~

The bispectrum needs each kernel's second-order part, in ``numeric_mu.kernels.K2``:

- ``'N'`` - Newtonian (Kaiser), including :math:`b_2` and :math:`\gamma_2`
- ``'LP'`` - local projection (relativistic) effects - agrees with ``GR1 + GR2`` through second order in :math:`\mathcal{H}/k`
- ``'Loc'``, ``'Eq'``, ``'Orth'`` or all three as ``'PNG'`` - the scale-dependent PNG bias

The PNG kernels are part of each field's :math:`Z_1` and :math:`Z_2` throughout, so the numeric :math:`B` keeps every power of :math:`f_{\rm NL}` at tree level (up to :math:`f_{\rm NL}^4`): products of the scale-dependent bias on several legs, cross terms between shapes, the scale-dependent bias on the legs of the primordial bispectrum, and the :math:`b_{02}\,\varphi_p^2` term of arXiv:2011.13660 eq (2.17). :math:`b_{02}` is included for ``'Loc'`` only - for ``'Eq'`` and ``'Orth'`` the response to :math:`\varphi_p^2` depends on the primordial trispectrum, which their templates do not fix. The analytic ``Loc``/``Eq``/``Orth`` terms are linear in :math:`f_{\rm NL}` (``bk_mt.Loc`` keeps the tree to :math:`f_{\rm NL}^3`, but without :math:`b_{02}` or the bias on the primordial legs), so they agree with the kernels at linear order only. Fisher derivatives at a fiducial :math:`f_{\rm NL}=0` are unchanged, as the central stencil cancels the even powers.

By default ``'LP'`` leaves out the :math:`(\mathcal{H}/k)^3` and :math:`(\mathcal{H}/k)^4` terms of the second-order kernel. Setting ``cosmo_funcs.full_betas = True`` adds them: :math:`\beta_1`-:math:`\beta_5` of arXiv:2011.13660 eq (3.24), and the coupling of the scale-dependent PNG bias to first-order projection effects (:math:`\Upsilon_3`-:math:`\Upsilon_5` of eq (3.26)), which enters a field only if it has both ``'LP'`` and a PNG kernel. :math:`\Upsilon_1` and :math:`\Upsilon_2` are already included through the relativistic :math:`Z_1` on the primordial bispectrum. The paper derives the :math:`\Upsilon`'s for local PNG only; ``'Eq'`` and ``'Orth'`` use the same couplings with their own :math:`b_{01}(z)`. The analytic ``GR1``/``GR2`` terms ignore the flag.

``'I'`` (integrated effects) has no second-order kernel yet.

Computing Spectra
-----------------

``pk.pk_func`` and ``bk.bk_func`` take the kernels as ``kernels``, summed onto an analytic ``term`` (``None`` for the kernels alone). With a list of multipoles the full :math:`P(k,\mu)` or :math:`B(\mu,\phi)` is computed once and projected onto each:

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap import bk, pk
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    cosmo_funcs = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo))
    k = np.linspace(0.01, 0.2, 50)
    z = 1.0

    # P_0, P_2, P_4 - Kaiser plus the local and integrated relativistic effects
    P_l = pk.pk_func(None, [0, 2, 4], cosmo_funcs, k, zz=z, kernels=['N', 'LP', 'I'])

    # B_l for l = 0..3 (and optionally (l, m) pairs) from one B(mu, phi) evaluation -
    # on a scalene triangle, as the odd multipoles vanish on equilateral ones
    B_l = bk.bk_func(None, [0, 1, 2, 3], cosmo_funcs, 0.1, 0.08, 0.05, zz=z,
                     kernels=['N', 'LP'], mu_grid=[8, 8])

    # add the analytic wide-separation corrections on top - WSGR as 'LP' is in (see below)
    P_l_WS = pk.pk_func('WSGR', [0, 2], cosmo_funcs, k, zz=z, kernels=['N', 'LP', 'I'])

``mu_grid`` sets the angular grid:

- **Power spectrum**: ``[n_mu, GL, los_n, deg]`` with an optional 5th entry ``n_p``, default ``[48, True, 8, 8]`` and ``n_p=2000`` - see ``numeric_mu.pk.get_multipoles`` in :doc:`integrated`.
- **Bispectrum**: ``[n_mu, n_phi]``, default ``[16, 16]``. Without FoG, :math:`B(\mu,\phi)` from local kernels is a polynomial of degree 8 in the LOS direction, so ``[8, 8]`` is exact through :math:`\ell = 4`.

``numeric_mu.pk.get_multipoles`` and ``numeric_mu.bk.get_multipoles`` are the lower-level entry points, with a separate kernel list for each field - e.g. ``['I']`` on one field and ``['N']`` on the other for the integrated x Newtonian cross term alone (see :doc:`integrated`).

FoG damping is applied inside the angular projection (``sigma``, per galaxy field - see :ref:`forecast-fog`).

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
     - At linear order in :math:`f_{\rm NL}` (see `Bispectrum kernels`_)
   * - ``IntNPP + IntInt`` (``GRI``), ``LxNPP``, ...
     - ``'I'``, ``'L'``, ...
     - The same effects; the kernels have an endpoint LOS on the first field, and also the cross terms with ``'LP'``
   * - ``WA1``, ``WA2``, ``RR1``, ``RR2``, ``WARR``, ``WAGR``, ``RRGR`` (``WS``, ``WSGR``)
     - none
     - Analytic only

Only the kernels have: the :math:`m > 0` bispectrum multipoles (``all_m``), FoG on the multi-tracer bispectrum, the bin's radial window on the integrated terms (``radial_window``), the higher-order relativistic bispectrum terms (``full_betas``), every power of :math:`f_{\rm NL}` in the bispectrum, kernel amplitudes, and the covariances. Only the analytic terms have the wide-separation corrections and the LOS choices ``t``, ``r`` and ``s`` that go with them.
