Power Spectrum
==============

CosmoWAP computes galaxy power spectrum multipoles in redshift space, including relativistic, integrated, PNG and wide-separation contributions. The signal is built numerically from the kernels of each field - the default - while the analytic multipoles exported from MathWAP (`Analytic Terms`_) supply the wide-separation corrections, which the kernels do not have. What goes into a signal is in :doc:`signal`; the simplest way to compute one is :doc:`model`.

.. toctree::
   :hidden:

   Integrated effects <integrated>

Multipoles
----------

The multipoles are Legendre moments in :math:`\mu = \hat{\mathbf{k}}\cdot\hat{\mathbf{d}}`, the cosine to the line of sight :math:`\mathbf{d} = (1-t)\,\mathbf{x}_1 + t\,\mathbf{x}_2` (``t = 0`` by default - see :doc:`signal`):

.. math::

   P_\ell(k) = \frac{2\ell+1}{2}\int_{-1}^{1} P(k,\mu)\, \mathcal{L}_\ell(\mu)\, d\mu.

For a pair of kernels :math:`P(k,\mu) = K_1 K_2^* P(k)`, where the line-of-sight integrals for integrated kernels are handled internally (see :doc:`integrated`). The multipoles are in :math:`(\mathrm{Mpc}/h)^3` and complex, with the odd ones imaginary. Field 1 sits at :math:`\mathbf{x}_1`, so swapping the fields flips the sign of the odd multipoles.

Computing them
--------------

:doc:`model` is the simplest way: it holds the kernels of each field and the wide-separation terms, and gives :math:`P_\ell`, :math:`P(k,\mu)`, :math:`\xi_\ell` and the multipoles through a survey window. Below it are ``pk_func`` - what the forecasts call - and the ``numeric_mu.pk`` functions.

pk_func
~~~~~~~

``pk.pk_func`` builds the full :math:`P(k,\mu)` from the kernels of each field (defined in ``numeric_mu/kernels.py``) and projects it onto the Legendre multipoles - with a list of multipoles it is computed once and projected onto each. The kernels of a field are summed before the spectrum is formed, so the cross terms between them are included. They are summed onto an analytic ``term`` (``None`` for the kernels alone):

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    import cosmo_wap.pk as pk
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    k = np.linspace(0.01, 0.3, 100)
    z = 1.0

    # P_0, P_2, P_4 - Kaiser
    P_l = pk.pk_func(None, [0, 2, 4], cosmo_funcs, k, zz=z, kernels=['N'])

    # with the local and integrated relativistic effects
    P_l_GR = pk.pk_func(None, [0, 2, 4], cosmo_funcs, k, zz=z, kernels=['N', 'LP', 'I'])

``mu_grid=[n_mu, GL, los_n, deg]``, with an optional 5th entry ``n_p``, sets the grid (default ``[48, True, 8, 8]`` and ``n_p=2000`` - see ``get_multipole`` below). FoG damping is applied before the projection (``sigma``, per galaxy field - see :ref:`forecast-fog`).

Lower-level functions
~~~~~~~~~~~~~~~~~~~~~

These take a separate kernel list for each field, e.g. ``['I']`` on one and ``['N']`` on the other for the integrated x Newtonian cross term alone (see :doc:`integrated`).

.. function:: numeric_mu.pk.get_multipole(kernel1, kernel2, l, cosmo_funcs, kk, zz, sigma=None, n=8, n_mu=48, deg=8, delta=0.1, GL=True)

   Compute the l-th multipole of the power spectrum for a given pair of kernels.

   :param list kernel1: Kernel(s) for field 1 (e.g. ``['N']``, ``['N', 'LP']``, ``['I']``)
   :param list kernel2: Kernel(s) for field 2
   :param int l: Multipole order
   :param cosmo_funcs: ``ClassWAP`` instance
   :param kk: Array-like wavevectors [h/Mpc]
   :param float zz: Redshift
   :param float sigma: FoG damping
   :param int n: Number of Gauss-Legendre nodes for the line-of-sight integral
   :param int n_mu: Number of :math:`\mu` integration points (split evenly across the four Gauss-Legendre panels when ``GL=True``)
   :param int deg: Polynomial degree for the radial fit of the integrated kernels
   :param float delta: Half-width of the central region: the panel split when ``GL=True``, the fine/coarse boundary of the non-uniform grid when ``GL=False``
   :param bool GL: Composite Gauss-Legendre over the panels :math:`[-1,-\delta]`, :math:`[-\delta,0]`, :math:`[0,\delta]`, :math:`[\delta,1]` (default). ``False`` uses a non-uniform grid with the trapezoid rule, which needs ``n_mu`` of a few hundred to match it
   :return: Power spectrum multipole [(Mpc/h)^3]

.. function:: numeric_mu.pk.get_multipoles(kernel1, kernel2, ln, cosmo_funcs, kk, zz, sigma=None, n=8, n_mu=48, deg=8, delta=0.1, GL=True)

   Like ``get_multipole`` but for a list of multipoles ``ln`` (e.g. ``[0, 2]``): the full :math:`P(k,\mu)` is computed once and projected onto each :math:`\ell`, so this is much cheaper than separate ``get_multipole`` calls.

   :param list ln: Multipole orders
   :return: Array of shape ``(len(ln), len(kk))``

.. function:: numeric_mu.pk.get_mu(mu, kernels1, kernels2, cosmo_funcs, kk, zz, n=8, deg=8, n_p=2000, **kwargs)

   Build the raw angle-dependent :math:`P(k,\mu)` for a given pair of kernel lists. This is the building block used by ``get_multipole``/``get_multipoles`` and by the covariance classes (see :doc:`covariance`).

   :param mu: Array-like :math:`\mu` values
   :return: Complex array broadcast over ``kk`` and ``mu``

.. function:: numeric_mu.pk.get_mu_sym(mu, kernels1, kernels2, cosmo_funcs, kk, zz, **kwargs)

   As ``get_mu`` but exploiting :math:`P(k,-\mu) = P(k,\mu)^*` (correlations are real in configuration space): for a symmetric 1D ``mu`` grid only :math:`\mu \geq 0` is computed and the negative half is mirrored, roughly halving the cost. Falls back to a full ``get_mu`` call for asymmetric grids. Used by ``get_multipole``/``get_multipoles`` and the covariance classes.

Analytic Terms
--------------

The ``pk`` module also holds the analytic power spectrum multipoles (see :doc:`signal`). In forecasts they are passed as ``terms``, summed onto the kernels. Every class takes ``sigma`` for FoG; "Kernel" is the kernel that computes the same contribution, which a forecast will not let you add twice.

.. _pk-terms:

**Wide-separation** - analytic only:

.. list-table::
   :header-rows: 1
   :widths: 22 48 12 18

   * - Term
     - Contribution
     - Kernel
     - :math:`\ell`
   * - ``WA1``, ``WA2``
     - Wide-angle corrections, first and second order
     - none
     - 1, 3, 5 / 0, 2, 4, 6
   * - ``RR1``, ``RR2``
     - Radial-redshift (evolution) corrections, first and second order
     - none
     - 1, 3, 5 / 0, 2, 4, 6
   * - ``WARR``
     - Mixed wide-angle and radial-redshift corrections
     - none
     - 0, 2, 4, 6
   * - ``WAGR``, ``RRGR``
     - Mixed wide-angle/GR and radial-redshift/GR corrections
     - none
     - 0, 2, 4, 6

**Local** - evaluated at the galaxy position; the multipoles in brackets are zero without FoG:

.. list-table::
   :header-rows: 1
   :widths: 22 48 12 18

   * - Term
     - Contribution
     - Kernel
     - :math:`\ell`
   * - ``NPP``
     - Newtonian plane-parallel (Kaiser RSD)
     - ``'N'``
     - 0, 2, 4, (6)
   * - ``GR1``, ``GR2``
     - Local relativistic (Doppler, gravitational redshift), first and second order in :math:`\mathcal{H}/k`
     - ``'LP'``
     - 1, 3, (5) / 0, 2, (4, 6)
   * - ``Loc``, ``Eq``, ``Orth``
     - PNG scale-dependent bias (local, equilateral, orthogonal)
     - the same names
     - 0, 2, (4, 6)

**Integrated** - line-of-sight integrals from observer to source (see :doc:`integrated`):

.. list-table::
   :header-rows: 1
   :widths: 22 48 12 18

   * - Term
     - Contribution
     - Kernel
     - :math:`\ell`
   * - ``IntNPP``, ``IntInt``
     - Integrated effects (lensing convergence, time delay, integrated Sachs-Wolfe) crossed with NPP and with themselves
     - ``'I'``
     - 0 - 4
   * - ``LxNPP``, ``TDxNPP``, ``ISWxNPP``
     - Individual lensing, time-delay and ISW contributions crossed with NPP
     - ``'L'``, ``'TD'``, ``'ISW'``
     - 0 - 4
   * - ``LxL``, ``LxTD``, ``LxISW``, ``TDxTD``, ``TDxISW``, ``ISWxISW``
     - Individual integrated-integrated contributions
     - the same pairs
     - 0 - 4

**Composites** - see :doc:`signal` for how they are summed:

.. list-table::
   :header-rows: 1
   :widths: 22 48 12 18

   * - Term
     - Sum of
     - Kernel
     - :math:`\ell`
   * - ``WS``
     - ``WA1 + RR1 + WA2 + RR2 + WARR`` - combined wide-separation (WA + RR)
     - none
     - 0 - 6
   * - ``WS_GR``
     - ``WAGR + RRGR``
     - none
     - 0, 2, 4, 6
   * - ``WSGR``
     - ``WS + WAGR + RRGR``
     - none
     - 0 - 6
   * - ``Full``
     - ``WSGR + GR1 + GR2`` - add ``NPP``, PNG and integrated contributions separately
     - ``'LP'``
     - 0 - 6
   * - ``GRL``
     - ``GR1 + GR2`` (local)
     - ``'LP'``
     - 0 - 6
   * - ``GRI``
     - ``IntNPP + IntInt`` (integrated)
     - ``'I'``
     - 0 - 4
   * - ``GR``
     - ``GRL + GRI``
     - ``'LP'``, ``'I'``
     - 0 - 4
   * - ``ISW``
     - ``ISWxNPP + ISWxISW + LxISW + TDxISW``
     - ``'I'``
     - 0 - 4

The wide-separation terms are generated by ``mathwap``: two tracers, any ``t``, multipoles up to :math:`\ell = 6`, and with ``sigma`` the FoG-damped ones, through moments that stay accurate as :math:`k\sigma \to 0`. The local and PNG terms' multipoles come from ``pk/Pk_PP.py``, generated the same way (the PNG classes' ``mu`` and ``l`` too), so they also go to :math:`\ell = 6` and stay accurate as :math:`k\sigma \to 0`. So the composites (and ``GRL``) have every multipole to :math:`\ell = 6`, and with ``sigma`` ``Full.l4`` includes the FoG-damped ``GR2.l4``. A part's multipole that is still missing would be projected from its ``mu``. The integrated terms' FoG forms are erf closed forms, which lose digits as :math:`k\sigma \to 0`.

These are analytic contribution names for ``pk.pk_func`` and forecast ``terms``. Field-kernel names for ``kernels`` and the default ``cov_terms`` follow a different convention - see :ref:`signal-kernels` and :doc:`covariance`. How closely each agrees with its kernel is in :doc:`signal`.

Higher-order products of the local relativistic kernels are available separately through ``pk.GRX.mu`` and ``pk.GRX.l(l, ...)``. ``GRX`` has no ``l0``/``l1`` dispatch methods, so it is not a ``terms`` entry for the standard forecast interface.

Methods
~~~~~~~

Each class provides multipole methods:

.. method:: lx(cosmo_funcs, k1, zz=0, t=0, sigma=None)

   Compute x-th multipole (``l0``, ``l2``, ``l4``, etc.).

   :param object cosmo_funcs: ``ClassWAP`` instance
   :param k1: Array-like wavevector [h/Mpc]
   :param float zz: Redshift
   :param float t: LoS parameter t ∈ [0,1] defining endpoint LOS choice
   :param float sigma: FoG damping
   :return: Power spectrum multipole [(Mpc/h)^3]

.. method:: l(l, cosmo_funcs, k1, zz=0, t=0, sigma=None, n_mu=16)

   Any multipole by numerical :math:`\mu` integration, for the classes that have it - not the composites or the integrated-integrated components.

Usage
~~~~~

.. code-block:: python

    # Wide-separation corrections - analytic only. WSGR = WS + WAGR + RRGR, so with their
    # mixing with the relativistic effects, for kernels with 'LP'
    P0_WS = pk.WSGR.l0(cosmo_funcs, k, zz=z)
    P2_WS = pk.WSGR.l2(cosmo_funcs, k, zz=z)

    # or summed onto the kernels in one call
    P0_tot, P2_tot = pk.pk_func('WSGR', [0, 2], cosmo_funcs, k, zz=z, kernels=['N', 'LP'])

    # The analytic Newtonian multipoles, e.g. to check the kernels against
    P0_NPP = pk.NPP.l0(cosmo_funcs, k, zz=z)

Angle-dependent forms
---------------------

.. method:: mu(mu, cosmo_funcs, k1, zz=0, t=0)

   The full :math:`P(k,\mu)` of a term. The wide-separation, ``NPP``, ``GR1``/``GR2`` and PNG classes have it as a polynomial in :math:`\mu`, without FoG (the PNG ones take ``fNL``); ``IntNPP``, ``IntInt`` and the integrated x NPP components also take ``sigma`` (see :doc:`integrated`). ``Model`` sums the wide-separation ones with the kernels, and the composites project them where a part has no closed form.

Multi-tracer
------------

With two tracers, ``ClassWAP(cosmo, [A, B])``, field 1 is ``survey[0]`` at :math:`\mathbf{x}_1` and field 2 ``survey[1]`` at :math:`\mathbf{x}_2` - in the kernels, the wide-separation terms and ``Model`` alike. Swapping the tracers is the same pair seen from the other end: ``t`` goes to ``1 - t``, and the odd multipoles change sign. A bright/faint split of one sample (``BF_split``, see :doc:`surveyparams`) gives the two tracers from one survey.

Covariance
----------

The covariances of the multipoles - Gaussian, with or without FoG, multi-tracer, and the non-Gaussian terms - are in :doc:`covariance`, and the super-sample covariance in :doc:`forecast`. The analytic Newtonian Gaussian covariance is ``pk.COV`` (see :doc:`pkmath`).
