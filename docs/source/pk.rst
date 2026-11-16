Power Spectrum
==============

CosmoWAP computes galaxy power spectrum multipoles in redshift space, including relativistic, integrated, PNG and wide-separation contributions. The signal is built numerically from the kernels of each field - the default, see `Kernel Path`_ - while the analytic multipoles exported from MathWAP (`Analytic Terms`_) supply the wide-separation corrections, which the kernels do not have. How the two combine in forecasts is in :doc:`signal`.

Kernel Path
-----------

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

.. _pk-kernels:

Kernels
~~~~~~~

Each kernel is one first-order contribution to the observed galaxy overdensity, in ``numeric_mu.kernels.K1`` (standard) and ``IntK1`` (integrated). The bispectrum needs their second-order parts too - see :ref:`bk-kernels`.

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

The power spectrum for any pair of kernels is :math:`P_\ell(k) = \frac{2\ell+1}{2}\int_{-1}^{1} K_1 K_2^* P(k)\, \mathcal{L}_\ell(\mu)\, d\mu`, where the line-of-sight integrals for integrated kernels are handled internally (see :doc:`integrated`).

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

The ``pk`` module also holds the analytic power spectrum multipoles, exported from MathWAP. They are what you need for the wide-separation corrections, which the kernels do not have, and are kept as a check on the kernels. In forecasts they are passed as ``terms``, summed onto the kernels.

.. _pk-terms:

**Contribution types:**

- **Wide-separation**: analytic only

  - ``WA1``, ``WA2``: Wide-angle corrections
  - ``RR1``, ``RR2``: Radial-redshift (evolution) corrections
  - ``WARR``: Mixed wide-angle and radial-redshift corrections
  - ``WS``: Wrapper for combined wide-separation (WA + RR)
  - ``WAGR``, ``RRGR``: Mixed wide-angle/GR and radial-redshift/GR corrections
  - ``WS_GR``: ``WAGR + RRGR``
  - ``WSGR``: ``WS + WAGR + RRGR``
  - ``Full``: ``WSGR + GR1 + GR2`` - add ``NPP``, PNG and integrated contributions separately

  These are generated by ``mathwap`` (the sympy port of MathWAP, in ``mathwap/`` in the MathWAP repository): two tracers, any ``t``, multipoles up to :math:`\ell = 6`, and with ``sigma`` the FoG-damped ones, through moments that stay accurate as :math:`k\sigma \to 0`.

  The composites (and ``GRL``) go to :math:`\ell = 6` too. Each multipole sums the parts of its parity. Where a part has no closed form for that multipole, it is projected from the part's ``mu`` polynomial instead, so with ``sigma`` ``Full.l4`` includes the FoG-damped ``GR2`` (``GR2.l4`` on its own is 0).

- **Local contributions** (evaluated at the galaxy position) - also computed by the kernels in brackets

  - ``NPP``: Newtonian plane-parallel (Kaiser RSD) (``'N'``)
  - ``GR1``, ``GR2``: Local relativistic (Doppler, gravitational redshift) (``'LP'``)
  - ``Loc``, ``Eq``, ``Orth``: PNG scale-dependent bias (local, equilateral, orthogonal) (``'Loc'``, ``'Eq'``, ``'Orth'``)

.. note::

   Equilateral and orthogonal PNG biases require ``compute_bias=True`` when initialising ``ClassWAP``. Local PNG also works with the default survey bias prescription; ``compute_bias=True`` derives its biases from the HOD/HMF instead.

- **Integrated contributions**: Line-of-sight integrals from observer to source (``'I'``, ``'L'``, ``'TD'``, ``'ISW'``)

  - ``IntNPP``, ``IntInt``: Integrated effects (lensing convergence, time delay, integrated Sachs-Wolfe) crossed with NPP and with themselves - see :doc:`integrated`
  - ``LxNPP``, ``TDxNPP``, ``ISWxNPP``: Individual lensing, time-delay and ISW contributions crossed with NPP
  - ``LxL``, ``LxTD``, ``LxISW``, ``TDxTD``, ``TDxISW``, ``ISWxISW``: Individual integrated-integrated contributions

- **Relativistic composites**:

  - ``GRL``: ``GR1 + GR2`` (local)
  - ``GRI``: ``IntNPP + IntInt`` (integrated)
  - ``GR``: ``GRL + GRI``
  - ``ISW``: ``ISWxNPP + ISWxISW + LxISW + TDxISW``

These are analytic contribution names for ``pk.pk_func`` and forecast ``terms``. Field-kernel names for ``kernels`` and the default ``cov_terms`` follow a different convention - see `Kernels`_ and :doc:`covariance`. How closely each agrees with its kernel is in :doc:`signal`.

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
