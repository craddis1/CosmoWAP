Analytic Power Spectrum
=======================

The ``pk`` module holds the analytic power spectrum multipoles in redshift space, exported from MathWAP. The signal is normally built from the numerical kernels instead (see :doc:`signal`) - the analytic terms are what you need for the wide-separation corrections, which the kernels do not have, and are kept as a check on the kernels. In forecasts they are passed as ``terms``, summed onto the kernels.

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

These are analytic contribution names for ``pk.pk_func`` and forecast ``terms``. Field-kernel names for ``kernels`` and the default ``cov_terms`` follow a different convention - see :doc:`signal` and :doc:`covariance`. A term that duplicates an active kernel (e.g. ``NPP`` with ``'N'``) raises an error in a forecast, and how closely each agrees with its kernel is in :doc:`signal`.

Higher-order products of the local relativistic kernels are available separately through ``pk.GRX.mu`` and ``pk.GRX.l(l, ...)``. ``GRX`` has no ``l0``/``l1`` dispatch methods, so it is not a ``terms`` entry for the standard forecast interface.

Methods
-------

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
-----

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

    # the signal from the kernels - Kaiser and the local relativistic effects (see Signal model)
    P0, P2 = pk.pk_func(None, [0, 2], cosmo_funcs, k, zz=z, kernels=['N', 'LP'])

    # Wide-separation corrections - analytic only. WSGR = WS + WAGR + RRGR, so with their
    # mixing with the relativistic effects, as 'LP' is in the kernels
    P0_WS = pk.WSGR.l0(cosmo_funcs, k, zz=z)
    P2_WS = pk.WSGR.l2(cosmo_funcs, k, zz=z)

    # or summed onto the kernels in one call
    P0_tot, P2_tot = pk.pk_func('WSGR', [0, 2], cosmo_funcs, k, zz=z, kernels=['N', 'LP'])

    # The analytic Newtonian multipoles, e.g. to check the kernels against
    P0_NPP = pk.NPP.l0(cosmo_funcs, k, zz=z)
