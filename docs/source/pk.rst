Power Spectrum Module
=====================

The ``pk`` module computes galaxy power spectrum multipoles in redshift space, including wide-separation, relativistic, and integrated contributions.

.. _pk-terms:

**Contribution types:**

- **Local contributions**: Evaluated at the galaxy position

  - ``NPP``: Newtonian plane-parallel (Kaiser RSD)
  - ``WA1``, ``WA2``: Wide-angle corrections
  - ``RR1``, ``RR2``: Radial-redshift (evolution) corrections
  - ``WARR``: Mixed wide-angle and radial-redshift corrections
  - ``WS``: Wrapper for combined wide-separation (WA + RR)
  - ``GR1``, ``GR2``: Local relativistic (Doppler, gravitational redshift)
  - ``WAGR``, ``RRGR``: Mixed wide-angle/GR and radial-redshift/GR corrections
  - ``WS_GR``: ``WAGR + RRGR``
  - ``WSGR``: ``WS + WAGR + RRGR``
  - ``Full``: ``WSGR + GR1 + GR2`` - add ``NPP``, PNG and integrated contributions separately
  - ``Loc``, ``Eq``, ``Orth``: PNG scale-dependent bias (local, equilateral, orthogonal)

.. note::

   Equilateral and orthogonal PNG biases require ``compute_bias=True`` when initialising ``ClassWAP``. Local PNG also works with the default survey bias prescription; ``compute_bias=True`` derives its biases from the HOD/HMF instead.

- **Integrated contributions**: Line-of-sight integrals from observer to source

  - ``IntNPP``, ``IntInt``: Integrated effects (lensing convergence, time delay, integrated Sachs-Wolfe) crossed with NPP and with themselves - see :doc:`integrated`
  - ``LxNPP``, ``TDxNPP``, ``ISWxNPP``: Individual lensing, time-delay and ISW contributions crossed with NPP
  - ``LxL``, ``LxTD``, ``LxISW``, ``TDxTD``, ``TDxISW``, ``ISWxISW``: Individual integrated-integrated contributions

- **Relativistic composites**:

  - ``GRL``: ``GR1 + GR2`` (local)
  - ``GRI``: ``IntNPP + IntInt`` (integrated)
  - ``GR``: ``GRL + GRI``
  - ``ISW``: ``ISWxNPP + ISWxISW + LxISW + TDxISW``

These are analytic contribution names for ``pk.pk_func`` and forecast ``terms``. Field-kernel names for ``kernels`` and the default ``cov_terms`` follow a different convention - see :doc:`integrated` and :doc:`covariance`.

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

    # Newtonian multipoles (Kaiser)
    P0 = pk.NPP.l0(cosmo_funcs, k, zz=z)
    P2 = pk.NPP.l2(cosmo_funcs, k, zz=z)
    P4 = pk.NPP.l4(cosmo_funcs, k, zz=z)

    # Wide-separation corrections
    P0_WS = pk.WS.l0(cosmo_funcs, k, zz=z)

    # Relativistic corrections
    P0_GR = pk.GR2.l0(cosmo_funcs, k, zz=z)

Numerical :math:`\mu` Pipeline
-------------------------------

Newtonian, local relativistic, PNG and integrated power spectrum contributions can also be computed via ``cosmo_wap.numeric_mu.pk``, which constructs :math:`P(k,\mu)` from first-order field kernels and projects onto Legendre multipoles. The analytic wide-angle and radial-redshift expansion is separate and is not supplied by these field kernels. See :doc:`integrated` for full details, available kernels, and usage examples.
