Power Spectrum Module
=====================

The ``pk`` module computes galaxy power spectrum multipoles in redshift space, including wide-separation, relativistic, and integrated contributions.

**Contribution types:**

- **Local contributions**: Evaluated at the galaxy position

  - ``NPP``: Newtonian plane-parallel (Kaiser RSD)
  - ``WA1``, ``WA2``: Wide-angle corrections
  - ``RR1``, ``RR2``: Radial-redshift (evolution) corrections
  - ``WS``: Wrapper for combined wide-separation (WA + RR)
  - ``GR1``, ``GR2``: Local relativistic (Doppler, gravitational redshift)
  - ``Loc``, ``Eq``, ``Orth``: PNG scale-dependent bias (local, equilateral, orthogonal)

.. note::

   PNG contributions (``Loc``, ``Eq``, ``Orth``) require ``compute_bias=True`` when initialising ``ClassWAP``.

- **Integrated contributions**: Line-of-sight integrals from observer to source

  - ``Int``: Lensing convergence, time delay, integrated Sachs-Wolfe (ISW)

Methods
-------

Each class provides multipole methods:

.. method:: lx(cosmo_functions, k1, zz=0, t=0, sigma=None)

   Compute x-th multipole (``l0``, ``l2``, ``l4``, etc.).

   :param object cosmo_functions: ``ClassWAP`` instance
   :param array-like k1: Wavevector [h/Mpc]
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

All power spectrum contributions (including standard terms like NPP and GR) can alternatively be computed via the numerical :math:`\mu` pipeline in ``pk_int``, which constructs :math:`P(k,\mu)` from first-order field kernels and projects onto Legendre multipoles. See :doc:`integrated` for full details, available kernels, and usage examples.
