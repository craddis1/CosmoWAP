Power Spectrum Module
=====================

The ``pk`` module computes galaxy power spectrum multipoles in redshift space.

Available Classes
-----------------

* ``NPP``: Newtonian plane-parallel (Kaiser)
* ``WA1``, ``WA2``: Wide-angle corrections (1st/2nd order)
* ``RR1``, ``RR2``: Radial-redshift corrections
* ``WS``: Combined wide-separation (WA + RR)
* ``GR1``, ``GR2``: Relativistic corrections
* ``Loc``: Local PNG

Methods
-------

Each class provides multipole methods:

.. method:: lx(cosmo_functions, k1, zz=0, t=0, sigma=None)

   Compute x-th multipole (``l0``, ``l2``, ``l4``, etc.).

   :param object cosmo_functions: ``ClassWAP`` instance
   :param array-like k1: Wavevector [h/Mpc]
   :param float zz: Redshift
   :param float t: LoS parameter
   :param float sigma: FoG damping
   :return: Power spectrum multipole [(Mpc/h)^3]

Usage
-----

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    import cosmo_wap.pk as pk
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31)
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    k = np.linspace(0.01, 0.3, 100)
    z = 1.0

    # Newtonian multipoles
    P0 = pk.NPP.l0(cosmo_funcs, k, zz=z)
    P2 = pk.NPP.l2(cosmo_funcs, k, zz=z)
    P4 = pk.NPP.l4(cosmo_funcs, k, zz=z)

    # Wide-angle corrections
    P0_WA = pk.WA1.l0(cosmo_funcs, k, zz=z)

    # Relativistic corrections
    P0_GR = pk.GR1.l0(cosmo_funcs, k, zz=z)
