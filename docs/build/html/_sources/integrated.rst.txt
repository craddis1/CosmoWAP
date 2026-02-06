Integrated Effects
==================

Integrated contributions arise from line-of-sight integrals between the observer and source, including lensing convergence, the integrated Sachs-Wolfe (ISW) effect, and time delay. These are important for unbiased :math:`f_\mathrm{NL}` constraints -- neglecting them can bias :math:`f_\mathrm{NL}` by several sigma.

Integration Methods
-------------------

CosmoWAP provides two underlying integration routines (in ``integrated.BaseInt``) for evaluating line-of-sight integrals:

.. method:: BaseInt.single_int(func, \*args, n=128, remove_div=True, source_func=None)

   Gauss-Legendre quadrature over a single line-of-sight integral :math:`\int_0^{1} f(y)\, dy`, where :math:`y = \chi/d` is the normalised radial distance. Used for RSD x Integrated terms.

   :param int n: Number of quadrature nodes
   :param bool remove_div: Excise numerical divergence near the source (default: True)
   :param source_func: Optional function returning the integrand value at the source, used for interpolation near the divergence

.. method:: BaseInt.double_int(func, \*args, n=128, n2=None, fast=True)

   Gauss-Legendre quadrature over a double line-of-sight integral :math:`\int_0^{1}\!\int_0^{1} f(y_1, y_2)\, dy_1\, dy_2`, where :math:`y_i = \chi_i/d`. Used for Integrated x Integrated terms. Exploits symmetry :math:`f(y_1, y_2) = f(y_2, y_1)` when ``fast=True`` to reduce memory and computation.

   :param int n: Number of quadrature nodes for first integral
   :param int n2: Nodes for second integral (default: same as ``n``)
   :param bool fast: Sum directly rather than building the full 2D grid (default: True)

Numerical :math:`\mu` Pipeline
-------------------------------

``pk_int.get_multipole`` provides an alternative numerical approach for computing power spectrum multipoles. Rather than using the analytically-derived multipole expressions (e.g. ``pk.NPP.l0``), it constructs the full :math:`P(k,\mu)` from first-order field kernels (defined in ``lib/kernels.py``) and numerically projects onto Legendre multipoles. This handles any combination of standard and integrated kernels with a single interface.

.. function:: pk_int.get_multipole(kernel1, kernel2, l, cosmo_funcs, kk, zz, sigma=None, n=32, n_mu=256, nr=2000, deg=8, delta=0.1, GL=False)

   Compute the l-th multipole of the power spectrum for a given pair of kernels.

   :param list kernel1: Kernel(s) for field 1 (e.g. ``['N']``, ``['N', 'LP']``, ``['I']``)
   :param list kernel2: Kernel(s) for field 2
   :param int l: Multipole order
   :param cosmo_funcs: ``ClassWAP`` instance
   :param array kk: Wavevectors [h/Mpc]
   :param float zz: Redshift
   :param float sigma: FoG damping
   :param int n: Number of Gauss-Legendre nodes for the line-of-sight integral
   :param int n_mu: Number of :math:`\mu` integration points
   :param int nr: Number of radial points for the kernel integration
   :param int deg: Polynomial degree for Filon integration
   :param float delta: Width of the central region for non-uniform :math:`\mu` grid
   :param bool GL: Use Gauss-Legendre for :math:`\mu` integration (default: non-uniform grid)
   :return: Power spectrum multipole [(Mpc/h)^3]

Available Kernels
~~~~~~~~~~~~~~~~~

The kernels are defined in ``lib/kernels.py`` (class ``K1``). Each kernel represents a first-order field contribution to the observed galaxy overdensity.

**Standard (evaluated at source):**

- ``'N'`` -- Newtonian (Kaiser RSD): :math:`D(z)[b_1 + f\mu^2]`
- ``'LP'`` -- Local projection effects (relativistic): :math:`D(z)[i\mu\,\beta_1/k + \beta_2/k^2]`

**Integrated (line-of-sight):**

- ``'I'`` -- All integrated effects combined (L + TD + ISW)
- ``'L'`` -- Lensing convergence
- ``'TD'`` -- Time delay
- ``'ISW'`` -- Integrated Sachs-Wolfe

The power spectrum for any pair of kernels is :math:`P_\ell(k) = \frac{2\ell+1}{2}\int_{-1}^{1} K_1 K_2^* P(k)\, \mathcal{L}_\ell(\mu)\, d\mu`, where the line-of-sight integrals for integrated kernels are handled internally.

Usage
~~~~~

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap.pk import pk_int
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    k = np.linspace(0.01, 0.3, 100)
    z = 1.0

    # Standard Newtonian (Kaiser x Kaiser) -- equivalent to pk.NPP
    P0_NPP = pk_int.get_multipole(['N'], ['N'], 0, cosmo_funcs, k, z)

    # Newtonian + local projection (Kaiser + GR)
    P0_full = pk_int.get_multipole(['N', 'LP'], ['N', 'LP'], 0, cosmo_funcs, k, z)

    # Integrated x NPP (lensing + ISW + TD crossed with Kaiser)
    P0_IntNPP = pk_int.get_multipole(['I'], ['N'], 0, cosmo_funcs, k, z)
    P2_IntNPP = pk_int.get_multipole(['I'], ['N'], 2, cosmo_funcs, k, z)

    # Integrated x Integrated
    P0_IntInt = pk_int.get_multipole(['I'], ['I'], 0, cosmo_funcs, k, z)

    # Individual components (e.g. Lensing x NPP only)
    P0_LxNPP = pk_int.get_multipole(['L'], ['N'], 0, cosmo_funcs, k, z)

Analytic Multipole Classes
--------------------------

Analytically-derived multipole expressions are also available in ``pk.int`` and ``pk.int_components``. These are computed in Mathematica and exported, with the line-of-sight integrals evaluated via Gauss-Legendre quadrature.

.. class:: pk.int.IntNPP

   RSD x Integrated (single line-of-sight integral).

   .. method:: l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)
   .. method:: l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)
   .. method:: l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)
   .. method:: l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)
   .. method:: l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)
   .. method:: l(l, cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n_mu=16)

      Generic l-th multipole via numeric :math:`\mu` integration.

   .. method:: mu(mu, cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128)

      Full :math:`P(k, \mu)`.

.. class:: pk.int.IntInt

   Integrated x Integrated (double line-of-sight integral).

   .. method:: l0(cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, fast=True)
   .. method:: l1(cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, fast=True)
   .. method:: l2(cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, fast=True)
   .. method:: l3(cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, fast=True)
   .. method:: l4(cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, fast=True)
   .. method:: l(l, cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128, n_mu=16)

      Generic l-th multipole via numeric :math:`\mu` integration.

   .. method:: mu(mu, cosmo_funcs, k1, zz=0, t=0.5, sigma=None, n=128)

      Full :math:`P(k, \mu)`.

Individual integrated component classes are also available in ``pk.int_components``:

- ``LxL``, ``LxTD``, ``LxISW``, ``TDxTD``, ``ISWxISW``, ``TDxISW``

Each provides ``l0`` through ``l4`` multipoles with the same interface as ``IntInt``.
