Integrated Effects
==================

CosmoWAP can also compute integrated contributions to the galaxy 3D power spectrum. See `arXiv:2511.09466 <https://arxiv.org/abs/2511.09466>`_ for full details.
These involve computing line-of-sight integrals between the observer and source, including lensing convergence, the integrated Sachs-Wolfe (ISW) effect, and time delay.

Integration Methods
-------------------

There are two pipelines for computing the integrated contributions (see Appendix G of arXiv:2511.09466 for full details):

- **Numerical** :math:`\mu`: for an endpoint line of sight the integrals can be rewritten so that the oscillatory parts reduce to 1D integrals of a single variable, which are precomputed with Filon-type quadrature and interpolated; the :math:`\mu` integration is then done numerically. This is :math:`\mathcal{O}(10)` faster for the multipoles and :math:`\mathcal{O}(1000)` or more for the covariances.
- **Analytic** :math:`\mu`: the :math:`\mu` integration is performed analytically (in Mathematica), leaving a single line-of-sight integral over :math:`r` for the Integrated x Standard (IxS) terms and a double integral over :math:`(r_1, r_2)` for the Integrated x Integrated (IxI) terms. These are evaluated with Gauss-Legendre quadrature.

In practice: the numerical pipeline is the default - it is what the integrated kernels (``'I'``, ``'L'``, ``'TD'``, ``'ISW'``) compute, in the signal of ``get_fish``/``sampler`` and in the covariance classes (see :doc:`signal`), while the analytic multipole classes are the standalone way to compute individual integrated multipoles and a check on it (see `Analytic Multipole Classes`_ below).

Numerical :math:`\mu` Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``numeric_mu.pk.get_multipole`` provides the default numerical approach for computing power spectrum multipoles. Rather than using the analytically-derived multipole expressions (e.g. ``pk.NPP.l0``), it constructs the full :math:`P(k,\mu)` from given kernels (defined in ``numeric_mu/kernels.py``) and numerically projects onto Legendre multipoles. This handles any combination of standard and integrated kernels with a single interface.
For an endpoint LOS the resulting (up to) 3D integrals can be rewritten so that they are much faster to evaluate (Appendix G.1 of arXiv:2511.09466), with Filon-type quadrature for the oscillatory parts.

The key idea (Appendix G.1.1 of arXiv:2511.09466): for an endpoint LOS (:math:`t=0`) the integrated kernel can be expanded in powers of :math:`k` and :math:`\mu`,

.. math::

   \mathcal{K}^{\rm I}(k, \mu, d, r) = \sum_{n}\sum_{m} k^n \mu^m \, \mathcal{Z}^{\rm I}_{mn}(d, y),

where :math:`y = \chi/d` is the normalised radial distance. The oscillatory part of the line-of-sight integral then only enters through 1D integrals of the coefficients,

.. math::

   I_{mn}(p) = \int_0^1 \mathrm{d}y \; e^{iyp} \, d \, \mathcal{Z}^{\rm I}_{mn}(d, y),

which depend on :math:`k`, :math:`\mu` (and, for IxI, the second radial variable :math:`y_2`) only through the single combination :math:`p = k\mu d/y_2`. Fitting the :math:`y` dependence of :math:`\mathcal{Z}^{\rm I}_{mn}` with a polynomial, these oscillatory integrals can be computed analytically (via integration by parts) over the whole :math:`p` range and interpolated. The IxI power spectrum then reduces to a single remaining radial integral,

.. math::

   P^{\rm I \times I}_{\rm loc}(k,\mu,d) = e^{-ik\mu d} \sum_n \sum_m \mu^m
   \int_0^1 \mathrm{d}y_2 \left(\frac{k}{y_2}\right)^{\!n} \frac{d}{y_2^{3}}\,
   \mathcal{K}^{\rm I}\!\left(\frac{k}{y_2}, -\mu, d, y_2\right)
   P\!\left(\frac{k}{y_2}\right) I_{mn}(p),

which is oscillatory close to the source and is evaluated with Filon-type quadrature. Since the :math:`\mu` integration is kept numerical, the full :math:`\mu`-dependent power spectrum is computed once per redshift bin, from which all multipoles follow trivially.

The functions (``numeric_mu.pk.get_multipole``, ``get_multipoles``, ``get_mu``, ``get_mu_sym``) and the kernels, of which the integrated ones are ``'I'``, ``'L'``, ``'TD'``, ``'ISW'`` and ``'kappa_g'``, are with the :doc:`power spectrum <pk>`.

Usage
^^^^^

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap.numeric_mu import pk as pk_int
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    k = np.linspace(0.01, 0.3, 100)
    z = 1.0

    # Standard Newtonian (Kaiser x Kaiser) -- equivalent to pk.NPP
    P0_NPP = pk_int.get_multipole(['N'], ['N'], 0, cosmo_funcs, k, z)

    # Integrated x NPP (lensing + ISW + TD crossed with Kaiser)
    P0_IntNPP = pk_int.get_multipole(['I'], ['N'], 0, cosmo_funcs, k, z) + pk_int.get_multipole(['N'], ['I'], 0, cosmo_funcs, k, z)
    P2_IntNPP = pk_int.get_multipole(['I'], ['N'], 2, cosmo_funcs, k, z) + pk_int.get_multipole(['N'], ['I'], 2, cosmo_funcs, k, z)

    # Integrated x Integrated
    P0_IntInt = pk_int.get_multipole(['I'], ['I'], 0, cosmo_funcs, k, z)

    # Individual components (e.g. Lensing x NPP only)
    P0_LxNPP = pk_int.get_multipole(['L'], ['N'], 0, cosmo_funcs, k, z)

Use in Forecasting
^^^^^^^^^^^^^^^^^^

In forecasts the integrated kernels go in ``kernels``: ``kernels=['N', 'LP', 'I']`` is equivalent to the analytic ``terms=['NPP', 'GR1', 'GR2', 'IntNPP', 'IntInt']`` but much faster. ``'I'`` has no second-order kernel yet, so the bispectrum leaves it out (``bk_kernels`` defaults to ``['N', 'LP']`` there). See :doc:`signal`.

Analytic :math:`\mu` Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user-facing entry points for this pipeline are the multipole classes ``pk.int.IntNPP`` and ``pk.int.IntInt`` - see `Analytic Multipole Classes`_ below for their full API. Under the hood these use integration routines (in ``lib.integrated.BaseInt``) for evaluating 1D and 2D (for IxI contributions) line-of-sight integrals using Gauss-Legendre quadrature:

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

The integrands oscillate in :math:`r_1, r_2` with a frequency that increases with :math:`k`, so more nodes are needed to resolve them on smaller scales: with :math:`n = 256` the IxI term is converged to percent level at :math:`k = 0.1\,h/\mathrm{Mpc}` (see Fig. 18 of arXiv:2511.09466). The cost of the IxI term scales as :math:`n^2 n_k` (halved for a single tracer with :math:`t = 1/2`, where the integrand is symmetric about :math:`y_1 = y_2`).

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
