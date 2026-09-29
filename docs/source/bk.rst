Bispectrum Module
=================

The bispectrum module (``bk``) computes contributions to the galaxy bispectrum in redshift space, including wide-separation and relativistic corrections.
The bispectrum output is available as multipole moments or the full angle-dependent local bispectrum.

Bispectrum Multipoles
---------------------

The bispectrum can be expanded in terms of the "Scoccimarro" spherical harmonic multipoles, which are defined with respect to the orientation of the triangle to the line-of-sight.

Here's an example class from the bispectrum module:

.. class:: bk.NPP

    This class computes the Newtonian plane-parallel constant redshift terms.

    Methods
    -------

    .. method:: lx(cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

        Compute the x-th multipole :math:`l_x` of the bispectrum for the Newtonian contribution.

        :param object cosmo_funcs: An instance of ``ClassWAP`` containing cosmology and survey biases.
        :param array-like k1: Wavevector magnitude 1, broadcastable array in units of [h/Mpc].
        :param array-like k2: Wavevector magnitude 2, broadcastable array in units of [h/Mpc].
        :param array-like k3: (Optional) Wavevector magnitude 3, broadcastable array in units of [h/Mpc]. Either `k3` or `theta` must be set.
        :param array-like theta: (Optional) Outside angle θ, broadcastable array. Either `theta` or `k3` must be set.
        :param array-like zz: Redshift, broadcastable array with k vectors. Default is 0.
        :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float sigma: (Optional) Linear dispersion that sets FoG damping. Default is None.
        :return: Bispectrum multipole contribution in units of [(Mpc/h)^6].

    .. method:: ylm(l, m, cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

        Compute the multipole :math:`(\ell, m)` of the bispectrum by performing the angular integral numerically.

        :param int l: The degree of the spherical harmonic.
        :param int m: The order of the spherical harmonic.

        The remaining arguments and the return value are as for ``lx``.

Available Bispectrum Classes
----------------------------

CosmoWAP provides multiple classes for different contributions to the bispectrum:

* `NPP`: Newtonian plane-parallel (Kaiser)
* `WA1`: First-order wide-angle corrections
* `WA2`: Second-order wide-angle corrections
* `RR1`: First-order radial-redshift corrections
* `RR2`: Second-order radial-redshift corrections
* `WS`: Full combined wide-separation terms (wide-angle + radial-redshift)
* `GR1`: First-order relativistic corrections (H/k)
* `GR2`: Second-order relativistic corrections (H/k)
* `Loc`, `Eq`, `Orth`: PNG contributions (local, equilateral, orthogonal)

.. note::

   PNG contributions (``Loc``, ``Eq``, ``Orth``) require ``compute_bias=True`` when initialising ``ClassWAP``.

Each class follows the same interface with `lx()` methods for computing multipoles of order x, and a `ylm()` method for numerical integration of arbitrary multipoles.

**LOS parameters (r, s):** The line-of-sight direction **d** for the local triplet is defined as:

.. math::

   \mathbf{d} = r \, \mathbf{x}_1 + s \, \mathbf{x}_2 + (1-r-s) \, \mathbf{x}_3

where r, s ∈ [0,1] and x_i are the triplet positions. Default r=s=0 corresponds to the "midpoint" LOS.

Full Local Bispectrum
---------------------

In addition to the multipole decomposition, CosmoWAP also provides functions to compute the full angle-dependent local bispectrum.

.. function:: bk.Bk_0(mu, phi, cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

    Compute the angle-dependent Newtonian bispectrum.

    :param float mu: Cosine of the angle between the LOS and :math:`k_1`
    :param float phi: Azimuthal angle between LOS and :math:`k_2` in plane normal to :math:`k_1`.

    The remaining arguments and the return value are as for ``NPP.lx``.

The full angle-dependent bispectrum is available for the Newtonian contribution. For other contributions, use the multipole decomposition via the class methods.

Numerical Bispectrum Kernels
----------------------------

As for the power spectrum (see :doc:`integrated`), the bispectrum signal can also be built numerically from the redshift-space kernels in ``numeric_mu``: the full :math:`B(\mu, \phi)` is computed once per tracer combination on a Gauss-Legendre :math:`(\mu, \phi)` grid and projected onto every requested :math:`(\ell, m)`, with FoG applied inside the projection. The kernels need a second-order part, in ``numeric_mu.kernels.K2``:

- ``'N'`` - Newtonian (Kaiser), including :math:`b_2` and :math:`\gamma_2`
- ``'LP'`` - local projection (relativistic) effects - agrees with ``GR1 + GR2`` through second order in :math:`\mathcal{H}/k`
- ``'Loc'``, ``'Eq'``, ``'Orth'`` or all three as ``'PNG'`` - the scale-dependent PNG bias

``'I'`` (integrated effects) has no second-order kernel yet. Use them through ``bk_func``, where they are summed onto the analytic ``term`` (``None`` for kernels only):

.. code-block:: python

    from cosmo_wap import bk

    # l = 0..3 (and optionally (l, m) pairs) from one B(mu, phi) evaluation
    B_l = bk.bk_func(None, [0, 1, 2, 3], cosmo_funcs, k1, k2, k3, zz=z,
                     kernels=['N', 'LP'], mu_grid=[8, 8])

``mu_grid`` is ``[n_mu, n_phi]`` (default ``[16, 16]``). Without FoG, :math:`B(\mu,\phi)` from local kernels is a polynomial of degree 8 in the LOS direction, so ``[8, 8]`` is exact through :math:`\ell = 4`. The analytic terms only give :math:`m = 0`, so :math:`m > 0` multipoles (``FullForecast(all_m=True)``) need the kernels. In forecasts they are passed as ``bk_kernels`` - see :doc:`forecast`. ``numeric_mu.bk.get_multipoles`` is the lower-level entry point, with separate kernel lists for each of the three fields.

Bispectrum Gaussian Covariance
------------------------------

CosmoWAP provides Gaussian covariance for bispectrum multipoles, with both analytical expressions and numerical :math:`\mu`-:math:`\phi` integration (used automatically with FoG damping). See :doc:`covariance` for full details, the multi-tracer forecasting covariance, the non-Gaussian terms, and a comparison with Quijote simulations.
