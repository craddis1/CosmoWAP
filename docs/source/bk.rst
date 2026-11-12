Analytic Bispectrum
===================

The bispectrum module (``bk``) computes contributions to the galaxy bispectrum in redshift space, including wide-separation and relativistic corrections, from the analytic expressions exported from MathWAP.
The bispectrum output is available as multipole moments or the full angle-dependent local bispectrum.

The signal is normally built from the numerical kernels instead (see :doc:`signal`). The analytic terms are what you need for the wide-separation corrections, which the kernels do not have, and are kept as a check on the kernels. With the compiled expressions and coefficient tables (see :doc:`performance`) they are also the faster bispectrum in an MCMC. In forecasts they are passed as ``bk_terms``, summed onto ``bk_kernels``.

Bispectrum Multipoles
---------------------

The bispectrum can be expanded in terms of the "Scoccimarro" spherical harmonic multipoles, which are defined with respect to the orientation of the triangle to the line-of-sight.

Here's an example class from the bispectrum module:

.. class:: bk.NPP

    This class computes the Newtonian plane-parallel constant redshift terms.

    Methods
    -------

    .. method:: lx(cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

        Compute the x-th multipole :math:`l_x` of the bispectrum for the Newtonian contribution.

        :param object cosmo_funcs: An instance of ``ClassWAP`` containing cosmology and survey biases.
        :param k1: Array-like wavevector magnitude 1, broadcastable array in units of [h/Mpc].
        :param k2: Array-like wavevector magnitude 2, broadcastable array in units of [h/Mpc].
        :param k3: (Optional) Wavevector magnitude 3, array-like and broadcastable in units of [h/Mpc]. Either `k3` or `theta` must be set.
        :param theta: (Optional) Outside angle θ, array-like and broadcastable. Either `theta` or `k3` must be set.
        :param zz: Array-like redshift, broadcastable array with k vectors. Default is 0.
        :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :return: Bispectrum multipole contribution in units of [(Mpc/h)^6].

    .. method:: ylm(l, m, cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

        Compute the multipole :math:`(\ell, m)` of the bispectrum by performing the angular integral numerically.

        :param int l: The degree of the spherical harmonic.
        :param int m: The order of the spherical harmonic.
        :param float sigma: (Optional) Per-field FoG dispersion in h⁻¹ Mpc. Default is None.

        The remaining arguments and the return value are as for ``lx``.

Available Bispectrum Classes
----------------------------

.. _bk-terms:

CosmoWAP provides multiple classes for different contributions to the bispectrum:

* `NPP`: Newtonian plane-parallel (Kaiser)
* `WA1`: First-order wide-angle corrections
* `WA2`: Second-order wide-angle corrections
* `RR1`: First-order radial-redshift corrections
* `RR2`: Second-order radial-redshift corrections
* `WARR`: Mixed wide-angle and radial-redshift corrections
* `WS`: Full combined wide-separation terms (wide-angle + radial-redshift)
* `GR1`: First-order relativistic corrections (H/k)
* `GR2`: Second-order relativistic corrections (H/k)^2
* `WAGR`, `RRGR`: Mixed wide-angle/GR and radial-redshift/GR corrections
* `WSGR`: ``WS + WAGR + RRGR``
* `Full`: ``WSGR + GR1 + GR2`` - add ``NPP`` and PNG separately
* `Loc`, `Eq`, `Orth`: PNG contributions (local, equilateral, orthogonal)

.. note::

   Equilateral and orthogonal PNG biases require ``compute_bias=True`` when initialising ``ClassWAP``. Local PNG also works with the default survey bias prescription; ``compute_bias=True`` derives its biases from the HOD/HMF instead.

.. note::

   ``Orth`` is the orthogonal-LSS template of Senatore, Smith & Zaldarriaga (2010, App. B), which goes as :math:`(k_L/k_S)^2` in the squeezed limit, so its scale-dependent bias goes as :math:`k^2/\mathcal{M}(k)`, as for ``Eq`` (:math:`\alpha=2`, :math:`A = 3 - 5q/9 \approx -1.74`). The standard (CMB) orthogonal template's :math:`k/\mathcal{M}(k)` bias is an artefact of its squeezed limit. See ``cosmo_wap.lib.orth_lss``.

The analytic classes provide ``lx()`` multipoles. In the single-tracer case, ``NPP``, ``GR1``, ``GR2``, ``Loc``, ``Eq`` and ``Orth`` also provide ``ylm()`` for numerical angular projection and FoG. The wide-separation expressions were not exported in full angle-dependent form, so their ``ylm()``/FoG projection is unavailable, including through composites such as ``WS`` and ``Full``. Multi-tracer analytic classes have no ``ylm()`` interface.

Use ``bk.bk_func(term, l, cosmo_funcs, k1, k2, k3=..., zz=..., sigma=...)`` to route FoG through the supported numerical projection, or use the numerical kernels (see :doc:`signal`). Direct analytic methods such as ``NPP.l0`` do not accept ``sigma``.

**LOS parameters (r, s):** The line-of-sight direction **d** for the local triplet is defined as:

.. math::

   \mathbf{d} = r \, \mathbf{x}_1 + s \, \mathbf{x}_2 + (1-r-s) \, \mathbf{x}_3

where r, s ∈ [0,1] and x_i are the triplet positions. Default r=s=0 corresponds to the endpoint x_3 LOS; r=s=1/3 selects the triplet centroid.

Full Local Bispectrum
---------------------

In addition to the multipole decomposition, CosmoWAP also provides functions to compute the full angle-dependent local bispectrum.

.. function:: bk.Bk_0(mu, phi, cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

    Compute the angle-dependent Newtonian bispectrum.

    :param float mu: Cosine of the angle between the LOS and :math:`k_1`
    :param float phi: Azimuthal angle between LOS and :math:`k_2` in plane normal to :math:`k_1`.

    The remaining arguments and the return value are as for ``NPP.lx``.

Full angle-dependent forms are also exported as ``bk.GR_1``, ``bk.GR_2``, ``bk.Loc_``, ``bk.Eq_`` and ``bk.Orth_``. Their class-level ``ylm()`` methods provide numerical projection. Wide-separation corrections are available as analytic multipoles.

Numerical Bispectrum Kernels
----------------------------

The kernels, their second-order parts, the :math:`(\mu, \phi)` grid and how they enter forecasts (``bk_kernels``) are in :doc:`signal`. The analytic forecast dispatch uses :math:`m = 0`, so :math:`m > 0` multipoles in ``FullForecast(all_m=True)`` need the kernels. Some raw analytic methods, such as ``bk.GR1.l1m1``, also exist outside that dispatch.

Bispectrum Gaussian Covariance
------------------------------

CosmoWAP provides Gaussian covariance for bispectrum multipoles, with both analytical expressions and numerical :math:`\mu`-:math:`\phi` integration. Forecasts use numerical integration with or without FoG damping. See :doc:`covariance` for full details, the multi-tracer forecasting covariance, the non-Gaussian terms, and a comparison with Quijote simulations.
