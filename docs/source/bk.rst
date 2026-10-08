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

The analytic classes provide ``lx()`` multipoles. In the single-tracer case, ``NPP``, ``GR1``, ``GR2``, ``Loc``, ``Eq`` and ``Orth`` also provide ``ylm()`` for numerical angular projection and FoG. The wide-separation expressions were not exported in full angle-dependent form, so their ``ylm()``/FoG projection is unavailable, including through composites such as ``WS`` and ``Full``. Multi-tracer analytic classes have no ``ylm()`` interface.

Use ``bk.bk_func(term, l, cosmo_funcs, k1, k2, k3=..., zz=..., sigma=...)`` to route FoG through the supported numerical projection, or use the numerical kernels below. Direct analytic methods such as ``NPP.l0`` do not accept ``sigma``.

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

As for the power spectrum (see :doc:`integrated`), the bispectrum signal can also be built numerically from the redshift-space kernels in ``numeric_mu``: the full :math:`B(\mu, \phi)` is computed once per tracer combination on a Gauss-Legendre :math:`(\mu, \phi)` grid and projected onto every requested :math:`(\ell, m)`, with FoG applied inside the projection. The kernels need a second-order part, in ``numeric_mu.kernels.K2``:

- ``'N'`` - Newtonian (Kaiser), including :math:`b_2` and :math:`\gamma_2`
- ``'LP'`` - local projection (relativistic) effects - agrees with ``GR1 + GR2`` through second order in :math:`\mathcal{H}/k`
- ``'Loc'``, ``'Eq'``, ``'Orth'`` or all three as ``'PNG'`` - the scale-dependent PNG bias

The PNG kernels are part of each field's :math:`Z_1` and :math:`Z_2` throughout, so the numeric :math:`B` keeps every power of :math:`f_{\rm NL}` at tree level (up to :math:`f_{\rm NL}^4`): products of the scale-dependent bias on several legs, cross terms between shapes, the scale-dependent bias on the legs of the primordial bispectrum, and the :math:`b_{02}\,\varphi_p^2` term of arXiv:2011.13660 eq (2.17). :math:`b_{02}` is included for ``'Loc'`` only - for ``'Eq'`` and ``'Orth'`` the response to :math:`\varphi_p^2` depends on the primordial trispectrum, which their templates do not fix. The analytic ``Loc``/``Eq``/``Orth`` terms are linear in :math:`f_{\rm NL}` (``bk_mt.Loc`` keeps the tree to :math:`f_{\rm NL}^3`, but without :math:`b_{02}` or the bias on the primordial legs), so they agree with the kernels at linear order only. Fisher derivatives at a fiducial :math:`f_{\rm NL}=0` are unchanged, as the central stencil cancels the even powers.

By default ``'LP'`` leaves out the :math:`(\mathcal{H}/k)^3` and :math:`(\mathcal{H}/k)^4` terms of the second-order kernel. Setting ``cosmo_funcs.full_betas = True`` adds them: :math:`\beta_1`-:math:`\beta_5` of arXiv:2011.13660 eq (3.24), and the coupling of the scale-dependent PNG bias to first-order projection effects (:math:`\Upsilon_3`-:math:`\Upsilon_5` of eq (3.26)), which enters a field only if it has both ``'LP'`` and a PNG kernel. :math:`\Upsilon_1` and :math:`\Upsilon_2` are already included through the relativistic :math:`Z_1` on the primordial bispectrum. The paper derives the :math:`\Upsilon`'s for local PNG only; ``'Eq'`` and ``'Orth'`` use the same couplings with their own :math:`b_{01}(z)`. The analytic ``GR1``/``GR2`` terms ignore the flag.

``'I'`` (integrated effects) has no second-order kernel yet. Use them through ``bk_func``, where they are summed onto the analytic ``term`` (``None`` for kernels only):

.. code-block:: python

    from cosmo_wap import bk

    # l = 0..3 (and optionally (l, m) pairs) from one B(mu, phi) evaluation
    B_l = bk.bk_func(None, [0, 1, 2, 3], cosmo_funcs, k1, k2, k3, zz=z,
                     kernels=['N', 'LP'], mu_grid=[8, 8])

``mu_grid`` is ``[n_mu, n_phi]`` (default ``[16, 16]``). Without FoG, :math:`B(\mu,\phi)` from local kernels is a polynomial of degree 8 in the LOS direction, so ``[8, 8]`` is exact through :math:`\ell = 4`. The analytic forecast dispatch uses :math:`m = 0`, so :math:`m > 0` multipoles in ``FullForecast(all_m=True)`` need the kernels. Some raw analytic methods, such as ``bk.GR1.l1m1``, also exist outside that dispatch. In forecasts the kernels are passed as ``bk_kernels`` - see :doc:`forecast`. ``numeric_mu.bk.get_multipoles`` is the lower-level entry point, with separate kernel lists for each of the three fields.

Bispectrum Gaussian Covariance
------------------------------

CosmoWAP provides Gaussian covariance for bispectrum multipoles, with both analytical expressions and numerical :math:`\mu`-:math:`\phi` integration. Forecasts use numerical integration with or without FoG damping. See :doc:`covariance` for full details, the multi-tracer forecasting covariance, the non-Gaussian terms, and a comparison with Quijote simulations.
