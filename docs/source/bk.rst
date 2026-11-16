Bispectrum
==========

The bispectrum module (``bk``) computes contributions to the galaxy bispectrum in redshift space, including wide-separation and relativistic corrections.
The bispectrum output is available as multipole moments or the full angle-dependent local bispectrum.

As for the power spectrum, the signal is built numerically from the kernels of each field - the default - while the analytic expressions exported from MathWAP (`Analytic Terms`_) supply the wide-separation corrections. What goes into a signal is in :doc:`signal`; the simplest way to compute one is :doc:`model`.

Bispectrum Multipoles
---------------------

The bispectrum can be expanded in terms of the "Scoccimarro" spherical harmonic multipoles, which are defined with respect to the orientation of the triangle to the line-of-sight. With :math:`\mu = \hat{\mathbf{k}}_1\cdot\hat{\mathbf{d}}` and :math:`\phi` the azimuth of :math:`\mathbf{k}_2` about :math:`\mathbf{k}_1`,

.. math::

   B_{\ell m}(k_1, k_2, k_3) = \int \mathrm{d}\Omega\; Y^*_{\ell m}(\mu, \phi)\, B(k_1, k_2, k_3, \mu, \phi).

The other two sides' cosines follow from the triangle: :math:`\mu_2 = \mu\cos\theta + \sqrt{1-\mu^2}\sin\theta\cos\phi`, with :math:`\theta` the outside angle (:math:`\cos\theta = \hat{\mathbf{k}}_1\cdot\hat{\mathbf{k}}_2`), and :math:`\mu_3 = -(k_1\mu + k_2\mu_2)/k_3`. The triangle is given by ``k1``, ``k2`` and either ``k3`` or ``theta``. :math:`B` depends on :math:`\phi` only through :math:`\cos\phi`, so :math:`B_{\ell,-m} = (-1)^m B_{\ell m}` and :math:`m \geq 0` is enough: ``l2`` is :math:`(\ell, m) = (2, 0)` and ``l2m1`` is :math:`(2, 1)`. The multipoles are in :math:`(\mathrm{Mpc}/h)^6`, and the odd ones vanish on equilateral triangles.

**LOS parameters (r, s):** The line-of-sight direction **d** for the local triplet is defined as:

.. math::

   \mathbf{d} = r \, \mathbf{x}_1 + s \, \mathbf{x}_2 + (1-r-s) \, \mathbf{x}_3

where r, s ∈ [0,1] and x_i are the triplet positions. Default r=s=0 corresponds to the endpoint x_3 LOS; r=s=1/3 selects the triplet centroid.

Computing them
--------------

``Model.bk`` is the simplest way: a kernel list per field and the wide-separation terms on one :math:`(\mu, \phi)` grid, at every :math:`(\ell, m)`, with FoG - see :ref:`its bispectrum <model-bk>`. Below it are ``bk_func`` - what the forecasts call - and ``numeric_mu.bk.get_multipoles``.

bk_func
~~~~~~~

The bispectrum signal is built numerically from the redshift-space kernels in ``numeric_mu``: the full :math:`B(\mu, \phi)` is computed once per tracer combination on a Gauss-Legendre :math:`(\mu, \phi)` grid and projected onto every requested :math:`(\ell, m)`, with FoG applied inside the projection. Use them through ``bk_func``, where they are summed onto the analytic ``term`` (``None`` for kernels only):

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap import bk
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    cosmo_funcs = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo))
    z = 1.0

    # l = 0..3 (and optionally (l, m) pairs) from one B(mu, phi) evaluation - on a scalene
    # triangle, as the odd multipoles vanish on equilateral ones
    B_l = bk.bk_func(None, [0, 1, 2, 3], cosmo_funcs, 0.1, 0.08, 0.05, zz=z,
                     kernels=['N', 'LP'], mu_grid=[8, 8])

``mu_grid`` is ``[n_mu, n_phi]`` (default ``[16, 16]``). Without FoG, :math:`B(\mu,\phi)` from local kernels is a polynomial of degree 8 in the LOS direction, so ``[8, 8]`` is exact through :math:`\ell = 4`. The analytic forecast dispatch uses :math:`m = 0`, so :math:`m > 0` multipoles in ``FullForecast(all_m=True)`` need the kernels. Some raw analytic methods, such as ``bk.GR1.l1m1``, also exist outside that dispatch. ``numeric_mu.bk.get_multipoles`` is the lower-level entry point, with separate kernel lists for each of the three fields.

.. _bk-kernels:

Second-order kernels
--------------------

The kernels need a second-order part, in ``numeric_mu.kernels.K2`` - which ones have it, and their first-order parts, are in :ref:`signal-kernels`. ``'LP'`` agrees with ``GR1 + GR2`` through second order in :math:`\mathcal{H}/k`.

The PNG kernels are part of each field's :math:`Z_1` and :math:`Z_2` throughout, so the numeric :math:`B` keeps every power of :math:`f_{\rm NL}` at tree level (up to :math:`f_{\rm NL}^4`): products of the scale-dependent bias on several legs, cross terms between shapes, the scale-dependent bias on the legs of the primordial bispectrum, and the :math:`b_{02}\,\varphi_p^2` term of arXiv:2011.13660 eq (2.17). :math:`b_{02}` is included for ``'Loc'`` only - for ``'Eq'`` and ``'Orth'`` the response to :math:`\varphi_p^2` depends on the primordial trispectrum, which their templates do not fix. The analytic ``Loc``/``Eq``/``Orth`` terms are linear in :math:`f_{\rm NL}` (``bk_mt.Loc`` keeps the tree to :math:`f_{\rm NL}^3`, but without :math:`b_{02}` or the bias on the primordial legs), so they agree with the kernels at linear order only. Fisher derivatives at a fiducial :math:`f_{\rm NL}=0` are unchanged, as the central stencil cancels the even powers.

By default ``'LP'`` leaves out the :math:`(\mathcal{H}/k)^3` and :math:`(\mathcal{H}/k)^4` terms of the second-order kernel. Setting ``cosmo_funcs.full_betas = True`` adds them: :math:`\beta_1`-:math:`\beta_5` of arXiv:2011.13660 eq (3.24), and the coupling of the scale-dependent PNG bias to first-order projection effects (:math:`\Upsilon_3`-:math:`\Upsilon_5` of eq (3.26)), which enters a field only if it has both ``'LP'`` and a PNG kernel. :math:`\Upsilon_1` and :math:`\Upsilon_2` are already included through the relativistic :math:`Z_1` on the primordial bispectrum. The paper derives the :math:`\Upsilon`'s for local PNG only; ``'Eq'`` and ``'Orth'`` use the same couplings with their own :math:`b_{01}(z)`. The analytic ``GR1``/``GR2`` terms ignore the flag.

``'I'`` (integrated effects) has no second-order kernel yet.

Analytic Terms
--------------

The analytic expressions are what you need for the wide-separation corrections, which the kernels do not have, and are kept as a check on the kernels. With the compiled expressions and coefficient tables (see :doc:`performance`) they are also the faster bispectrum in an MCMC. In forecasts they are passed as ``bk_terms``, summed onto ``bk_kernels``.

.. _bk-terms:

Available Bispectrum Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"Kernel" is the kernel that computes the same contribution, which a forecast will not let you add twice. Every class has a method for each :math:`(\ell, m)` to :math:`\ell = 4` - ``lx`` for :math:`m = 0`, ``lxmy`` for :math:`(x, y)` - and ``ylm()`` for any :math:`(\ell, m)` and FoG (see `Methods`_). A term's multipoles of the other parity vanish.

.. list-table::
   :header-rows: 1
   :widths: 18 52 14 16

   * - Term
     - Contribution
     - Kernel
     - :math:`\ell`
   * - ``NPP``
     - Newtonian plane-parallel (Kaiser)
     - ``'N'``
     - even
   * - ``GR1``
     - First-order relativistic corrections (:math:`\mathcal{H}/k`)
     - ``'LP'``
     - odd
   * - ``GR2``
     - Second-order relativistic corrections (:math:`\mathcal{H}/k`)\ :sup:`2`
     - ``'LP'``
     - even
   * - ``Loc``, ``Eq``, ``Orth``
     - PNG contributions (local, equilateral, orthogonal), linear in :math:`f_{\rm NL}`
     - the same names
     - even
   * - ``WA1``, ``RR1``
     - First-order wide-angle and radial-redshift corrections
     - none
     - odd
   * - ``WA2``, ``RR2``
     - Second-order wide-angle and radial-redshift corrections
     - none
     - even
   * - ``WARR``
     - Mixed wide-angle and radial-redshift corrections
     - none
     - even
   * - ``WAGR``, ``RRGR``
     - Mixed wide-angle/GR and radial-redshift/GR corrections
     - none
     - even
   * - ``WS``
     - Full combined wide-separation terms (wide-angle + radial-redshift)
     - none
     - all
   * - ``WSGR``
     - ``WS + WAGR + RRGR``
     - none
     - all
   * - ``Full``
     - ``WSGR + GR1 + GR2`` - add ``NPP`` and PNG separately
     - ``'LP'``
     - all

MathWAP's exports give the low multipoles (:math:`\ell \leq 3` or less, :math:`m = 0`, and ``GR1``'s ``l1m1``, ``l3m*``); the rest - :math:`\ell = 4` and the :math:`m > 0` ones - come from angle-dependent forms generated by ``mathwap`` (see `Angle-dependent forms`_): ``bk.WS_`` for the wide-separation classes, which get their ``ylm()`` from it too, and ``bk.PP_`` for ``NPP``, ``GR1``, ``GR2`` and the PNG ones. Both are projected exactly, and with FoG integrated numerically (``n=24``). The exports, with their C kernels and tables, stay where they exist, and so does the ``ylm()`` of ``NPP``, ``GR1``, ``GR2`` and the PNG classes - MathWAP's forms at 16 nodes, ~9x faster with FoG than ``n=24`` but a few % off at :math:`(4, 4)`. With every part complete, the composites (summed as in :doc:`signal`) have every :math:`(\ell, m)` to :math:`\ell = 4` too.

.. note::

   ``Orth`` is the orthogonal-LSS template of Senatore, Smith & Zaldarriaga (2010, App. B), which goes as :math:`(k_L/k_S)^2` in the squeezed limit, so its scale-dependent bias goes as :math:`k^2/\mathcal{M}(k)`, as for ``Eq`` (:math:`\alpha=2`, :math:`A = 3 - 5q/9 \approx -1.74`). The standard (CMB) orthogonal template's :math:`k/\mathcal{M}(k)` bias is an artefact of its squeezed limit. See ``cosmo_wap.lib.orth_lss``.

Methods
~~~~~~~

Here's an example class from the bispectrum module:

.. class:: bk.NPP

    This class computes the Newtonian plane-parallel constant redshift terms.

    .. method:: lx(cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

        Compute the x-th multipole :math:`l_x` of the bispectrum for the Newtonian contribution. Where a class has them, ``lxmy`` give the :math:`(x, y)` multipoles with the same arguments.

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

Use ``bk.bk_func(term, l, cosmo_funcs, k1, k2, k3=..., zz=..., sigma=...)`` to route FoG through the supported numerical projection, or use the numerical kernels (see `Computing them`_). Direct analytic methods such as ``NPP.l0`` do not accept ``sigma``.

Angle-dependent forms
---------------------

In addition to the multipole decomposition, CosmoWAP also provides functions to compute the full angle-dependent local bispectrum.

.. function:: bk.Bk_0(mu, phi, cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

    Compute the angle-dependent Newtonian bispectrum.

    :param float mu: Cosine of the angle between the LOS and :math:`k_1`
    :param float phi: Azimuthal angle between LOS and :math:`k_2` in plane normal to :math:`k_1`.

    The remaining arguments and the return value are as for ``NPP.lx``.

Full angle-dependent forms are also exported as ``bk.GR_1``, ``bk.GR_2``, ``bk.Loc_``, ``bk.Eq_`` and ``bk.Orth_``. Their class-level ``ylm()`` methods provide numerical projection.

The wide-separation forms are ``bk.WA_1``, ``bk.RR_1``, ``bk.WA_2``, ``bk.RR_2``, ``bk.WARR_``, ``bk.WAGR_`` and ``bk.RRGR_``, with the same signature as ``bk.Bk_0``, for three tracers (tracer :math:`i` at :math:`\mathbf{x}_i`, as in ``bk_mt``) and any ``r``, ``s``. They are generated by ``mathwap`` (see :doc:`signal`) as the coefficients of :math:`\mu_1^a\mu_2^b`, evaluated once per triangle. Without FoG their multipoles are those coefficients projected exactly, and with FoG ``ylm`` integrates numerically (``n=24`` - 16 nodes are not enough in :math:`\phi` for these). ``Model.bk_mu`` sums them with the kernels' :math:`B(\mu, \phi)`.

The plane-parallel forms in ``bk.PP_`` are ``bk.NPP_``, ``bk.GR1_``, ``bk.GR2_``, ``bk.PNG1_`` and ``bk.PNG3_``, generated the same way for three tracers; ``PNG1_`` is linear in :math:`f_{\rm NL}` and ``PNG3_`` keeps every power of the tree, with ``shape`` (``'Loc'``, ``'Eq'``, ``'Orth'``) and ``fNL`` as keywords. For one tracer they agree with ``bk.Bk_0``, ``bk.GR_1``, ``bk.GR_2`` and ``bk.Loc_``/``Eq_``/``Orth_``. ``GR2_`` has the :math:`(\mathcal{H}/k)^2` part of :math:`Z_2` (:math:`\beta_6`-:math:`\beta_{13}`), as ``'LP'`` without ``full_betas``.

Multi-tracer
------------

With more than one tracer, field :math:`i` sits at :math:`\mathbf{x}_i`, on the leg :math:`k_i`, and takes its biases from ``survey[i-1]`` - in the kernels, ``Model.bk`` and the multi-tracer analytic terms in ``bk_mt``, which ``bk_func`` uses whenever ``cosmo_funcs.multi_tracer`` is set. ``ClassWAP`` fills an empty slot with ``survey[0]``, so two surveys give :math:`B_{ABA}`; pass three for any other ordering. In ``bk_mt``:

- the wide-separation classes come from the same angle-dependent forms as ``bk``'s, for three tracers - every :math:`(\ell, m)` to :math:`\ell = 4`, and ``ylm()`` for FoG
- ``NPP``, ``GR1``, ``GR2``, ``Loc``, ``Eq`` and ``Orth`` keep their exported :math:`m = 0` multipoles, and get the rest to :math:`\ell = 4` and ``ylm()`` (so FoG) from ``bk.PP_`` - ``GR2.l4``, which the export never had, included
- so ``Full`` has every :math:`(\ell, m)` to :math:`\ell = 4`, with FoG
- the PNG classes keep the tree to :math:`f_{\rm NL}^3` (see `Second-order kernels`_), ``bk.PP_``'s ``PNG3`` too

Which leg placements a multi-tracer forecast uses (``all_tracer``) is in :doc:`forecast`.

Bispectrum Gaussian Covariance
------------------------------

CosmoWAP provides Gaussian covariance for bispectrum multipoles, with both analytical expressions and numerical :math:`\mu`-:math:`\phi` integration. Forecasts use numerical integration with or without FoG damping. See :doc:`covariance` for full details, the multi-tracer forecasting covariance, the non-Gaussian terms, and a comparison with Quijote simulations.
