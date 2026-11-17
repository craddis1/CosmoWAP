Model: P(k), ξ(r) and B
=========================

``cw.Model`` holds a signal - the kernels of each field, and the analytic wide-separation terms the kernels do not have - and computes its power spectrum multipoles, its :math:`P(k,\mu)` or its correlation function multipoles from it, and its bispectrum multipoles (see `Bispectrum`_). The pieces are those of :doc:`pk` and :doc:`signal`: the numeric kernels of ``numeric_mu`` and, for the wide-separation terms, the full angular polynomials of the ``pk`` classes (their ``mu`` methods). ``pk_mu`` sums them, and ``pk`` projects that sum onto the Legendre polynomials, so the wide-separation multipoles are exact for every :math:`\ell` - up to :math:`\ell = 6`, where their polynomials end - with or without FoG.

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    cosmo = utils.get_cosmo()
    cosmo_funcs = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo))

    m = cw.Model(cosmo_funcs, kernels=['N', 'LP', 'I'], wide_separation=True)

    k = np.logspace(-3, np.log10(0.3), 100)
    mu = np.linspace(-0.95, 0.95, 20)   # avoid mu = 0 itself with integrated kernels
    r = np.linspace(10, 200, 39)

    P_l = m.pk(k, l=[0, 2, 4], zz=1.0)        # (3, 100), complex - odd l imaginary
    P_kmu = m.pk_mu(k, mu, zz=1.0)            # (100, len(mu))
    xi_l = m.xi(r, l=[0, 2, 4], zz=1.0)       # (3, 39), real
    xi_bins = m.xi(r_edges=np.arange(10, 210, 10), l=0, zz=1.0)   # bin averages

.. toctree::
   :hidden:

   Survey window <window>

Fields and kernels
------------------

``kernels`` are the kernels of field 1, at :math:`\mathbf{x}_1`, and ``kernels2`` those of field 2, at :math:`\mathbf{x}_2` (default: the same as ``kernels``, and ``kernels`` defaults to ``['N']``). The bispectrum has a third field, ``kernels3`` at :math:`\mathbf{x}_3`, also defaulting to ``kernels``. The tracers are ``cosmo_funcs``' own - ``survey[0]`` for field 1, ``survey[1]`` for field 2 and ``survey[2]`` for field 3, the same tracer for a single survey. With two surveys field 3 is ``survey[0]`` again (ClassWAP fills the empty slot with it), so pass three for e.g. :math:`B_{ABB}`.

Different kernels on the two fields give a single cross term. The spectrum is linear in each field's kernels, so the pieces add up to the total:

.. code-block:: python

    import matplotlib.pyplot as plt

    S, I = ['N', 'LP'], ['I']
    for name, (a, b) in {'SS': (S, S), 'SI': (S, I), 'IS': (I, S), 'II': (I, I)}.items():
        plt.loglog(k, abs(cw.Model(cosmo_funcs, kernels=a, kernels2=b).pk(k, l=0, zz=1.0)), label=name)

The order matters for the odd multipoles: swapping the two fields - each tracer with its kernels - flips their sign, so IS and SI differ.

Wide separation
---------------

``wide_separation`` adds the analytic wide-separation terms on top of the kernels (which are leading order in :math:`r/d`):

- ``False`` (default) - none.
- ``True`` - all that fit the kernels: ``WA1``, ``RR1``, ``WA2``, ``RR2``, ``WARR``, and ``WAGR``, ``RRGR`` once ``'LP'`` is in the kernels (the forecasts' ``WS`` and ``WSGR``).
- One or more blocks or groups, as a list to combine:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Name
     - Blocks
   * - ``'WA'``
     - ``WA1``, ``WA2``
   * - ``'RR'``
     - ``RR1``, ``RR2``
   * - ``'WS1'``
     - ``WA1``, ``RR1`` (first order in :math:`r/d`, odd multipoles)
   * - ``'WS2'``
     - ``WA2``, ``RR2``, ``WARR`` (second order, even multipoles)
   * - ``'WS'``
     - all five above - ``WARR`` is in neither ``'WA'`` nor ``'RR'``
   * - ``'WSGR'``
     - ``'WS'`` with ``WAGR``, ``RRGR`` (the mixing with the local relativistic terms)

Each block is added once, however it is named. The wide-separation terms correct the Newtonian term, so they need ``'N'`` in the kernels (and ``WAGR``, ``RRGR`` also ``'LP'``), and the same kernels on every field - the analytic terms are not split by field.

``t`` sets their line of sight, :math:`\mathbf{d} = (1-t)\,\mathbf{x}_1 + t\,\mathbf{x}_2`: ``0``, the end point :math:`\mathbf{x}_1`, by default - as FFT (Yamamoto) power spectrum estimators, and the integrated kernels, which are built about :math:`\mathbf{x}_1` whatever ``t`` is - ``1`` at :math:`\mathbf{x}_2`, and ``0.5`` the midpoint, the usual choice for correlation function estimators (e.g. pycorr's ``los='midpoint'``). At the end point a single tracer has odd multipoles from the wide-separation terms (the wide-angle dipole); at the midpoint they vanish.

``r`` and ``s`` set the bispectrum's, :math:`\mathbf{d} = r\,\mathbf{x}_1 + s\,\mathbf{x}_2 + (1-r-s)\,\mathbf{x}_3`: ``r = s = 0``, the end point :math:`\mathbf{x}_3`, by default, as ``bk_func`` and the forecasts.

Correlation function multipoles
-------------------------------

``xi`` is the Hankel transform of ``pk``,

.. math::

    \xi_\ell(r) = \mathrm{Re}\left[\frac{i^\ell}{2\pi^2}\int \mathrm{d}k\, k^2 P_\ell(k)\, j_\ell(kr)\right],

so it is in the same basis: Legendre multipoles in :math:`\mu = \hat{\mathbf{d}}\cdot\hat{\mathbf{x}}_{12}` at fixed :math:`|\mathbf{d}|`, with :math:`\mathbf{x}_{12} = \mathbf{x}_1 - \mathbf{x}_2`. Any :math:`P_\ell` - an analytic ``.l0`` for example - can be transformed directly with ``cosmo_wap.lib.hankel.pk_to_xi``.

- :math:`P_\ell` is computed on ``n_kc=2000`` log-spaced :math:`k` from :math:`10^{-5}\,h\,\mathrm{Mpc}^{-1}` to ``K_MAX`` and splined onto ``n_k=2**14`` points (``n_kc=None`` computes it on all of them; the spline changes :math:`\xi_\ell` by :math:`\sim 10^{-7}`).
- Each :math:`r` gets its own smooth window in :math:`\ln k`, 1 below :math:`k = 200/r` and 0 above :math:`600/r` (``kr_taper``): beyond that the integral is a cancellation of many oscillations carrying no information. Against a brute-force grid of :math:`8\times 10^4` points this is within :math:`2\times10^{-5}` of :math:`\xi_\ell` out to :math:`400\,h^{-1}\,\mathrm{Mpc}`. :math:`\xi` at :math:`r \lesssim 600/` ``K_MAX`` is limited by ``K_MAX`` itself.
- The potential-like (:math:`k^{-4}`) terms make the monopole diverge as :math:`k_{\rm min} \to 0`. Below ``k_ir`` :math:`=10^{-4}\,h\,\mathrm{Mpc}^{-1}` :math:`j_0` is replaced by :math:`j_0 - 1` (smoothly), which removes an :math:`r`-independent constant only, so :math:`\xi_0` does not depend on :math:`k_{\rm min}`. For the Kaiser monopole the constant is below :math:`4\times10^{-7}` of :math:`\xi_0(400\,h^{-1}\,\mathrm{Mpc})`.
- ``r_edges`` gives :math:`r^2`-weighted bin averages (``n_gl=16`` Gauss-Legendre nodes per bin).

The flat-sky Kaiser :math:`\xi_{0,2,4}` agree with COFFE to 0.03% given the same inputs (``tests/test_model.py``).

Through a survey window
-----------------------

``pk(k, l, zz, window=W)`` and ``xi(r, l, zz, window=W)`` put the signal through a survey window - a ``SurveyWindow`` built once from the window's multipoles, with the integral constraint - see :doc:`window`. ``pk(..., by_order=True)`` and ``xi(..., by_order=True)`` return ``{n: P}`` split by wide-angle order :math:`n`, the power of :math:`1/d` (0 for the kernels, ``WS_ORDER`` for the wide-separation blocks), which a window can weight separately.

The window fixes the output, so ``k`` and ``l`` must be its ``W.k`` (bin centres for ``k_edges``) and ``W.ells`` - a likelihood that already asks for the data's :math:`k` and multipoles just adds ``window=W``. ``by_order`` splits the windowed result the same way (the window is linear, integral constraint and all). ``pk_mu`` takes no window, and ``xi`` with one takes points ``r``, not ``r_edges``. ``xi`` takes :math:`\xi` below ``k_split`` (default 0.02 :math:`h\,\mathrm{Mpc}^{-1}`) through the window's own :math:`k` quadrature, as the integral constraint, so the infrared constants cancel exactly, and above it from its own transform - see :doc:`window`.

.. _model-bk:

Bispectrum
----------

``bk`` gives the bispectrum multipoles of the same signal, in the Scoccimarro basis of :doc:`bk` (:math:`\mu = \hat{\mathbf{k}}_1\cdot\hat{\mathbf{d}}`, :math:`\phi` the azimuth of :math:`\mathbf{k}_2` about :math:`\mathbf{k}_1`), and ``bk_mu`` the full :math:`B(\mu, \phi)`:

.. code-block:: python

    m = cw.Model(cosmo_funcs, kernels=['N', 'LP'], wide_separation=True)

    k1, k2, k3 = np.array([0.1, 0.05]), np.array([0.08, 0.04]), np.array([0.05, 0.03])
    phi = np.linspace(0, 2 * np.pi, 16)
    B_lm = m.bk(k1, k2, k3, l=[0, 1, 2, (2, 1)], zz=1.0)   # (4, 2), complex
    B_21 = m.bk(k1, k2, theta=2.0, l=2, m=1, zz=1.0)       # theta in place of k3
    B_mp = m.bk_mu(k1, k2, k3, mu, phi, zz=1.0)            # (2, len(mu), len(phi))

Each entry of ``l`` is an :math:`\ell` (at ``m``, default 0) or an :math:`(\ell, m)` pair. The kernels' :math:`B(\mu,\phi)` (``numeric_mu.bk``) and the wide-separation blocks' angle-dependent forms (``bk.WA_1``, ...) are summed on one :math:`(\mu, \phi)` grid and projected onto each :math:`(\ell, m)`, with FoG damping both alike. Both are polynomials of degree 8 in the line-of-sight direction without FoG, so ``bk_mu_grid=[8, 8]`` is exact for every :math:`(\ell, m)` to :math:`\ell = 4`; the default ``[16, 16]`` matches the exact wide-separation multipoles to :math:`\sim 10^{-13}` with FoG too.

``r`` and ``s`` set the line of sight of the wide-separation terms (see `Wide separation`_). Unlike ``bk_func``, ``bk`` keeps every :math:`m` and the wide-separation terms together, with different kernels on each field.

The kernels need a second-order part (see :ref:`bk-kernels`), so the integrated ones raise here - ``pk`` of the same ``Model`` is unaffected. There is no 3-point function or survey window for the bispectrum, and ``bk`` evaluates exact triangles: the bin averages and leg placements of the forecasts are in :doc:`forecast`.

Precision
---------

``n_mu``, ``n_los`` (the line-of-sight nodes of the integrated kernels), ``deg`` and ``n_p`` set the precision of the kernels' multipoles - ``n_mu=256`` and ``n_los=32`` by default, above the forecasts' 48 and 8 (``mu_grid``): the integrated kernels' :math:`P_\ell` to :math:`\sim 0.1\%`, where 48 and 8 leave 5-30% at :math:`k < 0.03\,h\,\mathrm{Mpc}^{-1}`. Without FoG the local kernels are polynomials in :math:`\mu`, so exact either way. ``None`` keeps the ``get_multipoles`` defaults for ``deg`` and ``n_p``. With integrated kernels this makes ``xi`` ~1 s a call. ``bk_mu_grid`` (``[n_mu, n_phi]``, default ``[16, 16]``) is the bispectrum's grid. ``sigma`` is the FoG damping, as in ``pk_func`` and ``bk_func``, and other keywords (``fNL``, ...) go to the kernels - ``GL`` and ``delta`` also set the :math:`\mu` grid of ``pk``, as in ``get_multipoles``.
