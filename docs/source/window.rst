Survey window
=============

``cosmo_wap.lib.survey_window`` puts a model's power spectrum multipoles through a survey window - the convolution that FFT (Yamamoto) estimators measure - as one matrix, built once per window: :math:`\hat P = W P`.

In configuration space the window multiplies. With the multipoles of the window :math:`Q_L(s)`,

.. math::

    \hat\xi_\ell(s) = \sum_{\ell', L} (2\ell+1) \begin{pmatrix} \ell & \ell' & L \\ 0 & 0 & 0 \end{pmatrix}^2 \xi_{\ell'}(s)\, Q_L(s),
    \qquad
    \hat P_\ell(k) = 4\pi (-i)^\ell \int \mathrm{d}s\, s^2\, \hat\xi_\ell(s)\, j_\ell(ks),

(Wilson et al. 2017; Beutler et al. 2017). Each step is linear in :math:`P`, so the Hankel transform to :math:`\xi`, the product and the transform back compose into a matrix acting on :math:`P_{\ell'}` at a set of nodes ``k_in`` (Beutler & McDonald 2021). :math:`Q_L` vanishes beyond the survey, so the :math:`s` integral is finite and needs none of ``hankel``'s tapers.

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap.lib.survey_window import SurveyWindow, window_multipoles

    s, Q = window_multipoles(s_edges, mu_edges, RR, norm=A)   # from the randoms' pair counts
    W = SurveyWindow(s, Q, ells=[0, 2, 4], k_edges=np.arange(0, 0.305, 0.005))   # seconds, once

    m = cw.Model(cosmo_funcs, kernels=['N', 'LP'], wide_separation=True)
    P_hat = m.pk(W.k, W.ells, zz=1.0, window=W)                          # (3, 60), complex as pk
    xi_hat = m.xi(np.arange(5, 200, 5), W.ells, zz=1.0, window=W)        # (3, 39)

    P_hat = W(P)   # or any P_l' on W.k_in, shape (len(W.ells_in), len(W.k_in))

Window multipoles
-----------------

``window_multipoles(s_edges, mu_edges, RR, norm)`` turns the randoms' pair counts :math:`RR(s,\mu)` into

.. math::

    Q_L(s) = \frac{2L+1}{A\, V_s} \sum_{\mu\ \rm bins} RR(s,\mu)\, \bar{\mathcal{L}}_L(\mu),

with :math:`V_s` the volume of the :math:`s` shell and :math:`\bar{\mathcal{L}}_L` the Legendre polynomial averaged over each :math:`\mu` bin. ``norm`` is the estimator's own normalisation :math:`A = \alpha^2 \sum_i w_i n_i` (de Mattia & Ruhlmann-Kleider 2019, eq. 3.3), so :math:`Q_0(s\to 0) \simeq 1` and the normalisation of the measurement and of the model cancel; the :math:`s \to 0` limit itself is ill-defined with finite randoms. :math:`\mu` must be measured against the estimator's line of sight, the same as the model's ``t`` (the Yamamoto estimator's is the end point, ``t=0``) - an end-point line of sight gives odd :math:`Q_L`, which make a dipole out of even :math:`\xi_{\ell'}`. ``mu_edges`` from 0 fold :math:`\pm\mu` together and give even :math:`L` only.

Any :math:`Q_L(s)` will do as input - from pycorr, pypower, or an analytic geometry. ``s`` should run from :math:`\sim 0` to past where :math:`Q` vanishes; :math:`Q` is taken as 0 beyond ``s[-1]``.

The matrix
----------

- ``ells`` are the output multipoles, ``k`` or ``k_edges`` the output wavenumbers - or bins, for :math:`k^2`-weighted shell averages.
- ``ells_in`` are the model's multipoles, by default every :math:`\ell'` that couples to ``ells`` through the :math:`Q_L` given (e.g. :math:`\ell' \le 8` for :math:`\ell \le 4` and :math:`L \le 4`; the odd ones too once an odd :math:`\ell` is asked for), plus those needed by the integral constraint.
- ``k_in`` are the nodes the model is computed on: 256 log-spaced from :math:`10^{-4}` to ``k[-1]`` + 0.25 :math:`h\,\mathrm{Mpc}^{-1}` by default. :math:`P` is splined between them in :math:`\ln k`, and the spline - linear in :math:`P` - is folded into the matrix, so the model is only ever evaluated on ``k_in``.
- The quadratures are trapezoidal in :math:`s` (step :math:`\pi/2(k_{\rm max} + k_{\rm in,max})`) and :math:`k'` (step :math:`\pi/4 s_{\rm max}`); ``ds`` and ``dk`` override them.

Against Gaussian windows and power spectra, where the convolution has closed forms (isotropic, and anisotropic with every :math:`Q_L` and :math:`P_{\ell'}` coupling), the defaults are good to :math:`\sim 10^{-6}`-:math:`10^{-5}` of :math:`\hat P` (``tests/test_survey_window.py``). For a window out to :math:`s = 4000\,h^{-1}\,\mathrm{Mpc}` with bins to :math:`k = 0.3\,h\,\mathrm{Mpc}^{-1}` the build takes ~2.5 s, and applying it ~60 µs.

``W.matrix`` is real: it maps :math:`i^{\ell'} P_{\ell'}` to :math:`i^\ell \hat P_\ell`, flattened as (``ells``, ``k``) by (orders, ``ells_in``, ``k_in``). ``SurveyWindow.from_matrix(matrix, k_in, ells_in, k, ells)`` takes a matrix from elsewhere - pypower, a data release - that acts on :math:`P_{\ell'}` directly.

Integrated terms
----------------

The integrated kernels (``'I'``, ``'L'``, ``'TD'``, ``'ISW'``) need one more step. Lensing correlates galaxies along the line of sight out to the observer, so :math:`\xi_I` is a long streak along the line of sight, out to :math:`\sim d` (in :math:`P`, the spike at :math:`\mu = 0` of ``numeric_mu.pk.get_mu``). A redshift bin :math:`\Delta r` deep holds no pair that far apart radially - :math:`Q = 0` beyond :math:`|s_\parallel| = \Delta r` - so the window removes most of it. Two things follow:

- Don't also give the model the bin's radial window, ``Model(..., window=(dr, k_f))`` on its own. That is ``get_int_window``'s triangle :math:`1 - |s_\parallel|/\Delta r`, the top-hat's own pair counts, which :math:`Q` already has: the two together leave the integrated part at about 2/3 of its value at low :math:`k`, so ``Model.pk``/``xi`` raise if given a survey window with ``window`` and no ``box``. The triangle is for the bin without a survey window - the Gaussian covariances and ``FullForecast(radial_window=True)``.
- :math:`Q_L` to :math:`L = 8` can't draw that edge, and lets the streak through: for a :math:`\Delta z = 0.1` bin at :math:`z = 1.55` (:math:`d/\Delta r \approx 25`) the lensing quadrupole at :math:`k = 0.005\,h\,\mathrm{Mpc}^{-1}` comes out at -3% of :math:`\hat P_0`, where the bin's is ~0. It takes :math:`L \sim d/\Delta r` to converge. The local kernels stop at :math:`\mu^4`, so :math:`L \le 8` is exact for them.

``box=1`` deals with the second by cutting :math:`\xi_I` at :math:`|s_\parallel| < \Delta r` first - a convolution along :math:`k_\parallel` with :math:`\sin(k_\parallel \Delta r)/\pi k_\parallel`. Every pair in the bin is inside the cut, so with all :math:`L` it changes nothing; at :math:`L \le 8` it gives the exact windowed lensing to ~:math:`10^{-4}` of :math:`\hat P_0`:

.. code-block:: python

    dr = cosmo_funcs.comoving_dist(z_max) - cosmo_funcs.comoving_dist(z_min)
    m = cw.Model(cosmo_funcs, kernels=['N', 'LP', 'I'], window=(dr, k_f), box=1)
    P_hat = m.pk(W.k, W.ells, zz, window=W)   # ~0.1 s per call

Against the exact window of a slab (an analytic :math:`Q` for a bin wide across the sky; :math:`\Delta z = 0.1` is in ``tests/test_radial_window.py``): the box pays off for thin bins. At :math:`\Delta z = 0.3` (:math:`d/\Delta r \approx 8`) it cuts the low-:math:`k` quadrupole error ~10x, to a few % of the lensing quadrupole; at :math:`\Delta z = 0.9` (:math:`d/\Delta r \approx 2`) there is no tail to cut, and both are within ~2%. ``box = a > 1`` leaves room for wide-angle pairs (:math:`|s_\parallel| \le \Delta r/\cos(\theta/2)` about the midpoint), but the layer past :math:`\Delta r` leaks as the tail did, so keep any margin thin. About an end-point line of sight the bound grows as :math:`s_\perp^2/2d` - not tested. ``k_f`` floors :math:`k_\perp` as for the triangle.

Wide-angle orders
-----------------

Beyond the plane-parallel limit the window depends on the pair's distance as well (Beutler, Castorina & Zhang 2019): :math:`\xi_\ell(s, d) = \sum_n (s/d)^n \xi^{(n)}_\ell(s)`, and order :math:`n` sees its own window :math:`Q^{(n)}_L`, the pair counts weighted by :math:`(s/d)^n`. Pass them keyed by ``(L, n)``, with ``d`` the effective comoving distance. The wide-separation blocks carry an explicit :math:`1/d^n` (``WS_ORDER`` in ``model.py``: ``WA1``, ``RR1``, ``WAGR``, ``RRGR`` first order, ``WA2``, ``RR2``, ``WARR`` second), and ``Model.pk(..., by_order=True)`` splits :math:`P` that way.

An order with no :math:`Q^{(n)}` of its own uses :math:`Q^{(0)}` - the effective-distance approximation, which is the same as adding it to order 0. So a window with only :math:`Q_L` treats everything at :math:`d(z)`. The ``RR`` blocks come from evolution along the line of sight rather than a pure :math:`1/d`, so their split is approximate.

Integral constraints
--------------------

If the estimator takes the mean density from the data itself - :math:`\alpha = N_g/N_r`, as every FKP estimator does - the measured density contrast averages to zero over the survey, so the survey cannot see modes on its own scale: :math:`\hat P(k \to 0) \to 0`. This is the global integral constraint. If the randoms' :math:`n(z)` also comes from the data (shuffled or binned redshifts), the mean vanishes in every redshift shell as well, which removes the radial modes on the shell's scale - the radial integral constraint, which matters for large-scale signals such as :math:`f_{\rm NL}`. De Mattia & Ruhlmann-Kleider (2019) derive both:

.. math::

    \hat\xi^{\rm cic}_\ell = \hat\xi_\ell - IC^{\delta,\rm ic}_\ell - IC^{\rm ic,\delta}_\ell + IC^{\rm ic,ic}_\ell,
    \qquad
    IC^{i,j}_\ell(s) = \int \Delta^2 \mathrm{d}\Delta \sum_p \frac{4\pi}{2p+1}\, \xi_p(\Delta)\, \mathcal{W}^{i,j}_{\ell p}(s, \Delta),

where the :math:`\mathcal{W}^{i,j}` are 3- and 4-point windows of the randoms. For the global constraint :math:`\mathcal{W}^{\rm ic,ic}` is separable and needs only :math:`Q_L`. Keeping only :math:`-IC^{\rm ic,ic}` gives the standard :math:`\hat P_\ell(k) - \hat P_0(0)\, W^2_\ell(k)/W^2_0(0)` (Beutler et al. 2017), which they find accurate for both the global and radial constraints. Their other windows took ~1500 CPU hours to compute from randoms, so here they are inputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``ic``
     - Integral constraint
   * - ``'global'`` (default)
     - :math:`-IC^{\rm ic,ic}` from :math:`Q_L` alone - the standard correction
   * - ``None``
     - none
   * - ``dict(s=..., d_ic=...)``
     - exact global: the 3-point window :math:`\mathcal{W}^{\delta,\rm glo}_{\ell p}(s,\Delta)`, as ``{(l, p): array}`` on the grid ``s`` (both axes, normalised by :math:`A`)
   * - ``dict(s=..., ic_ic=...)``
     - radial (or any other), approximated by :math:`-IC^{\rm ic,ic}`
   * - ``dict(s=..., ic_ic=..., d_ic=...)``
     - radial, exact; ``ic_d`` defaults to ``d_ic`` (equal for even :math:`\ell` with the midpoint line of sight)

The constraint acts on :math:`\xi_{\ell'}` summed over wide-angle orders. An output :math:`\ell` missing from ``ic_ic`` gets no :math:`IC^{\rm ic,ic}`; with ``d_ic`` the windows given must cover every output :math:`\ell` (pass explicit zeros if they vanish), since the exact form flips the sign of the global :math:`IC^{\rm ic,ic}` and an :math:`\ell` without its cross terms would be worse than the approximation. Both the standard and the exact global constraint are checked against closed forms for a Gaussian window (``tests/test_survey_window.py``).

The constraint also fixes the infrared. A constant in :math:`\xi_0` - which the potential-like :math:`k^{-4}` terms make diverge, see ``k_ir`` in ``hankel`` - windows to that constant times :math:`Q_\ell(s)`, and any integral constraint removes this exactly. So with one, :math:`\hat P` of the relativistic and :math:`f_{\rm NL}` terms does not depend on ``k_in[0]``; without one, it does (logarithmically).

``W.shot_noise`` is the response to a unit Poisson shot noise: :math:`\delta_{\ell 0} - W^2_\ell(k)/W^2_0(0)` for the global constraint (de Mattia & Ruhlmann-Kleider section 3.3, unweighted), and ``None`` with a user-supplied :math:`\mathcal{W}^{\rm ic,ic}`, which needs their shot-noise windows.

Windowed correlation function
-----------------------------

:math:`\hat\xi_\ell` is the middle step. ``windowed_xi(xi, Q, ells)`` is the product alone, and ``W.xi(xi, r, P)`` adds the integral constraint, whose terms integrate :math:`\xi` over the whole window and so come from :math:`P` on ``k_in`` through the matrix's own maps. If ``xi`` is omitted, ``W.xi(r=r, P=P)`` computes it from the same :math:`k` quadrature as the matrix, keeping the infrared convention consistent with the IC. But that quadrature stops sharply at ``k_in[-1]``, which is sized for the output power spectrum, and the cut rings through :math:`\xi` at every separation - 10-30% in :math:`\hat\xi_0` at 10-100 :math:`h^{-1}\,\mathrm{Mpc}` for the default ``k_in``. :math:`\hat P` does not see it (the ringing sits at :math:`k \approx` ``k_in[-1]``).

``W.xi(xi, r, P, k_split)`` splits the two: ``xi`` is only :math:`P`'s part above ``k_split``, :math:`P(1-S)` with :math:`S` = ``hankel.window(k, k_split, 2 k_split)`` (1 below ``k_split``, 0 above twice it), from a converged transform, and :math:`PS` goes in through the matrix's quadrature - so every infrared constant still cancels with the IC. ``Model.xi(r, l, zz, window=W)`` does this, with ``k_split`` = 0.02 :math:`h\,\mathrm{Mpc}^{-1}` and the high part from its own transform (its options apply): it reproduces ``windowed_xi`` of ``Model.xi`` to :math:`\sim 10^{-10}` of the peak (Euclid, :math:`z = 1`). ``hankel.xi_to_pk`` transforms it to :math:`\hat P`.

If supplying all of ``xi`` yourself with an IC, use the same infrared convention as the IC's ``P``; an independently subtracted constant in :math:`\xi_0` leaves a window-shaped offset.

A Landy-Szalay :math:`\xi` divides by :math:`RR(s,\mu)`, which largely removes the window, so a standard :math:`\xi` analysis compares with the unwindowed ``Model.xi``. :math:`\hat\xi_\ell` is what an estimator normalised by the :math:`RR` monopole alone sees, and the configuration-space picture of what the power spectrum estimator measures. :math:`\hat\xi^{\rm cic}` divided by the window is the Landy-Szalay model with the integral constraint.

References
----------

- M. J. Wilson, J. A. Peacock, A. N. Taylor, S. de la Torre (2017), *Rapid modelling of the redshift-space power spectrum multipoles for a masked density field*, MNRAS 464, 3121, `arXiv:1511.07799 <https://arxiv.org/abs/1511.07799>`_
- F. Beutler et al. (2017), *The clustering of galaxies in the completed SDSS-III BOSS: anisotropic galaxy clustering in Fourier space*, MNRAS 466, 2242, `arXiv:1607.03150 <https://arxiv.org/abs/1607.03150>`_
- F. Beutler, E. Castorina, P. Zhang (2019), *Interpreting measurements of the anisotropic galaxy power spectrum*, JCAP 03 (2019) 040, `arXiv:1810.05051 <https://arxiv.org/abs/1810.05051>`_
- A. de Mattia, V. Ruhlmann-Kleider (2019), *Integral constraints in spectroscopic surveys*, JCAP 08 (2019) 036, `arXiv:1904.08851 <https://arxiv.org/abs/1904.08851>`_
- F. Beutler, P. McDonald (2021), *Unified galaxy power spectrum measurements from 6dFGS, BOSS, and eBOSS*, JCAP 11 (2021) 031, `arXiv:2106.06324 <https://arxiv.org/abs/2106.06324>`_
