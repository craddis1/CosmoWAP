Forecasting
===========

The ``forecast`` module provides classes for Fisher matrix forecasting using power spectrum and bispectrum data.
FullForecast is the main class for survey level forecast after initiating we can use it to computes Fisher matrices, SNR, best-fit bias and create MCMC samplers.
The resulting posteriors can be plotted with the built-in ChainConsumer interface.

FullForecast
------------

.. py:class:: forecast.FullForecast(cosmo_funcs, kmax_func=None, s_k=2, nonlin=False, N_bins=None, bkmax_func=None, WS_cut=True, n_mu=8, n_phi=8)

   Main class for full survey forecasts over redshift bins.

   **Parameters:**

   - **cosmo_funcs**: ``ClassWAP`` instance
   - **kmax_func**: Maximum k (float or callable ``kmax_func(z)``)
   - **s_k**: k-bins per decade (default: 2)
   - **nonlin**: Use HALOFIT power spectra (default: False)
   - **N_bins**: Number of redshift bins (default: auto)
   - **bkmax_func**: Separate kmax for bispectrum (default: same as kmax_func)
   - **WS_cut**: Apply wide-separation validity cuts (default: True)
   - **n_mu**, **n_phi**: Angular integration points for covariances

   **Attributes:**

   - **z_bins**: Redshift bin edges, shape ``(N_bins, 2)``
   - **z_mid**: Bin centre redshifts
   - **k_max_list**: Maximum k per bin
   - **N_bins**: Number of bins

   **Methods:**

   .. method:: get_fish(param_list, terms='NPP', cov_terms=None, pkln=None, bkln=None, verbose=True, sigma=None, bias_list=None, bk_bias_list=None, bk_terms=None, bk_st=False, per_bin_params=None, marginalize_per_bin=True, kernels=None, mu_grid=None)

      Compute Fisher matrix.

      :param list param_list: Global parameters — shared across all bins (e.g., ``['fNL', 'n_s']``)
      :param str terms: Contribution terms (see :ref:`available-terms`)
      :param list kernels: Numeric-:math:`\mu` kernel names (``'N'``, ``'LP'``, ``'I'``, or the finer ``'L'``/``'TD'``/``'ISW'``/``'kappa_g'``) summed onto ``terms``, computed via the fast kernel path — one :math:`P(k,\mu)` per tracer combination, projected onto each multipole. E.g. ``terms=None, kernels=['N','LP','I']`` replaces the analytic NPP/GR/IntNPP/IntInt terms. Pk-only — the bispectrum is unaffected. See :doc:`integrated`.
      :param list mu_grid: ``[n_mu, GL, los_n, deg]`` controlling the numeric-:math:`\mu` grid used by ``kernels`` (default: ``[256, False, 32, 8]``)
      :param str bk_terms: Separate terms for the bispectrum (default: same as ``terms``)
      :param bool bk_st: Force bispectrum onto single-tracer pipeline using ``cosmo_funcs.survey[0]`` (no-op when not multi-tracer). Pk side is unaffected.
      :param list pkln: Pk multipoles (e.g., ``[0, 2]``)
      :param list bkln: Bk multipoles (e.g., ``[0]``)
      :param bool verbose: Show progress
      :param float sigma: FoG damping
      :param bias_list: Terms for best-fit bias calculation (only evaluated against global params)
      :param bk_bias_list: Override ``bias_list`` on the bispectrum side. When set, both lists are collapsed to a single composite (sum) and one combined bfb is returned.
      :param list per_bin_params: Parameters that take an independent value in each redshift bin (e.g., ``['b_1']``). See :ref:`per-bin-marginalisation`.
      :param bool marginalize_per_bin: If ``True`` (default) per-bin params are marginalised out via a Schur complement and the returned Fisher covers only ``param_list``. If ``False`` the full block matrix is returned with expanded names like ``b_1[k]``.
      :param bool precondition: Use diagonal preconditioning when inverting Fisher blocks (default: ``True``). Improves numerical stability for multi-tracer per-bin setups where parameter scales span many orders of magnitude.
      :return: ``FisherMat`` object

   .. method:: get_cumulative_fish(param_list, per_bin_params=None, terms='NPP', cov_terms=None, pkln=None, bkln=None, verbose=True, sigma=None, bk_terms=None, bk_st=False, precondition=True, cumulative=True)

      Per-bin marginalised Fisher: returns a list of ``FisherMat``, one per redshift bin, with ``per_bin_params`` marginalised out in each bin. With ``cumulative=True`` (default) entry ``k`` uses bins ``0..k`` and the final entry matches ``get_fish(..., marginalize_per_bin=True)``; with ``cumulative=False`` entry ``k`` uses bin ``k`` alone. See :ref:`per-bin-marginalisation`.

      :return: ``list[FisherMat]`` of length ``N_bins``

   .. method:: pk_SNR(term, pkln, verbose=True, sigma=None)

      Compute power spectrum SNR per redshift bin.

   .. method:: bk_SNR(term, bkln, verbose=True, sigma=None)

      Compute bispectrum SNR per redshift bin.

   .. method:: combined_SNR(term, pkln, bkln, verbose=True, sigma=None)

      Compute combined Pk + Bk SNR.

   .. method:: best_fit_bias(param, bias_term, terms='NPP', pkln=None, bkln=None, verbose=True, sigma=None)

      Compute parameter bias from neglecting a contribution.

      :return: Tuple of (bias_dict, fisher_diagonal)

   .. method:: get_pk_bin(i=0)

      Get ``PkForecast`` for redshift bin ``i``.

   .. method:: get_bk_bin(i=0)

      Get ``BkForecast`` for redshift bin ``i``.

   .. method:: sampler(param_list, terms=None, pkln=None, bkln=None, R_stop=0.005, max_tries=100, name=None, planck_prior=False, verbose=True, sigma=None, bias_list=None, bk_bias_list=None, bk_terms=None, bk_st=False, per_bin_params=None, fisher_covmat=True, drag=True, kernels=None)

      Create ``Sampler`` instance for MCMC.

      :param list kernels: Numeric-:math:`\mu` kernels summed onto ``terms``, as in ``get_fish``. With ``terms=None`` the signal comes entirely from the kernels (requires ``bkln=None`` or analytic ``bk_terms``, since ``kernels`` supplies no bispectrum). See :doc:`integrated`.

      :param list per_bin_params: Nuisance parameters (e.g. ``['b_1', 'Q', 'be']``) sampled independently per redshift bin and marginalised over. Each is expanded to one multiplicative amplitude per bin (``b_1_0``, ``b_1_1``, …, ref 1.0) applied to that bin's theory only. ``b_1`` gets a tight prior (0.8–1.2); selection functions like ``Q``/``be`` inherit the wide prior of their global ``A_Q``/``A_be`` amplitudes. For multi-tracer forecasts the tracer-prefixed names (``Xb_1``, ``YQ``, …) scale only that tracer's bias, while the bare names scale both tracers together — same convention as ``get_fish`` (see :ref:`per-bin params <per-bin-marginalisation>`).
      :param bool fisher_covmat: Seed cobaya's proposal with the inverse-Fisher covariance over the global params (default: ``True``), giving the chains the correct degenerate correlation structure from the start. Per-bin nuisance params are Schur-marginalised out of this proposal (the per-bin entries themselves fall back to their proposal widths). Falls back to proposal widths entirely if the Fisher is singular.
      :param bool drag: Use cobaya's fast/slow dragging (default: ``True``). Cosmological parameters form the slow block and all other sampled parameters (e.g. ``fNL``, bias and per-bin amplitudes) the fast block; the fast-block oversampling factor is measured automatically in ``run()``. See :ref:`fast-slow-dragging`.

Usage
~~~~~

.. code-block:: python

    import cosmo_wap as cw
    from cosmo_wap.lib import utils
    from cosmo_wap.forecast import FullForecast

    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31)
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Fisher from Pk monopole + quadrupole
    fisher = forecast.get_fish(
        ["fNL_loc", "A_b_1", "A_Q", "A_be"],
        terms=["NPP", "Loc", "GR2", "IntNPP"],
        pkln=[0, 2],
        bkln=None
    )

    # Combined Pk + Bk
    fisher_combined = forecast.get_fish(
        ["fNL_loc", "A_b_1", "A_Q", "A_be"],
        terms=["NPP", "Loc", "GR2", "IntNPP"],
        pkln=[0, 2],
        bkln=[0]
    )

Available Parameters
~~~~~~~~~~~~~~~~~~~~

We can forecast over the core cosmological parameter as well as our bias parameters or simply some specific contribution

**Cosmological parameters:**

- ``A_s``, ``n_s``, ``h``, ``Omega_m``, ``Omega_cdm``, ``Omega_b``, ``sigma8``, ``w0``, ``wa``
- ``ln_A_s`` — :math:`\ln(10^{10}A_s)`, a drop-in replacement for ``A_s`` that is better conditioned numerically (use one or the other).

**Derived cosmological parameters:**

- ``gamma`` — growth index, :math:`f(z) = \Omega_m(z)^\gamma`. Fiducial is inferred from the fiducial :math:`f(z_{\rm mid})` and :math:`\Omega_m(z_{\rm mid})` (typically :math:`\approx 0.55`). When included, :math:`f(z)` is replaced by the parametric form and :math:`f_d`, :math:`f_{dd}` are rebuilt; :math:`D(z)` is left unchanged.
- ``S8`` — :math:`S_8 \equiv \sigma_8\sqrt{\Omega_m/0.3}`. Not a stencil parameter; obtained post-hoc from a Fisher run with both ``sigma8`` and ``Omega_m`` via :py:meth:`FisherMat.to_S8`.

**PNG amplitudes:**

- ``fNL`` (local), ``fNL_eq`` (equilateral), ``fNL_orth`` (orthogonal). ``fNL_loc`` is accepted as an alias for ``fNL``.

**Survey/nuisance parameters:**

- ``A_b_1``, ``A_b_2``, ``A_be``, ``A_Q``, ``A_loc_b_01``, ``A_loc_b_11`` (bias amplitude parameters)

We can also refer to bias amplitude linked to one survey:
e.g.
- ``X_b_1``, ``Y_b_1``

**Per-tracer biases (for** ``per_bin_params`` **):**

- ``Xb_1``, ``Yb_1``, ``Xb_2``, ``Yb_2``, ``Xbe``, ``Ybe``, ``XQ``, ``YQ``, ``Xg_2``, ``Yg_2``

  Restricted to a single tracer (``X`` or ``Y``); the bare names (``b_1``, ``be``, …) modify both tracers together. Use these in ``per_bin_params`` for multi-tracer per-bin marginalisation — in the Fisher they are additive derivatives wrt the bias itself (not its amplitude), in the sampler they become per-bin multiplicative amplitudes (``Xb_1_0``, ``YQ_1``, …).

.. _available-terms:

Available Terms
~~~~~~~~~~~~~~~

Terms can be passed as a single string (e.g. ``terms='NPP'``) or a list (e.g. ``terms=['NPP', 'Loc']``). The full list of available terms is:

**Base:**

- ``'NPP'`` -- Newtonian plane-parallel (Kaiser RSD)

**Wide-separation (WS):**

- ``'WS'`` -- Combined wide-separation (WA + RR)
- ``'WA1'`` -- Wide-angle first order
- ``'WA2'`` -- Wide-angle second order
- ``'RR1'`` -- Radial redshift first order
- ``'RR2'`` -- Radial redshift second order

**Relativistic (GR):**

- ``'GR'`` -- All relativistic corrections
- ``'GR1'`` -- First order relativistic
- ``'GR2'`` -- Second order relativistic
- ``'GRX'`` -- Cross relativistic terms

**Combined WS + GR:**

- ``'WAGR'`` -- Wide-angle + GR
- ``'WARR'`` -- Wide-angle + Radial redshift
- ``'RRGR'`` -- Radial redshift + GR
- ``'WSGR'`` -- Wide-separation + GR
- ``'Full'`` -- All terms

**Primordial non-Gaussianity (PNG):**

- ``'Loc'`` -- Local PNG
- ``'Eq'`` -- Equilateral PNG
- ``'Orth'`` -- Orthogonal PNG

**Integrated effects (Pk only):**

- ``'IntNPP'`` -- Integrated x NPP
- ``'IntInt'`` -- Integrated x Integrated
- ``'GRI'`` -- Integrated relativistic terms (IntNPP + IntInt)
- ``'GRL'`` -- Local relativistic terms (GR1 + GR2)

**Individual integrated components:**

- ``'LxNPP'`` -- Lensing x NPP
- ``'ISWxNPP'`` -- ISW x NPP
- ``'TDxNPP'`` -- Time-delay x NPP
- ``'LxL'`` -- Lensing x Lensing
- ``'LxTD'`` -- Lensing x Time-delay
- ``'LxISW'`` -- Lensing x ISW
- ``'ISWxISW'`` -- ISW x ISW
- ``'ISWxTD'`` -- ISW x Time-delay
- ``'TDxTD'`` -- Time-delay x Time-delay

**Numeric-** :math:`\mu` **kernels (via** ``kernels`` **):**

In addition to the analytic terms above, the power spectrum signal can be built from the numeric-:math:`\mu` kernels (``'N'``, ``'LP'``, ``'I'``, ``'L'``, ``'TD'``, ``'ISW'``, ``'kappa_g'``) passed as ``kernels`` to ``get_fish``/``sampler`` — much faster when integrated effects are included. See :doc:`integrated` for the kernel definitions and usage.

Multi-Tracer Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Bright-faint split for multi-tracer
    survey = cw.SurveyParams.Euclid(cosmo)
    bright, faint = survey.BF_split(5e-16)
    cosmo_funcs_mt = cw.ClassWAP(cosmo, [bright, faint])

    forecast_mt = FullForecast(cosmo_funcs_mt, kmax_func=0.15, N_bins=4)

    # Multi-tracer Fisher (accounts for cross-correlations)
    fisher_mt = forecast_mt.get_fish(
        ["fNL_loc", "A_b_1", "A_Q", "A_be"],
        terms=["NPP", "Loc", "GR2", "IntNPP"],
        pkln=[0, 2]
    )

.. _per-bin-marginalisation:

Per-Bin Marginalisation
~~~~~~~~~~~~~~~~~~~~~~~

Some nuisance parameters (typically galaxy bias) are not shared between redshift bins — each bin has its own independent value. ``get_fish`` handles this via the ``per_bin_params`` argument, which expands the named parameters internally to one copy per bin and marginalises them out of the global Fisher.

**Block structure.** Calling ``get_fish`` with ``per_bin_params=["b_1"]`` builds an extended Fisher matrix with a natural block structure: a dense global block ``F_AA``, block-diagonal per-bin blocks ``F_BB[k]`` (zero between different bins by construction), and cross blocks ``F_AB[k]``:

.. math::

   F = \begin{pmatrix} F_{AA} & F_{AB}^{0} & F_{AB}^{1} & \cdots \\
                       F_{AB}^{0\,T} & F_{BB}^{0} & 0 & \cdots \\
                       F_{AB}^{1\,T} & 0 & F_{BB}^{1} & \cdots \\
                       \vdots & \vdots & \vdots & \ddots \end{pmatrix}

**Two paths, same maths:**

- **Schur complement** (default, ``marginalize_per_bin=True``): marginalises the per-bin block analytically,

  .. math::

     F_{AA}^{\mathrm{marg}} = F_{AA} - \sum_k F_{AB}^{k}\,(F_{BB}^{k})^{-1}\,(F_{AB}^{k})^{T}

  and returns an ``(N_A × N_A)`` ``FisherMat`` on just the global parameters. Each small ``F_BB[k]`` inversion is stored as a byproduct for nuisance diagnostics (see :py:meth:`FisherMat.get_per_bin_error`).

- **Full matrix** (``marginalize_per_bin=False``): builds the full ``(N_A + N_B × N_bins)`` block matrix, with expanded parameter names like ``b_1[0], b_1[1], ...``. Useful for inspecting cross-bin correlations.

Both paths give mathematically identical global-parameter errors; Schur is faster and numerically more stable, so is preferred unless you specifically need to inspect the full matrix.

``bias_list`` contributions are only computed against the global parameters; per-bin nuisance params are ignored in the bias calculation.

.. code-block:: python

    # fNL marginalised over independent per-bin b_1
    fish = forecast.get_fish(
        param_list=["fNL", "n_s"],
        per_bin_params=["b_1"],
        terms="NPP",
        pkln=[0, 2],
    )
    print(fish.get_error("fNL"))
    print(fish.get_per_bin_error("b_1"))   # array length N_bins

    # Same calculation, full matrix kept for inspection
    fish_full = forecast.get_fish(
        param_list=["fNL", "n_s"],
        per_bin_params=["b_1"],
        terms="NPP",
        pkln=[0, 2],
        marginalize_per_bin=False,
    )
    print(fish_full.param_list)            # ['fNL', 'n_s', 'b_1[0]', 'b_1[1]', ...]
    print(fish_full.get_error("b_1[3]"))   # marginalised error on b_1 in bin 3
    print(fish_full.get_per_bin_error("b_1"))  # same but as array

Cumulative constraints
^^^^^^^^^^^^^^^^^^^^^^

To see how the constraint on a global parameter tightens as redshift bins are added,
:py:meth:`FullForecast.get_cumulative_fish` returns one ``FisherMat`` per cumulative cut:
entry ``k`` uses bins ``0..k`` (ascending redshift), with ``per_bin_params`` marginalised out
within each bin. The final entry is identical to ``get_fish(..., marginalize_per_bin=True)``.
This is exact and cheap — the Schur-marginalised Fisher is additive over bins, so the per-bin
blocks are simply accumulated rather than summed in one go.

.. code-block:: python

    # cumulative sigma(fNL) marginalised over independent per-bin bias
    fishers = forecast.get_cumulative_fish(
        "fNL",
        per_bin_params=["b_1", "b_2", "g_2"],
        terms="NPP",
        pkln=[0, 2],
    )

    sigma_fNL = [f.get_error("fNL") for f in fishers]  # length N_bins, non-increasing
    plt.plot(forecast.z_bins[:, 1], sigma_fNL)         # constraint vs max redshift

PNG Forecasting
~~~~~~~~~~~~~~~

.. code-block:: python

    # Include local PNG contribution
    fisher_png = forecast.get_fish(
        ["fNL_loc", "A_b_1", "A_Q", "A_be"],
        terms=["NPP", "Loc", "GR2", "IntNPP"],
        pkln=[0, 2],
        bkln=[0]
    )

    print(f"sigma(fNL_loc) = {fisher_png.get_error('fNL_loc'):.2f}")

MCMC Sampling
~~~~~~~~~~~~~

.. note::

   For sampling over cosmology we recommend using the CosmoPower extension. See :doc:`install` for setup instructions.

.. code-block:: python

    # Create sampler (requires CosmoPower)
    sampler = forecast.sampler(
        ["A_s", "n_s", "h"],
        terms="NPP",
        pkln=[0, 2],
        R_stop=0.01  # Convergence criterion
    )

    # Run MCMC
    sampler.run()

    # Plot posteriors
    sampler.plot()

By default the sampler seeds cobaya's proposal with the inverse-Fisher covariance
(``fisher_covmat=True``), so the chains start with the correct degenerate correlation
structure rather than a diagonal guess — this greatly reduces stuck chains for tightly
constrained, strongly degenerate posteriors. With ``planck_prior=True`` the Planck prior is
also added to this Fisher so the proposal matches the constrained posterior. Pass
``per_bin_params=['b_1']`` to marginalise over an independent ``b_1`` amplitude in each
redshift bin (expanded to ``b_1_0``, ``b_1_1``, …); the global block of the proposal covmat is
Schur-marginalised over these, while the per-bin amplitudes themselves are left to their
proposal widths. For multi-tracer runs the tracer-prefixed names (``Xb_1``, ``YQ``, …) sample
an independent amplitude per bin for that tracer only. Note a per-bin ``Q``/``be`` amplitude
only constrains anything when a term that depends on it (e.g. ``GR2``) is included in
``terms``. For long runs, raising ``max_tries`` (e.g.
``max_tries=10000``) prevents a transient stuck chain from tearing down an MPI run.

.. _fast-slow-dragging:

Fast/slow dragging
^^^^^^^^^^^^^^^^^^

With ``drag=True`` (default) the sampler uses cobaya's fast/slow dragging
(Neal 2005, arXiv:math/0502099; Lewis 2013, arXiv:1304.4473). Parameters are split into a
slow block — cosmological parameters, which rebuild the ``ClassWAP`` cosmology each step —
and a fast block of all other sampled parameters (e.g. ``fNL``, bias and per-bin amplitudes).
Many cheap fast steps are interpolated ("dragged") between each expensive cosmology
evaluation, improving mixing of the fast directions. The cosmology is cached and reused across the fast steps,
and the fast-block oversampling factor is set from the measured slow/fast cost ratio at the
start of ``run()`` (dragging is skipped when the ratio is below 2). Set ``drag=False`` to
sample all parameters in a single block.

FisherMat
---------

.. py:class:: forecast.FisherMat

   Stores Fisher matrix results.

   **Attributes:**

   - **fisher_matrix**: Fisher information matrix (numpy array)
   - **covariance**: Inverse Fisher (parameter covariance)
   - **errors**: 1-sigma marginalised errors
   - **correlation**: Correlation matrix
   - **param_list**: Parameter names
   - **bias**: Best-fit bias values (if computed)
   - **per_bin_cov**: Per-bin nuisance covariance stack, shape ``(N_bins, N_B, N_B)``. Set by the Schur path of :py:meth:`FullForecast.get_fish` when ``per_bin_params`` is provided; otherwise ``None``.
   - **per_bin_param_list**: Base names of per-bin parameters (e.g. ``['b_1']``). Set on both Schur and full-matrix paths when ``per_bin_params`` is provided.

   **Methods:**

   .. method:: get_error(param)

      Get 1-sigma error for parameter.

   .. method:: get_per_bin_error(param)

      Return array of per-bin 1-sigma errors for a per-bin parameter, shape ``(N_bins,)``. Behaviour depends on how the Fisher matrix was built:

      - Schur path (``marginalize_per_bin=True``): returns ``sqrt(diag(inv(F_BB[k])))``, i.e. errors **conditional on global params held fixed** at their fiducial values. Intended as a diagnostic for nuisance constraints only.
      - Full-matrix path (``marginalize_per_bin=False``): returns fully-marginalised errors read from the inverted full Fisher.

   .. method:: reduce(params)

      Extract marginalised covariance submatrix for a subset of parameters.

      :param list params: Subset of ``param_list`` to keep.
      :return: Reduced covariance matrix (numpy array).

   .. method:: to_S8()

      Rotate Fisher from :math:`(\sigma_8, \Omega_m, \ldots)` to :math:`(S_8, \Omega_m, \ldots)` via :math:`F' = J^\top F\,J`, evaluated at the fiducial cosmology. Requires both ``sigma8`` and ``Omega_m`` in ``param_list``; other parameters get identity rows/columns in :math:`J`. Returns a new ``FisherMat`` with ``sigma8`` replaced by ``S8``.

   .. method:: add_planck_prior()

      Return a new ``FisherMat`` with Planck CMB constraints added as a Gaussian prior,
      :math:`F' = F + C_{\rm Planck}^{-1}`. Only the parameters present in both
      ``param_list`` and the Planck set ``{Omega_b, Omega_cdm, theta, tau, A_s, n_s}``
      are affected; all others are unchanged. ``ln_A_s`` is accepted in place of ``A_s``
      (the prior is built in the matching amplitude's units). Returns ``self`` unchanged if none overlap.

   .. method:: get_correlation(param1, param2)

      Get correlation coefficient.

   .. method:: summary()

      Print formatted summary.

   .. method:: add_chain(c=None, bias_values=None, name=None, param_list=None)

      Add as chain to ChainConsumer for plotting. If ``param_list`` is provided,
      reduces the covariance to that subset of parameters.

   .. method:: plot_errors(relative=False, figsize=(8, 6))

      Plot parameter errors as bar chart.

   .. method:: plot_1D(param, ci=0.68, ax=None, shade=True, color='royalblue')

      Plot 1D Gaussian posterior.

   .. method:: save(filename)

      Save to ``.npz`` file.

   .. classmethod:: load(filename)

      Load from file.

Usage
~~~~~

.. code-block:: python

    fisher = forecast.get_fish(
        ["fNL_loc", "A_b_1", "A_Q", "A_be"],
        terms=["NPP", "Loc", "GR2", "IntNPP"],
        pkln=[0, 2]
    )

    fisher.summary()
    print(fisher.get_error("fNL_loc"))
    print(fisher.get_correlation("fNL_loc", "A_b_1"))

    # Add Planck CMB prior to tighten cosmological parameters post-hoc
    fisher_planck = fisher.add_planck_prior()
    print(fisher_planck.get_error("A_s"))   # tighter than fisher.get_error("A_s")

    # Plot with ChainConsumer
    c = fisher.add_chain(name="Pk only")

    # Plot a reduced subset of parameters alongside the full set
    c = fisher.add_chain(c=c, param_list=["fNL_loc", "A_b_1"], name="reduced")
    fig, c = fisher.corner_plot(c=c)

FisherList
----------

.. py:class:: forecast.FisherList

   Container for multiple Fisher matrices (e.g., varying flux cuts/splits).

   Created via ``FullForecast.get_fish_list()``.

PkForecast / BkForecast
-----------------------

.. py:class:: forecast.PkForecast
.. py:class:: forecast.BkForecast

   Single-bin forecast classes. Usually accessed via ``FullForecast.get_pk_bin()`` / ``get_bk_bin()``.

   **Methods:**

   .. method:: get_data_vector(terms, ln, param=None)

      Get data vector (or derivative w.r.t. param).

   .. method:: get_cov_mat(ln, sigma=None)

      Get covariance matrix.

   .. method:: SNR(term, ln, param=None)

      Compute SNR.

Sampler
-------

.. py:class:: forecast.Sampler

   MCMC sampler using cobaya. Created via ``FullForecast.sampler()``.
   Requires CosmoPower for efficient cosmology evaluation.

   **Methods:**

   .. method:: run()

      Run MCMC chains.

   .. method:: add_chain(c=None, name=None, bins=12, param_list=None)

      Add MCMC samples as a chain to ChainConsumer. If ``param_list`` is provided,
      only includes that subset of parameters.

   .. method:: plot(extents=None, truth=True)

      Plot posteriors with ChainConsumer.

   .. method:: save(filename)

      Save chains to file.

Usage
~~~~~

.. code-block:: python

    # Create sampler
    sampler = forecast.sampler(
        ["A_s", "n_s"],
        terms="NPP",
        pkln=[0, 2],
        R_stop=0.005,      # Gelman-Rubin convergence
        planck_prior=True  # Add Planck prior
    )

    # Run chains
    sampler.run()

    # Plot with fiducial values marked
    sampler.plot(truth=True)
