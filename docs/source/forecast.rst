Forecasting
===========

The ``forecast`` module provides classes for Fisher matrix forecasting using power spectrum and bispectrum data.

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

   .. method:: get_fish(param_list, terms='NPP', cov_terms=None, pkln=None, bkln=None, verbose=True, sigma=None, bias_list=None, bk_terms=None, per_bin_params=None, marginalize_per_bin=True)

      Compute Fisher matrix.

      :param list param_list: Global parameters — shared across all bins (e.g., ``['fNL', 'n_s']``)
      :param str terms: Contribution terms (see :ref:`available-terms`)
      :param str bk_terms: Separate terms for the bispectrum (default: same as ``terms``)
      :param list pkln: Pk multipoles (e.g., ``[0, 2]``)
      :param list bkln: Bk multipoles (e.g., ``[0]``)
      :param bool verbose: Show progress
      :param float sigma: FoG damping
      :param bias_list: Terms for best-fit bias calculation (only evaluated against global params)
      :param list per_bin_params: Parameters that take an independent value in each redshift bin (e.g., ``['b_1']``). See :ref:`per-bin-marginalisation`.
      :param bool marginalize_per_bin: If ``True`` (default) per-bin params are marginalised out via a Schur complement and the returned Fisher covers only ``param_list``. If ``False`` the full block matrix is returned with expanded names like ``b_1[k]``.
      :return: ``FisherMat`` object

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

   .. method:: sampler(param_list, terms=None, pkln=None, bkln=None, R_stop=0.005, max_tries=100, name=None, planck_prior=False, verbose=True, sigma=None, bk_terms=None)

      Create ``Sampler`` instance for MCMC.

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

**PNG amplitudes:**

- ``fNL`` (local), ``fNL_eq`` (equilateral), ``fNL_orth`` (orthogonal)

**Survey/nuisance parameters:**

- ``A_b_1``, ``A_b_2``, ``A_be``, ``A_Q``, ``A_loc_b_01``, ``A_loc_b_01`` (bias amplitude parameters)

We can also refer to bias amplitude linked to one survey:
e.g.
- ``X_b_1``, ``Y_b_1``

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
- ``'GRI'`` -- GR + Integrated
- ``'GRL'`` -- GR + Lensing

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
