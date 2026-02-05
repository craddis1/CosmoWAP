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

   .. method:: get_fish(param_list, terms='NPP', cov_terms=None, pkln=None, bkln=None, verbose=True, sigma=None, bias_list=None)

      Compute Fisher matrix.

      :param list param_list: Parameters (e.g., ``['A_s', 'n_s', 'h', 'Omega_m']``)
      :param str terms: Contributions (``'NPP'``, ``'WS'``, ``'GR'``, ``'PNG'``)
      :param list pkln: Pk multipoles (e.g., ``[0, 2]``)
      :param list bkln: Bk multipoles (e.g., ``[0]``)
      :param bool verbose: Show progress
      :param float sigma: FoG damping
      :param bias_list: Terms for best-fit bias calculation
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

   .. method:: sampler(param_list, terms=None, pkln=None, bkln=None, R_stop=0.005, max_tries=100, name=None, planck_prior=False, verbose=True, sigma=None)

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
        ["A_s", "n_s", "h"],
        terms="NPP",
        pkln=[0, 2],
        bkln=None
    )

    # Combined Pk + Bk
    fisher_combined = forecast.get_fish(
        ["A_s", "n_s"],
        terms="NPP",
        pkln=[0, 2],
        bkln=[0]
    )

Available Parameters
~~~~~~~~~~~~~~~~~~~~

**Cosmological parameters:**

- ``A_s``, ``n_s``, ``h``, ``Omega_m``, ``Omega_cdm``, ``Omega_b``, ``sigma8``

**PNG amplitudes:**

- ``fNL`` (local), ``fNL_eq`` (equilateral), ``fNL_orth`` (orthogonal)

**Survey/nuisance parameters:**

- ``b_1``, ``b_2``, ``be``, ``Q`` (bias parameters)

**Terms options:**

- ``'NPP'``: Newtonian plane-parallel (Kaiser)
- ``'WS'``: Wide-separation (WA + RR)
- ``'GR1'``, ``'GR2'``: Relativistic corrections
- ``'Loc'``, ``'Eq'``, ``'Orth'``: PNG contributions

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
        ["A_s", "n_s", "fNL"],
        terms=["NPP", "Loc"],
        pkln=[0, 2]
    )

PNG Forecasting
~~~~~~~~~~~~~~~

.. code-block:: python

    # Include local PNG contribution
    fisher_png = forecast.get_fish(
        ["A_s", "n_s", "fNL"],
        terms=["NPP", "Loc"],  # Include PNG
        pkln=[0, 2],
        bkln=[0]
    )

    print(f"sigma(fNL) = {fisher_png.get_error('fNL'):.2f}")

MCMC Sampling
~~~~~~~~~~~~~

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

   **Methods:**

   .. method:: get_error(param)

      Get 1-sigma error for parameter.

   .. method:: get_correlation(param1, param2)

      Get correlation coefficient.

   .. method:: summary()

      Print formatted summary.

   .. method:: add_chain(c=None, bias_values=None, name=None)

      Add as chain to ChainConsumer for plotting.

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

    fisher = forecast.get_fish(["A_s", "n_s", "h"], terms="NPP", pkln=[0, 2])

    fisher.summary()
    print(fisher.get_error("n_s"))
    print(fisher.get_correlation("A_s", "n_s"))

    # Plot with ChainConsumer
    c = fisher.add_chain(name="Pk only")
    c.plotter.plot()

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
