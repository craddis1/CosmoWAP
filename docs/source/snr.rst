Signal-to-Noise Ratio
=====================

Computing cumulative SNR for power spectrum and bispectrum measurements.

Single-Bin vs Full Forecast
---------------------------

CosmoWAP forecasts at two levels:

- **PkForecast / BkForecast**: Single redshift bin calculations
- **FullForecast**: Collates results over multiple redshift bins for a survey - assumes they are independent

Using get_fish for SNR
----------------------

For a parameter multiplying the signal by a unit amplitude, its one-parameter Fisher information is the squared SNR. For Newtonian Pk alone, the fractional precision on ``A_s`` gives the same result, since Pk is proportional to ``A_s`` at fixed cosmology:

.. code-block:: python

    import numpy as np
    import cosmo_wap as cw
    from cosmo_wap.lib import utils
    from cosmo_wap.forecast import FullForecast

    cosmo = utils.get_cosmo()
    survey = cw.SurveyParams.Euclid(cosmo)
    cosmo_funcs = cw.ClassWAP(cosmo, survey)

    # FullForecast splits survey redshift range into N_bins
    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Pk is proportional to A_s
    fisher = forecast.get_fish(
        ["A_s"],
        terms="NPP",
        pkln=[0, 2],
        bkln=None
    )

    # SNR = 1 / fractional error on amplitude
    snr = 1.0 / (fisher.get_error("A_s") / cosmo_funcs.A_s)

For a joint Pk + Bk forecast this amplitude interpretation differs: tree-level Bk is proportional to ``A_s**2``. At fixed covariance, ``A_s**2 * F_AsAs`` is then :math:`\mathrm{SNR}_P^2 + 4\mathrm{SNR}_B^2`, rather than the joint signal SNR squared. Use the dedicated methods below for the signal SNR.

Dedicated SNR Methods
---------------------

The dedicated methods return **squared SNR**, :math:`d^\dagger C^{-1}d`, per redshift bin, despite their names. Take a square root only after summing over independent redshift bins. The arrays may have a complex dtype; the self-contraction is real up to roundoff.

.. py:method:: FullForecast.pk_SNR(term, pkln, **kwargs)

   Compute power spectrum squared SNR per redshift bin.

   :return: Array of squared SNR per bin

.. py:method:: FullForecast.bk_SNR(term, bkln, **kwargs)

   Compute bispectrum squared SNR per redshift bin.

.. py:method:: FullForecast.combined_SNR(term, pkln, bkln, **kwargs)

   Compute joint Pk + Bk squared SNR per redshift bin, including their cross-covariance when ``cov_ng=True``.

Pass ``verbose=False`` to hide progress, ``sigma`` for FoG damping, ``cov_terms`` for the covariance kernels, and ``all_tracer=True`` for the full multi-tracer data vector. Numerical signals accept ``kernels``/``mu_grid`` for Pk and ``bk_kernels`` for Bk, as in :doc:`forecast`.

.. code-block:: python

    # Per-bin squared SNR arrays
    snr2_pk = forecast.pk_SNR("NPP", pkln=[0, 2])
    snr2_bk = forecast.bk_SNR("NPP", bkln=[0])
    snr2_joint = forecast.combined_SNR("NPP", pkln=[0, 2], bkln=[0])

    # Total SNR (sum in quadrature over bins)
    total_snr_pk = np.sqrt(np.real(np.sum(snr2_pk)))
    total_snr_joint = np.sqrt(np.real(np.sum(snr2_joint)))

Single-Bin Access
-----------------

For fine-grained control, access individual redshift bins:

.. code-block:: python

    # Get single-bin forecast objects
    pk_bin0 = forecast.get_pk_bin(i=0)
    bk_bin0 = forecast.get_bk_bin(i=0)

    # Single-bin SNR
    snr_single = np.sqrt(np.real(pk_bin0.SNR("NPP", ln=[0, 2])))
