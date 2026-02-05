Signal-to-Noise Ratio
=====================

Computing cumulative SNR for power spectrum and bispectrum measurements.

Single-Bin vs Full Forecast
---------------------------

CosmoWAP provides two levels of forecasting:

- **PkForecast / BkForecast**: Single redshift bin calculations
- **FullForecast**: Collates results over multiple redshift bins across the survey

For most use cases, work with ``FullForecast`` which handles the binning automatically.

Using get_fish for SNR
----------------------

The most efficient way to compute SNR is via the Fisher matrix, since ``get_fish`` precomputes and caches derivatives:

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

    # Fisher matrix gives SNR on amplitude parameter
    fisher = forecast.get_fish(
        ["A_s"],
        terms="NPP",
        pkln=[0, 2],
        bkln=[0]
    )

    # SNR = 1 / fractional error on amplitude
    snr = 1.0 / (fisher.get_error("A_s") / cosmo_funcs.A_s)

Dedicated SNR Methods
---------------------

For convenience, dedicated SNR methods are also available:

.. py:method:: FullForecast.pk_SNR(term, pkln, verbose=True, sigma=None)

   Compute power spectrum SNR per redshift bin.

   :return: Array of SNR per bin

.. py:method:: FullForecast.bk_SNR(term, bkln, verbose=True, sigma=None)

   Compute bispectrum SNR per redshift bin.

.. py:method:: FullForecast.combined_SNR(term, pkln, bkln, verbose=True, sigma=None)

   Compute combined Pk + Bk SNR.

.. code-block:: python

    # Per-bin SNR arrays
    snr_pk = forecast.pk_SNR("NPP", pkln=[0, 2])
    snr_bk = forecast.bk_SNR("NPP", bkln=[0])

    # Total SNR (sum in quadrature over bins)
    total_snr = np.sqrt(np.sum(snr_pk**2))

Single-Bin Access
-----------------

For fine-grained control, access individual redshift bins:

.. code-block:: python

    # Get single-bin forecast objects
    pk_bin0 = forecast.get_pk_bin(i=0)
    bk_bin0 = forecast.get_bk_bin(i=0)

    # Single-bin SNR
    snr_single = pk_bin0.SNR("NPP", ln=[0, 2])
