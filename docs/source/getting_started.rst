Getting Started with CosmoWAP
=============================

This guide provides examples of how to use CosmoWAP for various cosmological calculations.

Code Structure
--------------

.. code-block:: text

    ┌──────────────┐
    │    cosmo      │
    │   (CLASS)     │──┐
    └──────────────┘  │    ┌──────────────┐      ┌───────────────┐
                      ├──▶ │   ClassWAP   │─────▶│ Power Spectrum│
    ┌──────────────┐  │    │              │      │  & Bispectrum │
    │ SurveyParams │──┘    └──────┬───────┘      │  multipoles   │
    │ (Euclid,     │              │              └───────────────┘
    │  Roman, ...) │              ▼
    └──────┬───────┘   ┌─────────────────────┐
           ▲
    ┌──────┴───────┐
    │ Luminosity   │
    │ functions    │
    └──────────────┘
                       │    FullForecast      │
                       │  ┌───────────────┐  │
                       │  │   Forecast     │  │
                       │  │  (per z-bin)   │  │
                       │  └───────────────┘  │
                       └──────────┬──────────┘
                                  │
                        ┌─────────┴─────────┐
                        ▼                   ▼
                 ┌────────────┐     ┌─────────────┐
                 │ FisherMat  │     │   Sampler   │
                 │            │     │   (MCMC)    │
                 └────────────┘     └─────────────┘

A ``cosmo`` instance (from CLASS) and :doc:`SurveyParams <surveyparams>` are passed into
:doc:`ClassWAP <classwap>`, which provides the interface for computing power spectrum and bispectrum
multipoles. For parameter constraints, pass a ``ClassWAP`` instance into
:doc:`FullForecast <forecast>`, which runs a ``Forecast`` per redshift bin and returns either a
``FisherMat`` (analytic Fisher matrix) or a ``Sampler`` (MCMC via Cobaya).

Basic Setup
-----------

First, let's set up a cosmology and survey parameters and initialize the main class (``ClassWAP``):

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    # Initialize CLASS with Planck 2018 cosmology (default)
    # Can pass: h, Omega_m, Omega_b, A_s, n_s, sigma8, k_max, z_max
    cosmo = utils.get_cosmo()

    # Get survey parameters for Euclid Hα survey (0.9 < z < 1.8)
    survey_params = cw.SurveyParams.Euclid(cosmo)

    # Initialize ClassWAP (the main interface)
    cosmo_funcs = cw.ClassWAP(cosmo, survey_params)


Computing Power Spectra
-----------------------

Once you have a ``ClassWAP`` instance, you can very simply compute the power spectrum multipoles:

.. code-block:: python

    import cosmo_wap.pk as pk

    # Define k values and redshift
    k = np.linspace(0.01, 0.2, 50)
    z = 1.0

    # Compute the Newtonian plane-parallel monopole (l=0)
    P_0 = pk.NPP.l0(cosmo_funcs, k, zz=z)

    # Compute wide-separation corrections to the monopole
    P_0_WA = pk.WS.l0(cosmo_funcs, k, zz=z)

    # and we can plot contributions
    plt.figure(figsize=(8, 5))
    plt.plot(k, P_0, label='Kaiser')
    plt.plot(k, P_0_WA, label='WS')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P_0(k) [(Mpc/h)^3]')
    plt.legend()

Computing Bispectra
-------------------

Similarly for the bispectrum:

.. code-block:: python

    import cosmo_wap.bk as bk

    # Define triangle configuration
    k1, k2, k3 = 0.1, 0.1, 0.1  # equilateral
    z = 1.0

    # Compute the Newtonian monopole
    B0 = bk.NPP.l0(cosmo_funcs, k1, k2, k3=k3, zz=z)

    # Second-order relativistic corrections O((H/k)^2)
    B0_GR = bk.GR2.l0(cosmo_funcs, k1, k2, k3=k3, zz=z)

Forecasting with Fisher Matrices
--------------------------------

CosmoWAP includes a full forecasting pipeline for SNRs, Fisher matrices and MCMCs

Lets run through a basic example!

.. code-block:: python

    from cosmo_wap.forecast import FullForecast

    # so we are using a Euclid-like survey with Euclid range (0.9,1.8)

    # Create a forecast with 4 redshift bins, kmax = 0.15 h/Mpc
    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Compute Fisher matrix for cosmological parameters
    # using Pk monopole (l=0) and quadrupole (l=2)
    fisher = forecast.get_fish(
        ["fNL","A_s", "n_s"],
        terms="NPP",          # Newtonian plane-parallel
        pkln=[0, 2],          # Pk multipoles
        bkln=None,            # No bispectrum multipoles
        verbose=True
    )

    # Access results
    print("1-sigma errors:", fisher.errors)
    print("Correlation matrix:\n", fisher.correlation)

    # Get error on a specific parameter
    sigma_ns = fisher.get_error("fNL")

See the :doc:`Forecasting <forecast>` page for more details.

Example Notebooks
=================

See example notebooks in the ``examples/`` directory for usage of CosmoWAP:

- ``example_pk.ipynb`` - Power spectrum calculations
- ``example_bk.ipynb`` - Bispectrum calculations

..
   Binder links (temporarily disabled - notebooks being updated):

   .. image:: https://img.shields.io/badge/launch-powerspectrum-F5A252.svg
      :target: https://mybinder.org/v2/gh/craddis1/CosmoWAP_binder/HEAD?labpath=example_pk.ipynb

   .. image:: https://img.shields.io/badge/launch-bispectrum-F5A252.svg
      :target: https://mybinder.org/v2/gh/craddis1/CosmoWAP_binder/HEAD?labpath=example_bk.ipynb
