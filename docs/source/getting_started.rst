Getting Started with CosmoWAP
=============================

This guide provides examples of how to use CosmoWAP for various cosmological calculations.

Basic Setup
-----------

First, let's set up a cosmology and survey parameters:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import cosmo_wap as cw
    from cosmo_wap.lib import utils

    # Initialize CLASS with Planck-like cosmology
    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=4.0)

    # Get survey parameters for a Euclid-like survey
    survey_params = cw.SurveyParams.Euclid(cosmo)

    # Initialize ClassWAP (the main interface)
    cosmo_funcs = cw.ClassWAP(cosmo, survey_params)


Computing Power Spectra
-----------------------

Once you have a ``ClassWAP`` instance, you can compute power spectrum multipoles:

.. code-block:: python

    import cosmo_wap.pk as pk

    # Define k values and redshift
    k = np.linspace(0.01, 0.2, 50)
    z = 1.0

    # Compute the Newtonian plane-parallel monopole (l=0)
    P0 = pk.Pk0.l0(cosmo_funcs, k, zz=z)

    # Compute wide-angle corrections
    P0_WA = pk.WA1.l0(cosmo_funcs, k, zz=z)

Computing Bispectra
-------------------

Similarly for the bispectrum:

.. code-block:: python

    import cosmo_wap.bk as bk

    # Define triangle configuration
    k1, k2, k3 = 0.1, 0.1, 0.1  # equilateral
    z = 1.0

    # Compute the Newtonian monopole
    B0 = bk.GR0.l0(cosmo_funcs, k1, k2, k3=k3, zz=z)

    # Or with wide-angle corrections
    B0_WA = bk.WA1.l0(cosmo_funcs, k1, k2, k3=k3, zz=z)

Forecasting with Fisher Matrices
--------------------------------

CosmoWAP includes a full forecasting pipeline for computing Fisher matrices:

.. code-block:: python

    from cosmo_wap.forecast import FullForecast

    # Create a forecast with 4 redshift bins, kmax = 0.15 h/Mpc
    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Compute Fisher matrix for cosmological parameters
    # using Pk monopole (l=0) and quadrupole (l=2)
    fisher = forecast.get_fish(
        ["A_s", "n_s", "h"],
        terms="NPP",          # Newtonian plane-parallel
        pkln=[0, 2],          # Pk multipoles
        bkln=None,            # No bispectrum
        verbose=True
    )

    # Access results
    print("1-sigma errors:", fisher.errors)
    print("Correlation matrix:\n", fisher.correlation)

    # Get error on a specific parameter
    sigma_ns = fisher.get_error("n_s")

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
