Getting Started with CosmoWAP
=============================

This guide provides examples of how to use CosmoWAP for various cosmological calculations.

Code Structure
--------------

Very roughly, the code structure is as follows:

.. code-block:: text

                              ┌────────────────┐
                              │     cosmo      │
                              │    (CLASS)     │
                              └───────┬────────┘
                                      │
    ┌────────────────┐        ╔═══════╧════════╗       ┌─────────────────┐
    │  SurveyParams  │───────▶║    ClassWAP    ║──────▶│      Model      │
    │                │        ╚═══════╤════════╝       │  P(k), ξ(r), B  │
    └───────┬────────┘                │                │   multipoles    │
            ▲                         │                └─────────────────┘
    ┌───────┴────────┐                │
    │   Luminosity   │                ▼
    │   functions    │     ┌──────────┴──────────┐
    └────────────────┘     │    FullForecast     │
                           │  ┌───────────────┐  │
                           │  │   Forecast    │  │
                           │  │  (per z-bin)  │  │
                           │  └───────────────┘  │
                           └──────────┬──────────┘
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                     ┌─────────────┐     ┌─────────────┐
                     │  FisherMat  │     │   Sampler   │
                     └─────────────┘     │    (MCMC)   │
                                         └─────────────┘

A ``cosmo`` instance (from CLASS) and :doc:`SurveyParams <surveyparams>` (with optional input from
:doc:`luminosity functions <luminosityfuncs>`) are passed into :doc:`ClassWAP <classwap>`, the
central interface that holds the cosmology and biases. A :doc:`Model <model>` built on it computes the
power spectrum, correlation function and bispectrum multipoles of a chosen signal. For parameter constraints,
pass a ``ClassWAP`` instance into :doc:`FullForecast <forecast>`, which runs a ``Forecast`` per
redshift bin and returns either a ``FisherMat`` (analytic Fisher matrix) or a ``Sampler``
(MCMC via Cobaya).

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

Once you have a ``ClassWAP`` instance, you can very simply compute the power spectrum multipoles with a :doc:`Model <model>`. The signal is built from the kernels of each field (``'N'`` Newtonian, ``'LP'`` local relativistic, ``'I'`` integrated, ``'Loc'`` local PNG, ... - see :doc:`signal`), with the analytic wide-separation corrections added on top:

.. code-block:: python

    # Define k values and redshift
    k = np.linspace(0.01, 0.2, 50)
    z = 1.0

    # Newtonian (Kaiser) monopole and quadrupole - kernels=['N'] is the default
    m = cw.Model(cosmo_funcs)
    P_0, P_2 = m.pk(k, l=[0, 2], zz=z)  # complex - odd multipoles are imaginary

    # and with the wide-separation corrections (analytic, from MathWAP)
    m_ws = cw.Model(cosmo_funcs, wide_separation=True)
    P_ws = m_ws.pk(k, l=[0, 1, 2], zz=z)

    # correlation function multipoles of the same signal
    r = np.linspace(10, 200, 39)
    xi_0, xi_2 = m.xi(r, l=[0, 2], zz=z)

    # and we can plot contributions
    plt.figure(figsize=(8, 5))
    plt.plot(k, P_0.real, label='Kaiser')
    plt.plot(k, (P_ws[0] - P_0).real, label='WS')
    plt.plot(k, P_ws[1].imag, label='WS dipole')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P_l(k) [(Mpc/h)^3]')
    plt.legend()

``pk_func`` is the lower-level function the forecasts use, with the kernels and the analytic terms passed separately - it gives the same multipoles:

.. code-block:: python

    import cosmo_wap.pk as pk

    # Compute the Newtonian monopole and quadrupole (l=0, 2) - None as there is no analytic term
    P_0, P_2 = pk.pk_func(None, [0, 2], cosmo_funcs, k, zz=z, kernels=['N'])

    # Wide-separation corrections to the monopole - these are analytic (MathWAP) only
    P_0_WA = pk.WS.l0(cosmo_funcs, k, zz=z)

Computing Bispectra
-------------------

Similarly for the bispectrum, where each entry of ``l`` is an :math:`\ell` or an :math:`(\ell, m)` pair:

.. code-block:: python

    # Define triangle configuration
    k1, k2, k3 = 0.1, 0.08, 0.05  # scalene - the dipole vanishes on equilateral triangles
    z = 1.0

    # Compute the Newtonian monopole and dipole
    B0, B1 = cw.Model(cosmo_funcs).bk(k1, k2, k3, l=[0, 1], zz=z)

    # With the local relativistic effects - the dipole is now non-zero
    m_gr = cw.Model(cosmo_funcs, kernels=['N', 'LP'])
    B0_GR, B1_GR, B21_GR = m_gr.bk(k1, k2, k3, l=[0, 1, (2, 1)], zz=z)

and again with ``bk_func``, as the forecasts do:

.. code-block:: python

    import cosmo_wap.bk as bk

    B0_GR, B1_GR = bk.bk_func(None, [0, 1], cosmo_funcs, k1, k2, k3, zz=z, kernels=['N', 'LP'])

Forecasting with Fisher Matrices
--------------------------------

CosmoWAP includes a full forecasting pipeline for SNRs, Fisher matrices and MCMCs. A basic example:

.. code-block:: python

    from cosmo_wap.forecast import FullForecast

    # so we are using a Euclid-like survey with Euclid range (0.9,1.8)

    # Create a forecast with 4 redshift bins, kmax = 0.15 h/Mpc
    forecast = FullForecast(cosmo_funcs, kmax_func=0.15, N_bins=4)

    # Compute Fisher matrix for cosmological parameters
    # using Pk monopole (l=0) and quadrupole (l=2)
    fisher = forecast.get_fish(
        ["fNL","A_s", "n_s"],
        kernels=["N", "Loc"], # Newtonian + local PNG
        pkln=[0, 2],          # Pk multipoles
        bkln=None,            # No bispectrum multipoles
        verbose=True
    )

    # Access results
    print("1-sigma errors:", fisher.errors)
    print("Correlation matrix:\n", fisher.correlation)

    # Get error on a specific parameter
    sigma_fNL = fisher.get_error("fNL")

See the :doc:`Forecasting <forecast>` page for more details.

Example Notebooks
=================

See example notebooks in the ``examples/`` directory for usage of CosmoWAP:

- ``example_pk.ipynb`` - Power spectrum and correlation function multipoles with ``Model``: kernels, contributions, the multi-tracer dipole
- ``example_bk.ipynb`` - Bispectrum multipoles with ``Model``: :math:`B_{\ell m}`, :math:`B(\mu, \phi)` and triangle-shape maps
- ``example_forecast.ipynb`` - SNRs, Fisher matrices, priors, best-fit shifts and MCMC with ``FullForecast``

The interactive cells need ``ipywidgets`` for their sliders (``pip install ipywidgets``) - without it they just plot the defaults.

..
   Binder links (temporarily disabled - notebooks being updated):

   .. image:: https://img.shields.io/badge/launch-powerspectrum-F5A252.svg
      :target: https://mybinder.org/v2/gh/craddis1/CosmoWAP_binder/HEAD?labpath=example_pk.ipynb

   .. image:: https://img.shields.io/badge/launch-bispectrum-F5A252.svg
      :target: https://mybinder.org/v2/gh/craddis1/CosmoWAP_binder/HEAD?labpath=example_bk.ipynb
