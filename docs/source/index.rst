.. CosmoWAP documentation master file, created by
   sphinx-quickstart on Thu Oct 10 16:18:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CosmoWAP Documentation
=================================

..
   Unicode block art commented out (doesn't render well on ReadTheDocs):

   .. code-block:: text
      ░░      ░░░░      ░░░░      ░░░  ░░░░  ░░░      ░░░  ░░░░  ░░░      ░░░       ░░
      ▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒   ▒▒   ▒▒  ▒▒▒▒  ▒▒  ▒  ▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
      ▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓▓      ▓▓▓        ▓▓  ▓▓▓▓  ▓▓        ▓▓  ▓▓▓▓  ▓▓       ▓▓
      █  ████  ██  ████  ████████  ██  █  █  ██  ████  ██   ██   ██        ██  ███████
      ██      ████      ████      ███  ████  ███      ███  ████  ██  ████  ██  ███████

.. code-block:: text

      ______                         _       _____    ____
     / ____/___  _________ ___  ____| |     / /   |  / __ \
    / /   / __ \/ ___/ __ `__ \/ __ \ | /| / / /| | / /_/ /
   / /___/ /_/ (__  ) / / / / / /_/ / |/ |/ / ___ |/ ____/
   \____/\____/____/_/ /_/ /_/\____/|__/|__/_/  |_/_/

COSMOlogy with Wide-separation, relAtivistic and Primordial non-gaussian contributions

`CosmoWAP <https://github.com/craddis1/CosmoWAP>`_ is a Python framework for forecasts of large-scale galaxy clustering using 3D power spectrum and/or bispectrum multipoles. The modelling covers relativistic effects, wide-separation corrections, and Primordial non-Gaussianity (PNG) in standard perturbation theory.

The core analytical expressions are derived in Mathematica using `MathWAP <https://github.com/craddis1/MathWAP>`_ and exported as `.py` files. These expressions take a cosmology (via CLASS) and a set of bias models as input. Built around this core is the forecasting machinery: bias modelling with HOD/luminosity functions, theoretical covariances, Fisher matrices, and MCMC modules.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   Installation <install>
   Getting started <getting_started>

.. toctree::
   :maxdepth: 2
   :caption: Biases:

   Survey parameters <surveyparams>
   Luminosity functions <luminosityfuncs>
   Bias modelling <biasmodel>

.. toctree::
   :maxdepth: 2
   :caption: Power Spectra and Bispectra:

   ClassWAP <classwap>
   Power Spectrum <pk>
   Integrated Effects <integrated>
   Bispectrum <bk>

.. toctree::
   :maxdepth: 2
   :caption: Forecasting:

   Gaussian Covariance <covariance>
   Forecasting <forecast>
   SNR <snr>
   Best-fit Bias <bfb>

.. toctree::
   :maxdepth: 2
   :caption: MathWAP:

   Overview <overview>
   Kernels <kernels>
   Power Spectrum <pkmath>
   Bispectrum <bkmath>

.. toctree::
   :maxdepth: 1
   :caption: Reference:

   Citation <citation>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
