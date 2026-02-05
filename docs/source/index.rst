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

CosmoWAP is an effort to provide a (hopefully) self-consistent framework to compute contributions to the Fourier power spectrum and bispectrum from wide-separation and relativistic effects as well as contributions from Primordial non-Gaussianity (PNG).

These expressions can be very cumbersome, and it can be tricky to check for consistency in the community, so hopefully this code should be useful in that regard.

`CosmoWAP <https://github.com/craddis1/CosmoWAP>`_ is a *Python* package for analyzing the power spectra and bispectra, but the analytical expressions themselves are computed in *Mathematica* using routines which are publicly available at the repository `MathWAP <https://github.com/craddis1/MathWAP>`_. These expressions are then exported as `.py` files. Therefore, the main functionality of *CosmoWAP* is to take these expressions and implement them for a given cosmology (from CLASS) and survey parameters.


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
   Bispectrum <bk>

.. toctree::
   :maxdepth: 2
   :caption: Forecasting:

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
