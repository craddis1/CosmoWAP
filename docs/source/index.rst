.. CosmoWAP documentation master file, created by
   sphinx-quickstart on Thu Oct 10 16:18:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CosmoWAP Documentation
=================================

.. code-block:: text
   ░░      ░░░░      ░░░░      ░░░  ░░░░  ░░░      ░░░  ░░░░  ░░░      ░░░       ░░
   ▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒   ▒▒   ▒▒  ▒▒▒▒  ▒▒  ▒  ▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
   ▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓▓▓      ▓▓▓        ▓▓  ▓▓▓▓  ▓▓        ▓▓  ▓▓▓▓  ▓▓       ▓▓
   █  ████  ██  ████  ████████  ██  █  █  ██  ████  ██   ██   ██        ██  ███████
   ██      ████      ████      ███  ████  ███      ███  ████  ██  ████  ██  ███████
                                                                                 

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

Citation
========

If you use CosmoWAP in your research, please cite the original cosmowap paper:

.. code-block:: bibtex

   @ARTICLE{2025JCAP...04..080A,
         author = {{Addis}, Chris and {Guandalin}, Caroline and {Clarkson}, Chris},
         title = "{Multipoles of the galaxy bispectrum on a light cone: wide-separation and relativistic corrections}",
         journal = {\jcap},
      keywords = {power spectrum, redshift surveys, cosmological parameters from LSS, Cosmology and Nongalactic Astrophysics},
            year = 2025,
         month = apr,
         volume = {2025},
         number = {4},
            eid = {080},
         pages = {080},
            doi = {10.1088/1475-7516/2025/04/080},
   archivePrefix = {arXiv},
         eprint = {2407.00168},
   primaryClass = {astro-ph.CO},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2025JCAP...04..080A},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

If you use anything related to integrated effects or multi-tracer/luminosity function functionality please also cite:

.. code-block:: bibtex

   @ARTICLE{2025arXiv251109466A,
         author = {{Addis}, Chris and {Guedezounme}, S{\^e}cloka L. and {Hammond}, Jessie and {Clarkson}, Chris and {Montano}, Federico and {Camera}, Stefano and {Jolicoeur}, Sheean and {Maartens}, Roy},
         title = "{Unbiased analysis of primordial non-Gaussianity: the multipoles of the full relativistic power spectrum}",
         journal = {arXiv e-prints},
      keywords = {Cosmology and Nongalactic Astrophysics},
            year = 2025,
         month = nov,
            eid = {arXiv:2511.09466},
         pages = {arXiv:2511.09466},
            doi = {10.48550/arXiv.2511.09466},
   archivePrefix = {arXiv},
         eprint = {2511.09466},
   primaryClass = {astro-ph.CO},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv251109466A},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
