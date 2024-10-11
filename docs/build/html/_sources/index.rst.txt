.. CosmoWAP documentation master file, created by
   sphinx-quickstart on Thu Oct 10 16:18:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CosmoWAP Documentation
======================

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contributions

CosmoWAP is an effort to provide a (hopefully) self-consistent framework to compute contributions to the Fourier power spectrum and bispectrum from wide-separation and relativistic effects as well as contributions from Primordial non-Gaussianity (PNG). 

These expressions can be very cumbersome, and it can be tricky to check for consistency in the community, so hopefully this code should be useful in that regard.

CosmoWAP is a *Python* package for analyzing the power spectra and bispectra, but the analytical expressions themselves are computed in *Mathematica* using routines which are publicly available at the repository *MathWAP*. These expressions are then exported as `.py` files. Therefore, the main functionality of CosmoWAP is to take these expressions and implement them for a given cosmology (from CLASS) and survey parameters.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   
   Installation <install>
   Getting started <getting_started>
   
.. toctree::
   :maxdepth: 2
   :caption: Biases:
   
   Survey parameters <surveyparams>
   Bias modelling <biasmodel>
   
.. toctree::
   :maxdepth: 2
   :caption: MathWAP:
   
   Overview <overview>
   Kernels <kernels>
   Powerspectrum <pkmath>
   Bispectrum <bkmath>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
