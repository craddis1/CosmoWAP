# CosmoWAP

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contibutions.

CosmoWAP is an effort to provide a (hopefully) self consistent framwork to compute contribtuions to the fourier power spectrum and bispectrum from wide-separation and relatvisitic effects as well as contribution from Primordial non-Gaussianity (PNG).
These expression can be very cumbersome and it can be tricky to check for consistency in the community and so hopefully this code should be useful in that regard.

CosmoWAP is a *Python* package to analyse the power spectra and bispectra but the analytical expressions themselves are computed analytically in Mathematica using routines which are publicly avalable at [*MathWAP*](https://github.com/craddis1/MathWAP) and then exported as .py files. therefore the main functionality of CosmoWAP is to take these expressions and implement them for a given cosmology (from CLASS) and survey parameters.

## Installation



## Features

CosmoWAP computes redshift spadce expressions for the 3D Fourier *power spectrum* (and it's multipoles with multi-tracer capabilites) as well as the *bispectrum* (with Sccoccimarro spherical harmonic multipoles), it also can be used for:

- Wide separation (WS) effects (i.e. wide angle and radial redshift contributions) up to second order in the WS expansion
- Relativistic (GR) effects (inlding projection and dynamical effects - see ... for currently implemented kernels) up to $\left(\frac{\mathcal{H}}{k}\right)^2$
- Primordial non-Gaussian (PNG) contribution for local, equilateral and orthogonal types (in progress)

### additional features

- Gaussian covariances along with routines for Fisher and SNR analyses (in progress)
- Inclusion of Finger-of-God damping

## Documentation 

see documentation .... (Add)

## Usage
Based on work in [arXiv:2407.00168](https://arxiv.org/abs/2407.00168) 

Also for PNG stuff please refer too: arXiv:24xx.xxxx


Updating documentiation - old stuff below...

## Overview

- \mathematica_routines\The_bispectrum.nb outputs expressions for each different in the bispectrum for different multipoles

- These are then stored in \bkterms as .py files

- \Library\WS_cosmo: takes in a cosmology from class and survey specifications and returns required parameters to use files in \bkterms - see example.py for usage

## quickstart

Download the repository...

To get started see example.ipynb or example_pk.ipynb for a quick guide on using the computed expressions...

CLASS, matplotlib, scipy and numpy are the only dependencies

## python functions

Outputs are converted to python format and stored in \bk_terms

### Notebooks

For an example notebook for using these expressions see:

- example.ipynb 

- bk_SNR.ipynb includes code to compute and plot the SNR and fisher stuff for multipoles

- bk_plots.ipynb contains a bunch of functions that maybe be useful but is not clean

In these notebooks everything is designed to use \Library\WS_cosmo.py where cosmology and survey biases are defined and from that object all bispectrum terms can be computed.


## Multi-tracer Power spectrum multipoles

This includes relatvistic and wide separation effects up to second order. The format should be similar as to that of the bispectrum.

- The_powerspectrum.nb, Pk_funcsandrules.nb, Pk_expansions.nb: mathematica routines for power spectrum computation.

- Library/WS_cosmo.py: Again is the backend to get input parameters for a given survey and cosmology for the saved python functions.

- See example_pk.ipynb for example usage.


## Errors

Some syntax has been updated so there will inevitably be several errors lying around - particular in the bigger notebooks - please let me know!

## Usage

Feel free to edit or just take any part that may be useful but please refer to:
[arXiv:2407.00168](https://arxiv.org/abs/2407.00168)

## Contact

Stuff will be added as time goes by - I plan on adding PNG stuff with scale dependent biases for different shapes of PNG (therefore including HOD for bias modelling)

Feel free to drop me a line if you have any feedback!
If youre having any problems or have any ideas to make it better-  Feel free to get in contact :) - c.l.j.addis@qmul.ac.uk
