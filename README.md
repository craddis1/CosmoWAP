# CosmoWAP

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contibutions.

CosmoWAP is an effort to provide a (hopefully) self consistent framwork to compute contribtuions to the fourier power spectrum and bispectrum from wide-separation and relatvisitic effects as well as contribution from Primordial non-Gaussianity (PNG).
These expression can be very cumbersome and it can be tricky to check for consistency in the community and so hopefully this code should be useful in that regard.

CosmoWAP is a *Python* package to analyse the power spectra and bispectra but the analytical expressions themselves are computed analytically in Mathematica using routines which are publicly avalable at [*MathWAP*](https://github.com/craddis1/MathWAP) and then exported as .py files. therefore the main functionality of CosmoWAP is to take these expressions and implement them for a given cosmology (from CLASS) and survey parameters.

## Installation

``` sh
python -m pip install Pylians
```

## Features

CosmoWAP computes redshift spadce expressions for the 3D Fourier *power spectrum* (and it's multipoles with multi-tracer capabilites) as well as the *bispectrum* (with Sccoccimarro spherical harmonic multipoles), it also can be used for:

- Wide separation (WS) effects (i.e. wide angle and radial redshift contributions) up to second order in the WS expansion
- Relativistic (GR) effects (inlding projection and dynamical effects - see ... for currently implemented kernels) up to $\left(\frac{\mathcal{H}}{k}\right)^2$
- Primordial non-Gaussian (PNG) contribution for local, equilateral and orthogonal types (in progress)

### additional features

- Gaussian covariances along with routines for Fisher and SNR analyses (in progress)
- Inclusion of Finger-of-God damping

## Documentation 

see documentation .... coming... (Add)

## Usage
Based on work in [arXiv:2407.00168](https://arxiv.org/abs/2407.00168) 

Also for PNG stuff please refer too: arXiv:24xx.xxxx


## Errors

In progress- still got old stuff lying around...

## Usage

Feel free to edit or just take any part that may be useful but please refer to:
[arXiv:2407.00168](https://arxiv.org/abs/2407.00168)

## Contact

If youre having any problems or have any ideas to make it better-  Feel free to get in contact :) - c.l.j.addis@qmul.ac.uk
