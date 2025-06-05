# CosmoWAP
```
   ______                         _       _____    ____ 
  / ____/___  _________ ___  ____| |     / /   |  / __ \
 / /   / __ \/ ___/ __ `__ \/ __ \ | /| / / /| | / /_/ /
/ /___/ /_/ (__  ) / / / / / /_/ / |/ |/ / ___ |/ ____/ 
\____/\____/____/_/ /_/ /_/\____/|__/|__/_/  |_/_/      
                                                        
```

[![PyPI version](https://img.shields.io/pypi/v/cosmowap)](https://pypi.org/project/cosmowap/)
[![Licence](https://img.shields.io/github/license/craddis1/CosmoWAP?label=licence&style=flat-square&color=informational)](https://github.com/craddis1/CosmoWAP/blob/main/LICENCE)
[![Docs](https://img.shields.io/readthedocs/cosmowap/latest?logo=ReadtheDocs)](https://readthedocs.org/projects/cosmowap/builds/)

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contibutions.

CosmoWAP is an effort to provide a (hopefully) self consistent framework to compute contribtuions within standard peturbation theory to the fourier power spectrum and bispectrum including wide-separation and relatvisitic effects as well as Primordial non-Gaussianity (PNG).
These expression can be very cumbersome and it can be tricky to check for consistency in the community and so hopefully this code should be useful in that regard.

CosmoWAP is a *Python* package to analyse the power spectra and bispectra but the analytical expressions themselves are computed analytically in Mathematica using routines which are publicly avalable at [*MathWAP*](https://github.com/craddis1/MathWAP) and then exported as .py files. therefore the main functionality of CosmoWAP is to take these expressions and implement them for a given cosmology (from CLASS) and set of survey parameters.

## [*Documentation*](https://cosmowap.readthedocs.io/en/latest/)

Documentation still in progress as this is an evolving repo!

## Installation

``` sh
pip install cosmowap

```
Or clone repository...

See requirements.txt for full list of dependencies (most are common python libraries). classy (CLASS python wrapper) is necessecary to fully use CosmoWAP.

## Features

CosmoWAPs aim is to provide self-consistent modelling for the linear bispectrum and power spectrum. It contains redshift space expressions for the 3D Fourier (multipoles and full LOS dependent expressions) *power spectrum* (with multi-tracer capabilites) as well as the *bispectrum* (with Sccoccimarro spherical harmonic multipoles), including terms from:

- Wide separation (WS) effects (i.e. wide angle and radial redshift contributions) up to second order in the WS expansion
- Local Relativistic (GR) effects (including projection and dynamical effects) up to $\left(\frac{\mathcal{H}}{k}\right)^2$
- Integrated Effects (IntInt, IntNPP), (e.g. lensing + ISW...) (power spectrum only currently)
- Primordial non-Gaussian (PNG) contribution for local, equilateral and orthogonal types 

It also has a fully integrated forecasting and plotting library that allows these expressions to be explored.

### additional features

- Gaussian multipole covariances 
- Finger-of-God damping and non-linear corrections for covariance.
- TriPOSH bispectrum expansion terms (Coming soon)

## Usage
Based on work in [arXiv:2407.00168](https://arxiv.org/abs/2407.00168) 

Also for PNG and covraiance related stuff please also refer too: arXiv:25xx.xxxx

## Contact

If you find any bugs or errors or have any questions and suggestions feel free to get in touch :) - c.l.j.addis@qmul.ac.uk
