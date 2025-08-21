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
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fcc0f69852984e01b101fc56d67c43f4)](https://app.codacy.com/gh/craddis1/CosmoWAP/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![PyPI version](https://img.shields.io/badge/ascl-2507.020-blue.svg?colorB=262255)](https://ascl.net/2507.020)

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contibutions.

CosmoWAP is an effort to provide a (hopefully) self consistent framework to compute contribtuions within standard peturbation theory to the fourier power spectrum and bispectrum including wide-separation and relatvisitic effects as well as Primordial non-Gaussianity (PNG).
These expression can be very cumbersome and it can be tricky to check for consistency in the community and so hopefully this code should be useful in that regard.

CosmoWAP is a *Python* package to analyse the power spectra and bispectra but the analytical expressions themselves are computed analytically in Mathematica using routines which are publicly avalable at [*MathWAP*](https://github.com/craddis1/MathWAP) and then exported as .py files. therefore the main functionality of CosmoWAP is to take these expressions and implement them for a given cosmology (from CLASS) and set of survey parameters.

## [*Documentation*](https://cosmowap.readthedocs.io/en/latest/)

[![Documentation](https://img.shields.io/badge/Read%20the%20Docs-latest-informational?logo=ReadtheDocs)](https://cosmowap.readthedocs.io/en/latest/)

Full documentation is available at [*ReadtheDocs*](https://cosmowap.readthedocs.io/en/latest/).

> [!NOTE]
> Note this is still in progress as this is an evolving repo! 
> Occassionaly parts will be outdated and will contain deprecated methods.

## Installation

> [!NOTE]
> Requires at least Python >=3.10 for full functionality.
> For use of CosmoPower emulators we recommend using Python 3.10 or 3.11 - See Docs for full details.

``` sh
pip install cosmowap

```
For Development mode...

Clone repository:

``` sh
git clone https://github.com/craddis1/CosmoWAP.git
```
and then make editable install:
``` sh
cd cosmowap
pip install -e .
```

See requirements.txt for full list of dependencies (most are common python libraries). classy (CLASS python wrapper) is necessary to fully use CosmoWAP.

## Features

CosmoWAPs aim is to provide self-consistent modelling for the linear bispectrum and power spectrum. It contains redshift space expressions for the 3D Fourier (multipoles and full LOS dependent expressions) *power spectrum* (with multi-tracer capabilites) as well as the *bispectrum* (with Sccoccimarro spherical harmonic multipoles), including terms from:

- Wide separation (WS) effects (i.e. wide angle and radial redshift contributions) up to second order in the WS expansion
- Local Relativistic (GR) effects (including projection and dynamical effects) up to $\left(\frac{\mathcal{H}}{k}\right)^2$
- Integrated Effects (IntInt, IntNPP), (e.g. lensing + ISW...) (power spectrum only currently)
- Primordial non-Gaussian (PNG) contribution for local, equilateral and orthogonal types 

It also has a fully integrated forecasting and plotting library that allows these expressions to be explored.

### additional features

- Bias modelling through Luminosity functions and HOD/HMF
- Gaussian multipole covariances (Multi-tracer power spectrum)
- Finger-of-God damping and non-linear corrections
- TriPOSH bispectrum expansion terms (Coming soon)

## Usage
Base code based on work in [arXiv:2407.00168](https://arxiv.org/abs/2407.00168)

For PNG and Forecasting routines related please also refer too: arXiv:25xx.xxxx

For Integrated effects see: arXiv:25xx.xxxx


## Contact

If you find any bugs or errors or have any questions and suggestions feel free to get in touch :) - c.l.j.addis@qmul.ac.uk
