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
[![ASCL](https://img.shields.io/badge/ascl-2507.020-blue.svg?colorB=262255)](https://ascl.net/2507.020)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cosmowap?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=downloads)](https://pepy.tech/projects/cosmowap)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Cosmo**logy with **W**ide-separation, rel**A**tivistic and **P**rimordial non-Gaussian contributions.

CosmoWAP is a project that has grown organically alongside several research works, taking an increasingly central role in their development. It is designed to be a flexible and user-friendly Python framework to enable quick and easy forecasts of large-scale galaxy clustering using 3D power spectrum and/or bispectrum multipoles.

The modelling covers several key areas: relativistic effects, wide-separation corrections, and Primordial non-Gaussianity (PNG) in standard perturbation theory. The core analytical expressions are derived in Mathematica (see [MathWAP](https://github.com/craddis1/MathWAP)) and stored as exported .py files for efficiency. These expressions ingest a cosmology (via CLASS) and a flexible set of bias models, and have been thoroughly validated with sanity checks and against existing literature results.

To enable realistic forecasts, the associated machinery has been built around the core: thorough bias modelling with HOD/luminosity functions, theoretical covariances, Fisher matrices, and MCMC modules. These tools are easily accessible and can be used flexibly for a wide range of survey scenarios.

## [Documentation](https://cosmowap.readthedocs.io/en/latest/)

> [!NOTE]
> Note this is still in progress as this is an evolving repo!
> Occasionally parts will be outdated and will contain deprecated methods.

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
cd CosmoWAP
pip install -e .
```

See requirements.txt for full list of dependencies (most are common python libraries). classy (CLASS python wrapper) is necessary to fully use CosmoWAP.

## Features

**Core Observables**
* **3D redshift space Fourier statistics:** Multipoles and full line-of-sight (LOS) dependent expressions (fully multi-tracer compatible).
* **$P_{\ell}(k)$:** Legendre Power spectrum multipoles.
* **$B_{\ell,m}(k_1,k_2,k_3)$:** Scoccimarro Bispectrum multipoles (TriPosH extension to come).

---

**Physical Contributions**
* **Wide-separation (WS) effects:** Wide-angle and radial redshift contributions up to second order in the WS expansion.
* **Local Relativistic (GR) effects:** Projection and dynamical effects up to $(\mathcal{H}/k)^2$.
* **Integrated effects:** Lensing, ISW, and time delay contributions (currently implemented for the power spectrum only).
* **Primordial non-Gaussianity (PNG):** Contributions for local, equilateral, and orthogonal types — uses peak background split approach for biases.
* **Non-linearities:** Finger-of-God damping and non-linear `HaloFit`/`HMcode` $P(k)$.

---

**Forecasting and Analysis**
* **Bias modelling:** Uses literature Luminosity Functions and HOD/HMF.
* **Multi-tracer covariances:** Multipole covariances assuming Gaussianity for power spectrum (including wide-separation corrections) and bispectrum.
* **Forecasting modules:** Fisher matrices and MCMC capabilities through [`Cobaya`](https://github.com/CobayaSampler/cobaya) and emulated $P(k)$ with [`CosmoPower`](https://github.com/alessiospuriomancini/cosmopower).
* **Built-in plotting library:** Designed for exploring theoretical expressions and forecasts — uses [`ChainConsumer`](https://github.com/Samreay/ChainConsumer).

## Usage

If you use CosmoWAP in your research, please cite the relevant papers:

* Base code: [arXiv:2407.00168](https://arxiv.org/abs/2407.00168)
* Integrated effects: [arXiv:2511.09466](https://arxiv.org/abs/2511.09466)
* PNG and forecasting routines: arXiv:25xx.xxxx


## Contact

If you find any bugs or errors or have any questions and suggestions feel free to get in touch :) - c.l.j.addis@qmul.ac.uk
