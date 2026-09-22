Installation Guide
==================

Installing via pip
------------------

Requires Python >= 3.10.

.. code-block:: bash

    pip install cosmowap

This installs CosmoWAP and all its required dependencies (numpy, scipy, classy, cobaya and others - see ``pyproject.toml``).

Development Version
-------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/craddis1/CosmoWAP.git
    cd CosmoWAP
    pip install -e .

Optional: CosmoPower
--------------------

For MCMC sampling over cosmology, CosmoWAP supports CosmoPower emulators. We recommend Python 3.11 which has been tested to work with pip. Install separately:

.. code-block:: bash

    pip install cosmopower==0.2.0

.. note::

   CosmoPower is mainly required if you want to sample over cosmological parameters in MCMC. Fisher matrix forecasting and all other CosmoWAP functionality are pretty quick without it.

.. note::

   ``cosmopower`` pulls in ``tensorflow<2.14``, which caps numpy below what recent matplotlib needs, so installing it separately can leave matplotlib broken. Install them together so pip resolves both at once:

   .. code-block:: bash

       pip install cosmopower==0.2.0 "matplotlib<3.11"

Optional accelerators
---------------------

CosmoWAP works out of the box in pure numpy. numba is installed with it and speeds up the
hottest spline and quadrature blocks automatically. For bispectrum MCMC there are two further
build-once speedups, compiled C kernels and cached coefficient tables, plus some threading
settings that matter under MPI. All of these are described on the
:doc:`Performance and acceleration <performance>` page.
