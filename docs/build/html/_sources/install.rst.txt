Installation Guide
==================

Installing via pip
------------------

Requires Python >= 3.10.

.. code-block:: bash

    pip install cosmowap

This installs CosmoWAP and all required dependencies (numpy, scipy, classy, matplotlib, tqdm, cython, ChainConsumer, cobaya).

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

   CosmoPower is only required if you want to sample over cosmological parameters in MCMC. Fisher matrix forecasting and all other CosmoWAP functionality work without it.
