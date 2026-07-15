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

   CosmoPower is mainly required if you want to sample over cosmological parameters in MCMC. Fisher matrix forecasting and all other CosmoWAP functionality are pretty quick without it.

Advanced: compiled bispectrum kernels
-------------------------------------

For MCMC with the bispectrum, most of each likelihood evaluation is spent in the wide-separation bispectrum expressions (``WA2``, ``WARR``, ``RR2``, ``WSGR``). These can optionally be compiled to C, which speeds up each expression by roughly 10x and a typical bispectrum likelihood call by ~3x overall. Requires ``gcc``; build once per machine:

.. code-block:: bash

    python -m cosmo_wap.bk.c_compile

The build takes ~45 minutes (almost all of it compiling ``RR2``) and writes the compiled kernels next to the expression files in ``cosmo_wap/bk/c_lib/``. Once built, they are picked up automatically on import and transparently replace the numpy implementations - results are identical to float precision, and nothing changes for anyone who never runs the build.

To go back to pure numpy, set the environment variable ``COSMOWAP_DISABLE_C=1`` or delete the ``c_lib`` directory. If the underlying expression files ever change (e.g. on updating CosmoWAP), the stale kernels are detected and skipped with a warning telling you to rebuild.

The compiled kernels can also thread over triangles with OpenMP: they run single-threaded unless ``OMP_NUM_THREADS`` is set, so MPI/multi-chain jobs are never oversubscribed by default. If you have more cores than chains (e.g. 10 chains on a 40-core node), set ``OMP_NUM_THREADS`` to the cores available per chain for a further speedup; results are identical at any thread count.
