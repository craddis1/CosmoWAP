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

.. note::

   On a machine with 8 GB of RAM or less, build with ``COSMOWAP_BUILD_JOBS=1``. ``RR2`` produces two translation units of roughly 3.3 GB each, and compiling them concurrently gets them OOM-killed - leaving a partial build that silently falls back to numpy. Serial takes ~23 minutes for ``RR2`` alone.

Advanced: coefficient tables for MCMC
--------------------------------------

For MCMC only, the bispectrum can be rewritten as a sum over redshift monomials, ``B = sum_m C_m(k1,k2,k3,Pk) * M_m(z)``. The coefficients ``C_m`` depend on cosmology but not on bias, so under cobaya's fast/slow dragging they are built once per slow step and every fast step in the drag block becomes a dot product against the cached table. Generating and building the tables for ``WA2`` and ``RR2``:

.. code-block:: bash

    python -m cosmo_wap.bk.table.convert WA2,RR2       # ~50 min, sympy only
    python -m cosmo_wap.bk.c_compile WA2_tab,RR2_tab   # ~12 min, needs gcc

The generated ``*_tab.py`` modules are shipped with the repository, so normally only the second step is needed. Then run with ``COSMOWAP_BK_TABLE=1``. Measured at 1056 triangles, a fast step drops from 3.9 ms to 0.08 ms for ``WA2`` (50x) and from 34.8 ms to 0.53 ms for ``RR2`` (66x), with the table build costing 1.8-3.5 kernel calls, so it pays for itself after about two to four fast steps.

.. warning::

   Unlike the compiled kernels, tables are **not** identical to the numpy result: regrouping sums the fully expanded expression, which carries more cancellation. Measured against the kernel, the error relative to max\|B\| is ~3e-7 for ``WA2`` and ~3e-6 for ``RR2``, with median relative errors around 1e-12. Check this is comfortably below your statistical errors before using it.

The table path is deliberately hard to enter by accident: it needs ``COSMOWAP_BK_TABLE=1`` *and* a sampler that has registered a cosmology version, which ``Sampler`` does automatically. Fisher forecasts, notebooks and one-off calls therefore keep the ordinary kernel, and so cannot be slowed down by a table they would build once and discard. An array-valued ``zz``, or the ``nonlin``/``growth2`` options, also fall back to the kernel.

Tables are cached per cosmology and per triangle set, bounded by ``COSMOWAP_BK_TABLE_MB`` (default 256, **per process**, so N MPI chains want N times that). If the budget is too small for your bins and terms the cache thrashes - rebuilding a table costs several kernel calls, which is slower than not using tables at all - so a warning is issued after repeated evictions.
