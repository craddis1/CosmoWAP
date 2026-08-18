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

Optional: compiled bispectrum kernels
-------------------------------------

CosmoWAP works out of the box in pure numpy; this is a build-once speedup that only matters if you run bispectrum MCMC.

Most of a bispectrum likelihood call is spent in the wide-separation expressions (``WA2``, ``WARR``, ``RR2``, ``WSGR``). These can be compiled to C: roughly 10x per expression and ~3x per likelihood call. Requires ``gcc``; build once per machine:

.. code-block:: bash

    python -m cosmo_wap.bk.c_compile

The build takes ~45 minutes (almost all of it ``RR2``) and writes to ``cosmo_wap/bk/c_lib/``. Compiled kernels are picked up automatically on import and give bit-identical results to numpy (they are built with ``-ffp-contract=off``, which is what guarantees that). Set ``COSMOWAP_DISABLE_C=1`` or delete ``c_lib`` to go back; if the expression files change, stale kernels are detected and skipped with a warning telling you to rebuild.

.. note::

   On 8 GB of RAM or less, build with ``COSMOWAP_BUILD_JOBS=1``. ``RR2`` produces two ~3.3 GB translation units, and compiling them concurrently gets them OOM-killed - leaving a partial build that silently falls back to numpy.

The kernels thread over triangles with OpenMP, but run single-threaded unless ``OMP_NUM_THREADS`` is greater than 1, so MPI/multi-chain jobs are never oversubscribed by default. If you have spare cores per chain, set it to that number; results are identical at any thread count.

*Tuning, rarely needed.* On x86-64 the build uses ``-mavx2 -mfma``; elsewhere the vector types are generic and gcc lowers them onto whatever SIMD the target has. ``COSMOWAP_ARCH_FLAGS`` and ``COSMOWAP_SIMD_WIDTH`` override this. AVX-512 at width 8 gains 1.5-2x on ``WA2`` and ``WARR`` but doubles each module's scratch, so concurrent chains can overflow a shared L3 (measured 0.59x with six chains on a 16 MB machine), and the library crashes where those instructions are missing. Width 8 suits single-chain runs only; it is recorded per module in ``manifest.json``.

Threads: BLAS and the compiled kernels
--------------------------------------

Two independent thread pools can be live during a likelihood call, and they want opposite settings.

**Hold numpy's BLAS to one thread.** The arrays in a likelihood call are too small for threaded BLAS to pay for itself, and its pool spins against the OpenMP pool of the compiled kernels. Measured on 12 cores, 5 bins, pk 320 + bk 5650, an OpenBLAS build runs 2x slower for 12x the CPU (187 ms -> 95 ms once limited). Set this before Python starts:

.. code-block:: bash

    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

``Sampler`` also applies the limit around each likelihood call through ``threadpoolctl`` (``blas_threads=1``; pass ``None`` to disable). That covers OpenBLAS but not MKL, whose pool is created when numpy is imported and keeps busy-waiting regardless. The environment variables are the only thing that reaches every BLAS, and they matter most under MPI, where each of N chains otherwise tries to claim every core.

.. warning::

   ``OMP_NUM_THREADS`` controls the compiled kernels, not BLAS - setting it to 1 does nothing for the BLAS oversubscription above. Watch instead for a scheduler setting it above 1: under MPI that gives every chain a full thread team on top of its own process.

Optional: coefficient tables for MCMC
--------------------------------------

For MCMC only, the bispectrum can be rewritten as a sum over redshift monomials, ``B = sum_m C_m(k1,k2,k3,Pk) * M_m(z)``. The coefficients ``C_m`` depend on cosmology but not on bias, so under cobaya's fast/slow dragging they are built once per slow step and every fast step in the drag block becomes a dot product against the cached table. Generating and building the tables:

.. code-block:: bash

    python -m cosmo_wap.bk.table.convert WA2,RR2,WARR,WSGR,PNG,GR0,GR1,GR2   # ~1.5 h, sympy only
    python -m cosmo_wap.bk.c_compile WA2_tab,RR2_tab,WARR_tab,WAGR_tab,RRGR_tab,Loc_tab,NPP_tab,GR1_tab,GR2_tab

The generated ``*_tab.py`` modules are shipped with the repository, so normally only the second step is needed (~20 min). One module is written per expression *class*, so ``WSGR`` produces both ``WAGR_tab`` and ``RRGR_tab``. Then run with ``COSMOWAP_BK_TABLE=1``.

Measured at 1056 triangles, the eighteen tabulated methods together drop from ~124 ms to ~2.9 ms per fast step. A drag block needs 1-4 fast steps before a table has repaid the cost of building it.

``GR1`` is the one tabulated class with odd multipoles, so its coefficients are complex and its cached table is complex128, costing twice the bytes of a real one of the same shape.

The table path needs ``COSMOWAP_BK_TABLE=1`` *and* a sampler that has registered a cosmology version, which ``Sampler`` does automatically. Fisher forecasts, notebooks and one-off calls therefore keep the ordinary kernel, as do an array-valued ``zz`` and the ``nonlin``/``growth2`` options.

Tables are cached per cosmology and per triangle set, bounded by ``COSMOWAP_BK_TABLE_MB`` (default 256, **per process**, so N MPI chains want N times that). All eighteen methods come to ~38 MB per (cosmology, redshift bin) and dragging keeps two cosmologies resident, so budget roughly ``76 MB x N_bins``; the default covers about three bins. Too small a budget thrashes the cache and is slower than not using tables at all, so a warning is issued after repeated evictions. ``runtime.cache_stats()`` reports entries, bytes held and evictions.

Multi-tracer
------------

The multi-tracer expressions (``cosmo_wap.bk_mt``, used automatically when ``cosmo_funcs.multi_tracer`` is set) take both optimisations, and both are cheap here: they only implement ``NPP``, ``GR1``, ``GR2`` and local PNG, which are small expressions next to the wide-separation classes.

.. code-block:: bash

    python -m cosmo_wap.bk.table.convert --pkg bk_mt GR0,GR1,GR2,PNG   # seconds, sympy only
    python -m cosmo_wap.bk.c_compile mt_GR0,mt_GR1,mt_GR2,mt_PNG       # seconds
    python -m cosmo_wap.bk.c_compile mt_NPP_tab,mt_GR1_tab,mt_GR2_tab,mt_Loc_tab

The multi-tracer modules are built by ``python -m cosmo_wap.bk.c_compile`` with no arguments as well; the ``mt_`` prefix on a module key is what selects ``bk_mt``, and it exists because the two packages define classes of the same name (``NPP``, ``GR1``, ``GR2``, ``Loc``) while ``c_lib/`` and its manifest are flat. Generated tables go to ``cosmo_wap/bk_mt/table/`` for the same reason and are shipped with the repository.

Measured at 1056 triangles on a bright/faint Euclid split, the nine tabulated methods together:

======================  =========================
 configuration           per likelihood call
======================  =========================
 numpy                   21.2 ms
 compiled kernels         6.9 ms
 compiled + table         2.5 ms
======================  =========================

Rebuilding all nine tables costs 6.5 ms with the compiled table kernels (19.1 ms without), so a drag block repays the build in about one fast step. The tables come to 12 MB per (cosmology, redshift bin). Accuracy is ~1e-15 relative to the kernel - better than the wide-separation tables' ~1e-6, because these expressions expand to 600-2300 terms rather than ``RR2``'s 605k.

With ``all_tracer=True`` the sampler evaluates every tracer combination (XX/XY/YY for the power spectrum, XXX/XXY/XYY/YYY for the bispectrum), so the table cache holds one entry per (combination, bin, method). Measured on the 5-bin, ``bkln=[0,1,2]`` configuration above: 180 entries and ~15 MB per cosmology, and ``Sampler`` keeps four cosmologies resident, so budget ~64 MB - ``COSMOWAP_BK_TABLE_MB=128`` leaves headroom. The per-combination views are built once per cosmology and cached on it; rebuilding them per likelihood call makes every table lookup miss, since the cache keys on the identity of the ``cosmo_funcs`` it was built from.

``Eq`` and ``Orth`` are not tabulated, in ``bk_mt`` as in ``bk``: their cube-root shape functions leave ``D1`` in the coefficients, which would freeze one redshift bin's growth into a table shared by all of them, so ``convert`` refuses rather than emit it. They keep the compiled kernel.
