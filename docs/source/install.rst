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

.. note::

   Optional. CosmoWAP works out of the box in pure numpy - this is a build-once speedup that only matters if you run bispectrum MCMC.

Most of a bispectrum likelihood call is spent in the wide-separation expressions (``WA2``, ``WARR``, ``RR2``, ``WSGR``). These can be compiled to C: roughly 10x per expression and ~3x per likelihood call. Requires ``gcc``; build once per machine:

.. code-block:: bash

    python -m cosmo_wap.bk.c_compile

The build takes ~45 minutes (almost all of it ``RR2``) and writes to ``cosmo_wap/bk/c_lib/``. Compiled kernels are picked up automatically on import and give bit-identical results to numpy (they are built with ``-ffp-contract=off``, which is what guarantees that). Set ``COSMOWAP_DISABLE_C=1`` or delete ``c_lib`` to go back; if the expression files change, stale kernels are detected and skipped with a warning telling you to rebuild.

.. note::

   On 8 GB of RAM or less, build with ``COSMOWAP_BUILD_JOBS=1``. ``RR2`` produces two ~3.3 GB translation units, and compiling them concurrently gets them OOM-killed - leaving a partial build that silently falls back to numpy.

The kernels thread over triangles with OpenMP, but run single-threaded unless ``OMP_NUM_THREADS`` is set, so MPI/multi-chain jobs are never oversubscribed by default. If you have spare cores per chain, set it to that number; results are identical at any thread count.

*Tuning, rarely needed.* On x86-64 the build uses ``-mavx2 -mfma`` (present on every part since ~2013, so a login-node build runs on the compute nodes); elsewhere the vector types are generic and gcc lowers them onto whatever SIMD the target has. ``COSMOWAP_ARCH_FLAGS`` and ``COSMOWAP_SIMD_WIDTH`` override this. AVX-512 at width 8 gains 1.5-2x on ``WA2`` and ``WARR``, but doubles each module's scratch: ``RR2`` grows to 3.9 MB and several concurrent chains overflow a shared L3 (measured 0.59x with six chains on a 16 MB machine), and the library crashes anywhere those instructions are missing. Width 8 suits single-chain runs only; it is recorded per module in ``manifest.json``, so modules can be built at different widths.

Threads: BLAS and the compiled kernels
--------------------------------------

Two independent thread pools can be live during a likelihood call, and they want opposite settings. Getting this wrong costs more than most of the optimisations on this page win.

**Hold numpy's BLAS to one thread.** The arrays in a likelihood call are far too small for threaded BLAS to pay for itself, and its pool spins against the OpenMP pool of the compiled kernels. Measured on 12 cores, 5 bins, pk 320 + bk 5650:

==================  ===========  ==========  ===============
BLAS                default      1 thread    CPU per wall-ms
==================  ===========  ==========  ===============
OpenBLAS (pip)      187 ms       95 ms       11.0 -> 1.0
MKL (conda)         111 ms       94 ms       6.0 -> 1.0
==================  ===========  ==========  ===============

An OpenBLAS build spends 12x the CPU to run 2x *slower*. Set this before Python starts:

.. code-block:: bash

    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

``Sampler`` also applies the limit around each likelihood call through ``threadpoolctl`` (``blas_threads=1``; pass ``None`` to disable), which covers OpenBLAS without affecting anything else in the process. It cannot rescue an MKL build: MKL's pool is created when numpy is imported and its idle threads keep busy-waiting, so a limit taken afterwards caps new parallel regions without stopping the spin. The environment variables are the only thing that reaches every BLAS, and they matter most under MPI, where each of N chains otherwise tries to claim every core.

.. warning::

   Do **not** set ``OMP_NUM_THREADS=1`` as a precaution. That variable controls the compiled kernels, not BLAS, and they test only whether it is *set*: giving it any value switches their OpenMP on, with a team of one. Leaving it unset is what keeps them serial. Use it only to give a chain real cores, as described above.

Optional: coefficient tables for MCMC
--------------------------------------

For MCMC only, the bispectrum can be rewritten as a sum over redshift monomials, ``B = sum_m C_m(k1,k2,k3,Pk) * M_m(z)``. The coefficients ``C_m`` depend on cosmology but not on bias, so under cobaya's fast/slow dragging they are built once per slow step and every fast step in the drag block becomes a dot product against the cached table. Generating and building the tables:

.. code-block:: bash

    python -m cosmo_wap.bk.table.convert WA2,RR2,WARR,WSGR,PNG,GR0,GR1,GR2   # ~1.5 h, sympy only
    python -m cosmo_wap.bk.c_compile WA2_tab,RR2_tab,WARR_tab,WAGR_tab,RRGR_tab,Loc_tab,NPP_tab,GR1_tab,GR2_tab

The generated ``*_tab.py`` modules are shipped with the repository, so normally only the second step is needed (~20 min). One module is written per expression *class*, so ``WSGR`` produces both ``WAGR_tab`` and ``RRGR_tab``. Then run with ``COSMOWAP_BK_TABLE=1``.

Measured at 1056 triangles:

===========  ==========  ========  ===========  =========  ============
class        monomials   table     kernel       fast step  break-even
===========  ==========  ========  ===========  =========  ============
``WA2``      96          0.8 MB    3.8 ms       0.075 ms   3.9 steps
``RR2``      1254        10.6 MB   35.0 ms      0.48 ms    1.6 steps
``WARR``     282         2.4 MB    10.9 ms      0.155 ms   2.6 steps
``WAGR``     105         0.9 MB    1.9 ms       0.13 ms    2.0 steps
``RRGR``     363         3.1 MB    6.0 ms       0.30 ms    1.0 steps
``Loc``      21 / 15     0.2 MB    4.3 ms       0.115 ms   1.0 steps
``NPP``      20 / 16     0.2 MB    0.36 ms      0.055 ms   1.2 steps
``GR1``      35 / 26     0.6 MB    0.91 ms      0.089 ms   1.1 steps
``GR2``      66 / 56     0.6 MB    0.60 ms      0.087 ms   1.1 steps
===========  ==========  ========  ===========  =========  ============

Together these eighteen methods drop from ~124 ms to ~2.9 ms per fast step. "Break-even" is how many fast steps a drag block needs before the table has repaid the cost of building it. The four smaller classes (``Loc``, ``NPP``, ``GR1``, ``GR2``) are the ones whose two multipoles need different numbers of monomials, and they are much the cheapest to generate - seconds of sympy rather than the hour ``RR2`` takes.

``GR1`` is the one tabulated class with odd multipoles, so its coefficients are complex: the imaginary unit is a number rather than a redshift symbol and rides along in the coefficient, while the monomial basis stays real. The cached table is then complex128 and costs twice the bytes of a real one of the same shape.

.. note::

   ``PNG`` also defines the ``Eq`` and ``Orth`` shapes, and those **cannot** be tabulated. Their shape functions contain ``(Mk1*Mk2*Mk3)**(-1/3)``, and a redshift symbol under a fractional power cannot be split out into the monomial, so the growth factor would end up frozen into coefficients that are shared across redshift bins. ``convert`` detects this and refuses rather than emitting a table that is wrong per bin, so ``convert PNG`` writes ``Loc_tab.py`` and then stops with an error naming the symbol that survived. That is expected.

.. warning::

   Unlike the compiled kernels, tables are **not** identical to the numpy result: regrouping sums the fully expanded expression, which carries more cancellation. Measured against the kernel, the error relative to max\|B\| ranges from ~1e-13 (``GR2``, ``GR1``, ``Loc``) through ~9e-12 (``NPP``) and ~2e-10 (``WAGR``, ``RRGR``) and ~3e-7 (``WA2``) to ~3e-6 (``WARR``, ``RR2``), with median relative errors of 1e-12 or better throughout. The larger the expression, the more cancellation it carries. Check the worst of these is comfortably below your statistical errors before using it.

The table path is deliberately hard to enter by accident: it needs ``COSMOWAP_BK_TABLE=1`` *and* a sampler that has registered a cosmology version, which ``Sampler`` does automatically. Fisher forecasts, notebooks and one-off calls therefore keep the ordinary kernel, and so cannot be slowed down by a table they would build once and discard. An array-valued ``zz``, or the ``nonlin``/``growth2`` options, also fall back to the kernel.

Tables are cached per cosmology and per triangle set, bounded by ``COSMOWAP_BK_TABLE_MB`` (default 256, **per process**, so N MPI chains want N times that). All eighteen methods together come to ~38 MB per (cosmology, redshift bin), and dragging keeps two cosmologies resident, so budget roughly ``76 MB x N_bins``: the 256 MB default covers about three bins. If the budget is too small the cache thrashes - rebuilding a table costs several kernel calls, which is slower than not using tables at all - so a warning is issued after repeated evictions. ``runtime.cache_stats()`` reports entries, bytes held and evictions.
