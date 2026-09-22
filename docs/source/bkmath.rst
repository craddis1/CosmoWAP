Bispectrum
==========

``The_bispectrum.nb`` is the main notebook. It builds the tree-level bispectrum from the kernels,
expands in the wide-separation parameters, projects onto the Scoccimarro spherical-harmonic
multipoles, and exports each term as a ``.json`` file. It uses:

- ``Expansions.nb`` - the wide-angle and radial-evolution series expansions.
- ``Bk_funcsandrules.nb`` - helper functions and replacement rules used throughout.

Exported expressions
--------------------

The bispectrum is a sum over the three permutations of the triangle, and each is exported
separately: ``perm12/``, ``perm13/`` and ``perm23/`` hold the same set of terms with the
wavevectors permuted. Files are named by permutation, multipole and term, so
``perm12/P12l1m1GR1.json`` is the :math:`(\ell, m) = (1, 1)` multipole of the first-order
relativistic term for the 12 permutation; CosmoWAP sums the three when evaluating ``bk.GR1.l1``.

- ``perm12/``, ``perm13/``, ``perm23/`` - the NPP, WA, RR, GR and mixed terms per permutation
- ``BPNG/`` - the local, equilateral and orthogonal PNG terms
- ``covariance/`` - the Gaussian covariance of the multipoles, named by :math:`(\ell, m)`
  (``covl2m0``, ``covl00m00``); these are ``bk.COV`` in CosmoWAP
