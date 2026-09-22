MathWAP overview
================

`MathWAP <https://github.com/craddis1/MathWAP>`_ is the companion repository to CosmoWAP. It holds
the Mathematica notebooks that derive the analytic expressions CosmoWAP evaluates: the multipoles
of the linear power spectrum and tree-level bispectrum, with wide-separation (WS) and relativistic
(GR) contributions up to second order, local, equilateral and orthogonal primordial non-Gaussianity
(PNG), and the Gaussian covariances. The derivations follow
`arXiv:2407.00168 <https://arxiv.org/abs/2407.00168>`_ and
`arXiv:2511.09466 <https://arxiv.org/abs/2511.09466>`_.

Doing this symbolically is particularly useful for the wide-separation terms: the series
expansion in the wide-angle and radial-evolution parameters, and the derivatives of the power
spectrum and bias functions it brings in, are all computed analytically rather than numerically.

You only need MathWAP if you want to see how an expression was derived, or to add a new kernel or
term. Everything it produces is already shipped inside CosmoWAP.

From Mathematica to CosmoWAP
----------------------------

1. A notebook in ``mathematica_routines/`` derives an expression and exports it as a ``.json``
   file into ``mathematica_expr/``, one file per term and multipole.
2. ``mathematica_expr/read_mathematica.ipynb`` parses a ``.json`` file with sympy's Mathematica
   parser and rewrites it in numpy syntax.
3. The result is pasted into the corresponding class in ``cosmo_wap/pk`` or ``cosmo_wap/bk``
   (for example ``bk.WA1.l1``), where it is evaluated on the cosmology and biases held by
   :doc:`ClassWAP <classwap>`.

Repository layout
-----------------

- ``mathematica_routines/`` - the notebooks:

  - ``kernels.nb`` - redshift-space kernels shared by both spectra (see :doc:`kernels`)
  - ``The_powerspectrum.nb``, ``Pk_expansions.nb``, ``Pk_funcsandrules.nb`` (see :doc:`pkmath`)
  - ``The_bispectrum.nb``, ``Expansions.nb``, ``Bk_funcsandrules.nb`` (see :doc:`bkmath`)
  - ``int.nb`` - the line-of-sight integrated contributions of :doc:`integrated`

- ``mathematica_expr/`` - the exported ``.json`` expressions and ``read_mathematica.ipynb``.
