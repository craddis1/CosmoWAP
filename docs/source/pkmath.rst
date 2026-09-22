Power spectrum
==============

``The_powerspectrum.nb`` is the main notebook. It builds the power spectrum from the kernels,
expands in the wide-separation parameters, integrates over :math:`\mu` analytically to obtain
the multipoles, and exports each term as a ``.json`` file. It uses:

- ``Pk_expansions.nb`` - the wide-angle and radial-evolution series expansions.
- ``Pk_funcsandrules.nb`` - helper functions and replacement rules used throughout.

Exported expressions
--------------------

Files in ``mathematica_expr/Pk/`` are named by multipole and term, so ``Pkl2GR2.json`` is the
:math:`\ell = 2` multipole of the second-order relativistic term, matching ``pk.GR2.l2`` in
CosmoWAP. The ``PkFOG`` prefix marks the variant with Finger-of-God damping.

- ``Pk/`` - the WA, RR, GR and mixed (WAGR, WARR, RRGR) terms
- ``Pk_PNG/`` - the local, equilateral and orthogonal PNG terms
- ``Pk/covariance/`` - the Newtonian Gaussian covariance :math:`C[P_{\ell_1}, P_{\ell_2}]`
  (``covl2l0``), with ``covdfog`` for the damped case; these are ``pk.COV`` in CosmoWAP
- ``covariance/covWS*`` - wide-separation corrections to the covariance
