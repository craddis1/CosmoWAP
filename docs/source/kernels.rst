Kernels
=======

``kernels.nb`` defines the redshift-space kernels used by both ``The_powerspectrum.nb`` and
``The_bispectrum.nb``: the first- and second-order kernels for the observed galaxy number counts,
including the Newtonian (Kaiser) terms, the wide-angle and radial-evolution corrections that make
up the wide-separation expansion, and the relativistic projection terms at :math:`\mathcal{O}(\mathcal{H}/k)`
and :math:`\mathcal{O}(\mathcal{H}^2/k^2)`.

The kernels are written in terms of the same quantities CosmoWAP later supplies numerically:
the growth rate and factor, the conformal Hubble rate and its derivative, the comoving distance,
and the bias functions :math:`b_1`, :math:`b_2`, :math:`\gamma_2`, :math:`b_e` and :math:`Q`.
Adding a new contribution means adding its kernel here and then re-running the relevant
spectrum notebook.
