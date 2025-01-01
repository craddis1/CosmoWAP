
Bispectrum functions
====================

The bisepctrum output exported from matehamtica are stored in two different formats: in terms of the "Scoccimarro" spherical harmonic multipoles and in terms of the full local bisepctrum.

Bispectrum multipoles
---------------------

For example the Newtonian plane-parallel constant redshift terms are stored in the "Bk0" class.

.. class:: Bk0

    Methods
    -------

    .. method:: ln(cosmo_functions, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

        Compute the monopole term of the bispectrum.

        :param object cosmo_functions: An instance of `ClassWAP` containing cosmological functions.
        :param array-like k1: Wavevector magnitude 1, broadcastable array in units of [Mpc/h].
        :param array-like k2: Wavevector magnitude 2, broadcastable array in units of [Mpc/h].
        :param array-like k3: (Optional) Wavevector magnitude 3, broadcastable array in units of [Mpc/h]. Either `k3` or `theta` must be set.
        :param array-like theta: (Optional) Outside angle θ, broadcastable array. Either `theta` or `k3` must be set.
        :param array-like zz: Redshift, broadcastable array in units of [dimensionless]. Default is 0.
        :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :return: Bispectrum contribution in units of [h/Mpc]^6.
        :rtype: float
        
    .. method:: ylm(l, m, cosmo_functions, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

        Compute the spherical harmonic coefficient for a given set of parameters.

        :param int l: The degree of the spherical harmonic.
        :param int m: The order of the spherical harmonic.
        :param object cosmo_functions: An instance containing cosmological functions.
        :param float k1: Wavevector magnitude 1.
        :param float k2: Wavevector magnitude 2.
        :param float k3: (Optional) Wavevector magnitude 3.
        :param float theta: (Optional) Angle parameter.
        :param float zz: (Optional) Redshift value. Default is 0.
        :param float r: (Optional) Parameter `r`. Default is 0.
        :param float s: (Optional) Parameter `s`. Default is 0.
        :param float sigma: (Optional) Standard deviation or another parameter. Default is None.
        :return: The computed spherical harmonic coefficient.
        :rtype: float





Gaussian Covariance
-------------------

Gaussian covariance compared to the measured covariance from 100 fiducial Quijote `Quijote <https://quijote-simulations.readthedocs.io/en/latest/index.html>`_ sims.

.. image:: images/Covariance_comp.png
   :alt: Comparison of theory to measured covariance
   :width: 400px
   :align: center
