
Bispectrum functions
====================

The bispectrum output exported from matehamtica are stored in two different formats: in terms of the "Scoccimarro" spherical harmonic multipoles and in terms of the full local bisepctrum.

Bispectrum multipoles
---------------------

Lets look at example class:

`Bk0': the Newtonian plane-parallel constant redshift terms

.. class:: bk.Bk0

    Methods
    -------

    .. method:: lx(cosmo_functions, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

        Compute the x-th multipole \( l_x \) of the bispectrum for the given contribution.

        :param object cosmo_functions: An instance of `ClassWAP` containing cosmology and survey biases.
        :param array-like k1: Wavevector magnitude 1, broadcastable array in units of [Mpc/h].
        :param array-like k2: Wavevector magnitude 2, broadcastable array in units of [Mpc/h].
        :param array-like k3: (Optional) Wavevector magnitude 3, broadcastable array in units of [Mpc/h]. Either `k3` or `theta` must be set.
        :param array-like theta: (Optional) Outside angle θ, broadcastable array. Either `theta` or `k3` must be set.
        :param array-like zz: Redshift, broadcastable array with k vectors. Default is 0.
        :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :return: Bispectrum multipole contribution in units of [h/Mpc]^6.
        
    .. method:: ylm(l, m, cosmo_functions, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, sigma=None)

        Compute the multipole \(\ell,m\) of the bispectrum doing the angular integral numerically with gauss-legendre integration

        :param int l: The degree of the spherical harmonic.
        :param int m: The order of the spherical harmonic.
        :param object cosmo_functions: An instance of `ClassWAP` containing cosmology and survey biases.
        :param array-like k1: Wavevector magnitude 1, broadcastable array in units of [Mpc/h].
        :param array-like k2: Wavevector magnitude 2, broadcastable array in units of [Mpc/h].
        :param array-like k3: (Optional) Wavevector magnitude 3, broadcastable array in units of [Mpc/h]. Either `k3` or `theta` must be set.
        :param array-like theta: (Optional) Outside angle θ, broadcastable array. Either `theta` or `k3` must be set.
        :param array-like zz: Redshift, broadcastable array with k vectors. Default is 0.
        :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
        :param float sigma: (Optional) Linear disperion that sets FoG damping. Default is None.
        :return: Bispectrum multipole contribution in units of [h/Mpc]^6.

This class stucutre exists for all contributions including:
GR1,GR2,WA1,WA2,RR1,RR2 where we follow the notation of arXiv:2407.00168

Loc, Eq and Orth for PNG terms

Also composite wide separation terms WS

Full local bispectra
--------------------

We can also just directly call the \(\mu and \phi\) dependent terms.
Again for the Newtonian, plane-parallel constant redshift limit `Bk_0'

.. function:: bk.Bk_0(mu, phi, cosmo_functions, k1, k2, k3=None, theta=None, zz=0, r=0, s=0)

    Compute the angle dependent bispectrum

    This function calculates the bispectrum contribution \( B_k \) using cosmological parameters and wavevectors provided. The function accounts for the orientation of the wavevectors relative to the line-of-sight and computes terms based on input angles \( \mu \) and \( \phi \).

    :param float mu: Cosine of the angle between the LOS and the \(k_1\)
    :param float phi: Azimuthal angle between LOS and \(k_2\) in plane normal to \(k_1\).
    :param object cosmo_functions: An instance of `ClassWAP` containing cosmology and survey biases.
    :param array-like k1: Wavevector magnitude 1, broadcastable array in units of [Mpc/h].
    :param array-like k2: Wavevector magnitude 2, broadcastable array in units of [Mpc/h].
    :param array-like k3: (Optional) Wavevector magnitude 3, broadcastable array in units of [Mpc/h]. Either `k3` or `theta` must be set.
    :param array-like theta: (Optional) Outside angle θ, broadcastable array. Either `theta` or `k3` must be set.
    :param array-like zz: Redshift, broadcastable array with k vectors. Default is 0.
    :param float r: Parameter `r` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
    :param float s: Parameter `s` that sets the Line of Sight (LoS) in the local triplet. Default is 0.
    :param float sigma: (Optional) Linear disperion that sets FoG damping. Default is None.
    :return: The bispectrum contribution \( B_k \), in units of [h/Mpc]^6.


Gaussian Covariance
-------------------

Gaussian covariance compared to the measured covariance from 100 fiducial Quijote `Quijote <https://quijote-simulations.readthedocs.io/en/latest/index.html>`_ sims.

.. image:: images/Covariance_comp.png
   :alt: Comparison of theory to measured covariance
   :width: 400px
   :align: center
