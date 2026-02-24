Luminosity Functions
====================

CosmoWAP includes luminosity function classes that compute number densities, magnification bias Q, and evolution bias for a given model. See `arXiv:2107.13401 <https://arxiv.org/abs/2107.13401>`_ for an overview.

**Key definitions** (for flux-limited surveys):

.. math::

   b_e = -\frac{\partial \ln \bar{n}_g}{\partial \ln(1+z)}\bigg|_{F_c}, \quad
   Q = -\frac{\partial \ln \bar{n}_g}{\partial \ln L}\bigg|_{F_c}

Hα Luminosity Functions
-----------------------

For flux-limited Hα surveys (Euclid, Roman). See Pozzetti et al. (2016) [arXiv:1603.01453] for model detials:

.. math::

   \Phi(z, y) = \phi^*(z) \, g(y), \quad y \equiv L/L^*

where the shape function g(y) and characteristic density φ∗(z) are model-dependent.

.. py:class:: lib.luminosity_funcs.Model1LuminosityFunction(cosmo)

   Standard Schechter function with g(y) = y^α exp(-y), α = -1.35.

.. py:class:: lib.luminosity_funcs.Model3LuminosityFunction(cosmo)

   Broken power-law fit to luminosity function data with g(y) = y^α / (1 + (e - 1) * y^ν) (α = -1.587, ν = 2.288). Updated model from [arXiv:1910.09273] with reduced redshift range.

**Methods:**

.. method:: luminosity_function(L, zz)

   Compute Φ(L, z) [h³/Mpc³].

.. method:: number_density(F_c, zz)

   Integrate luminosity function above flux cut F_c [erg/cm²/s].

.. method:: get_Q(F_c, zz)

   Magnification bias: Q = y_c g(y_c) / G(F_c, z).

.. method:: get_be(F_c, zz)

   Evolution bias from number density evolution and Q.

.. method:: get_b_1(F_c, zz)

   Flux-averaged linear bias using semi-analytic model from `arXiv:1909.12069 <https://arxiv.org/abs/1909.12069>`_ (Table 2).

Magnitude-Limited Surveys (k-correction)
----------------------------------------

If a survey measures galaxy fluxes in fixed wavelength bands, this leads to a K-correction
for the redshifting effect on the bands. In that case, it is standard to work in terms of
dimensionless magnitudes.

Here these surveys can detect objects above a minimum apparent magnitude (m_c) which is linked to the threshold absolute magnitude:

.. math::

   M_c(z) = m_c - 5 \log_{10}\left[\frac{d_L(z)}{10\,\mathrm{pc}}\right] - K(z)

Works with schechter type luminosity functions where:

.. math::

    Φ(z, y) = φ∗(z) g(y) where y ≡ M - M*(z)

φ∗(z), g(y) are defined in the child classes for a specific luminosity function

See: arXiv:2107.13401 for an overview

.. py:class:: lib.luminosity_funcs.BGSLuminosityFunction(cosmo)

   DESI BGS r-band luminosity function. Schechter with α = -1.23, K(z) = 0.87z.

.. py:class:: lib.luminosity_funcs.LBGLuminosityFunction(cosmo)

   Lyman Break Galaxy UV luminosity function for MegaMapper. Parameters from `arXiv:1904.13378 <https://arxiv.org/abs/1904.13378>`_ (Table 3). K(z) = -2.5 log₁₀(1+z). Bias model from Eq. (2.7).

**Methods:** Same as Hα classes, but with magnitude cut ``m_c`` instead of flux cut.

For magnitude-limited surveys:

.. math::

   Q(z, m_c) = \frac{5}{2 \ln(10)} \frac{\Phi(z, M_c)}{\bar{n}_g(z, m_c)}

Usage
-----

.. code-block:: python

    from cosmo_wap.lib.luminosity_funcs import Model3LuminosityFunction
    from cosmo_wap.lib import utils
    import numpy as np

    cosmo = utils.get_cosmo()
    LF = Model3LuminosityFunction(cosmo)

    z = np.linspace(0.9, 1.8, 50)
    F_c = 2e-16  # erg/cm²/s

    # Number density 
    n_g = LF.number_density(F_c, z)

    # Magnification and evolution biases
    Q = LF.get_Q(F_c, z)
    be = LF.get_be(F_c, z)

    # Linear bias - from a magnitude dependent parameterization
    b1 = LF.get_b_1(F_c, z)
