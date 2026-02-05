Luminosity Functions
====================

CosmoWAP includes luminosity function classes that compute number densities, magnification bias Q, and evolution bias be from physical models. These are used internally by the preset survey classes but can also be used directly.

Hα Luminosity Functions
-----------------------

For flux-limited Hα surveys (Euclid, Roman):

.. py:class:: lib.luminosity_funcs.Model1LuminosityFunction(cosmo)
.. py:class:: lib.luminosity_funcs.Model3LuminosityFunction(cosmo)

   Schechter-type Hα luminosity functions from Pozzetti et al. (2016). Model 3 includes a modified faint-end slope.

   :param cosmo: CLASS cosmology instance

   **Methods:**

   .. method:: luminosity_function(L, zz)

      Compute Φ(L, z) [h³/Mpc³].

   .. method:: number_density(F_c, zz)

      Compute n_g above flux cut F_c [erg/cm²/s].

   .. method:: get_Q(F_c, zz)

      Compute magnification bias Q(z).

   .. method:: get_be(F_c, zz)

      Compute evolution bias be(z).

   .. method:: get_b_1(F_c, zz)

      Compute flux-averaged linear bias b₁(z) using semi-analytic model.

Magnitude-Limited Surveys
-------------------------

For apparent magnitude-limited surveys (BGS, MegaMapper):

.. py:class:: lib.luminosity_funcs.BGSLuminosityFunction(cosmo)

   DESI BGS r-band luminosity function.

.. py:class:: lib.luminosity_funcs.LBGLuminosityFunction(cosmo)

   Lyman Break Galaxy UV luminosity function for MegaMapper.

   **Methods:** Same as Hα classes, but with magnitude cut ``m_c`` instead of flux cut.

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

    # Number density vs redshift
    n_g = LF.number_density(F_c, z)

    # Magnification and evolution bias
    Q = LF.get_Q(F_c, z)
    be = LF.get_be(F_c, z)

    # Linear bias (semi-analytic)
    b1 = LF.get_b_1(F_c, z)
