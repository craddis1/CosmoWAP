"""
Halo Mass Function (HMF) multiplicity functions and halo bias models.

Multiplicity functions:
    - Tinker et al. 2010 (arXiv:1001.3162)
    - Sheth-Tormen (arXiv:astro-ph/9901122)

Halo Mass Function:
    - n_h(zz) - R, M, sigma_R are all defined/computed as arrays in ClassWAP

Bias models:
    - Lagrangian bias from Numeric or analytic derivatives of multiplicity functions
"""

from __future__ import annotations

import numpy as np
import scipy.special as sp


class HMF:
    def __init__(self, cosmo_funcs, hmf="Tinker10"):
        self.cosmo_funcs = cosmo_funcs
        self.delta_c = cosmo_funcs.delta_c

        # peak height (z,R)
        self.nu_func = lambda xx: self.delta_c / np.sqrt(self.cosmo_funcs.sig_R["0"](xx))  # interpolate along z

        if hmf == "Tinker10":
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias_Tinker10(self)
        elif hmf == "ST":
            self.multiplicity = self.multiplicity_ST
            self.lagbias = self.LagBias_ST(self)

        else:  # we could easily integrate HMcode for literature HMFs
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias(
                self
            )  # do numerics derivs of multiplicity - default is tinker2010 mass function

    #######################################  Which HMF: f(nu)
    def halo_bias_params(self, zz: float) -> tuple[float, float, float, float, float]:
        """
        Define initial free parameters from Tinker 2010 table 4, Delta=200
        """
        # alpha0 = 0.368
        beta0 = 0.589
        gamma0 = 0.864
        eta0 = -0.243
        phi0 = -0.729

        # alpha,beta,gamma,eta,psi
        beta = beta0 * (1 + zz) ** (0.2)
        gamma = gamma0 * (1 + zz) ** (-0.01)
        eta = eta0 * (1 + zz) ** (0.27)
        phi = phi0 * (1 + zz) ** (-0.08)

        alpha = 1 / (
            2 ** (eta - phi - 0.5)
            * beta ** (-2 * phi)
            * gamma ** (-0.5 - eta)
            * (2**phi * beta ** (2 * phi) * sp.gamma(eta + 0.5) + gamma**phi * sp.gamma(0.5 + eta - phi))
        )

        return alpha, beta, gamma, eta, phi

    def multiplicity_Tinker10(self, zz: float) -> np.ndarray:
        alpha, beta, gamma, eta, psi = self.halo_bias_params(zz)
        nu = self.nu_func(zz)

        return nu * alpha * (1 + (beta * nu) ** (-2 * psi)) * nu ** (2 * eta) * np.exp(-gamma * nu**2 / 2)

    def multiplicity_ST(self, zz: float) -> np.ndarray:
        A = 0.3221
        p = 0.3
        a = 0.707  # 1

        nu = self.nu_func(zz)

        part1 = A * (1 + (a * nu**2) ** (-p))
        part2 = nu * (2 * a / (np.pi)) ** (1 / 2) * (np.exp(-a * nu**2 / 2))
        return part1 * part2

    #############################################################################################
    def n_h(self, zz: float) -> np.ndarray:
        """
        define halo mass function - number density of halos per unit mass - Tinker10
        Return array in (R,z)
        """
        sig_R = self.cosmo_funcs.sig_R["0"](zz)  # array in R,z
        # derivative of sigma wrt to M
        dSdM = np.gradient(np.sqrt(sig_R), self.cosmo_funcs.M_halo, axis=-1)  # is 2D array R,z

        return (
            (self.cosmo_funcs.rho_m(0) / self.cosmo_funcs.M_halo)
            * self.multiplicity(zz)
            * np.abs(dSdM)
            / np.sqrt(sig_R)
        )

    ###############################################################################################
    # get langrangian and then eulerian biases
    class LagBias:
        """
        Get numeric local-in-matter bias expansion - from derivatives of the multiplicity function - e.g. see Appendix C arXiv:1911.03964v3
        """

        def __init__(self, parent_class):
            self.pc = parent_class

        def get_bn(self, zz, N):
            coef = (-self.pc.nu_func(zz)) ** N / (self.pc.delta_c**N * self.pc.multiplicity(zz))
            # computes derivatives of the multiplicity function
            deriv = self.pc.multiplicity(zz)  # so only 1D - only for 1 z value
            for _ in range(N):
                deriv = np.gradient(deriv, self.pc.nu_func(zz), axis=-1)

            return coef * deriv

        def b1(self, zz):
            return self.get_bn(zz, 1)

        def b2(self, zz):
            return self.get_bn(zz, 2)

    # get langrangian biases from either numerical or analytic derivatives of the multiplicity function
    # e.g. see Appendix C arXiv:1911.03964v3
    class LagBias_Tinker10:
        """
        define lagrangian biases in terms of z, M and the halo bias params - these are all arrays in (z,R)
        """

        def __init__(self, parent_class):
            self.pc = parent_class

        def b1(self, zz):
            _, beta, gamma, eta, psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2 * psi) / (delta_c * ((beta * nu) ** (2 * psi) + 1)) + (gamma * nu**2 - 2 * eta - 1) / delta_c

        def b2(self, zz):
            _, beta, gamma, eta, psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2 * psi * (2 * gamma * nu**2 - 4 * eta + 2 * psi - 1)) / (
                delta_c**2 * ((beta * nu) ** (2 * psi) + 1)
            ) + (gamma**2 * nu**4 - 4 * gamma * eta * nu**2 - 3 * gamma * nu**2 + 4 * eta**2 + 2 * eta) / delta_c**2

    class LagBias_ST:  # Sheth-Tormen mass functions biases
        """
        define lagrangian biases in terms of z,M and the halo bias params - these are all arrays in (z,M)
        """

        def __init__(self, parent_class):
            self.pc = parent_class

        def b1(self, zz):
            # A = 0.322
            p = 0.3
            q = 0.707
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c

            return (q * nu**2 - 1) / delta_c + 2 * p / (delta_c * (1 + (q * nu**2) ** p))

        def b2(self, zz):
            # A = 0.322
            p = 0.3
            q = 0.707
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c

            part1 = q * nu**2 * (q * nu**2 - 3) / delta_c**2
            part2 = (1 + 2 * p + 2 * (q * nu**2 - 1)) * (2 * p) / (delta_c**2 * (1 + q * nu ** (2 * p)))
            return part1 + part2
