from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson

import cosmo_wap as cw

# from cosmo_wap.lib import utils


class HaLuminosityFunction:
    """Parent class of H-alpha luminosity function e.g. Euclid, Roman
    Works with schechter type luminosity functions where:
    Φ(z, y) = φ∗(z) g(y) where y ≡ L/L∗

    φ∗(z), g(y) are defined in the child classes for a specific luminosity function

    Here these surveys can detect a minimum flux (F_c)

    See: arXiv:2107.13401 for an overview"""

    def luminosity_function(self, L: ArrayLike, zz: ArrayLike) -> np.ndarray:
        """
        Schechter luminosity function values for  given luminosity L and redshift zz

        Φ(z, y) = φ∗(z) g(y) where y ≡ L/L∗

        Parameters:
        -----------
        L : array or float
            Luminosity of source
        zz: array or float
            Redshift

        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]
        """

        # make sure mm and zz are 2D arrays for broadcasting
        if isinstance(zz, (np.ndarray)) and isinstance(L, (np.ndarray)) and zz.size != L.size:
            zz = zz[:, np.newaxis]
            L = L[np.newaxis, :]

        # get y L/L*
        y = self.get_y(L, zz)

        return self.get_phi_star(zz) * self.g(y)

    def L_c(self, F_c: float, zz: ArrayLike) -> np.ndarray:
        """
        Convert flux [erg cm^−2 s^−1] to luminosity [erg s^−1] at redshift z
        """
        convert_cm_to_mpc = 3.0857e24  # Mpc in cm
        return F_c * (1 + zz) ** 2 * 4 * np.pi * self.cosmo.comoving_distance(zz) ** 2 * convert_cm_to_mpc**2

    def number_density(self, F_c: float, zz: np.ndarray) -> np.ndarray:
        """
        Calculate the number density of H-alpha emitters for a given flux cut F_c and redshift zz

        n_g(z,F_c) = φ∗(z) G(F_c,z) where G(y) = ∫_0^y g(y') dy'

        Parameters:
        -----------
        F_c : float
            Flux cut [erg cm^−2 s^−1]

        zz : float or array
            Redshift

        Returns:
        --------
        Number density : float or array
            Total number density [h^3 Mpc^-3]
        """

        return self.get_phi_star(zz) * self.get_G(F_c, zz)

    def get_G(self, F_c: float, zz: np.ndarray) -> np.ndarray:
        """
        G(y) = ∫_0^y g(y') dy'
        """
        # so this is 2D array 1st dimension is redshift, 2nd is luminosity
        L = np.logspace(np.log10(self.L_c(F_c, zz)), 47, 1000, axis=-1)  # integrate over luminosity with a given cut

        y = self.get_y(L, zz[:, np.newaxis])  # zz 2D for broadcasting

        return simpson(self.g(y), y, axis=-1)

    def get_Q(self, F_c: float, zz: np.ndarray) -> np.ndarray:
        """
        Q(z, Mc) =
        """
        L_c = self.L_c(F_c, zz)
        y_c = self.get_y(L_c, zz)

        return y_c * self.g(y_c) / self.get_G(F_c, zz)

    def get_nQ(self, F_c: float, zz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Number density and magnification bias sharing a single G integral -
        same as (number_density, get_Q) but cheaper when both are needed"""
        G = self.get_G(F_c, zz)
        n_g = self.get_phi_star(zz) * G

        y_c = self.get_y(self.L_c(F_c, zz), zz)
        return n_g, y_c * self.g(y_c) / G

    def get_be(
        self, F_c: float, zz: np.ndarray, n_g: np.ndarray | None = None, Q: np.ndarray | None = None
    ) -> np.ndarray:
        """Get evolution bias - Eq. 2.25 in arXiv:2107.13401 - compute total derivate and correct"""

        # if not defined use values for given cut
        if n_g is None:
            n_g = self.number_density(F_c, zz)
        if Q is None:
            Q = self.get_Q(F_c, zz)

        # change in number density
        d_ln_ng_dln = np.gradient(np.log(n_g), np.log(1 + zz))

        terms = 2 * (1 + (1 + zz) / (self.cosmo.Hubble(zz) * self.cosmo.comoving_distance(zz))) * Q
        return -d_ln_ng_dln - terms

    def b_1(self, x: ArrayLike, zz: ArrayLike) -> np.ndarray:
        a = 0.844
        b = 0.116
        c = 42.623
        d = 1.186
        e = 1.765

        return a + b * (1 + zz) ** e * (1 + np.exp((x - c) * d))

    def get_b_1(self, F_c: float, zz: np.ndarray) -> np.ndarray:
        r"""Semi-anlaytic model with free parameters from table 2 in 1909.12069
        (∫_x^inf \phi(x) b_1(x) dx)/(∫_x^inf \phi(x) dx)
        Returns linear bias as an array in redshift above a given flux cut
        """
        # so this is 2D array 1st dimension is redshift, 2nd is luminosity
        x = np.zeros((len(zz), 100))

        integrand = np.zeros((len(zz), 100))
        ng_integrand = np.zeros((len(zz), 100))
        for i in range(len(zz)):  # loop over z - for each z we have different cut in luminosity
            x_arr = np.linspace(np.log10(self.L_c(F_c, zz[i])), 47, 100)  # integrate over luminosity with a given cut
            x[i] = x_arr
            lf = self.luminosity_function(10**x_arr, zz[i])
            b1 = self.b_1(x_arr, zz[i])
            integrand[i] = lf * b1
            ng_integrand[i] = lf

        return simpson(integrand, x, axis=-1) / simpson(
            ng_integrand, x, axis=-1
        )  # integrate over luminosities above flux cut


class Model1LuminosityFunction(HaLuminosityFunction):
    def __init__(self, cosmo: object | None = None) -> None:
        """
        H-alpha Luminosity Function calculator
        Luminsotiy in Units h^3 Mpc^-3
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7, 2, 1000)

        # fitted free parameters - stored as attributes so they can be perturbed when
        # forward-modelling fit errors onto b_e/Q (see cosmo_wap.lib.lf_priors)
        self.alpha = -1.35
        self.log_phi_star = -2.8  # log10(phi* at z=0 [Mpc^-3])
        self.eta = 1  # phi*(z) evolution index
        self.z_b = 1.3  # break redshift for phi*(z)
        self.log_L0_star = 41.5  # log10(L* at z=0 [erg/s])
        self.delta = 2  # L*(z) evolution index
        self.fit_params = ["alpha", "log_phi_star", "eta", "z_b", "log_L0_star", "delta"]
        # diagonal 2-sigma fit errors, symmetrised as (upper+lower)/2 from the asymmetric
        # +upper/-lower quoted values - keyed by fit_params name
        self.fit_errors = {
            "alpha": 0.125,  # -1.35 +0.10 -0.15
            "log_phi_star": 0.165,  # -2.80 +0.15 -0.18
            "eta": 0.1,  # 1.0 +0.1 -0.1
            "z_b": 0.1,  # 1.3 +0.1 (-0.1 assumed symmetric)
            "log_L0_star": 0.11,  # 41.50 +0.11 -0.11
            "delta": 0.1,  # 2.0 +0.1 -0.1
        }

    def g(self, y: ArrayLike) -> np.ndarray:
        return y**self.alpha * np.exp(-y)

    def get_phi_star(self, zz: ArrayLike) -> np.ndarray:
        def phi_star_phi_star0(zz):
            """
            L* as a function of redshift
            """
            return np.where(
                zz < self.z_b, (1 + zz) ** self.eta, (1 + self.z_b) ** (2 * self.eta) * (1 + zz) ** (-self.eta)
            )

        # all redshift dependent
        phi_star0 = 10**self.log_phi_star / self.cosmo.h() ** 3  # phi* at z=0 in h^3 Mpc^-3
        phi_star = phi_star_phi_star0(zz) * phi_star0
        return phi_star

    def get_y(self, L: ArrayLike, zz: ArrayLike) -> np.ndarray:
        """Calculate y = L/L* for given luminosity and redshift"""
        L_star = 10**self.log_L0_star * (1 + zz) ** self.delta  # L*
        return L / L_star


class Model2LuminosityFunction(HaLuminosityFunction):
    def __init__(self, cosmo: object | None = None) -> None:
        """
        H-alpha Luminosity Function calculator - Pozzetti et al. (2016) [arXiv:1603.01453] Model 2.
        Luminsotiy in Units h^3 Mpc^-3

        Schechter shape g(y) = y^α exp(-y) (as Model 1) but with a constant φ*(z) (no density
        evolution) and a quadratic-in-z evolution of the characteristic luminosity peaking at
        z_break: log10 L*(z) = -c (z - z_break)^2 + log10 L*_break.
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7, 2, 1000)

        # fitted free parameters - stored as attributes so they can be perturbed when
        # forward-modelling fit errors onto b_e/Q (see cosmo_wap.lib.lf_priors)
        self.alpha = -1.40
        self.log_phi_star = -2.70  # log10(phi* [Mpc^-3]) - constant in z (no density evolution)
        self.log_L_star_break = 42.59  # log10(L* at z_break [erg/s])
        self.c = 0.22  # quadratic coefficient of log10 L*(z)
        self.z_break = 2.23  # peak redshift of L*(z) - held fixed in the Pozzetti fit
        self.fit_params = ["alpha", "log_phi_star", "log_L_star_break", "c"]
        # diagonal 2-sigma fit errors, symmetrised as (upper+lower)/2 from the asymmetric
        # +upper/-lower quoted values - keyed by fit_params name
        self.fit_errors = {
            "alpha": 0.125,  # -1.40 +0.10 -0.15
            "log_phi_star": 0.17,  # -2.70 +0.17 -0.17
            "log_L_star_break": 0.11,  # 42.59 +0.10 -0.12
            "c": 0.05,  # 0.22 +0.05 -0.05
        }

    def g(self, y: ArrayLike) -> np.ndarray:
        return y**self.alpha * np.exp(-y)

    def get_phi_star(self, zz: ArrayLike) -> np.ndarray:
        return 10**self.log_phi_star / self.cosmo.h() ** 3  # constant phi* at z=0 in h^3 Mpc^-3

    def get_y(self, L: ArrayLike, zz: ArrayLike) -> np.ndarray:
        """Calculate y = L/L* with log10 L*(z) = -c (z - z_break)^2 + log10 L*_break"""
        log_L_star = -self.c * (zz - self.z_break) ** 2 + self.log_L_star_break
        return L / 10**log_L_star


class Model3LuminosityFunction(HaLuminosityFunction):
    def __init__(self, cosmo: object | None = None) -> None:
        """
        H-alpha Luminosity Function calculator
        Luminsotiy in Units h^3 Mpc^-3
        Broken power-law fit to luminosity function data with g(y) (α = -1.587, ν = 2.288). Updated model from [arXiv:1910.09273] with reduced redshift range.
        g(y) = y^α / (1 + (e - 1) * y^ν)
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7, 2, 1000)

        # fitted free parameters - stored as attributes so they can be perturbed when
        # forward-modelling fit errors onto b_e/Q (see cosmo_wap.lib.lf_priors)
        self.alpha = -1.587
        self.nu = 2.288
        self.log_phi_star = -2.920  # log10(phi* at z=0 [Mpc^-3])
        self.log_L_star_inf = 42.956  # 42.557  # log10(L* as z -> inf [erg/s])
        self.log_L_star_half = 41.733  # log10(L* at z=0.5 [erg/s])
        self.beta = 1.615
        self.fit_params = ["alpha", "nu", "log_phi_star", "log_L_star_inf", "log_L_star_half", "beta"]
        # diagonal 2-sigma fit errors, symmetrised as (upper+lower)/2 from the asymmetric
        # +upper/-lower quoted values - keyed by fit_params name
        self.fit_errors = {
            "alpha": 0.1255,  # -1.587 +0.132 -0.119
            "nu": 0.3945,  # 2.288 +0.410 -0.379
            "log_phi_star": 0.179,  # -2.920 +0.183 -0.175
            "log_L_star_inf": 0.114,  # 42.557 +0.109 -0.119
            "log_L_star_half": 0.146,  # 41.733 +0.150 -0.142
            "beta": 1.0715,  # 1.615 +0.947 -1.196
        }

    def g(self, y: ArrayLike) -> np.ndarray:
        return y**self.alpha / (1 + (np.e - 1) * y**self.nu)

    def get_phi_star(self, zz: ArrayLike) -> np.ndarray:
        return 10**self.log_phi_star / self.cosmo.h() ** 3  # phi* at z=0 in h^3 Mpc^-3

    def get_y(self, L: ArrayLike, zz: ArrayLike) -> np.ndarray:
        """Calculate y = L/L* for given luminosity and redshift"""
        log_L_star = self.log_L_star_inf + (1.5 / (1 + zz)) ** self.beta * (self.log_L_star_half - self.log_L_star_inf)

        return L / (10**log_L_star)


########################################################################## apparent magnitude limited surveys


class KCorrectionLuminosityFunction:
    """
    Parent class for K-corrected luminosity functions e.g. BGS, Megamapper
    If a survey measures galaxy fluxes in fixed wavelength bands, this leads to a K-correction
    for the redshifting effect on the bands. In that case, it is standard to work in terms of
    dimensionless magnitudes.

    Here these surveys can detect objects above a minimum apparent magnitude (m_c) which is linked to the threshold absolute magnitude:

    M_c(z) = m_c − 5 log[ dL(z)/10 pc] − K(z)

    Works with schechter type luminosity functions where:
    Φ(z, y) = φ∗(z) g(y) where y ≡ M - M*(z)

    φ∗(z), g(y) are defined in the child classes for a specific luminosity function

    See: arXiv:2107.13401 for an overview
    """

    def M_UV(self, mm: ArrayLike, zz: ArrayLike, ref_z: float = 0) -> np.ndarray:
        """
        Convert apparent to absolute UV magnitude
        M(z) = m − 5 log[ dL(z)/10 pc] − K(z)
        """

        if zz is None:
            zz = self.z_values

        if isinstance(mm, (np.ndarray)):
            mm = mm[:, np.newaxis]

        # Luminosity distance in pc
        D_L = 1e6 * self.cosmo.luminosity_distance(zz)  # *self.cosmo.get_current_derived_parameters(['h'])['h']

        # Distance modulus
        distance_modulus = 5 * np.log10(D_L / 10.0)

        # K-correction
        k_correction = self.K(zz, ref_z)

        # Equation 2.6
        M_UV = mm - distance_modulus - k_correction

        return M_UV

    def number_density(self, m_cut: float, zz: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the number density for k corrected survey for given apparent magnitude cut

        Parameters:
        -----------
        m_cut : float
            Apparent magnitude cut
        zz : float or array
            redshift

        Returns:
        --------
        Number density : float or array
            Total number density [h^3 Mpc^-3]
        """

        if zz is None:
            zz = self.z_values

        mm = np.linspace(15, m_cut, 1000)  # apparent magntiude values to integrate over
        luminosity_arr = self.luminosity_function(mm, zz)

        return simpson(luminosity_arr, self.M_UV(mm, zz), axis=0)

    def get_Q(self, m_c: float, zz: np.ndarray | None = None) -> np.ndarray:
        """
        Q(z, Mc) = (5/(2 * ln(10))) * (Φ(z, Mc) / ng(z, Mc))
        """
        return (5 / (2 * np.log(10))) * self.luminosity_function(m_c, zz) / self.number_density(m_c, zz)

    def get_Q2(self, m_c: float, zz: np.ndarray | None = None) -> np.ndarray:
        """
        Q(z, Mc) = (5/2) * (∂ log10 ng(z, Mc) / ∂Mc)
        In terms of number density - matches above
        """

        h_m = 0.01  # for deriv

        # change in number density
        deriv = np.log10(self.number_density(m_c + h_m, zz)) - np.log10(self.number_density(m_c - h_m, zz))

        d_log10_ng_dMc = (deriv) / (2 * h_m)
        return (5 / 2) * d_log10_ng_dMc

    def get_nQ(self, m_c: float, zz: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Number density and magnification bias sharing a single luminosity integral -
        same as (number_density, get_Q) but cheaper when both are needed"""
        n_g = self.number_density(m_c, zz)
        return n_g, (5 / (2 * np.log(10))) * self.luminosity_function(m_c, zz) / n_g

    def get_be(
        self, m_c: float, zz: np.ndarray | None = None, n_g: np.ndarray | None = None, Q: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Eq. 3.6 in arXiv:2107.13401 - evolution bias with K-correction
        """
        if zz is None:
            zz = self.z_values

        # if not defined use values for given cut
        if n_g is None:
            n_g = self.number_density(m_c, zz)
        if Q is None:
            Q = self.get_Q(m_c, zz)

        # change in number density
        d_ln_ng_dln = np.gradient(np.log(n_g), np.log(1 + zz))

        terms = (
            2
            * (
                1
                + (1 + zz) / (self.cosmo.Hubble(zz) * self.cosmo.comoving_distance(zz))
                + 2 * np.log(10) / (5) * np.gradient(self.K(zz), np.log(1 + zz))
            )
            * Q
        )

        return -d_ln_ng_dln - terms


class LBGLuminosityFunction(KCorrectionLuminosityFunction):
    def __init__(self, cosmo: object | None = None) -> None:
        """
        Lyman Break Galaxy Luminosity Function calculator (MegaMapper)
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()

        # Here we have luminosity function parameters fitted for some given redshifts
        # Redshifts and parameters from Table 3 [1904.13378]
        self.z_values = np.array([2.0, 3.0, 3.8, 4.9, 5.9])
        self.M_star = np.array([-20.60, -20.86, -20.63, -20.96, -20.91])
        self.phi_star = np.array([9.70e-3, 5.04e-3, 9.25e-3, 3.22e-3, 1.64e-3])  # h^-3 Mpc^3
        self.alpha = np.array([-1.60, -1.78, -1.57, -1.60, -1.87])

    def luminosity_function(self, mm: ArrayLike, zz: ArrayLike | None = None) -> np.ndarray:
        """
        Schechter luminosity function values for given apparent magnitude and fitted params

        Parameters:
        -----------
        m_obs : float
            Observed apparent magnitude

        zz : float or array
            Redshift

        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]
        """
        # Convert to absolute magnitude at this redshift
        M_UV = self.M_UV(mm, self.z_values)

        if isinstance(mm, (np.ndarray)):
            mm = mm[:, np.newaxis]

        # Calculate Schechter function (Equation 2.5)
        L_Lstar = 10 ** (-0.4 * (M_UV - self.M_star))

        phi_values = (
            (np.log(10) / 2.5)
            * self.phi_star
            * 10 ** (-0.4 * (1 + self.alpha) * (M_UV - self.M_star))
            * np.exp(-L_Lstar)
        )

        return phi_values

    def K(self, zz: ArrayLike, ref_z: float = 0) -> np.ndarray:
        """
        K-correction for LBGs
        """
        return -2.5 * np.log10(1 + (zz - ref_z))  # so this is the K-correction relative to z=0

    def b_1(self, mm: ArrayLike, zz: ArrayLike | None = None) -> np.ndarray:
        """From 1904.13378 - magnitude and redshift dependent biass"""
        if isinstance(mm, (np.ndarray)):
            mm = mm[:, np.newaxis]

        A = -0.98 * (mm - 25) + 0.11  # from Eq.(2.7) 1904.13378v2
        B = 0.12 * (mm - 25) + 0.17
        return A * (1 + zz) + B * (1 + zz) ** 2

    def get_b_1(self, m_c: float, zz: np.ndarray) -> np.ndarray:
        r"""(∫_x^inf \phi(x) b_1(x) dx)/(∫_x^inf \phi(x) dx)
        Returns linear bias as an array in redshift above a given flux cut
        """

        mm = np.linspace(15, m_c, 1000)  # apparent magntiude values to integrate over
        luminosity_arr = self.luminosity_function(mm, zz)  # so array m,zz
        bias_arr = self.b_1(mm, zz)

        # integrate over apparent magnitudes for a given cut
        return simpson(luminosity_arr * bias_arr, self.M_UV(mm, zz), axis=0) / simpson(
            luminosity_arr, self.M_UV(mm, zz), axis=0
        )


class BGSLuminosityFunction(KCorrectionLuminosityFunction):
    def __init__(self, cosmo: object | None = None) -> None:
        """
        BGS Luminosity function class
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.05, 0.6, 200)

    def luminosity_function(self, mm: ArrayLike, zz: ArrayLike) -> np.ndarray:
        """
        Schechter luminosity function values for given apparent magnitude and fitted params

        Parameters:
        -----------
        m_obs : float or array
            Observed apparent magnitude

        zz : float or array
            Redshift

        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]
        """
        if zz is None:
            zz = self.z_values

        # Convert to absolute magnitude at this redshift
        M_UV = self.M_UV(mm, zz)

        # make sure mm and zz are 2D arrays for broadcasting
        if isinstance(zz, (np.ndarray)) and isinstance(mm, (np.ndarray)):
            mm = mm[:, np.newaxis]

        alpha = -1.23
        M_star = 5 * np.log10(self.cosmo.h()) - 20.64 - 0.6 * zz
        y = M_UV - M_star

        g = (np.log(10) / 2.5) * 10 ** (-0.4 * (1 + alpha) * y) * np.exp(-(10 ** (-0.4 * y)))

        phi_star = 10 ** (-2.022 + 0.92 * zz)
        phi = phi_star * g
        return phi

    def K(self, zz: ArrayLike, ref_z: float = 0) -> np.ndarray:
        """
        K-correction accounting for redshifting effect on the band - 2004.12981
        """
        return 0.87 * (
            zz - ref_z
        )  # so this is the K-correction relative to z=0, but for example Smith parameters are fitted to z=0.1
