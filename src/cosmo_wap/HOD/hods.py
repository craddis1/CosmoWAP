# from matplotlib.pylab import ArrayLike
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from scipy.special import erf

from cosmo_wap.lib import utils
from cosmo_wap.lib.luminosity_funcs import BGSLuminosityFunction


class BaseHOD(ABC):
    """Abstract base class for Halo Occupation Distribution models."""

    @abstractmethod
    def get_hod_params(self, zz: np.ndarray, m_c: float | None = None) -> tuple:
        """Return HOD parameters as a tuple for the given redshift(s)."""
        ...

    @abstractmethod
    def HOD(self, zz: float, *args, **kwargs) -> np.ndarray:
        """Return the mean number of galaxies per halo N(M) at redshift zz."""
        ...


class YP(BaseHOD):
    """Halo occupation distribution (HOD) model from Yankelevich and Porciani 2018"""

    def __init__(self, PBBias, cosmo_funcs, survey_params, M0_func=None, NO_func=None):
        self.PBBias = PBBias
        self.survey_params = survey_params
        self.cosmo_funcs = cosmo_funcs
        self.M0_func = M0_func
        self.NO_func = NO_func

    def fit_params(self):
        """get fitted values of NO and M0 - speed up by using less points"""
        z_arr = np.linspace(self.cosmo_funcs.z_min, self.cosmo_funcs.z_max, 10)
        self.M0_func = self.fit_M0(z_arr)
        self.NO_func = self.fit_NO(z_arr)

    def get_hod_params(self, zz, m_c=None):
        """
        get HOD params for given redshift from fits to survey specifications
        """
        M0 = self.M0_func(zz)
        NO = self.NO_func(zz)
        return (M0, NO)

    @utils.array_memo()
    def HOD(self, zz, M0, NHO) -> np.ndarray:
        """
        Define the HOD - N(M, z) mean number of galaxies per halo - from Yankelevich and porciani 2018: arXiv:1807.07076

        Memoised: once fitted, PBBias evaluates this at one identical set of arguments nine
        times - get_number_density plus the eight get_galaxy_bias calls, which differ only
        in the halo bias weighting it. The fit itself moves M0/NHO, so those calls miss.
        """
        M = self.cosmo_funcs.M_halo[:, None]  # (R,1) for broadcasting

        theta_HoD = 1 + erf(2 * np.log10(M / M0))
        N_c = np.exp(-10 * (np.log10(M / M0)) ** 2) + 0.05 * theta_HoD  # central galaxies
        N_s = 0.003 * (M / M0) * theta_HoD

        return NHO * (N_c + N_s)  # (R,z)

    ########################### fit M0 and NO to given n_g and b_1
    def fit_M0(self, z_arr: np.ndarray) -> CubicSpline:
        """
        fit M0 from linear bias (b_1 is independent of NO)

        Solved in log10(M0), not M0 - b_1(M0) is ~flat in linear M0 near the fit's x0, so the
        secant method's finite-difference slope there is unreliable and can overshoot to a
        negative M0, which is nan under the HOD's log10(M/M0) and poisons every later
        iterate. b_1(M0) is well-behaved in log-M0, and M0 legitimately spans many decades
        across the fit's z range, so this is the natural variable regardless.
        """

        def objective(logM0, zz, NO=1):
            """
            diff between linear bias from PBS and survey specifications
            """
            M0 = 10**logM0
            return self.PBBias.general_galaxy_bias(self.PBBias.eulbias.b1, zz, M0, NO) / self.PBBias.number_density(
                zz, M0, NO
            ) - self.survey_params.b_1(zz)

        logM0_arr = np.array([newton(objective, x0=12.0, args=(z,), rtol=1e-8) for z in z_arr])

        # Spline log10(M0) itself, not M0 - only 10 knots span the whole z range and M0(z)
        # can fall by orders of magnitude between them (same reasoning as fit_NO below), so a
        # plain CubicSpline through the raw values can overshoot to negative M0 between knots.
        log_M0_spline = CubicSpline(z_arr, logM0_arr)
        return lambda zz: 10 ** log_M0_spline(zz)

    # now can find NO from n_g
    def fit_NO(self, z_arr: np.ndarray) -> CubicSpline:
        """
        fit NO from number density
        """

        def objective(NO, zz, M0):
            """
            diff between number density from PBS and survey specifications
            """
            return self.PBBias.number_density(zz, M0, NO) - self.survey_params.n_g(zz)

        # Use the secant method by not providing a derivative
        NO_arr = np.array([newton(objective, x0=2.0, args=(z, self.M0_func(z)), rtol=1e-5) for z in z_arr])

        # Spline log(NO), not NO - NO(z) can fall by orders of magnitude between the 10 fit
        # knots (e.g. steep dropoff at high z), and a plain CubicSpline through such values
        # overshoots below zero between knots, giving negative NO -> negative n_g downstream.
        # Splining in log guarantees NO > 0 everywhere by construction.
        log_NO_spline = CubicSpline(z_arr, np.log(NO_arr))
        return lambda zz: np.exp(log_NO_spline(zz))


class Smith_BGS(BaseHOD):
    def __init__(self, cosmo_funcs):
        self.cosmo_funcs = cosmo_funcs
        cosmo = cosmo_funcs.cosmo
        self.lf = BGSLuminosityFunction(cosmo)
        self.lf.get_Q = self.lf.get_Q2  # use number density based Q method

        # read the best-fitting HOD parameters
        # The parameters in the file are in the order:
        # Index:     0      1      2      3      4        5        6        7        8    9    10   11   12   13   14       15       16
        # Parameter: A_min, B_min, C_min, D_min, A_sigma, B_sigma, C_sigma, D_sigma, A_0, B_0, A_1, B_1, C_1, D_1, A_alpha, B_alpha, C_alpha
        _dir = os.path.dirname(__file__)
        self.params = np.loadtxt(os.path.join(_dir, "AbacusSummit_base_c000_ph000_best_params.txt"))

    def get_M_c(self, m_c, zz):
        """convert to absolute magnitude with K-correction to z=0.1, since the parameters are fitted to z=0.1
        Smith et al. use d_L in Mpc/h so absolute magnitudes include implicit h factor: M_h = M_phys - 5*log10(h)"""
        return (
            self.lf.M_UV(m_c, zz, ref_z=0.1)
            - self.evolutionary_correction(zz, z_ref=0.1)
            - 5 * np.log10(self.cosmo_funcs.h)
        )

    # so define the HOD parameters as a function of magnitude, using the fits from Smith et al. 2024
    # Original code taken from example notebooks in Smith et al. 2024.

    # NOTE: these functions are defined with repsect to a threshold magnitude - not magnitude
    def M_function(self, magnitude, A, B, C, D):
        # Function for Mmin and M1
        return (A + 12) + B * (magnitude + 20) + C * (magnitude + 20) ** 2 + D * (magnitude + 20) ** 3

    def M0_function(self, magnitude, A, B):  # logM0
        M0s = (A + 11) + B * (magnitude + 20)
        # M0s[M0s <=1.0] = 1.0
        return M0s

    def sigma_function(self, magnitude, A, B, C, D):
        return A + (B - A) / (1.0 + np.exp((C * (magnitude + 20 + D))))

    def alpha_function(self, magnitude, A, B, C):
        return A + B ** (-magnitude - 20 + C)

    def get_params(self, zz, m_c=20):
        M_c = self.get_M_c(m_c, zz)  # get cut in absolute magnitude
        # ok so all of these are functions of m_c and z
        Mmin = self.M_function(M_c, *self.params[:4])
        sigma = self.sigma_function(M_c, *self.params[4:8])
        M0 = self.M0_function(M_c, *self.params[8:10])
        M1 = self.M_function(M_c, *self.params[10:14])
        alpha = self.alpha_function(M_c, *self.params[14:])

        return Mmin, sigma, M0, M1, alpha

    #####################################################################################################

    def get_hod_params(self, zz, m_c=20):
        return (m_c,)

    def evolutionary_correction(self, zz, z_ref=0.1):
        """
        luminosity evolution correction E(z)=Q(z−z0​) (Q from McNaught-Roberts et al. 2014 (GAMA))
        So only applies to make sure the we have right units for the HOD parameters, since they are fitted to absolute magnitude at z=0.1
        """
        return 0.97 * (zz - z_ref)

    # HODs
    def number_centrals(self, logM, logMmin, sigma):  # Smith 2017/Zheng 2005
        return 0.5 * (1 + erf((logM - logMmin) / sigma))

    def number_satellites(self, logM, logM0, logM1, alpha):  # e.g. Smith 2017/Zehavi 2011
        N = (np.maximum(0, 10**logM - 10**logM0) / 10**logM1) ** alpha
        return N

    def HOD(self, zz, m_c=20):  # Smith 2017
        # use best fits from AbacusSummit_base_c000_ph000_best_params.txt with m_c =20
        logMmin, sigma, logM0, logM1, alpha = self.get_params(zz, m_c)

        logM = np.log10(self.cosmo_funcs.M_halo[:, None])  # (R,1) for broadcasting with z
        Ncen = self.number_centrals(logM, logMmin, sigma)
        Nsat = self.number_satellites(logM, logM0, logM1, alpha)
        return Ncen + Nsat * Ncen
