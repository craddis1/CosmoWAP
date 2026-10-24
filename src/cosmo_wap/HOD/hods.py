# from matplotlib.pylab import ArrayLike
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar, newton
from scipy.special import erf, gamma, gammaincc

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

        def solve(z):
            """
            root of objective - or, if b_1 is below what the HOD can reach, its least-biased M0
            """
            try:
                logM0 = newton(objective, x0=12.0, args=(z,), rtol=1e-8)
                if abs(objective(logM0, z)) < 1e-6:
                    return logM0
            except RuntimeError:
                pass
            # b_1(M0) bottoms out near log10(M0) ~ 11, where the satellite term takes over
            floor = minimize_scalar(objective, bounds=(8.0, 14.0), args=(z,), method="bounded")
            if floor.fun < 0:
                # reachable - newton set off from the flat bottom and overshot the halo mass grid (from b_1 ~ 1.7
                # at z = 0.1). Bracket the root above the minimum: b_1 rises again below it, and that high-mass
                # branch is where newton lands whenever it converges
                top = np.log10(self.cosmo_funcs.M_halo.max())
                ceiling = objective(top, z)
                if ceiling < 0:
                    raise ValueError(
                        f"YP HOD cannot reach b_1 = {self.survey_params.b_1(z):.3f} at z = {z:.2f} - "
                        f"at most {ceiling + self.survey_params.b_1(z):.3f} at the top of the halo mass grid"
                    )
                return brentq(objective, floor.x, top, args=(z,), rtol=1e-8)
            warnings.warn(
                f"YP HOD cannot reach b_1 = {self.survey_params.b_1(z):.3f} at z = {z:.2f} - "
                f"using its minimum, {floor.fun + self.survey_params.b_1(z):.3f}"
            )
            return floor.x

        logM0_arr = np.array([solve(z) for z in z_arr])

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
    """Luminosity-dependent HOD for the DESI BGS from Smith et al. 2024 (2312.08792), fitted on the AbacusSummit z = 0.2
    snapshot, and carried to other redshifts as in Smith et al. 2017 (1701.06581) Sec. 3.2.

    Ported from hodpy (https://github.com/amjsmith/hodpy), the code behind both papers - BSD-3, notice in
    HODPY_LICENSE.txt. HOD_BGS with redshift_evolution=True in hodpy/hod_bgs_abacus.py (abacus branch, c0259e4), and
    HOD_BGS in hodpy/hod_bgs.py (master, d303bef) for Smith et al. 2017:
    1. the observed threshold at z maps to the threshold at z_fit with the same number density in the target
       luminosity function - HOD_BGS.Mmin etc. (lf.Phi_cumulative, then lf.magnitude at z0)
    2. the HOD parameters are evaluated there, so sigma and alpha are fixed at fixed number density
    3. M_min, M_0 and M_1 slide by a factor f(z) until the HOD on this cosmology's halo mass function reproduces the
       target number density - HOD_BGS.slide_factor
    So n_g is the target luminosity function by construction, and the HOD sets the clustering.

    The observed threshold takes the survey's K, the GAMA colour-dependent K-correction at its cut (threshold_K below)
    - the same K its b_e, lum_derivs and SetSurveyFunctions.set_selection see.
    """

    def __init__(self, PBBias, cosmo_funcs):
        self.PBBias = PBBias  # its halo mass function and mass grid weights, for the slide factor
        self.cosmo_funcs = cosmo_funcs
        cosmo = cosmo_funcs.cosmo
        self.lf = BGSLuminosityFunction(cosmo)
        self.lf.get_Q = self.lf.get_Q2  # use number density based Q method
        # the survey's K-correction (SurveyParams.BGS), fixed at its own cut - so a bright cut, shifted cuts and other
        # cosmologies all keep it. lf.get_be and lum_derivs then differentiate the same K
        self.K = PBBias.survey_params.K
        self.lf.K = self.K

        # read the best-fitting HOD parameters
        # The parameters in the file are in the order:
        # Index:     0      1      2      3      4        5        6        7        8    9    10   11   12   13   14       15       16
        # Parameter: A_min, B_min, C_min, D_min, A_sigma, B_sigma, C_sigma, D_sigma, A_0, B_0, A_1, B_1, C_1, D_1, A_alpha, B_alpha, C_alpha
        _dir = os.path.dirname(__file__)
        self.params = np.loadtxt(os.path.join(_dir, "AbacusSummit_base_c000_ph000_best_params.txt"))

        self.z_fit = 0.2  # redshift of the snapshot the HODs are fitted on - hodpy HOD_BGS z0
        self.mag_faint = -17  # the fits are not trusted fainter than this - hodpy HOD_BGS mag_faint_extrapolate

        # target luminosity function: the Schechter fit to GAMA of Loveday et al. 2012, M* and phi* evolving through Q
        # and P - Smith et al. 2017 Sec. 3.2, values from hodpy lookup/gama/lf_params.dat. Smith et al. switch to an
        # SDSS luminosity function below z ~ 0.15; GAMA is kept throughout here, as that switch is sharp enough to
        # spike b_e
        self.phi_star, self.M_star, self.alpha_lf, self.P_lf, self.Q_lf = 0.94e-2, -20.70, -1.23, 1.8, 0.7

        # threshold <-> number density at z_fit, on hodpy's grid (LuminosityFunction.magnitude, master)
        self._M_grid = np.arange(-25, 0, 0.001)
        self._log_n_fit = np.log10(self.target_lf(self._M_grid, self.z_fit))

    def get_M_obs(self, m_c, zz):
        """convert to absolute magnitude with K-correction to z=0.1 - the 0.1M_r band of the fits, which is not the
        redshift they are fitted at (z_fit)
        Smith et al. use d_L in Mpc/h so absolute magnitudes include implicit h factor: M_h = M_phys - 5*log10(h)"""
        if callable(m_c):  # a cut that moves with z - the bright side of a fixed-fraction split
            m_c = m_c(zz)
        return absolute_magnitude(self.cosmo_funcs.cosmo, m_c, zz, self.K(zz))

    def get_M_c(self, m_c, zz):
        """cut in absolute magnitude at z_fit, where the HOD parameters are defined"""
        return self.match_z_fit(self.get_M_obs(m_c, zz), zz)

    def match_z_fit(self, M_obs, zz):
        """threshold at z_fit with the same target number density as M_obs at zz - Smith et al. 2017 Sec. 3.2, hodpy
        HOD_BGS.Mmin etc. (abacus branch, redshift_evolution=True)"""
        return np.interp(np.log10(self.target_lf(M_obs, zz)), self._log_n_fit, self._M_grid)

    def target_lf(self, M, zz):
        """number density brighter than M [h^3 Mpc^-3] in the target luminosity function, M*(z) = M* - Q(z - 0.1) and
        phi*(z) = phi* 10^(0.4 P z) - hodpy LuminosityFunctionSchechter.Phi_cumulative (luminosity_function.py,
        master)"""
        t = 10 ** (0.4 * (self.M_star - self.Q_lf * (zz - 0.1) - M))
        a = self.alpha_lf
        # upper incomplete Gamma(a + 1, t) by recurrence, as hodpy - gammaincc needs a positive first argument
        gamma_t = (gammaincc(a + 2, t) * gamma(a + 2) - t ** (a + 1) * np.exp(-t)) / (a + 1)
        return self.phi_star * 10 ** (0.4 * self.P_lf * zz) * gamma_t

    def target_number_density(self, m_c, zz):
        """n_g(m_c, zz) - the target luminosity function at the observed threshold, which the slide factor makes the HOD
        reproduce. Same signature as a luminosity function's number_density, so PBBias can hand it to lf"""
        return self.target_lf(self.get_M_obs(m_c, zz), zz)

    # so define the HOD parameters as a function of magnitude, using the fits from Smith et al. 2024
    # Original code taken from example notebooks in Smith et al. 2024.

    # NOTE: these functions are defined with repsect to a threshold magnitude - not magnitude
    def M_function(self, magnitude, A, B, C, D):
        # Function for Mmin and M1
        logM = (A + 12) + B * (magnitude + 20) + C * (magnitude + 20) ** 2 + D * (magnitude + 20) ** 3
        # the cubic shoots off fainter than the fitted magnitudes - past mag_faint hodpy (HOD_BGS.__Mmin_z0/__M1_z0,
        # abacus branch) swaps it for a curve with the same value and slope there that flattens to 1 dex below it
        x = self.mag_faint
        slope = B + 2 * C * (x + 20) + 3 * D * (x + 20) ** 2
        logM_x = (A + 12) + B * (x + 20) + C * (x + 20) ** 2 + D * (x + 20) ** 3
        return np.where(magnitude > x, logM_x - 1 + np.exp(slope * (magnitude - x)), logM)

    def M0_function(self, magnitude, A, B):  # logM0
        M0s = (A + 11) + B * (magnitude + 20)
        # M0s[M0s <=1.0] = 1.0
        # and M0 = 0 fainter than mag_faint "to avoid any problems" - hodpy HOD_BGS.M0 (abacus branch)
        return np.where(magnitude > self.mag_faint, -np.inf, M0s)

    def sigma_function(self, magnitude, A, B, C, D):
        return A + (B - A) / (1.0 + np.exp((C * (magnitude + 20 + D))))

    def alpha_function(self, magnitude, A, B, C):
        return A + B ** (-magnitude - 20 + C)

    def get_params(self, zz, m_c=20):
        return self.evolved_params(zz, self.get_M_obs(m_c, zz))

    def evolved_params(self, zz, M_obs):
        """HOD parameters at zz for the observed threshold M_obs, mass parameters as log10 - steps 1-3 of the class
        docstring"""
        M_c = self.match_z_fit(M_obs, zz)  # get cut in absolute magnitude at z_fit
        # ok so all of these are functions of m_c and z
        # all five at the matched magnitude, as hodpy master - the abacus branch takes M0 at the unmatched one
        Mmin = self.M_function(M_c, *self.params[:4])
        sigma = self.sigma_function(M_c, *self.params[4:8])
        M0 = self.M0_function(M_c, *self.params[8:10])
        M1 = self.M_function(M_c, *self.params[10:14])
        alpha = self.alpha_function(M_c, *self.params[14:])

        log_f = self.slide_factor(zz, M_obs, Mmin, sigma, M0, M1, alpha)
        return Mmin + log_f, sigma, M0 + log_f, M1 + log_f, alpha

    def slide_factor(self, zz, M_obs, logMmin, sigma, logM0, logM1, alpha):
        """log10 f - the factor Smith et al. 2017 Sec. 3.2 multiply M_min, M_0 and M_1 by so the HOD on the halo mass
        function at zz reproduces the target number density. hodpy (HOD_BGS.slide_factor) tabulates f on a grid of
        magnitude and z with a scalar root find per point; one cut on one z grid only needs a secant solve over all z"""
        n_target = self.target_lf(M_obs, zz)
        if not np.all(np.isfinite(n_target) & (n_target > 0)):
            raise ValueError("Smith BGS target number density must be finite and positive")
        log_n = np.log(n_target)
        n_h = self.PBBias.hmf.n_h(zz)

        def residual(log_f):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                N = self.occupation(logMmin + log_f, sigma, logM0 + log_f, logM1 + log_f, alpha)
                return np.log((self.PBBias._mass_weights @ (N * n_h)).squeeze()) - log_n

        x0, x1 = np.zeros_like(log_n), np.full_like(log_n, -0.05)
        r0, r1 = residual(x0), residual(x1)
        for _ in range(30):
            converged = np.isfinite(r1) & (np.abs(r1) < 1e-8)
            if np.all(converged):
                return x1
            active = np.isfinite(r0) & np.isfinite(r1) & (r1 != r0) & ~converged
            if not np.any(active):
                break
            # a redshift that has already converged has r1 == r0 - it stays put
            with np.errstate(invalid="ignore", over="ignore"):
                step = np.divide(r1 * (x1 - x0), r1 - r0, out=np.zeros_like(r1), where=active)
            x0, r0, x1 = x1, r1, x1 - step
            r1 = residual(x1)

        # Only the failed redshifts need a bracketed solve. Use n - n_target here: zero occupation is a valid
        # endpoint, while its log is -inf and cannot give the secant method a slope.
        roots = np.atleast_1d(x1).copy()
        n_h = np.broadcast_to(n_h, (n_h.shape[0], roots.size))
        params = np.broadcast_arrays(logMmin, sigma, logM0, logM1, alpha, n_target)
        failed = ~np.isfinite(r1) | (np.abs(r1) >= 1e-8)
        for i in np.flatnonzero(failed):
            Mmin_i, sigma_i, M0_i, M1_i, alpha_i, target_i = (p.flat[i] for p in params)

            def objective(log_f):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    N = self.occupation(Mmin_i + log_f, sigma_i, M0_i + log_f, M1_i + log_f, alpha_i)[:, 0]
                    return self.PBBias._mass_weights @ (N * n_h[:, i]) - target_i

            lo, hi = -1.0, 1.0
            for _ in range(10):
                if objective(lo) >= 0 and objective(hi) <= 0:
                    break
                lo, hi = 2 * lo, 2 * hi
            else:
                raise ValueError(
                    "Smith BGS slide factor did not converge - target number density out of the HOD's reach?"
                )
            roots[i] = brentq(objective, lo, hi, xtol=1e-12)
        return roots.reshape(np.shape(log_n))

    #####################################################################################################

    def get_hod_params(self, zz, m_c=20):
        return (m_c,)

    # HODs
    def number_centrals(self, logM, logMmin, sigma):  # Smith 2017/Zheng 2005
        # Smith et al. 2024 Sec. 2.2: the error function step with a pseudo-Gaussian spline kernel in place of the
        # Gaussian, so the tails are exactly 0 and 1 - hodpy spline.cumulative_spline_kernel(logM, logMmin,
        # sigma/sqrt(2))
        x = (logM - logMmin) / (sigma * np.sqrt(6))  # kernel width sig sqrt(12), with sig = sigma/sqrt(2)
        a = np.abs(x)
        a2, u = a * a, 1 - np.minimum(a, 1)  # u = 0 past |x| = 1, where the integral is constant
        integral = np.where(a < 0.5, a - 2 * a2 * a + 1.5 * a2 * a2, 0.375 - 0.5 * (u * u) ** 2)
        return 0.5 + np.sign(x) * integral / 0.75

    def number_satellites(self, logM, logM0, logM1, alpha):  # e.g. Smith 2017/Zehavi 2011
        N = (np.maximum(0, 10**logM - 10**logM0) / 10**logM1) ** alpha
        return N

    def occupation(self, logMmin, sigma, logM0, logM1, alpha):
        """N(M) on the halo mass grid for given HOD parameters - shared by HOD and the slide factor's root find"""
        logM = np.log10(self.cosmo_funcs.M_halo[:, None])  # (R,1) for broadcasting with z
        Ncen = self.number_centrals(logM, logMmin, sigma)
        Nsat = self.number_satellites(logM, logM0, logM1, alpha)
        return Ncen + Nsat * Ncen

    def HOD(self, zz, m_c=20):  # Smith 2017
        # use best fits from AbacusSummit_base_c000_ph000_best_params.txt with m_c =20
        return self.evolved_HOD(zz, self.get_M_obs(m_c, zz))

    @utils.array_memo()
    def evolved_HOD(self, zz, M_obs):
        """N(M, z) for the observed threshold M_obs

        Memoised: PBBias evaluates it at one cut ten times - get_number_density plus the nine get_galaxy_bias calls -
        and a miss costs a slide-factor root find on top of the occupation."""
        return self.occupation(*self.evolved_params(zz, M_obs))


#####################################################################################################
# K-correction of the Smith BGS pipeline - Smith et al. 2017 Sec. 4.3, ported from hodpy as Smith_BGS above

# median GAMA r-band K-correction to the 0.1 band in 7 bins of 0.1(g-r), sum_i A_i (z - 0.1)^(4-i) with A_4 =
# -2.5 log10(1.1) in every bin - Smith et al. 2017 Sec. 4.3 table, to full precision from hodpy
# lookup/k_corr_rband_z01.dat (master). Columns: median colour, A_0, A_1, A_2, A_3
_GAMA_K = np.array(
    [
        [0.130634, -45.32783517, 35.27660354, -6.60434164, -0.480538],
        [0.298124, -20.07718761, 20.14480008, -4.6200285, -0.04824625],
        [0.443336, -10.98217672, 14.35677081, -3.67641154, 0.33946073],
        [0.603434, -3.42775232, 9.47765132, -2.70330179, 0.76463324],
        [0.784644, 6.71683404, 3.25021491, -1.17611261, 1.11334705],
        [0.933226, 16.76083847, -2.51354452, 0.35130908, 1.30680013],
        [1.06731, 20.30236651, -4.18856337, 0.56185062, 1.49435891],
    ]
)
_GAMA_K_A4 = -0.10348171


def absolute_magnitude(cosmo, m, zz, K):
    """0.1M_r - 5 log10 h for apparent magnitude m at zz, d_L in Mpc/h - Smith et al. 2017 Sec. 4.3, hodpy
    GAMA_KCorrection.absolute_magnitude"""
    return m - 5 * np.log10(cosmo.luminosity_distance(zz) * cosmo.h()) - 25 - K


def gama_k(zz, colour):
    """K-correction [mag] at zz of a galaxy of 0.1(g-r) colour - the coefficients interpolated linearly between the bin
    medians, the colour clipped to them, and past z = 0.5 the line through K(0.4) and K(0.5) - hodpy
    GAMA_KCorrection.k (k_correction.py, master, its default linear interpolation)"""
    med = _GAMA_K[:, 0]
    c = np.clip(colour, med[0], med[-1])
    A0, A1, A2, A3 = (np.interp(c, med, _GAMA_K[:, i]) for i in range(1, 5))

    def poly(z):
        dz = z - 0.1
        return (((A0 * dz + A1) * dz + A2) * dz + A3) * dz + _GAMA_K_A4

    K_4, K_5 = poly(0.4), poly(0.5)
    return np.where(zz <= 0.5, poly(np.minimum(zz, 0.5)), K_5 + (zz - 0.5) * (K_5 - K_4) / 0.1)


def gama_k_mean(zz, mean, rms):
    """gama_k averaged over a Gaussian in colour, in closed form: at fixed z it is linear between the bin medians and
    flat outside them, so a sum of hinges max(c - median, 0), each with a closed-form Gaussian mean"""
    med = _GAMA_K[:, 0]
    k = gama_k(zz, med[:, None])  # (7, z)
    slope = np.diff(k, axis=0) / np.diff(med)[:, None]
    kink = np.diff(slope, axis=0, prepend=0, append=0)  # change of slope at each median
    t = (mean - med[:, None]) / rms
    hinge = rms * (np.exp(-t * t / 2) / np.sqrt(2 * np.pi) + t * 0.5 * (1 + erf(t / np.sqrt(2))))
    return k[0] + np.sum(kink * hinge, axis=0)


def gama_colours(M, zz):
    """0.1(g-r) colours at magnitude M (0.1M_r - 5 log10 h) and zz: the blue fraction, then the mean and rms of the blue
    and of the red sequence, each a Gaussian - Smith et al. 2017 Sec. 4.2, as hodpy Colour (colour.py, master)
    fraction_blue, blue_mean, blue_rms, red_mean and red_rms. Where the paper's text differs (red rms evolving as
    0.5(z - 0.1), faint blue slope 0.03 and a hard switch to it at -18.7, no floor of 4.45 in f_blue) this follows
    hodpy, which made the mocks"""
    dz = np.maximum(zz - 0.1, 0)  # the sequences are held fixed below z = 0.1
    dz_mean = np.minimum(dz, 0.3)  # and their means above z = 0.4
    blue_mean = np.log10(1e9 ** (0.62 - 0.11 * (M + 20)) + 1e9 ** (0.4 - 0.0286 * (M + 16))) / 9 - 0.25 * dz_mean
    blue_rms = np.clip(0.12 + 0.02 * (M + 20), 0, 0.15) + 0.2 * dz
    red_mean = 0.932 - 0.032 * (M + 20) - 0.18 * dz_mean
    red_rms = 0.07 + 0.01 * (M + 20) + (0.05 + 0.1 * dz) * dz
    f_blue = 0.2 * M + np.clip(4.4 + (1.2 + 0.5 * (zz - 0.1)) * (zz - 0.1), 4.45, 10)
    f_blue = np.clip(np.maximum(f_blue, 0.46 + 0.07 * (M + 20)), 0, 1)
    return f_blue, (blue_mean, blue_rms), (red_mean, red_rms)


def threshold_K(cosmo, m_c, zz):
    """K-correction [mag] at the flux limit m_c: gama_k averaged over the colours of galaxies at the threshold
    magnitude, which K itself sets - so the two are iterated to agreement.

    The mocks K-correct each galaxy by its own colour (hodpy galaxy_catalogue.add_apparent_magnitude); a forecast needs
    one K(z), and the colour average at the threshold reproduces their counts at the flux limit to first order in the
    spread of K. Not in hodpy - the one piece of the pipeline that is not a port.

    Q and b_e then come from a sharp threshold, which the per-galaxy K of the mocks is not: counted galaxy by galaxy,
    Q is up to ~10% higher and b_e lower by ~0.2 at z = 0.2 and ~1 at z = 0.4 (r < 20.175), b_e because a single K's
    slope books the colour mix changing at the threshold as threshold motion rather than as evolution"""
    if callable(m_c):
        m_c = m_c(zz)
    K = np.zeros_like(zz)
    for _ in range(50):
        f_blue, blue, red = gama_colours(absolute_magnitude(cosmo, m_c, zz, K), zz)
        K_new = f_blue * gama_k_mean(zz, *blue) + (1 - f_blue) * gama_k_mean(zz, *red)
        if np.max(np.abs(K_new - K)) < 1e-10:
            return K_new
        K = K_new
    raise ValueError("K-correction at the BGS threshold did not converge")
