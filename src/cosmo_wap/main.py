from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from cosmo_wap.HOD.peak_background_bias import PBBias
from cosmo_wap.lib import utils
from cosmo_wap.lib.unpack import UnpackClassWAP
from cosmo_wap.survey_params import SetSurveyFunctions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # for type checking
    from cosmo_wap.lib.utils import Emulator
    from cosmo_wap.survey_params import SurveyParams

__all__ = ["ClassWAP"]


class ClassWAP(UnpackClassWAP):
    r"""
        Willkommen, Bienvenue, Welcome...
           ______                         _       _____    ____
          / ____/___  _________ ___  ____| |     / /   |  / __ \
         / /   / __ \/ ___/ __ `__ \/ __ \ | /| / / /| | / /_/ /
        / /___/ /_/ (__  ) / / / / / /_/ / |/ |/ / ___ |/ ____/
        \____/\____/____/_/ /_/ /_/\____/|__/|__/_/  |_/_/

        Main class - takes in cosmology from CLASS and survey parameters and then can called to generate cosmology (f,P(k),P'(k),D(z) etc) and all other biases including relativstic parts
    """

    def __init__(
        self,
        cosmo,
        survey_params: SurveyParams.SurveyBase | list[SurveyParams.SurveyBase] | None = None,
        compute_bias: bool = False,
        hmf: str = "Tinker10",
        hod: str = "YP",
        emulator: bool | Emulator = False,
        verbose: bool = True,
        params: dict[str, float] | None = None,
        fast: bool = False,
        nonlin: bool = False,
    ) -> None:
        """
        Initialise CosmoWAP from a CLASS cosmology and (optionally) survey parameters.

        Interpolates cosmological background quantities (e.g. D, f, H etc) from CLASS,
        Compute linear P(k) and optionally non-linear P(k) using halofit or an emulator,
        If survey parameters are passed, compute bias funtions for given survey(s) and optionally compute bias functions from HMF/HOD relations.

        Parameters
        ----------
        cosmo : classy.Class
            A CLASS instance.
        survey_params : SurveyBase or list[SurveyBase], optional
            List or single survey parameter objects.
        compute_bias : bool
            If True, derive  higher order bias functions and scale-dependent PNG biases from the HMF/HOD.
        hmf : str
            Halo mass function model passed to ``PBBias`` (default ``'Tinker10'``).
        emulator : bool or Emulator
            If True, initialise a CosmoPower emulator internally. Pass a pre-loaded
            ``Emulator`` instance to reuse it across multiple ``ClassWAP`` objects.
        verbose : bool
        params : dict, optional
            Pre-computed cosmological parameters (h, Omega_m, ...) to load directly
            instead of querying CLASS — useful for speeding up MCMC sampling.
        fast : bool
            If True (and ``nonlin`` is False), skip building the non-linear P(k,z) grid.
        nonlin : bool
            If True, always build the non-linear halofit/emulator P(k,z) grid.
        """
        self.nonlin = nonlin  # use nonlin halofit powerspectra
        self.growth2 = False  # second order growth corrections to F2 and G2 kernels
        self.n = 128  # default n for integrated terms - used currently in forecast stuff
        intcomponents = ["LxNPP", "ISWxNPP", "TDxNPP", "LxL", "LxTD", "LxISW", "ISWxISW", "ISWxTD", "TDxTD"]
        self.term_list = [
            "NPP",
            "RR1",
            "RR2",
            "WA1",
            "WA2",
            "WAGR",
            "WARR",
            "WS",
            "RRGR",
            "WSGR",
            "Full",
            "GR1",
            "GR2",
            "GRX",
            "Loc",
            "Eq",
            "Orth",
            "IntInt",
            "IntNPP",
            "GRI",
            "GRL",
            "GR",
        ] + intcomponents  # list of terms currently implemented. Includes composites - see pk/combined.py etc

        # so we can use emulators for Pk to speed up sampling cosmological parameter space
        if emulator:
            self.emulator = True
            K_MAX_h = 10  # in Units of [1/Mpc] - is limit of emulator
            if emulator is True:
                self.emu = utils.Emulator()  # initiate nested emulator class using CosmoPower
            else:
                self.emu = emulator  # use pre-loaded estimators

        else:
            self.emu = None
            self.emulator = False
            K_MAX_h = cosmo.pars["P_k_max_1/Mpc"]  # in Units of [1/Mpc]

        #############  interpolate cosmological functions  - this is quick as we call class to calculate a limited set of parameters
        self.compute_background(cosmo, params)

        ##################################################################################
        # get powerspectra
        self.K_MAX = K_MAX_h / self.h  # in Units of [h/Mpc]
        k = np.logspace(-5, np.log10(self.K_MAX), num=400)
        self.Pk, self.Pk_d, self.Pk_dd = self.get_pk(k)

        # setup surveys and compute all bias params including for multi tracer case...
        self.setup_hod_hmf(compute_bias, hmf, hod)
        if survey_params:
            self.update_survey(survey_params, verbose=verbose)

        if nonlin or not fast:
            # get 2D interpolated halofit powerspectrum function (k,z) - need maximum redshift here
            z_range = np.linspace(0, 5, 50)  # for integrated effects need all values below maximum redshift
            self.Pk_NL = self.get_Pk_NL(k, z_range)

    def compute_background(self, cosmo, params: dict[str, float] | None) -> None:
        """Use class to compute background without much overhead."""
        zz = np.linspace(0, 10, 100)  # for now we have a redshift range up z=10 - so sample every 0.1 z
        self.D = CubicSpline(zz, cosmo.scale_independent_growth_factor(zz))
        self.f = CubicSpline(zz, cosmo.scale_independent_growth_factor_f(zz))
        self.cosmo = cosmo
        self.load_cosmology(params)  # load cosmological paramerters into object
        self.H_c = CubicSpline(zz, cosmo.Hubble(zz) * (1 / (1 + zz)) / self.h)  # now in h/Mpc! - is conformal
        self.dH_c = self.H_c.derivative(nu=1)  # first derivative wrt z
        xi_zz = self.h * cosmo.comoving_distance(zz)  # Mpc/h
        self.comoving_dist = CubicSpline(zz, xi_zz)
        self.d_to_z = CubicSpline(xi_zz, zz)  # useful to map other way
        # let see if faster use simpler stuff for Om_m # less than 0.2% at z=5 and 0.5% at z=10
        # self.Om_m          = CubicSpline(zz,cosmo.Om_m(zz))
        self.Om_m = CubicSpline(zz, (2 / 3) * (1 + (1 + zz) * self.dH_c(zz) / self.H_c(zz)))
        # misc
        self.c = 2.99792e5  # km/s

    def load_cosmology(self, params: dict[str, float] | None) -> ClassWAP:
        """Unify the way we call cosmological parameters so they are defined within cosmo_funcs"""

        if params is None:
            self.A_s = self.cosmo.get_current_derived_parameters(["A_s"])["A_s"]  # from cosmo (classy)
            self.h = self.cosmo.h()
            self.Omega_m = self.cosmo.Omega_m()
            self.Omega_b = self.cosmo.Omega_b()
            self.Omega_cdm = self.Omega_m - self.Omega_b
            self.n_s = self.cosmo.n_s()
            if not self.emulator:
                self.sigma8 = self.cosmo.sigma8()  # not computed in classy if it doesn't compute P(k)
        else:
            self.__dict__.update(params)  # get from params dict (quicker)

        return self

    ####################################################################################
    # get power spectras - linear and non-linear Halofit
    def get_class_powerspectrum(
        self, kk: np.ndarray, zz: float = 0
    ) -> np.ndarray:  # h are needed to convert to 1/Mpc for k then convert pk back to (Mpc/h)^3
        return np.array([self.cosmo.pk_lin(ki, zz) for ki in kk * self.h]) * self.h**3

    def get_pk(self, k: np.ndarray) -> tuple[CubicSpline, CubicSpline, CubicSpline]:
        """get Pk and its k derivatives"""
        if self.emulator:
            params_lin = {
                "omega_b": [self.Omega_b * self.h**2],  # in terms of Omega_b*h**2
                "omega_cdm": [self.Omega_m * self.h**2 - self.Omega_b * self.h**2],
                "h": [self.h],
                "n_s": [self.n_s],
                "ln10^{10}A_s": [np.log(10**10 * self.A_s)],
                "z": [0],  # just linear pk at z=0
            }

            # originally maps to log10(Pk)
            plin = self.emu.Pk.predictions_np(params_lin)[0]  # can use for non-lin part as well
            Plin = 10.0 ** (plin) * self.h**3  # is array in k (k defined by emu_k)
            k = self.emu.k / self.h  # set k_modes to output of emulator
        else:
            Plin = self.get_class_powerspectrum(k, 0)  # just always get present day power spectrum

        Pk = CubicSpline(k, Plin)  # get linear power spectrum
        Pk_d = Pk.derivative(nu=1)
        Pk_dd = Pk.derivative(nu=2)
        return Pk, Pk_d, Pk_dd

    def get_Pk_NL(
        self, kk: np.ndarray, zz: np.ndarray
    ) -> Callable[[ArrayLike, ArrayLike], np.ndarray]:  # for halofit non-linear power spectrum
        """
        Get 2D (k,z) interpolated nonlinear power spectrum - has non-trivial time dependence
        only want non-linear correction on small scales - use linear P(k) for large scales
        We also factor in (self.D(zz)**2) dependence as this is already factored out.
        """

        if self.emulator:
            # all input arrays must have the same shape
            n = len(zz)
            batch_params_lin = {
                "omega_b": [self.Omega_b * self.h**2] * n,
                "omega_cdm": [self.Omega_m * self.h**2 - self.Omega_b * self.h**2] * n,
                "h": [self.h] * n,
                "n_s": [self.n_s] * n,
                "ln10^{10}A_s": [np.log(10**10 * self.A_s)] * n,
                "z": zz,  # just linear pk at z=0
            }

            # hmcode parameters - can play around here
            batch_params_hmcode = {"c_min": [3] * n, "eta_0": [0.6] * n}

            # combine parameters
            batch_params_nlboost = {**batch_params_lin, **batch_params_hmcode}
            total_log_power = self.emu.Pk.predictions_np(batch_params_lin) + self.emu.Pk_NL.predictions_np(
                batch_params_nlboost
            )
            pks = (10.0 ** (total_log_power) * self.h**3).T / (self.D(zz) ** 2)  # make (k,z) shape
            kk = self.emu.k / self.h  # set k_modes to output of emulator

        else:
            # so most efficiently get 2D grid using cosmo.get_pk - class
            kk_base = kk[:, np.newaxis, np.newaxis] * self.h
            kk_arr = np.broadcast_to(kk_base, (kk.size, zz.size, 1))  # we do this rearranging for cosmo.get_pk
            pk_nonlin = (
                self.h**3
                * self.cosmo.get_pk(kk_arr, zz, kk.size, zz.size, 1)[..., 0]
                / (self.D(zz) ** 2)[np.newaxis, :]
            )

            # use linear on large scales...
            pk_lin = np.broadcast_to(self.Pk(kk)[:, np.newaxis], (pk_nonlin.shape))
            pks = np.where(pk_nonlin > pk_lin, pk_nonlin, pk_lin)

        interp = RegularGridInterpolator((kk, zz), pks, bounds_error=False)

        def f(x, y):
            """
            A wrapper for the RegularGridInterpolator that allows calling with individual coordinates.
            """
            return interp((x, y))

        return f

    def pk(self, k: ArrayLike) -> np.ndarray:
        """After K_MAX we just have K^{-3} power law - just linear power spectra"""
        return np.where(k > self.K_MAX, self.Pk(self.K_MAX) * (k / self.K_MAX) ** (-3), self.Pk(k))

    ###########################################################
    def setup_hod_hmf(self, compute_bias: bool, hmf: str, hod: str = "YP", R: np.ndarray = None) -> None:
        """
        Setup for HOD/HMF stuff and store some computation - cosmology stuff
        """
        self.compute_bias = compute_bias
        self.hmf = hmf
        self.hod = hod
        if compute_bias:  # precompute for HOD/HMF
            if R is not None:
                self.R = R
            else:
                self.R = np.logspace(-1.5, 1.5, 100, dtype=np.float32)  # radius [Mpc/h]

            # precompute sigma^2 - for HMF/HOD
            self.sigmaR0 = self.sigma_R_n(self.R, 0)
            self.sigmaR1 = self.sigma_R_n(self.R, -1)
            self.sigmaR2 = self.sigma_R_n(self.R, -2)

            # store sigmas as functions of (z,R)
            self.sig_R = {}
            self.sig_R["0"] = lambda xx: self.sigmaR0 * self.D(xx) ** 2
            self.sig_R["1"] = lambda xx: self.sigmaR1 * self.D(xx) ** 2
            self.sig_R["2"] = lambda xx: self.sigmaR2 * self.D(xx) ** 2

            self.delta_c = 1.686  # from spherical collapse model

            # for critical density
            GG = 4.300917e-3  # [pc M_sun^-1 (km/s)^2]
            G = GG / (1e6 * self.c**2)  # gravitational constant # [Mpc M_sun^-1]
            self.rho_crit = lambda xx: (
                3 * (self.H_c(xx) * (1 + xx)) ** 2 / (8 * np.pi * G)
            )  # in units of h^2 Mo/ Mpc^3 where Mo is solar mass
            self.rho_m = lambda xx: self.rho_crit(xx) * self.Om_m(xx)  # in units of h^2 Mo/ Mpc^3

            # M_halo as function of z and R - refers to the mass enclosed within the radius
            # CHANGED: so defined at redshift 0
            self.M_halo = (4 * np.pi * self.rho_m(0) * self.R**3) / 3  # [M_sun/h] - array in R as func of z

    # read in survey_params class and define self.survey
    def _process_survey(
        self, survey_params: SurveyParams.SurveyBase, compute_bias: bool, hmf: str, hod: str, verbose: bool = True
    ) -> list[SetSurveyFunctions]:
        """
        Get bias funcs for a given survey - compute biases from HMF and HOD relations if flagged.
        Returns a list of SetSurveyFunctions (one for single-tracer, two for split-tracer).
        """
        class_bias = SetSurveyFunctions(survey_params, compute_bias)
        class_bias.z_survey = np.linspace(class_bias.z_range[0], class_bias.z_range[1], 100)

        if compute_bias:  # compute biases from HMF and HOD
            if verbose:
                logger.info("Computing bias functions...")

            if survey_params.need_hod and hasattr(survey_params, "split"):
                # for multi-tracer stuff for BGS with HOD we need to compute split stuff now
                total = class_bias
                bright = utils.copy(class_bias)
                faint = utils.copy(class_bias)
                # ok so call total and bright
                pb_class_T = PBBias(self, survey_params, hmf, hod, m_c=survey_params.cut)
                pb_class_T.add_bias_attr(class_bias)
                pb_class_B = PBBias(self, survey_params, hmf, hod, m_c=survey_params.split)
                pb_class_B.add_bias_attr(bright)

                # now get faint from total and bright
                zz = class_bias.z_survey
                n_T = total.n_g(zz)
                n_B = bright.n_g(zz)
                w_F = 1 / (n_T - n_B)  # faint weight

                def get_faint_bias(total_bias, bright_bias):
                    return CubicSpline(zz, (n_T * total_bias(zz) - n_B * bright_bias(zz)) * w_F)

                faint.n_g = CubicSpline(zz, n_T - n_B)
                faint.b_1 = get_faint_bias(total.b_1, bright.b_1)
                faint.b_2 = get_faint_bias(total.b_2, bright.b_2)
                faint.g_2 = get_faint_bias(total.g_2, bright.g_2)
                faint.Q = CubicSpline(zz, (n_T * total.Q(zz) - n_B * bright.Q(zz)) * w_F)
                faint.be = CubicSpline(zz, np.gradient(np.log(n_T - n_B), np.log(1 + zz)))

                # PNG biases
                for png_type in ("loc", "eq", "orth"):
                    total_png = getattr(total, png_type)
                    bright_png = getattr(bright, png_type)
                    faint_png = utils.copy(total_png)
                    faint_png.b_01 = get_faint_bias(total_png.b_01, bright_png.b_01)
                    faint_png.b_11 = get_faint_bias(total_png.b_11, bright_png.b_11)
                    setattr(faint, png_type, faint_png)
                return [bright, faint]
            else:
                pb_class = PBBias(self, survey_params, hmf, hod, m_c=survey_params.cut)
                pb_class.add_bias_attr(class_bias)  # adds b_1,b_2 and PNG biases
        return [class_bias]

    def update_survey(
        self, survey_params: SurveyParams.SurveyBase | list[SurveyParams.SurveyBase], verbose: bool = True
    ) -> ClassWAP:
        """
        Update survey bias functions and initialize multi-tracer configurations.

        Parameters
        ----------
        survey_params : SurveyBase | list[SurveyBase]
            Parameters for one or more surveys. Triggers multi-tracer mode if multiple unique surveys are provided.
        verbose : bool, default=True
            Print progress during bias computation.

        Returns
        -------
        ClassWAP
            Self instance with updated survey state.

        Notes
        -----
        Internal 'betas' are reset and re-computed lazily. For bispectrum multi-tracer (X, Y),
        the surveys are expanded to [X, Y, X].
        """
        self.multi_tracer = False  # is it multi-tracer
        self.survey = [None, None, None]  # allow for bispectrum

        # If survey_params is a list
        if not isinstance(survey_params, list):
            survey_params = [survey_params]

        # is multi-tracer if more then 1 unique survey_params instance (or BGS hod stuff)
        if len(set(survey_params)) > 1 or hasattr(survey_params[0], "split"):
            self.multi_tracer = True

        # set redshift range and fsky - initial vlaues - we update as we loop over tracers
        self.z_min, self.z_max = survey_params[0].z_range
        self.f_sky = survey_params[0].f_sky

        idx = 0
        for sp in survey_params:  # loop over tracers
            self.z_min = max([self.z_min, sp.z_range[0]])
            self.z_max = min([self.z_max, sp.z_range[1]])
            self.f_sky = min([self.f_sky, sp.f_sky])

            results = self._process_survey(sp, self.compute_bias, self.hmf, self.hod, verbose=verbose)
            for result in results:  # can be a list if a BGS split
                self.survey[idx] = result
                result.t = idx
                idx += 1

        # remove Nones - if not specified use first value
        for i, survey in enumerate(self.survey):
            if survey is None:
                self.survey[i] = self.survey[0]

        self.update_shared_survey()  # update z_range,f_sky,n_g etc
        self.survey_params = survey_params
        self.N_tracers = len(
            {item for item in survey_params if item is not None}
        )  # Number of unique tracers - so probably 2 if multi-tracer

        return self

    def update_shared_survey(self) -> ClassWAP:
        """
        Configure parameters shared across all tracers (z_range, f_sky, n_g).

        Returns
        -------
        ClassWAP
            Self instance with synchronized shared parameters.
        """
        if self.z_min >= self.z_max:
            raise ValueError("Incompatible survey redshifts.")
        self.z_survey = np.linspace(self.z_min, self.z_max, int(1e2))
        if self.multi_tracer:
            self.n_g = lambda xx: 0 * xx + 1e10  # for multi-tracer set shot noise to zero...
        else:
            self.n_g = self.survey[0].n_g

        # and also derivatives for radial evolution effects over survey range
        self.compute_derivs_cosmo()  # set up derivatives for cosmology dependent functions
        return self

    #######################################################################################################

    def Pk_phi(self, k: ArrayLike, k0: float = 0.05) -> np.ndarray:
        """
        Power spectrum of the Bardeen potential Phi in the matter-dominated era.

        Parameters
        ----------
        k : ArrayLike
            Wavemodes in units of [h/Mpc].
        k0 : float, default=0.05
            Pivot scale in units of [1/Mpc].

        Returns
        -------
        np.ndarray
            Bardeen potential power spectrum in units of [Mpc/h]^3.
        """
        k_pivot = k0 / self.h  # get pivot scale in [h/Mpc]
        resp = (9.0 / 25.0) * self.A_s * (k / k_pivot) ** (self.n_s - 1.0)

        resp *= 2 * np.pi**2.0 / k**3.0  # [Mpc/h]^3

        return resp

    def M(self, k: ArrayLike, z: ArrayLike) -> np.ndarray:
        """
        Compute transfer function from primordial scalar power spectrum to the late-time
        linear matter power spectrum.

        Parameters
        ----------
        k : ArrayLike
            Wavemodes in units of [h/Mpc].
        z : ArrayLike
            Redshift.

        Returns
        -------
        np.ndarray
            Scaling factor M(k, z).
        """
        return np.sqrt(self.D(z) ** 2 * self.Pk(k) / self.Pk_phi(k))

    #################################################################################

    def sigma_R_n(self, R: np.ndarray, n: int, K_MIN: float = 5e-5, steps: int = int(1e3)) -> np.ndarray:
        """
        Compute sigma^2 for a given radius and n, i.e. does integral over k.
        Works well and is vectorised - in agreement with other codes.
        Uses differential equation approach.
        """

        def deriv_sigma(x, y, R, n):  # adapted from Pylians
            kR = x * R
            W = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
            return [x ** (2 + n) * self.pk(x) * W**2]

        # this function computes sigma(R)
        def sigma_sq(R, n):
            k_limits = [K_MIN, self.K_MAX]
            yinit = [0.0]

            I = solve_ivp(deriv_sigma, k_limits, yinit, args=(R, n), method="RK45")

            return I.y[0][-1]  # get last value

        sigma_squared = np.zeros((len(R)))
        for i, _ in enumerate(R):
            sigma_squared[i] = sigma_sq(R[i], n) / (2.0 * np.pi**2)

        return sigma_squared

    ############################################################################################################

    def solve_second_order_KC(self) -> None:
        """
        Get second order growth factors - redshift dependent corrections to F2 and G2 kernels (very minimal)
        """
        dD_dz = self.D.derivative(nu=1)  # first derivative wrt to z

        def F_func(u, zz):  # so variables are F and H and D
            f, fd = u  # unpack u vector
            D_zz = self.D(zz)
            return [
                fd,
                (
                    -self.H_c(zz) * self.dH_c(zz) * (1 + zz) ** 2 * fd
                    + ((3 * (self.H_c(0)) ** 2 * self.Om_m(0) * (1 + zz)) / (2)) * (f + D_zz**2)
                )
                / (self.H_c(zz) ** 2 * (1 + zz) ** 2),
            ]

        odeint_zz = np.linspace(10, 0.05, int(1e5))  # so z=10 is peak matter domination...

        # set initial params for F
        F0 = [(3 / 7) * self.D(odeint_zz[0]) ** 2, (3 / 7) * 2 * self.D(odeint_zz[0]) * dD_dz(odeint_zz[0])]
        sol1 = odeint(F_func, F0, odeint_zz)
        K = sol1[:, 0] / self.D(odeint_zz) ** 2
        C = sol1[:, 1] / (2 * self.D(odeint_zz) * dD_dz(odeint_zz))
        self.K_intp = CubicSpline(odeint_zz[::-1], K[::-1])  # strictly increasing
        self.C_intp = CubicSpline(odeint_zz[::-1], C[::-1])

    def lnd_derivatives(self, functions_to_differentiate: list[Callable], ti: int = 0) -> list[CubicSpline]:
        """
        Calculates derivatives of a list of functions wrt log comoving dist numerically
        """
        tracer = self.survey[ti]

        # Store first derivatives in a list
        function_derivatives = []

        for func in functions_to_differentiate:
            # Calculate numerical derivatives of the function with respect to ln(d)
            derivative_func = CubicSpline(
                tracer.z_survey, np.gradient(func(tracer.z_survey), np.log(self.comoving_dist(tracer.z_survey)))
            )
            function_derivatives.append(derivative_func)

        return function_derivatives

    def get_PNG_bias(self, zz: ArrayLike, ti: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
        """Get b_01 and b_11 arrays depending on tracer redshift and shape"""
        tracer = self.survey[ti]

        if shape == "Loc":
            bE01 = tracer.loc.b_01(zz)
            bE11 = tracer.loc.b_11(zz)

        elif shape == "Eq":
            try:
                bE01 = tracer.eq.b_01(zz)
                bE11 = tracer.eq.b_11(zz)

            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")

        elif shape == "Orth":
            try:
                bE01 = tracer.orth.b_01(zz)
                bE11 = tracer.orth.b_11(zz)

            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")
        else:
            raise ValueError("Select PNG shape: Loc,Eq,Orth")

        return bE01, bE11

    def compute_derivs_cosmo(self) -> ClassWAP:
        """
        Compute derivatives wrt comoving distance of redshift dependent parameters for radial evolution terms
        Computes derivatives for cosmology dependent functions.
        """
        self.f_d, self.D_d = self.lnd_derivatives([self.f, self.D])
        self.f_dd, self.D_dd = self.lnd_derivatives([self.f_d, self.D_d])
        return self  # just for chaining

    def compute_derivs_survey(self, ti: int = 0) -> SetSurveyFunctions:
        """
        Compute derivatives wrt comoving distance of redshift dependent parameters for radial evolution terms
        Computes survey dependent derivatives for the given tracer (ti).
        """
        tracer = self.survey[ti]
        tracer.deriv = {}  # create dict
        # first order derivatives
        tracer.deriv["b1_d"], tracer.deriv["b2_d"], tracer.deriv["g2_d"] = self.lnd_derivatives(
            [tracer.b_1, tracer.b_2, tracer.g_2], ti=ti
        )
        # second order derivatives
        tracer.deriv["b1_dd"], tracer.deriv["b2_dd"], tracer.deriv["g2_dd"] = self.lnd_derivatives(
            [tracer.deriv["b1_d"], tracer.deriv["b2_d"], tracer.deriv["g2_d"]], ti=ti
        )

        return tracer
