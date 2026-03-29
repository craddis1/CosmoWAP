from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from cosmo_wap.HOD import HMF
from cosmo_wap.HOD.hods import YP, Smith_BGS
from cosmo_wap.lib.luminosity_funcs import BGSLuminosityFunction_HOD

if TYPE_CHECKING:
    from cosmo_wap.main import ClassWAP
    from cosmo_wap.survey_params import SetSurveyFunctions, SurveyParams


class PBBias:
    """Peak-background split bias computation.

    Computes higher order Eulerian and non-Gaussian biases for a given halo mass function (HMF) (Tinker 2010 (or ST))
    and halo occupation distribution (HOD), (Yankelevich and Porciani 2018).

    Parameters
    ----------
    cosmo_funcs : ClassWAP
        Cosmological functions (growth factor, power spectrum, etc.).
    survey_params : SurveyParams.SurveyBase
        Survey specification (redshift range, number density, linear bias).
    HMF : str
        Halo mass function model: ``"Tinker10"`` or ``"ST"`` (Sheth-Tormen).
    """

    def __init__(
        self,
        cosmo_funcs: ClassWAP,
        survey_params: SurveyParams.SurveyBase,
        hmf: str = "Tinker10",
        hod: str = "YP",
        m_c: float | None = None,
    ):
        self.cosmo_funcs = cosmo_funcs  # for later
        self.survey_params = survey_params
        # unpack from cosmo_funcs
        self.R = cosmo_funcs.R  # [Mpc/h]
        self.M = cosmo_funcs.M_halo  # [Msun/h] # array in R
        self.sig_R = cosmo_funcs.sig_R
        self.delta_c = cosmo_funcs.delta_c
        self.m_c = m_c  # only needed for Smith_BGS HOD, is ignored for YP

        # init hmf
        self.hmf = HMF(
            cosmo_funcs, hmf
        )  # initiate HMF class to get multiplicity function and lagrangian bias functions
        # and Eulerian bias conversions
        self.eulbias = self.EulBias(self)

        # initialise HOD
        if hod == "Smith_BGS":
            self.hod = Smith_BGS(cosmo_funcs)
        else:
            self.hod = YP(self, cosmo_funcs, survey_params)  # has two free parameters which we fit
            self.hod.fit_params()

        #################################################################################################
        # so save all required params to object
        self.z_samps = np.linspace(self.survey_params.z_range[0], self.survey_params.z_range[1], int(40))
        self.params = self.hod.get_hod_params(self.z_samps, self.m_c)

        self.n_g = self.get_number_density()
        self.b_1 = self.get_galaxy_bias(self.eulbias.b1)
        self.b_2 = self.get_galaxy_bias(self.eulbias.b2)
        self.g_2 = lambda xx: -(2 / 7) * (self.b_1(xx) - 1)  # tidal bias - e.g. baldauf

        # get PNG biases for each type
        self.loc = self.Loc(self)
        self.eq = self.Eq(self)
        self.orth = self.Orth(self)

        if hod == "Smith_BGS":
            # ok so we need an m_c dependent n_g!
            # compute Q and b_e from luminosity function using HOD-derived n_g
            def ng_tmp(m_c, zz):
                """n_g(m_c,zz) - compatible with lum_funcs"""
                return self.number_density(zz, m_c)

            lf = BGSLuminosityFunction_HOD(cosmo_funcs.cosmo, n_g=ng_tmp)
            zz = np.linspace(survey_params.z_range[0], survey_params.z_range[1], 40)
            self.Q = CubicSpline(zz, lf.get_Q(self.m_c, zz))
            self.be = CubicSpline(zz, lf.get_be(self.m_c, zz))

    #########################################################################################
    def number_density(self, zz: float | np.ndarray, *hod_params) -> float | np.ndarray:
        """
        Number density as function of redshift and HOD params
        Integrate HOD-weighted halo mass function to get galaxy number density at redshift zz.
        """
        if not np.isscalar(zz):
            return np.array(
                [
                    self.number_density(z, *[p[i] if not np.isscalar(p) else p for p in hod_params])
                    for i, z in enumerate(zz)
                ]
            )

        return simpson(self.hod.HOD(zz, *hod_params) * self.hmf.n_h(zz), self.M, axis=-1)

    # so implement
    def general_galaxy_bias(
        self, b_h: Callable, zz: float | np.ndarray, *hod_params, A: float = 1, alpha: int = 0
    ) -> float | np.ndarray:
        """
        Galaxy bias as function of redshift and HOD params
        Compute HOD-weighted average of halo bias b_h over the halo mass function at redshift zz.
        """
        if not np.isscalar(zz):
            return np.array(
                [
                    self.general_galaxy_bias(
                        b_h, z, *[p[i] if not np.isscalar(p) else p for p in hod_params], A=A, alpha=alpha
                    )
                    for i, z in enumerate(zz)
                ]
            )

        # Integrate over M for each value of z
        integral_values = simpson(b_h(zz, A, alpha) * self.hmf.n_h(zz) * self.hod.HOD(zz, *hod_params), self.M, axis=-1)

        return integral_values

    #########################  return cubic spline objects
    def get_number_density(self):
        """
        Fit cubic spline to the galaxy number density over the survey redshift range.
        """
        n_g = self.number_density(self.z_samps, *self.params)

        return CubicSpline(self.z_samps, n_g)

    def get_galaxy_bias(self, b_h, A=1, alpha=0):
        """
        set z_range and fit cubic spline for a given bias function - needs to be normalised by n_g
        """
        bias_arr = self.general_galaxy_bias(b_h, self.z_samps, *self.params, A=A, alpha=alpha) / self.n_g(self.z_samps)

        return CubicSpline(self.z_samps, bias_arr)

    #############################################################################################################

    class EulBias:
        """
        define Eulerian biases in terms of lagrangian biases
        """

        def __init__(self, parent_class):
            self.pc = parent_class

        def b1(self, zz, A=1, alpha=0):
            return 1 + self.pc.hmf.lagbias.b1(zz)

        def b2(self, zz, A=1, alpha=0):
            return self.pc.hmf.lagbias.b2(zz) + (8 / 21) * self.pc.hmf.lagbias.b1(zz)

        def b_01(self, zz, A=1, alpha=0):
            delta_c = self.pc.delta_c
            return (
                A
                * (
                    2 * delta_c * self.pc.hmf.lagbias.b1(zz)
                    + 4 * (self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)), np.log(self.pc.sig_R["0"](zz))) - 1)
                )
                * (self.pc.sig_R[str(alpha)](zz) / self.pc.sig_R["0"](zz))
            )

        def b_11(self, zz, A=1, alpha=0):
            delta_c = self.pc.delta_c
            return (
                A
                * (
                    delta_c * (self.b2(zz) + (13 / 21) * (self.b1(zz) - 1))
                    + self.b1(zz)
                    * (2 * self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)), np.log(self.pc.sig_R["0"](zz))) - 3)
                    + 1
                )
                * (self.pc.sig_R[str(alpha)](zz) / self.pc.sig_R["0"](zz))
            )

    # get halo biases these will be arrays with repsect to M - for integration
    def dy_ov_dx(self, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Compute dy/dx using finite differences along the R axis.
        """
        # Fit cubic splines and caclulate their derivatives for each R coord
        dum1 = np.gradient(dy, self.R, axis=-1)
        dum2 = np.gradient(dx, self.R, axis=-1)

        return dum1 / dum2  # Compute the ratio of the derivatives

    ###############################################################################################################

    # for the 3 different types of PNG
    class Loc:
        def __init__(self, parent):
            self.A = 1
            self.alpha = 0
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01, A=self.A, alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11, A=self.A, alpha=self.alpha)

    class Eq:
        def __init__(self, parent):
            self.A = 3
            self.alpha = 2
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01, A=self.A, alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11, A=self.A, alpha=self.alpha)

    class Orth:
        def __init__(self, parent):
            self.A = -3
            self.alpha = 1
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01, A=self.A, alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11, A=self.A, alpha=self.alpha)

    def add_bias_attr(self, other_class: SetSurveyFunctions) -> None:
        """
        Collect computed biases
        """
        other_class.n_g = self.n_g
        other_class.b_1 = self.b_1
        other_class.b_2 = self.b_2
        other_class.g_2 = self.g_2

        # Q and be if computed from HOD
        if hasattr(self, "Q"):
            other_class.Q = self.Q
            other_class.be = self.be

        # get PNG biases for each type
        other_class.loc = self.loc
        other_class.eq = self.eq
        other_class.orth = self.orth
