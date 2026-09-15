# Define the path to the data file relative to the current script location
import os

import numpy as np

from cosmo_wap.lib import utils
from cosmo_wap.lib.luminosity_funcs import (
    BGSLuminosityFunction,
    LBGLuminosityFunction,
    Model1LuminosityFunction,
    Model3LuminosityFunction,
    WISELuminosityFunction,
)
from cosmo_wap.lib.utils import CachedSpline


class SurveyParams:
    def get(self, cosmo, survey):
        """
        Initialize and get survey parameters for some set surveys
        """
        return getattr(self, survey)(cosmo)

    # ok want to inherit this function to update variables - could use dataclasses
    class SurveyBase:
        p = 1.0  # UMF: b_phi = 2 delta_c (b_1 - p)
        need_hod = False  # only compute_luminosity sets it; surveys that skip it never do

        def update(self, **kwargs):
            """update survey class parameters-
            Input dictionary with parameters you want to change."""
            new_self = utils.copy(self)

            for key, value in kwargs.items():
                if not hasattr(new_self, key):  # else a typo'd name would silently do nothing
                    raise AttributeError(f"{type(self).__name__} has no parameter '{key}'")
                setattr(new_self, key, value)
            return new_self

        def compute_luminosity(self, LF, cut, zz, need_hod=False):
            """Get biases from given luminosity function and magnitude/luminosity cut

            LF: Luminosity function class
            cut: Magnitude/flux cut
            zz: redshift
            need_hod: if True, defer bias computation to HOD later
            Returns:
            Object with Q,be and n_g defined"""
            self.cut = cut
            self.need_hod = need_hod
            if not need_hod:
                self.Q = CachedSpline(zz, LF.get_Q(cut, zz))
                self.be = CachedSpline(zz, LF.get_be(cut, zz))
                self.n_g = CachedSpline(zz, LF.number_density(cut, zz))
                # then also get linear bias from fits in Table. 2 1909.12069 - `is not None` as
                # get_b_1 lives on the H-alpha base class, and the WISE LF opts out of it
                if getattr(LF, "get_b_1", None) is not None:
                    self.b_1 = CachedSpline(zz, LF.get_b_1(cut, zz))
            return self

        def _get_faint(self, split):
            """
            We already have total sample biases.
            n_F = n_T - n_B
            bF = n_T b_T − n_B b_B /nF
            Q_F = n_T/(n_T - n_B) Q_T - n_B/(n_T - n_B)*Q_B
            be_F = d ln(n_T - n_B)/ d ln (1 + z)
            """
            zz = self.zz
            # define total survey biases
            Q_T = self.Q(zz)
            n_T = self.n_g(zz)

            Q_B = self.LF.get_Q(split, zz)
            be_B = self.LF.get_be(split, zz)
            n_B = self.LF.number_density(split, zz)
            n_F = n_T - n_B

            self.bright.Q = CachedSpline(zz, Q_B)
            self.bright.be = CachedSpline(zz, be_B)
            self.bright.n_g = CachedSpline(zz, n_B)

            # so for faint
            self.faint.n_g = CachedSpline(zz, n_F)
            self.faint.Q = utils.get_faint_bias(zz, n_T, n_B, Q_T, Q_B)
            self.faint.be = CachedSpline(
                zz, self.LF.get_be(None, zz, n_g=n_F, Q=self.faint.Q(zz))
            )  # get be for faint from luminosity function using faint n_g and Q

            # then for linear bias if we can use semi-analytical fit from 1909.12069
            if getattr(self.LF, "get_b_1", None) is not None:
                b_T = self.b_1(zz)
                b_B = self.LF.get_b_1(split, zz)
                self.bright.b_1 = CachedSpline(zz, b_B)
                self.faint.b_1 = utils.get_faint_bias(zz, n_T, n_B, b_T, b_B)

            return [self.bright, self.faint]  # two tracers defined

        def BF_split(self, split):
            """
            Split is magnitude or flux cut at which we seperate our samples
            Call from survey class with a luminosity function.
            """
            if not hasattr(self, "LF"):
                raise ValueError("No luminosity function - use survey with defined luminosity function!")

            self.split = split  # store for later
            self.bright = utils.copy(self)  # bright
            self.faint = utils.copy(self)  # faint
            if not self.need_hod:  # get faint from bright and total sample.
                return self._get_faint(split)
            else:
                return self

        def load_SKAO_data(self):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.SKAO1Data = np.loadtxt(os.path.join(module_dir, "data_library/SKAO1Data.txt"))
            self.SKAO2Data = np.loadtxt(os.path.join(module_dir, "data_library/SKAO2Data.txt"))

        def load_SPHEREx_data(self):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.SPHERExData = np.loadtxt(os.path.join(module_dir, "data_library/SPHERExData.txt"))

    class Euclid(SurveyBase):
        def __init__(self, cosmo, fitting=False, model3=True, cut=None):
            self.cosmo = cosmo
            self.b_1 = lambda xx: 0.9 + 0.4 * xx
            self.f_sky = 15000 / 41253
            self.z_range = [0.9, 1.8]  # get zmin and zmax

            if fitting:
                self.be = lambda xx: -7.29 + 0.470 * xx + 1.17 * xx**2 - 0.290 * xx**3  # euclid_data[:,2]
                self.Q = lambda xx: 0.583 + 2.02 * xx - 0.568 * xx**2 + 0.0411 * xx**3
                self.n_g = lambda zz: 0.0193 * zz ** (-0.0282) * np.exp(-2.81 * zz)
            else:
                self.zz = np.linspace(self.z_range[0], self.z_range[1], 100)
                # from lumnosity function
                if model3:
                    if cut is None:  # set defualt values - this one agrees with fitting functions above
                        cut = 2e-16
                    self.LF = Model3LuminosityFunction(cosmo)
                else:
                    if cut is None:
                        cut = 3e-16
                    self.LF = Model1LuminosityFunction(cosmo)

                self.compute_luminosity(self.LF, cut, self.zz)

    class Roman(SurveyBase):
        def __init__(self, cosmo, model3=False, cut=None):
            self.cosmo = cosmo
            self.b_1 = lambda xx: 0.9 + 0.4 * xx
            self.f_sky = 2000 / 41253
            self.z_range = [0.5, 2.0]  # get zmin and zmax
            self.zz = np.linspace(self.z_range[0], self.z_range[1], 100)

            # from lumnosity function
            if model3:
                if cut is None:  # set defualt values
                    cut = 1e-16
                self.LF = Model3LuminosityFunction(cosmo)
            else:
                if cut is None:
                    cut = 1e-16
                self.LF = Model1LuminosityFunction(cosmo)

            self.compute_luminosity(self.LF, cut, self.zz)

    class BGS(SurveyBase):
        def __init__(self, cosmo, cut=20.175, flag="HOD"):
            self.cosmo = cosmo
            self.b_1 = lambda xx: 1.34 / cosmo.scale_independent_growth_factor(xx)
            self.z_range = [0.05, 0.5]
            self.f_sky = 15000 / 41253

            # default
            self.be = lambda xx: -2.25 - 4.02 * xx + 0.318 * xx**2 - 14.6 * xx**3
            self.Q = lambda xx: 0.282 + 2.36 * xx + 2.27 * xx**2 + 11.1 * xx**3
            self.n_g = lambda zz: 0.023 * zz ** (-0.471) * np.exp(-5.17 * zz) - 0.002  # fitting from Maartens
            if flag == "LF":
                # from lumnosity function
                self.zz = np.linspace(self.z_range[0], self.z_range[1], 100)
                self.LF = BGSLuminosityFunction(cosmo)
                self.compute_luminosity(self.LF, cut, self.zz)
            elif flag == "HOD":
                self.zz = np.linspace(self.z_range[0], self.z_range[1], 100)
                self.LF = BGSLuminosityFunction(cosmo)
                self.compute_luminosity(self.LF, cut, self.zz, need_hod=True)  # compute biases with HOD later

    class MegaMapper(SurveyBase):
        def __init__(self, cosmo, cut=24.5):
            self.cosmo = cosmo
            self.A = -0.98 * (cut - 25) + 0.11  # from Eq.(2.7) 1904.13378v2
            self.B = 0.12 * (cut - 25) + 0.17
            self.b_1 = lambda xx: (
                self.A * (1 + xx) + self.B * (1 + xx) ** 2
            )  # so linear bias for given apparent magnitude limit
            self.z_range = [2.1, 5]  # get zmin and zmax
            self.f_sky = 20000 / 41253

            # from lumnosity function
            self.LF = LBGLuminosityFunction(cosmo)
            self.zz = self.LF.z_values
            self.compute_luminosity(self.LF, cut, self.zz)

    class SKAO1(SurveyBase):
        def __init__(self, cosmo):
            self.cosmo = cosmo
            self.load_SKAO_data()
            self.b_1 = lambda xx: 0.616 * np.exp(1.017 * xx)
            self.z_range = [self.SKAO1Data[:, 0][0], self.SKAO1Data[:, 0][-1]]
            self.be = CachedSpline(self.SKAO1Data[:, 0], self.SKAO1Data[:, 4])
            self.Q = CachedSpline(self.SKAO1Data[:, 0], self.SKAO1Data[:, 3])
            self.n_g = CachedSpline(self.SKAO1Data[:, 0], self.SKAO1Data[:, 2])  # fitting from Maartens
            self.f_sky = 5000 / 41253

    class SKAO2(SurveyBase):
        def __init__(self, cosmo):
            self.cosmo = cosmo
            self.load_SKAO_data()
            self.b_1 = lambda xx: 0.554 * np.exp(0.783 * xx)
            self.z_range = [self.SKAO2Data[:, 0][0], self.SKAO2Data[:, 0][-1]]
            self.be = CachedSpline(self.SKAO2Data[:, 0], self.SKAO2Data[:, 4])
            self.Q = CachedSpline(self.SKAO2Data[:, 0], self.SKAO2Data[:, 3])
            self.n_g = CachedSpline(self.SKAO2Data[:, 0], self.SKAO2Data[:, 2])  # fitting from Maartens
            self.f_sky = 30000 / 41253

    class SPHEREx(SurveyBase):
        # b_1 fit (A, beta, gamma) per subsample, from least squares of Eq. (5.2) of
        # arXiv:2608.18334 to the tabulated biases.
        b_1_fits = [
            (0.122, 7.93e4, 0.262),
            (0.689, 14.2, 0.461),
            (0.925, 1.96, 0.723),
            (0.717, 5.58, 0.547),
            (0.522, 5.62, 0.710),
        ]

        def __init__(self, cosmo, sample=0, cut=2e-16):
            """
            SPHEREx all-sky spectral survey - Dore et al. (2014) [arXiv:1412.4872].

            Number density and linear bias come from the SPHEREx public products (see
            data_library/SPHERExData.txt); Q and b_e from the WISE 2.4 micron luminosity
            function, following arXiv:2608.18334.

            sample: 0-4, selects the redshift-accuracy subsample sigma_z/(1+z) < 0.003,
                    0.01, 0.03, 0.1 or 0.2. Sample 0 here is default for 3D P(k)
            cut: flux cut [erg/cm^2/s at 2.4 micron] used for Q and b_e.

            Note on the flux cut: SPHEREx selects on template-fitted photometric redshifts
            across many bands, not on 2.4 micron flux, so it has no true flux limit and
            Dore et al. quote no magnification bias. The default 2e-16 is the effective
            value used in arXiv:2608.18334, roughly 320x deeper than SPHEREx's actual 5
            sigma point-source depth at 2.4 micron (19.63 AB, i.e. 51 uJy). At the real
            depth the WISE luminosity function - calibrated at z <~ 1 - puts essentially no
            galaxies above z ~ 1, so the cut is best read as the knob that places Q in a
            plausible range, and is exposed here for that reason.
            """
            self.cosmo = cosmo
            self.load_SPHEREx_data()
            zz_data = np.mean(self.SPHERExData[:, :2], axis=1)  # bin centres
            n_g_data = self.SPHERExData[:, 2 + sample]
            self.sample = sample

            A, beta, gamma = self.b_1_fits[sample]
            self.b_1 = lambda xx: A * (1 + beta * xx) ** gamma

            self.z_range = [zz_data[0], zz_data[-1]]  # bin centres, so n_g is never extrapolated
            self.zz = np.linspace(self.z_range[0], self.z_range[1], 100)
            self.f_sky = 0.75  # 75% of sky after galactic masking - Dore et al. Sec. VI H

            # Q and b_e from the luminosity function... - BF split does not make sense here
            self.LF = WISELuminosityFunction(cosmo)
            self.compute_luminosity(self.LF, cut, self.zz)

            # ...but n_g from the survey itself. Splined in log as it spans four decades -
            # note it is too coarse (11 bins) and too noisy to differentiate, which is why
            # b_e above is left to the luminosity function.
            log_n_g = CachedSpline(zz_data, np.log(n_g_data))
            self.n_g = lambda xx: np.exp(log_n_g(xx))

        def BF_split(self, split):
            raise ValueError(
                "SPHEREx takes n_g from the survey, not from its luminosity function, so a "
                "bright/faint split would be inconsistent - use two of the five subsamples "
                "as tracers instead, e.g. [SPHEREx(cosmo, sample=0), SPHEREx(cosmo, sample=1)]."
            )

    class DM_part(SurveyBase):
        def __init__(self, cosmo):
            self.cosmo = cosmo
            self.b_1 = lambda xx: 1 + 0 * xx  # for dark matter particles
            self.b_2 = lambda xx: 0 * xx
            self.g_2 = lambda xx: 0 * xx
            self.z_range = [0.01, 5]
            self.be = lambda xx: 0 * xx
            self.Q = lambda xx: 0 * xx
            self.n_g = lambda xx: 1e5 + 0 * xx
            self.f_sky = 1

    class InitNew(SurveyBase):  # for adding new surveys... # so could add some sort of dict unpacking method?
        def __init__(self, cosmo):
            self.cosmo = cosmo
            self.b_1 = lambda xx: 1 + 0 * xx  # for dark matter particles
            self.z_range = [0.01, 5]
            self.be = lambda xx: 0 * xx
            self.Q = lambda xx: 0 * xx
            self.n_g = lambda xx: 1e5 + 0 * xx
            self.f_sky = 1


################################################################################


class SetSurveyFunctions:
    """Reads in survey specific params and calculates higher order ones if unprovided throguh set relations - if empty use default values.
    This is called once for each tracer

    p: overrides the survey's UMF merger exponent in b_phi = 2 delta_c (b_1 - p) - if None use survey_params.p
    """

    def __init__(self, survey_params, compute_bias=False, p=None):
        self.be = getattr(survey_params, "be", lambda xx: 0 * xx)
        self.Q = getattr(survey_params, "Q", lambda xx: 0 * xx)
        self.p = getattr(survey_params, "p", 1.0) if p is None else p

        if not compute_bias:  # then just use default or assigned
            self.n_g = getattr(survey_params, "n_g", lambda xx: 1e5 + 0 * xx)  # no shot noise
            self.b_1 = getattr(survey_params, "b_1", lambda xx: np.sqrt(1 + xx))  # stupid linear bias model
            self.g_2 = getattr(
                survey_params, "g_2", lambda xx: -(2 / 7) * (self.b_1(xx) - 1)
            )  # from local langragian expression
            self.b_2 = getattr(
                survey_params,
                "b_2",
                lambda xx: (
                    0.412
                    - 2.143 * self.b_1(xx)
                    + 0.929 * self.b_1(xx) ** 2
                    + 0.008 * self.b_1(xx) ** 3
                    + 4 / 3 * self.g_2(xx)
                ),
            )  # quadratic fit: 1511.01096

            # Local PNG from universality
            class Loc:
                def __init__(self, parent):
                    delta_c = 1.686
                    bL10 = lambda xx: parent.b_1(xx) - 1
                    bL20 = lambda xx: parent.b_2(xx) - (8 / 21) * bL10(xx)

                    bL01 = lambda xx: 2 * delta_c * (parent.b_1(xx) - parent.p)
                    bL11 = lambda xx: 2 * (delta_c * bL20(xx) - bL10(xx))
                    bL02 = lambda xx: 4 * delta_c * (delta_c * bL20(xx) - 2 * bL10(xx))

                    self.b_01 = bL01
                    self.b_11 = lambda xx: bL01(xx) + bL11(xx)
                    self.b_02 = bL02

            self.loc = Loc(self)

        self.f_sky = getattr(survey_params, "f_sky", 1)
        self.z_range = getattr(survey_params, "z_range", [0, 5])  # read as [z_min, z_max]

        self.betas = None
        self.deriv = {}

    def reset_cache(self):
        """Clear cached betas and derivatives — call when bias functions change."""
        self.betas = None
        self.deriv = {}
