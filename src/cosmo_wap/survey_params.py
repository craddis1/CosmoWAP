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
    fraction_cut,
    lum_derivs,
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
            self.K = LF.K  # K-correction - with it b_e is derived from n_evol (see SetSurveyFunctions.be)
            if not need_hod:
                self.Q = CachedSpline(zz, LF.get_Q(cut, zz))
                self.be = CachedSpline(zz, LF.get_be(cut, zz))
                self.n_g = CachedSpline(zz, LF.number_density(cut, zz))
                dQ, db1, dlnLc = lum_derivs(LF, cut, zz)  # for the betas - see lib.betas
                self.dQ_dlnL = CachedSpline(zz, dQ)
                self.dlnLc_dz = CachedSpline(zz, dlnLc)
                # then also get linear bias from fits in Table. 1 1909.12069 - `is not None` as
                # get_b_1 lives on the H-alpha base class, and the WISE LF opts out of it
                if getattr(LF, "get_b_1", None) is not None:
                    self.b_1 = CachedSpline(zz, LF.get_b_1(cut, zz))
                    self.db1_dlnL = CachedSpline(zz, db1)
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

            # luminosity derivatives - dlnLc_dz does not depend on the cut, so both keep the total's
            dQ_B, db1_B, _ = lum_derivs(self.LF, split, zz)
            self.bright.dQ_dlnL = CachedSpline(zz, dQ_B)
            self.faint.dQ_dlnL = utils.get_faint_lum_deriv(zz, n_T, n_B, Q_T, Q_B, Q_T, Q_B, self.dQ_dlnL(zz), dQ_B)

            # then for linear bias if we can use semi-analytical fit from 1909.12069
            if getattr(self.LF, "get_b_1", None) is not None:
                b_T = self.b_1(zz)
                b_B = self.LF.get_b_1(split, zz)
                self.bright.b_1 = CachedSpline(zz, b_B)
                self.faint.b_1 = utils.get_faint_bias(zz, n_T, n_B, b_T, b_B)
                self.bright.db1_dlnL = CachedSpline(zz, db1_B)
                self.faint.db1_dlnL = utils.get_faint_lum_deriv(
                    zz, n_T, n_B, Q_T, Q_B, b_T, b_B, self.db1_dlnL(zz), db1_B
                )

            return [self.bright, self.faint]  # two tracers defined

        def _get_fraction(self, fraction):
            """
            Bright = the brightest `fraction` of the sample at each redshift, faint = the rest, so
            n_B = fraction n_T and n_evol is the same for total, bright and faint (2406.19908 Sec. 3.2).
            The bright cut then moves with z: ln(L_B/L_T) from fraction_cut, drifting against a fixed
            flux by D = d ln(L_B/L_T)/d ln(1+z), which goes into both b_e (see SetSurveyFunctions.be).
            """
            zz, LF = self.zz, self.LF
            n_T, Q_T = self.n_g(zz), self.Q(zz)

            def at_cut(func, dlnL):
                """func(cut, z) one redshift at a time - the magnitude-limited LFs broadcast an array cut against zz"""
                return np.array([func(LF.shift_cut(self.cut, d), np.array([z]))[0] for d, z in zip(dlnL, zz)])

            def d_dlnL(func, h=0.01):
                """d func/d ln L at the cut - the central difference of lum_derivs, which also wants a z grid"""
                return lambda c, z: (func(LF.shift_cut(c, h), z) - func(LF.shift_cut(c, -h), z)) / (2 * h)

            dlnL = fraction_cut(lambda d: at_cut(LF.number_density, d), fraction * n_T)
            D = np.gradient(dlnL, np.log(1 + zz))
            n_B, n_F = fraction * n_T, (1 - fraction) * n_T
            Q_B = at_cut(LF.get_Q, dlnL)
            Q_F = utils.get_faint_bias(zz, n_T, n_B, Q_T, Q_B)(zz)
            dQ_B = at_cut(d_dlnL(LF.get_Q), dlnL)

            # b_e at a fixed flux (get_be) and the cut drift on top: n_F b_eF = n_T b_eT - n_B b_eB
            be_B = LF.get_be(None, zz, n_g=n_B, Q=Q_B) - D * Q_B
            be_F = LF.get_be(None, zz, n_g=n_F, Q=Q_F) + fraction / (1 - fraction) * D * Q_B

            self.bright.n_g = CachedSpline(zz, n_B)
            self.bright.Q = CachedSpline(zz, Q_B)
            self.bright.be = CachedSpline(zz, be_B)
            self.bright.dQ_dlnL = CachedSpline(zz, dQ_B)
            self.bright.dlnF_cut = CachedSpline(zz, D)

            self.faint.n_g = CachedSpline(zz, n_F)
            self.faint.Q = CachedSpline(zz, Q_F)
            self.faint.be = CachedSpline(zz, be_F)
            self.faint.dQ_dlnL = utils.get_faint_lum_deriv(zz, n_T, n_B, Q_T, Q_B, Q_T, Q_B, self.dQ_dlnL(zz), dQ_B)

            b_1 = None
            if getattr(LF, "get_b_1", None) is not None:  # as _get_faint
                b_T = self.b_1(zz)
                b_B = at_cut(LF.get_b_1, dlnL)
                db1_B = at_cut(d_dlnL(LF.get_b_1), dlnL)
                self.bright.b_1 = CachedSpline(zz, b_B)
                self.faint.b_1 = utils.get_faint_bias(zz, n_T, n_B, b_T, b_B)
                self.bright.db1_dlnL = CachedSpline(zz, db1_B)
                self.faint.db1_dlnL = utils.get_faint_lum_deriv(
                    zz, n_T, n_B, Q_T, Q_B, b_T, b_B, self.db1_dlnL(zz), db1_B
                )
                b_1 = (b_B, self.faint.b_1(zz), db1_B)

            extras = utils.fraction_split_faint(zz, fraction, D, Q_B, Q_F, dQ_B, self.dlnLc_dz(zz), be_B, be_F, b_1)
            for name, value in extras.items():
                setattr(self.faint, name, value)

            return [self.bright, self.faint]

        def BF_split(self, split=None, fraction=None):
            """
            Split is magnitude or flux cut at which we seperate our samples
            Call from survey class with a luminosity function.

            Or fraction: the bright fraction n_B/n_T, kept the same at every redshift by a cut that moves with z -
            see _get_fraction.
            """
            if not hasattr(self, "LF"):
                raise ValueError("No luminosity function - use survey with defined luminosity function!")
            if (split is None) == (fraction is None):
                raise ValueError("Give one of split (a fixed cut) or fraction (a fixed bright fraction).")
            if fraction is not None and not 0 < fraction < 1:
                raise ValueError("fraction is the bright fraction n_B/n_T - it must lie in (0, 1).")

            self.split = split  # store for later
            self.fraction = fraction
            self.bright = utils.copy(self)  # bright
            self.faint = utils.copy(self)  # faint
            if not self.need_hod:  # get faint from bright and total sample.
                return self._get_faint(split) if fraction is None else self._get_fraction(fraction)
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
                self.K = lambda xx: 0 * xx  # H-alpha line flux - no K-correction
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
            self.z_range = [0.1, 0.5]
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
            self.K = lambda xx: 0 * xx  # HI line flux - no K-correction
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
            self.K = lambda xx: 0 * xx  # HI line flux - no K-correction
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
        self.be = getattr(survey_params, "be", lambda xx: 0 * xx)  # replaced by n_evol in set_selection if K is known
        self.Q = getattr(survey_params, "Q", lambda xx: 0 * xx)
        # K-correction [mag] - None for an unselected tracer, which keeps b_e
        self.K = getattr(survey_params, "K", None)
        self.p = getattr(survey_params, "p", 1.0) if p is None else p
        # luminosity derivatives at the cut (luminosity_funcs.lum_derivs) - zero without a luminosity function
        self.dQ_dlnL = getattr(survey_params, "dQ_dlnL", lambda xx: 0 * xx)
        self.dlnLc_dz = getattr(survey_params, "dlnLc_dz", lambda xx: 0 * xx)
        self.db1_dlnL = lambda xx: 0 * xx  # belongs to the luminosity function's b_1, which compute_bias replaces
        # fixed-fraction split (SurveyBase.BF_split): the bright cut's drift against a fixed flux; on the faint, the
        # bright Q its b_e carries (see be) and the fixed-L corrections from the bright cut (utils.fraction_split_faint)
        self.dlnF_cut = getattr(survey_params, "dlnF_cut", None)
        self.w_split = getattr(survey_params, "w_split", None)
        self.Q_split = getattr(survey_params, "Q_split", None)
        self.split_corr = getattr(survey_params, "split_corr", {})
        self.split_be_lin = getattr(survey_params, "split_be_lin", None)  # the b_e term as A + B c_T
        if compute_bias:  # the HMF/HOD b_1 replaces the luminosity function's, with no luminosity derivatives (above)
            self.split_corr = {k: v for k, v in self.split_corr.items() if k != "b_1"}

        if not compute_bias:  # then just use default or assigned
            self.n_g = getattr(survey_params, "n_g", lambda xx: 1e5 + 0 * xx)  # no shot noise
            self.b_1 = getattr(survey_params, "b_1", lambda xx: np.sqrt(1 + xx))  # stupid linear bias model
            self.db1_dlnL = getattr(survey_params, "db1_dlnL", self.db1_dlnL)
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
                    self.db_01_dlnL = lambda xx: 2 * delta_c * parent.db1_dlnL(xx)  # for the full_betas Upsilons
                    if "b_1" in parent.split_corr:  # fixed-fraction faint: b_01's second-cut term, p cancels in it
                        corr_b1 = parent.split_corr["b_1"]
                        self.split_corr_b01 = lambda xx: 2 * delta_c * corr_b1(xx)

            self.loc = Loc(self)

        self.f_sky = getattr(survey_params, "f_sky", 1)
        self.z_range = getattr(survey_params, "z_range", [0, 5])  # read as [z_min, z_max]

        self.betas = None
        self.betas_HO = None  # beta1-beta5, only built with full_betas
        self.deriv = {}

    def __setattr__(self, name, value):
        # with n_evol stored b_e is derived - an assigned b_e would shadow be() and silently go stale
        if name == "be" and "n_evol" in self.__dict__:
            raise AttributeError("b_e is derived from n_evol here - shift n_evol instead (see utils.selection_key)")
        super().__setattr__(name, value)

    def be(self, zz):
        """Evolution bias from the observed count slope n_evol = d ln N/d ln(1+z), N = r^2 n_g/H the counts per z
        per solid angle: b_e = -n_evol - d ln H/d ln(1+z) + 2/(chi H) - c_Q Q, with c_Q = d ln L_c/d ln(1+z).

        A method so it reads this tracer's n_evol and Q when called - a copy shifted by the forecast moves b_e with it.
        Only used once set_selection has stored n_evol; until then (and always without a K-correction) an instance
        attribute `be` shadows it."""
        be = -self.n_evol(zz) + self.be_vol(zz) - self.c_Q(zz) * self.Q(zz)
        if self.w_split is not None:  # faint of a fixed-fraction split - see utils.fraction_split_faint
            be = be + self.w_split(zz) * self.Q_split(zz)
        return be

    def set_selection(self, cf):
        """The cosmology half of b_e (be) for the cosmology of cf - so a cosmology derivative holds n_evol and Q fixed.

        The first call, on the fiducial ClassWAP, inverts the b_e the survey came with into n_evol. Later calls (the
        copies adopt_survey makes) only rebuild be_vol and c_Q, and the dlnLc_dz the betas take from c_Q where the
        tracer has luminosity derivatives. Does nothing without a K-correction: b_e then stays the survey's own."""
        if self.K is None:
            return
        zz = self.z_survey
        H_c = cf.H_c(zz)
        inv_xH = 1 / (cf.comoving_dist(zz) * H_c)
        dlnH = 1 + (1 + zz) * cf.dH_c(zz) / H_c  # d ln H/d ln(1+z) for physical H = (1+z) H_c
        c_Q = 2 * (1 + inv_xH) + 0.4 * np.log(10) * np.gradient(self.K(zz), np.log(1 + zz))
        if self.dlnF_cut is not None:  # a cut moving against a fixed flux - the bright of a fixed-fraction split
            c_Q = c_Q + self.dlnF_cut(zz)
        # both smooth, so splined on zz - and on the fiducial they cancel exactly against the same splines in n_evol
        be_vol = CachedSpline(zz, -dlnH + 2 * inv_xH)  # from the volume factor r^2/H
        c_Q = CachedSpline(zz, c_Q)

        if "n_evol" not in self.__dict__:
            # a closure over the survey's own functions, not a spline on zz - n_evol is a small difference of large
            # terms at low z (c_Q Q and 2/(chi H) ~ 40 for BGS), so re-splining it moves b_e between the nodes.
            # A pure function of z, so scalar redshifts are remembered as CachedSpline does
            be_fid, Q_fid, w, Q_split = self.__dict__.pop("be"), self.Q, self.w_split, self.Q_split
            memo = {}

            def n_evol(z):
                scalar = isinstance(z, CachedSpline._SCALARS)
                if scalar and (hit := memo.get(float(z))) is not None:
                    return hit
                n = be_vol(z) - c_Q(z) * Q_fid(z) - be_fid(z)
                if w is not None:
                    n = n + w(z) * Q_split(z)
                if scalar:
                    if len(memo) >= CachedSpline._MAX:
                        memo.clear()
                    memo[float(z)] = n
                return n

            self.n_evol = n_evol
        self.be_vol, self.c_Q = be_vol, c_Q
        if self.split_be_lin is not None:  # the faint's b_e second-cut term at this c_Q (= c_T) - rebound, not edited
            A, B = self.split_be_lin
            self.split_corr = {**self.split_corr, "be": CachedSpline(zz, A(zz) + B(zz) * c_Q(zz))}
        if np.any(self.dlnLc_dz(zz)):  # only alongside the luminosity derivatives it multiplies - see lum_derivs
            self.dlnLc_dz = CachedSpline(zz, c_Q(zz) / (1 + zz))
        self.reset_cache()

    def reset_cache(self):
        """Clear cached betas and derivatives — call when bias functions change."""
        self.betas = None
        self.betas_HO = None
        self.deriv = {}
