from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from cosmo_wap.lib import betas, utils

if TYPE_CHECKING:
    from cosmo_wap.main import ClassWAP


class UnpackClassWAP:
    """Mixin providing parameter unpacking methods for power spectrum and bispectrum functions from ClassWAP."""

    self: ClassWAP

    ###################### helper function to unpack PNG
    def get_PNGparams(
        self, zz: ArrayLike, k1: ArrayLike, k2: ArrayLike, k3: ArrayLike, ti: int = 0, shape: str = "Loc"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        returns terms needed to compute PNG contribution including scale-dependent bias for bispectrum
        """

        bE01, bE11 = self.get_PNG_bias(zz, ti, shape)

        Mk1 = self.M(k1, zz)
        Mk2 = self.M(k2, zz)
        Mk3 = self.M(k3, zz)

        return bE01, bE11, Mk1, Mk2, Mk3

    def get_PNGparams_pk(
        self, zz: ArrayLike, k1: ArrayLike, ti: int = 0, shape: str = "Loc"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        returns terms needed to compute PNG contribution including scale-dependent bias for power spectra
        """

        bE01, _ = self.get_PNG_bias(zz, ti, shape)  # only need b_phi

        Mk1 = self.M(k1, zz)

        return bE01, Mk1

    ################################# helper function to unpack beta functions for GR terms

    def get_beta_funcs(self, zz: ArrayLike, ti: int = 0) -> list[np.ndarray]:
        """Get betas for given redshifts for given tracer if they are not already computed.
        If not computed then compute them using betas.interpolate_beta_funcs"""

        tracer = self.survey[ti]

        if tracer.betas is not None:
            return [tracer.betas[i](zz) for i in range(len(tracer.betas))]
        else:
            tracer.betas = betas.interpolate_beta_funcs(self, ti=ti)
            if not self.multi_tracer:  # no need to recompute for second survey
                self.survey[1].betas = tracer.betas

            return [tracer.betas[i](zz) for i in range(len(tracer.betas))]

    def get_beta_derivs(self, zz: ArrayLike, ti: int = 0) -> list[np.ndarray]:
        """Get betas derivatives wrt comoving distance for given redshifts for given tracer if they are not already computed.
        If not computed then compute"""

        tracer = self.survey[ti]

        if "beta" in tracer.deriv:
            return [tracer.deriv["beta"][i](zz) for i in range(len(tracer.deriv["beta"]))]
        else:
            # get betad - derivatives wrt to ln(d)  - for radial evolution terms
            betad = np.array(self.lnd_derivatives(tracer.betas[-6:]), dtype=object)  # beta14-19
            grd1 = self.lnd_derivatives([tracer.betas[0]])
            tracer.deriv["beta"] = np.concatenate((grd1, betad))

            if not self.multi_tracer:  # no need to recompute for second survey
                self.survey[1].deriv = tracer.deriv

            return [tracer.deriv["beta"][i](zz) for i in range(len(tracer.deriv["beta"]))]

    ############################################## unpacking function for power spectrum
    def unpack_pk(
        self,
        k1: ArrayLike,
        zz: ArrayLike,
        GR: bool = False,
        fNL_type: str | None = None,
        WS: bool = False,
        RR: bool = False,
    ) -> list:
        """Helper function to unpack all necessary terms with flag for each different type of term
        Should reduce the number of duplicated lines and make maintanence easier

        Is multi-tracer compliant

        Returns: list of parameters in order:
        - Base: [Pk, f, D1, b1, xb1]
        - +GR: [gr1, gr2, xgr1, xgr2]
        - +fNL: [bE01, Mk1, xbE01]
        - +WS/RR: [Pkd1, Pkdd1, d]
        - +RR: [fd, Dd, bd1, xbd1, fdd, Ddd, bdd1, xbdd1]
        - +RR+GR: [grd1, xgrd1]
        Total: [Pk, f, D1, b1, xb1, gr1, gr2, xgr1, xgr2, bE01, Mk1, xbE01, Pkd1, Pkdd1, d, fd, Dd, bd1, xbd1, fdd, Ddd, bdd1, xbdd1, grd1, xgrd1]"""

        # basic params
        if self.nonlin:
            Pk1 = self.Pk_NL(k1, zz)
        else:
            Pk1 = self.Pk(k1)

        f = self.f(zz)
        D1 = self.D(zz)

        b1 = self.survey[0].b_1(zz)
        xb1 = self.survey[1].b_1(zz)

        params = [Pk1, f, D1, b1, xb1]

        if GR:
            gr1, gr2 = self.get_beta_funcs(zz, ti=0)[:2]
            xgr1, xgr2 = self.get_beta_funcs(zz, ti=1)[:2]
            params.extend([gr1, gr2, xgr1, xgr2])

        if fNL_type is not None:
            bE01, Mk1 = self.get_PNGparams_pk(zz, k1, ti=0, shape=fNL_type)
            xbE01, _ = self.get_PNGparams_pk(zz, k1, ti=1, shape=fNL_type)
            params.extend([bE01, Mk1, xbE01])

        if WS or RR:
            Pkd1 = self.Pk_d(k1)
            Pkdd1 = self.Pk_dd(k1)
            d = self.comoving_dist(zz)
            params.extend([Pkd1, Pkdd1, d])

            if RR:
                if not self.survey[0].deriv or not self.survey[1].deriv:
                    self.survey[0] = self.compute_derivs_survey(ti=0)
                    if self.multi_tracer:  # no need to recompute for second survey
                        self.survey[1] = self.compute_derivs_survey(ti=1)
                    else:
                        self.survey[1].deriv = self.survey[0].deriv

                fd = self.f_d(zz)
                Dd = self.D_d(zz)
                bd1 = self.survey[0].deriv["b1_d"](zz)
                xbd1 = self.survey[1].deriv["b1_d"](zz)
                fdd = self.f_dd(zz)
                Ddd = self.D_dd(zz)
                bdd1 = self.survey[0].deriv["b1_dd"](zz)
                xbdd1 = self.survey[1].deriv["b1_dd"](zz)
                params.extend([fd, Dd, bd1, xbd1, fdd, Ddd, bdd1, xbdd1])

                if GR:
                    # beta derivs
                    grd1 = self.get_beta_derivs(zz, ti=0)[0]
                    xgrd1 = self.get_beta_derivs(zz, ti=1)[0]
                    params.extend([grd1, xgrd1])

        return params

    # unpacking function for bispectrum
    def get_params(
        self,
        k1: ArrayLike,
        k2: ArrayLike,
        k3: ArrayLike | None = None,
        theta: ArrayLike | None = None,
        zz: ArrayLike = 0,
        t_n: int = 0,
        nonlin: bool = False,
        growth2: bool = False,
    ) -> tuple:
        """
        return arrays of redshift and k dependent parameters for bispectrum - nonlin and growth2 are legacy and are slowly being removed
        """
        k3, theta = utils.get_theta_k3(k1, k2, k3, theta)

        tracer = self.survey[t_n]

        Pk1 = self.Pk(k1)
        Pk2 = self.Pk(k2)
        Pk3 = self.Pk(k3)

        Pkd1 = self.Pk_d(k1)
        Pkd2 = self.Pk_d(k2)
        Pkd3 = self.Pk_d(k3)

        Pkdd1 = self.Pk_dd(k1)
        Pkdd2 = self.Pk_dd(k2)
        Pkdd3 = self.Pk_dd(k3)

        if self.nonlin:
            Pk1 = self.Pk_NL(k1, zz)
            Pk2 = self.Pk_NL(k2, zz)
            Pk3 = self.Pk_NL(k3, zz)

        # redshift dependendent terms
        d = self.comoving_dist(zz)

        K = 3 / 7  # from einstein-de-sitter
        C = 3 / 7
        if self.growth2:
            self.solve_second_order_KC()  # get K and C
            K = self.K_intp(zz)
            C = self.C_intp(zz)

        f = self.f(zz)
        D1 = self.D(zz)

        # survey stuff
        b1 = tracer.b_1(zz)
        b2 = tracer.b_2(zz)
        g2 = tracer.g_2(zz)
        return k1, k2, k3, theta, Pk1, Pk2, Pk3, Pkd1, Pkd2, Pkd3, Pkdd1, Pkdd2, Pkdd3, d, K, C, f, D1, b1, b2, g2

    def get_derivs(self, zz: ArrayLike, t_n: int = 0) -> tuple:
        """
        Get derivatives wrt comoving distance of redshift dependent parameters for radial evolution terms.
        """

        if "g2_d" not in self.survey[t_n].deriv:
            self.survey[t_n] = self.compute_derivs_survey(ti=t_n)  # compute derivs lazily

        tracer = self.survey[t_n]

        # 1st deriv
        fd = self.f_d(zz)
        Dd = self.D_d(zz)
        gd2 = tracer.deriv["g2_d"](zz)
        bd2 = tracer.deriv["b2_d"](zz)
        bd1 = tracer.deriv["b1_d"](zz)
        # 2nd deriv
        fdd = self.f_dd(zz)
        Ddd = self.D_dd(zz)
        gdd2 = tracer.deriv["g2_dd"](zz)
        bdd2 = tracer.deriv["b2_dd"](zz)
        bdd1 = tracer.deriv["b1_dd"](zz)
        return fd, Dd, gd2, bd2, bd1, fdd, Ddd, gdd2, bdd2, bdd1

    ############################################################## for multi_tracer bk
    def get_all_bias(self, bias: str, zz: ArrayLike) -> tuple:
        """Gets biases for all tracers - bispectrum"""
        biases = []
        for i in range(3):
            bias_func = getattr(self.survey[i], bias)
            biases.append(bias_func(zz))
        return tuple(biases)

    def unpack_bk(
        self,
        k1: ArrayLike,
        k2: ArrayLike,
        k3: ArrayLike | None = None,
        theta: ArrayLike | None = None,
        zz: ArrayLike = 0,
        GR: bool = False,
        fNL_type: str | None = None,
        WS: bool = False,
        RR: bool = False,
    ) -> list:
        """Helper function to unpack all necessary terms with flag for each different type of term

        Is multi-tracer compliant"""

        k3, theta = utils.get_theta_k3(k1, k2, k3, theta)

        # basic params
        if self.nonlin:
            Pk1 = self.Pk_NL(k1, zz)
            Pk2 = self.Pk_NL(k2, zz)
            Pk3 = self.Pk_NL(k3, zz)
        else:
            Pk1 = self.Pk(k1)
            Pk2 = self.Pk(k2)
            Pk3 = self.Pk(k3)

        f = self.f(zz)
        D1 = self.D(zz)

        K = 3 / 7  # from einstein-de-sitter
        C = 3 / 7
        if self.growth2:
            self.solve_second_order_KC()  # get K and C
            K = self.K_intp(zz)
            C = self.C_intp(zz)

        b1, xb1, yb1 = self.get_all_bias("b_1", zz)
        b2, xb2, yb2 = self.get_all_bias("b_2", zz)  # second order bias
        g2, xg2, yg2 = self.get_all_bias("g_2", zz)  # tidal bias

        params = [k1, k2, k3, theta, Pk1, Pk2, Pk3, f, D1, K, C, b1, xb1, yb1, b2, xb2, yb2, g2, xg2, yg2]

        return params
