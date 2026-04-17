"""
Main frontend forecasting class for forecasts over full surveys not just single bin
"""

from __future__ import annotations

import logging

# from numpy.typing import ArrayLike
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.survey_params import SurveyParams

from .core import BkForecast, Forecast, PkForecast
from .fisher import FisherMat
from .fisher_list import FisherList
from .sampler import Sampler

if TYPE_CHECKING:
    from cosmo_wap.main import ClassWAP


class FullForecast:
    def __init__(
        self,
        cosmo_funcs: ClassWAP,
        kmax_func: float | Callable | None = None,
        s_k: int = 2,
        nonlin: bool = False,
        N_bins: int | None = None,
        bkmax_func: float | Callable | None = None,
        WS_cut: bool = True,
        n_mu: int = 8,
        n_phi: int = 8,
    ) -> None:
        """
        Do full survey forecast over redshift bins
        First get relevant redshifts and ks for each redshift bin
        Calls BkForecast and PkForecast which compute for particular bin
        """

        # get number of redshift bins survey is split into for forecast...
        if not N_bins:
            N_bins = round((cosmo_funcs.z_max - cosmo_funcs.z_min) * 10)
        self.N_bins = N_bins

        z_lims = np.linspace(cosmo_funcs.z_min, cosmo_funcs.z_max, N_bins + 1)
        self.z_mid = (z_lims[:-1] + z_lims[1:]) / 2  # get bin centers
        self.z_bins = np.column_stack((z_lims[:-1], z_lims[1:]))

        if kmax_func is None:  # k -limit of analysis
            kmax_func = 0.1  # 0.1 *cosmo_funcs.h*(1+zz)**(2/(2+cosmo_funcs.n_s]))

        self.k_max_list = self.get_kmax_list(kmax_func)

        if bkmax_func is None:  # allow for different kmax for the bispectrum
            self.bk_max_list = self.k_max_list
        else:
            self.bk_max_list = self.get_kmax_list(bkmax_func)

        self.kmax_func = kmax_func
        self.bkmax_func = bkmax_func

        self.s_k = s_k
        self.WS_cut = WS_cut  # cut scales where WS expansion breaks down

        # basically we dont have an amazing system of including nonlinear effects
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            cosmo_funcs = utils.copy(cosmo_funcs)
            cosmo_funcs.nonlin = True
        self.nonlin = nonlin
        self.cosmo_funcs = cosmo_funcs

        # for covariances - need to increase when included integrated effects
        self.n_mu = n_mu
        self.n_phi = n_phi

        self.cf_mat = self.setup_multitracer()
        self.cf_mat_bk = self.setup_multitracer_bk()
        self.set_bias_placeholders()

    def set_bias_placeholders(self) -> None:
        """define lists of bias parameters for forecasting"""
        self.amp_bias = [
            "A_b_1",
            "X_b_1",
            "Y_b_1",
            "X_be",
            "X_Q",
            "Y_be",
            "Y_Q",
            "A_be",
            "A_Q",
            "X_b_2",
            "Y_b_2",
            "A_b_2",
        ]
        self.png_amp_bias = [
            "X_loc_b_01",
            "Y_loc_b_01",
            "A_loc_b_01",
            "X_eq_b_01",
            "Y_eq_b_01",
            "A_eq_b_01",
            "X_orth_b_01",
            "Y_orth_b_01",
            "A_orth_b_01",
            "A_loc_b_11",
        ]

    def get_kmax_list(self, kmax_func: float | Callable) -> np.ndarray:
        """Get list of k_max"""
        if callable(
            kmax_func
        ):  # is it a function - if not then it just constant (or an array if you really wanted it to be)
            return kmax_func(self.z_mid)
        else:
            return np.ones_like(self.z_mid) * kmax_func

    def setup_multitracer(self, cosmo_funcs: ClassWAP | None = None) -> list[list[ClassWAP]]:
        """So lets set up cosmo_funcs objects for each multi-tracer combination and store in a list.
        If multi-tracer:
        | XX XY |
        | YX YY |
        Could be 3x3 matrix if we define 3 tracers...
        we also need to define for bispectrum even with two tracers
        """
        if cosmo_funcs is None:
            cosmo_funcs = self.cosmo_funcs  # use the cosmo_funcs of the forecast object

        if cosmo_funcs.multi_tracer:
            cf_mat = []
            for i in range(cosmo_funcs.N_tracers):
                cf_row = []
                for j in range(cosmo_funcs.N_tracers):
                    cf = utils.copy(cosmo_funcs)  # create a copy of cosmo_funcs
                    cf.survey = [cosmo_funcs.survey[i], cosmo_funcs.survey[j]]
                    # cf.survey_params = [cosmo_funcs.survey_params[i], cosmo_funcs.survey_params[j]]
                    if i == j:  # then auto-correlation
                        cf.multi_tracer = False  # now single tracer
                        cf.n_g = cf.survey[0].n_g

                    cf_row.append(cf)
                cf_mat.append(cf_row)

            return cf_mat

        return [[cosmo_funcs]]

    def setup_multitracer_bk(
        self, cosmo_funcs: ClassWAP | None = None
    ) -> list[list[list[ClassWAP]]] | list[list[ClassWAP]]:
        """Now for bisepctrum
        If multi-tracer: 2 x 2 x 2 shape
        Could be 3x3x3 matrix if we define 3 tracers...
        """
        if cosmo_funcs is None:
            cosmo_funcs = self.cosmo_funcs  # use the cosmo_funcs of the forecast object

        if cosmo_funcs.multi_tracer:
            cf_matrix = []
            N = cosmo_funcs.N_tracers
            for i in range(N):
                cf_mat = []  # 2D slice
                for j in range(N):
                    cf_row = []
                    for k in range(N):
                        # Create a copy for the specific tracer combination
                        cf = utils.copy(cosmo_funcs)

                        # Map the three tracers
                        cf.survey = [cosmo_funcs.survey[i], cosmo_funcs.survey[j], cosmo_funcs.survey[k]]
                        # cf.survey_params = [cosmo_funcs.survey_params[i],cosmo_funcs.survey_params[j],cosmo_funcs.survey_params[k]]

                        # Logic for auto-correlation (when all three are the same)
                        if i == j == k:
                            cf.multi_tracer = False
                            cf.n_g = cf.survey[0].n_g
                        else:
                            cf.multi_tracer = True

                        cf_row.append(cf)
                    cf_mat.append(cf_row)
                cf_matrix.append(cf_mat)

            return cf_matrix

        return [[cosmo_funcs]]

    ######################################################### helper functions - just simplify calls slightly

    def get_pk_bin(
        self, i: int = 0, all_tracer: bool = False, cache: list[dict] | None = None, cov_terms: str | None = None
    ) -> PkForecast:
        """Get PkForecast object for a single redshift bin"""
        return PkForecast(
            self.z_bins[i],
            self.cosmo_funcs,
            self,
            k_max=self.k_max_list[i],
            all_tracer=all_tracer,
            cache=cache,
            cov_terms=cov_terms,
        )

    def get_bk_bin(
        self, i: int = 0, all_tracer: bool = False, cache: list[dict] | None = None, cov_terms: str | None = None
    ) -> BkForecast:
        """Get BkForecast object for a single redshift bin"""
        return BkForecast(
            self.z_bins[i],
            self.cosmo_funcs,
            self,
            k_max=self.bk_max_list[i],
            all_tracer=all_tracer,
            cache=cache,
            cov_terms=cov_terms,
        )

    ############################################################### simple SNR forecasts
    def pk_SNR(
        self,
        term: str,
        pkln: str,
        param: str | list[str] | None = None,
        param2: str | None = None,
        t: int = 0,
        verbose: bool = True,
        sigma: float | None = None,
        cov_terms: str | None = None,
        all_tracer: bool = False,
    ) -> np.ndarray:
        """
        Get SNR at several redshifts for a given survey and contribution - power spectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)), dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):
            foreclass = self.get_pk_bin(i, all_tracer=all_tracer, cov_terms=cov_terms)
            snr[i] = foreclass.SNR(term, ln=pkln, param=param, param2=param2, t=t, sigma=sigma)
        return snr

    def bk_SNR(
        self,
        term: str,
        bkln: str,
        param: str | list[str] | None = None,
        param2: str | None = None,
        m: int = 0,
        r: int = 0,
        s: int = 0,
        verbose: bool = True,
        sigma: float | None = None,
        cov_terms: str | None = None,
        all_tracer: bool = False,
    ) -> np.ndarray:
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.bk_max_list)), dtype=np.complex64)
        for i in tqdm(range(len(self.bk_max_list))) if verbose else range(len(self.k_max_list)):
            foreclass = self.get_bk_bin(i, all_tracer=all_tracer, cov_terms=cov_terms)
            snr[i] = foreclass.SNR(term, ln=bkln, param=param, param2=param2, m=m, r=r, s=s, sigma=sigma)
        return snr

    def combined_SNR(
        self,
        term: str,
        pkln: str,
        bkln: str,
        param: str | list[str] | None = None,
        param2: str | None = None,
        m: int = 0,
        t: int = 0,
        r: int = 0,
        s: int = 0,
        verbose: bool = True,
        sigma: float | None = None,
        cov_terms: str | None = None,
        all_tracer: bool = False,
    ) -> np.ndarray:
        """
        Get SNR at several redshifts for a given survey and contribution - powerspectrum + bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)), dtype=np.complex64)
        for i in range(len(self.k_max_list)):
            foreclass = Forecast(
                self.z_bins[i],
                self.cosmo_funcs,
                self,
                k_max=self.k_max_list[i],
                all_tracer=all_tracer,
                cov_terms=cov_terms,
            )
            snr[i] = foreclass.combined(
                term, pkln=pkln, bkln=bkln, param=param, param2=param2, t=t, r=r, s=s, sigma=sigma
            )
        return snr

    ######################################################## Routines for fishers and MCMC

    def _precompute_cache(self, param_list: list[str], dh: float = 1e-3) -> list[dict[str, Any]] | None:
        """
        Pre-computes the four cosmo_funcs objects needed for the five-point stencil - is less necessary now cosmo_funcs has been sped up x50
        for each cosmological parameter. This is done only once for the entire forecast instead of for each bin.
        """

        cache = [{}, {}, {}, {}, {}]  # for five point stencil - need 4 points - last entry is for h

        cosmo_params = [
            p for p in param_list if p in ["Omega_m", "Omega_b", "Omega_cdm", "A_s", "sigma8", "n_s", "h", "w0", "wa"]
        ]
        if cosmo_params:
            for param in cosmo_params:
                current_value = getattr(self.cosmo_funcs, param)
                h = dh * current_value if current_value != 0 else dh  # fallback for zero fiducial (e.g. wa)
                cache[-1][param] = h

                K_MAX = self.cosmo_funcs.K_MAX
                if K_MAX > 1 and not self.cosmo_funcs.compute_bias:  # also no point computing all of it!
                    K_MAX = 1

                for i, n in enumerate([2, 1, -1, -2]):
                    if self.cosmo_funcs.emulator:
                        cosmo_h, params = utils.get_cosmo(
                            **{param: current_value + n * h}, emulator=self.cosmo_funcs.emulator
                        )
                        kwargs = {"emulator": self.cosmo_funcs.emu, "params": params}
                    else:
                        cosmo_h = utils.get_cosmo(**{param: current_value + n * h}, k_max=K_MAX * self.cosmo_funcs.h)
                        kwargs = {}
                    cache[i][param] = cw.ClassWAP(cosmo_h, **kwargs)  # does not initialise survey

                    cosmo_h.struct_cleanup()
                    cosmo_h.empty()

        if cache[0] == {}:  # if empty
            # print('empty cache')
            return None
        return cache

    def _precompute_derivatives_and_covariances(
        self,
        param_list: list[str],
        terms: str = "NPP",
        cov_terms: str | None = None,
        pkln: str | None = None,
        bkln: str | None = None,
        t: int = 0,
        r: int = 0,
        s: int = 0,
        sigma: float | None = None,
        verbose: bool = True,
        all_tracer: bool = False,
        use_cache: bool = True,
        compute_cov: bool = True,
        bk_terms: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, np.ndarray]]], list[dict[str, np.ndarray]]]:
        """
        Precompute all values for fisher matrix - computes covariance and data vector for each parameter once for each bin
        Also can be used for just getting full data vector and covariance - used in Sampler
        """
        num_params = len(param_list)
        num_bins = len(self.z_bins)

        if use_cache:
            cache = self._precompute_cache(param_list)
        else:
            cache = None

        # Caching structures
        # derivs[param_idx][bin_idx] = {'pk': pk_deriv, 'bk': bk_deriv}
        data_vector = [[{} for _ in range(num_bins)] for _ in range(num_params)]
        inv_covs = [{} for _ in range(num_bins)]

        if verbose:
            logger.info("Step 1: Pre-computing derivatives and inverse covariances...")
        for i in tqdm(range(num_bins), disable=not verbose, desc="Bin Loop"):
            # --- Covariance Calculation (once per bin) ---New method to compute and cache all derivatives and inverse covariances once.
            if pkln:
                pk_fc = self.get_pk_bin(i, all_tracer=all_tracer, cache=cache, cov_terms=cov_terms)
                if compute_cov:
                    pk_cov_mat = pk_fc.get_cov_mat(pkln, sigma=sigma, n_mu=self.n_mu)
                    inv_covs[i]["pk"] = pk_fc.invert_matrix(pk_cov_mat)

            if bkln:
                bk_fc = self.get_bk_bin(i, all_tracer=all_tracer, cache=cache, cov_terms=cov_terms)
                if compute_cov:
                    bk_cov_mat = bk_fc.get_cov_mat(bkln, sigma=sigma, n_mu=self.n_mu, n_phi=self.n_phi)
                    inv_covs[i]["bk"] = bk_fc.invert_matrix(bk_cov_mat)

            # --- Get data vector (once per parameter per bin) - if parameter is not a term it computes the derivative of the terms wrt parameter 5 ---
            for j, param in enumerate(param_list):
                if pkln:
                    pk_deriv = pk_fc.get_data_vector(terms, pkln, param=param, t=t, sigma=sigma, **kwargs)
                    data_vector[j][i]["pk"] = pk_deriv
                if bkln:
                    bk_deriv = bk_fc.get_data_vector(bk_terms, bkln, param=param, r=r, s=s, sigma=sigma, **kwargs)
                    data_vector[j][i]["bk"] = bk_deriv

        return data_vector, inv_covs

    def _rename_composite_params(self, param_list: list[str | list[str]]) -> list[str]:
        """so if one "param" is a list itself - then lets just call our parameter in some frankenstein way"""
        param_list_names = []
        for param in param_list:
            if isinstance(param, list):
                param = "_".join(param)
            param_list_names.append(param)
        return param_list_names

    def get_fish(
        self,
        param_list: str | list[str],
        terms: str = "NPP",
        cov_terms: str | None = None,
        pkln: str | None = None,
        bkln: str | None = None,
        m: int = 0,
        t: int = 0,
        r: int = 0,
        s: int = 0,
        all_tracer: bool = False,
        verbose: bool = True,
        sigma: float | None = None,
        bias_list: str | list[str] | None = None,
        use_cache: bool = True,
        bk_terms: str | None = None,
        per_bin_params: str | list[str] | None = None,
        marginalize_per_bin: bool = True,
        **kwargs: Any,
    ) -> FisherMat:
        """
        Compute fisher minimising redundancy (only compute each data vector/covariance one for each bin (and parameter of relevant).
        This routine computes covariance and data vector for each parameter once for each bin, then assembles the Fisher matrix.
        Also allows for computation of best fit bias using bias terms which can be a list. - this is also the most efficient way to do this!

        per_bin_params: parameters that take an independent value in every redshift bin
            (e.g. galaxy bias). With marginalize_per_bin=True (default) the per-bin block
            is marginalised out via a Schur complement and the returned FisherMat covers
            only the global params (per-bin covariances available via per_bin_cov).
            With marginalize_per_bin=False the full block matrix is returned with expanded
            names like "b_1[k]".
        """
        if not isinstance(param_list, list):  # if item is not a list, make it one
            param_list = [param_list]

        if per_bin_params is None:
            per_bin_params = []
        elif not isinstance(per_bin_params, list):
            per_bin_params = [per_bin_params]

        if bk_terms is None:
            bk_terms = terms  # if not specified, use same terms for bispectrum as power spectrum

        param_list_names = self._rename_composite_params(
            param_list
        )  # get combined names for list of params - is used here to save biases
        per_bin_names = self._rename_composite_params(per_bin_params)

        N_A = len(param_list)
        N_B = len(per_bin_params)
        N_bins = len(self.z_bins)

        if bias_list:
            if not isinstance(bias_list, list):  # if item is not a list, make it one
                bias_list = [bias_list]

            # Order: [globals, per-bin, biases] so existing bias indexing logic stays simple
            all_param_list = param_list + per_bin_params + bias_list

            N_b = len(bias_list)
            bias = [{} for _ in range(N_b)]  # empty dicts for each bias term
        else:
            bias = None
            all_param_list = param_list + per_bin_params

        # Precompute
        derivs, inv_covs = self._precompute_derivatives_and_covariances(
            all_param_list,
            terms,
            cov_terms,
            pkln,
            bkln,
            t,
            r,
            s,
            sigma=sigma,
            verbose=verbose,
            all_tracer=all_tracer,
            use_cache=use_cache,
            bk_terms=bk_terms,
            **kwargs,
        )

        # lets save them
        self.derivs = derivs
        self.inv_covs = inv_covs

        def _fij(i: int, j: int, bin_idx: int) -> float:
            """Single-bin Fisher contribution between params i and j (indices into all_param_list). Uses cached derivatives and inverse covariances."""
            val = 0.0
            if pkln:
                d1 = derivs[i][bin_idx]["pk"]
                d2 = derivs[j][bin_idx]["pk"]
                inv_cov = inv_covs[bin_idx]["pk"]
                val += np.sum(np.einsum("ik,ijk,jk->k", d1, inv_cov, np.conjugate(d2)).real)
            if bkln:
                d1 = derivs[i][bin_idx]["bk"]
                d2 = derivs[j][bin_idx]["bk"]
                inv_cov = inv_covs[bin_idx]["bk"]
                val += np.sum(np.einsum("ik,ijk,jk->k", d1, inv_cov, np.conjugate(d2)).real)
            return val

        if verbose:
            logger.info("Step 2: Assembling Fisher matrix...")

        # Block accumulators (per-bin params live in indices [N_A : N_A+N_B] of all_param_list)
        F_AA = np.zeros((N_A, N_A))
        # fishers per bin
        F_AB = np.zeros((N_bins, N_A, N_B)) if N_B else None
        F_BB = np.zeros((N_bins, N_B, N_B)) if N_B else None

        for k in range(N_bins):
            # global-global (sum over all bins)
            for i in range(N_A):
                for j in range(i, N_A):
                    f = _fij(i, j, k)
                    F_AA[i, j] += f
                    if i != j:
                        F_AA[j, i] += f

            # cross global x per-bin (only bin k contributes to bin k's block)
            if N_B:
                for i in range(N_A):
                    for j in range(N_B):
                        F_AB[k, i, j] = _fij(i, N_A + j, k)
                # per-bin self
                for i in range(N_B):
                    for j in range(i, N_B):
                        f = _fij(N_A + i, N_A + j, k)
                        F_BB[k, i, j] = f
                        if i != j:
                            F_BB[k, j, i] = f

        # Bias terms — only computed against global params (per-bin nuisance ignored)
        if bias_list:
            bias_offset = N_A + N_B  # bias entries start here in all_param_list
            for i in range(N_A):
                for j in range(N_b):
                    bij = 0.0
                    for k in range(N_bins):
                        bij += _fij(i, bias_offset + j, k)
                    bias[j][param_list_names[i]] = bij / F_AA[i, i]

        # store stuff for use in FishMat
        config = {"terms": terms, "pkln": pkln, "bkln": bkln, "t": t, "r": r, "s": s, "sigma": sigma, "bias": bias}

        if N_B == 0:
            return FisherMat(F_AA, self, param_list, config=config)

        if marginalize_per_bin:
            # Schur complement: F_marg = F_AA - sum_k F_AB[k] @ inv(F_BB[k]) @ F_AB[k].T
            per_bin_cov = np.zeros((N_bins, N_B, N_B))
            F_marg = F_AA.copy()
            for k in range(N_bins):
                Fbb_inv = np.linalg.inv(F_BB[k])
                per_bin_cov[k] = Fbb_inv
                F_marg -= F_AB[k] @ Fbb_inv @ F_AB[k].T
            return FisherMat(
                F_marg,
                self,
                param_list,
                config=config,
                per_bin_cov=per_bin_cov,
                per_bin_param_list=per_bin_names,
            )

        # Full block matrix: [globals, per_bin_bin0, per_bin_bin1, ...]
        # Build it!
        N_full = N_A + N_B * N_bins
        F_full = np.zeros((N_full, N_full))
        F_full[:N_A, :N_A] = F_AA
        for k in range(N_bins):
            sl = slice(N_A + k * N_B, N_A + (k + 1) * N_B)
            F_full[:N_A, sl] = F_AB[k]
            F_full[sl, :N_A] = F_AB[k].T
            F_full[sl, sl] = F_BB[k]
        expanded_names = list(param_list_names) + [f"{name}[{k}]" for k in range(N_bins) for name in per_bin_names]
        return FisherMat(
            F_full,
            self,
            expanded_names,
            config=config,
            per_bin_param_list=per_bin_names,
        )

    def best_fit_bias(
        self,
        param: str | list[str],
        bias_term: str | list[str],
        terms: str = "NPP",
        pkln: str | None = None,
        bkln: str | None = None,
        t: int = 0,
        r: int = 0,
        s: int = 0,
        verbose: bool = True,
        sigma: float | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Get best fit bias on one parameter if a particular contribution is ignored
        New, more efficient method uses FisherMat instance - basically is just a little wrapper of get fish method.
        bfb is a dictionary and if bias_term is a list - bfb is the sum from all the terms."""

        fish_mat = self.get_fish(
            param, terms=terms, pkln=pkln, bkln=bkln, t=t, r=r, s=s, verbose=verbose, sigma=sigma, bias_list=bias_term
        )

        bfb = fish_mat.bias[-1]  # is list containing a dictionary for each bias term
        fish = np.diag(fish_mat.fisher_matrix)  # is array - ignore marginalisation

        return bfb, fish

    def get_fish_list(
        self,
        param_list: list[str],
        cuts: list[float],
        splits: list[float],
        terms: str = "NPP",
        cov_terms: str | None = None,
        pkln: str | None = None,
        bkln: str | None = None,
        m: int = 0,
        t: int = 0,
        r: int = 0,
        s: int = 0,
        all_tracer: bool = False,
        verbose: bool = True,
        bk_terms: str | None = None,
        **kwargs: Any,
    ) -> FisherList:
        fish_list = [[None for _ in range(len(splits))] for _ in range(len(cuts))]
        survey_params = SurveyParams()  # initialise SurveyParams object
        for i, cut in tqdm(enumerate(cuts), disable=not verbose):  # loop over cuts
            for j, split in enumerate(splits):  # loop over splits
                if split > cut:
                    cosmo = self.cosmo_funcs.cosmo
                    cosmo_funcs = cw.ClassWAP(
                        cosmo,
                        survey_params.Euclid(cosmo, cut=cut).BF_split(split),
                        compute_bias=self.cosmo_funcs.compute_bias,
                        verbose=False,
                    )
                    forecast = cw.forecast.FullForecast(
                        cosmo_funcs,
                        s_k=self.s_k,
                        kmax_func=self.kmax_func,
                        bkmax_func=self.bkmax_func,
                        N_bins=self.N_bins,
                        nonlin=self.nonlin,
                    )
                    fish_list[i][j] = forecast.get_fish(
                        param_list,
                        terms=terms,
                        pkln=pkln,
                        bkln=bkln,
                        cov_terms=cov_terms,
                        all_tracer=all_tracer,
                        verbose=False,
                        bk_terms=bk_terms,
                        **kwargs,
                    )
        return FisherList(fish_list, self, param_list, cuts, splits)

    def sampler(
        self,
        param_list: list[str],
        terms: str | None = None,
        cov_terms: str | None = None,
        bias_list: list[str] | None = None,
        pkln: str | None = None,
        bkln: str | None = None,
        R_stop: float = 0.005,
        max_tries: int = 100,
        name: str | None = None,
        planck_prior: bool = False,
        all_tracer: bool = False,
        verbose: bool = True,
        sigma: float | None = None,
        bk_terms: str | None = None,
        **kwargs: Any,
    ) -> Sampler:
        """Define Sampler instance which is used for MCMC samples"""

        return Sampler(
            self,
            param_list,
            terms=terms,
            cov_terms=cov_terms,
            bias_list=bias_list,
            pkln=pkln,
            bkln=bkln,
            R_stop=R_stop,
            max_tries=max_tries,
            name=name,
            planck_prior=planck_prior,
            all_tracer=all_tracer,
            verbose=verbose,
            sigma=sigma,
            bk_terms=bk_terms,
            **kwargs,
        )
