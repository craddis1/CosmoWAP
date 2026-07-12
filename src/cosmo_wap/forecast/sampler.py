"""
MCMC Sampler using Cobaya for cosmological parameter sampling.

Uses Cobaya MCMC sampler to sample for a given likelihoods.
Allows us to drop the assumption of gaussianity of the posterior we have in the fisher.
Heavily reliant on CosmoPower to make sampling over cosmological parameters efficient.
"""

import logging
import os
import pickle
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
from chainconsumer import Chain, ChainConfig
from cobaya import run
from scipy import stats
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

import cosmo_wap as cw
import cosmo_wap.bk as bk
import cosmo_wap.pk as pk
from cosmo_wap.lib import utils
from cosmo_wap.lib.lf_priors import LFBiasPrior

from .base_posterior import BasePosterior


class Sampler(BasePosterior):
    """MCMC Sampler with cobaya with ChainConsumer plots.
    Assumes gaussian likelihood with parameter independent covariances."""

    # params whose move forces a cosmology rebuild (slow block for fast/slow dragging)
    COSMO_PARAMS = {"Omega_m", "Omega_cdm", "Omega_b", "A_s", "ln_A_s", "sigma8", "n_s", "h", "gamma"}

    def __init__(
        self,
        forecast,
        param_list,
        terms=None,
        cov_terms=None,
        all_tracer=False,
        bias_list=None,
        bk_bias_list=None,
        pkln=None,
        bkln=None,
        R_stop=0.005,
        max_tries=200,
        name=None,
        planck_prior=False,
        lf_prior=False,
        bk_terms=None,
        bk_st=False,
        kernels=None,
        mu_grid=None,
        per_bin_params=None,
        fisher_covmat=True,
        drag=True,
        **kwargs,
    ):
        super().__init__(forecast, param_list, name=name)

        self.pkln = pkln
        self.bkln = bkln
        # terms which to compute that are parameter dependent. terms=None is a power-spectrum-only
        # kernel model (the signal comes entirely from `kernels`); it requires bkln=None since
        # kernels does not supply a bispectrum.
        self.terms = terms
        if bk_terms is None:
            bk_terms = terms
        self.bk_terms = bk_terms
        if self.bkln and self.bk_terms is None:
            raise ValueError(
                "terms=None is power-spectrum-only (kernels has no bispectrum); pass bk_terms or set bkln=None."
            )
        # numeric-mu pk kernels summed onto `terms` (pk only); None keeps the analytic-only model
        self.kernels = kernels
        self.mu_grid = mu_grid
        # use planck covariance as prior
        self.planck_prior = planck_prior
        # forward-modelled luminosity-function prior on the per-bin b_e/Q (True or a LFBiasPrior)
        self.lf_prior = lf_prior
        # full mutli-tracer analysis
        self.all_tracer = all_tracer
        # bk_st: force bispectrum onto single-tracer pipeline using survey[0] (no-op when not multi-tracer)
        self.bk_st = bk_st
        # config needed to rebuild a matching Fisher for the proposal covmat
        self.cov_terms = cov_terms
        self.fisher_covmat = fisher_covmat
        # fast/slow dragging - split cosmology (slow) from all other (fast) params
        self.drag = drag
        # LRU of built cosmologies keyed on the cosmo sub-vector; size >= 2 so dragging's
        # two slow anchors (s0, s1) both stay cached across the interpolation loop
        self._cosmo_cache = OrderedDict()
        self._cosmo_cache_size = 4

        # per-bin nuisance params: one amplitude on b_1 (etc.) per redshift bin, marginalised in the MCMC.
        # Tracer-specific entries use the fisher convention: 'Xb_1'/'YQ' apply to that tracer only.
        if per_bin_params is None:
            per_bin_params = []
        elif not isinstance(per_bin_params, list):
            per_bin_params = [per_bin_params]
        self.per_bin_params = self.forecast._rename_composite_params(per_bin_params)
        for p in self.per_bin_params:
            base, tracers = self._split_tracer(p)
            if base not in forecast.biases:
                raise NotImplementedError(f"per_bin_params entry '{p}' is not a supported per-bin bias.")
            if len(tracers) == 1 and not self.cosmo_funcs.multi_tracer:
                raise ValueError(f"per_bin_params entry '{p}' is tracer-specific but the survey is single-tracer.")
        # global params before expansion - kept for save/load reconstruction
        self.global_param_list = list(self.param_list)
        # sampled names like 'b_1_0' ... 'b_1_{N-1}', appended to the sampled parameter list
        self.per_bin_names = [f"{p}_{i}" for p in self.per_bin_params for i in range(forecast.N_bins)]
        self.param_list = self.param_list + self.per_bin_names
        for name in self.per_bin_names:
            self.fiducial[name] = 1.0  # amplitude fiducial

        if "fNL" in kwargs.keys():
            self.fNL = kwargs["fNL"]
        else:
            self.fNL = 0

        if bias_list is None:
            bias_list = []
        # bk_bias_list overrides bias_list on the bk side for the 'true' theory term collection
        bk_bias_list = bias_list if bk_bias_list is None else bk_bias_list

        self.pk_fc = []
        self.bk_fc = []
        for i in range(forecast.N_bins):
            self.pk_fc.append(forecast.get_pk_bin(i, all_tracer=all_tracer, cov_terms=cov_terms))
            self.bk_fc.append(
                forecast.get_bk_bin(
                    i,
                    all_tracer=False if bk_st else all_tracer,
                    cov_terms=cov_terms,
                    cosmo_funcs=forecast._bk_st_cosmo(bk_st),
                )
            )

        all_terms = [
            term for term in (terms or []) + param_list + bias_list if term in self.cosmo_funcs.term_list
        ]  # get list of needed terms to compute full 'true' theory (terms=None -> kernels-only)
        all_bk_terms = [
            term for term in (self.bk_terms or []) + param_list + bk_bias_list if term in self.cosmo_funcs.term_list
        ]
        # so this just gets total contribution - i.e. true theory - and also parameter independent covariance
        self.data, self.inv_covs = forecast._precompute_derivatives_and_covariances(
            [all_terms],
            pkln=pkln,
            bkln=bkln,
            verbose=False,
            all_tracer=all_tracer,
            bk_st=bk_st,
            cov_terms=cov_terms,
            bk_terms=all_bk_terms,
            bk_param_list=[all_bk_terms],
            kernels=self.kernels,
            mu_grid=self.mu_grid,
            fNL=0,
        )

        # set up cobaya sampler - define priors, starting value and initial step
        # Cosmological Priors
        cosmo_params = {
            "n_s": self.get_prior(0.84, 1.1, 0.9665, 5e-5),
            "h": self.get_prior(0.64, 0.82, 0.6776, 1e-3),
            "A_s": self.get_prior(6e-10, 4.8e-9, 2.105e-9, 1e-11),
            "ln_A_s": self.get_prior(1.79, 3.87, 3.047, 5e-3),  # ln(10^10 A_s): same range as A_s, well-conditioned
            "Omega_m": self.get_prior(0.17, 0.42, 0.31, 5e-5),
            "Omega_cdm": self.get_prior(0.13, 0.38, 0.26, 5e-5),
            "Omega_b": self.get_prior(0.041, 0.057, 0.049, 1e-5),
        }

        # fNL Parameters (Wide priors)
        fnl_params = {
            k: self.get_prior(-100, 100, ref=0, proposal=0.01) for k in ["fNL", "fNL_loc", "fNL_eq", "fNL_orth"]
        }

        # Theory Amplitude Parameters
        theory_params = {k: self.get_prior(-100, 100) for k in ["GR2", "WS2", "WA2"]}

        # b1 amplitude Parameters (Narrow priors around 1.0)
        b1_prior = {k: self.get_prior(0.8, 1.2, 1.0, 1e-2) for k in ["A_b_1", "X_b_1", "Y_b_1"]}

        # per-bin nuisance amplitudes (multiplicative, around 1.0), one per redshift bin.
        # b_1 is tightly constrained; selection functions (Q, be) inherit the wide prior of
        # their global A_Q/A_be counterparts since they are far less certain.
        per_bin_bounds = {"b_1": (0.8, 1.2, 1e-2)}  # others fall back to the wide lum-style prior
        per_bin_prior = {}
        for p in self.per_bin_params:
            # look up by the tracer-stripped base so Xb_1/Yb_1 share the tight b_1 prior
            lo, hi, prop = per_bin_bounds.get(self._split_tracer(p)[0], (-50, 50, None))
            for i in range(forecast.N_bins):
                per_bin_prior[f"{p}_{i}"] = self.get_prior(lo, hi, ref=1.0, proposal=prop)

        # Luminosity priors - Q and be and b_2
        lum_prior = {k: self.get_prior(-50, 50) for k in forecast.amp_bias[3:]}

        # PNG amplitude Parameters
        pngbias_prior = {k: self.get_prior(-100, 100) for k in forecast.png_amp_bias}

        # Combine everything
        self.prior_dict = {
            **cosmo_params,
            **fnl_params,
            **theory_params,
            **b1_prior,
            **per_bin_prior,
            **lum_prior,
            **pngbias_prior,
        }

        self.set_info(self.param_list, R_stop, max_tries)

    @staticmethod
    def _split_tracer(param):
        """Split a per-bin/LF param into (base, tracers) - fisher convention:
        'Xb_1' -> ('b_1', ('X',)); 'b_1' -> ('b_1', ('X', 'Y'))."""
        if param[:1] in ("X", "Y"):
            return param[1:], (param[0],)
        return param, ("X", "Y")

    def get_prior(self, min_val, max_val, ref=1, proposal=None):
        """Helper to standardize dictionary creation."""
        return {
            "prior": {"min": min_val, "max": max_val},
            "ref": ref,
            "proposal": proposal or (max_val - min_val) / 100,  # Default proposal if not provided
        }

    def set_info(self, param_list, R_stop, max_tries):
        """Sets cobaya info for given parameters"""
        mcmc = {"Rminus1_stop": R_stop, "max_tries": max_tries}
        if self.fisher_covmat:  # seed the proposal with the inverse-Fisher covariance
            covmat, covmat_params = self.get_fisher_covmat()
            if covmat is not None:
                mcmc["covmat"] = covmat
                mcmc["covmat_params"] = covmat_params

        # fast/slow split - one external likelihood takes all params so cobaya can't infer it.
        # slow = cosmology (rebuilds ClassWAP), fast = all other params; pays off via update_cosmo_funcs cache
        slow = [p for p in param_list if p in self.COSMO_PARAMS]
        fast = [p for p in param_list if p not in self.COSMO_PARAMS]
        if self.drag and slow and fast:
            # cobaya can't measure our speeds (one likelihood owns all params), so the fast
            # oversampling factor is set from the measured cache speed-up in run()
            mcmc["blocking"] = [[1, slow], [1, fast]]  # order slow->fast; fast factor tuned in run()
            mcmc["drag"] = True  # Neal fast-dragging over the fast block

        self.info = {
            "likelihood": {"cosmowap_likelihood": {"external": self.get_likelihood, "input_params": param_list}},
            "params": {key: self.prior_dict[key] for key in param_list},
            "sampler": {"mcmc": mcmc},
        }
        if self.planck_prior:
            self.info = self.set_planck_prior(self.info)
        if self.lf_prior is not False:
            self.info = self.set_lf_prior(self.info)

    def get_fisher_covmat(self):
        """Inverse-Fisher over the global params -> cobaya initial proposal covmat.

        Gives the MCMC the correct (tight, degenerate) correlation structure from the
        start instead of guessing a diagonal proposal, computed for the same data/cov
        config as the likelihood. Per-bin nuisance params are Schur-marginalised out of
        the global block (fisher works in absolute bias units, but only the marginalised
        global block is used so the units drop out); cobaya fills the per-bin entries
        from their proposal widths (a partial covmat is fine).
        Returns (None, None) if the Fisher is singular/non-finite so the run falls
        back to proposal widths rather than crashing.
        """
        try:
            fish = self.forecast.get_fish(
                self.global_param_list,
                terms=self.terms,
                bk_terms=self.bk_terms,
                pkln=self.pkln,
                bkln=self.bkln,
                cov_terms=self.cov_terms,
                all_tracer=self.all_tracer,
                bk_st=self.bk_st,
                kernels=self.kernels,
                mu_grid=self.mu_grid,
                per_bin_params=self.per_bin_params or None,
                lf_prior=self.lf_prior if self.per_bin_params else False,
                marginalize_per_bin=True,
                verbose=False,
            )
            if self.planck_prior:  # if we have planck prior
                fish = fish.add_planck_prior()
        except Exception as exc:  # best-effort: a covmat failure must not kill the run
            logger.warning("Fisher proposal covmat failed (%s); falling back to proposal widths.", exc)
            return None, None

        covmat = np.asarray(fish.covariance)
        if not np.all(np.isfinite(covmat)):
            logger.warning("Fisher covariance is non-finite (singular?); falling back to proposal widths.")
            return None, None
        # unconstrained param (sigma wider than its sampled range) - proposals would never be accepted
        sigma = np.sqrt(np.abs(np.diag(covmat)))
        widths = [self.prior_dict[p]["prior"]["max"] - self.prior_dict[p]["prior"]["min"] for p in fish.param_list]
        unconstrained = [p for p, sig, width in zip(fish.param_list, sigma, widths) if sig > width]
        if unconstrained:
            raise ValueError(f"Fisher gives no constraint on {unconstrained} - do they enter the theory (terms)?")
        # cobaya requires a symmetric, positive-definite proposal covmat. inv(Fisher)
        # carries floating-point asymmetry, and the strong A_s/n_s/Omega_b/h (and bias
        # amplitude) degeneracies leave the Fisher near-singular -> tiny negative
        # eigenvalues. Symmetrize and clip eigenvalues to a small positive floor so we
        # keep the real degenerate correlation structure instead of discarding it.
        covmat = 0.5 * (covmat + covmat.T)
        w, V = np.linalg.eigh(covmat)
        if w.max() <= 0:
            logger.warning("Fisher covariance has no positive directions; falling back to proposal widths.")
            return None, None
        floor = 1e-12 * w.max()
        if w.min() < floor:
            logger.warning("Fisher covariance not positive-definite (min eig %.2e); regularizing.", w.min())
            w = np.clip(w, floor, None)
            covmat = (V * w) @ V.T
            covmat = 0.5 * (covmat + covmat.T)
        return covmat, list(fish.param_list)

    def set_planck_prior(self, info):
        """Use planck constraints to set priors.
        So we need to define function to describe the planck prior likelihood:
        log(L)=-(1/2)*(p-mu)*c^{-1}(p-mu)^T"""

        cov = self.planck_cov()  # get NxN parameter covariance
        inv_cov = np.linalg.inv(cov)  # NxN

        # the cosmology params this prior constrains (ln_A_s and A_s are mutually exclusive;
        # planck_cov is built in the matching amplitude's units). Same order as planck_cov.
        amp = "ln_A_s" if "ln_A_s" in self.param_list else "A_s"
        params = ["Omega_b", "Omega_cdm", "theta", "tau", amp, "n_s"]
        prior_params = [p for p in self.param_list if p in params]
        means = np.array([getattr(self.cosmo_funcs, p) for p in prior_params])  # fiducials

        def planck_prior(**kwargs):
            # cobaya passes the parameters by name; index by name (input_params = prior_params)
            d = np.array([kwargs[p] for p in prior_params]) - means
            return -0.5 * d @ inv_cov @ d

        info["likelihood"]["prior"] = {"external": planck_prior, "input_params": prior_params}
        return info

    def set_lf_prior(self, info):
        """Forward-modelled luminosity-function prior on the per-bin b_e/Q.
        We work with amplitudes as it allows us to have the same proposal and same scale.
        Adds the quadratic penalty: log L = -1/2 sum_k (a_k - 1)^T Cinv_k (a_k - 1), coupling b_e_k <-> Q_k per bin.
        For a bright/faint split the components are the per-tracer 'Xbe'/'XQ'/'Ybe'/'YQ' and the
        bright-faint correlation through the shared luminosity function is kept.
        """

        z_mid = self.forecast.z_mid
        if self.lf_prior is True:
            survey = next((s for s in self.cosmo_funcs.survey_params if hasattr(s, "LF")), None)
            if survey is None:
                raise ValueError("No survey with a luminosity function found - cannot build lf_prior.")
            bias_prior = LFBiasPrior.from_survey(survey)
        else:  # pre-built prior (LFBiasPrior or ConstantBiasPrior)
            bias_prior = self.lf_prior

        targets = [p for p in self.per_bin_params if p in bias_prior.components]
        if not targets:
            raise ValueError(f"lf_prior needs per_bin_params overlapping {bias_prior.components}.")

        cov = bias_prior.covariance(z_mid, targets)  # (N_bins, m, m) absolute (b_e, Q)

        # fiducial selection functions map the absolute covariance into amplitude space -
        # each target reads its own tracer's survey (X = tracer 0, Y = tracer 1)
        tracer_survey = {["X", "Y"][s.t]: s for s in set(self.cosmo_funcs.survey)}
        fid = {}
        for target in targets:
            base, tracers = self._split_tracer(target)
            fid[target] = getattr(tracer_survey[tracers[0]], base)(z_mid)

        inv_cov = np.zeros((self.forecast.N_bins, len(targets), len(targets)))
        for k in range(self.forecast.N_bins):
            Dinv = np.diag([1.0 / fid[t][k] for t in targets])
            inv_cov[k] = np.linalg.inv(Dinv @ cov[k] @ Dinv)

        # sampled amplitude names per bin, e.g. [['be_0','Q_0'], ['be_1','Q_1'], ...]
        bin_names = [[f"{t}_{k}" for t in targets] for k in range(self.forecast.N_bins)]
        lf_params = [n for names in bin_names for n in names]

        def lf_prior(**kwargs):
            # cobaya passes the parameters by name; index by name (input_params = lf_params)
            logp = 0.0
            for k in range(self.forecast.N_bins):
                d = np.array([kwargs[n] for n in bin_names[k]]) - 1.0  # deviation from amplitude fiducial 1.0
                logp += -0.5 * d @ inv_cov[k] @ d
            return logp

        info["likelihood"]["lf_prior"] = {"external": lf_prior, "input_params": lf_params}
        return info

    def update_cosmo_funcs(self, param_vals):
        """Update the cosmology for each sample.

        Caches the built ClassWAP on the cosmology sub-vector so steps that leave the
        cosmology unchanged reuse it instead of rebuilding CLASS/the emulator. Dragging
        alternates between two slow anchors per step, hence an LRU (not a single slot);
        cosmologies evicted past the cache size have their CLASS C-memory freed.
        """
        cosmo_kwargs = {}
        for i, param in enumerate(self.param_list):
            if param in ["Omega_m", "Omega_cdm", "Omega_b", "A_s", "ln_A_s", "sigma8", "n_s", "h"]:
                cosmo_kwargs[param] = param_vals[i]
        gamma = param_vals[self.param_list.index("gamma")] if "gamma" in self.param_list else None

        # reuse the cached cosmology when only non-cosmology params changed
        key = (tuple(sorted(cosmo_kwargs.items())), gamma)
        if key in self._cosmo_cache:
            self._cosmo_cache.move_to_end(key)  # mark most-recently used
            return self._cosmo_cache[key]

        # change survey params
        if cosmo_kwargs:
            if self.cosmo_funcs.emulator:  # much quicker!
                cosmo_kwargs["emulator"] = True
                cosmo, params = utils.get_cosmo(
                    **cosmo_kwargs, k_max=self.cosmo_funcs.K_MAX * self.cosmo_funcs.h
                )  # update cosmology for change in param
                other_kwarg = {"emulator": self.cosmo_funcs.emu, "params": params}

            else:
                cosmo = utils.get_cosmo(**cosmo_kwargs, k_max=self.cosmo_funcs.K_MAX * self.cosmo_funcs.h)
                other_kwarg = {}

            cosmo_funcs = cw.ClassWAP(
                cosmo,
                self.cosmo_funcs.survey_params,
                compute_bias=self.cosmo_funcs.compute_bias,
                verbose=self.cosmo_funcs.verbose,
                fast=True,
                **other_kwarg,
            )
        else:
            cosmo_funcs = utils.copy(self.cosmo_funcs)

        # override growth rate: f(z) = Omega_m(z)^gamma
        if gamma is not None:
            zz = np.linspace(0, cosmo_funcs.z_max, 100)
            cosmo_funcs.f = CubicSpline(zz, cosmo_funcs.Om_m(zz) ** gamma)
            cosmo_funcs.compute_derivs_cosmo()

        self._cosmo_cache[key] = cosmo_funcs
        # evict least-recently-used past the cache size, freeing its CLASS C-memory
        # (copies of the master share its cosmo, so never free that one)
        while len(self._cosmo_cache) > self._cosmo_cache_size:
            _, old = self._cosmo_cache.popitem(last=False)
            if old.cosmo is not self.cosmo_funcs.cosmo:
                old.cosmo.struct_cleanup()
                old.cosmo.empty()
        return cosmo_funcs

    def get_theory(self, param_vals):
        """
        Get data vector for given MCMC call - data vector is shape [z_bin]['pk'][k_bin]
        """
        cosmo_funcs = self.update_cosmo_funcs(param_vals)  # the cosmology part (cached)

        kwargs = {}  # create dict which is fed into function
        kwargs["fNL"] = self.fNL  # useful to set default to 0 - otherwise without fNL as parameter default would be 1
        for i, param in enumerate(self.param_list):
            if param in [
                "fNL",
                "fNL_loc",
                "fNL_eq",
                "fNL_orth",
                "t",
                "r",
                "s",
            ]:  # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
                kwargs[param] = param_vals[i]

        # scale the global amplitude-bias params on the (cached) survey bias, restored on exit
        with self._amplitude_bias(cosmo_funcs, param_vals):
            # setup multiracer permutations - get cf_list
            if self.all_tracer:
                cf_mat = self.forecast.setup_multitracer(cosmo_funcs)
                cf_mat_bk = self.forecast.setup_multitracer_bk(cosmo_funcs)
                cf_list = [cf_mat[0][0], cf_mat[0][1], cf_mat[1][1]]
                cf_list_bk = [cf_mat_bk[0][0][0], cf_mat_bk[0][0][1], cf_mat_bk[0][1][1], cf_mat_bk[1][1][1]]
            else:
                cf_list = [cosmo_funcs]
                cf_list_bk = [cosmo_funcs]

            if self.bk_st:  # collapse Bk to single-tracer auto on survey[0]
                cf_list_bk = cf_list_bk[:1]

            # Caching structures
            # derivs[bin_idx] = {'pk': pk_deriv, 'bk': bk_deriv}
            d_v = [{} for _ in range(self.forecast.N_bins)]

            # now change this for full multi-tracer lengths with odd pk_l
            for i in range(self.forecast.N_bins):
                # apply this bin's per-bin nuisance amplitudes (e.g. b_1) in place, restored after the bin
                with self._per_bin_bias(cosmo_funcs, param_vals, i):
                    # get powerspectrum data vector
                    if self.pkln:
                        d_v[i]["pk"] = self.get_pk_d1(i, self.terms, self.pkln, cf_list, cosmo_funcs, **kwargs)
                    if self.bkln:  # get bispectrum data vector
                        d_v[i]["bk"] = self.get_bk_d1(i, self.bk_terms, self.bkln, cf_list_bk, **kwargs)

            # ok a little weird but may be useful later i guess - allows sample of term like alpha_GR
            for i, param in enumerate(self.param_list):
                if param in self.cosmo_funcs.term_list:
                    for j in range(self.forecast.N_bins):
                        if self.pkln:
                            # kernels are already in the base-terms vector above - don't re-add them here
                            d_v[j]["pk"] += (param_vals[i]) * self.get_pk_d1(
                                j, param, self.pkln, cf_list, cosmo_funcs, with_kernels=False, **kwargs
                            )
                        if self.bkln:
                            d_v[j]["bk"] += (param_vals[i]) * self.get_bk_d1(j, param, self.bkln, cf_list_bk, **kwargs)

        return d_v

    @contextmanager
    def _amplitude_bias(self, cosmo_funcs, param_vals):
        """Temporarily scale the global amplitude-bias params (e.g. X_b_1, A_loc_b_01) on the
        survey bias, restored on exit - so the cached cosmo_funcs is reused without the edits
        accumulating across samples. Mirrors the per-bin path but for the global amplitudes.
        """
        cf_surveys = list(set(cosmo_funcs.survey))  # get unique tracers
        restore = []  # (obj, attr, original_func), undone last-applied-first
        try:
            for i, param in enumerate(self.param_list):
                # so now only for each survey...
                if param in self.forecast.amp_bias:
                    tmp_param = param[2:]  # i.e get b_1 from X_b_1
                    for cf_survey in cf_surveys:
                        if param[0] in ["X", "Y"] and ["X", "Y"][cf_survey.t] is not param[0]:
                            continue  # tracer-specific bias: skip the other tracer
                        restore.append((cf_survey, tmp_param, getattr(cf_survey, tmp_param)))
                        utils.modify_func(cf_survey, tmp_param, lambda f, par=param_vals[i]: f * (par), do_copy=False)

                if param in self.forecast.png_amp_bias:
                    par1 = param[2:-5]  # separate param: e.g. loc_b_01 -> loc and b_01
                    par2 = param[-4:]
                    for cf_survey in cf_surveys:
                        if param[0] in ["X", "Y"] and ["X", "Y"][cf_survey.t] is not param[0]:
                            continue
                        cf_survey_type = getattr(cf_survey, par1)  # get survey.loc etc
                        restore.append((cf_survey_type, par2, getattr(cf_survey_type, par2)))
                        utils.modify_func(cf_survey_type, par2, lambda f, par=param_vals[i]: f * (par), do_copy=False)
            yield
        finally:
            for obj, attr, func in reversed(restore):
                setattr(obj, attr, func)
                if hasattr(obj, "reset_cache"):
                    obj.reset_cache()
            if restore:  # clear any survey-level cache built from the scaled bias
                for cf_survey in cf_surveys:
                    if hasattr(cf_survey, "reset_cache"):
                        cf_survey.reset_cache()

    @contextmanager
    def _per_bin_bias(self, cosmo_funcs, param_vals, bin_idx):
        """Temporarily scale this bin's per-bin nuisance amplitudes (e.g. b_1) on the survey bias.

        Scales survey.b_1 by the sampled 'b_1_{bin_idx}' in place, yields, then restores the
        original bias functions - so each redshift bin gets its own independently-marginalised
        b_1 without deep-copying cosmo_funcs (which can't be copied: it holds a CLASS object).
        Tracer-prefixed params ('Xb_1', 'YQ') scale only that tracer's survey; the multi-tracer
        cf_list copies share these survey objects so the edit reaches every XX/XY/YY combination.
        Mirrors the global amplitude-bias path but applied per bin and reverted afterwards.
        """
        if not self.per_bin_params:
            yield
            return

        surveys = list(set(cosmo_funcs.survey))
        restore = []  # (survey, attr, original_func), undone last-applied-first
        try:
            for bin_param in self.per_bin_params:
                base, tracers = self._split_tracer(bin_param)
                amp = param_vals[self.param_list.index(f"{bin_param}_{bin_idx}")]
                for s in surveys:
                    if ["X", "Y"][s.t] not in tracers:
                        continue  # tracer-specific bias: skip the other tracer
                    restore.append((s, base, getattr(s, base)))
                    utils.modify_func(s, base, lambda f, par=amp: f * par, do_copy=False)
            yield
        finally:
            for s, attr, func in reversed(restore):
                setattr(s, attr, func)
            for s in surveys:
                if hasattr(s, "reset_cache"):
                    s.reset_cache()

    def get_pk_d1(self, index, term, ln, cf_list, cosmo_funcs, with_kernels=True, **kwargs):
        """Helper function to get power spectrum data vector in right form.

        All multipoles are computed in one pk_func call per tracer so the numeric-mu
        kernels reuse P(k,mu) across l (mirrors PkForecast.get_data_vector)."""
        kernels = self.kernels if with_kernels else None  # numeric-mu kernels summed onto analytic `term`

        cache = {}  # per tracer combination: all multipoles at once

        def data(cf):
            return pk.pk_func(
                term, list(ln), cf, *self.pk_fc[index].args[1:], kernels=kernels, mu_grid=self.mu_grid, **kwargs
            )

        d1 = []
        for i, l in enumerate(ln):
            cfs = [cosmo_funcs] if l & 1 else cf_list  # odd multipoles only ever care about XY
            for cf in cfs:
                if id(cf) not in cache:
                    cache[id(cf)] = data(cf)
                d1.append(cache[id(cf)][i])
        return np.array(d1)

    def get_bk_d1(self, index, term, ln, cf_list, **kwargs):
        """Helper function to get bispectrum data vector in right form.

        l-major ordering (l outer, tracer inner) to match BkForecast.get_data_vector
        and the covariance built in FullCovBk.get_multi_tracer."""
        return np.array([bk.bk_func(term, l, cf, *self.bk_fc[index].args[1:], **kwargs) for l in ln for cf in cf_list])

    def get_likelihood(self, **kwargs):
        # cobaya passes the parameters by name (as keyword arguments)
        param_vals = list(kwargs.values())

        # incomplete theory
        theory = self.get_theory(param_vals)

        chi2 = 0
        for bin_idx in range(len(self.forecast.z_bins)):  # so loop over redshift bins...
            if self.pkln:  # for power spectrum
                d1 = self.data[0][bin_idx]["pk"] - theory[bin_idx]["pk"]
                InvCov = self.inv_covs[bin_idx]["pk"]

                chi2 += np.sum(np.einsum("ik,ijk,jk->k", np.conjugate(d1), InvCov, d1)).real

            if self.bkln:  # for bispectrum
                d1 = self.data[0][bin_idx]["bk"] - theory[bin_idx]["bk"]
                InvCov = self.inv_covs[bin_idx]["bk"]

                chi2 += np.sum(np.einsum("ik,ijk,jk->k", np.conjugate(d1), InvCov, d1)).real

        return -(1 / 2) * chi2

    def run(self, skip_samples=0.3, output=None, resume=True):
        """Run cobaya sampler - save mcmc and samples_df.

        With output set cobaya writes resumable chain files so an
        interrupted run can be continued by re-running with resume=True.
        """
        if output is not None:  # enable cobaya's on-disk (resumable) chain output
            output = self.name if output is True else output
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)  # cobaya won't create parent dirs
            self.info["output"] = output
            self.info["resume"] = resume
        self._tune_drag_factor()
        self.updated_info, self.mcmc = run(self.info)
        self.samples_df = self.mcmc.samples(skip_samples=skip_samples).data[self.param_list]

    def _tune_drag_factor(self):
        """
        We need to get times for slow and fast block manually,
        Set the fast-block oversampling factor from the measured cache speed-up.

        Times a cosmology-changing (slow, cache-miss) vs other (fast, cache-hit)
        likelihood call; the cost ratio is how many fast sub-steps fit in one slow step.
        Left at 1 when the cache gives < 2x, so cobaya then disables dragging by itself.
        """
        blocking = self.info["sampler"]["mcmc"].get("blocking")
        if not blocking:
            return

        base = [self.fiducial.get(p, 0.0) for p in self.param_list]
        call = lambda v: self.get_likelihood(**dict(zip(self.param_list, v)))

        def time_eval(idx, perturb):
            best = np.inf
            for n in range(2):
                v = list(base)
                v[idx] = perturb(n)
                t0 = time.perf_counter()
                call(v)
                best = min(best, time.perf_counter() - t0)
            return best

        i_slow = self.param_list.index(blocking[0][1][0])  # cosmology param
        i_fast = self.param_list.index(blocking[1][1][0])  # other param

        call(base)  # warm the cache at the fiducial cosmology
        t_fast = time_eval(i_fast, lambda n: base[i_fast] + 0.05 + 0.01 * n)  # cache hit
        t_slow = time_eval(i_slow, lambda n: base[i_slow] * (1.001 + 0.001 * n))  # cache miss

        ratio = t_slow / t_fast if t_fast > 0 else 1.0
        factor = 1 if ratio < 2 else min(int(round(ratio)), 30)
        blocking[1][0] = factor
        logger.info("drag: slow/fast cost ratio %.1f -> fast oversampling x%d", ratio, factor)

    def update_df(self, skip_samples=0.3):
        """basically just change skipsamples - as we now work with samples_df more!"""
        self.samples_df = self.mcmc.samples(skip_samples=skip_samples).data

    def add_chain(self, c=None, name=None, bins=12, param_list=None, **kwargs):
        """
        Add MCMC sample as a chain to a ChainConsumer object.

        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            name - name of chain
            bins - Number of bins to plot posterior - more bins higher resolution
            skip_samples - skip burn in of chains
            param_list -  plot subset of parameters

        Returns:
            ChainConsumer: ChainConsumer object with MCMC sample added as a chain.
        """
        c, name = self._name_chain(c, name)

        if not param_list:
            param_list = self.param_list

        # raise error if we do not have computed/loaded dataframe
        if not hasattr(self, "samples_df"):
            raise ValueError("Run/load a sample first!")

        # default bar_shade (1D fill) to match shade (2D fill) so a single flag controls both
        if "shade" in kwargs:
            kwargs.setdefault("bar_shade", kwargs["shade"])

        c.add_chain(Chain(samples=self.samples_df[param_list], name=name, **kwargs))
        c.set_override(ChainConfig(bins=bins))

        return c

    def get_summary(self, param, ci=0.68):
        """Get Median and n-sigma errors"""

        sample = self.samples_df[param]

        # Determine the lower and upper quantiles from the confidence interval
        lower_quantile = (1.0 - ci) / 2.0  # For ci=0.68, this is 0.16
        upper_quantile = 1.0 - lower_quantile  # For ci=0.68, this is 0.84

        # Calculate the median and the bounds of the interval
        median = sample.quantile(0.50)
        lower_bound = sample.quantile(lower_quantile)
        upper_bound = sample.quantile(upper_quantile)

        # Calculate the positive and negative errors
        positive_error = upper_bound - median
        negative_error = median - lower_bound

        return median, positive_error, negative_error

    def summary(self, skip_samples=0.3, ci=0.68):
        """Summarize chain"""

        print("---------------------------------------------------------")
        # 3. Iterate through the list of parameter names
        for param in self.param_list:
            if param in self.samples_df.columns:
                median, positive_error, negative_error = self.get_summary(param, ci=ci)

                if False:  # could use matplotlib to render strings...
                    print(rf"{self.latex[param]}: ${median:.2f}^{{+{positive_error:.2f}}}_{{-{negative_error:.2f}}}$")
                else:
                    print(f"{param}: {median:.2f} (+{positive_error:.2f} / -{negative_error:.2f})")

    def plot_1D(
        self,
        param,
        ci=0.68,
        ax=None,
        shade=True,
        color="royalblue",
        normalise_height=False,
        figsize=(8, 5),
        shade_alpha=0.2,
        **kwargs,
    ):
        """1D PDF plots for a given param from the mcmc samples"""
        if not ax:
            ax = self._setup_1Dplot(param, figsize=figsize, fontsize=22)

        # get sample for given param
        sample_data = self.samples_df[param]
        # get approximate pdf function
        kde = stats.gaussian_kde(sample_data)

        x_eval = np.linspace(sample_data.min() - 1, sample_data.max() + 1, 500)
        pdf_values = kde(x_eval)  # get y_vals

        if normalise_height:  # normalise height of peak to 1
            peak = np.max(pdf_values)
            boost = 1 / peak
        else:
            boost = 1

        if shade:
            # get median and 1-sigma errors
            m, uq, lq = self.get_summary(param, ci=ci)

            # shade 1 sigma region
            x_fill = np.linspace(m - lq, m + uq, 100)
            y_fill = kde(x_fill)
            ax.fill_between(x_fill, boost * y_fill, color=color, alpha=shade_alpha)

        # Plot the main KDE curve
        ax.plot(x_eval, boost * pdf_values, color=color, **kwargs)

        return ax

    def save(self, filepath):
        """
        Saves the Sampler's state to a file using pickle.

        This method serializes the important attributes of the sampler, including
        the MCMC results, parameters, and configuration. It explicitly excludes
        the 'forecast' object and 'info' dictionary (which contains a non-picklable
        external function) to ensure compatibility.

        Args:
            filepath (str): The path to the file where the sampler state will be saved.
        """

        # The 'info' dict contains a reference to the 'get_likelihood' method,
        # which can't be pickled. We can reconstruct it during loading.
        attributes_to_save = {
            "param_list": self.param_list,
            "global_param_list": self.global_param_list,
            "per_bin_params": self.per_bin_params,
            "terms": self.terms,
            "bk_terms": self.bk_terms,
            "kernels": self.kernels,
            "mu_grid": self.mu_grid,
            "pkln": self.pkln,
            "bkln": self.bkln,
            "all_tracer": self.all_tracer,
            "bk_st": self.bk_st,
            "cov_terms": self.cov_terms,
            "data": self.data,
            "prior_dict": self.prior_dict,
            "samples_df": self.samples_df,
            "name": self.name,
            "R_stop": self.info["sampler"]["mcmc"]["Rminus1_stop"],  # Save necessary info values
            "max_tries": self.info["sampler"]["mcmc"]["max_tries"],
        }

        with open(filepath, "wb") as f:
            pickle.dump(attributes_to_save, f)
        logger.info("Sampler state saved to %s", filepath)

    @staticmethod
    def _load_pickle_compat(f):
        """Unpickle, tolerating NumPy 2.x pickles under NumPy 1.x.

        NumPy 2.0 renamed the internal ``numpy.core`` package to
        ``numpy._core``. Arrays pickled with NumPy >= 2.0 therefore reference
        ``numpy._core`` submodules that don't exist under NumPy 1.x, raising
        ``ModuleNotFoundError: No module named 'numpy._core'``. This unpickler
        redirects those references back to ``numpy.core`` when the new module
        is unavailable, leaving behaviour unchanged under NumPy >= 2.0.
        """

        class _NumpyCompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    try:
                        return super().find_class(module, name)
                    except ModuleNotFoundError:
                        module = module.replace("numpy._core", "numpy.core", 1)
                return super().find_class(module, name)

        return _NumpyCompatUnpickler(f).load()

    @classmethod
    def load(cls, filepath, forecast):
        """
        Loads a Sampler's state from a file.

        This class method reconstructs a Sampler instance from a saved file.
        Because the 'forecast' object is not saved, it must be provided
        manually upon loading.

        Args:
            filepath (str): The path to the file containing the saved sampler state.
            forecast (object): The forecast object required to initialize the sampler.
                               This must be the same type of object used originally.

        Returns:
            Sampler: A reconstructed instance of the Sampler class.
        """
        with open(filepath, "rb") as f:
            saved_attrs = cls._load_pickle_compat(f)

        # Create a new instance of the class
        # The __init__ will run, but we will overwrite its products with our saved data.
        new_sampler = cls(
            forecast=forecast,
            # reconstruct from the global params; per_bin_params re-expands the rest
            param_list=saved_attrs.get("global_param_list", saved_attrs["param_list"]),
            terms=saved_attrs["terms"],
            bk_terms=saved_attrs.get("bk_terms"),
            kernels=saved_attrs.get("kernels"),
            mu_grid=saved_attrs.get("mu_grid"),
            pkln=saved_attrs["pkln"],
            bkln=saved_attrs["bkln"],
            all_tracer=saved_attrs.get("all_tracer", False),
            bk_st=saved_attrs.get("bk_st", False),
            cov_terms=saved_attrs.get("cov_terms"),
            R_stop=saved_attrs["R_stop"],
            max_tries=saved_attrs["max_tries"],
            name=saved_attrs["name"],
            per_bin_params=saved_attrs.get("per_bin_params"),
            fisher_covmat=False,  # no proposal needed when just loading saved chains
        )

        # Overwrite the attributes with the loaded state
        # This is more robust than trying to prevent __init__ from running.
        for key, value in saved_attrs.items():
            setattr(new_sampler, key, value)

        # for backwards compatability with saved chains
        if hasattr(new_sampler, "dataframe"):
            new_sampler.samples_df = new_sampler.dataframe

        logger.info("Sampler state loaded from %s", filepath)
        return new_sampler
