"""
MCMC Sampler using Cobaya for cosmological parameter sampling.

Uses Cobaya MCMC sampler to sample for a given likelihoods.
Allows us to drop the assumption of gaussianity of the posterior we have in the fisher.
Heavily reliant on CosmoPower to make sampling over cosmological parameters efficient.
"""

import logging
import pickle
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

from .base_posterior import BasePosterior


class Sampler(BasePosterior):
    """MCMC Sampler with cobaya with ChainConsumer plots.
    Assumes gaussian likelihood with parameter independent covariances."""

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
        bk_terms=None,
        bk_st=False,
        per_bin_params=None,
        fisher_covmat=True,
        **kwargs,
    ):
        super().__init__(forecast, param_list, name=name)

        self.pkln = pkln
        self.bkln = bkln
        # terms which to compute that are parameter dependent
        self.terms = terms
        if bk_terms is None:
            bk_terms = terms
        self.bk_terms = bk_terms
        # use planck covariance as prior
        self.planck_prior = planck_prior
        # full mutli-tracer analysis
        self.all_tracer = all_tracer
        # bk_st: force bispectrum onto single-tracer pipeline using survey[0] (no-op when not multi-tracer)
        self.bk_st = bk_st
        # config needed to rebuild a matching Fisher for the proposal covmat
        self.cov_terms = cov_terms
        self.fisher_covmat = fisher_covmat

        # per-bin nuisance params: one amplitude on b_1 (etc.) per redshift bin, marginalised in the MCMC
        if per_bin_params is None:
            per_bin_params = []
        elif not isinstance(per_bin_params, list):
            per_bin_params = [per_bin_params]
        if per_bin_params and all_tracer:
            raise NotImplementedError("per_bin_params is not yet supported with all_tracer=True")
        self.per_bin_params = self.forecast._rename_composite_params(per_bin_params)
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
            term for term in terms + param_list + bias_list if term in self.cosmo_funcs.term_list
        ]  # get list of needed terms to compute full 'true' theory
        all_bk_terms = [
            term for term in self.bk_terms + param_list + bk_bias_list if term in self.cosmo_funcs.term_list
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
            lo, hi, prop = per_bin_bounds.get(p, (-50, 50, None))
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
        self.info = {
            "likelihood": {"cosmowap_likelihood": {"external": self.get_likelihood, "input_params": param_list}},
            "params": {key: self.prior_dict[key] for key in param_list},
            "sampler": {"mcmc": mcmc},
        }
        if self.planck_prior:
            self.info = self.set_planck_prior(self.info)

    def get_fisher_covmat(self):
        """Inverse-Fisher over the global params -> cobaya initial proposal covmat.

        Gives the MCMC the correct (tight, degenerate) correlation structure from the
        start instead of guessing a diagonal proposal, computed for the same data/cov
        config as the likelihood. Per-bin nuisance params are deliberately left out -
        cobaya fills them from their proposal widths (a partial covmat is fine).
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
                verbose=False,
            )
        except Exception as exc:  # best-effort: a covmat failure must not kill the run
            logger.warning("Fisher proposal covmat failed (%s); falling back to proposal widths.", exc)
            return None, None

        covmat = np.asarray(fish.covariance)
        if not np.all(np.isfinite(covmat)):
            logger.warning("Fisher covariance is non-finite (singular?); falling back to proposal widths.")
            return None, None
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

        def planck_prior(**kwargs):
            # cobaya passes the parameters by name (as keyword arguments)
            param_vals = list(kwargs.values())

            # find what parameters in this prior we are sampling over! (ln_A_s and A_s are
            # mutually exclusive; planck_cov is built in the matching amplitude's units)
            amp = "ln_A_s" if "ln_A_s" in self.param_list else "A_s"
            params = ["Omega_b", "Omega_cdm", "theta", "tau", amp, "n_s"]
            selected_params = []
            means = []
            values = []
            for i, param in enumerate(self.param_list):
                for prior_param in params:
                    if param == prior_param:
                        selected_params.append(param)  # get the cosmology params
                        means.append(getattr(self.cosmo_funcs, param))  # get fiducial
                        values.append(param_vals[i])

            data_vector = np.array(values) - np.array(means)  # so N array

            return -(1 / 2) * np.sum(data_vector[:, np.newaxis] * inv_cov * data_vector[np.newaxis, :])

        # info['params'] = {'planck': planck_prior} # add prior to conbaya setup
        info["likelihood"]["prior"] = {"external": planck_prior, "input_params": self.param_list}
        return info

    def update_cosmo_funcs(self, param_vals):
        """Update the cosmology for each sample"""
        cosmo_kwargs = {}
        for i, param in enumerate(self.param_list):
            if param in ["Omega_m", "Omega_cdm", "Omega_b", "A_s", "ln_A_s", "sigma8", "n_s", "h"]:
                cosmo_kwargs[param] = param_vals[i]

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
        if "gamma" in self.param_list:
            i_g = self.param_list.index("gamma")
            zz = np.linspace(0, cosmo_funcs.z_max, 100)
            cosmo_funcs.f = CubicSpline(zz, cosmo_funcs.Om_m(zz) ** param_vals[i_g])
            cosmo_funcs.compute_derivs_cosmo()

        return cosmo_funcs

    def get_theory(self, param_vals):
        """
        Get data vector for given MCMC call - data vector is shape [z_bin]['pk'][k_bin]
        """
        cosmo_funcs = self.update_cosmo_funcs(param_vals)  # the cosmology part

        # lets make this multi-tracer
        cf_surveys = list(set(cosmo_funcs.survey))  # get unique tracers

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

            # so now only for each survey...
            if param in self.forecast.amp_bias:
                tmp_param = param[2:]  # i.e get b_1 from X_b_1
                # now lets also be able to marginalise over the amplitude parameters
                for cf_survey in cf_surveys:
                    if param[0] in ["X", "Y"]:  # if tracer specific bias
                        if ["X", "Y"][cf_survey.t] is param[0]:
                            cf_survey = utils.modify_func(
                                cf_survey, tmp_param, lambda f, par=param_vals[i]: f * (par), do_copy=False
                            )  # default argument solves late binding
                    else:  # then edit all surveys
                        cf_survey = utils.modify_func(
                            cf_survey, tmp_param, lambda f, par=param_vals[i]: f * (par), do_copy=False
                        )

            if param in self.forecast.png_amp_bias:
                par1 = param[2:-5]  # separate param: e.g. loc_b_01 -> loc and b_01
                par2 = param[-4:]
                # now lets also be able to marginalise over the amplitude parameters
                for cf_survey in cf_surveys:
                    cf_survey_type = getattr(cf_survey, par1)  # get survey.loc etc
                    if param[0] in ["X", "Y"]:  # if tracer specific bias
                        if ["X", "Y"][cf_survey.t] is param[0]:
                            cf_survey_type = utils.modify_func(
                                cf_survey_type, par2, lambda f, par=param_vals[i]: f * (par), do_copy=False
                            )
                    else:  # then edit all surveys
                        cf_survey_type = utils.modify_func(
                            cf_survey_type, par2, lambda f, par=param_vals[i]: f * (par), do_copy=False
                        )  # default argument solves late binding

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
                        d_v[j]["pk"] += (param_vals[i]) * self.get_pk_d1(
                            i, param, self.pkln, cf_list, cosmo_funcs, **kwargs
                        )
                    if self.bkln:
                        d_v[j]["bk"] += (param_vals[i]) * self.get_bk_d1(i, param, self.bkln, cf_list_bk, **kwargs)

        # free this step's CLASS C-memory (cosmo_funcs is in a reference cycle so isn't reclaimed
        # promptly) - else per-sample Class() structures pile up. Skip the shared master cosmology.
        if cosmo_funcs.cosmo is not self.cosmo_funcs.cosmo:
            cosmo_funcs.cosmo.struct_cleanup()
            cosmo_funcs.cosmo.empty()

        return d_v

    @contextmanager
    def _per_bin_bias(self, cosmo_funcs, param_vals, bin_idx):
        """Temporarily scale this bin's per-bin nuisance amplitudes (e.g. b_1) on the survey bias.

        Scales survey.b_1 by the sampled 'b_1_{bin_idx}' in place, yields, then restores the
        original bias functions - so each redshift bin gets its own independently-marginalised
        b_1 without deep-copying cosmo_funcs (which can't be copied: it holds a CLASS object).
        Mirrors the global amplitude-bias path but applied per bin and reverted afterwards.
        """
        if not self.per_bin_params:
            yield
            return

        surveys = list(set(cosmo_funcs.survey))
        originals = [{p: getattr(s, p) for p in self.per_bin_params} for s in surveys]
        try:
            for s in surveys:
                for bin_param in self.per_bin_params:
                    amp = param_vals[self.param_list.index(f"{bin_param}_{bin_idx}")]
                    utils.modify_func(s, bin_param, lambda f, par=amp: f * par, do_copy=False)
            yield
        finally:
            for s, orig in zip(surveys, originals):
                for p, func in orig.items():
                    setattr(s, p, func)
                if hasattr(s, "reset_cache"):
                    s.reset_cache()

    def get_pk_d1(self, index, term, ln, cf_list, cosmo_funcs, **kwargs):
        """Helper function to get power spectrum data vector in right form"""
        d1 = []
        for l in ln:
            if l & 1:
                cfs = [cosmo_funcs]  # odd multipoles only ever care about XY
            else:
                cfs = cf_list

            d1 += [pk.pk_func(term, l, cf, *self.pk_fc[index].args[1:], **kwargs) for cf in cfs]
        return np.array(d1)

    def get_bk_d1(self, index, term, ln, cf_list, **kwargs):
        """Helper function to get bispectrum data vector in right form"""
        return np.array([bk.bk_func(term, l, cf, *self.bk_fc[index].args[1:], **kwargs) for cf in cf_list for l in ln])

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

    def run(self, skip_samples=0.3):
        """Run cobaya sampler - save mcmc and samples_df"""
        self.updated_info, self.mcmc = run(self.info)
        self.samples_df = self.mcmc.samples(skip_samples=skip_samples).data[self.param_list]

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
            "pkln": self.pkln,
            "bkln": self.bkln,
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
            saved_attrs = pickle.load(f)

        # Create a new instance of the class
        # The __init__ will run, but we will overwrite its products with our saved data.
        new_sampler = cls(
            forecast=forecast,
            # reconstruct from the global params; per_bin_params re-expands the rest
            param_list=saved_attrs.get("global_param_list", saved_attrs["param_list"]),
            terms=saved_attrs["terms"],
            bk_terms=saved_attrs.get("bk_terms"),
            pkln=saved_attrs["pkln"],
            bkln=saved_attrs["bkln"],
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
