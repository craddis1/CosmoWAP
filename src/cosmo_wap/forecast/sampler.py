"""
MCMC Sampler using Cobaya for cosmological parameter sampling.

Uses Cobaya MCMC sampler to sample for a given likelihoods.
Allows us to drop the assumption of gaussianity of the posterior we have in the fisher.
Heavily reliant on CosmoPower to make sampling over cosmological parameters efficient.
"""
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import stats

import cosmo_wap.bk as bk
import cosmo_wap.pk as pk
import cosmo_wap as cw
from cosmo_wap.lib import utils
from chainconsumer import Chain, ChainConfig
from cobaya import run

from .base_posterior import BasePosterior


class Sampler(BasePosterior):
    """MCMC Sampler with cobaya with ChainConsumer plots.
    Assumes gaussian likelihood with parameter independent covariances."""
    def __init__(self, forecast, param_list,terms=None,cov_terms=None,all_tracer=False,bias_list=None,pkln=None,bkln=None,R_stop=0.005,max_tries=200,name=None,planck_prior=False, **kwargs):
        super().__init__(forecast, param_list, name=name)

        self.pkln = pkln
        self.bkln = bkln
        # terms which to compute that are parameter dependent
        self.terms = terms
        # use planck covariance as prior
        self.planck_prior = planck_prior
        # full mutli-tracer analysis
        self.all_tracer = all_tracer

        if 'fNL' in kwargs.keys():
            self.fNL = kwargs['fNL']
        else:
            self.fNL = 0

        if bias_list is None:
            bias_list = []

        self.pk_fc = []
        self.bk_fc = []
        for i in range(forecast.N_bins):
            self.pk_fc.append(forecast.get_pk_bin(i,all_tracer=all_tracer,cov_terms=cov_terms))
            self.bk_fc.append(forecast.get_bk_bin(i,all_tracer=all_tracer,cov_terms=cov_terms))

        all_terms = [term for term in terms+param_list+bias_list if term in self.cosmo_funcs.term_list] # get list of needed terms to compute full 'true' theory
        # so this just gets total contribution - i.e. true theory - and also parameter independent covariance
        self.data, self.inv_covs = forecast._precompute_derivatives_and_covariances([all_terms],pkln=pkln,bkln=bkln,verbose=False,all_tracer=all_tracer,cov_terms=cov_terms,fNL=0)

        # set up cobaya sampler - define priors, starting value and initial step
        # Cosmological Priors
        cosmo_params = {
            "n_s":       self.get_prior(0.84, 1.1, 0.9665, 5e-5),
            "h":         self.get_prior(0.64, 0.82, 0.6776, 1e-3),
            "A_s":       self.get_prior(6e-10, 4.8e-9, 2.105e-9, 1e-11),
            "Omega_m":   self.get_prior(0.17, 0.42, 0.31, 5e-5),
            "Omega_cdm": self.get_prior(0.13, 0.38, 0.26, 5e-5),
            "Omega_b":   self.get_prior(0.041, 0.057, 0.049, 1e-5),
        }

        # fNL Parameters (Wide priors)
        fnl_params = {k: self.get_prior(-100,100, ref=0,proposal=0.01)
                    for k in ["fNL", "fNL_eq", "fNL_orth"]}

        # Theory Amplitude Parameters
        theory_params = {k: self.get_prior(-100,100)
                        for k in ["GR2", "WS2", "WA2"]}

        # b1 amplitude Parameters (Narrow priors around 1.0)
        b1_prior = {k: self.get_prior(0.8, 1.2, 1.0, 1e-2)
                for k in ["A_b_1", "X_b_1", "Y_b_1"]}

        # Luminosity priors - Q and be and b_2
        lum_prior = {k: self.get_prior(-50, 50)
                for k in forecast.amp_bias[3:]}

        # PNG amplitude Parameters
        pngbias_prior = {k: self.get_prior(-100, 100)
                    for k in forecast.png_amp_bias}

        # Combine everything
        self.prior_dict = {**cosmo_params, **fnl_params, **theory_params, **b1_prior,**lum_prior, **pngbias_prior}

        self.set_info(param_list,R_stop,max_tries)

    def get_prior(self, min_val, max_val, ref=1, proposal=None):
        """Helper to standardize dictionary creation."""
        return {
            "prior": {"min": min_val, "max": max_val},
            "ref": ref,
            "proposal": proposal or (max_val - min_val) / 100  # Default proposal if not provided
        }

    def set_info(self,param_list,R_stop,max_tries):
        """Sets cobaya info for given parameters"""
        self.info = {
            "likelihood": {
                "cosmowap_likelihood": {
                    "external": self.get_likelihood,
                    "input_params": param_list
                }
            },

            "params": {key: self.prior_dict[key] for key in param_list},

            "sampler": {"mcmc": {"Rminus1_stop": R_stop, "max_tries": max_tries}}
        }
        if self.planck_prior:
            self.info = self.set_planck_prior(self.info)

    def set_planck_prior(self,info):
        """Use planck constraints to set priors.
        So we need to define function to describe the planck prior likelihood:
        log(L)=-(1/2)*(p-mu)*c^{-1}(p-mu)^T"""

        cov = self.planck_cov() # get NxN parameter covariance
        inv_cov = np.linalg.inv(cov) # NxN

        def planck_prior(**kwargs):
            # cobaya passes the parameters by name (as keyword arguments)
            param_vals = list(kwargs.values())

            #find what parameters in this prior we are sampling over!
            params = ['Omega_b','Omega_cdm','theta','tau','A_s','n_s']
            selected_params = []
            means = []
            values = []
            for i,param in enumerate(self.param_list):
                for prior_param in params:
                    if param==prior_param:
                        selected_params.append(param) # get the cosmology params
                        means.append(getattr(self.cosmo_funcs,param)) # get fiducial
                        values.append(param_vals[i])

            data_vector = np.array(values)-np.array(means) # so N array

            return -(1/2)*np.sum(data_vector[:,np.newaxis] *inv_cov *data_vector[np.newaxis,:])

        #info['params'] = {'planck': planck_prior} # add prior to conbaya setup
        info['likelihood']['prior'] = {
                    "external": planck_prior,
                    "input_params": self.param_list
                }
        return info

    def update_cosmo_funcs(self,param_vals):
        """Update the cosmology for each sample"""
        cosmo_kwargs = {}
        for i, param in enumerate(self.param_list):
            if param in ['Omega_m','Omega_cdm','Omega_b','A_s','sigma8','n_s','h']:
                cosmo_kwargs[param] = param_vals[i]

        # change survey params
        if cosmo_kwargs:
            if self.cosmo_funcs.emulator: # much quicker!
                cosmo_kwargs['emulator'] = True
                cosmo,params = utils.get_cosmo(**cosmo_kwargs,k_max=self.cosmo_funcs.K_MAX*self.cosmo_funcs.h) # update cosmology for change in param
                other_kwarg = {'emulator':self.cosmo_funcs.emu,'params':params}

            else:
                cosmo = utils.get_cosmo(**cosmo_kwargs,k_max=self.cosmo_funcs.K_MAX*self.cosmo_funcs.h)
                other_kwarg = {}

            cosmo_funcs = cw.ClassWAP(cosmo,self.cosmo_funcs.survey_params,compute_bias=self.cosmo_funcs.compute_bias,fast=True,**other_kwarg)
        else:
            cosmo_funcs = utils.create_copy(self.cosmo_funcs)

        return cosmo_funcs

    def get_theory(self,param_vals):
        """
        Get data vector for given MCMC call - data vector is shape [z_bin]['pk'][k_bin]
        """
        cosmo_funcs = self.update_cosmo_funcs(param_vals) # the cosmology part

        # lets make this multi-tracer
        cf_surveys = list(set(cosmo_funcs.survey)) # get unique tracers

        kwargs = {} # create dict which is fed into function
        kwargs['fNL'] = self.fNL  # useful to set default to 0 - otherwise without fNL as parameter default would be 1
        for i, param in enumerate(self.param_list):
            if param in ['fNL','fNL_loc','fNL_eq','fNL_orth','t','r','s']: # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
                kwargs[param] = param_vals[i]

            # so now only for each survey...
            if param in self.forecast.amp_bias:
                tmp_param = param[2:] # i.e get b_1 from X_b_1
                # now lets also be able to marginalise over the amplitude parameters
                for cf_survey in cf_surveys:
                    if param[0] in ['X','Y']: # if tracer specific bias
                        if ['X','Y'][cf_survey.t] is param[0]:
                            cf_survey = utils.modify_func(cf_survey, tmp_param, lambda f,par=param_vals[i]: f*(par),copy=False) # default argument solves late binding
                    else: # then edit all surveys
                        cf_survey = utils.modify_func(cf_survey, tmp_param, lambda f,par=param_vals[i]: f*(par),copy=False)

                    if tmp_param in ['be','Q']: # reset betas - as they need to be recomputed with the new biases
                        cf_survey.betas = None

            if param in self.forecast.png_amp_bias:
                par1 = param[2:-5];     # separate param: e.g. loc_b_01 -> loc and b_01
                par2 =  param[-4:]
                # now lets also be able to marginalise over the amplitude parameters
                for cf_survey in cf_surveys:
                    cf_survey_type = getattr(cf_survey,par1) # get survey.loc etc
                    if param[0] in ['X','Y']: # if tracer specific bias
                        if ['X','Y'][cf_survey.t] is param[0]:
                            cf_survey_type = utils.modify_func(cf_survey_type, par2, lambda f,par=param_vals[i]: f*(par),copy=False)
                    else: # then edit all surveys
                        cf_survey_type = utils.modify_func(cf_survey_type, par2, lambda f,par=param_vals[i]: f*(par),copy=False) # default argument solves late binding

        # setup multiracer permutations - get cf_list
        if self.all_tracer:
            cf_mat = self.forecast.setup_multitracer(cosmo_funcs)
            cf_mat_bk = self.forecast.setup_multitracer_bk(cosmo_funcs)
            cf_list = [cf_mat[0][0],cf_mat[0][1],cf_mat[1][1]]
            cf_list_bk = [cf_mat_bk[0][0][0],cf_mat_bk[0][0][1],cf_mat_bk[0][1][1],cf_mat_bk[1][1][1]]
        else:
            cf_list = [cosmo_funcs]
            cf_list_bk = [cosmo_funcs]

        # Caching structures
        # derivs[bin_idx] = {'pk': pk_deriv, 'bk': bk_deriv}
        d_v = [{} for _ in range(self.forecast.N_bins)]

        # now change this for full multi-tracer lengths with odd pk_l
        for i in range(self.forecast.N_bins):
            # get powerspectrum data vector
            if self.pkln:
                d_v[i]['pk'] = self.get_pk_d1(i,self.terms,self.pkln,cf_list,cosmo_funcs,**kwargs)
            if self.bkln: # get bispectrum data vector
                d_v[i]['bk'] = self.get_bk_d1(i,self.terms,self.bkln,cf_list_bk,**kwargs)

        # ok a little weird but may be useful later i guess - allows sample of term like alpha_GR
        for i, param in enumerate(self.param_list):
            if param in self.cosmo_funcs.term_list:
                for j in range(self.forecast.N_bins):
                    if self.pkln:
                        d_v[j]['pk'] += (param_vals[i])*self.get_pk_d1(i,param,self.pkln,cf_list,cosmo_funcs,**kwargs)
                    if self.bkln:
                        d_v[j]['bk'] += (param_vals[i])*self.get_bk_d1(i,param,self.bkln,cf_list_bk,**kwargs)

        return d_v

    def get_pk_d1(self,index,term,ln,cf_list,cosmo_funcs,**kwargs):
        """Helper function to get power spectrum data vector in right form"""
        d1 = []
        for l in ln:
            if l & 1:
                cfs = [cosmo_funcs] # odd multipoles only ever care about XY
            else:
                cfs = cf_list

            d1 += [pk.pk_func(term,l,cf,*self.pk_fc[index].args[1:],**kwargs) for cf in cfs]
        return np.array(d1)

    def get_bk_d1(self,index,term,ln,cf_list,**kwargs):
        """Helper function to get bispectrum data vector in right form"""
        return np.array([bk.bk_func(term,l,cf,*self.bk_fc[index].args[1:],**kwargs) for cf in cf_list for l in ln])

    def get_likelihood(self,**kwargs):

        # cobaya passes the parameters by name (as keyword arguments)
        param_vals = list(kwargs.values())

        # incomplete theory
        theory = self.get_theory(param_vals)

        chi2 = 0
        for bin_idx in range(len(self.forecast.z_bins)): # so loop over redshift bins...
            if self.pkln: # for power spectrum
                d1 = self.data[0][bin_idx]['pk'] - theory[bin_idx]['pk']
                InvCov = self.inv_covs[bin_idx]['pk']

                chi2 += np.sum(np.einsum('ik,ijk,jk->k', np.conjugate(d1), InvCov, d1)).real

            if self.bkln: # for bispectrum
                d1 = self.data[0][bin_idx]['bk'] - theory[bin_idx]['bk']
                InvCov = self.inv_covs[bin_idx]['bk']

                chi2 += np.sum(np.einsum('ik,ijk,jk->k', np.conjugate(d1), InvCov, d1)).real

        return - (1/2)*chi2

    def run(self,skip_samples=0.3):
        """Run cobaya sampler - save mcmc and samples_df"""
        self.updated_info, self.mcmc = run(self.info)
        self.samples_df = self.mcmc.samples(skip_samples=skip_samples).data[self.param_list]

    def update_df(self,skip_samples=0.3):
        """basically just change skipsamples - as we now work with samples_df more!"""
        self.samples_df = self.mcmc.samples(skip_samples=skip_samples).data

    def add_chain(self,c=None,name=None,bins=12,param_list=None,**kwargs):
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
        c,name = self._name_chain(c,name)

        if not param_list:
            param_list = self.param_list

        # raise error if we do not have computed/loaded dataframe
        if not hasattr(self,'samples_df'):
            raise ValueError("Run/load a sample first!")

        c.add_chain(Chain(samples=self.samples_df, name=name,**kwargs))
        c.set_override(ChainConfig(bins=bins))

        return c

    def get_summary(self,param,ci=0.68):
        """Get Median and n-sigma errors"""

        sample = self.samples_df[param]

        # Determine the lower and upper quantiles from the confidence interval
        lower_quantile = (1.0 - ci) / 2.0  # For ci=0.68, this is 0.16
        upper_quantile = 1.0 - lower_quantile # For ci=0.68, this is 0.84

        # Calculate the median and the bounds of the interval
        median = sample.quantile(0.50)
        lower_bound = sample.quantile(lower_quantile)
        upper_bound = sample.quantile(upper_quantile)

        # Calculate the positive and negative errors
        positive_error = upper_bound - median
        negative_error = median - lower_bound

        return median,positive_error,negative_error

    def summary(self,skip_samples=0.3,ci=0.68):
        """Summarize chain"""

        print("---------------------------------------------------------")
        # 3. Iterate through the list of parameter names
        for param in self.param_list:
            if param in self.samples_df.columns:
                median,positive_error,negative_error = self.get_summary(param,ci=ci)

                if False: # could use matplotlib to render strings...
                    print(rf"{self.latex[param]}: ${median:.2f}^{{+{positive_error:.2f}}}_{{-{negative_error:.2f}}}$")
                else:
                    print(f"{param}: {median:.2f} (+{positive_error:.2f} / -{negative_error:.2f})")

    def plot_1D(self,param,ci=0.68,ax=None,shade=True,color='royalblue',normalise_height=False, figsize=(8,5),shade_alpha=0.2,**kwargs):
        """1D PDF plots for a given param from the mcmc samples"""
        if not ax:
            ax = self._setup_1Dplot(param,figsize=figsize,fontsize=22)

        # get sample for given param
        sample_data = self.samples_df[param]
        # get approximate pdf function
        kde = stats.gaussian_kde(sample_data)

        x_eval = np.linspace(sample_data.min() - 1, sample_data.max() + 1, 500)
        pdf_values = kde(x_eval) # get y_vals

        if normalise_height: # normalise height of peak to 1
            peak  = np.max(pdf_values)
            boost = 1/peak
        else:
            boost = 1

        if shade:
            # get median and 1-sigma errors
            m,uq,lq = self.get_summary(param,ci=ci)

            # shade 1 sigma region
            x_fill = np.linspace(m-lq, m+uq, 100)
            y_fill = kde(x_fill)
            ax.fill_between(x_fill, boost*y_fill, color=color, alpha=shade_alpha)

        # Plot the main KDE curve
        ax.plot(x_eval, boost*pdf_values, color=color, **kwargs)

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
            'param_list': self.param_list,
            'terms': self.terms,
            'pkln': self.pkln,
            'bkln': self.bkln,
            'data': self.data,
            'prior_dict': self.prior_dict,
            'samples_df': self.samples_df,
            'name': self.name,
            'R_stop': self.info['sampler']['mcmc']['Rminus1_stop'], # Save necessary info values
            'max_tries': self.info['sampler']['mcmc']['max_tries']
        }

        with open(filepath, 'wb') as f:
            pickle.dump(attributes_to_save, f)
        print(f"Sampler state saved to {filepath}")

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
        with open(filepath, 'rb') as f:
            saved_attrs = pickle.load(f)

        # Create a new instance of the class
        # The __init__ will run, but we will overwrite its products with our saved data.
        new_sampler = cls(
            forecast=forecast,
            param_list=saved_attrs['param_list'],
            terms=saved_attrs['terms'],
            pkln=saved_attrs['pkln'],
            bkln=saved_attrs['bkln'],
            R_stop=saved_attrs['R_stop'],
            max_tries=saved_attrs['max_tries'],
            name=saved_attrs['name']
        )

        # Overwrite the attributes with the loaded state
        # This is more robust than trying to prevent __init__ from running.
        for key, value in saved_attrs.items():
            setattr(new_sampler, key, value)

        # for backwards compatability with saved chains
        if hasattr(new_sampler,'dataframe'):
            new_sampler.samples_df = new_sampler.dataframe

        print(f"Sampler state loaded from {filepath}")
        return new_sampler
