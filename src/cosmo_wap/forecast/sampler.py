"""
Uses Cobaya MCMC sampler to sample for a given likelihoods. 
Allows us to drop the assumption of gaussianity of the posterior we have in the fisher.
Heavily reliant on CosmoPower to make sampling over cosmological parameters efficient.
"""
import numpy as np

import cosmo_wap.bk as bk
import cosmo_wap.pk as pk 
import cosmo_wap as cw 
from cosmo_wap.forecast.core import PkForecast, BkForecast
from cosmo_wap.lib import utils

from cobaya import run

class Sampler:
    """Return efficient datavector for given parameters - useful for MCMC samples - initiate with a FullForecast class"""
    def __init__(self, forecast, param_list, terms=None, bias_terms=None, pkln=None,bkln=None,R_stop=0.005,max_tries=100):
        self.cosmo_funcs = forecast.cosmo_funcs
        self.forecast = forecast
        self.pkln = pkln
        self.bkln = bkln
        
        self.terms = terms
        self.param_list = param_list
        if bias_terms is None:
            bias_terms = []

        # inititalize Pk/BkFroecast classes
        self.pk_args = []
        self.bk_args = []
        for i in range(forecast.num_bins):
            self.pk_args.append(PkForecast(forecast.z_bins[i], self.cosmo_funcs, k_max=forecast.k_max_list[i], s_k=forecast.s_k).args)
            self.bk_args.append(BkForecast(forecast.z_bins[i], self.cosmo_funcs, k_max=forecast.k_max_list[i], s_k=forecast.s_k).args)
        
        all_terms = [term for term in terms+param_list+bias_terms if term in self.cosmo_funcs.term_list] # get list of needed terms to compute full 'true' theory
        # so this just gets total contribution - i.e. true theory - and also parameter independent covariance
        self.data, self.inv_covs = forecast._precompute_derivatives_and_covariances([all_terms],pkln=pkln,bkln=bkln,verbose=False,fNL=0)

        # set up cobaya sampler
        #standard term:
        standard_dict = {"prior": {"min": -20, "max": 30},"ref": 0,"proposal": 2,"latex": "fNL"}
        self.prior_dict = {
                "fNL": standard_dict.update({"latex": "fNL"}),
                "GR2": standard_dict.update({"latex": "GR2"}),
                "WS2": standard_dict.update({"latex": "WS2"}),
                "WA2": standard_dict.update({"latex": "WA2"}),
                "n_s": {
                    "prior": {"min": 0.84, "max": 1.1},"ref": 0.9665,"proposal": 0.01,
                    "latex": "n_s"
                },
                "h": {
                    "prior": {"min": 0.64, "max": 0.82},"ref": 0.6776,"proposal": 0.01,
                    "latex": "n_s"
                },
                "A_s": {
                    "prior": {"min": 6e-10, "max": 4.8e-9},"ref": 2.105e-9,"proposal": 2e-10,
                    "latex": "A_s"
                },
                "Omega_m": {
                    "prior": {"min": 0.17, "max": 0.58},"ref": 2.105e-9,"proposal": 2e-10,
                    "latex": "Omega_m"
                },
                "Omega_b": {
                    "prior": {"min": 0.041, "max": 0.057},"ref": 0.049,"proposal": 0.01,
                    "latex": "Omega_b"
                }
            }
        
        self.set_info(param_list,R_stop,max_tries)


    def set_info(self,param_list,R_stop,max_tries):
        self.info = {
            "likelihood": {
                "my_flexible_likelihood": {
                    "external": self.get_likelihood,
                    "input_params": param_list
                }
            },
            
            "params": {key: self.prior_dict[key] for key in param_list},
            
            "sampler": {"mcmc": {"Rminus1_stop": R_stop, "max_tries": max_tries}}
        }


    def get_theory(self,param_vals):
        """
        Get data vector for given MCMC call - data vector is shape [z_bin]['pk'][k_bin]
        """
        cosmo_kwargs = {}
        for i, param in enumerate(self.param_list):
            if param in ['Omega_m','A_s','sigma8','n_s','h']:
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
            
            cosmo_funcs = cw.ClassWAP(cosmo,self.cosmo_funcs.survey_params,compute_bias=self.cosmo_funcs.compute_bias,**other_kwarg)
        else:
            cosmo_funcs = self.cosmo_funcs

        kwargs = {} # create dict which is fed into function
        for i, param in enumerate(self.param_list):
            if param in ['fNL','t','r','s']: # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
                kwargs[param] = param_vals[i]

        # Caching structures
        # derivs[bin_idx] = {'pk': pk_deriv, 'bk': bk_deriv}
        d_v = [{} for _ in range(self.forecast.num_bins)]

        for i in range(self.forecast.num_bins):
            # get data vector
            if self.pkln:
                d_v[i]['pk'] = np.array([pk.pk_func(self.terms,l,cosmo_funcs,*self.pk_args[i][1:],**kwargs) for l in self.pkln])
            if self.bkln:
                d_v[i]['bk'] = np.array([bk.bk_func(self.terms,l,cosmo_funcs,*self.bk_args[i][1:],**kwargs) for l in self.bkln])

        # ok a little weird but may be useful later i guess - allows sample of term like alpha_GR 
        for i, param in enumerate(self.param_list):
            if param in self.cosmo_funcs.term_list:
                for j in range(self.forecast.num_bins):
                    if self.pkln:
                        d_v[j]['pk'] += (param_vals[i])*np.array([pk.pk_func(param,l,cosmo_funcs,*self.pk_args[j][1:],**kwargs) for l in self.pkln])
                    if self.bkln:
                        d_v[j]['bk'] += (param_vals[i])*np.array([bk.bk_func(param,l,cosmo_funcs,*self.bk_args[j][1:],**kwargs) for l in self.bkln])

        return d_v
    
    def get_likelihood(self,**kwargs):
        # cobaya passes the parameters by name (as keyword arguments)
        param_vals = list(kwargs.values())

        # incomplete theory
        theory = self.get_theory(param_vals)

        chi2 = 0
        for bin_idx in range(len(self.forecast.z_bins)):
            d1 = self.data[0][bin_idx]['pk'] - theory[bin_idx]['pk']
            InvCov = self.inv_covs[bin_idx]['pk']

            chi2 += np.sum(np.einsum('ik,ijk,jk->k', d1, InvCov, d1)).real

        return - (1/2)*chi2
    
    def run(self):
        """Run cobaya smapler"""
        self.updated_info, self.samples = run(self.info)