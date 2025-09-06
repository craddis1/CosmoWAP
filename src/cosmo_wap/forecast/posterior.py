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
from matplotlib import pyplot as plt
import pickle
import warnings

from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig, ChainConfig # plots with chainconsumer - https://samreay.github.io/ChainConsumer/
from cobaya import run

# be proper
#from typing import Any, List
from abc import ABC #,abstractmethod # not implemented

class BasePosterior(ABC):
    """Base class for different meethod of analysing the posterior distributions
    Either for Fishers or MCMC samples.
    Shared functionality of storing parameters and plotting with Chainconsumer."""
    def __init__(self,forecast, param_list, name=None):
        self.forecast = forecast
        self.cosmo_funcs = forecast.cosmo_funcs
        self.param_list = param_list
        self.name = name or "_".join(param_list) # sample name is amalgamation of parameters
        self.handle_latex() # use latex label if latex is available
        self.fiducial = self._get_fiducial()

    def handle_latex(self):    
        try:
            # A lightweight test to see if LaTeX is available
            fig = plt.figure(figsize=(0.1, 0.1))
            plt.text(0, 0, 'test', usetex=True)
            plt.close(fig)
            self.USE_LATEX = True
        except RuntimeError:
            warnings.warn("LaTeX not found. Will someone think of the plots!!")
            self.USE_LATEX = False
        
        if self.USE_LATEX:
            self.latex = {"fNL": r"$f_{\rm NL}$",
                      "n_s": "$n_s$",
                      "A_s": "$A_s$",
                      "h"  : "$h$",
                      "Omega_m": r"$\Omega_m$",
                      "Omega_cdm": r"$\Omega_{cdm}$",
                      "Omega_b": r"$\Omega_{b}$",
                      "X_b_1": r"$\alpha^X_{b_1}$",
                      "X_be": r"$\alpha^X_{be}$",
                      "X_Q": r"$\alpha^X_{Q}$",
                      "Y_b_1": r"$\alpha^Y_{b_1}$",
                      "Y_be": r"$\alpha^Y_{be}$",
                      "Y_Q": r"$\alpha^Y_{Q}$",
                      "A_b_1": r"$\alpha_{b_1}$",
                      "A_be": r"$\alpha_{be}$",
                      "A_Q": r"$\alpha_{Q}$"} # define dictionary of latex strings for plotting for all of our parameters
            
            self.columns = [self.latex.get(param, param) for param in self.param_list] # have latex version of param_list
        else:
            # don't use latex
            self.latex = {}
            self.columns = self.param_list

    def _get_fiducial(self):
        """
        Get a dictionary of fiducial values for all free parameters.
        
        For redshift-dependent parameters, the value at the mean redshift of the survey is used.
        Parameters not explicitly defined default to 0, except for term amplitudes which default to 1.
        """
        fid_dict = {}
        for param in self.param_list: # Default to 0
            fid_dict[param] = 0
            
        mid_z = (self.cosmo_funcs.z_min + self.cosmo_funcs.z_max) / 2

        # Fiducial values for bias parameters
        for param in ['b_1', 'b_2', 'g_2', 'be', 'Q']:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs.survey, param)(mid_z)
        
        # Fiducial values for standard cosmological parameters
        for param in ['Omega_m', 'Omega_cdm', 'Omega_b', 'A_s', 'n_s', 'h']:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs, param)

        # Amplitudes of each contribution default to 1
        for param in self.cosmo_funcs.term_list:
            if param in self.param_list:
                fid_dict[param] = 1

        # Amplitude of bias parameters (Nuisance parameters)
        for param in ['X_b_1','X_be','X_Q','Y_b_1','Y_be','Y_Q','A_b_1','A_be','A_Q']:
            if param in self.param_list:
                fid_dict[param] = 1
        
        return fid_dict
    
    def planck_cov(self):
        """Returns Planck-BAO parameter covariance:
        Uses: parameter covariance from base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO"""

        full_cov = np.array([
            [1.8259383e-08, -4.9452862e-08, 1.1363663e-08, 1.8051082e-07, 3.3472141e-07, 1.3359350e-07],
            [-4.9452862e-08, 8.3384991e-07, -5.0175067e-08, -1.9209722e-06, -2.1282794e-06, -1.9412337e-06],
            [1.1363663e-08, -5.0175067e-08, 8.5677597e-08, 2.2499557e-07, 4.5210639e-07, 2.0928688e-07],
            [1.8051082e-07, -1.9209722e-06, 2.2499557e-07, 5.0196921e-05, 9.1926469e-05, 6.8814113e-06],
            [3.3472141e-07, -2.1282794e-06, 4.5210639e-07, 9.1926469e-05, 1.9787144e-04, 8.1080876e-06],
            [1.3359350e-07, -1.9412337e-06, 2.0928688e-07, 6.8814113e-06, 8.1080876e-06, 1.4266360e-05]])
        
        # ok so lets convert units: ['omega_b','omega_cdm','theta','tau','logA','n_s']
        full_cov[:2] *= 1/self.cosmo_funcs.h**2
        full_cov[:,:2] *= 1/self.cosmo_funcs.h**2
        # lnAs to A_s - conversion factor is actually A_s from error propogation
        full_cov[4] = full_cov[4]*self.cosmo_funcs.A_s
        full_cov[:,4] = full_cov[:,4]*self.cosmo_funcs.A_s
        
        # if we wanted to switch to using Omega_m (instead of Omega_cdm/Omega_b then we can combine errors)

        #find what parameters in this prior we are sampling over!
        params = ['Omega_b','Omega_cdm','theta','tau','A_s','n_s']
        columns = []
        for param in self.param_list:
            for j,prior_param in enumerate(params):
                if param==prior_param:
                    columns.append(j) # get columns/rows in cov_mat

        return full_cov[columns][:,columns] # NxN matrix

    def _name_chain(self,c,name):
        """define chainconsumer object and name of chain if none"""
        # Create ChainConsumer object
        if c == None:
            c = ChainConsumer()

        if name is None:
            # Generate unique name based on existing chains
            existing_names = c.get_names()
            
            # Find the next available number
            chain_number = 1
            while f"chain_{chain_number}" in existing_names:
                chain_number += 1
            
            name = f"chain_{chain_number}"

        return c,name

    def add_chain_cov(self,c=None,bias_values=None,name=None,cov=None):
        """
        Get chain from a covariance matrix - defualt is planck-covaraince-
        But later uses inverse fisher matrices
        
        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            bias_values (dict, optional): Best fit bias on parameter mean - calculate using get_bias etc.
                Keys should match parameter names, e.g., {'b_1': 1.0, 'sigma_8': 0.1}.
                If not provided then default is 0.
        
        Returns:
            ChainConsumer: ChainConsumer object with a brand new chain!
        """
        if cov is None:
            cov = self.planck_cov()
            param_list = [param for param in self.param_list if param in ['Omega_b','Omega_cdm','theta','tau','A_s','n_s']]
        else:
            param_list = self.param_list
        
        if bias_values:
            if isinstance(bias_values, list):
                bias_values = bias_values[-1] # use last entry which is sum of all terms if bias list is a list
        else:
            bias_values = {} # then yeet is empty

        # Use fiducial parameter values and apply bias if provided
        mean_values = np.zeros(len(param_list))
        for i, param in enumerate(param_list):
            if param in bias_values:
                offset = bias_values[param]
            else:
                offset = 0
            
            if param in self.fiducial:
                fid = self.fiducial[param]
            else:
                fid = 0

            mean_values[i] = fid + offset

        c,name = self._name_chain(c,name)
            
        # Create chain from covariance
        ch = Chain.from_covariance(
            mean_values, 
            cov,
            columns=param_list,
            name=name
        )
        c.add_chain(ch)

        return c

    def corner_plot(self, c=None, extents=None, figsize=None, truth=True, width=3, fid2=None, fontsize =16,tick_fontsize=12, **plot_kwargs):
        """
        Plot parameter contours using ChainConsumer.
        
        Args:
        c (ChainConsumer, optional): ChainConsumer object containing chains to plot.
            If None, creates a new ChainConsumer with just this Fisher matrix.
        extents (dict, optional): Plot extents (as tuples) for specific parameters.
            e.g., {'Omega_m': (0.2, 0.4), 'sigma_8': (0.6, 1.0)}.
        figsize (tuple, optional): Figure size as (width, height). If None, uses
            default size plus 3 inches in each dimension.
        truth (bool, optional): If True, adds fiducial parameter values as truth points
            on the plot. Default is True.
        **plot_kwargs: Additional keyword arguments passed to ChainConsumer's plot method.
        """
        if c is None:
            c = self.add_chain()
        
        # Add fiducial values as truth lines
        if truth and self.fiducial:
            c.add_truth(Truth(location=self.fiducial, color="#500724"))
        if fid2:
            c.add_truth(Truth(location=fid2, color="#16A085"))
        
        plot_config = PlotConfig(usetex=True,label_font_size=fontsize,tick_font_size=tick_fontsize)
        if extents:
            plot_config.extents = extents
        else:
            # Auto-generate extents to fit all chains and truth lines
            extents = {}
            for i,param in enumerate(self.param_list):
                mins, maxs = [], []
                largest_error = 0

                for chain_name in c.get_names():
                    samps = c.get_chain(name=chain_name).samples[self.param_list[i]]
                    mean, error = samps.mean(), samps.std()
                    mins.append(mean - width * error)
                    maxs.append(mean + width * error)
                    largest_error = max(largest_error, error)

                if truth and param in self.fiducial:
                    mins.append(self.fiducial[param] - largest_error * 0.1)
                    maxs.append(self.fiducial[param] + largest_error * 0.1)
                if fid2 and param in fid2:
                    mins.append(fid2[param] - largest_error * 0.1)
                    maxs.append(fid2[param] + largest_error * 0.1)

                if mins and maxs:
                    extents[self.param_list[i]] = (min(mins), max(maxs))
            plot_config.extents = extents
        
        plot_config.labels = self.latex # use latex labels
        c.set_plot_config(plot_config)
        
        fig = c.plotter.plot(**plot_kwargs)
        
        if figsize:
            fig.set_size_inches(figsize)
        else:
            current_size = fig.get_size_inches()
            fig.set_size_inches(current_size + 3)
            
        return fig, c

class FisherMat(BasePosterior):
    """
    Class to store and handle Fisher matrix results with built-in plotting capabilities.
    """ 
    def __init__(self, fisher_matrix, forecast, param_list, term=None, config=None, name=None):
        """
        Initialize Fisher result object.
        
        Args:
            fisher_matrix (np.ndarray): The Fisher information matrix
            param_list (list): List of parameter names
            term (str or list): The term(s) used in the forecast
            config (dict): Configuration used for the forecast (pkln, bkln, etc.)
            name (str): Optional name for this result
        """
        super().__init__(forecast, param_list, name=name)
        
        self.fisher_matrix = fisher_matrix
        self.term = term
        self.config = config or {}
        
        # if not computed then is None and if it is and a list then add all previous entries to get sum bias
        if isinstance(config['bias'],list) and len(config['bias'])>1:
            self.bias = config['bias']
            keys = self.bias[0].keys()
            tot = {}
            for key in keys:
                tot[key] = 0
                for i,b_dict in enumerate(self.bias):
                    tot[key] += b_dict[key]
            self.bias.append(tot)
                    
        self.bias = config['bias']
        
        # Compute derived quantities
        self.covariance = np.linalg.inv(fisher_matrix)
        self.errors = np.sqrt(np.diag(self.covariance))
        # add check for singular values:
        if np.isnan(self.errors).any():
            nan_indices = np.where(np.isnan(self.errors))[0]
            nan_params = [param_list[i] for i in nan_indices]
            warnings.warn(f"Singular matrix in {nan_params}")
            #raise ValueError(f"Singular matrix in {nan_params}")
        
        self.correlation = self._compute_correlation()
    
    def _compute_correlation(self):
        """Compute correlation matrix from covariance."""
        correlation = np.zeros_like(self.covariance)
        for i in range(len(self.param_list)):
            for j in range(len(self.param_list)):
                correlation[i,j] = self.covariance[i,j] / (self.errors[i] * self.errors[j])
        return correlation
    
    def get_error(self, param):
        """Get 1-sigma error for a specific parameter."""
        if param in self.param_list:
            idx = self.param_list.index(param)
            return self.errors[idx]
        else:
            raise ValueError(f"Parameter {param} not found in {self.param_list}")
    
    def get_correlation(self, param1, param2):
        """Get correlation coefficient between two parameters."""
        if param1 in self.param_list and param2 in self.param_list:
            i = self.param_list.index(param1)
            j = self.param_list.index(param2)
            return self.correlation[i,j]
        else:
            raise ValueError(f"One or both parameters not found in {self.param_list}")
    
    def summary(self):
        """Print summary of Fisher matrix results."""
        print(f"\n=== Fisher Matrix Results: {self.name} ===")
        print(f"Parameters: {self.param_list}")
        print(f"Term(s): {self.term}")
        print("\n1-sigma errors:")
        for i, param in enumerate(self.param_list):
            print(f"  σ({param}) = {self.errors[i]:.4f}")
        
        print("\nCorrelation matrix:")
        print("     ", end="")
        for param in self.param_list:
            print(f"{param:>8}", end="")
        print()
        for i, param1 in enumerate(self.param_list):
            print(f"{param1:>4} ", end="")
            for j, _ in enumerate(self.param_list):
                print(f"{self.correlation[i,j]:>8.3f}", end="")
            print()
    
    def add_chain(self,c=None,bias_values=None,name=None,cov=None):
        """
        Add the covariance (inverse Fisher) matrix as a chain to ChainConsumer object.
        Is wrapper for add_chain_cov in base class
        
        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            bias_values (dict, optional): Best fit bias on parameter mean - calculate using get_bias etc.
                Keys should match parameter names, e.g., {'b_1': 1.0, 'sigma_8': 0.1}.
                If not provided then default is 0.
        
        Returns:
            ChainConsumer: ChainConsumer object with this Fisher matrix added as a chain.
        """
        # reset defaults for FisherMat object
        if cov is None: 
            cov = self.covariance
        if bias_values is None:
            bias_values = self.bias
        return self.add_chain_cov(c=c,bias_values=bias_values,name=name,cov=cov)
    
    def compute_biases(self,bias_term,verbose=True):
        """Wrapper function of best_fit_bias in FullForecast:
        Compute biases for all parameters in fisher matrix
        Uses same config used to compute fisher."""

        terms = self.config['terms']
        pkln = self.config['pkln']
        bkln = self.config['bkln']
        t = self.config['t'] 
        s = self.config['s'] 
        r = self.config['r']
        sigma = self.config['sigma'] 
        nonlin = self.config['nonlin'] 

        bias_dict,_ = self.forecast.best_fit_bias(self.param_list, bias_term, terms,
                                                pkln,bkln,t=t,r=r,s=s,verbose=verbose,sigma=sigma,nonlin=nonlin)
        return bias_dict
    
    def plot_errors(self, relative=False, figsize=(8, 6)):
        """
        Plot parameter errors as a bar chart.
        parameters
        Args:
            relative (bool): Plot relative errors (σ/|mean|) if True
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if relative:
            # Compute relative errors (avoid division by zero)
            mean_values = np.zeros(len(self.param_list))  # Could be modified to use actual means
            rel_errors = []
            labels = []
            for i, param in enumerate(self.param_list):
                if abs(mean_values[i]) > 1e-10:
                    rel_errors.append(self.errors[i] / abs(mean_values[i]))
                    labels.append(param)
                else:
                    rel_errors.append(self.errors[i])
                    labels.append(param)
            
            ax.bar(labels, rel_errors)
            ax.set_ylabel('Relative Error (σ/|μ|)')
            ax.set_title(f'Relative Parameter Errors: {self.name}')
        else:
            ax.bar(self.param_list, self.errors)
            ax.set_ylabel('1-σ Error')
            ax.set_title(f'Parameter Errors: {self.name}')
        
        ax.set_xlabel('Parameters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig, ax
    
    def save(self, filename):
        """Save Fisher result to file."""
        np.savez(filename, 
                fisher_matrix=self.fisher_matrix,
                param_list=self.param_list,
                term=self.term,
                config=self.config,
                name=self.name)
    
    @classmethod
    def load(cls, filename):
        """Load Fisher result from file."""
        data = np.load(filename, allow_pickle=True)
        return cls(
            data['fisher_matrix'],
            data['param_list'].tolist(),
            data['term'].item(),
            data['config'].item(),
            data['name'].item()
        )

class Sampler(BasePosterior):
    """MCMC Sampler with cobaya with ChainConsumer plots.
    Assumes gaussian likelihood with parameter independent covariances."""
    def __init__(self, forecast, param_list, terms=None, bias_list=None, pkln=None,bkln=None, R_stop=0.005, max_tries=200, name=None, planck_prior=False, **kwargs):
        super().__init__(forecast, param_list, name=name)

        self.pkln = pkln
        self.bkln = bkln
        # terms which to compute that are parameter dependent
        self.terms = terms
        # use planck covariance as prior
        self.planck_prior = planck_prior

        if bias_list is None:
            bias_list = []

        # inititalize Pk/BkFroecast classes
        self.pk_args = []
        self.bk_args = []
        for i in range(forecast.num_bins):
            self.pk_args.append(PkForecast(forecast.z_bins[i], self.cosmo_funcs, k_max=forecast.k_max_list[i], s_k=forecast.s_k).args)
            self.bk_args.append(BkForecast(forecast.z_bins[i], self.cosmo_funcs, k_max=forecast.k_max_list[i], s_k=forecast.s_k).args)
        
        all_terms = [term for term in terms+param_list+bias_list if term in self.cosmo_funcs.term_list] # get list of needed terms to compute full 'true' theory
        # so this just gets total contribution - i.e. true theory - and also parameter independent covariance
        self.data, self.inv_covs = forecast._precompute_derivatives_and_covariances([all_terms],pkln=pkln,bkln=bkln,verbose=False,fNL=0)

        # set up cobaya sampler - define priors, starting value and initial step
        #standard term:
        standard_dict = {"prior": {"min": -100, "max": 100},"ref": 0,"proposal": 1}
        self.prior_dict = {
                "fNL": standard_dict,
                "GR2": standard_dict,
                "WS2": standard_dict,
                "WA2": standard_dict,
                "A_b_1": {"prior": {"min": 0.8, "max": 1.2},"ref": 1.0,"proposal": 5e-5},
                "n_s": {
                    "prior": {"min": 0.84, "max": 1.1},"ref": 0.9665,"proposal": 5e-5
                },
                "h": {
                    "prior": {"min": 0.64, "max": 0.82},"ref": 0.6776,"proposal": 1e-3
                },
                "A_s": {
                    "prior": {"min": 6e-10, "max": 4.8e-9},"ref": 2.105e-9,"proposal": 1e-11
                },
                "Omega_m": {
                    "prior": {"min": 0.17, "max": 0.42},"ref": 0.31,"proposal": 5e-5
                },
                "Omega_cdm": {
                    "prior": {"min": 0.13, "max": 0.38},"ref": 0.26,"proposal": 5e-5
                },
                "Omega_b": {
                    "prior": {"min": 0.041, "max": 0.057},"ref": 0.049,"proposal": 1e-5
                }
            }
        
        self.set_info(param_list,R_stop,max_tries)

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
        cosmo_funcs = self.update_cosmo_funcs(param_vals)

        kwargs = {} # create dict which is fed into function
        for i, param in enumerate(self.param_list):
            if param in ['fNL','t','r','s']: # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
                kwargs[param] = param_vals[i]

            if param in ['A_b_1']:
                tmp_param = param[2:] # i.e get b_1 from X_b_1
                # now lets also be able to marginalise over the amplitude parameters
                cosmo_funcs.survey = utils.modify_func(cosmo_funcs.survey, tmp_param, lambda f,par=param_vals[i]: f*(par))# default argument solves late binding
                cosmo_funcs.survey1 = utils.modify_func(cosmo_funcs.survey1, tmp_param, lambda f,par=param_vals[i]: f*(par))

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
        """Run cobaya sampler"""
        self.updated_info, self.mcmc = run(self.info)

    def add_chain(self,c=None,name=None,bins=12,skip_samples=0.3):
        """
        Add MCMC sample as a chain to a ChainConsumer object.
        
        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
        
        Returns:
            ChainConsumer: ChainConsumer object with MCMC sample added as a chain.
        """
        c,name = self._name_chain(c,name)

        # get pandas dataframe of samples
        if hasattr(self,'mcmc'):
            data_frame = self.mcmc.samples(skip_samples=skip_samples).data[self.param_list]
        elif hasattr(self,'dataframe'):
            data_frame = self.dataframe # loaded samples
        else:
            raise ValueError("Run/load a sample first!")
            
        c.add_chain(Chain(samples=data_frame, name=name))
        c.set_override(ChainConfig(bins=bins))

        return c
    
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

        data_frame = self.mcmc.samples(skip_samples=0.3).data[self.param_list]

        # The 'info' dict contains a reference to the 'get_likelihood' method,
        # which can't be pickled. We can reconstruct it during loading.
        attributes_to_save = {
            'param_list': self.param_list,
            'terms': self.terms,
            'pkln': self.pkln,
            'bkln': self.bkln,
            'data': self.data,
            'prior_dict': self.prior_dict,
            'dataframe': data_frame,
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
            
        print(f"Sampler state loaded from {filepath}")
        return new_sampler