"""
Main frontend forecasting class for forecasts over full surveys not just single bin
"""
import numpy as np
from tqdm.auto import tqdm

import cosmo_wap as cw 
from .core import Forecast,PkForecast, BkForecast
from .posterior import FisherMat, Sampler
from cosmo_wap.lib import utils

class FullForecast:
    def __init__(self,cosmo_funcs,kmax_func=None,s_k=2,nonlin=False,N_bins=None,bkmax_func=None,WS_cut=True,n_mu=8,n_phi=8):
        """
        Do full survey forecast over redshift bins
        First get relevant redshifts and ks for each redshift bin
        Calls BkForecast and PkForecast which compute for particular bin
        """
    
        # get number of redshift bins survey is split into for forecast...
        if not N_bins:
            N_bins = round((cosmo_funcs.z_max - cosmo_funcs.z_min)*10) 

        z_lims = np.linspace(cosmo_funcs.z_min,cosmo_funcs.z_max,N_bins+1)
        self.z_mid = (z_lims[:-1] + z_lims[1:])/ 2 # get bin centers
        self.z_bins = np.column_stack((z_lims[:-1], z_lims[1:]))

        if kmax_func is None: # k -limit of analysis
            kmax_func = 0.1 #0.1 *cosmo_funcs.h*(1+zz)**(2/(2+cosmo_funcs.n_s]))
        
        self.k_max_list = self.get_kmax_list(kmax_func)

        if bkmax_func is None: # allow for different kmax for the bispectrum
            self.bk_max_list = self.k_max_list
        else:
            self.bk_max_list = self.get_kmax_list(bkmax_func)

        self.s_k = s_k
        self.WS_cut = WS_cut # cut scales where WS expansion breaks down

        # basically we dont have an amazing system of including nonlinear effects
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            cosmo_funcs = utils.create_copy(cosmo_funcs)
            cosmo_funcs.nonlin = True
        self.cosmo_funcs = cosmo_funcs

        #for covariances - need to increase when included integrated effects
        self.n_mu = n_mu
        self.n_phi = n_phi

        # get args for each bin (basically just get k-vectors!)
        self.num_bins = len(self.z_bins)

        self.cf_mat = self.setup_multitracer()
        self.cf_mat_bk = self.setup_multitracer_bk()
        self.set_bias_placeholders()

    def set_bias_placeholders(self):
        """define lists of bias parameters for forecasting"""
        self.amp_bias     = ['A_b_1','X_b_1','Y_b_1','X_be','X_Q','Y_be','Y_Q','A_be','A_Q','X_b_2','Y_b_2','A_b_2']
        self.png_amp_bias = ['X_loc_b_01','Y_loc_b_01','A_loc_b_01','X_eq_b_01','Y_eq_b_01','A_eq_b_01','X_orth_b_01','Y_orth_b_01','A_orth_b_01','A_loc_b_11']


    def get_kmax_list(self,kmax_func):
        """Get list of k_max"""
        if callable(kmax_func): # is it a function - if not then it just constant (or an array if you really wanted it to be)
            return kmax_func(self.z_mid)
        else:
            return np.ones_like(self.z_mid)*kmax_func

    def setup_multitracer(self,cosmo_funcs=None):
        """So lets set up cosmo_funcs objects for each multi-tracer combination and store in a list.
        If multi-tracer:
        | XX XY |
        | YX YY |
        Could be 3x3 matrix if we define 3 tracers...
        we also need to define for bispectrum even with two tracers
        """
        if cosmo_funcs is None:
            cosmo_funcs = self.cosmo_funcs # use the cosmo_funcs of the forecast object

        if cosmo_funcs.multi_tracer:
            cf_mat = []  
            for i in range(cosmo_funcs.N_tracers):
                cf_row = []
                for j in range(cosmo_funcs.N_tracers):
                    cf = utils.create_copy(cosmo_funcs) # create a copy of cosmo_funcs
                    cf.survey = [cosmo_funcs.survey[i],cosmo_funcs.survey[j]]
                    cf.survey_params = [cosmo_funcs.survey_params[i],cosmo_funcs.survey_params[j]]
                    if i == j: #then auto-correlation
                        cf.multi_tracer = False # now single tracer
                        cf.n_g = cf.survey[0].n_g
                    
                    cf_row.append(cf)
                cf_mat.append(cf_row)

            return cf_mat

        return [[cosmo_funcs]]
    
    def setup_multitracer_bk(self,cosmo_funcs=None):
        """Now for bisepctrum
        If multi-tracer: 2 x 2 x 2 shape
        Could be 3x3x3 matrix if we define 3 tracers...
        """
        if cosmo_funcs is None:
            cosmo_funcs = self.cosmo_funcs # use the cosmo_funcs of the forecast object

        if cosmo_funcs.multi_tracer:
            cf_matrix = [] 
            N = cosmo_funcs.N_tracers
            for i in range(N):
                cf_mat = [] # 2D slice
                for j in range(N):
                    cf_row = []
                    for k in range(N):
                        # Create a copy for the specific tracer combination
                        cf = utils.create_copy(cosmo_funcs)
                        
                        # Map the three tracers
                        cf.survey = [cosmo_funcs.survey[i], cosmo_funcs.survey[j], cosmo_funcs.survey[k]]
                        cf.survey_params = [cosmo_funcs.survey_params[i], cosmo_funcs.survey_params[j], cosmo_funcs.survey_params[k]]
                        
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
    
    ######################################################### helper functions
    
    def get_pk_bin(self,i=0,all_tracer=False,cache=None,cov_terms=None):
        """Get PkForecast object for a single redshift bin"""
        return PkForecast(self.z_bins[i], self.cosmo_funcs, self, k_max=self.k_max_list[i], all_tracer=all_tracer, cache=cache,cov_terms=cov_terms)

    def get_bk_bin(self,i=0,all_tracer=False,cache=None,cov_terms=None):
        """Get BkForecast object for a single redshift bin"""
        return BkForecast(self.z_bins[i], self.cosmo_funcs, self, k_max=self.bk_max_list[i], all_tracer=all_tracer, cache=cache,cov_terms=cov_terms)
    
    ############################################################### simple SNR forecasts
    def pk_SNR(self,term,pkln,param=None,param2=None,t=0,verbose=True,sigma=None,cov_terms=None,all_tracer=False):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = self.get_pk_bin(i,all_tracer=all_tracer,cov_terms=cov_terms)
            snr[i] = foreclass.SNR(term,ln=pkln,param=param,param2=param2,t=t,sigma=sigma)
        return snr
    
    def bk_SNR(self,term,bkln,param=None,param2=None,m=0,r=0,s=0,verbose=True,sigma=None,cov_terms=None,all_tracer=False):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.bk_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.bk_max_list))) if verbose else range(len(self.k_max_list)):
            foreclass = self.get_bk_bin(i,all_tracer=all_tracer,cov_terms=cov_terms)
            snr[i] = foreclass.SNR(term,ln=bkln,param=param,param2=param2,m=m,r=r,s=s,sigma=sigma)
        return snr
    
    def combined_SNR(self,term,pkln,bkln,param=None,param2=None,m=0,t=0,r=0,s=0,verbose=True,sigma=None,all_tracer=False):
        """
        Get SNR at several redshifts for a given survey and contribution - powerspectrum + bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in range(len(self.k_max_list)):

            foreclass = Forecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],all_tracer=False)
            snr[i] = foreclass.combined(term,pkln=pkln,bkln=bkln,param=param,param2=param2,t=t,r=r,s=s,sigma=sigma)
        return snr

    ######################################################## Routines for fishers and MCMC
    
    def _precompute_cache(self, param_list, dh=1e-3):
        """
        Pre-computes the four cosmo_funcs objects needed for the five-point stencil - is less necessary now cosmo_funcs has been sped up x50
        for each cosmological parameter. This is done only once for the entire forecast instead of for each bin.
        """

        cache = [{},{},{},{},{}] # for five point stencil - need 4 points - last entry is for h
        
        cosmo_params = [p for p in param_list if p in ['Omega_m','Omega_b','Omega_cdm', 'A_s', 'sigma8', 'n_s', 'h']]
        if cosmo_params:
            for param in cosmo_params:
                current_value = getattr(self.cosmo_funcs,param)
                h = dh * current_value
                cache[-1][param] = h
                
                K_MAX = self.cosmo_funcs.K_MAX
                if K_MAX > 1 and not self.cosmo_funcs.compute_bias: # also no point computing all of it!
                    K_MAX = 1
                                                                           
                for i,n in enumerate([2, 1, -1, -2]):
                    if self.cosmo_funcs.emulator:
                        cosmo_h,params = utils.get_cosmo(**{param: current_value + n * h},emulator=self.cosmo_funcs.emulator)
                        kwargs = {'emulator':self.cosmo_funcs.emu,'params':params}
                    else:
                        cosmo_h = utils.get_cosmo(**{param: current_value + n * h},k_max=K_MAX*self.cosmo_funcs.h)
                        kwargs = {}
                    cache[i][param] = cw.ClassWAP(cosmo_h,**kwargs) # does not initialise survey
                    
                    cosmo_h.struct_cleanup()
                    cosmo_h.empty()

        if cache[0] is {}: # if empty
            #print('empty cache')
            return None
        return cache
    
    def _precompute_derivatives_and_covariances(self,param_list,terms='NPP',cov_terms=None,pkln=None,bkln=None,t=0,r=0,s=0,sigma=None,verbose=True,all_tracer=False,use_cache=True,compute_cov=True,**kwargs):
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

        if verbose: print("Step 1: Pre-computing derivatives and inverse covariances...")
        for i in tqdm(range(num_bins), disable=not verbose, desc="Bin Loop"):
            # --- Covariance Calculation (once per bin) ---New method to compute and cache all derivatives and inverse covariances once.
            if pkln:
                pk_fc = self.get_pk_bin(i,all_tracer=all_tracer,cache=cache,cov_terms=cov_terms)
                if compute_cov:
                    pk_cov_mat = pk_fc.get_cov_mat(pkln, sigma=sigma,n_mu=self.n_mu)
                    inv_covs[i]['pk'] = pk_fc.invert_matrix(pk_cov_mat)
 
            if bkln:
                bk_fc = self.get_bk_bin(i,all_tracer=all_tracer,cache=cache,cov_terms=cov_terms)
                if compute_cov:
                    bk_cov_mat = bk_fc.get_cov_mat(bkln, sigma=sigma,n_mu=self.n_mu,n_phi=self.n_phi)
                    inv_covs[i]['bk'] = bk_fc.invert_matrix(bk_cov_mat)

            # --- Get data vector (once per parameter per bin) - if parameter is not a term it computes the derivative of the terms wrt parameter 5 ---
            for j, param in enumerate(param_list):
                if pkln:
                    pk_deriv = pk_fc.get_data_vector(terms, pkln, param=param, t=t, sigma=sigma,**kwargs)
                    data_vector[j][i]['pk'] = pk_deriv
                if bkln:
                    bk_deriv = bk_fc.get_data_vector(terms, bkln, param=param, r=r, s=s, sigma=sigma,**kwargs)
                    data_vector[j][i]['bk'] = bk_deriv

        return data_vector, inv_covs
    
    def _rename_composite_params(self,param_list):
        """ so if one "param" is a list itself - then lets just call our parameter in some frankenstein way"""
        param_list_names = []
        for param in param_list:
            if isinstance(param, list):
                param = "_".join(param)
            param_list_names.append(param)
        return param_list_names

    def get_fish(self, param_list, terms='NPP', cov_terms=None, pkln=None, bkln=None, m=0, t=0, r=0, s=0, all_tracer=False, verbose=True, sigma=None, bias_list=None, use_cache=True,**kwargs):
        """
        Compute fisher minimising redundancy (only compute each data vector/covariance one for each bin (and parameter of relevant).
        This routine computes covariance and data vector for each parameter once for each bin, then assembles the Fisher matrix. 
        Also allows for computation of best fit bias using bias terms which can be a list. - this is also the most efficient way to do this!
        """
        if not isinstance(param_list, list):  # if item is not a list, make it one
            param_list = [param_list]

        param_list_names = self._rename_composite_params(param_list)# get combined names for list of params - is used here to save biases
        
        N = len(param_list)
        fish_mat = np.zeros((N, N))

        if bias_list:
            if not isinstance(bias_list, list):  # if item is not a list, make it one
                bias_list = [bias_list]

            all_param_list = param_list + bias_list  # Add bias term to end of param list for precomputation

            N_b = len(bias_list)    
            bias = [{} for _ in range(N_b)] # empty dicts for each bias term
        else:
            bias = None
            all_param_list = param_list

        # Precompute
        derivs, inv_covs = self._precompute_derivatives_and_covariances(
            all_param_list, terms, cov_terms, pkln, bkln, t, r, s, sigma=sigma, verbose=verbose, all_tracer=all_tracer, use_cache=use_cache, **kwargs)

        # lets save them
        self.derivs = derivs
        self.inv_covs = inv_covs

        if verbose: print("\nStep 2: Assembling Fisher matrix...")
        # 2. Assemble the matrix using cached derivatives
        for i in range(N):
            for j in range(i, N):
                f_ij = 0
                # Sum contributions from each redshift bin
                for bin_idx in range(len(self.z_bins)):
                    # Power spectrum contribution
                    if pkln:
                        d1 = derivs[i][bin_idx]['pk']
                        d2 = derivs[j][bin_idx]['pk']
                        inv_cov = inv_covs[bin_idx]['pk']

                        # Perform the matrix multiplication part of the SNR calculation
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, np.conjugate(d2)).real)

                    # Bispectrum contribution
                    if bkln:
                        d1 = derivs[i][bin_idx]['bk']
                        d2 = derivs[j][bin_idx]['bk']
                        inv_cov = inv_covs[bin_idx]['bk']
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, np.conjugate(d2)).real)
                
                fish_mat[i, j] = f_ij.real
                if i != j:
                    fish_mat[j, i] = fish_mat[i, j]

            if bias_list: # integrate bias terms in as well
                for j in range(N_b):
                    bias[j][param_list_names[i]] = 0
                    # Sum contributions from each redshift bin
                    for bin_idx in range(len(self.z_bins)):
                        # Power spectrum contribution
                        if pkln:
                            d1 = derivs[i][bin_idx]['pk']
                            d2 = derivs[N+j][bin_idx]['pk'] # access bias parts of derivs
                            inv_cov = inv_covs[bin_idx]['pk']
                            # Perform the matrix multiplication part of the SNR calculation
                            bias[j][param_list_names[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, np.conjugate(d2)).real)

                        # Bispectrum contribution
                        if bkln:
                            d1 = derivs[i][bin_idx]['bk']
                            d2 = derivs[N+j][bin_idx]['bk']                            
                            inv_cov = inv_covs[bin_idx]['bk']
                            bias[j][param_list_names[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, np.conjugate(d2)).real)
                    
                    bias[j][param_list_names[i]] *= 1/fish_mat[i,i]

        config = {'terms':terms,'pkln':pkln,'bkln':bkln,'t':t,'r':r,'s':s,'sigma':sigma,'bias':bias}
        return FisherMat(fish_mat, self, param_list, config=config)
    
    def best_fit_bias(self,param,bias_term,terms='NPP',pkln=None,bkln=None,t=0,r=0,s=0,verbose=True,sigma=None):
        """ Get best fit bias on one parameter if a particular contribution is ignored 
        New, more efficient method uses FisherMat instance - basically is just a little wrapper of get fish method.
        bfb is a dictionary and if bias_term is a list - bfb is the sum from all the terms."""

        fish_mat = self.get_fish(param,terms=terms, pkln=pkln, bkln=bkln, t=t, r=r, s=s, verbose=verbose, sigma=sigma, bias_list=bias_term)

        bfb = fish_mat.bias[-1] # is list containing a dictionary for each bias term
        fish = np.diag(fish_mat.fisher_matrix) # is array - ignore marginalisation

        return bfb,fish
    
    def sampler(self, param_list, terms=None, cov_terms=None, bias_list=None, pkln=None,bkln=None,R_stop=0.005,max_tries=100, name=None,planck_prior=False, all_tracer=False, verbose=True, sigma=None):
        """Define Sampler instance which is used for MCMC samples"""

        return Sampler(self, param_list, terms=terms, cov_terms=cov_terms, bias_list=bias_list, pkln=pkln,bkln=bkln,R_stop=R_stop,max_tries=max_tries,name=name,planck_prior=planck_prior, all_tracer=all_tracer, verbose=verbose, sigma=sigma)
