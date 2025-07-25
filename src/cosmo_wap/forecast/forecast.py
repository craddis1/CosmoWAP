"""
Main frontend forecasting class for forecasts over full surveys not just single bin
"""
import numpy as np
from tqdm.auto import tqdm

import cosmo_wap as cw 
from .core import PkForecast, BkForecast
from .posterior import FisherMat
from cosmo_wap.lib import utils

class FullForecast:
    def __init__(self,cosmo_funcs,kmax_func=None,s_k=1,nonlin=True,N_bins=None, all_tracer=False):
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

        if kmax_func is None:
            kmax_func = lambda zz: 0.1 + zz*0 #0.1 *cosmo_funcs.h*(1+zz)**(2/(2+cosmo_funcs.n_s]))

        self.z_bins = np.column_stack((z_lims[:-1], z_lims[1:]))
        self.k_max_list = kmax_func(self.z_mid)
            
        self.nonlin = nonlin # use Halofit Pk for covariance    
        self.cosmo_funcs = cosmo_funcs
        self.s_k = s_k

        # get args for each bin (basically just get k-vectors!)
        self.num_bins = len(self.z_bins)
 
    def pk_SNR(self,term,pkln,param=None,param2=None,t=0,verbose=True,sigma=None,all_tracer=False):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """    
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.PkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,all_tracer=all_tracer)
            snr[i] = foreclass.SNR(term,ln=pkln,param=param,param2=param2,t=t,sigma=sigma)
        return snr
    
    def bk_SNR(self,term,bkln,param=None,param2=None,m=0,r=0,s=0,verbose=True,sigma=None):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """

        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.BkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,all_tracer=False)
            snr[i] = foreclass.SNR(term,ln=bkln,param=param,param2=param2,m=m,r=r,s=s,sigma=sigma)
        return snr
    
    def combined_SNR(self,term,pkln,bkln,param=None,param2=None,m=0,t=0,r=0,s=0,verbose=True,sigma=None,all_tracer=False):
        """
        Get SNR at several redshifts for a given survey and contribution - powerspectrum + bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in range(len(self.k_max_list)):

            foreclass = cw.forecast.Forecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,all_tracer=all_tracer)
            snr[i] = foreclass.combined(term,pkln=pkln,bkln=bkln,param=param,param2=param2,t=t,r=r,s=s,sigma=sigma)
        return snr
    
    def _precompute_cache(self, param_list, dh=1e-3):
        """
        Pre-computes the four cosmo_funcs objects needed for the five-point stencil - is less necessary now cosmo_funcs has been sped up x50
        for each cosmological parameter. This is done only once for the entire forecast instead of for each bin.
        """

        cache = [{},{},{},{},{}] # for five point stencil - need 4 points - last entry is for h
        
        cosmo_params = [p for p in param_list if p in ['Omega_m','Omega_b', 'A_s', 'sigma8', 'n_s', 'h']]
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
                    cache[i][param] = cw.ClassWAP(cosmo_h, self.cosmo_funcs.survey_params, 
                                                    compute_bias=self.cosmo_funcs.compute_bias,**kwargs)
                    
                    cosmo_h.struct_cleanup()
                    cosmo_h.empty()
        
        bias_params = [p for p in param_list if p in ['b_1','be','Q', 'b_2', 'g_2']]
        if bias_params:
            for param in bias_params:
                h = dh*getattr(self.cosmo_funcs.survey,param)((self.cosmo_funcs.z_min+self.cosmo_funcs.z_max)/2)
                # now need to store h!
                cache[-1][param] = h
                                                                                                       
                for i,n in enumerate([2, 1, -1, -2]):
                    cosmo_funcs_h = utils.create_copy(self.cosmo_funcs) # make copy is good
                    # Default arguments are defined at the time of the function created
                    cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, param, lambda f, shift=n*h: f + shift)
                    cache[i][param] = cosmo_funcs_h
        
        if cache[0] is {}: # if empty
            print('empty cache')
            return None
        return cache
    
    def _precompute_derivatives_and_covariances(self,param_list,base_term='NPP',pkln=None,bkln=None,t=0,r=0,s=0,sigma=None,verbose=True,all_tracer=False,use_cache=True,compute_cov=True,**kwargs):
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
                pk_fc = PkForecast(self.z_bins[i], self.cosmo_funcs, k_max=self.k_max_list[i], s_k=self.s_k, cache=cache, all_tracer=all_tracer)
                if compute_cov:
                    pk_cov_mat = pk_fc.get_cov_mat(pkln, sigma=sigma,terms=base_term)
                    inv_covs[i]['pk'] = pk_fc.invert_matrix(pk_cov_mat)
 
            if bkln:
                bk_fc = BkForecast(self.z_bins[i], self.cosmo_funcs, k_max=self.k_max_list[i], s_k=self.s_k, cache=cache, all_tracer=all_tracer)
                if compute_cov:
                    bk_cov_mat = bk_fc.get_cov_mat(bkln, sigma=sigma,terms=base_term)
                    inv_covs[i]['bk'] = bk_fc.invert_matrix(bk_cov_mat)

            # --- Get data vector (once per parameter per bin) - if parameter is not a term it computes the derivative of the base_term wrt parameter 5 ---
            for j, param in enumerate(param_list):
                if pkln:
                    pk_deriv = pk_fc.get_data_vector(base_term, pkln, param=param, t=t, sigma=sigma,**kwargs)
                    data_vector[j][i]['pk'] = pk_deriv
                if bkln:
                    bk_deriv = bk_fc.get_data_vector(base_term, bkln, param=param, r=r, s=s, sigma=sigma,**kwargs)
                    data_vector[j][i]['bk'] = bk_deriv

        return data_vector, inv_covs

    def get_fish(self, param_list, base_term='NPP', pkln=None, bkln=None, m=0, t=0, r=0, s=0, all_tracer=False, verbose=True, sigma=None, bias_list=None, use_cache=True,**kwargs):
        """
        Compute fisher minimising redundancy (only compute each data vector/covariance one for each bin (and parameter of relevant).
        This routine computes covariance and data vector for each parameter once for each bin, then assembles the Fisher matrix. 
        Also allows for computation of best fit bias using bias terms which can be a list. - this is also the most efficient way to do this!
        """
        if not isinstance(param_list, list):  # if item is not a list, make it one
            param_list = [param_list]

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
            all_param_list, base_term, pkln, bkln, t, r, s, sigma=sigma, verbose=verbose, all_tracer=all_tracer, use_cache=use_cache, **kwargs)

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
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2).real)

                    # Bispectrum contribution
                    if bkln:
                        d1 = derivs[i][bin_idx]['bk']
                        d2 = derivs[j][bin_idx]['bk']
                        inv_cov = inv_covs[bin_idx]['bk']
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2).real)
                
                fish_mat[i, j] = f_ij.real
                if i != j:
                    fish_mat[j, i] = fish_mat[i, j]

            if bias_list: # integrate bias terms in as well
                for j in range(N_b):
                    bias[j][param_list[i]] = 0
                    # Sum contributions from each redshift bin
                    for bin_idx in range(len(self.z_bins)):
                        # Power spectrum contribution
                        if pkln:
                            d1 = derivs[i][bin_idx]['pk']
                            d2 = derivs[N+j][bin_idx]['pk'] # access bias parts of derivs
                            inv_cov = inv_covs[bin_idx]['pk']
                            # Perform the matrix multiplication part of the SNR calculation
                            bias[j][param_list[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2).real)

                        # Bispectrum contribution
                        if bkln:
                            d1 = derivs[i][bin_idx]['bk']
                            d2 = derivs[N+j][bin_idx]['bk']                            
                            inv_cov = inv_covs[bin_idx]['bk']
                            bias[j][param_list[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2).real)
                    
                    bias[j][param_list[i]] *= 1/fish_mat[i,i]

        config = {'base_term':base_term,'pkln':pkln,'bkln':bkln,'t':t,'r':r,'s':s,'sigma':sigma,'bias':bias}
        return FisherMat(fish_mat, param_list, self, config=config)
    
    def best_fit_bias(self,param,bias_term,base_term='NPP',pkln=None,bkln=None,t=0,r=0,s=0,verbose=True,sigma=None):
        """ Get best fit bias on one parameter if a particular contribution is ignored 
        New, more efficient method uses FisherMat instance - basically is just a little wrapper of get fish method.
        bfb is a dictionary and if bias_term is a list - bfb is the sum from all the terms."""

        fish_mat = self.get_fish(param,base_term=base_term, pkln=pkln, bkln=bkln, t=t, r=r, s=s, verbose=verbose, sigma=sigma, bias_list=bias_term)

        bfb = fish_mat.bias[-1] # is list containing a dictionary for each bias term
        fish = np.diag(fish_mat.fisher_matrix) # is array 

        return bfb,fish
    
########################################################################################################### plotting code: Uses chainconsumer - https://samreay.github.io/ChainConsumer/

