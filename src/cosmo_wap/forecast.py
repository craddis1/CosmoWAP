import numpy as np
from tqdm.auto import tqdm
import copy
import cosmo_wap.bk as bk
import cosmo_wap.pk as pk 
import cosmo_wap as cw 
from cosmo_wap.lib import utils
from matplotlib import pyplot as plt

#for plotting
from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig
from itertools import combinations
from cobaya import run

# be proper
from typing import Any, List
from abc import ABC, abstractmethod
#from typing import override

# lets define a base forecast class
class Forecast(ABC):
    def __init__(
        self,
        z_bin: tuple[float],
        cosmo_funcs: Any,
        k_max: float = 0.1,
        s_k: float = 1,
        cache: dict = None
    ):
        """
        Base initialization for power spectrum and bispectrum forecasts.
        Computes a forecast for a single redshift bin.

        Args:
            z_bin (List[float]): Redshift bin as [z_min, z_max].
            cosmo_funcs (Any): ClassWAP object has main functionality.
            k_max (float, optional): Maximum k value. Defaults to 0.1.
            s_k (float, optional): Scaling for k-bin width. Defaults to 1.
            verbose (bool, optional): Verbosity flag. Defaults to False.
        """

        self.cosmo_funcs = cosmo_funcs
        self.cache = cache

        z_mid = (z_bin[0] + z_bin[1])/2 + 1e-6
        delta_z = (z_bin[1] - z_bin[0])
        
        V_s = self.bin_volume(z_mid, delta_z, f_sky=cosmo_funcs.f_sky) # survey volume in [Mpc/h]^3
        self.k_f = 2*np.pi*V_s**(-1/3)  # fundamental frequency of survey
        
        delta_k = s_k*self.k_f  # k-bin width
        k_bin = np.arange(delta_k, k_max, delta_k)  # define k-bins
        
        # Cut based on comoving distance where WA expansion breaks... for endpoint LOS, minimum redshift
        com_dist = cosmo_funcs.comoving_dist(z_mid) #z_min
        k_cut = 2*np.pi/com_dist
        self.k_cut_bool = np.where(k_bin > k_cut, True, False)
        
        self.z_mid = z_mid
        self.k_bin = k_bin
        self.s_k   = s_k
        self.k_max = k_max
        self.z_bin = z_bin

        self.propogate = False # when calculate derivatives also change modelling for down the line functions - mainly for b_1
    
    def bin_volume(self,z,delta_z,f_sky=0.365): # get d volume/dz assuming spherical shell bins
        """Returns volume of a spherical shell in Mpc^3/h^3"""
        # V = 4 * pi * R**2 * (width)
        return f_sky*4*np.pi*self.cosmo_funcs.comoving_dist(z)**2 *(self.cosmo_funcs.comoving_dist(z+delta_z/2)-self.cosmo_funcs.comoving_dist(z-delta_z/2))
    
    def five_point_stencil(self,param,term,l,*args,dh=1e-3, **kwargs):
        """
        Computes the numerical derivative of a function with respect to a given param using the five-point stencil method.
        This method supports different parameter types, including survey biases, cosmology, and miscellaneous.
        Different parameter types will have different methods of computing derivatives.
        
        l th multipole is differentiated for given term for either bispectrum or power spectrum.
        Args:
            param (str): The name of the parameter with respect to which the derivative is computed.
            term: Which contribution.
            l: Multipole.
            *args: Regular power spectrum or bisepctrum arguments of cosmowap.
            dh (float, optional): Relative step size for finite differencing. Default is 1e-3.
            **kwargs: Additional keyword arguments to be passed to the target function.
        Returns:
            float: Numerical derivative
        Raises:
            Exception: If the specified parameter is not implemented for differentiation.
        """
        if hasattr(self,'V123'): # then we must be working with the bispectrum
            func = bk.bk_func
        else:
            func = pk.pk_func

        #handle lists by summing terms - enables functionality to combine terms
        if type(param) is list:
            tot = []
            for par in param:
                tot.append(self.five_point_stencil(par,term,l,*args,dh=1e-3, **kwargs))
            return np.sum(tot,axis=0)

        if param in ['b_1','be','Q', 'b_2', 'g_2']:# only for b1, b_e, Q and also potentially b_2, g_2
            h = dh*getattr(self.cosmo_funcs.survey,param)(self.z_mid)
            if self.propogate and param not in ['be','Q', 'b_2', 'g_2']: # change model changes other biases too!
                cosmo_funcs = utils.copy(self.cosmo_funcs)
                def get_func_h(h,l):
                    if type(cosmo_funcs.survey_params)==list:
                        obj = cosmo_funcs.survey_params[0]
                    else:
                        obj = cosmo_funcs.survey_params
                    
                    cosmo_funcs = cosmo_funcs.update_survey(utils.modify_func(obj, param, lambda f: f + h),verbose=False)
                    return func(term,l,cosmo_funcs, *args[1:], **kwargs) # args normally contains cosmo_funcs
                
            else: # in this case just b_1 changes but not say b_2 which can be dependent on b_1 in modelling...
                if self.cache:
                    h = self.cache[-1][param]
                    # now compute with existing expressions...
                    def wrap_func(i,l):
                        return func(term,l,self.cache[i][param], *args[1:], **kwargs)
                    
                    #return 5 point stencil
                    return (-wrap_func(0,l)+8*wrap_func(1,l)-8*wrap_func(2,l)+wrap_func(3,l))/(12*h) 

                def get_func_h(h,l):
                    cosmo_funcs_h = utils.create_copy(self.cosmo_funcs) # make copy is good
                    cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, param, lambda f: f + h)
                    if param in ['be','Q']: # reset betas
                        cosmo_funcs_h.survey.betas = None
                    return func(term,l,cosmo_funcs_h, *args[1:], **kwargs) # args normally contains cosmo_funcs
            
        elif param in ['fNL','t','r','s']:
            # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
            # also could make broadcastable....
            h = kwargs[param]*dh
            def get_func_h(h,l):
                wargs = copy.copy(kwargs)
                wargs[param] += h
                return func(term,l,*args, **wargs)

        elif param in ['Omega_m','Omega_b','A_s','sigma8','n_s','h']:
            # so for cosmology we recall ClassWAP with updated class cosmology  
            if self.cache:
                h = self.cache[-1][param]

                # now compute with existing expressions...
                def wrap_func(i,l):
                    return func(term,l,self.cache[i][param], *args[1:], **kwargs)
                
                #return 5 point stencil
                return (-wrap_func(0,l)+8*wrap_func(1,l)-8*wrap_func(2,l)+wrap_func(3,l))/(12*h) 
            else:
                current_value = getattr(self.cosmo_funcs,param) # get current value of param
                h = dh*current_value
                def get_func_h(h,l):

                    if self.cosmo_funcs.emulator:
                        cosmo_h,params = utils.get_cosmo(**{param: current_value + h},emulator=self.cosmo_funcs.emulator)# update cosmology for change in param
                        other_kwargs = {'emulator':self.cosmo_funcs.emu,'params':params}
                    else:
                        cosmo_h = utils.get_cosmo(**{param: current_value + h},k_max=K_MAX*self.cosmo_funcs.h)
                        other_kwargs = {}

                    cosmo_funcs_h = cw.ClassWAP(cosmo_h,self.cosmo_funcs.survey_params,compute_bias=self.cosmo_funcs.compute_bias,**other_kwargs)
                    # need to clean up cython structures as it gives warnings
                    cosmo_h.struct_cleanup()
                    cosmo_h.empty()
                    return func(term,l,cosmo_funcs_h, *args[1:], **kwargs)

        # and lastly for term contributions
        elif param in self.cosmo_funcs.term_list:
            return func(param,l, *args, **kwargs) # no need to differentiate, just return the function value
        else:
            raise Exception(param+" Is not implemented in this method yet...")
        
        # useful for in the case fNL =0
        if h == 0:
            h = 0.1
        # return array
        return (-get_func_h(2*h,l)+8*get_func_h(h,l)-8*get_func_h(-h,l)+get_func_h(-2*h,l))/(12*h)
    
    def invert_matrix(self,A):
        """
        invert array of matrices efficiently - https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        """
        identity = np.identity(A.shape[0], dtype=A.dtype)

        inv_mat = np.zeros_like(A)
        for i in range(A.shape[2]):
            inv_mat[:,:,i] = np.linalg.solve(A[:,:,i], identity)
        return inv_mat

    def SNR(self,func,ln,param=None,param2=None,m=0,t=0,r=0,s=0,sigma=None,nonlin=False):
        """Compute SNR"""
        if ln is None:
            return None
        elif type(ln) is not list:
            ln = [ln] # make compatible

        #data vector
        d1 = self.get_data_vector(func,ln,param=param,sigma=sigma,t=t,r=r,s=s) # they should be shape [len(ln),Number of k-bins/triangles]
        if param2 is not None and param2 is not param:  # for non-diagonal fisher terms
            d2 = self.get_data_vector(func,ln,param=param2,sigma=sigma,t=t,r=r,s=s)
        else:
            d2 = d1

        self.cov_mat = self.get_cov_mat(ln,sigma=sigma,nonlin=nonlin)
        
        #invert covariance and sum
        InvCov = self.invert_matrix(self.cov_mat)# invert array of matrices

        """
            (d1 d2)(C11 C12)^{-1}  (d1)
                   (C21 C22)       (d2)
        """
        # contract stuff
        return np.sum(np.einsum('ik,ijk,jk->k', d1, InvCov, d2)) 
    
    def combined(self,term,pkln=None,bkln=None,param=None,param2=None,m=0,t=0,r=0,s=0,sigma=None,nonlin=False):
        """for a combined pk+bk analysis - because we limit to gaussian covariance we have block diagonal covriance matrix"""
        
        # get both classes
        pkclass = PkForecast(self.z_bin, self.cosmo_funcs, self.k_max, self.s_k)
        bkclass = BkForecast(self.z_bin, self.cosmo_funcs, self.k_max, self.s_k)
            
        # get full contribution
        if pkln:
            pk_snr = pkclass.SNR(term,pkln,param=param,param2=param2,t=t,sigma=sigma,nonlin=nonlin)
        else:
            pk_snr = 0
        if bkln:
            bk_snr = bkclass.SNR(term,bkln,param=param,param2=param2,r=r,s=s,sigma=sigma,nonlin=nonlin)
        else:
            bk_snr = 0

        return pk_snr + bk_snr
        

class PkForecast(Forecast):
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, cache=None):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, cache)
        
        self.N_k = 4*np.pi*self.k_bin**2 * (s_k*self.k_f)
        self.args = cosmo_funcs,self.k_bin,self.z_mid
    
    def get_cov_mat(self,ln,sigma=None,nonlin=False):
        """compute covariance matrix for different multipoles. Shape: (ln x ln)"""
        
        # create an instance of covariance class...
        cov = pk.COV(*self.args,sigma=sigma,nonlin=nonlin) 
        const =  self.k_f**3 /self.N_k # from comparsion with Quijote sims 
        
        N = len(ln) #NxNxlen(k) covariance matrix
        cov_mat = np.zeros((N,N,len(self.args[1])))
        for i in range(N):
            for j in range(i,N): #only compute upper traingle of covariance matrix
                cov_mat[i, j] = (getattr(cov,f'N{ln[i]}{ln[j]}')()) * const
                if i != j:  # Fill in the symmetric entry
                    cov_mat[j, i] = cov_mat[i, j]
        return cov_mat
    
    def get_data_vector(self,func,ln,param=None,m=0,sigma=None,t=0,r=0,s=0,**kwargs):
        """
        Get datavactor for each multipole...
        If parameter providede return numerical derivative wrt to parameter - for fisher matrix
        Will vectorize with l as well....
        """
        if param is None:# If no parameter is specified, compute the data vector directly without derivatives.
            d_v = np.array([pk.pk_func(func,l,*self.args,t=t,sigma=sigma,**kwargs) for l in ln])
        else:
            #compute derivatives wrt to parameter
            d_v = np.array([self.five_point_stencil(param,func,l,*self.args,dh=1e-3,sigma=sigma,t=t,**kwargs) for l in ln]) 

        return d_v
    
class BkForecast(Forecast):
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, cache=None):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, cache)
        self.cosmo_funcs = cosmo_funcs

        k1,k2,k3 = np.meshgrid(self.k_bin ,self.k_bin ,self.k_bin ,indexing='ij')

        # so k1,k2,k3 have shape (N,N,N) 
        #create bool for closed traingles with k1>k2>k3
        is_triangle = np.full_like(k1,False).astype(np.bool_)
        s123 = np.ones_like(k1) # 2 isoceles and 6 equilateral
        beta = np.ones_like(k1) # e.g. see Eq 24 - arXiv:1610.06585v3
        for i in range(k1.shape[0]+1):
            if np.logical_or(i == 0,self.k_cut_bool[i-1]==False):
                continue
            for j in range(i+1):#enforce k1>k2
                if np.logical_or(j == 0,self.k_cut_bool[j-1]==False):
                    continue
                for k in range(i-j,j+1):# enforce k2>k3 and triangle condition |k1-k2|<k3
                    if np.logical_or(k == 0,self.k_cut_bool[k-1]==False):
                        continue

                    #for indexing-
                    ii = i-1; jj = j-1; kk = k-1
                    is_triangle[ii,jj,kk] = True # is a triangle

                    #get beta
                    if i + j == k:
                        beta[ii,jj,kk] = 1/2

                    #get s123
                    if i==j:
                        if j==k:
                            s123[ii,jj,kk]=6
                        else:
                            s123[ii,jj,kk]=2
                    elif j==k:
                        s123[ii,jj,kk]=2
                        
        #define attributes
        self.is_triangle = is_triangle
        self.beta = self.tri_filter(beta)
        self.s123 = self.tri_filter(s123)
        
        # filter array and flatten - now 1D arrays
        k1 = self.tri_filter(k1)
        k2 = self.tri_filter(k2)
        k3 = self.tri_filter(k3)

        #get theta and consider floating point errors
        theta = utils.get_theta(k1,k2,k3)
        
        self.V123 = 8*np.pi**2*k1*k2*k3*(s_k)**3 * self.beta #from thin bin limit -Ntri
        self.args = cosmo_funcs,k1,k2,k3,theta,self.z_mid # usual args - excluding r and s

    def tri_filter(self,arr):
        """
        flattens and selects closed triangles
        """
        return arr.flatten()[self.is_triangle.flatten()]
    
    ################ functions for computing SNR #######################################
    def get_cov_mat(self,ln,mn=(0,0),sigma=None,nonlin=False):
        """
        compute covariance matrix for different multipoles
        """
        # create an instance of covariance class...
        cov = bk.COV(*self.args,sigma=sigma)
        const = (4*np.pi)**2  *2 # from comparsion with Quijote sims 
  
        N = len(ln) #NxNxlen(k) covariance matrix
        cov_mat = np.zeros((N,N,len(self.args[1])))
        for i in range(N):
            for j in range(i,N): #only compute upper triangle of covariance matrix
                cov_mat[i, j] = (self.s123*cov.cov([ln[i],ln[j]],mn,nonlin=nonlin).real)/self.V123  * const
                if i != j:  # Fill in the symmetric entry
                    cov_mat[j, i] = cov_mat[i, j]
        return cov_mat
    
    def get_data_vector(self,func,ln,param=None,m=0,sigma=None,t=0,r=0,s=0,**kwargs):
        """Get data vector"""
            
        if param is None: # If no parameter is specified, compute the data vector directly without derivatives.
            d_v = np.array([bk.bk_func(func,l,*self.args,r,s,sigma=sigma,**kwargs) for l in ln])
        else:
            d_v = np.array([self.five_point_stencil(param,func,l,*self.args,dh=1e-3,sigma=sigma,r=r,s=s,**kwargs) for l in ln]) 
                
        return d_v

#################################################################################### Main frontend forecasting class for full survey not just single redshift bin

class FullForecast:
    def __init__(self,cosmo_funcs,kmax_func=None,s_k=1,nonlin=True,N_bins=None):
        """
        Do full survey forecast over redshift bins.
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
 
    def pk_SNR(self,term,pkln,param=None,param2=None,t=0,verbose=True,sigma=None,nonlin=False):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """    
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.PkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k)
            snr[i] = foreclass.SNR(term,ln=pkln,param=param,param2=param2,t=t,sigma=sigma,nonlin=nonlin)
        return snr
    
    def bk_SNR(self,term,bkln,param=None,param2=None,m=0,r=0,s=0,verbose=True,sigma=None,nonlin=False):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """

        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.BkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k)
            snr[i] = foreclass.SNR(term,ln=bkln,param=param,param2=param2,m=m,r=r,s=s,sigma=sigma,nonlin=nonlin)
        return snr
    
    def combined_SNR(self,term,pkln,bkln,param=None,param2=None,m=0,t=0,r=0,s=0,verbose=True,sigma=None,nonlin=False):
        """
        Get SNR at several redshifts for a given survey and contribution - powerspectrum + bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in range(len(self.k_max_list)):

            foreclass = cw.forecast.Forecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k)
            snr[i] = foreclass.combined(term,pkln=pkln,bkln=bkln,param=param,param2=param2,t=t,r=r,s=s,sigma=sigma,nonlin=nonlin)
        return snr
    
    def _precompute_cache(self, param_list, dh=1e-3):
        """
        Pre-computes the four cosmo_funcs objects needed for the five-point stencil
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
    
    def _precompute_derivatives_and_covariances(self,param_list,base_term='NPP',pkln=None,bkln=None,t=0,r=0,s=0,sigma=None,nonlin=False,verbose=True,use_cache=True,compute_cov=True,**kwargs):
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
                pk_fc = PkForecast(self.z_bins[i], self.cosmo_funcs, k_max=self.k_max_list[i], s_k=self.s_k, cache=cache)
                if compute_cov:
                    pk_cov_mat = pk_fc.get_cov_mat(pkln, sigma=sigma, nonlin=nonlin)
                    inv_covs[i]['pk'] = pk_fc.invert_matrix(pk_cov_mat)
 
            if bkln:
                bk_fc = BkForecast(self.z_bins[i], self.cosmo_funcs, k_max=self.k_max_list[i], s_k=self.s_k, cache=cache)
                if compute_cov:
                    bk_cov_mat = bk_fc.get_cov_mat(bkln, sigma=sigma, nonlin=nonlin)
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

    def get_fish(self, param_list, base_term='NPP', pkln=None, bkln=None, m=0, t=0, r=0, s=0, verbose=True, sigma=None, nonlin=False, bias_list=None, use_cache=True,**kwargs):
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
            all_param_list, base_term, pkln, bkln, t, r, s, sigma, nonlin,verbose, use_cache, **kwargs)

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
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2))

                    # Bispectrum contribution
                    if bkln:
                        d1 = derivs[i][bin_idx]['bk']
                        d2 = derivs[j][bin_idx]['bk']
                        inv_cov = inv_covs[bin_idx]['bk']
                        f_ij += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2))
                
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
                            bias[j][param_list[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2))

                        # Bispectrum contribution
                        if bkln:
                            d1 = derivs[i][bin_idx]['bk']
                            d2 = derivs[N+j][bin_idx]['bk']                            
                            inv_cov = inv_covs[bin_idx]['bk']
                            bias[j][param_list[i]] += np.sum(np.einsum('ik,ijk,jk->k', d1, inv_cov, d2))
                    
                    bias[j][param_list[i]] *= 1/fish_mat[i,i]

        config = {'base_term':base_term,'pkln':pkln,'bkln':bkln,'t':t,'r':r,'s':s,'sigma':sigma,'nonlin':nonlin,'bias':bias}
        return FisherMat(fish_mat, param_list, self, config=config)
    
    def best_fit_bias(self,param,bias_term,base_term='NPP',pkln=None,bkln=None,t=0,r=0,s=0,verbose=True,sigma=None,nonlin=False):
        """ Get best fit bias on one parameter if a particular contribution is ignored 
        New, more efficient method uses FisherMat instance - basically is just a little wrapper of get fish method.
        bfb is a dictionary and if bias_term is a list - bfb is the sum from all the terms."""

        fish_mat = self.get_fish(param,base_term=base_term, pkln=pkln, bkln=bkln, t=t, r=r, s=s, verbose=verbose, sigma=sigma, nonlin=nonlin,bias_list=bias_term)

        bfb = fish_mat.bias[-1] # is list containing a dictionary for each bias term
        fish = np.diag(fish_mat.fisher_matrix) # is array 

        return bfb,fish

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

    
########################################################################################################### plotting code: Uses chainconsumer - https://samreay.github.io/ChainConsumer/

class FisherMat:
    """
    Class to store and handle Fisher matrix results with built-in plotting capabilities.
    """
    
    def __init__(self, fisher_matrix, param_list, forecast_obj, term=None, config=None, name=None):
        """
        Initialize Fisher result object.
        
        Args:
            fisher_matrix (np.ndarray): The Fisher information matrix
            param_list (list): List of parameter names
            term (str or list): The term(s) used in the forecast
            config (dict): Configuration used for the forecast (pkln, bkln, etc.)
            name (str): Optional name for this result
        """
        
        self.fisher_matrix = fisher_matrix
        self.param_list = param_list
        self.forecast = forecast_obj
        self.term = term
        self.config = config or {}
        self.fiducial = self._get_fiducial()
        self.name = name or "_".join(param_list) # fisher name is amalgamation of parameters
        
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
            raise ValueError(f"Singular matrix in {nan_params}")
        
        self.correlation = self._compute_correlation()
    
    def _get_fiducial(self):
        """ Get dictionary of fiducial values for non-zero params. For redshift dependent parameters use mean redshift of bin.
        Parameters not added will default to 0."""
        fid_dict = {}
        for param in self.param_list: # default to 0 and overwrite later...
            fid_dict[param] = 0 
        cosmo_funcs = self.forecast.cosmo_funcs
        mid_z = (cosmo_funcs.z_min+cosmo_funcs.z_max)/2 #should be volume average tbh
        cosmo = cosmo_funcs.cosmo

        for param in ['b_1','b_2','g_2','be','Q']: # get biases
            fid_dict[param] = getattr(cosmo_funcs.survey,param)(mid_z)
        
        for param in ['Omega_m','Omega_b','A_s','n_s','h']: # cosmological parameters
            fid_dict[param] = getattr(cosmo_funcs,param)

        for param in cosmo_funcs.term_list:
            fid_dict[param] = 1
        
        return fid_dict
    
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
    
    def add_chain(self,c=None,bias_values=None,name=None):
        """
        Add this Fisher matrix as a chain to a ChainConsumer object.
        
        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            bias_values (dict, optional): Best fit bias on parameter mean - calculate using get_bias etc.
                Keys should match parameter names, e.g., {'b_1': 1.0, 'sigma_8': 0.1}.
                If not provided then default is 0.
        
        Returns:
            ChainConsumer: ChainConsumer object with this Fisher matrix added as a chain.
        """
        if not bias_values:# use default
            bias_values = self.bias
            if isinstance(bias_values, list):
                bias_values = bias_values[-1] # use last entry which is sum of all terms if bias list is a list

        # Use fiducial parameter values and apply bias if provided
        mean_values = np.zeros(len(self.param_list))
        for i, param in enumerate(self.param_list):
            if bias_values and param in bias_values:
                offset = bias_values[param]
            else:
                offset = 0
            
            if param in self.fiducial:
                fid = self.fiducial[param]
            else:
                fid = 0

            mean_values[i] = fid + offset

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
            
        # Create chain from covariance
        ch = Chain.from_covariance(
            mean_values, 
            self.covariance,
            columns=self.param_list,
            name=name
        )
        c.add_chain(ch)

        return c
    
    def corner_plot(self, c=None, extents=None, 
                     figsize=None, truth=True, width=3, fid2=None,  **plot_kwargs):
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
        
        # Add fiducial values if provided
        if truth:
            c.add_truth(Truth(location=self.fiducial, color="#500724"))#2E86C1
        if fid2:
            c.add_truth(Truth(location=fid2, color="#16A085"))
        
        # Set plot configuration
        plot_config = PlotConfig()
        if extents:
            plot_config.extents = extents
        else:
            #make sure ellipse AND fiducial are included...
            extents = {}
            for i,param in enumerate(self.param_list):
                mins = []
                maxs = []
                largest_error = 0
                # get means it different samples have different biases
                for name in c.get_names(): # is list
                    samps = c.get_chain(name=name).samples[param]
                    mean = samps.mean() # get mean (i.e. fiducial + bias)
                    error = samps.std() # error also gives a good scaling

                    mins.append(mean-width*error) # so for all chains
                    maxs.append(mean+width*error)
                    
                    if error > largest_error: # also want largest error for scale with plot
                        largest_error = error

                mins.append(self.fiducial[param]-largest_error*0.1)
                maxs.append(self.fiducial[param]+largest_error*0.1)
                if fid2:
                    mins.append(fid2[param]-largest_error*0.1)
                    maxs.append(fid2[param]+largest_error*0.1)

                extents[param] =  (min(mins),max(maxs))
            plot_config.extents = extents
        c.set_plot_config(plot_config)
        
        # Create plot
        fig = c.plotter.plot(**plot_kwargs)
        
        # Adjust figure size if requested
        if figsize:
            fig.set_size_inches(figsize)
        else:
            # Default: add some space - I like'em large
            current_size = fig.get_size_inches()
            fig.set_size_inches(current_size + 3)
        
        return fig, c
    
    def compute_biases(self,bias_term,verbose=True):
        """Wrapper function of best_fit_bias in FullForecast:
        Compute biases for all parameters in fisher matrix
        Uses same config used to compute fisher."""

        base_term = self.config['base_term']
        pkln = self.config['pkln']
        bkln = self.config['bkln']
        t = self.config['t'] 
        s = self.config['s'] 
        r = self.config['r']
        sigma = self.config['sigma'] 
        nonlin = self.config['nonlin'] 

        bias_dict,_ = self.forecast.best_fit_bias(self.param_list, bias_term, base_term,
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