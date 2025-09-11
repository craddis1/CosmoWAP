"""
Forecasting classes for single redshift bins.

This module provides the building blocks for cosmological forecasts.
Computes SNR for a single bin - allowinng for fishers

Classes
-------
Forecast
    Abstract base class for a single-bin forecast calculation.
PkForecast
    Handles power spectrum (P(k)) forecasts.
BkForecast
    Handles bispectrum (B(k1, k2, k3)) forecasts.
"""

import numpy as np
import copy
import cosmo_wap.bk as bk
import cosmo_wap.pk as pk 
import cosmo_wap as cw
from cosmo_wap.lib import utils
from cosmo_wap.forecast.covariances import FullCov

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
        cache: dict = None,
        all_tracer: bool = False,
        cov_terms: list = None
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
        self.all_tracer = all_tracer

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

        if cov_terms is None:
            self.cov_terms = ['NPP'] # just newtonian covariance (+ shot)
        else:
            self.cov_terms = cov_terms
    
    def bin_volume(self,z,delta_z,f_sky=0.365): # get d volume/dz assuming spherical shell bins
        """Returns volume of a spherical shell in Mpc^3/h^3"""
        # V = 4 * pi * R**2 * (width)
        return f_sky*4*np.pi*self.cosmo_funcs.comoving_dist(z)**2 *(self.cosmo_funcs.comoving_dist(z+delta_z/2)-self.cosmo_funcs.comoving_dist(z-delta_z/2))
    
    def five_point_stencil(self,param,term,l,*args,dh=1e-3,cosmo_funcs=None, **kwargs):
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

        if cosmo_funcs is None:
            cosmo_funcs = self.cosmo_funcs

        #handle lists by summing terms - enables functionality to combine terms
        if type(param) is list:
            tot = []
            for par in param:
                tot.append(self.five_point_stencil(par,term,l,*args,dh=1e-3, **kwargs))
            return np.sum(tot,axis=0)

        if param in ['b_1','be','Q', 'b_2', 'g_2']:# only for b1, b_e, Q and also potentially b_2, g_2
            h = dh*getattr(cosmo_funcs.survey,param)(self.z_mid)
            if self.propogate and param not in ['be','Q', 'b_2', 'g_2']: # change model changes other biases too!
                def get_func_h(h,l):
                    cosmo_funcs = utils.create_copy(cosmo_funcs)
                    if type(cosmo_funcs.survey_params)==list:
                        obj = cosmo_funcs.survey_params[0]
                    else:
                        obj = cosmo_funcs.survey_params
                    
                    cosmo_funcs = cosmo_funcs.update_survey(utils.modify_func(obj, param, lambda f: f + h),verbose=False)
                    return func(term,l,cosmo_funcs, *args[1:], **kwargs) # args normally contains cosmo_funcs
                
            else: # in this case just b_1 changes but not say b_2 which can be dependent on b_1 in modelling...
                def get_func_h(h,l):
                    cosmo_funcs_h = utils.create_copy(cosmo_funcs) # make copy is good
                    cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, param, lambda f: f + h)
                    cosmo_funcs_h.survey1 = utils.modify_func(cosmo_funcs_h.survey1, param, lambda f: f + h)
                    if param in ['be','Q']: # reset betas - as they need to be recomputed with the new biases
                        cosmo_funcs_h.survey.betas = None
                        cosmo_funcs_h.survey1.betas = None
                    return func(term,l,cosmo_funcs_h, *args[1:], **kwargs) # args normally contains cosmo_funcs
                
        # ok lets add a way to marginalize over amplitude of biases with flexibility for multi-tracer
        elif param in ['X_b_1','X_be','X_Q','Y_b_1','Y_be','Y_Q','A_b_1','A_be','A_Q']:
            # so X is survey1 bias, Y survey2 bias and A both
            h = dh

            def get_func_h(h,l):
                cosmo_funcs_h = utils.create_copy(cosmo_funcs) # make copy is good
                tmp_param = param[2:] # i.e get b_1 from X_b_1
                
                if param[0] == 'X':
                    if cosmo_funcs_h.survey.t1: # is tracer1?
                        cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, tmp_param, lambda f: f*(1+h))
                    if cosmo_funcs_h.survey1.t1: # is tracer1?
                        cosmo_funcs_h.survey1 = utils.modify_func(cosmo_funcs_h.survey1, tmp_param, lambda f: f*(1+h))
                elif param[0] == 'Y': #  Y - so we affect tracer2
                    if not cosmo_funcs_h.survey.t1:
                        cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, tmp_param, lambda f: f*(1+h))
                    if not cosmo_funcs_h.survey1.t1:
                        cosmo_funcs_h.survey1 = utils.modify_func(cosmo_funcs_h.survey1, tmp_param, lambda f: f*(1+h))
                else: # so affects both tracers (affect or effect)
                    cosmo_funcs_h.survey = utils.modify_func(cosmo_funcs_h.survey, tmp_param, lambda f: f*(1+h))
                    cosmo_funcs_h.survey1 = utils.modify_func(cosmo_funcs_h.survey1, tmp_param, lambda f: f*(1+h))
                    
                if tmp_param in ['be','Q']: # reset betas - as they need to be recomputed with the new biases
                    cosmo_funcs_h.survey.betas = None
                    cosmo_funcs_h.survey1.betas = None
                elif tmp_param in ['b_1']: # reset deriv - could make so we dont do it for both but it not time cosuming
                    cosmo_funcs_h.survey.deriv = {}
                    cosmo_funcs_h.survey1.deriv = {}
                
                return func(term,l,cosmo_funcs_h, *args[1:], **kwargs) # args normally contains cosmo_funcs
            
        elif param in ['fNL','t','r','s']:
            # mainly for fnl but for any kwarg. fNL shape is determine by whats included in base terms...
            # also could make broadcastable....
            h = kwargs[param]*dh
            def get_func_h(h,l):
                wargs = copy.copy(kwargs)
                wargs[param] += h
                return func(term,l,*args, **wargs)

        elif param in ['Omega_m','Omega_cdm','Omega_b','A_s','sigma8','n_s','h']:
            # so for cosmology we recall ClassWAP with updated class cosmology  
            if self.cache:
                h = self.cache[-1][param]

                # now compute with existing expressions...
                def wrap_func(i,l):
                    cosmo_funcs_h = self.cache[i][param]
                    cosmo_funcs_h.update_survey(cosmo_funcs.survey_params) # so has right tracers with different cosmology
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
                        cosmo_h = utils.get_cosmo(**{param: current_value + h},k_max=self.cosmo_funcs.K_MAX*self.cosmo_funcs.h)
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
        """Compute SNR:
        
        Data vector is shape (ln,kk) in single tracer and (ln,3,kk) for multi-tracer (PK ONLY)
        Covariance is shape (ln,ln,kk) in single tracer and (ln,ln,3,3,kk) for multi-tracer (PK ONLY)
        """
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

        self.cov_mat = self.get_cov_mat(ln,sigma=sigma)

        #invert covariance and sum
        InvCov = self.invert_matrix(self.cov_mat) # invert array of matrices

        """
            (d1 d2)(C11 C12)^{-1}  (d1)
                   (C21 C22)       (d2)
        """
        # contract stuff
        return np.sum(np.einsum('ik,ijk,jk->k', d1, InvCov, np.conjugate(d2))) 
    
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
    """Now with multi-tracer capability: cosmo_funcs_list holds the information for XX,XY,YX and YY- so we can get full data vector but also covariances"""
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, cache=None,all_tracer=False,cov_terms=None,cosmo_funcs_list=None, fast=False):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, cache, all_tracer, cov_terms)
        
        self.N_k = 4*np.pi*self.k_bin**2 * (s_k*self.k_f)
        self.args = cosmo_funcs,self.k_bin,self.z_mid

        self.fast = False # can quicken covariance calculations but be careful with mu integral cancellations

        if cosmo_funcs_list is None:
            self.cosmo_funcs_list = [[cosmo_funcs]] # make single tracer case compatible with updated get_data_vector and get_cov_mat
        else:
            self.cosmo_funcs_list = cosmo_funcs_list

    def get_cov_mat(self,ln,sigma=None,n_mu=128):
        """compute covariance matrix for different multipoles. Shape: (ln x ln x kk) for single tracer
        Shape: (ln x ln x 3 x 3 x kk) for multi tracer

        so what we want is C = | C_l1l1   C_l1l2 |
                               | C_l2l1   C_l2l2 |
        """
            
        self.cov = FullCov(self,self.cosmo_funcs_list,self.cov_terms,sigma=sigma,n_mu=n_mu,fast=self.fast)
        cov_ll = self.cov.get_cov(ln,sigma)*self.k_f**3 /self.N_k # from comparsion with Quijote sims

        return cov_ll
    
    def get_cov_mat1(self,ln,sigma=None,nonlin=False):
        """
        Older version that does the mu integral analytically -
        Is faster but only has newtonian terms and for single tracers.
        compute covariance matrix for different multipoles. Shape: (ln x ln x kk)
        """
        
        # create an instance of covariance class...
        cov = pk.COV(*self.args,sigma=sigma,nonlin=nonlin) 
        const =  self.k_f**3 /self.N_k # from comparsion with Quijote sims 
        
        N = len(ln) #NxNxlen(k) covariance matrix
        cov_mat = np.zeros((N,N,len(self.args[1])))
        for i in range(N):
            for j in range(i,N): #only compute upper triangle of covariance matrix
                cov_mat[i, j] = (getattr(cov,f'N{ln[i]}{ln[j]}')()) * const
                if i != j:  # Fill in the symmetric entry
                    cov_mat[j, i] = cov_mat[i, j]
        return cov_mat
    
    def get_data_vector(self,func,ln,param=None,m=0,sigma=None,t=0,r=0,s=0,**kwargs):
        """
        Get datavactor for each multipole...
        If parameter provided return numerical derivative wrt to parameter - for fisher matrix routine
        """
        if self.all_tracer:
            cosmo_funcs_list = [self.cosmo_funcs_list[0][0],self.cosmo_funcs_list[0][1],self.cosmo_funcs_list[1][1]]
        else:
            cosmo_funcs_list = [self.cosmo_funcs]

        d1 = []
        for l in ln:
            if l & 1:
                cf_list = [self.cosmo_funcs] # odd multipoles only ever care about XY
            else:
                cf_list = cosmo_funcs_list

            if param is None: # If no parameter is specified, compute the data vector directly without derivatives.
                d1 += [pk.pk_func(func,l,cf,*self.args[1:],t=t,sigma=sigma,**kwargs) for cf in cf_list]
            else:
                #compute derivatives wrt to parameter
                d1 += [self.five_point_stencil(param,func,l,cf,*self.args[1:],dh=1e-3,sigma=sigma,t=t,**kwargs) for cf in cf_list]

        return np.array(d1)
    
class BkForecast(Forecast):
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, cache=None,all_tracer=False):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, cache, all_tracer)
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
    def get_cov_mat(self,ln,mn=(0,0),sigma=None,nonlin=False,**kwargs):
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