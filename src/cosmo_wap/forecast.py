
import numpy as np
from tqdm.auto import tqdm
import copy
import cosmo_wap.bk as bk
import cosmo_wap.pk as pk 
import cosmo_wap as cw 
from cosmo_wap.lib import utils

# be proper
from typing import Any, List
from abc import ABC, abstractmethod
#from typing import override

# lets define a base forecast class
class Forecast(ABC):
    def __init__(
        self,
        z_bin: List[float],
        cosmo_funcs: Any,
        k_max: float = 0.1,
        s_k: float = 1,
        verbose: bool = False
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
    
    def bin_volume(self,z,delta_z,f_sky=0.365): # get d volume/dz assuming spherical shell bins
        """Returns volume of a spherical shell in Mpc^3/h^3"""
        # V = 4 * pi * R**2 * (width)
        return f_sky*4*np.pi*self.cosmo_funcs.comoving_dist(z)**2 *(self.cosmo_funcs.comoving_dist(z+delta_z/2)-self.cosmo_funcs.comoving_dist(z-delta_z/2))
    
    def five_point_stencil(self,parameter,term,l,*args,dh=1e-3, **kwargs):
        """
        Computes the numerical derivative of a function with respect to a given parameter using the five-point stencil method.
        This method supports different parameter types, including survey biases, cosmology, and miscellaneous.
        Different parameter types will have different methods of computing derivatives.
        
        l th multipole is differentiated for given term for either bispectrum or power spectrum.
        Args:
            parameter (str): The name of the parameter with respect to which the derivative is computed.
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

        if parameter in ['b_1','be_survey','Q_survey']:# only for b1, b_e, Q and also potentially b_2, g_2
            h = dh*getattr(self.cosmo_funcs.survey,parameter)(self.z_mid)
            def get_func_h(h):
                if type(self.cosmo_funcs.survey_params)==list:
                    cosmo_funcs_h = self.cosmo_funcs.update_survey(self.cosmo_funcs.survey_params[0].modify_func(parameter, lambda f: f + h),verbose=False)
                else:
                    cosmo_funcs_h = self.cosmo_funcs.update_survey(self.cosmo_funcs.survey_params.modify_func(parameter, lambda f: f + h),verbose=False)
                return func(term,l,cosmo_funcs_h, *args[1:], **kwargs) # args normally contains cosmo_funcs
            
        elif parameter in ['fNL','t','r','s']:
            # mainly for fnl but for any kwarg...
            # also could make broadcastable....
            h = kwargs[parameter]*dh
            def get_func_h(h):
                wargs = copy.copy(kwargs)
                wargs[parameter] += h
                return func(term,l,*args, **wargs)

        elif parameter in ['Omega_m','A_s','sigma8','n_s']:
            # so for cosmology we recall ClassWAP with updated class cosmology
            current_value = getattr(self.cosmo_funcs,parameter)
            h = dh*current_value
            def get_func_h(h):
                cosmo_h = utils.get_cosmo(**{parameter: current_value + h}) # update cosmology for change in parameter
                cosmo_funcs_h = cw.ClassWAP(cosmo_h,self.cosmo_funcs.survey_params,compute_bias=self.cosmo_funcs.compute_bias)
                # need to clean up cython structures as it gives warnings
                cosmo_h.struct_cleanup()
                cosmo_h.empty()
                return func(term,l,cosmo_funcs_h, *args[1:], **kwargs)
        
        else:
            raise Exception(parameter+" Is not implemented in this method yet...")
            
        return (-get_func_h(2*h)+8*get_func_h(h)-8*get_func_h(-h)+get_func_h(-2*h))/(12*h)
    
    def invert_matrix(self,A):
        """
        invert array of matrices efficiently - https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        """
        identity = np.identity(A.shape[0], dtype=A.dtype)

        inv_mat = np.zeros_like(A)
        for i in range(A.shape[2]):
            inv_mat[:,:,i] = np.linalg.solve(A[:,:,i], identity)
        return inv_mat

    def SNR(self,func,ln,m=0,func2=None,t=0,r=0,s=0,sigma=None,nonlin=False,parameter=None):
        """Compute SNR"""

        if type(ln) is not list:
            ln = [ln] # make compatible

        #data vector
        d1,d2 = self.get_data_vector(func,ln,func2=func2,sigma=sigma,t=t,r=r,s=s,parameter=parameter) # they should be shape [len(ln),Number of triangles]

        self.cov_mat = self.get_cov_mat(ln,sigma=sigma,nonlin=nonlin)
        
        #invert covariance and sum
        InvCov = self.invert_matrix(self.cov_mat)# invert array of matrices

        """
        (d1 d2)(C11 C12)^{-1}  (d1)
               (C21 C22)       (d2)
        """
        result = 0
        for i in range(len(d1)):
            for j in range(len(d1)):
                result += np.sum(d1[i]*d2[j]*InvCov[i,j])

        return result
    
    def combined(self,term,pkln=[0,2],bkln=[0],m=0,term2=None,t=0,r=0,s=0,sigma=None,nonlin=False,parameter=None):
        """for a combined pk+bk analysis - because we limit to gaussian covariance we have block diagonal covriance matrix"""
        
        # get both classes
        pkclass = PkForecast(self.z_bin, self.cosmo_funcs, self.k_max, self.s_k)
        bkclass = BkForecast(self.z_bin, self.cosmo_funcs, self.k_max, self.s_k)
        
        if term2 == None:
            term2 = term
            
        # get full contribution
        if pkln:
            pk_snr = pkclass.SNR(getattr(pk,term),pkln,func2=getattr(pk,term2),t=t,sigma=sigma,nonlin=nonlin,parameter=parameter)
        else:
            pk_snr = 0
        if bkln:
            bk_snr = bkclass.SNR(getattr(bk,term),bkln,func2=getattr(bk,term2),r=r,s=s,sigma=sigma,nonlin=nonlin,parameter=parameter)
        else:
            bk_snr = 0

        return pk_snr + bk_snr
        

class PkForecast(Forecast):
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, verbose=False):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, verbose)
        
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
    
    def get_data_vector(self,func,ln,m=0,func2=None,sigma=None,t=0,r=0,s=0,parameter=None):
        """
        Get datavactor for each multipole...
        If parameter providede return numerical derivative wrt to parameter - for fisher matrix
        """
        pk_power  = []
        pk_power2 = []
        for l in ln:# loop over all multipoles and append to lists
            if l == 0:
                tt=1/2
            else:
                tt=t
            
            if parameter is None: 
                # If no parameter is specified, compute the data vector directly without derivatives.
                pk_power.append(pk.pk_func(func,l,*self.args,tt,sigma=sigma))
            else:
                #compute derivatives wrt to parameter
                pk_power.append(self.five_point_stencil(parameter,func,l,*self.args,dh=1e-3,sigma=sigma,t=tt))

            if func2 != None:  # for non-diagonal fisher terms
                if parameter is None:
                    pk_power2.append(pk.pk_func(func2,l,*self.args,tt,sigma=sigma))
                else:
                    #compute derivatives wrt to parameter
                    pk_power2.append(self.five_point_stencil(parameter,func2,l,*self.args,dh=1e-3,sigma=sigma,t=tt))
            else:
                pk_power2 = pk_power
                
        return pk_power,pk_power2
    
class BkForecast(Forecast):
    def __init__(self, z_bin, cosmo_funcs, k_max=0.1, s_k=1, verbose=False):
        super().__init__(z_bin, cosmo_funcs, k_max, s_k, verbose)
        self.cosmo_funcs = cosmo_funcs

        k1,k2,k3 = np.meshgrid(self.k_bin ,self.k_bin ,self.k_bin ,indexing='ij')

        # so k1,k2,k3 have shape (N,N,N) 
        #create bool for closed traingles with k1>k2>k3
        is_triangle = np.full_like(k1,False).astype(np.bool_)
        s123 = np.ones_like(k1) # 2 isoceles and 6 equilateral
        beta = np.ones_like(k1) #e.g. see Eq 24 - arXiv:1610.06585v3
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
    def get_cov_mat(self,ln,mn=[0,0],sigma=None,nonlin=False):
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
    
    def get_data_vector(self,func,ln,m=0,func2=None,sigma=None,t=0,r=0,s=0,parameter=None):
        """Get data vector -call differents funcs if sigma!=None"""
        bk_tri  = []
        bk_tri2 = []
        for l in ln: # loop over all multipoles and append to lists
            if l == 0:
                rr=1/3;ss=1/3
            else:
                rr=r;ss=s
            
            if parameter is None:
                # If no parameter is specified, compute the data vector directly without derivatives.
                bk_tri.append(bk.bk_func(func,l,*self.args,rr,ss,sigma=sigma))
            else:
                bk_tri.append(self.five_point_stencil(parameter,func,l,*self.args,dh=1e-3,sigma=sigma,r=rr,s=ss))

            if func2 != None:  # for non-diagonal fisher terms
                if parameter is None:
                    bk_tri2.append(bk.bk_func(func2,l,*self.args,rr,ss,sigma=sigma))
                else:
                    bk_tri2.append(self.five_point_stencil(parameter,func2,l,*self.args,dh=1e-3,sigma=sigma,r=rr,s=ss))
            else:
                bk_tri2 = bk_tri
                
        return bk_tri,bk_tri2
    
def get_SNR(func,l,m,cosmo_funcs,r=0,s=0,func2=None,verbose=True,s_k=1,kmax_func=None,sigma=None):
    """
    Get SNR at several redshifts for a given survey and contribution - bispectrum
    """

    # get number of redshift bins survey is split into for forecast...
    if not hasattr(cosmo_funcs,'z_bins'):
        cosmo_funcs.z_bins = round((cosmo_funcs.z_max - cosmo_funcs.z_min)*10) + 1

    z_lims=np.linspace(cosmo_funcs.z_min,cosmo_funcs.z_max,cosmo_funcs.z_bins)
    z_mid =(z_lims[:-1] + z_lims[1:])/ 2 # get bin centers

    if kmax_func is None:
        kmax_func = lambda zz: 0.1 + zz*0 #0.1 *h*(1+zz)**(2/(2+cosmo_funcs.cosmo.get_current_derived_parameters(['n_s'])['n_s']))#

    # get SNRs for each redshift bin
    snr = np.zeros((len(z_lims)-1),dtype=np.complex64)
    for i in tqdm(range(len(z_lims)-1)) if verbose else range(len(z_lims)-1):
        z_bin = [z_lims[i],z_lims[i+1]]

        foreclass = cw.forecast.BkForecast(z_bin,cosmo_funcs,k_max=kmax_func(z_mid[i]),s_k=s_k,verbose=verbose)
        snr[i] = foreclass.SNR(func,ln=l,m=m,func2=func2,sigma=sigma,r=r,s=s)
    return snr, z_mid

def fisherij(func,func2,l,m,cosmo_funcs,r=0,s=0,sigma=None,verbose=False,kmax_func=None):
    return np.sum(get_SNR(func,l,m,cosmo_funcs,func2=func2,r=r,s=s,sigma=sigma,kmax_func=kmax_func,verbose=verbose)[0].real)

class FullForecast:
    def __init__(self,cosmo_funcs,kmax_func=None,s_k=1,nonlin=True):
        """
        Do full survey forecast over redshift bins.
        First get relevant redshifts and ks for each redshift bin
        Calls BkForecast and PkForecast which compute for particular bin
        """

        # get number of redshift bins survey is split into for forecast...
        if not hasattr(cosmo_funcs,'z_bins'):
            cosmo_funcs.z_bins = round((cosmo_funcs.z_max - cosmo_funcs.z_min)*10) + 1

        z_lims=np.linspace(cosmo_funcs.z_min,cosmo_funcs.z_max,cosmo_funcs.z_bins)
        z_mid =(z_lims[:-1] + z_lims[1:])/ 2 # get bin centers

        if kmax_func is None:
            kmax_func = lambda zz: 0.1 + zz*0 #0.1 *h*(1+zz)**(2/(2+cosmo_funcs.cosmo.get_current_derived_parameters(['n_s'])['n_s']))#

        self.z_bins = []
        self.k_max_list = []
        for i in range(len(z_lims)-1):
            self.z_bins.append([z_lims[i],z_lims[i+1]])
            self.k_max_list.append(kmax_func(z_mid[i]))
            
        self.nonlin = nonlin # use Halofit Pk for covariance    
        self.cosmo_funcs = cosmo_funcs
        self.s_k = s_k
    
    def pk_SNR(self,term,pkln,term2=None,t=0,verbose=True,sigma=None,nonlin=False,parameter=None):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """
        if term2 is not None: # get function from term unless None 
            func2 = getattr(pk,term2)
        else:
            func2 = term2
            
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.PkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,verbose=verbose)
            snr[i] = foreclass.SNR(getattr(pk,term),ln=pkln,func2=func2,t=t,sigma=sigma,nonlin=nonlin,parameter=parameter)
        return snr
    
    def bk_SNR(self,term,bkln,term2=None,m=0,r=0,s=0,verbose=True,sigma=None,nonlin=False,parameter=None):
        """
        Get SNR at several redshifts for a given survey and contribution - bispectrum
        """
        if term2 is not None: # get function from term unless None 
            func2 = getattr(bk,term2)
        else:
            func2 = term2
            
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.BkForecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,verbose=verbose)
            snr[i] = foreclass.SNR(getattr(bk,term),ln=bkln,m=m,func2=func2,r=r,s=s,sigma=sigma,nonlin=nonlin,parameter=parameter)
        return snr
    
    def combined_SNR(self,term,pkln,bkln,term2=None,m=0,t=0,r=0,s=0,verbose=True,sigma=None,nonlin=False,parameter=None):
        """
        Get SNR at several redshifts for a given survey and contribution - powerspectrum + bispectrum
        """
        # get SNRs for each redshift bin
        snr = np.zeros((len(self.k_max_list)),dtype=np.complex64)
        for i in tqdm(range(len(self.k_max_list))) if verbose else range(len(self.k_max_list)):

            foreclass = cw.forecast.Forecast(self.z_bins[i],self.cosmo_funcs,k_max=self.k_max_list[i],s_k=self.s_k,verbose=verbose)
            snr[i] = foreclass.combined(term,pkln=pkln,bkln=bkln,term2=term2,t=t,r=r,s=s,sigma=sigma,nonlin=nonlin,parameter=parameter)
        return snr
    
    def fisherij(self,term,pkln=[],bkln=[],term2=None,m=0,t=0,r=0,s=0,verbose=True,sigma=None,nonlin=False,parameter=None):
        "get fisher matrix component for a survey"
        if pkln != []:
            if bkln != []:
                f_ij = np.sum(self.combined_SNR(term,pkln,bkln,term2=term2,m=m,t=t,r=r,s=s,
                                                verbose=verbose,sigma=sigma,nonlin=nonlin,parameter=parameter).real)
            else:
                f_ij = np.sum(self.pk_SNR(term,pkln,term2=term2,t=t,verbose=verbose,sigma=sigma,nonlin=nonlin,parameter=parameter).real)
        else:
            if bkln != []:
                f_ij = np.sum(self.bk_SNR(term,bkln,term2=term2,m=m,r=r,s=s,verbose=verbose,sigma=sigma,nonlin=nonlin,parameter=parameter).real)
            else:
                raise Exception("No multipoles selected!")
        return f_ij
    
    
    def get_fish(self,term_list,pkln=[],bkln=[],m=0,t=0,r=0,s=0,verbose=True,sigma=None,nonlin=False):
        """
        get fisher matrix for list of terms 
        """

        N = len(term_list)#get size of matrix
        fish_mat = np.zeros((N,N))

        for i in tqdm(range(N)):
            for j in range(i,N): # only compute top half
                fish_mat[i,j] =  fisherij(term_list[i],pkln,bkln,term2=term_list[j],t=t,r=r,s=s,
                                          verbose=verbose,sigma=sigma,nonlin=nonlin)
                if i != j:  # Fill in the symmetric entry
                    fish_mat[j, i] = fish_mat[i, j]

        return fish_mat
    
    def best_fit_bias(self,term,term2,pkln=[],bkln=[],t=0,r=0,s=0,verbose=True,sigma=None,parameter=None):
        """ Get best fit bias on one parameter if a particular contribution is ignored """

        fisher = self.fisherij(term,pkln=pkln,bkln=bkln,t=t,r=r,s=s,verbose=verbose,sigma=sigma,parameter=parameter)

        fishbias = self.fisherij(term,term2=term2,pkln=pkln,bkln=bkln,t=t,r=r,s=s,verbose=verbose,sigma=sigma,parameter=parameter)

        return fishbias/fisher,fisher
