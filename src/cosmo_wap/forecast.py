import numpy as np
from tqdm.auto import tqdm
import cosmo_wap.bk as bk #import terms for the bispectrum
import cosmo_wap as cw 


def bin_volume(cosmo_funcs,z,delta_z=0.1,f_sky=0.365): # get d volume/dz assuming spherical shell bins
    return f_sky*4*np.pi*cosmo_funcs.comoving_dist(z)**2 *(cosmo_funcs.comoving_dist(z+delta_z)-cosmo_funcs.comoving_dist(z-delta_z))

class PkForecast:
    def __init__(self,z_bin,cosmo_funcs,k_max=0.1,s_k=1,verbose=False):
        
        z_mid = (z_bin[0]+z_bin[1])/2 + 0.0001
        delta_z = (z_bin[1]-z_bin[0])/2
        
        V_s = bin_volume(cosmo_funcs,z_mid,delta_z,f_sky=cosmo_funcs.f_sky)# in [Mpc/h]^3 used to get fundamental fequency

class BkForecast:
    def __init__(self,z_bin,cosmo_funcs,k_max=0.1,s_k=1,verbose=False):
        """
        Initialize k-binning scheme and sum over triangle all k-triangles
        """
        self.cosmo_funcs = cosmo_funcs
        z_mid = (z_bin[0]+z_bin[1])/2 + 0.0001
        delta_z = (z_bin[1]-z_bin[0])/2

        V_s = bin_volume(cosmo_funcs,z_mid,delta_z,f_sky=cosmo_funcs.f_sky)# in [Mpc/h]^3 used to get fundamental fequency

        self.k_f = 2*np.pi*V_s**(-1/3)# fundamental frequency of survey

        delta_k = s_k*self.k_f # k-bin width
        k_bin = np.arange(delta_k,k_max,delta_k) # define k-bins

        #so lets say we remove signal where WA expansion breaks... for endpoint this is at # only relevant for DESI
        #remove from low redshift point of bin
        com_dist = cosmo_funcs.comoving_dist(z_mid)#k_bin[0]
        k_cut = 2*np.pi/com_dist
        k_cut_bool = np.where(k_bin>k_cut,True,False)

        k1,k2,k3 = np.meshgrid(k_bin,k_bin,k_bin,indexing='ij')

        # so k1,k2,k3 have shape (N,N,N) 
        #create bool for closed traingles with k1>k2>k3
        is_triangle = np.full_like(k1,False).astype(np.bool_)
        s123 = np.ones_like(k1) # 2 isoceles and 6 equilateral
        beta = np.ones_like(k1) #e.g. see Eq 24 - arXiv:1610.06585v3
        for i in range(k1.shape[0]+1):
            if np.logical_or(i == 0,k_cut_bool[i-1]==False):
                continue
            for j in range(i+1):#enforce k1>k2
                if np.logical_or(j == 0,k_cut_bool[j-1]==False):
                    continue
                for k in range(i-j,j+1):# enforce k2>k3 and triangle condition |k1-k2|<k3
                    if np.logical_or(k == 0,k_cut_bool[k-1]==False):
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
        theta = cosmo_funcs.get_theta(k1,k2,k3)
        
        self.k_bin = k_bin
        self.V123 = 8*np.pi**2*k1*k2*k3*(delta_k/self.k_f)**3 * self.beta #from thin bin limit -Ntri
        self.args = cosmo_funcs,k1,k2,k3,theta,z_mid # usual args - excluding r and s

    def tri_filter(self,arr):
        """
        flattens and selects closed triangles
        """
        return arr.flatten()[self.is_triangle.flatten()]
    
    ################ functions for computing SNR #######################################
    def get_cov_mat(self,ln,m=0):
        """
        compute covariance matrix for different multipoles
        """
        # create an instance of covariance class...
        cov = bk.COV(*self.args)
        const = (4*np.pi)**2  *2 # from comparsion with Quijote sims 
        
        N = len(ln) #NxNxlen(k) covariance matrix
        cov_mat = np.zeros((N,N,len(self.args[1])))
        for i in range(N):
            for j in range(i,N): #only compute upper traingle of covariance matrix
                cov_mat[i, j] = (self.s123*getattr(cov,f'N{ln[i]}{ln[j]}_00')())/self.V123  * const
                if i != j:  # Fill in the symmetric entry
                    cov_mat[j, i] = cov_mat[i, j]
        return cov_mat
    
    def invert_matrix(self,A):
        """
        invert array of matrices somewhat quickly - check - https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        """
        identity = np.identity(A.shape[0], dtype=A.dtype)
        
        inv_mat = np.zeros_like(A)
        for i in range(A.shape[2]):
            inv_mat[:,:,i] = np.linalg.solve(A[:,:,i], identity)
        return inv_mat
    
    def get_data_vector(self,func,ln,m=0,func2=None,sigma=None,r=0,s=0):
        
        bk_tri  = []
        bk_tri2 = []
        for l in ln:# loop over all multipoles and append to lists
            if l == 0:
                rr=1/3;ss=1/3
            else:
                rr=r;ss=s
            
            if sigma is None:
                funcl  = getattr(func,"l"+str(l)) 
                bk_tri.append(funcl(*self.args,rr,ss))
            else:
                bk_tri.append(func.ylm(l,m,*self.args,rr,ss,sigma=sigma))

            if func2 != None:  # for non-diagonal fisher terms
                if sigma is None:
                    func2l  = getattr(func2,"l"+str(l))
                    bk_tri2.append(func2l(*self.args,rr,ss))
                else:
                    bk_tri2.append(func2.ylm(l,m,*self.args,rr,ss,sigma=sigma))
            else:
                bk_tri2 = bk_tri
                
        return bk_tri,bk_tri2
    
    def SNR(self,func,ln,m=0,func2=None,sigma=None,r=0,s=0):
        if type(ln) is not list:
            ln = [ln] # make compatible

        #data vector
        d1,d2 = self.get_data_vector(func,ln,m=m,func2=func2,sigma=sigma,r=r,s=s)# they should be shape [len(ln),Number of triangles]
        
        self.cov_mat = self.get_cov_mat(ln,m)
        
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
    
def get_SNR(func,l,m,cosmo_funcs,r=0,s=0,func2=None,verbose=True,s_k=1,kmax_func=None,sigma=None):
    """
    Get SNR at several redshifts for a given survey and contribution
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
        
############### old SNR func - updating to class and cross-multipole ###################################################       
def SNR(z_bin,func,l,m,cosmo_funcs,r=0,s=0,k_max=0.1,func2=None,sigma=None,s_k=1):
    """
    Sums over all traingles and calculates the SNR for a given redshift bin
    """
    
    z_mid = (z_bin[0]+z_bin[1])/2 + 0.0001
    delta_z = (z_bin[1]-z_bin[0])/2

    V_s = bin_volume(cosmo_funcs,z_mid,delta_z,f_sky=cosmo_funcs.f_sky)# in [Mpc/h]^3 used to get fundamental fequency

    k_f = 2*np.pi*V_s**(-1/3)# fundamental frequency of survey
    
    delta_k = s_k*k_f # k-bin width

    k_bin = np.arange(delta_k,k_max,delta_k)
    
    #so lets say we remove signal where WA expansion breaks... for endpoint this is at # only relevant for DESI
    #remove from low redshift point of bin
    com_dist = cosmo_funcs.comoving_dist(z_mid)#k_bin[0]
    k_cut = 2*np.pi/com_dist
    k_cut_bool = np.where(k_bin>k_cut,True,False)
    
    #print(k_cut)
    #print('k_f=',k_f)

    k1,k2,k3 = np.meshgrid(k_bin,k_bin,k_bin,indexing='ij')
    
    # so k1,k2,k3 have shape (N,N,N) - work with array until we sum at the end
    #create bool for closed traingles with k1>k2>k3...
    tri_bool = np.full_like(k1,False).astype(np.bool_)
    s123 = np.ones_like(k1) # 2 isoceles and 6 equilateral
    beta = np.ones_like(k1) #e.g. see Eq 24 - arXiv:1610.06585v3
    for i in range(k1.shape[0]+1):
        if np.logical_or(i == 0,k_cut_bool[i-1]==False):
            continue
        for j in range(i+1):#enforce k1>k2
            if np.logical_or(j == 0,k_cut_bool[j-1]==False):
                continue
            for k in range(i-j,j+1):# enforce k2>k3 and triangle condition |k1-k2|<k3
                if np.logical_or(k == 0,k_cut_bool[k-1]==False):
                    continue

                #for indexing-
                ii = i-1
                jj = j-1
                kk = k-1
                tri_bool[ii,jj,kk] = True
                
                #get beta
                if i + j == k:
                    beta[ii,jj,kk] = 1/2
                        
                #get s123
                if i==j:
                    if j==k:
                        #tri_bool[ii,jj,kk] = False
                        s123[ii,jj,kk]=6
                    else:
                        #tri_bool[ii,jj,kk] = False
                        s123[ii,jj,kk]=2
                elif j==k:
                    #tri_bool[ii,jj,kk] = False
                    s123[ii,jj,kk]=2
                    
                    
    V123 = 8*np.pi**2*k1*k2*k3*(delta_k/k_f)**3 * beta #from thin bin limit -Ntri

    #get theta and consider floating point errors
    theta = cosmo_funcs.get_theta(k1,k2,k3)

    args = cosmo_funcs,k1,k2,k3,theta,z_mid,r,s # usual args
    
    def snr_part(func,func2,l,m,s123):
        """
        returns snr contribution for given triangle
        """
        cov_lm = f'N{l}{m}'# get label
        
        cov = bk.COV(*args)
        
        const = (4*np.pi)**2  *2 # comparsion with sims 
        covariance = (s123*getattr(cov,cov_lm)())/V123  * const
        
        if l==0:
            covariance += (s123*cov.NL00())/V123  * const
        
        if sigma is None:
            funcl = getattr(func,"l"+str(l)) 
            bk_tri = funcl(*args)
        else:
            bk_tri = cw.integrate.ylm(func,l,m,*args,sigma=sigma)

        if func2 != None:  # for non-diagonal fisher terms
            if sigma is None:
                func2l = getattr(func2,"l"+str(l))
                bk_tri2 = func2l(*args)
            else:
                bk_tri2 = cw.integrate.ylm(func2,l,m,*args,sigma=sigma)
        else:
            bk_tri2 = bk_tri

        return bk_tri*np.conjugate(bk_tri2)/covariance
    
    #sum snr for all tri
    tri_snr = snr_part(func,func2,l,m,s123) # SNR for each triangle
    snr_tot = np.sum(tri_snr.flatten()[tri_bool.flatten()])

    return snr_tot

def get_SNR2(func,l,m,cosmo_funcs,r=0,s=0,func2=None,verbose=True,kmax_func=None,sigma=None):
    """
    Get SNR at several redshifts for a given survey and contribution
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
        
        snr[i] = SNR(z_bin,func,l,m,cosmo_funcs,r,s,k_max=kmax_func(z_mid[i]),func2=func2,sigma=sigma)
    return snr, z_mid