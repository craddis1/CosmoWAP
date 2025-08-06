"""Just for Pk for now but contains class to get multi-tracer multipole covariances for a series of terms."""
import numpy as np
from scipy.special import eval_legendre

import cosmo_wap.pk as pk
from cosmo_wap.lib import utils

class FullCov:
    def __init__(self,fc,cosmo_funcs_list,cov_terms,sigma=None,n_mu=64,fast=False,nonlin=False):
        """
        Does full (multi-tracer) multipole covariance for given terms in a single redshift bin.
        Takes in PkForecast object.
        Do numerical mu integrals over regular expressions to get everything we need!"""
        self.fc = fc
        self.terms = cov_terms
        self.sigma = sigma
        self.nonlin = nonlin

        self.cosmo_funcs_list = cosmo_funcs_list

        nodes, self.weights = np.polynomial.legendre.leggauss(n_mu)#legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)
        if fast: # only go from 0,1 and use symmetry - cut mu integral in half - just need to know when it cancels!
            self.mu = (1)*(nodes+1)/2.0  # sample mu range [0,1]
        else:
            self.mu = (2)*(nodes+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!

        # make k,z broadcastable
        cosmo_funcs,kk,zz = fc.args
        self.kk_shape = len(kk)
        kk,zz = utils.enable_broadcasting(kk,zz,n=1) # if arrays add newaxis at the end so is broadcastable with mu!
        self.args = (cosmo_funcs,kk,zz)

        # basically we dont have an amazing system of including nonlinear effects in the covariance
        # (bispectrum is different and not yet implemented under the same method)
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            initial_state = cosmo_funcs_list[0]
            for cf in cosmo_funcs_list:
                cf.nonlin = True

        self.create_cache(*self.args)

        if nonlin:
            for cf in cosmo_funcs_list:
                cf.nonlin = initial_state

    def get_cov(self,ln,sigma=None):
        """Gets full covariance matrix"""
        self.sigma = sigma

        if self.fc.all_tracer:
            ll_cov = np.zeros((len(ln),len(ln),3,3,self.kk_shape),dtype=np.complex128)
        else:
            ll_cov = np.zeros((len(ln),len(ln),self.kk_shape),dtype=np.complex128)

        for i in range(len(ln)):
            for j in range(i,len(ln)):
                if self.fc.all_tracer:
                    ll_cov[i,j] = self.get_multi_tracer_ll(self.terms,ln[i],ln[j])
                else:
                    ll_cov[i,j] = self.get_single_tracer_ll(self.terms,ln[i],ln[j])

                if i!=j: # only need to compute top half!
                    if self.fc.all_tracer:
                        # so for this bit we use the transpose (in the case of no WS then this matrix is already hermitian)
                        ll_cov[j,i] = ll_cov[i,j].transpose(1, 0, 2)
                    else:
                        ll_cov[j,i] = ll_cov[i,j]

        return ll_cov

    def get_coef(self,l1,l2,mu):
        if self.sigma:
            return (2*l1+1)*(2*l2+1)*eval_legendre(l1,mu)*eval_legendre(l2,mu)#*np.exp)*np.exp()
        return (2*l1+1)*(2*l2+1)*eval_legendre(l1,mu)*eval_legendre(l2,mu) # So k_f**3/N_k will be included on the forecast end...
    
    def create_cache(self,*args,**kwargs):
        """Store all Pks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store dictionary of each term for each tracer combination
        | XX XY |
        | YX YY | where YX = np.conjugate(XY) in this case"""

        if len(self.cosmo_funcs_list)>1:
            size = 2
            self.pk_cache = [[{},{}],
                             [{},{}]]
        else:
            size = 1
            self.pk_cache = [[{}]]
        
        for i in range(size):
            for j in range(size):
                 if self.cosmo_funcs_list[i][j]: # we can skip some calculation for the XY non all-tracer case
                    for term in self.terms:
                        self.pk_cache[i][j][term] = getattr(pk,term).mu(self.mu,self.cosmo_funcs_list[i][j],*args[1:],**kwargs)
    
    def integrate_mu(self,t1,t2,t3,t4,terms,l1,mu):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer t1=t2=t3=t4=0 (i.e. P_XX P_XX)
        For say: P_XY P_XX t1=t2=t4=0;t3=1 - P_t1t3 P_t2t4
        """
        coef = eval_legendre(l1,mu)*self.weights

        _,kk,zz = self.args
        tot_cov = np.zeros(self.kk_shape,dtype=np.complex128) # so shape kk

        N_terms = len(terms)
        for i in range(N_terms+1): # ok we need to get all pairs of Pk_term_i()xPk_term_j() etc
             for j in range(N_terms+1):
                if i == N_terms: #add shot noise
                    a = 1/self.cosmo_funcs_list[t1][t3].n_g(zz) # t1+t3 has range [0,2] and gets the correct cosmo_funcs for shot noise - is zero in XY case
                else:
                    a = self.pk_cache[t1][t3][terms[i]]

                if j == N_terms:
                    b = 1/self.cosmo_funcs_list[t2][t4].n_g(zz)
                else:
                    b = self.pk_cache[t2][t4][terms[j]]

                tot_cov += np.sum(coef*a*b, axis=(-1)) # sum over last axis - mu
        return tot_cov
    
    def get_single_tracer_ll(self,terms,l1,l2):
        """Get full single-tracer covariance for multipole pair"""
        if len(self.cosmo_funcs_list)>1: # for XY covariance
            return (1/2)*(self.integrate_mu(0,1,0,1,terms,l1,l2) + self.integrate_mu(0,0,1,1,terms,l1,l2))
        return self.integrate_mu(0,0,0,0,terms,l1,l2)
    
    def get_single_tracer_ll(self,terms,l1,l2):
        """Get C[P^ab_{l}, P^cd_{l2}](k)
        C[P^ab_{l1}, P^cd_{l2}](k) = ((2*l1 + 1)(2*l2 + 1) / N_k) ( Int (d(Omega_k) / 4*pi) * L_1(mu) * 
                                        [L_2(mu)*P^ad(k,mu)*P^bc(k,mu)^* + L_2(-mu)*P^ac(k,mu)*P^bd(k,mu)^*]"""
        

        coef = (2*l1+1)*(2*l2+1)

        term1 = self.integrate_mu(t1,t2,t3,t4,terms,l1,self.mu) + self.integrate_mu(t1,t2,t4,t3,terms,l2,-self.mu)

        return self.integrate_mu(0,0,0,0,terms,l1,l2)

    def get_multi_tracer_ll(self,terms,l1,l2):
        """Get full multi-tracer matrix for multipole pair:
        Cov(P_i, P_j) = | P̃_XX²             P̃_XX⋅P̃_XY             P̃_XY²        |
                        | P̃_XX⋅P̃_YX    ½(P̃_XX⋅P̃_YY + P̃_XY⋅P̃_YX)   P̃_XY⋅P̃_YY    |
                        | P̃_YX²             P̃_YX⋅P̃_YY             P̃_YY²        |

        C[P^ab_{l_i}, P^cd_{l_j}](k) = ((2*l_i + 1)(2*l_j + 1) / N_k) ( Int (d(Omega_k) / 4*pi) * L_i(mu) * 
                                        [L_j(mu)*P^ad(k,mu)*P^bc(k,mu)^* + L_j(-mu)*P^ac(k,mu)*P^bd(k,mu)^*]

        So only l_odd x l_even thing are imaginary - the rest are purely real after mu integration

        | 00x00         00x01          01x01   |
        | 00x10    (00x11 + 10x01)     01x11   |
        | 10x10         10x11          11x11   |

        So real diagonal and complex off diagonals! For GR terms - WS makes it all complex

        Sometimes i use notation P_XY which is 01
        """
        
        cov_mt = np.zeros((3, 3, self.kk_shape),dtype=np.complex128) #create empty complex array

        # ok so this is so we can unpack which tracers in our powerspectrum - see matrix above!
        tracers1 = [(0,0),(0,1),(1,1)] # first digits eg iA x jB
        tracers2 = [(0,0),(1,0),(1,1)] # second digits eg Ai x Bj
        for i in range(3): # so this is loop over cov matrix above
            for j in range(3):
                if i==j==1: # special case
                    #                                                 01x10                                                      00x11
                    cov_mt[i,j] = (1/2)*(self.integrate_mu(*tracers1[i],*tracers2[j],terms,l1,l2) + self.integrate_mu(*tracers1[i],*tracers1[j],terms,l1,l2))
                else:
                    cov_mt[i,j] = self.integrate_mu(*tracers1[i],*tracers2[j],terms,l1,l2)
                
        return cov_mt