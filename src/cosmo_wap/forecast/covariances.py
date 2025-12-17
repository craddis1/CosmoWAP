"""Just for Pk for now but contains class to get multi-tracer multipole covariances for a series of terms."""
import numpy as np
from scipy.special import eval_legendre, sph_harm

import cosmo_wap.pk as pk
from cosmo_wap.lib import utils

from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm

__all__ = ['FullCovPk', 'FullCovBk']

class FullCovPk:
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

        # make k,z broadcastable # z will always be a float tbh
        cosmo_funcs,kk,self.zz = fc.args
        self.kk_shape = len(kk)
        kk = utils.enable_broadcasting(kk,n=1) # if arrays add newaxis at the end so is broadcastable with mu!
        self.args = (cosmo_funcs,kk,self.zz)

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
            ll_cov = self.get_multi_tracer(self.terms,ln) # full covariance matrix
        else:
            # simple single tracer case:
            ll_cov = np.zeros((len(ln),len(ln),self.kk_shape),dtype=np.complex128)

            for i in range(len(ln)):
                for j in range(i,len(ln)):
                        ll_cov[i,j] = self.get_single_tracer_ll(self.terms,ln[i],ln[j])
                        if i!=j: # only need to compute top half!
                            ll_cov[j,i] = np.conjugate(ll_cov[i,j])
        return ll_cov

    def create_cache(self,*args,**kwargs):
        """Store all Pks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store dictionary of each term for each tracer combination
        | XX XY |
        | YX YY | where YX = np.conjugate(XY) in this case"""

        N = len(self.cosmo_funcs_list)
        self.pk_cache = [[{} for _ in range(N)] for _ in range(N)] # create nested list of empty dicts
        
        for i in range(N):
            for j in range(i,N):
                 if self.cosmo_funcs_list[i][j]: # we can skip some calculation for the XY non all-tracer case
                    for term in self.terms:
                        self.pk_cache[i][j][term] = getattr(pk,term).mu(self.mu,self.cosmo_funcs_list[i][j],*args[1:],**kwargs)
                        if i != j:
                            self.pk_cache[j][i][term] = np.conjugate(self.pk_cache[i][j][term]) # this holds currently P_YX = P_XY*
    
    def integrate_mu(self,i1,i2,j1,j2,terms,l1,l2,mu):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer t1=t2=t3=t4=0 (i.e. P_XX P_XX)
        For say: P_XY P_XX t1=t2=t4=0;t3=1 - P_t1t3 P_t2t4
        """
        coef = (2*l1+1)*(2*l2+1)*eval_legendre(l1,self.mu)*eval_legendre(l2,mu)*self.weights

        tot_cov = np.zeros(self.kk_shape,dtype=np.complex128) # so shape kk

        N_terms = len(terms)
        for i in range(N_terms+1): # ok we need to get all pairs of Pk_term_i()xPk_term_j() etc
             for j in range(N_terms+1):
                if i == N_terms: #add shot noise
                    a = 1/self.cosmo_funcs_list[i1][i2].n_g(self.zz) # is zero in XY case
                else:
                    a = self.pk_cache[i1][i2][terms[i]]

                if j == N_terms:
                    b = 1/self.cosmo_funcs_list[j1][j2].n_g(self.zz)
                else:
                    b = self.pk_cache[j1][j2][terms[j]]

                tot_cov += np.sum(coef*a*np.conjugate(b), axis=(-1)) # sum over last axis - mu
        return tot_cov
    
    def get_tracer(self,a,b,c,d,terms,l1,l2):
        """Get C[P^ab_{l}, P^cd_{l2}](k)
        C[P^ab_{l1}, P^cd_{l2}](k) = ((2*l1 + 1)(2*l2 + 1) / N_k) ( Int (d(Omega_k) / 4*pi) * L_1(mu) *
                                        [L_2(mu)*P^ac(k,mu)*P^bd(k,mu)^* + L_2(-mu)*P^ad(k,mu)*P^bc(k,mu)^*]"""

        return (1/2)*(self.integrate_mu(a,c,b,d,terms,l1,l2,self.mu) + self.integrate_mu(a,d,b,c,terms,l1,l2,-self.mu))
    
    def get_single_tracer_ll(self,terms,l1,l2):
        """Get full single-tracer covariance for multipole pair"""
        if len(self.cosmo_funcs_list)>1: # then we have XY term
            return self.get_tracer(0,1,0,1,terms,l1,l2)
        return self.get_tracer(0,0,0,0,terms,l1,l2)
    
    def get_multi_tracer(self,terms,ln):
        """Now compute full matrix:
        
        Get full multi-tracer matrix for multipole pair:
        C(Pi, Pj) = │ C[P_li^XX, P_lj^XX]   C[P_li^XY, P_lj^XX]   C[P_li^YY, P_lj^XX] │
                    │ C[P_li^XX, P_lj^XY]   C[P_li^XY, P_lj^XY]   C[P_li^YY, P_lj^XY] │
                    │ C[P_li^XX, P_lj^YY]   C[P_li^XY, P_lj^YY]   C[P_li^YY, P_lj^YY] │

        So only l_odd x l_even thing are imaginary - the rest are purely real after mu integration
        """

        # find shape of covariance matrix: (len(data_vector),len(data_vector))
        length = 0
        for l in ln:
            if l & 1: #if odd
                length += 1
            else:
                length += 3

        cov_mt = np.zeros((length, length, self.kk_shape),dtype=np.complex128) #create empty complex array

        # lets build our covariance matix!
        # so first we loop over l and then over tracers
        # even multipoles have tracers XX,XY,YY but odd just have XY

        # keep track of row and column of each submatrix
        row = 0
        column = 0

        tt = [(0,0),(0,1),(1,1)]# XX,XY,YY
        for _,li in enumerate(ln):
            for _,lj in enumerate(ln):
                if li & 1: # binary operator to specify odd
                    tracer = [tt[1]]
                else:
                    tracer = tt
                
                if lj & 1:
                    tracer2 = [tt[1]]
                else:
                    tracer2 = tt

                # now loop over tracers - k1,k2 keep track of where we are in this submatrix
                for k1,t1 in enumerate(tracer):
                    for k2,t2 in enumerate(tracer2):
                        cov_mt[row+k1,column+k2] = self.get_tracer(*t1,*t2,terms,li,lj) # get matrix element

                # update what bit of covariance is being calculated - overarching (not on the level of the submatrices)
                column += len(tracer2)
            row += len(tracer)
            column = 0
                
        return cov_mt
    
    def plot_cov(self,ln,kn=0,real=True,log=True,vmin=None,vmax=None,cmap='RdBu',lnrwidth=None,**kwargs):
        """Lets plot the covariance"""
        cov = self.get_cov(ln)

        labels = []
        for l in ln:
            if l & 1: 
                labels.append(rf"$P^{{\rm BF}}_{l}$")
            else: # If even
                labels.extend([
                    rf"$P^{{\rm BB}}_{{{l}}}$",
                    rf"$P^{{\rm BF}}_{{{l}}}$",
                    rf"$P^{{\rm FF}}_{{{l}}}$"
                ])

        if log and not lnrwidth: # for regular log plots set zero value to white
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_under('white') # You can also use 'white', '#dddddd', etc.
                
        plt.figure(figsize=(10,7))
        if real:
            if log:
                if not vmin:
                    vmin = np.abs(cov[...,kn].real).min()
                if not vmax:
                    vmax = np.abs(cov[...,kn].real).max()
                
                if lnrwidth:
                    plt.pcolormesh(cov[...,kn].real, cmap=cmap,norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax))
                else:
                    plt.pcolormesh(np.abs(cov[...,kn].real), cmap=cmap,norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[...,kn].real, cmap=cmap)
        else:
            if log:
                if not vmin:
                    vmin = np.abs(cov[...,kn].imag).min()
                if not vmax:
                    vmax = np.abs(cov[...,kn].imag).max()

                if lnrwidth:
                    plt.pcolormesh(cov[...,kn].imag, cmap=cmap,norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax))
                else:
                    plt.pcolormesh(np.abs(cov[...,kn].imag), cmap=cmap,norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[...,kn].imag, cmap=cmap)

        plt.xticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        plt.yticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        cbar = plt.colorbar()
        cbar.set_label(r'$|C[P^{ab}_{\ell_i},P^{cd}_{\ell_j}](k)|$', **kwargs)
        return cbar

class FullCovBk:
    def __init__(self,fc,cosmo_funcs_list,cov_terms,sigma=None,n_mu=64,n_phi=32,fast=False,nonlin=False):
        """
        Does full (multi-tracer) multipole covariance for given terms in a single redshift bin.
        Takes in BkForecast object.
        Do numerical mu integrals over regular expressions to get everything we need!
        Same as Pk but now for bispectrum!
        Covariance has shape [k1,k2,k3,mu,phi]"""
        self.fc = fc
        self.terms = cov_terms
        self.sigma = sigma
        self.nonlin = nonlin

        self.cosmo_funcs_list = cosmo_funcs_list

        nodes, weights_mu = np.polynomial.legendre.leggauss(n_mu)#legendre gauss - get nodes and weights for given n
        if fast: # only go from 0,1 and use symmetry - cut mu integral in half - just need to know when it cancels!
            mu = (1)*(nodes+1)/2.0  # sample mu range [0,1]
        else:
            mu = nodes # sample mu range [-1,1] - so this is the natural gauss legendre range!

        # for phi
        nodes, weights_phi = np.polynomial.legendre.leggauss(n_phi)#legendre gauss - get nodes and weights for given n
        phi = (2*np.pi)*(nodes+1)/2.0  # sample mu range [0,2 *np.pi]
        self.weights = weights_mu[:,np.newaxis]*weights_phi # 2D GL weights

        # make k1,k2,k3,z broadcastable
        _,k1,k2,k3,_,self.zz = fc.args
        mu = mu[:,np.newaxis]
        k1,k2,k3 = utils.enable_broadcasting(k1,k2,k3,n=2) # if arrays add newaxis at the end so is broadcastable with mu!
        
        # lets define some bispectrum stuff
        _,theta = utils.get_theta_k3(k1,k2,k3,None)

        mu2 = mu*np.cos(theta) + np.sqrt(1-mu**2)*np.sin(theta)*np.cos(phi)
        mu3 = -(mu*k1+mu2*k2)/k3
        self.mus = mu,mu2,mu3

        self.ks = k1,k2,k3
        self.N_tri = len(k1) # is literally number of triangles - k1,k2,k3 are flattened to this shape

        # basically we dont have an amazing system of including nonlinear effects in the covariance
        # (bispectrum is different and not yet implemented under the same method)
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            initial_state = cosmo_funcs_list[0][0].nonlin
            for i in range(len(cosmo_funcs_list)):
                for j in range(len(cosmo_funcs_list[i])):
                    cosmo_funcs_list[i][j].nonlin = True

        self.create_cache()

        if nonlin:
            for i in range(len(cosmo_funcs_list)):
                for j in range(len(cosmo_funcs_list[i])):
                    cosmo_funcs_list[i][j].nonlin = initial_state

    def get_cov(self,ln): # is the same - so could create a parent class.
        """Gets full covariance matrix"""

        if self.fc.all_tracer:
            ll_cov = self.get_multi_tracer(self.terms,ln) # full covariance matrix
        else:
            # simple single tracer case:
            ll_cov = np.zeros((len(ln),len(ln),self.N_tri),dtype=np.complex128)

            for i in range(len(ln)):
                for j in range(i,len(ln)):
                        ll_cov[i,j] = self.get_single_tracer_ll(self.terms,ln[i],ln[j])
                        if i!=j: # only need to compute top half!
                            ll_cov[j,i] = np.conjugate(ll_cov[i,j])
        return ll_cov

    def create_cache(self,**kwargs):
        """Store all Bks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store dictionary of each term for each tracer combination

        | XX XY XZ |
        | YX YY YZ | 
        | ZX ZY ZZ | where YX = np.conjugate(XY) in this case -so traingle numbers compute (n+1)*n/2 pairs!
        However the standard case will be working with 2 tracers - like the power spectrum
        
        also store for k1,k2,k3 etc 
        so shape probably 2,2,3,N
        We also now do it with numerical mu-int
        """

        N = len(self.cosmo_funcs_list)
        self.pk_cache = [[[{} for _ in range(N)] for _ in range(N)] for _ in range(3)] # create nested list of empty dicts
        
        for i in range(N):
            for j in range(i,N):
                for ki in range(3): #  if self.cosmo_funcs_list[i][j]: # we can skip some calculation for the XY non all-tracer case
                    for term in self.terms:
                        if True: # analytic mus
                            self.pk_cache[ki][i][j][term] = getattr(pk,term).mu(self.mus[ki],self.cosmo_funcs_list[i][j],self.ks[ki],self.zz,**kwargs) # so can change to new mechanism -new int
                        else:
                            self.pk_cache[ki][i][j][term] = pk.get_mu(self.mus[ki],term,term,self.cosmo_funcs_list[i][j],self.ks[ki],self.zz,n=32) # so can change to new mechanism -new int
                        if i != j:
                            self.pk_cache[ki][j][i][term] = np.conjugate(self.pk_cache[ki][i][j][term]) # this holds currently P_YX = P_XY*
    
    def integrate_mu(self,i1,i2,j1,j2,k1,k2,terms,l1,l2,mu):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer i1=i2=j1=j2=k1=k2=0 (i.e. P_XX P_XX P_XX)
        For say: P_XY P_XX P_XX -> i1=j1=j2=k1=k2=0;j1=1 
        """
        m = 0
        phi = 0 # can edit later for m\neq0
        coef = 4*np.pi*np.conjugate(sph_harm(m, l1, phi, np.arccos(mu)))*sph_harm(m, l2, phi, np.arccos(mu))*self.weights

        tot_cov = np.zeros(self.N_tri,dtype=np.complex128) # so shape kk

        if self.sigma is None: # for FOG
            sigma = 0
        else:
            sigma=self.sigma

        # Note with numerical int we do not really need a for loop as we can just call each term altogether
        N_terms = len(terms)
        for i in range(N_terms+1): # ok we need to get all triplets of Pk_term_i()xPk_term_j()xPk_term_k() etc
             for j in range(N_terms+1):
                for k in range(N_terms+1):
                    if i == N_terms: #add shot noise
                        a = 1/self.cosmo_funcs_list[i1][i2].n_g(self.zz) # is zero in XY case
                    else:
                        a = self.pk_cache[0][i1][i2][terms[i]]*np.exp(-(1/2)*((self.ks[0]*self.mus[0])**2)*sigma**2)

                    if j == N_terms:
                        b = 1/self.cosmo_funcs_list[j1][j2].n_g(self.zz)
                    else:
                        b = self.pk_cache[1][j1][j2][terms[j]]*np.exp(-(1/2)*((self.ks[1]*self.mus[1])**2)*sigma**2)

                    if k == N_terms:
                        c = 1/self.cosmo_funcs_list[k1][k2].n_g(self.zz)
                    else:
                        c = self.pk_cache[2][k1][k2][terms[k]]*np.exp(-(1/2)*((self.ks[2]*self.mus[2])**2)*sigma**2)

                    tot_cov += (2*np.pi)/2.0 *np.sum(coef*a*b*c, axis=(-2,-1)) # sum over last 2 axes - mu and phi
        return tot_cov
    
    def get_tracer(self,a,b,c,d,e,f,terms,l1,l2):
        """Get C[B^abc_{l}, B^def_{l2}](k) - i.e. PPP term to bispectrum covariance
        C[B^abc_{l}, B^def_{l2}](k1,k2,k3) = ( Int (d(Omega_k) / 4*pi) * Y_l1m1(mu,phi) *Y_l2m2(mu,phi)
                                        [P^ad(k1,mu)*P^be(k2,mu2)*P^cf(k3,mu3)]"""

        return (self.integrate_mu(a,d,b,e,c,f,terms,l1,l2,self.mus[0]))
    
    def get_single_tracer_ll(self,terms,l1,l2):
        """Get full single-tracer covariance for multipole pair"""
        if len(self.cosmo_funcs_list)>1: # then we have XY term
            return self.get_tracer(0,1,0,0,1,0,terms,l1,l2)
        return self.get_tracer(0,0,0,0,0,0,terms,l1,l2)
    
    def get_multi_tracer(self,terms,ln):
        """Now compute full matrix:
        
        Get full multi-tracer matrix for multipole pair:
        C(Bi, Bj) = │ C[B_li^xXX, B_lj^xXX]   C[B_li^XXY, B_lj^xXX]   C[B_li^XYY, B_lj^xXX]   C[B_li^YYY, B_lj^xXX] │
                    │ C[B_li^xXX, B_lj^XXY]   C[B_li^XXY, B_lj^XXY]   C[B_li^XYY, B_lj^XXY]   C[B_li^YYY, B_lj^XXY] │
                    │ C[B_li^xXX, B_lj^XYY]   C[B_li^XXY, B_lj^XYY]   C[B_li^XYY, B_lj^XYY]   C[B_li^YYY, B_lj^XYY] │
                    │ C[B_li^xXX, B_lj^YYY]   C[B_li^XXY, B_lj^YYY]   C[B_li^XYY, B_lj^YYY]   C[B_li^YYY, B_lj^YYY] │

        So only l_odd x l_even thing are imaginary - the rest are purely real after mu integration
        Shape [4xln,4xln]
        """
        nl = len(ln)
        cov_mt = np.zeros((4*nl, 4*nl, self.N_tri),dtype=np.complex128) #create empty complex array

        # lets build our covariance matix!
        # so first we loop over l and then over tracers
        # keep track of row and column of each submatrix
        row = 0
        column = 0

        tracers = [(0,0,0),(0,0,1),(0,1,1),(1,1,1)]# XXx,XXY,XYY,YYY
        for _,li in enumerate(ln):
            for _,lj in enumerate(ln):
                # now loop over tracers - k1,k2 keep track of where we are in this submatrix
                for k1,t1 in enumerate(tracers):
                    for k2,t2 in enumerate(tracers):
                        cov_mt[row+k1,column+k2] = self.get_tracer(*t1,*t2,terms,li,lj) # get matrix element

                # update what bit of covariance is being calculated - overarching (not on the level of the submatrices)
                column += len(tracers)
            row += len(tracers)
            column = 0
                
        return cov_mt
    
    def plot_cov(self,ln,kn=0,real=True,log=True,vmin=None,vmax=None,cmap='RdBu',lnrwidth=None,**kwargs):
        """Lets plot the covariance"""
        cov = self.get_cov(ln)

        labels = []
        for l in ln:
            labels.extend([
                rf"$B^{{\rm BBB}}_{{{l}}}$",
                rf"$B^{{\rm BBF}}_{{{l}}}$",
                rf"$B^{{\rm BFF}}_{{{l}}}$",
                rf"$B^{{\rm FFF}}_{{{l}}}$"
            ])

        if log and not lnrwidth: # for regular log plots set zero value to white
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_under('white') # You can also use 'white', '#dddddd', etc.
                
        plt.figure(figsize=(10,7))
        if real:
            if log:
                if not vmin:
                    vmin = np.abs(cov[...,kn].real).min()
                if not vmax:
                    vmax = np.abs(cov[...,kn].real).max()
                
                if lnrwidth:
                    plt.pcolormesh(cov[...,kn].real, cmap=cmap,norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax))
                else:
                    plt.pcolormesh(np.abs(cov[...,kn].real), cmap=cmap,norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[...,kn].real, cmap=cmap)
        else:
            if log:
                if not vmin:
                    vmin = np.abs(cov[...,kn].imag).min()
                if not vmax:
                    vmax = np.abs(cov[...,kn].imag).max()

                if lnrwidth:
                    plt.pcolormesh(cov[...,kn].imag, cmap=cmap,norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax))
                else:
                    plt.pcolormesh(np.abs(cov[...,kn].imag), cmap=cmap,norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[...,kn].imag, cmap=cmap)

        plt.xticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        plt.yticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        cbar = plt.colorbar()
        cbar.set_label(r'$|C[B^{abc}_{\ell_i},B^{def}_{\ell_j}](k)|$', **kwargs)
        return cbar