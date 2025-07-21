"""Bunch of integration mainly for numerical integration over LOS orientation- mu (P(k,mu)) and mu,phi (B(k1,k2,k3,mu,phi))"""
import numpy as np
import scipy
from cosmo_wap.lib import utils

def int_mu(func,n_mu,cosmo_funcs,k1,zz,fast=False,**kwargs):
    """
    implements single legendre guass integral for mu integral
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    if fast: # only go from 0,1 and use symmetry - cut mu integral in half - just need to know when it cancels!
        mu_nodes = (1)*(nodes+1)/2.0  # sample mu range [0,1]
    else:
        mu_nodes = (2)*(nodes+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!

    # make k,z broadcastable
    kk,zz = utils.enable_broadcasting(k1,zz,n=1) # if arrays add newaxis at the end so is broadcastable with mu!

    #if fast:return np.sum(weights*func(mu_nodes,cosmo_funcs,kk,zz,**kwargs), axis=(-1)) # so i think it's still the same as 2 cancels with 2 from range
         
    return np.sum(weights*func(mu_nodes,cosmo_funcs,kk,zz,**kwargs), axis=(-1)) # sum over last axis - mu

#for numerical angular derivates - useful for FOG and consistency otherwise precompute analytic are quicker
def legendre(func,l,cosmo_funcs,k1,zz,t=0,sigma=None,n_mu=16,fast=False,**kwargs):
    """
    implements single legendre guass integral over mu for powerspectrum term
    """
    def integrand(mu,cosmo_funcs,k1,zz,t,sigma,**kwargs):
        leg = scipy.special.eval_legendre(l,mu)
        expression = func(mu,cosmo_funcs,k1,zz,t,sigma,**kwargs)
        
        if sigma is None: #no FOG
            dfog_val = 1
        else:
            dfog_val = np.exp(-(1/2)*((k1*mu)**2)*sigma**2)
            
        return ((2*l+1)/2)*leg*expression*dfog_val

    result = int_mu(integrand,n_mu,cosmo_funcs,k1,zz,t=t,sigma=sigma,fast=fast,**kwargs)
        
    return result

def int_gl_dbl(func,n,*args,**kwargs):
    """
    implements double legendre guass integral for mu and phi integrals - extension and specialiation
    of integrate.fixed_quad() - flexible for signal and covariance functions
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    mesh_nodes1,mesh_nodes2 = np.meshgrid(nodes,nodes,indexing ='ij')        #mesh gridding as 2d
    mesh_weights1,mesh_weights2 = np.meshgrid(weights,weights,indexing ='ij')

    phi_nodes = (2*np.pi)*(mesh_nodes1+1)/2.0 #sample phi range [0,2*np.pi]
    mu_nodes = (2)*(mesh_nodes2+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!
    
    return (2*np.pi)/2.0 * np.sum(mesh_weights1*mesh_weights2*func(phi_nodes, mu_nodes,*args,**kwargs), axis=(-2,-1)) #sum over last two axes (mu and phi)
    
#for numerical angular derivates - usfule for FOG and consistency otherwise precompute analytic are quicker
def ylm(func,l,m,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,sigma=None,n=16):
    """
    does numerical integration with double legendre gauss integration implemented in int_gl_dbl()
    is vectorised just last two axes need to be dimension 1 for numpy broadcasting
    """
    
    # Add size 1 dimensions to the last 2 axes if arrays to allow broadcasting with mu and phi
    k1, k2, k3, theta, zz = utils.enable_broadcasting(k1, k2, k3, theta, zz)
    
    def integrand(phi,mu,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma):
        ylm = scipy.special.sph_harm(m, l, phi, np.arccos(mu))
        expression = func(mu,phi,cosmo_funcs,k1,k2,k3,theta,zz,r,s)
        
        if sigma is None: #no FOG
            dfog_val = 1
        else:
            k3,theta = utils.get_theta_k3(k1,k2,k3,theta)
            mu2 = mu*np.cos(theta)+ (1-mu**2)**(1/2) *np.sin(theta)*np.cos(phi)
            mu3 = -(mu*k1+mu2*k2)/k3
            dfog_val = np.exp(-(1/2)*((k1*mu)**2+(k2 *mu2)**2+(k3 *mu3)**2)*sigma**2)
            
        return np.conjugate(ylm)*expression*dfog_val

    result = int_gl_dbl(integrand,n,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma)
        
    return result

#same as above with changes to work with covariance style functions
def cov_ylm(func,ln,mn,params,sigma=None,n=16):
    """
    does numerical integration with double legendre gauss integration implemented in int_gl_dbl()
    is vectorised just last two axes need to be dimension 1 for numpy broadcasting
    """
    def integrand(phi,mu,params,sigma):
        ylm = scipy.special.sph_harm(mn[0], ln[0], phi, np.arccos(mu))
        ylm1 = scipy.special.sph_harm(mn[1], ln[1], phi, np.arccos(mu))
        
        if sigma is not None: # include FOG in a way that it does not act on shot noise
            k1,k2,k3,theta,Pk1,Pk2,Pk3,_,_,_,_,_,_,_,_,_,f,D1,b1,_,_ = params
            mu2 = mu*np.cos(theta)+ (1-mu**2)**(1/2) *np.sin(theta)*np.cos(phi)
            mu3 = -(mu*k1+mu2*k2)/k3
            
            #add dfog to relevant parts (does not act of shot noise)
            Pk1 = Pk1*np.exp(-(1/2)*((k1*mu)**2)*sigma**2)
            Pk2 = Pk2*np.exp(-(1/2)*((k2*mu2)**2)*sigma**2)
            Pk3 = Pk3*np.exp(-(1/2)*((k3*mu3)**2)*sigma**2)
            
            params = k1,k2,k3,theta,Pk1,Pk2,Pk3,_,_,_,_,_,_,_,_,_,f,D1,b1,_,_

        expression = func(mu,phi,params)
        
        return 4*np.pi*ylm1*np.conjugate(ylm)*expression
   
    result = int_gl_dbl(integrand,n,params,sigma)
        
    return result