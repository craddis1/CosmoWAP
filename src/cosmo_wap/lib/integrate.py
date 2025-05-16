import numpy as np
import scipy
from cosmo_wap import utils

def single_int(func,cosmo_funcs,k1,zz=0,t=0,sigma=None,n=16):
    """ Do single integral for RSDxIntegrated term"""
    
    # create a [k,dx] 2D meshgrid
    k_grid  = k1[:,np.newaxis]
    
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    
    # so limits [0,d]
    d = cosmo_funcs.comoving_dist(zz)

    # define nodes in comoving distance: for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
    dx_nodes = (d)*(nodes+1)/2.0 # sample phi range [0,d]
    
    # call term func
    int_grid = func(dx_nodes,cosmo_funcs,k_grid,zz,t,sigma)
    
    #(x1-x0)/2
    return (d)/2.0 * np.sum(weights*int_grid, axis=(-1)) # sum over last


def double_int(func,cosmo_funcs,k1,zz=0,t=0,sigma=None,n=16):
    """ Do double integral for IntegratedxIntegrated term"""
    
    # create a [k,dx1,dx2] 3D meshgrid
    k_grid  = k1[:,np.newaxis,np.newaxis]
    
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    
    # so limits [0,d]
    d = cosmo_funcs.comoving_dist(zz)

    #  define nodes in comoving distance for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
    dx_nodes = (d)*(nodes+1)/2.0 #sample phi range [0,d]
    
    # so for last two axis we need to define the nodes and weights on the grid
    dx1 = dx_nodes[:,np.newaxis]
    dx2 = dx_nodes[np.newaxis,:]
    
    weights1 = weights[:,np.newaxis]
    weights2 = weights[np.newaxis,:]
    
    # call term func
    int_grid = func(dx1,dx2,cosmo_funcs,k_grid,zz,t,sigma)
    
    #(x1-x0)/2
    return ((d)/2.0)**2  * np.sum(weights1*weights2*int_grid, axis=(-2,-1)) # sum over last 2 axis


def int_gl_dbl(func,n,*args):
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
    
    return (2*np.pi)/2.0 * np.sum(mesh_weights1*mesh_weights2*func(phi_nodes, mu_nodes,*args), axis=(-2,-1)) #sum over last two axes (mu and phi)
    
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
        
        if sigma is None: #no FOG
            dfog_val = 1
        else:
            k1,k2,k3,theta,Pk1,Pk2,Pk3,_,_,_,_,_,_,_,_,_,f,D1,b1,_,_ = params
            mu2 = mu*np.cos(theta)+ (1-mu**2)**(1/2) *np.sin(theta)*np.cos(phi)
            mu3 = -(mu*k1+mu2*k2)/k3
            
            #add dfog to relevant parts (does not act of shot noise)
            Pk1 *= np.exp(-(1/2)*((k1*mu)**2)*sigma**2)
            Pk2 *= np.exp(-(1/2)*((k2*mu2)**2)*sigma**2)
            Pk3 *= np.exp(-(1/2)*((k3*mu3)**2)*sigma**2)
            
            params = k1,k2,k3,theta,Pk1,Pk2,Pk3,_,_,_,_,_,_,_,_,_,f,D1,b1,_,_

        expression = func(mu,phi,params)
        
        return 4*np.pi*ylm1*np.conjugate(ylm)*expression
   
    result = int_gl_dbl(integrand,n,params)
        
    return result