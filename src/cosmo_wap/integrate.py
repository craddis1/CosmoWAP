import numpy as np
import scipy

def single_int(func,cosmo_funcs,k1,zz=0,t=0,sigma=None,n=16):
    """ Do single integral for RSDxIntegrated term"""
    
    # create a [k,dx] 2D meshgrid
    k_grid  = k1[:,np.newaxis]
    
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
    nodes = np.real(nodes)
    
    # so limits [0,d]
    d = cosmo_funcs.comoving_dist(zz)

    # define nodes in comoving distance: for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
    dx_nodes = (d)*(nodes+1)/2.0 #sample phi range [0,d]
    
    # call term func
    int_grid = func(dx_nodes,cosmo_funcs,k_grid,zz,t,sigma)
    
    #(x1-x0)/2
    return (d)/2.0 * np.sum(weights*int_grid, axis=(-1)) #sum over last


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
 
    
#for numerical angular derivates - usfule for FOG and consistency otherwise precompute analytic are quicker
def ylm(func,l,m,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,sigma=None,n=16):
    """
    does numerical integration with double legendre gauss integration implemented in int_gl_dbl()
    is vectorised just last two axes need to be dimension 1 # could just add new axis for that reason...
    """
    
   #make sure k3 and theta are defined
    if theta is None:
        if k3 is None:
            raise  ValueError('Define either theta or k3')
        else:
            theta = cosmo_funcs.get_theta(k1,k2,k3)
            
    if k3 is None:
        k3 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(theta))#get k3 from triangle condition
        k3 = np.where(k3==0,1e-4,k3)
    
    # need to make sure the last 2 axes are empty if they are arrays such that they can be broadcasted with mu and phi
    bk_shape = [k1, k2, k3, theta, zz]

    for i, var in enumerate(bk_shape):
        if isinstance(var, np.ndarray):
            bk_shape[i] = var[..., None, None]

    k1, k2, k3, theta, zz = bk_shape

    def int_gl_dbl(func,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma,n):
        """
        implements double legendre guass integral for mu and phi integrals - extension and specialiation
        of integrate.fixed_quad()
        """

        nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)
        mesh_nodes1,mesh_nodes2 = np.meshgrid(nodes,nodes,indexing ='ij')        #mesh gridding as 2d
        mesh_weights1,mesh_weights2 = np.meshgrid(weights,weights,indexing ='ij')

        phi_nodes = (2*np.pi)*(mesh_nodes1+1)/2.0 #sample phi range [0,2*np.pi]
        mu_nodes = (2)*(mesh_nodes2+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!

        return (2*np.pi)/2.0 * np.sum(mesh_weights1*mesh_weights2*func(phi_nodes, mu_nodes,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma), axis=(-2,-1)) #sum over last two axes (mu and phi)
    
    def integrand(phi,mu,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma):
        ylm = scipy.special.sph_harm(m, l, phi, np.arccos(mu))
        expression = func(mu,phi,cosmo_funcs,k1,k2,k3,theta,zz,r,s)
        
        if sigma is None: #no FOG
            dfog_val = 1
        else:
            mu2 = mu*np.cos(theta)+ (1-mu**2)**(1/2) *np.sin(theta)*np.cos(phi)
            mu3 = -(mu*k1+mu2*k2)/k3
            dfog_val = np.exp(-(1/2)*((k1*mu)**2+(k2 *mu2)**2+(k3 *mu3)**2)*sigma**2)
            
        return np.conjugate(ylm)*expression*dfog_val

    
    result = int_gl_dbl(integrand,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma,n)
        
    return result