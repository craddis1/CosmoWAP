#for numerical angular derivates - usfule for FOG and consistency otherwise precompute analytic are quicker
def bk_ylm(func,l,m,params,derivs,betas,r=0,s=0,sigma=None,paramsPNG=[]):
    """
    does numerical integration with double legendre gauss integration implemented in int_gl_dbl()
    is vectorised just last two axes need to be dimension 1 
    """

    def int_gl_dbl(func,*args):
        """
        implements double legendre guass integral for mu and phi integrals - extension and specialiation
        of integrate.fixed_quad()
        """
        n=16

        nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)
        mesh_nodes1,mesh_nodes2 = np.meshgrid(nodes,nodes,indexing ='ij')        #mesh gridding as 2d
        mesh_weights1,mesh_weights2 = np.meshgrid(weights,weights,indexing ='ij')

        phi_nodes = (2*np.pi)*(mesh_nodes1+1)/2.0 #sample phi range [0,2*np.pi]
        mu_nodes = (2)*(mesh_nodes2+1)/2.0 - 1 # sample mu range [-1,1] - so this is the natural gauss legendre range!

        return (2*np.pi)/2.0 * np.sum(mesh_weights1*mesh_weights2*func(phi_nodes, mu_nodes, *args), axis=(-2,-1)) #sum over last two axes (mu and phi)
    
    k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2 = params
    def integrand(phi,mu,*args):
        ylm = scipy.special.sph_harm(m, l, phi, np.arccos(mu))
        expression = func(*args,mu,phi)
        
        if sigma is None: #no FOG
            dfog_val = 1
        else:
            mu2 = mu*np.cos(theta)+ (1-mu**2)**(1/2) *np.sin(theta)*np.cos(phi)
            mu3 = -(mu*k1+mu2*k2)/k3
            dfog_val = np.exp(-(1/2)*((k1*mu)**2+(k2 *mu2)**2+(k3 *mu3)**2)*sigma**2)
            
        return np.conjugate(ylm)*expression*dfog_val
    
    args = get_args(func,params,derivs,betas,r,s,paramsPNG=paramsPNG) # get necessary args for the func
    
    result = int_gl_dbl(integrand,*args)
        
    return result