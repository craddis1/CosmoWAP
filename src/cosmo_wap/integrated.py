import numpy as np

class BaseInt:
    """Base Integral class with defined integration funcs and parameter getter funcs"""
    def __init__(self, cosmo_funcs):
        """Initialize with cosmo_funcs"""
        self.cosmo_funcs = cosmo_funcs

    @staticmethod
    def get_int_params(cosmo_funcs, zz=0):
        """Get Source quatities for integrated power spectra"""
        d = cosmo_funcs.comoving_dist(zz)
        H = cosmo_funcs.H_c(zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz) # dH_dt - deriv wrt to conformal time! - equivalently: (1-(3/2)*cosmo_funcs.Om_m(zz))*H**2
        #OM = cosmo_funcs.Om_m(zz)
        Qm = cosmo_funcs.survey.Q(zz)
        xQm = cosmo_funcs.survey1.Q(zz)

        be = cosmo_funcs.survey.be(zz)
        xbe = cosmo_funcs.survey1.be(zz)

        return d, H, Hp, Qm, xQm, be, xbe

    @staticmethod
    def get_integrand_params(cosmo_funcs, xd):
        """Get parameters that are funcs of xd"""
        # convert comoving distance to redshift
        zzd = cosmo_funcs.d_to_z(xd)
        # get interpolated values
        fd = cosmo_funcs.f(zzd)
        D1d = cosmo_funcs.D(zzd)
        Hd = cosmo_funcs.H_c(zzd)
        OMd = cosmo_funcs.Om_m(zzd)
        return zzd, fd, D1d, Hd, OMd
    
    @staticmethod
    def int_2Dgrid(xd1,xd2,diag_func,off_diag_func,*args,full=True,fast=True,dtype=np.complex128,weights1=None,weights2=None):

        """ 
        Return Integrated 2D grid - only thing that is 2D is int grid.
        Uses symmetry A[i,j] == A[j,i] if full=False and computes a different expression for the diagonal.

        So in the fast case we dont actually compute grid - we just sum
        the main reason for "fast" is to save memory
        """

        grid_size = xd1.size

        if fast:
            mask = np.triu(np.ones((grid_size, grid_size), dtype=bool), k=1) # get top half - exclude diagonal

            i_upper, j_upper = np.where(mask)
            xd1new = xd1[i_upper,0]; xd2new = xd2[0,j_upper]
            weights1new = weights1[i_upper,0]; weights2new = weights2[0,j_upper]
            
            section = off_diag_func(xd1new, xd2new, *args)
            int_grid = np.sum(weights1new*weights2new*section,axis=-1)

            if full:
                mask_lower = np.tril(np.ones((grid_size, grid_size), dtype=bool), k=-1)
                i_lower, j_lower = np.where(mask_lower)
                xd1new = xd1[i_lower,0]; xd2new = xd2[0,j_lower]
                weights1new = weights1[i_lower,0]; weights2new = weights2[0,j_lower]

                section = off_diag_func(xd1new, xd2new, *args)
                int_grid += np.sum(weights1new*weights2new*section,axis=-1) # just sum over last axis
            else:
                # Use symetry A[i,j] == A[j,i]
                int_grid *= 2

            #diagonal part
            int_grid += np.sum(weights1[:,0]**2 *diag_func(xd1[:,0], *args),axis=-1)

            return int_grid

        else:
            # so legacy code really but if you want to explore the 2D integrand!
            # trust me this will get the shape required of the zz k1 broadcast
            if isinstance(args[0],(np.ndarray,list)):
                mu,_,k1,zz = args[:4] # but sometimes we have mu
                int_grid = np.zeros((*(mu*k1*zz).shape[:-1], grid_size, grid_size),dtype=dtype) # can be complex
            else:
                k1,zz = args[1:3]
                int_grid = np.zeros((*(k1*zz).shape[:-1], grid_size, grid_size),dtype=dtype) # can be complex

            # Use symetry A[i,j] == A[j,i]
            mask = np.triu(np.ones((grid_size, grid_size), dtype=bool), k=1) # get top half - exclude diagonal

            i_upper, j_upper = np.where(mask)
            xd1new = xd1[i_upper,0]; xd2new = xd2[0,j_upper]
            
            section = off_diag_func(xd1new, xd2new, *args)
            int_grid[..., i_upper, j_upper] = section

            if full:
                mask_lower = np.tril(np.ones((grid_size, grid_size), dtype=bool), k=-1)
                i_lower, j_lower = np.where(mask_lower)
                xd1new = xd1[i_lower,0]; xd2new = xd2[0,j_lower]

                section = off_diag_func(xd1new, xd2new, *args)
                int_grid[..., i_lower, j_lower] = section
            else:
                # use symmetry A[i,j] == A[j,i]
                int_grid[..., j_upper, i_upper] = section

            #diagonal part
            int_grid[...,np.arange(grid_size),np.arange(grid_size)] = diag_func(xd1[:,0], *args)

            return int_grid
    
    def pk(self,x,zz,zz2=None): # k**-3 scaling for k > 10
        """Integrated terms integrate over all scales after K_MAX we just have K^{-3} power law"""
        if zz2 is None:
            zz2 = zz
            
        K_MAX = self.cosmo_funcs.K_MAX
        if self.cosmo_funcs.nonlin: # for nonlinear power spectrum modelling
            # so we correlated two points at unequal redshift 
            pk_nl = np.sqrt(self.cosmo_funcs.Pk_NL(x,zz))*np.sqrt(self.cosmo_funcs.Pk_NL(x,zz2))

            # at end of pk
            pk_lim = np.sqrt(self.cosmo_funcs.Pk_NL(K_MAX,zz))*np.sqrt(self.cosmo_funcs.Pk_NL(K_MAX,zz2))

            return np.where(x >K_MAX,pk_lim*(x/K_MAX)**(-3),pk_nl)
        
        return np.where(x >K_MAX,self.cosmo_funcs.Pk(K_MAX)*(x/K_MAX)**(-3),self.cosmo_funcs.Pk(x))

    @staticmethod
    def single_int(func, *args, n=128, remove_div=True,**kwargs):
        """Do single integral for RSDxIntegrated term"""

        nodes, weights = np.polynomial.legendre.leggauss(n)  # legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)

        # so limits [0,d]
        if isinstance(args[0],(np.ndarray,list)):# unpack args we need
            cosmo_funcs,_,zz = args[1:4] # but sometimes we have mu
        else:
            cosmo_funcs,_,zz = args[0:3]

        d = cosmo_funcs.comoving_dist(zz)

        # define nodes in comoving distance: for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
        xd_nodes = (d) * (nodes + 1) / 2.0  # sample range [0,d]

        # call term func # so 2D array in kk,xd
        int_grid = func(xd_nodes, *args,**kwargs)

        if remove_div:
            # complete hack to take care of divergence that should not be here
            # at this point it's good enough of me - we can show it is legit compared to the mu terms
            # so remove so nodes and replace them with the value before it diverges
            x = int(n/16) # so remove this many nodes
            int_grid[...,-x:] = int_grid[...,-x][...,np.newaxis] # remove divergence in last axis

        # (x1-x0)/2
        return (d) / 2.0 * np.sum(weights * int_grid, axis=(-1))  # sum over last

    @staticmethod
    def double_int(func, *args, n=128, n2=None,fast=True,**kwargs):
        """Do double integral for IntegratedxIntegrated term
        1. Defines grid using legendre guass
        2. Calls int_2Dgrid which returns 2D grid of integrand values
        or 
        2. If fast int_2Dgrid returns already summed grid
        3. Sums over last two axes and returns the result
        """
        
        # legendre gauss - get nodes and weights for given n
        nodes1, weights1 = np.polynomial.legendre.leggauss(n)
        nodes1 = np.real(nodes1)
        
        if n2 is None:
            n2 = n
            nodes2 = nodes1
            weights2 = weights1
        else:
            nodes2, weights2 = np.polynomial.legendre.leggauss(n2)
            nodes2 = np.real(nodes2)

        # so limits [0,d]
        if isinstance(args[0],(np.ndarray,list)):# unpack args we need
            cosmo_funcs,_,zz = args[1:4] # but sometimes we have mu
        else:
            cosmo_funcs,_,zz = args[0:3]

        d = cosmo_funcs.comoving_dist(zz)

        #  define nodes in comoving distance for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
        xd_nodes1 = (d) * (nodes1 + 1) / 2.0  # sample range [0,d]
        xd_nodes2 = (d) * (nodes2 + 1) / 2.0

        # so for last two axis we need to define the nodes and weights on the grid
        xd1 = xd_nodes1[:, np.newaxis]
        xd2 = xd_nodes2[np.newaxis, :]

        weights1 = weights1[:, np.newaxis]
        weights2 = weights2[np.newaxis, :]

        if fast:
            return ((d) / 2.0) ** 2 *func(xd1,xd2,*args,fast=fast,weights1=weights1,weights2=weights2,**kwargs) # already summed - is memory efficient!
        
        int_grid = func(xd1,xd2,*args,fast=fast,**kwargs) # get back 2D grid
        #  (x1-x0)/2 sum over last 2 axis
        return ((d) / 2.0) ** 2 * np.sum(weights1 * weights2 * int_grid, axis=(-2, -1))


