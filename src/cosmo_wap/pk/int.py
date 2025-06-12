import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

""" To Do:
Remove function calls: cosmo_funcs.Pk() and G_expr(xd1, xd2, d) from expr
Simplify int_terms2 as xd1==xd2
"""    
class IntNPP(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l0_integrand, cosmo_funcs, k1, zz=zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz)
        _, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd, d):
            return (d + xd) / (2 * d)
            
        expr = -6*D1*D1d*Hd**2*OMd*baseint.pk(k1/G_expr(xd, d))*(G_expr(xd, d)*(2*H*xd*(Qm - 1)*(2*G_expr(xd, d)**2*f*(8*H - Hd*fd + Hd) - b1*k1**2*xd**2*(H - Hd*fd + Hd) + f*k1**2*xd**2*(-7*H + Hd*(fd - 1))) + Hd*d**4*k1**2*(b1 + f)*(fd - 1)*(2*H**2*Qm - H**2*be + Hp) + d**3*k1**2*(-Hd*b1*(fd - 1)*(-3*H**2*xd*(-2*Qm + be) + 2*H*(Qm - 1) + 3*Hp*xd) + f*(H**2*(3*Hd*be*xd*(fd - 1) + Qm*(-6*Hd*xd*(fd - 1) + 4) - 4) - 2*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1))) + d**2*(-2*G_expr(xd, d)**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) + H**2*Hd*be*(fd - 1)*(2*G_expr(xd, d)**2*f - 3*b1*k1**2*xd**2 - 3*f*k1**2*xd**2) + b1*k1**2*xd*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 2) + 2) + 6*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1)) + f*k1**2*xd*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 22) + 22) + 6*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1))) + d*(2*G_expr(xd, d)**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + b1*k1**2*xd**2*(H**2*(Hd*be*xd*(fd - 1) + Qm*(-2*Hd*xd*(fd - 1) + 4) - 4) - 6*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*xd*(fd - 1)) + f*k1**2*xd**2*(H**2*(Hd*be*xd*(fd - 1) + Qm*(-2*Hd*xd*(fd - 1) + 32) - 32) - 6*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*xd*(fd - 1))))*np.sin(k1*(d - xd)/G_expr(xd, d)) - 2*k1*(d - xd)*(H**2*d**3*k1**2*(Qm - 1)*(b1 + f) + H*xd*(Qm - 1)*(2*G_expr(xd, d)**2*f*(8*H - Hd*fd + Hd) - 2*H*b1*k1**2*xd**2 - 2*H*f*k1**2*xd**2) + d**2*(G_expr(xd, d)**2*H**2*Hd*be*f*(fd - 1) - G_expr(xd, d)**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) - 4*H**2*b1*k1**2*xd*(Qm - 1) - 4*H**2*f*k1**2*xd*(Qm - 1)) + d*(G_expr(xd, d)**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 5*H**2*b1*k1**2*xd**2*(Qm - 1) + 5*H**2*f*k1**2*xd**2*(Qm - 1)))*np.cos(k1*(d - xd)/G_expr(xd, d)))/(G_expr(xd, d)*H**2*d*k1**5*(d - xd)**4)
        return expr
    
    
class IntInt(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n1=128, n2=None):
        return BaseInt.double_int(IntInt.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n1=n1, n2=n2)
        
    @staticmethod    
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
        
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
        _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)

        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1*D1d2*(-18*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xd1**2 + 4*xd1*xd2 + xd2**2) - xd1*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - xd2*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**2*d**2*k1**4*(xd1 - xd2)**4) + 3*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(2*G_expr(xd1, xd2, d)**2*H**2*xd1*(3 - 3*Qm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 2*G_expr(xd1, xd2, d)**2*H**2*xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 12*H**4*(Qm - 1)**2*(d - xd1)*(d - xd2)*(-G_expr(xd1, xd2, d)**2*(xd1**2 + 4*xd1*xd2 + xd2**2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**5*(-xd1 + xd2)**5))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2 # can simplify this slightly as xd1==xd2 by definition
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1**2*Hd1**4*OMd1**2*(45*G_expr(xd1, xd2, d)**4*(2*H**2*(Qm - 1) + Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))**2 + 60*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)**2*(d - xd1)**2 + 20*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1**2*(Qm - 1)**2*(d - xd1)**2)*baseint.pk(k1/G_expr(xd1, xd2, d))/(5*G_expr(xd1, xd2, d)**3*H**4*d**2*k1**4)
            
            return expr

        #lets make this more efficient # removes some redundancy # only thing that is 2D is int grid
        if True: # Use symetry
            grid_size = xd1.size
            # trust me this will get the shape required of the zz k1 broadcast
            int_grid = np.zeros((*(k1*zz).shape[:-1], grid_size, grid_size))
            
            #diagonal part
            _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1[:,0])
            _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1[:,0])
            int_grid[...,np.arange(grid_size),np.arange(grid_size)] = int_terms2(xd1[:,0], cosmo_funcs, k1, zz)#[...,0]
            
            # not the rest...
            mask_upper = np.triu(np.ones((grid_size, grid_size), dtype=bool), k=1) # get top half - exclude diagonal 
            #like mesh grid - get flattened list of indices for top half
            i_upper, j_upper = np.where(mask_upper)

            xd1new = xd1[i_upper,0]; xd2new = xd2[0,j_upper]
            
            _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1new)
            _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2new)
            section = int_terms1(xd1new, xd2new, cosmo_funcs, k1, zz)
            # res = 2*np.sum(section)
            int_grid[..., i_upper, j_upper] = section
            int_grid[..., j_upper, i_upper] = section
        else:
            grid_size = xd1.size

            int_grid[..., j_upper, i_upper] = int_terms1(xd1, xd2, cosmo_funcs, k1, zz)

            int_grid[...,np.arange(grid_size),np.arange(grid_size)] = int_terms2(xd1, cosmo_funcs, k1, zz)[...,0]
            
        return int_grid
    
    @staticmethod
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n1=128, n2=None):
        return BaseInt.double_int(IntInt.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n1=n1, n2=n2)
        
    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
        _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1*D1d2*(-15*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(24*H**4*(Qm - 1)**2*(d - xd1)*(d - xd2)*(-27*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) - 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 2*H**2*(Qm - 1)*(d - xd2)*(-9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**6*(xd1 - xd2)**6) + 15*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2))*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd2)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G_expr(xd1, xd2, d)**2*(3*G_expr(xd1, xd2, d)**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**4*(Qm - 1)**2*(d - xd1)*(d - xd2)*(27*G_expr(xd1, xd2, d)**4*(xd1**2 + 3*xd1*xd2 + xd2**2) - G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(11*xd1**2 + 35*xd1*xd2 + 11*xd2**2) + k1**4*xd1*xd2*(xd1 - xd2)**4))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2 # can simplify this slightly as xd1==xd2 by definition
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1**2*Hd1**4*OMd1**2*(45*G_expr(xd1, xd2, d)**4*(2*H**2*(Qm - 1) + Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))**2 + 60*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)**2*(d - xd1)**2 + 20*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1**2*(Qm - 1)**2*(d - xd1)**2)*baseint.pk(k1/G_expr(xd1, xd2, d))/(5*G_expr(xd1, xd2, d)**3*H**4*d**2*k1**4)
            
            return expr

        #lets make this more efficient # removes some redundancy # only thing that is 2D is int grid
        if True: # Use symetry
            grid_size = xd1.size
            # trust me this will get the shape required of the zz k1 broadcast
            int_grid = np.zeros((*(k1*zz).shape[:-1], grid_size, grid_size))
            
            #diagonal part
            _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1[:,0])
            _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1[:,0])
            int_grid[...,np.arange(grid_size),np.arange(grid_size)] = int_terms2(xd1[:,0], cosmo_funcs, k1, zz)#[...,0]
            
            # not the rest...
            mask_upper = np.triu(np.ones((grid_size, grid_size), dtype=bool), k=1) # get top half - exclude diagonal 
            #like mesh grid - get flattened list of indices for top half
            i_upper, j_upper = np.where(mask_upper)

            xd1new = xd1[i_upper,0]; xd2new = xd2[0,j_upper]
            
            _, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1new)
            _, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2new)
            section = int_terms1(xd1new, xd2new, cosmo_funcs, k1, zz)
            # res = 2*np.sum(section)
            int_grid[..., i_upper, j_upper] = section
            int_grid[..., j_upper, i_upper] = section
        else:
            grid_size = xd1.size

            int_grid[..., j_upper, i_upper] = int_terms1(xd1, xd2, cosmo_funcs, k1, zz)

            int_grid[...,np.arange(grid_size),np.arange(grid_size)] = int_terms2(xd1, cosmo_funcs, k1, zz)[...,0]
            
        return int_grid
