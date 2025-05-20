import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils
    
class IntRSD(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntRSD.l0_integrand, cosmo_funcs, k1, zz=zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        b1, xb1, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        k1, Pk, _, _, d, f, D1 = cosmo_funcs.get_params_pk(k1, zz)
        zzd, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd)
        
        def G_expr(xd):
            return (d + xd) / (2 * d)
            
        expr = (3 * D1 * Hd**2 * OMd * (-1j * G_expr(xd) ** 3 * Hd * b1 * k1**2 * (d - xd) ** 3 * (fd - 1) * (4 * H * (Qm - 1) + d * (3 * OM - 4 * Qm + 2 * be - 2)) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) + G_expr(xd) ** 3 * Hd * f * (d - xd) * (fd - 1) * (4 * H * (Qm - 1) + d * (3 * OM - 4 * Qm + 2 * be - 2)) * (2 * 1j * G_expr(xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) + 2 * G_expr(xd) * k1 * (d - xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) + 1) - 1j * k1**2 * (d - xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1)) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) - 4 * 1j * G_expr(xd) ** 3 * b1 * k1**2 * (Qm - 1) * (d - xd) ** 3 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) + 4 * G_expr(xd) ** 3 * f * (Qm - 1) * (d - xd) * (2 * 1j * G_expr(xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) + 2 * G_expr(xd) * k1 * (d - xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) + 1) - 1j * k1**2 * (d - xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1)) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) + 4 * G_expr(xd) ** 2 * b1 * k1**2 * (Qm - 1) * (d - xd) ** 3 * (1j * G_expr(xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) + k1 * (d - xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) + 1)) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) - 2 * 1j * G_expr(xd) * b1 * k1**4 * xd * (Qm - 1) * (d - xd) ** 4 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) + 2 * G_expr(xd) * b1 * k1**2 * xd * (Qm - 1) * (d - xd) ** 2 * (-2 * 1j * G_expr(xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) - 2 * G_expr(xd) * k1 * (d - xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) + 1) + 1j * k1**2 * (d - xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1)) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) + 2 * G_expr(xd) * f * k1**2 * xd * (Qm - 1) * (d - xd) ** 2 * ( 2 * 1j * G_expr(xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1) + 2 * G_expr(xd) * k1 * (d - xd) * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) + 1) - 1j * k1**2 * (d - xd) ** 2 * (np.exp(2 * 1j * k1 * (d - xd) / G_expr(xd)) - 1)) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)) - 4 * 1j * G_expr(xd) * f * (1 - Qm) * (d - xd) * (6 * G_expr(xd) ** 4 - G_expr(xd) * (6 * G_expr(xd) ** 3 + 6 * 1j * G_expr(xd) ** 2 * k1 * (-d + xd) - 3 * G_expr(xd) * k1**2 * (d - xd) ** 2 + 1j * k1**3 * (d - xd) ** 3) * np.exp(1j * k1 * (d - xd) / G_expr(xd)) - G_expr(xd) * (6 * G_expr(xd) ** 3 * (np.exp(1j * k1 * (d - xd) / G_expr(xd)) - 1) - 6 * 1j * G_expr(xd) ** 2 * k1 * (d - xd) + 3 * G_expr(xd) * k1**2 * (d - xd) ** 2 + 1j * k1**3 * (d - xd) ** 3) * np.exp(-1j * k1 * (d - xd) / G_expr(xd))) + 2 * f * xd * (1 - Qm) * (24 * 1j * G_expr(xd) ** 5 + G_expr(xd) * (-24 * 1j * G_expr(xd) ** 4 - 24 * G_expr(xd) ** 3 * k1 * (d - xd) + 12 * 1j * G_expr(xd) ** 2 * k1**2 * (d - xd) ** 2 + 4 * G_expr(xd) * k1**3 * (d - xd) ** 3 - 1j * k1**4 * (d - xd) ** 4) * np.exp(1j * k1 * (d - xd) / G_expr(xd)) + G_expr(xd) * (-24 * 1j * G_expr(xd) ** 4 * (np.exp(1j * k1 * (d - xd) / G_expr(xd)) - 1) - 24 * G_expr(xd) ** 3 * k1 * (d - xd) - 12 * 1j * G_expr(xd) ** 2 * k1**2 * (d - xd) ** 2 + 4 * G_expr(xd) * k1**3 * (d - xd) ** 3 + 1j * k1**4 * (d - xd) ** 4) * np.exp(-1j * k1 * (d - xd) / G_expr(xd)))) * cosmo_funcs.Pk(k1 / G_expr(xd)) / (G_expr(xd) ** 3 * d * k1**5 * (d - xd) ** 4))
        return expr
    
    
class IntInt(BaseInt):
    #need to remove function calls: cosmo_funcs.Pk() and G_expr(xd1, xd2, d) from expr
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n1=128, n2=None):
        return BaseInt.double_int(IntInt.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n1=n1, n2=n2)
        
    @staticmethod    
    def l0_integrand(xd1, xd2, cosmo_funcs, kk, z=0, t=0, sigma=None):
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(kk,z,n=2)
        
        b1, xb1, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        k1, Pk, Pkd, Pkdd, d, f, D1 = cosmo_funcs.get_params_pk(k1, zz)

        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1*D1d2*(-18*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xd1**2 + 4*xd1*xd2 + xd2**2) - xd1*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - xd2*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**2*d**2*k1**4*(xd1 - xd2)**4) + 3*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(2*G_expr(xd1, xd2, d)**2*H**2*xd1*(3 - 3*Qm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 2*G_expr(xd1, xd2, d)**2*H**2*xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 12*H**4*(Qm - 1)**2*(d - xd1)*(d - xd2)*(-G_expr(xd1, xd2, d)**2*(xd1**2 + 4*xd1*xd2 + xd2**2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**5*(-xd1 + xd2)**5))*cosmo_funcs.Pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 = xd2 # can simplify this slightly as xd1==xd2 by definition
        def int_terms2(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

            expr = D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*(45*G_expr(xd1, xd2, d)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 60*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)**2*(d - xd1)*(d - xd2) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(Qm - 1) - Hd2*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd2*(3 - 3*Qm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1*xd2*(Qm - 1)**2*(d - xd1)*(d - xd2))*cosmo_funcs.Pk(k1/G_expr(xd1, xd2, d))/(5*H**4*d**2*k1**4)
            return expr
        

        #lets make this more efficient # removes some redundancy # only thing that is 2D is int grid
        int_grid = int_terms1(xd1, xd2, cosmo_funcs, k1, zz)
        print(int_grid.shape)
        print(xd1.shape)
        print(int_terms2(xd1, xd1, cosmo_funcs, k1, zz).shape)
        #np.fill_diagonal(int_grid, int_terms2(xd1, xd1, cosmo_funcs, k1, zz)[0])
        
        return int_grid
