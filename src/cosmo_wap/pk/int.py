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
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd, d):
            return (d + xd) / (2 * d)
            
        expr = -6*D1*D1d*Hd**2*OMd*baseint.pk(k1/G_expr(xd, d),zzd1)*(G_expr(xd, d)*(2*H*xd*(Qm - 1)*(2*G_expr(xd, d)**2*f*(8*H - Hd*fd + Hd) - b1*k1**2*xd**2*(H - Hd*fd + Hd) + f*k1**2*xd**2*(-7*H + Hd*(fd - 1))) + Hd*d**4*k1**2*(b1 + f)*(fd - 1)*(2*H**2*Qm - H**2*be + Hp) + d**3*k1**2*(-Hd*b1*(fd - 1)*(-3*H**2*xd*(-2*Qm + be) + 2*H*(Qm - 1) + 3*Hp*xd) + f*(H**2*(3*Hd*be*xd*(fd - 1) + Qm*(-6*Hd*xd*(fd - 1) + 4) - 4) - 2*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1))) + d**2*(-2*G_expr(xd, d)**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) + H**2*Hd*be*(fd - 1)*(2*G_expr(xd, d)**2*f - 3*b1*k1**2*xd**2 - 3*f*k1**2*xd**2) + b1*k1**2*xd*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 2) + 2) + 6*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1)) + f*k1**2*xd*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 22) + 22) + 6*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1))) + d*(2*G_expr(xd, d)**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + b1*k1**2*xd**2*(H**2*(Hd*be*xd*(fd - 1) + Qm*(-2*Hd*xd*(fd - 1) + 4) - 4) - 6*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*xd*(fd - 1)) + f*k1**2*xd**2*(H**2*(Hd*be*xd*(fd - 1) + Qm*(-2*Hd*xd*(fd - 1) + 32) - 32) - 6*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*xd*(fd - 1))))*np.sin(k1*(d - xd)/G_expr(xd, d)) - 2*k1*(d - xd)*(H**2*d**3*k1**2*(Qm - 1)*(b1 + f) + H*xd*(Qm - 1)*(2*G_expr(xd, d)**2*f*(8*H - Hd*fd + Hd) - 2*H*b1*k1**2*xd**2 - 2*H*f*k1**2*xd**2) + d**2*(G_expr(xd, d)**2*H**2*Hd*be*f*(fd - 1) - G_expr(xd, d)**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) - 4*H**2*b1*k1**2*xd*(Qm - 1) - 4*H**2*f*k1**2*xd*(Qm - 1)) + d*(G_expr(xd, d)**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 5*H**2*b1*k1**2*xd**2*(Qm - 1) + 5*H**2*f*k1**2*xd**2*(Qm - 1)))*np.cos(k1*(d - xd)/G_expr(xd, d)))/(G_expr(xd, d)*H**2*d*k1**5*(d - xd)**4)
        return expr
    
    
class IntInt(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None):
        return BaseInt.double_int(IntInt.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2)
        
    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
        
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)

        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            expr = D1d1*D1d2*(-6*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(6*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1**2 + 4*xd1*xd2 + xd2**2) + xd1*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**2*d**2*k1**4*(xd1 - xd2)**4) + 3*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(2*G_expr(xd1, xd2, d)**2*H**2*xd1*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 2*G_expr(xd1, xd2, d)**2*H**2*xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-G_expr(xd1, xd2, d)**2*(xd1**2 + 4*xd1*xd2 + xd2**2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**5*(-xd1 + xd2)**5))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions

            expr = D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(45*G_expr(xd1, xd2, d)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 60*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G_expr(xd1, xd2, d))/(5*G_expr(xd1, xd2, d)**3*H**4*d**2*k1**4)
            
            return expr
            
        return BaseInt.int_2Dgrid(xd1,xd2,cosmo_funcs, k1, zz,int_terms2,int_terms1) # parse functions as well
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None):
        return BaseInt.double_int(IntInt.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2)
        
    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            expr = 9*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*(1j*G_expr(xd1, xd2, d)*(-24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(G_expr(xd1, xd2, d)**2*(xd1 + 2*xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(G_expr(xd1, xd2, d)**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-G_expr(xd1, xd2, d)**2*(2*xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))**2 + k1*(xd1 - xd2)*(2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(xd1 + 2*xd2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(2*xd1 + xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*xd1*xd2*(xd1 - xd2)**2))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))**2 + (G_expr(xd1, xd2, d)*(-24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(G_expr(xd1, xd2, d)**2*(xd1 + 2*xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(G_expr(xd1, xd2, d)**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-G_expr(xd1, xd2, d)**2*(2*xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2))) + 1j*k1*(xd1 - xd2)*(2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(xd1 + 2*xd2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(2*xd1 + xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*xd1*xd2*(xd1 - xd2)**2)))*np.sin(2*k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/2)*baseint.pk(k1/G_expr(xd1, xd2, d),zzd1,zzd2)*np.exp(1j*k1*(xd1 - xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**6*(xd1 - xd2)**6)

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            _, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions

            expr = D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(45*G_expr(xd1, xd2, d)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 60*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 10*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G_expr(xd1, xd2, d),zzd1)/(5*G_expr(xd1, xd2, d)**3*H**4*d**2*k1**4)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,cosmo_funcs, k1, zz,int_terms2,int_terms1) # parse functions as well
    
    @staticmethod
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None):
        return BaseInt.double_int(IntInt.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2)
        
    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            expr = D1d1*D1d2*(-15*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-27*G_expr(xd1, xd2, d)**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) - 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(3*G_expr(xd1, xd2, d)**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**6*(xd1 - xd2)**6) + 15*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(9*G_expr(xd1, xd2, d)**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G_expr(xd1, xd2, d)**2*(3*G_expr(xd1, xd2, d)**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(27*G_expr(xd1, xd2, d)**4*(xd1**2 + 3*xd1*xd2 + xd2**2) - G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(11*xd1**2 + 35*xd1*xd2 + 11*xd2**2) + k1**4*xd1*xd2*(xd1 - xd2)**4))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions

            expr = 2*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(84*G_expr(xd1, xd2, d)**2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 7*G_expr(xd1, xd2, d)**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 7*G_expr(xd1, xd2, d)**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**2*k1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G_expr(xd1, xd2, d))/(7*G_expr(xd1, xd2, d)**3*H**2*d**2*k1**2)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,cosmo_funcs, k1, zz,int_terms2,int_terms1) # parse functions as well
    
    @staticmethod
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None):
        return BaseInt.double_int(IntInt.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2)
        
    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            expr = D1d1*D1d2*(-21*1j*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G_expr(xd1, xd2, d)**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d))*(30*G_expr(xd1, xd2, d)**4*(3*xd1 + 2*xd2) - 9*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4) + (xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(9*G_expr(xd1, xd2, d)**2*(5*G_expr(xd1, xd2, d)**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(3 - 3*xQm)*(d - xd2)*(30*G_expr(xd1, xd2, d)**4*(2*xd1 + 3*xd2) - 9*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**8*(xd1 - xd2)**8) - 21*1j*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(30*G_expr(xd1, xd2, d)**2*(3*xd1 + 2*xd2) - k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(30*G_expr(xd1, xd2, d)**2*(2*xd1 + 3*xd2) - k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G_expr(xd1, xd2, d)**2*(15*G_expr(xd1, xd2, d)**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G_expr(xd1, xd2, d)**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2 
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions

            expr = 2*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(84*G_expr(xd1, xd2, d)**2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 7*G_expr(xd1, xd2, d)**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 7*G_expr(xd1, xd2, d)**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**2*k1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G_expr(xd1, xd2, d))/(7*G_expr(xd1, xd2, d)**3*H**2*d**2*k1**2)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,cosmo_funcs, k1, zz,int_terms2,int_terms1) # parse functions as well
    
    @staticmethod
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None):
        return BaseInt.double_int(IntInt.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2)
        
    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz)
        
        def G_expr(xd1, xd2, d):
            return (xd1 + xd2) / (2 * d)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            expr = D1d1*D1d2*(-21*1j*G_expr(xd1, xd2, d)**4*Hd1**2*Hd2**2*OMd1*OMd2*(12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G_expr(xd1, xd2, d)**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d))*(30*G_expr(xd1, xd2, d)**4*(3*xd1 + 2*xd2) - 9*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4) + (xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(9*G_expr(xd1, xd2, d)**2*(5*G_expr(xd1, xd2, d)**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(3 - 3*xQm)*(d - xd2)*(30*G_expr(xd1, xd2, d)**4*(2*xd1 + 3*xd2) - 9*G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)))*np.sin(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**8*(xd1 - xd2)**8) - 21*1j*G_expr(xd1, xd2, d)**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(30*G_expr(xd1, xd2, d)**2*(3*xd1 + 2*xd2) - k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G_expr(xd1, xd2, d)**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(30*G_expr(xd1, xd2, d)**2*(2*xd1 + 3*xd2) - k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G_expr(xd1, xd2, d)**2*(15*G_expr(xd1, xd2, d)**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G_expr(xd1, xd2, d)**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G_expr(xd1, xd2, d)**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G_expr(xd1, xd2, d))/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G_expr(xd1, xd2, d))/G_expr(xd1, xd2, d)**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions

            expr = 9*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(525*G_expr(xd1, xd2, d)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 1260*G_expr(xd1, xd2, d)**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 70*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 70*G_expr(xd1, xd2, d)**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 136*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G_expr(xd1, xd2, d))/(70*G_expr(xd1, xd2, d)**3*H**4*d**2*k1**4)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,cosmo_funcs, k1, zz,int_terms2,int_terms1) # parse functions as well
