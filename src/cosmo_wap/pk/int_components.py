import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

class LxL(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxL.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 
    
        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-36*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1**2 + 4*xd1*xd2 + xd2**2)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**4*(xd1 - xd2)**4) - 36*D1d1*D1d2*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G**2*(xd1**2 + 4*xd1*xd2 + xd2**2) - 2*k1**2*xd1*xd2*(xd1 - xd2)**2)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2)**5))/G**3
            
            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 12*D1d1**2*Hd1**4*OMd1**2*Pk*(5*G**2 + 2*k1**2*xd1*xd2)*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(5*G**3*d**2*k1**2)
            
            return expr
        
        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxL.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(216*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(3*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) - k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2))*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**6) + 216*1j*D1d1*D1d2*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(3*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) - k1**2*xd1*xd2*(xd1 - xd2)**2)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2)**5))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 36*1j*D1d1**2*Hd1**4*OMd1**2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)

            return expr
        
        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxL.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(360*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(27*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) - 2*k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2))*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**6) + 360*D1d1*D1d2*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(27*G**4*(xd1**2 + 3*xd1*xd2 + xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(11*xd1**2 + 35*xd1*xd2 + 11*xd2**2) + k1**4*xd1*xd2*(xd1 - xd2)**4)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**7))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 24*D1d1**2*Hd1**4*OMd1**2*Pk*(7*G**2 - 2*k1**2*xd1*xd2)*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(7*G**3*d**2*k1**2)
    
            return expr
        
        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxL.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-252*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2))*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**8) - 252*1j*D1d1*D1d2*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**7))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = -36*1j*D1d1**2*Hd1**4*OMd1**2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxL.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-324*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(1575*G**4*(5*xd1**2 + 18*xd1*xd2 + 5*xd2**2) - 15*G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 218*xd1*xd2 + 61*xd2**2) + 11*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2))*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**8) - 324*D1d1*D1d2*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(1575*G**6*(5*xd1**2 + 18*xd1*xd2 + 5*xd2**2) - 60*G**4*k1**2*(xd1 - xd2)**2*(59*xd1**2 + 212*xd1*xd2 + 59*xd2**2) + 3*G**2*k1**4*(xd1 - xd2)**4*(47*xd1**2 + 168*xd1*xd2 + 47*xd2**2) - 2*k1**6*xd1*xd2*(xd1 - xd2)**6)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**9*(xd1 - xd2)**9))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 72*D1d1**2*Hd1**4*OMd1**2*Pk*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(35*G**3*d**2)
    
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

class LxTD(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxTD.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-36*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(d*(xd1 + xd2) - 2*xd1*xd2)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2)**3) - 36*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(d*(xd1 + xd2) - 2*xd1*xd2)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**4*(xd1 - xd2)**2))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = -12*D1d1**2*Hd1**4*OMd1**2*Pk*(Qm - 1)*(xQm - 1)*(-d*(xd1 + xd2) + xd1**2 + xd2**2)/(G*d**2*k1**2)

            return expr
        
        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxTD.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-108*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-3*d*(xd1 + xd2) + xd1**2 + 4*xd1*xd2 + xd2**2)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2)**3) + 108*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-G**2*(xd1**2 + 4*xd1*xd2 + xd2**2) - d*(-3*G**2 + k1**2*(xd1 - xd2)**2)*(xd1 + xd2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**4))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 36*1j*D1d1**2*Hd1**4*OMd1**2*Pk*(Qm - 1)*(xQm - 1)*(xd1 - xd2)/(d**2*k1**3)

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxTD.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(180*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-9*G**2*(xd1 + xd2)**2 + d*(18*G**2 - 7*k1**2*(xd1 - xd2)**2)*(xd1 + xd2) + k1**2*(xd1 - xd2)**2*(3*xd1**2 + 8*xd1*xd2 + 3*xd2**2))*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**5) + 180*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-9*G**2*(xd1 + xd2)**2 - d*(-18*G**2 + k1**2*(xd1 - xd2)**2)*(xd1 + xd2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**4))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 12*D1d1**2*Hd1**4*OMd1**2*Pk*(Qm - 1)*(xQm - 1)*(-d*(xd1 + xd2) + xd1**2 + xd2**2)/(G*d**2*k1**2)

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxTD.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-252*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-30*G**2*(3*xd1**2 + 4*xd1*xd2 + 3*xd2**2) + d*(150*G**2 - 13*k1**2*(xd1 - xd2)**2)*(xd1 + xd2) + 2*k1**2*(xd1 - xd2)**2*(3*xd1**2 + 7*xd1*xd2 + 3*xd2**2))*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**5) - 252*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-30*G**4*(3*xd1**2 + 4*xd1*xd2 + 3*xd2**2) + 18*G**2*k1**2*(xd1 - xd2)**2*(2*xd1**2 + 3*xd1*xd2 + 2*xd2**2) + d*(xd1 + xd2)*(150*G**4 - 63*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4) - 2*k1**4*xd1*xd2*(xd1 - xd2)**4)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**6))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 0

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxTD.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-324*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-1050*G**4*(xd1**2 + xd1*xd2 + xd2**2) + 30*G**2*k1**2*(xd1 - xd2)**2*(15*xd1**2 + 16*xd1*xd2 + 15*xd2**2) + 3*d*(xd1 + xd2)*(525*G**4 - 230*G**2*k1**2*(xd1 - xd2)**2 + 7*k1**4*(xd1 - xd2)**4) - 2*k1**4*(xd1 - xd2)**4*(5*xd1**2 + 11*xd1*xd2 + 5*xd2**2))*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**9*(xd1 - xd2)**7) - 324*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(-1050*G**4*(xd1**2 + xd1*xd2 + xd2**2) + 10*G**2*k1**2*(xd1 - xd2)**2*(10*xd1**2 + 13*xd1*xd2 + 10*xd2**2) + d*(xd1 + xd2)*(1575*G**4 - 165*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4) - 2*k1**4*xd1*xd2*(xd1 - xd2)**4)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**6))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 0

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

class LxISW(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxISW.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(6*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(3*Hd1*xd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + 3*Hd2*xd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.sin(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**5*(-xd1 + xd2)**3) - 6*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(3*Hd1*xd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + 3*Hd2*xd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**4*(xd1 - xd2)**2))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 2*D1d1**2*Hd1**5*OMd1**2*Pk*(fd1 - 1)*(3*xd1*(Qm - 1)*(d - xd1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2) + 3*xd2*(d - xd2)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2))/(G*d*k1**2)

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxISW.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(54*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(-Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(2*xd1 + xd2)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) - Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(xd1 + 2*xd2)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**5*(-xd1 + xd2)**3) + 18*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(-3*Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(-G**2*(2*xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + 3*Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(-G**2*(xd1 + 2*xd2) + k1**2*xd2*(xd1 - xd2)**2)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*np.sin(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**6*(xd1 - xd2)**4))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 6*1j*D1d1**2*Hd1**5*OMd1**2*Pk*(fd1 - 1)*(-3*(Qm - 1)*(d - xd1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2) + 3*(d - xd2)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2))/(d*k1**3)

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxISW.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(90*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(-9*G**2*(xd1 + xd2) + k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(-9*G**2*(xd1 + xd2) + k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2))*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.sin(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**7*(-xd1 + xd2)**5) + 90*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(-Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(-9*G**2*(xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) - Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(-9*G**2*(xd1 + xd2) + k1**2*xd2*(xd1 - xd2)**2)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**6*(xd1 - xd2)**4))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 2*D1d1**2*Hd1**5*OMd1**2*Pk*(fd1 - 1)*(-3*xd1*(Qm - 1)*(d - xd1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2) - 3*xd2*(d - xd2)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2))/(G*d*k1**2)
    
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxISW.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(126*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(-Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(-30*G**2*(2*xd1 + 3*xd2) + k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) - Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(-30*G**2*(3*xd1 + 2*xd2) + k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**7*(-xd1 + xd2)**5) - 126*1j*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(30*G**4*(2*xd1 + 3*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(30*G**4*(3*xd1 + 2*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.sin(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**8*(xd1 - xd2)**6))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 0

            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(LxISW.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-54*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(-Hd1*(1 - fd1)*(3 - 3*xQm)*(d - xd2)*(525*G**4*(xd1 + 2*xd2) - 30*G**2*k1**2*(xd1 - xd2)**2*(8*xd1 + 15*xd2) + k1**4*(xd1 - xd2)**4*(11*xd1 + 10*xd2))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + Hd2*(1 - fd2)*(3 - 3*Qm)*(d - xd1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)*(525*G**4*(2*xd1 + xd2) - 30*G**2*k1**2*(xd1 - xd2)**2*(15*xd1 + 8*xd2) + k1**4*(xd1 - xd2)**4*(10*xd1 + 11*xd2)))*np.sin(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**9*(-xd1 + xd2)**7) - 162*D1d1*D1d2*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(Hd1*(d - xd2)*(fd1 - 1)*(xQm - 1)*(525*G**4*(xd1 + 2*xd2) - 5*G**2*k1**2*(xd1 - xd2)**2*(13*xd1 + 20*xd2) + k1**4*xd1*(xd1 - xd2)**4)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + Hd2*(Qm - 1)*(d - xd1)*(fd2 - 1)*(525*G**4*(2*xd1 + xd2) - 5*G**2*k1**2*(xd1 - xd2)**2*(20*xd1 + 13*xd2) + k1**4*xd2*(xd1 - xd2)**4)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**8*(xd1 - xd2)**6))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 0
        
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
class TDxTD(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(TDxTD.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = -36*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2))

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 36*D1d1**2*G*Hd1**4*OMd1**2*Pk*(Qm - 1)*(xQm - 1)/(d**2*k1**4)
    
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(TDxTD.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(108*1j*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**2) + 108*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**5*(xd1 - xd2)))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
    
        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(TDxTD.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(540*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**6*(xd1 - xd2)**2) + 180*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(3*G**2 - k1**2*(xd1 - xd2)**2)*(Qm - 1)*(xQm - 1)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**3))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
    
        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(TDxTD.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-756*1j*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*(Qm - 1)*(xQm - 1)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**4) - 252*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(15*G**2 - k1**2*(xd1 - xd2)**2)*(Qm - 1)*(xQm - 1)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**7*(xd1 - xd2)**3))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
    
        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(TDxTD.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-1620*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(Qm - 1)*(xQm - 1)*np.cos(k1*(-xd1 + xd2)/G)/(d**2*k1**8*(xd1 - xd2)**4) - 324*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(Qm - 1)*(xQm - 1)*(105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4)*np.sin(k1*(-xd1 + xd2)/G)/(d**2*k1**9*(xd1 - xd2)**5))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
class ISWxISW(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = -9*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**5*(xd1 - xd2))

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # can remove

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 9*D1d1**2*G*Hd1**6*OMd1**2*Pk*(fd1 - 1)**2*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)/(H**4*d**2*k1**4)
    
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(27*1j*D1d1*D1d2*G**6*Hd1**3*Hd2**3*OMd1*OMd2*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**6*(xd1 - xd2)**2) + 27*1j*D1d1*D1d2*G**5*Hd1**3*Hd2**3*OMd1*OMd2*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**5*(xd1 - xd2)))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(135*D1d1*D1d2*G**6*Hd1**3*Hd2**3*OMd1*OMd2*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**6*(xd1 - xd2)**2) + 45*D1d1*D1d2*G**5*Hd1**3*Hd2**3*OMd1*OMd2*(3*G**2 - k1**2*(xd1 - xd2)**2)*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(xd1 - xd2)**3))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-189*1j*D1d1*D1d2*G**6*Hd1**3*Hd2**3*OMd1*OMd2*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**4) - 63*1j*D1d1*D1d2*G**5*Hd1**3*Hd2**3*OMd1*OMd2*(15*G**2 - k1**2*(xd1 - xd2)**2)*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(xd1 - xd2)**3))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-405*D1d1*D1d2*G**6*Hd1**3*Hd2**3*OMd1*OMd2*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**4) - 81*D1d1*D1d2*G**5*Hd1**3*Hd2**3*OMd1*OMd2*(fd1 - 1)*(fd2 - 1)*(105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**9*(xd1 - xd2)**5))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
class TDxISW(BaseInt):
    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        @staticmethod
        def l0_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 9*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.sin(k1*(-xd1 + xd2)/G)/(d*k1**5*(-xd1 + xd2))

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l0_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1)

            G = (xd1) / (d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = 18*D1d1**2*G*Hd1**5*OMd1**2*Pk*(fd1 - 1)*(H**2*d*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 4*H*(Qm - 1)*(xQm - 1) - Hp*d*(Qm + xQm - 2))/(H**2*d**2*k1**4)
    
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,l0_terms2, l0_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well

    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l1_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(27*1j*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.sin(k1*(-xd1 + xd2)/G)/(d*k1**6*(xd1 - xd2)**2) - 27*1j*D1d1*D1d2*G**5*Hd1**2*Hd2**2*OMd1*OMd2*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.cos(k1*(-xd1 + xd2)/G)/(d*k1**5*(-xd1 + xd2)))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l1_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l1_terms2, l1_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l2_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(45*D1d1*D1d2*G**7*Hd1**2*Hd2**2*OMd1*OMd2*(-3 + k1**2*(xd1 - xd2)**2/G**2)*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.sin(k1*(-xd1 + xd2)/G)/(d*k1**7*(-xd1 + xd2)**3) + 135*D1d1*D1d2*G**6*Hd1**2*Hd2**2*OMd1*OMd2*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.cos(k1*(-xd1 + xd2)/G)/(d*k1**6*(xd1 - xd2)**2))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l2_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l2_terms2, l2_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l3_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(-189*1j*D1d1*D1d2*G**8*Hd1**2*Hd2**2*OMd1*OMd2*(5 - 2*k1**2*(xd1 - xd2)**2/G**2)*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.sin(k1*(-xd1 + xd2)/G)/(d*k1**8*(xd1 - xd2)**4) - 63*1j*D1d1*D1d2*G**7*Hd1**2*Hd2**2*OMd1*OMd2*(-15 + k1**2*(xd1 - xd2)**2/G**2)*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.cos(k1*(-xd1 + xd2)/G)/(d*k1**7*(-xd1 + xd2)**3))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l3_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l3_terms2, l3_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=0, n=128, n2=None, fast=True):
        return BaseInt.double_int(ISWxISW.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)

    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)

        # for when xd1 != xd2
        @staticmethod
        def l4_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=0):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)

            G = (xd1 + xd2) / (2 * d)
            Pk = baseint.pk(k1/G, zzd1, zzd2)

            expr = Pk*(81*D1d1*D1d2*G**9*Hd1**2*Hd2**2*OMd1*OMd2*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*(105 - 45*k1**2*(xd1 - xd2)**2/G**2 + k1**4*(xd1 - xd2)**4/G**4)*np.sin(k1*(-xd1 + xd2)/G)/(d*k1**9*(-xd1 + xd2)**5) + 405*D1d1*D1d2*G**8*Hd1**2*Hd2**2*OMd1*OMd2*(-21 + 2*k1**2*(xd1 - xd2)**2/G**2)*(2*Hd1*(fd1 - 1)*(xQm - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2))*np.cos(k1*(-xd1 + xd2)/G)/(d*k1**8*(xd1 - xd2)**4))/G**3

            return expr

        # for when xd1 = xd2
        @staticmethod
        def l4_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=0):
            return 0
        
        return BaseInt.int_2Dgrid(xd1,xd2,l4_terms2, l4_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well