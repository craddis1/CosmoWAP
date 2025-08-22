import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils
from cosmo_wap.lib import integrate

class LxL(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxL.l0_terms1, LxL.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G*k1*(5*xd1**3 - 21*xd1**2*xd2 + 21*xd1*xd2**2 - 5*xd2**3)*np.cos(k1*(xd1 - xd2)/G) + (G**2*(-5*xd1**2 + 16*xd1*xd2 - 5*xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**5*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 12*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(5*G**2 + 2*k1**2*xd1*xd2)*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(5*G**3*d**2*k1**2)
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxL.l1_terms1, LxL.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 216*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G*(G**2*(9*xd1**2 - 33*xd1*xd2 + 9*xd2**2) - 2*k1**2*(xd1 - xd2)**2*(2*xd1**2 - 7*xd1*xd2 + 2*xd2**2))*np.sin(k1*(xd1 - xd2)/G) + k1*(xd1 - xd2)*(G**2*(-9*xd1**2 + 33*xd1*xd2 - 9*xd2**2) + k1**2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**6*(xd1 - xd2)**6)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -36*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxL.l2_terms1, LxL.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 360*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G*k1*(xd1 - xd2)*(-9*G**2*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) + k1**2*(xd1 - xd2)**2*(7*xd1**2 - 26*xd1*xd2 + 7*xd2**2))*np.cos(k1*(xd1 - xd2)/G) + (9*G**4*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(28*xd1**2 - 113*xd1*xd2 + 28*xd2**2) + k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**7*(xd1 - xd2)**7)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 24*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(7*G**2 - 2*k1**2*xd1*xd2)*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(7*G**3*d**2*k1**2)
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxL.l3_terms1, LxL.l3_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 252*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G*(-150*G**4*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) + 3*G**2*k1**2*(xd1 - xd2)**2*(181*xd1**2 - 822*xd1*xd2 + 181*xd2**2) - k1**4*(xd1 - xd2)**4*(23*xd1**2 - 88*xd1*xd2 + 23*xd2**2))*np.sin(k1*(xd1 - xd2)/G) + k1*(xd1 - xd2)*(150*G**4*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) - 11*G**2*k1**2*(xd1 - xd2)**2*(13*xd1**2 - 56*xd1*xd2 + 13*xd2**2) + 2*k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**8*(xd1 - xd2)**8)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)

    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxL.l4_terms1, LxL.l4_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 324*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(G*k1*(xd1 - xd2)*(1575*G**4*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) - 255*G**2*k1**2*(xd1 - xd2)**2*(7*xd1**2 - 34*xd1*xd2 + 7*xd2**2) + k1**4*(xd1 - xd2)**4*(35*xd1**2 - 136*xd1*xd2 + 35*xd2**2))*np.cos(k1*(xd1 - xd2)/G) + (-1575*G**6*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) + 30*G**4*k1**2*(xd1 - xd2)**2*(217*xd1**2 - 1094*xd1*xd2 + 217*xd2**2) - 3*G**2*k1**4*(xd1 - xd2)**4*(105*xd1**2 - 472*xd1*xd2 + 105*xd2**2) + 2*k1**6*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**9*(xd1 - xd2)**9)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 72*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(35*G**3*d**2)
    
class LxTD(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxTD.l0_terms1, LxTD.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -36*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(-G*k1*(xd1 - xd2)*(d**2*(5*xd1**2 - 16*xd1*xd2 + 5*xd2**2) - 6*d*(xd1**3 - 2*xd1**2*xd2 - 2*xd1*xd2**2 + xd2**3) + 2*xd1**4 - xd1**3*xd2 - 8*xd1**2*xd2**2 - xd1*xd2**3 + 2*xd2**4)*np.cos(k1*(xd1 - xd2)/G) + (G**2*(xd1**4 + 3*xd1**3*xd2 - 14*xd1**2*xd2**2 + 3*xd1*xd2**3 + xd2**4) + d**2*(G**2*(5*xd1**2 - 16*xd1*xd2 + 5*xd2**2) - 2*k1**2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)) - 2*d*(3*G**2 - k1**2*(xd1 - xd2)**2)*(xd1**3 - 2*xd1**2*xd2 - 2*xd1*xd2**2 + xd2**3) - 2*k1**2*xd1*xd2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**5*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 12*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(15*G**4 + 5*G**2*k1**2*(d**2 - xd1**2 + xd1*xd2 - xd2**2) + 2*k1**4*xd1*xd2*(d - xd1)*(d - xd2))/(5*G**3*d**2*k1**4)
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxTD.l1_terms1, LxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 108*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(G*(G**2*(4*xd1**4 + 8*xd1**3*xd2 - 54*xd1**2*xd2**2 + 8*xd1*xd2**3 + 4*xd2**4) + 2*d**2*(G**2*(9*xd1**2 - 33*xd1*xd2 + 9*xd2**2) - 2*k1**2*(xd1 - xd2)**2*(2*xd1**2 - 7*xd1*xd2 + 2*xd2**2)) + 3*d*(xd1 + xd2)*(G**2*(-7*xd1**2 + 24*xd1*xd2 - 7*xd2**2) + k1**2*(xd1 - xd2)**2*(3*xd1**2 - 10*xd1*xd2 + 3*xd2**2)) - 2*k1**2*(xd1 - xd2)**2*(xd1**4 + xd1**3*xd2 - 10*xd1**2*xd2**2 + xd1*xd2**3 + xd2**4))*np.sin(k1*(xd1 - xd2)/G) - k1*(xd1 - xd2)*(G**2*(4*xd1**4 + 8*xd1**3*xd2 - 54*xd1**2*xd2**2 + 8*xd1*xd2**3 + 4*xd2**4) + 2*d**2*(G**2*(9*xd1**2 - 33*xd1*xd2 + 9*xd2**2) - k1**2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)) + d*(xd1 + xd2)*(-3*G**2*(7*xd1**2 - 24*xd1*xd2 + 7*xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)) - 2*k1**2*xd1*xd2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**6*(xd1 - xd2)**6)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -36*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(5*G**2 + k1**2*(d - xd1)*(d - xd2))*(Qm - 1)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1**3)
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxTD.l2_terms1, LxTD.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 180*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(-G*k1*(xd1 - xd2)*(6*G**2*(4*xd1**4 + 11*xd1**3*xd2 - 75*xd1**2*xd2**2 + 11*xd1*xd2**3 + 4*xd2**4) + 2*d**2*(9*G**2*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) - k1**2*(xd1 - xd2)**2*(7*xd1**2 - 26*xd1*xd2 + 7*xd2**2)) + 3*d*(xd1 + xd2)*(-6*G**2*(8*xd1**2 - 31*xd1*xd2 + 8*xd2**2) + k1**2*(xd1 - xd2)**2*(5*xd1**2 - 18*xd1*xd2 + 5*xd2**2)) - 2*k1**2*(xd1 - xd2)**2*(xd1**4 + 4*xd1**3*xd2 - 22*xd1**2*xd2**2 + 4*xd1*xd2**3 + xd2**4))*np.cos(k1*(xd1 - xd2)/G) + (6*G**4*(4*xd1**4 + 11*xd1**3*xd2 - 75*xd1**2*xd2**2 + 11*xd1*xd2**3 + 4*xd2**4) - 2*G**2*k1**2*(xd1 - xd2)**2*(5*xd1**4 + 15*xd1**3*xd2 - 97*xd1**2*xd2**2 + 15*xd1*xd2**3 + 5*xd2**4) + 2*d**2*(9*G**4*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(28*xd1**2 - 113*xd1*xd2 + 28*xd2**2) + k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) - d*(xd1 + xd2)*(18*G**4*(8*xd1**2 - 31*xd1*xd2 + 8*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(21*xd1**2 - 80*xd1*xd2 + 21*xd2**2) + 2*k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**7*(xd1 - xd2)**7)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 12*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(7*G**2*(xd1 + xd2)**2 + 2*d**2*(7*G**2 - 2*k1**2*xd1*xd2) + d*(-21*G**2 + 4*k1**2*xd1*xd2)*(xd1 + xd2) - 4*k1**2*xd1**2*xd2**2)/(7*G**3*d**2*k1**2)
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxTD.l1_terms1, LxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -252*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(G*(15*G**4*(13*xd1**4 + 48*xd1**3*xd2 - 332*xd1**2*xd2**2 + 48*xd1*xd2**3 + 13*xd2**4) - 3*G**2*k1**2*(xd1 - xd2)**2*(28*xd1**4 + 111*xd1**3*xd2 - 738*xd1**2*xd2**2 + 111*xd1*xd2**3 + 28*xd2**4) + d**2*(150*G**4*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(181*xd1**2 - 822*xd1*xd2 + 181*xd2**2) + k1**4*(xd1 - xd2)**4*(23*xd1**2 - 88*xd1*xd2 + 23*xd2**2)) - 6*d*(xd1 + xd2)*(75*G**4*(3*xd1**2 - 13*xd1*xd2 + 3*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(101*xd1**2 - 432*xd1*xd2 + 101*xd2**2) + k1**4*(xd1 - xd2)**4*(4*xd1**2 - 15*xd1*xd2 + 4*xd2**2)) + k1**4*(xd1 - xd2)**4*(xd1**2 + 12*xd1*xd2 + xd2**2)*(2*xd1**2 - 7*xd1*xd2 + 2*xd2**2))*np.sin(k1*(xd1 - xd2)/G) - k1*(xd1 - xd2)*(15*G**4*(13*xd1**4 + 48*xd1**3*xd2 - 332*xd1**2*xd2**2 + 48*xd1*xd2**3 + 13*xd2**4) - G**2*k1**2*(xd1 - xd2)**2*(19*xd1**4 + 93*xd1**3*xd2 - 554*xd1**2*xd2**2 + 93*xd1*xd2**3 + 19*xd2**4) + d**2*(150*G**4*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) - 11*G**2*k1**2*(xd1 - xd2)**2*(13*xd1**2 - 56*xd1*xd2 + 13*xd2**2) + 2*k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) - 2*d*(xd1 + xd2)*(225*G**4*(3*xd1**2 - 13*xd1*xd2 + 3*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(26*xd1**2 - 107*xd1*xd2 + 26*xd2**2) + k1**4*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**8*(xd1 - xd2)**8)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)
    
    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxTD.l1_terms1, LxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 324*D1d1*D1d2*G*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(105*G**4*(19*xd1**4 + 89*xd1**3*xd2 - 636*xd1**2*xd2**2 + 89*xd1*xd2**3 + 19*xd2**4) - 5*G**2*k1**2*(xd1 - xd2)**2*(44*xd1**4 + 247*xd1**3*xd2 - 1602*xd1**2*xd2**2 + 247*xd1*xd2**3 + 44*xd2**4) + d**2*(1575*G**4*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) - 255*G**2*k1**2*(xd1 - xd2)**2*(7*xd1**2 - 34*xd1*xd2 + 7*xd2**2) + k1**4*(xd1 - xd2)**4*(35*xd1**2 - 136*xd1*xd2 + 35*xd2**2)) - 6*d*(xd1 + xd2)*(525*G**4*(5*xd1**2 - 24*xd1*xd2 + 5*xd2**2) - 25*G**2*k1**2*(xd1 - xd2)**2*(13*xd1**2 - 60*xd1*xd2 + 13*xd2**2) + k1**4*(xd1 - xd2)**4*(6*xd1**2 - 23*xd1*xd2 + 6*xd2**2)) + k1**4*(xd1 - xd2)**4*(2*xd1**4 + 29*xd1**3*xd2 - 128*xd1**2*xd2**2 + 29*xd1*xd2**3 + 2*xd2**4))*np.cos(k1*(xd1 - xd2)/G)/(d**2*k1**8*(xd1 - xd2)**8) - 324*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(105*G**6*(19*xd1**4 + 89*xd1**3*xd2 - 636*xd1**2*xd2**2 + 89*xd1*xd2**3 + 19*xd2**4) - 15*G**4*k1**2*(xd1 - xd2)**2*(59*xd1**4 + 290*xd1**3*xd2 - 2018*xd1**2*xd2**2 + 290*xd1*xd2**3 + 59*xd2**4) + G**2*k1**4*(xd1 - xd2)**4*(31*xd1**4 + 233*xd1**3*xd2 - 1314*xd1**2*xd2**2 + 233*xd1*xd2**3 + 31*xd2**4) + d**2*(1575*G**6*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) - 30*G**4*k1**2*(xd1 - xd2)**2*(217*xd1**2 - 1094*xd1*xd2 + 217*xd2**2) + 3*G**2*k1**4*(xd1 - xd2)**4*(105*xd1**2 - 472*xd1*xd2 + 105*xd2**2) - 2*k1**6*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*d*(xd1 + xd2)*(-1575*G**6*(5*xd1**2 - 24*xd1*xd2 + 5*xd2**2) + 900*G**4*k1**2*(xd1 - xd2)**2*(4*xd1**2 - 19*xd1*xd2 + 4*xd2**2) - 3*G**2*k1**4*(xd1 - xd2)**4*(56*xd1**2 - 243*xd1*xd2 + 56*xd2**2) + k1**6*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2)) - 2*k1**6*xd1*xd2*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2))*np.sin(k1*(xd1 - xd2)/G)/(d**2*k1**9*(xd1 - xd2)**9)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 72*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(35*G**3*d**2)

class LxISW(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxISW.l0_terms1, LxISW.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 9*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(2*G*H**2*k1*(xd1 - xd2)*(2*H*(Qm - 1)*(xQm - 1)*(H*xd1*xd2*(5*xd1**2 - 16*xd1*xd2 + 5*xd2**2) - Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2)*(xd1 - xd2)**2 + Hd2*xd1*(fd2 - 1)*(xd1 - xd2)**2*(2*xd1 - xd2)) + d**2*(H**2*(xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(xQm - 1) + Hd1*be*(fd1 - 1)*(xQm - 1) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(-4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(8*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 10*xQm + 10) + xd1*xd2*(5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 2*Qm*(5*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) + 32*xQm - 32) + xd2**2*(-2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + Qm*(4*Hd1*xd2*(fd1 - 1)*(xQm - 1) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 10*xQm + 10)) - Hp*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2))) + d*(H**2*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 5*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 10*xQm - 10) + 2*xd1**2*xd2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 4*Hd2*fd2*xQm*xd2 - 2*Hd2*fd2*xbe*xd2 - 4*Hd2*xQm*xd2 + 2*Hd2*xbe*xd2 + Qm*(-4*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 11*xQm - 11) - 11*xQm + 11) + xd1*xd2**2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(10*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 22*xQm + 22) - 2*xd2**3*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + Qm*(2*Hd1*xd2*(fd1 - 1) + 5) - 5)) + 2*H*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xd1 - 2*xd2) - Hd2*(fd2 - 1)*(2*xd1 - xd2)) - Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2))))*np.cos(k1*(-xd1 + xd2)/G) + (4*H**2*(Qm - 1)*(xQm - 1)*(G**2*(H**2*xd1*xd2*(5*xd1**2 - 16*xd1*xd2 + 5*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2) + Hd2*xd1*(fd2 - 1)*(2*xd1 - xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 2*H**2*k1**2*xd1*xd2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*H*d*(G**2*(H**3*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 5*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 10*xQm - 10) + 2*xd1**2*xd2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 4*Hd2*fd2*xQm*xd2 - 2*Hd2*fd2*xbe*xd2 - 4*Hd2*xQm*xd2 + 2*Hd2*xbe*xd2 + Qm*(-4*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 11*xQm - 11) - 11*xQm + 11) + xd1*xd2**2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(10*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 22*xQm + 22) - 2*xd2**3*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + Qm*(2*Hd1*xd2*(fd1 - 1) + 5) - 5)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + xQm - 1) - xQm + 1) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 4*Qm*(xQm - 1) - 4*xQm + 4)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(2*xd1 - xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + 4*H**3*k1**2*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(xd1**3 - 2*xd1**2*xd2 - 2*xd1*xd2**2 + xd2**3)) + d**2*(G**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 10*xQm + 10) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) - 32*xQm + 32) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) + 4*Hd2*fd2*xQm*xd2 - 2*Hd2*fd2*xbe*xd2 - 4*Hd2*xQm*xd2 + 2*Hd2*xbe*xd2 + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + 10*xQm - 10) - 20*xQm + 20)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + xQm - 1) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 4*xQm - 4)) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 8*H**4*k1**2*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)))*np.sin(k1*(-xd1 + xd2)/G))/(H**4*d**2*k1**5*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 3*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(15*G**4*Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + 10*G**2*H**2*k1**2*(2*H*(Qm - 1)*(xQm - 1)*(-Hd2*xd1**2*(fd2 - 1) + xd2*(H*xd1 - Hd1*xd2*(fd1 - 1))) + d**2*(H**2*(-Hd2*xd1*(fd2 - 1)*(-2*xQm + xbe) + Qm*(Hd2*xd1*(fd2 - 1)*(-2*xQm + xbe) - 2*(xQm - 1)*(Hd1*xd2*(fd1 - 1) - 1)) + (xQm - 1)*(Hd1*be*xd2*(fd1 - 1) - 2)) + Hp*(Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) + Hd2*xd1*(-Qm*fd2 + Qm + fd2 - 1))) + d*(H**2*(-Hd2*xd1**2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1*(-Qm*xQm + Qm + xQm - 1) + xd2*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + 2*Qm*(Hd1*xd2*(fd1 - 1) - 1) + 2)) + 2*H*(Qm - 1)*(xQm - 1)*(Hd1*xd2*(fd1 - 1) + Hd2*xd1*(fd2 - 1)) + Hp*(Hd1*xd2**2*(fd1 - 1)*(xQm - 1) + Hd2*xd1**2*(Qm - 1)*(fd2 - 1)))) + 8*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))/(5*G**3*H**4*d**2*k1**4)
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxISW.l1_terms1, LxISW.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -27*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(G*(4*H**2*(Qm - 1)*(xQm - 1)*(G**2*(6*H**2*xd1*xd2*(3*xd1**2 - 11*xd1*xd2 + 3*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(2*xd1 - 5*xd2) + Hd2*xd1*(fd2 - 1)*(5*xd1 - 2*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - H*k1**2*(xd1 - xd2)**2*(4*H*xd1*xd2*(2*xd1**2 - 7*xd1*xd2 + 2*xd2**2) - Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2)*(xd1 - xd2)**2 + Hd2*xd1*(fd2 - 1)*(xd1 - xd2)**2*(2*xd1 - xd2))) + 2*H*d*(G**2*(H**3*(5*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 6*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(xQm - 1) - 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 9*xQm + 9) + 18*xQm - 18) - 3*xd1**2*xd2*(-3*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 6*Hd2*fd2*xQm*xd2 + 3*Hd2*fd2*xbe*xd2 + 6*Hd2*xQm*xd2 - 3*Hd2*xbe*xd2 + Qm*(6*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 32*xQm + 32) + 32*xQm - 32) + 2*xd1*xd2**2*(-6*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(12*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 48*xQm - 48) - 48*xQm + 48) - xd2**3*(xQm - 1)*(-5*Hd1*be*xd2*(fd1 - 1) + 2*Qm*(5*Hd1*xd2*(fd1 - 1) + 18) - 36)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 2*Qm*(xQm - 1) - 2*xQm + 2) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 10*Qm*(xQm - 1) - 10*xQm + 10)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(5*xd1 - 2*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(2*xd1 - 5*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(5*xd1 - 2*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - H*k1**2*(xd1 - xd2)**2*(H**2*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 10*Hd2*fd2*xQm*xd2 + 5*Hd2*fd2*xbe*xd2 + 10*Hd2*xQm*xd2 - 5*Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 16*xQm + 16) + 16*xQm - 16) - 4*xd1**2*xd2*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 10*xQm + 10) + 10*xQm - 10) + xd1*xd2**2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(10*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) - 40*xQm + 40) - 2*xd2**3*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + 2*Qm*(Hd1*xd2*(fd1 - 1) + 4) - 8)) + 2*H*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xd1 - 2*xd2) - Hd2*(fd2 - 1)*(2*xd1 - xd2)) - Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))) + d**2*(G**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-4*Hd1*Qm*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) + 2*Hd1*be*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) - 5*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1) + 2*Qm*(Hd1*xd2*(fd1 - 1) + 2) - 4)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) - 2*xd1*xd2*(-2*Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) - 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 132*xQm - 132) - 132*xQm + 132) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) + 8*Hd2*fd2*xQm*xd2 - 4*Hd2*fd2*xbe*xd2 - 8*Hd2*xQm*xd2 + 4*Hd2*xbe*xd2 + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 4*Hd2*fd2*xQm*xd2 + 2*Hd2*fd2*xbe*xd2 + 4*Hd2*xQm*xd2 - 2*Hd2*xbe*xd2 + 36*xQm - 36) - 72*xQm + 72)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 2*xQm - 2) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 10*xQm - 10)) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(5*xd1 - 2*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + 2*H**2*k1**2*(xd1 - xd2)**2*(H**2*(xd1**3*(2*Hd1*Qm*(fd1 - 1)*(xQm - 1) + Hd1*be*(-fd1*xQm + fd1 + xQm - 1) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 10*Hd2*fd2*xQm*xd2 + 5*Hd2*fd2*xbe*xd2 + 10*Hd2*xQm*xd2 - 5*Hd2*xbe*xd2 + Qm*(-8*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 16*xQm + 16) + 16*xQm - 16) + xd1*xd2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 8*Hd2*fd2*xQm*xd2 - 4*Hd2*fd2*xbe*xd2 - 8*Hd2*xQm*xd2 + 4*Hd2*xbe*xd2 + 2*Qm*(5*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) - 56*xQm + 56) + xd2**2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(-4*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 16*xQm + 16) + 16*xQm - 16)) + Hp*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))))*np.sin(k1*(-xd1 + xd2)/G) + k1*(xd1 - xd2)*(4*H**2*(Qm - 1)*(xQm - 1)*(G**2*(6*H**2*xd1*xd2*(3*xd1**2 - 11*xd1*xd2 + 3*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(2*xd1 - 5*xd2) + Hd2*xd1*(fd2 - 1)*(5*xd1 - 2*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 2*H**2*k1**2*xd1*xd2*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*H*d*(G**2*(H**3*(5*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 6*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(xQm - 1) - 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 9*xQm + 9) + 18*xQm - 18) - 3*xd1**2*xd2*(-3*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 6*Hd2*fd2*xQm*xd2 + 3*Hd2*fd2*xbe*xd2 + 6*Hd2*xQm*xd2 - 3*Hd2*xbe*xd2 + Qm*(6*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 32*xQm + 32) + 32*xQm - 32) + 2*xd1*xd2**2*(-6*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(12*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 48*xQm - 48) - 48*xQm + 48) - xd2**3*(xQm - 1)*(-5*Hd1*be*xd2*(fd1 - 1) + 2*Qm*(5*Hd1*xd2*(fd1 - 1) + 18) - 36)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 2*Qm*(xQm - 1) - 2*xQm + 2) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 10*Qm*(xQm - 1) - 10*xQm + 10)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(5*xd1 - 2*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(2*xd1 - 5*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(5*xd1 - 2*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + 4*H**3*k1**2*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(xd1**3 - 2*xd1**2*xd2 - 2*xd1*xd2**2 + xd2**3)) + d**2*(G**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-4*Hd1*Qm*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) + 2*Hd1*be*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + xQm - 1) - 5*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1) + 2*Qm*(Hd1*xd2*(fd1 - 1) + 2) - 4)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) - 2*xd1*xd2*(-2*Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) - 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 132*xQm - 132) - 132*xQm + 132) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) + 8*Hd2*fd2*xQm*xd2 - 4*Hd2*fd2*xbe*xd2 - 8*Hd2*xQm*xd2 + 4*Hd2*xbe*xd2 + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 4*Hd2*fd2*xQm*xd2 + 2*Hd2*fd2*xbe*xd2 + 4*Hd2*xQm*xd2 - 2*Hd2*xbe*xd2 + 36*xQm - 36) - 72*xQm + 72)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 2*xQm - 2) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 10*xQm - 10)) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(5*xd1 - 2*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 8*H**4*k1**2*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(xd1**2 - 3*xd1*xd2 + xd2**2)))*np.cos(k1*(-xd1 + xd2)/G))/(H**4*d**2*k1**6*(xd1 - xd2)**6)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -18*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(2*H*(Qm - 1)*(xQm - 1)*(5*G**2*(-Hd1*xd2*(fd1 - 1) + Hd2*xd1*(fd2 - 1)) + H*k1**2*xd1*xd2*(xd1 - xd2)) + d**2*(5*G**2*H**2*Hd1*be*(fd1 - 1)*(xQm - 1) - 5*G**2*(Hd1*(fd1 - 1)*(xQm - 1)*(2*H**2*Qm + Hp) - Hd2*(Qm - 1)*(fd2 - 1)*(-H**2*(-2*xQm + xbe) + Hp)) + 2*H**2*k1**2*(Qm - 1)*(xQm - 1)*(xd1 - xd2)) + d*(5*G**2*(H**2*(-Hd1*xd2*(-2*Qm + be)*(fd1 - 1)*(xQm - 1) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*H*(Qm - 1)*(xQm - 1)*(Hd1*(fd1 - 1) - Hd2*fd2 + Hd2) + Hp*(Hd1*xd2*(fd1 - 1)*(xQm - 1) + Hd2*xd1*(-Qm*fd2 + Qm + fd2 - 1))) - 2*H**2*k1**2*(Qm - 1)*(xQm - 1)*(xd1**2 - xd2**2)))/(5*G**2*H**2*d**2*k1**3)
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxISW.l2_terms1, LxISW.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 45*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(-G*k1*(xd1 - xd2)*(4*H**2*(Qm - 1)*(xQm - 1)*(3*G**2*(6*H**2*xd1*xd2*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) + 3*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xd1 - 3*xd2) + Hd2*xd1*(fd2 - 1)*(3*xd1 - xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - H*k1**2*(xd1 - xd2)**2*(2*H*xd1*xd2*(7*xd1**2 - 26*xd1*xd2 + 7*xd2**2) - Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2)*(xd1 - xd2)**2 + Hd2*xd1*(fd2 - 1)*(xd1 - xd2)**2*(2*xd1 - xd2))) + 2*H*d*(3*G**2*(3*H**3*(3*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 7*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) + 28*xQm - 28) + xd1**2*xd2*(-10*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 88*Qm*(xQm - 1) - 88*xQm + 88) + xd1*xd2**2*(14*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 7*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 88*Qm*(xQm - 1) - 88*xQm + 88) - xd2**3*(xQm - 1)*(-3*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 28*Qm - 28)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 3*xQm + 3) + 3*xQm - 3) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 18*xQm - 18) - 18*xQm + 18)) + 6*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(3*xd1 - xd2)) - 3*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 3*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(3*xd1 - xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - H*k1**2*(xd1 - xd2)**2*(H**2*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 5*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 28*Qm*(xQm - 1) + 28*xQm - 28) + 4*xd1**2*xd2*(2*Hd1*Qm*xd2*(-fd1*xQm + fd1 + xQm - 1) + Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 19*Qm*(xQm - 1) - 19*xQm + 19) + xd1*xd2**2*(10*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 76*Qm*(xQm - 1) - 76*xQm + 76) - 2*xd2**3*(xQm - 1)*(-Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 14*Qm - 14)) + 2*H*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xd1 - 2*xd2) + Hd2*(fd2 - 1)*(-2*xd1 + xd2)) + Hp*(xd1 - xd2)**2*(Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))) + d**2*(3*G**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) - 9*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) - 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) - 28*xQm + 28) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 21*xQm - 21) - 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 21*xQm - 21) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 348*xQm - 348) - 348*xQm + 348) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 6*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) + 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 84*xQm - 84) - 168*xQm + 168)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*xd1*(-Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 3*xQm + 3) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 18*xQm - 18)) - 6*Hd2*(Qm - 1)*(fd2 - 1)*(3*xd1 - xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + 2*H**2*k1**2*(xd1 - xd2)**2*(H**2*(xd1**3*(2*Hd1*Qm*(fd1 - 1)*(xQm - 1) + Hd1*be*(-fd1*xQm + fd1 + xQm - 1) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(8*Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 28*xQm + 28) + 28*xQm - 28) + xd1*xd2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(5*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 52*xQm - 52) - 104*xQm + 104) + xd2**2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 28*xQm + 28) + 28*xQm - 28)) + Hp*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) + Hd2*(Qm - 1)*(fd2 - 1)*(-2*xd1 + xd2)))))*np.cos(k1*(-xd1 + xd2)/G) + (-4*H**2*(Qm - 1)*(xQm - 1)*(3*G**4*(6*H**2*xd1*xd2*(7*xd1**2 - 29*xd1*xd2 + 7*xd2**2) + 3*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xd1 - 3*xd2) + Hd2*xd1*(fd2 - 1)*(3*xd1 - xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + G**2*k1**2*(-2*H**2*xd1*xd2*(xd1 - xd2)**2*(28*xd1**2 - 113*xd1*xd2 + 28*xd2**2) - H*(xd1 - xd2)**4*(-Hd1*xd2*(fd1 - 1)*(4*xd1 - 11*xd2) + Hd2*xd1*(fd2 - 1)*(11*xd1 - 4*xd2)) + Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**6) + 2*H**2*k1**4*xd1*xd2*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*H*d*(-3*G**4*(3*H**3*(3*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 7*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 4*xQm - 4) + 28*xQm - 28) + xd1**2*xd2*(-10*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 88*Qm*(xQm - 1) - 88*xQm + 88) + xd1*xd2**2*(14*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 7*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 88*Qm*(xQm - 1) - 88*xQm + 88) - xd2**3*(xQm - 1)*(-3*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 28*Qm - 28)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 3*xQm + 3) + 3*xQm - 3) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 18*xQm - 18) - 18*xQm + 18)) + 6*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(3*xd1 - xd2)) - 3*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 3*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(3*xd1 - xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + G**2*k1**2*(xd1 - xd2)**2*(H**3*(11*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(4*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 13*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 13*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 56*Qm*(xQm - 1) + 56*xQm - 56) + xd1**2*xd2*(Qm*(-38*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 19*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 340*xQm - 340) - 340*xQm + 19*xd2*(Hd1*be*(fd1 - 1)*(xQm - 1) - Hd2*(fd2 - 1)*(-2*xQm + xbe)) + 340) + 2*xd1*xd2**2*(26*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) - 13*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 170*Qm*(xQm - 1) - 170*xQm + 170) - xd2**3*(xQm - 1)*(-11*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 112*Qm - 112)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 4*xQm + 4) + 4*xQm - 4) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 22*xQm - 22) - 22*xQm + 22)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(11*xd1 - 4*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(4*xd1 - 11*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(11*xd1 - 4*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + 4*H**3*k1**4*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**4*(xd1 + xd2)*(xd1**2 - 3*xd1*xd2 + xd2**2)) + d**2*(-3*G**4*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 3*xQm - 3) - 9*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) - 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) - 28*xQm + 28) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 21*xQm - 21) - 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 21*xQm - 21) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 348*xQm - 348) - 348*xQm + 348) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 6*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) + 3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 84*xQm - 84) - 168*xQm + 168)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*xd1*(-Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 3*xQm + 3) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 18*xQm - 18)) - 6*Hd2*(Qm - 1)*(fd2 - 1)*(3*xd1 - xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + G**2*k1**2*(xd1 - xd2)**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-4*Hd1*Qm*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*Hd1*be*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 11*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 19*xQm - 19) - 26*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 19*xQm - 19) + 13*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 56*xQm - 56) - 112*xQm + 112) - 2*xd1*xd2*(-2*Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 13*xQm - 13) - 19*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 13*xQm - 13) + 19*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 452*xQm - 452) - 452*xQm + 452) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 8*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) + 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 112*xQm - 112) - 224*xQm + 224)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*xd1*(-Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 4*xQm + 4) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 22*xQm - 22)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-11*xd1 + 4*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 8*H**4*k1**4*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)))*np.sin(k1*(-xd1 + xd2)/G))/(H**4*d**2*k1**7*(xd1 - xd2)**7)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -6*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(-2*H*(Qm - 1)*(xQm - 1)*(7*G**2*(Hd2*xd1**2*(fd2 - 1) + xd2*(2*H*xd1 + Hd1*xd2*(fd1 - 1))) - 4*H*k1**2*xd1**2*xd2**2) + d**2*(7*G**2*(H**2*(-Hd2*xd1*(fd2 - 1)*(-2*xQm + xbe) + Qm*(Hd2*xd1*(fd2 - 1)*(-2*xQm + xbe) - 2*(xQm - 1)*(Hd1*xd2*(fd1 - 1) + 2)) + (xQm - 1)*(Hd1*be*xd2*(fd1 - 1) + 4)) + Hp*(Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) + Hd2*xd1*(-Qm*fd2 + Qm + fd2 - 1))) + 8*H**2*k1**2*xd1*xd2*(Qm - 1)*(xQm - 1)) + d*(-7*G**2*(H**2*(Hd2*xd1**2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 4*xd1*(-Qm*xQm + Qm + xQm - 1) - xd2*(xQm - 1)*(-Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 4*Qm - 4)) - 2*H*(Qm - 1)*(xQm - 1)*(Hd1*xd2*(fd1 - 1) + Hd2*xd1*(fd2 - 1)) + Hp*(Hd1*xd2**2*(-fd1*xQm + fd1 + xQm - 1) + Hd2*xd1**2*(-Qm*fd2 + Qm + fd2 - 1))) - 8*H**2*k1**2*xd1*xd2*(Qm - 1)*(xQm - 1)*(xd1 + xd2)))/(7*G**3*H**2*d**2*k1**2)
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxISW.l3_terms1, LxISW.l3_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 63*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(G*(4*H**2*(Qm - 1)*(xQm - 1)*(15*G**4*(10*H**2*xd1*xd2*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) + 2*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(2*xd1 - 7*xd2) + Hd2*xd1*(fd2 - 1)*(7*xd1 - 2*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 3*G**2*k1**2*(xd1 - xd2)**2*(H**2*xd1*xd2*(181*xd1**2 - 822*xd1*xd2 + 181*xd2**2) + 3*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(3*xd1 - 10*xd2) + Hd2*xd1*(fd2 - 1)*(10*xd1 - 3*xd2)) - 2*Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + H*k1**4*(H*xd1*xd2*(xd1 - xd2)**4*(23*xd1**2 - 88*xd1*xd2 + 23*xd2**2) - Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2)*(xd1 - xd2)**6 + Hd2*xd1*(fd2 - 1)*(xd1 - xd2)**6*(2*xd1 - xd2))) + 2*H*d*(15*G**4*(2*H**3*(7*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 8*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 20*xQm + 20) + 40*xQm - 40) + xd1**2*xd2*(11*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(-22*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 290*xQm - 290) - 290*xQm + 290) + 2*xd1*xd2**2*(8*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(16*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 145*xQm - 145) - 145*xQm + 145) - xd2**3*(xQm - 1)*(-7*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 80*Qm - 80)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 4*xQm + 4) + 4*xQm - 4) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 28*xQm - 28) - 28*xQm + 28)) + 4*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(7*xd1 - 2*xd2)) - 2*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(2*xd1 - 7*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(7*xd1 - 2*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - 3*G**2*k1**2*(xd1 - xd2)**2*(H**3*(30*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(9*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 69*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(18*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 69*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 362*xQm + 362) + 362*xQm - 362) + 2*xd1**2*xd2*(Qm*(-48*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 24*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 641*xQm - 641) - 641*xQm + 24*xd2*(Hd1*be*(fd1 - 1)*(xQm - 1) - Hd2*(fd2 - 1)*(-2*xQm + xbe)) + 641) + xd1*xd2**2*(-69*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(138*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 1282*xQm - 1282) - 1282*xQm + 1282) - 2*xd2**3*(xQm - 1)*(-15*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 181*Qm - 181)) - 2*H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + xd1*(2*Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-2*Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 9*xQm + 9) + 9*xQm - 9) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 30*xQm - 30) - 30*xQm + 30)) + 3*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(10*xd1 - 3*xd2)) - 3*H*Hp*(xd1 - xd2)**2*(Hd1*xd2*(fd1 - 1)*(xQm - 1)*(-3*xd1 + 10*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(10*xd1 - 3*xd2)) + 2*Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + H*k1**4*(xd1 - xd2)**4*(H**2*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 5*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 46*Qm*(xQm - 1) + 46*xQm - 46) + 2*xd1**2*xd2*(4*Hd1*Qm*xd2*(-fd1*xQm + fd1 + xQm - 1) + 2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) - 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 65*Qm*(xQm - 1) - 65*xQm + 65) + xd1*xd2**2*(10*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 130*Qm*(xQm - 1) - 130*xQm + 130) - 2*xd2**3*(xQm - 1)*(-Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 23*Qm - 23)) + 2*H*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xd1 - 2*xd2) - Hd2*(fd2 - 1)*(2*xd1 - xd2)) + Hp*(xd1 - xd2)**2*(Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))) + d**2*(15*G**4*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 4*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + Hd1*be*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 7*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 32*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) + 16*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 80*xQm - 80) - 160*xQm + 160) - 4*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) - 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) + 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 370*xQm - 370) - 370*xQm + 370) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) - 8*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) + 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 160*xQm - 160) - 320*xQm + 320)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*xd1*(-Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 4*xQm + 4) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 28*xQm - 28)) - 4*Hd2*(Qm - 1)*(fd2 - 1)*(7*xd1 - 2*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 6*G**2*k1**2*(xd1 - xd2)**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 9*xQm - 9) + Hd1*be*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 9*xQm - 9) - 30*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(-6*Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 8*xQm - 8) - 69*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(12*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 8*xQm - 8) + 69*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 362*xQm - 362) - 362*xQm + 362) + xd1*xd2*(Hd1*be*xd2*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 69*xQm - 69) + 48*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 2*Qm*(Hd1*xd2*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 69*xQm - 69) + 24*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 822*xQm - 822) + 1644*xQm - 1644) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 30*xQm - 30) - 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 30*xQm - 30) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 362*xQm - 362) - 362*xQm + 362)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + xd1*(-2*Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 9*xQm + 9) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 30*xQm - 30)) - 3*Hd2*(Qm - 1)*(fd2 - 1)*(10*xd1 - 3*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 2*H**2*k1**4*(xd1 - xd2)**4*(H**2*(xd1**3*(2*Hd1*Qm*(fd1 - 1)*(xQm - 1) + Hd1*be*(-fd1*xQm + fd1 + xQm - 1) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(8*Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 46*xQm + 46) + 46*xQm - 46) + xd1*xd2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(5*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 88*xQm - 88) - 176*xQm + 176) + xd2**2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(-fd1*xQm + fd1 + xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 46*xQm + 46) + 46*xQm - 46)) + Hp*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))))*np.sin(k1*(-xd1 + xd2)/G) + k1*(xd1 - xd2)*(4*H**2*(Qm - 1)*(xQm - 1)*(15*G**4*(10*H**2*xd1*xd2*(8*xd1**2 - 37*xd1*xd2 + 8*xd2**2) + 2*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(2*xd1 - 7*xd2) + Hd2*xd1*(fd2 - 1)*(7*xd1 - 2*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - G**2*k1**2*(xd1 - xd2)**2*(11*H**2*xd1*xd2*(13*xd1**2 - 56*xd1*xd2 + 13*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(7*xd1 - 20*xd2) + Hd2*xd1*(fd2 - 1)*(20*xd1 - 7*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + 2*H**2*k1**4*xd1*xd2*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*H*d*(15*G**4*(2*H**3*(7*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 8*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 20*xQm + 20) + 40*xQm - 40) + xd1**2*xd2*(11*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(-22*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 290*xQm - 290) - 290*xQm + 290) + 2*xd1*xd2**2*(8*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(16*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 145*xQm - 145) - 145*xQm + 145) - xd2**3*(xQm - 1)*(-7*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 80*Qm - 80)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 4*xQm + 4) + 4*xQm - 4) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 28*xQm - 28) - 28*xQm + 28)) + 4*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(7*xd1 - 2*xd2)) - 2*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(2*xd1 - 7*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(7*xd1 - 2*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - G**2*k1**2*(xd1 - xd2)**2*(H**3*(20*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(14*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + 7*Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 47*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 47*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 286*Qm*(xQm - 1) + 286*xQm - 286) + 2*xd1**2*xd2*(Qm*(-34*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 17*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 473*xQm - 473) - 473*xQm + 17*xd2*(Hd1*be*(fd1 - 1)*(xQm - 1) - Hd2*(fd2 - 1)*(-2*xQm + xbe)) + 473) + xd1*xd2**2*(94*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) - 47*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 7*Hd2*Qm*xd2*(fd2 - 1)*(-2*xQm + xbe) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 946*Qm*(xQm - 1) - 946*xQm + 946) - 2*xd2**3*(xQm - 1)*(-10*Hd1*xd2*(-2*Qm + be)*(fd1 - 1) + 143*Qm - 143)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + 2*xd1*(Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(-Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 7*xQm + 7) + 7*xQm - 7) + xd2*(-Hd2*xd2*(fd2 - 1)*(be - xQm*(be + 2) + xbe) + Qm*(Hd2*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 40*xQm - 40) - 40*xQm + 40)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(20*xd1 - 7*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(7*xd1 - 20*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(20*xd1 - 7*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - 4*H**3*k1**4*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**4*(xd1 + xd2)*(xd1**2 - 3*xd1*xd2 + xd2**2)) + d**2*(15*G**4*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 4*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + Hd1*be*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 7*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 32*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 22*xQm - 22) + 16*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 80*xQm - 80) - 160*xQm + 160) - 4*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) - 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 16*xQm - 16) + 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 370*xQm - 370) - 370*xQm + 370) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) - 8*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 28*xQm - 28) + 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 160*xQm - 160) - 320*xQm + 320)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*xd1*(-Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 4*xQm + 4) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 28*xQm - 28)) - 4*Hd2*(Qm - 1)*(fd2 - 1)*(7*xd1 - 2*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - G**2*k1**2*(xd1 - xd2)**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 7*xQm - 7) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 7*xQm - 7) - 20*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 34*xQm - 34) - 47*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 34*xQm - 34) + 47*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 286*xQm - 286) - 286*xQm + 286) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 47*xQm - 47) - 34*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 47*xQm - 47) + 17*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 616*xQm - 616) - 1232*xQm + 1232) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) - 14*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) + 7*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 286*xQm - 286) - 572*xQm + 572)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 7*xQm - 7) + xd2*(Hd2*xd2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + 40*xQm - 40)) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(20*xd1 - 7*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + 8*H**4*k1**4*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**4*(xd1**2 - 3*xd1*xd2 + xd2**2)))*np.cos(k1*(-xd1 + xd2)/G))/(H**4*d**2*k1**8*(xd1 - xd2)**8)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)/(5*G**2*d**2*k1)
    
    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(LxISW.l4_terms1, LxISW.l4_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 81*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(G*k1*(xd1 - xd2)*(4*H**2*(Qm - 1)*(xQm - 1)*(105*G**4*(15*H**2*xd1*xd2*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) + 5*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xd1 - 4*xd2) + Hd2*xd1*(fd2 - 1)*(4*xd1 - xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 5*G**2*k1**2*(xd1 - xd2)**2*(51*H**2*xd1*xd2*(7*xd1**2 - 34*xd1*xd2 + 7*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(13*xd1 - 46*xd2) + Hd2*xd1*(fd2 - 1)*(46*xd1 - 13*xd2)) - 2*Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + H*k1**4*(xd1 - xd2)**4*(H*xd1*xd2*(35*xd1**2 - 136*xd1*xd2 + 35*xd2**2) - Hd1*xd2*(fd1 - 1)*(xd1 - 2*xd2)*(xd1 - xd2)**2 + Hd2*xd1*(fd2 - 1)*(xd1 - xd2)**2*(2*xd1 - xd2))) + 2*H*d*(105*G**4*(5*H**3*(4*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 9*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) + 54*xQm - 54) - 6*xd1**2*xd2*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 37*xQm + 37) + 37*xQm - 37) + xd1*xd2**2*(-9*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(18*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 222*xQm - 222) - 222*xQm + 222) - 2*xd2**3*(xQm - 1)*(-2*Hd1*be*xd2*(fd1 - 1) + Qm*(4*Hd1*xd2*(fd1 - 1) + 27) - 27)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 5*Qm*(xQm - 1) - 5*xQm + 5) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 40*Qm*(xQm - 1) - 40*xQm + 40)) + 10*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(4*xd1 - xd2)) - 5*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 4*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(4*xd1 - xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - 5*G**2*k1**2*(xd1 - xd2)**2*(H**3*(46*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(26*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) - 13*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 105*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 21*Qm*(5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 34*xQm - 34) + 714*xQm - 714) - 18*xd1**2*xd2*(-4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 8*Hd2*fd2*xQm*xd2 + 4*Hd2*fd2*xbe*xd2 + 8*Hd2*xQm*xd2 - 4*Hd2*xbe*xd2 + Qm*(8*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 153*xQm + 153) + 153*xQm - 153) + xd1*xd2**2*(-105*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 26*Hd2*fd2*xQm*xd2 + 13*Hd2*fd2*xbe*xd2 + 26*Hd2*xQm*xd2 - 13*Hd2*xbe*xd2 + Qm*(210*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 13*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2754*xQm - 2754) - 2754*xQm + 2754) - 2*xd2**3*(xQm - 1)*(-23*Hd1*be*xd2*(fd1 - 1) + Qm*(46*Hd1*xd2*(fd1 - 1) + 357) - 357)) - 2*H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) + xd1*(-2*Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) - 2*Hd2*be*xd2*(fd2 - 1)*(xQm - 1) - 4*Hd2*fd2*xQm*xd2 + 2*Hd2*fd2*xbe*xd2 + 4*Hd2*xQm*xd2 - 2*Hd2*xbe*xd2 - 13*Qm*(xQm - 1) + 13*xQm - 13) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 46*Qm*(xQm - 1) - 46*xQm + 46)) + Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(46*xd1 - 13*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(13*xd1 - 46*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(46*xd1 - 13*xd2)) + 2*Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + H*k1**4*(xd1 - xd2)**4*(H**2*(2*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 70*xQm + 70) + 70*xQm - 70) + 2*xd1**2*xd2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 4*Hd2*fd2*xQm*xd2 - 2*Hd2*fd2*xbe*xd2 - 4*Hd2*xQm*xd2 + 2*Hd2*xbe*xd2 + Qm*(-4*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 101*xQm - 101) - 101*xQm + 101) + xd1*xd2**2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(10*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 202*xQm - 202) - 202*xQm + 202) - 2*xd2**3*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + Qm*(2*Hd1*xd2*(fd1 - 1) + 35) - 35)) + 2*H*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xd1 - 2*xd2) - Hd2*(fd2 - 1)*(2*xd1 - xd2)) - Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))) + d**2*(105*G**4*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) - 20*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 90*xQm - 90) - 90*xQm + 90) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 45*xQm - 45) - 30*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 45*xQm - 45) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 690*xQm - 690) - 1380*xQm + 1380) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) - 10*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 270*xQm - 270) - 540*xQm + 540)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 5*xQm - 5) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 40*xQm - 40)) - 10*Hd2*(Qm - 1)*(fd2 - 1)*(4*xd1 - xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 10*G**2*k1**2*(xd1 - xd2)**2*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 13*xQm - 13) + Hd1*be*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 13*xQm - 13) - 46*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 3*xd1**2*(-2*Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 12*xQm - 12) - 35*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(4*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 12*xQm - 12) + 35*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 238*xQm - 238) - 238*xQm + 238) + xd1*xd2*(Hd1*be*xd2*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 105*xQm - 105) + 72*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 2*Qm*(Hd1*xd2*(fd1 - 1)*(4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 105*xQm - 105) + 36*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 1734*xQm - 1734) + 3468*xQm - 3468) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 46*xQm - 46) + 26*Hd2*fd2*xQm*xd2 - 13*Hd2*fd2*xbe*xd2 - 26*Hd2*xQm*xd2 + 13*Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 46*xQm - 46) - 26*Hd2*fd2*xQm*xd2 + 13*Hd2*fd2*xbe*xd2 + 26*Hd2*xQm*xd2 - 13*Hd2*xbe*xd2 + 714*xQm - 714) - 714*xQm + 714)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) + xd1*(-2*Hd2*be*xd2*(fd2 - 1) + 2*Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) - 13*xQm + 13) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 46*xQm - 46)) - Hd2*(Qm - 1)*(fd2 - 1)*(46*xd1 - 13*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 2*H**2*k1**4*(xd1 - xd2)**4*(H**2*(xd1**3*(2*Hd1*Qm*(fd1 - 1)*(xQm - 1) + Hd1*be*(-fd1*xQm + fd1 + xQm - 1) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + xd1**2*(4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(-8*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 70*xQm + 70) + 70*xQm - 70) + xd1*xd2*(-5*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(5*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 136*xQm - 136) - 272*xQm + 272) + xd2**2*(2*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(-4*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 70*xQm + 70) + 70*xQm - 70)) + Hp*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(xQm - 1)*(xd1 - 2*xd2) - Hd2*(Qm - 1)*(fd2 - 1)*(2*xd1 - xd2)))))*np.cos(k1*(-xd1 + xd2)/G) + (4*H**2*(Qm - 1)*(xQm - 1)*(105*G**6*(15*H**2*xd1*xd2*(9*xd1**2 - 46*xd1*xd2 + 9*xd2**2) + 5*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xd1 - 4*xd2) + Hd2*xd1*(fd2 - 1)*(4*xd1 - xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 15*G**4*k1**2*(xd1 - xd2)**2*(2*H**2*xd1*xd2*(217*xd1**2 - 1094*xd1*xd2 + 217*xd2**2) + 2*H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(8*xd1 - 31*xd2) + Hd2*xd1*(fd2 - 1)*(31*xd1 - 8*xd2)) - 3*Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + G**2*k1**4*(xd1 - xd2)**4*(3*H**2*xd1*xd2*(105*xd1**2 - 472*xd1*xd2 + 105*xd2**2) + H*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(11*xd1 - 32*xd2) + Hd2*xd1*(fd2 - 1)*(32*xd1 - 11*xd2)) - Hd1*Hd2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 2*H**2*k1**6*xd1*xd2*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2)) + 2*H*d*(105*G**6*(5*H**3*(4*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(2*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) + Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) + 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 9*Qm*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 6*xQm - 6) + 54*xQm - 54) - 6*xd1**2*xd2*(Hd1*be*xd2*(-fd1*xQm + fd1 + xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 37*xQm + 37) + 37*xQm - 37) + xd1*xd2**2*(-9*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 2*Hd2*fd2*xQm*xd2 + Hd2*fd2*xbe*xd2 + 2*Hd2*xQm*xd2 - Hd2*xbe*xd2 + Qm*(18*Hd1*xd2*(fd1 - 1)*(xQm - 1) - Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 222*xQm - 222) - 222*xQm + 222) - 2*xd2**3*(xQm - 1)*(-2*Hd1*be*xd2*(fd1 - 1) + Qm*(4*Hd1*xd2*(fd1 - 1) + 27) - 27)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 5*Qm*(xQm - 1) - 5*xQm + 5) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 40*Qm*(xQm - 1) - 40*xQm + 40)) + 10*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(4*xd1 - xd2)) - 5*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(xd1 - 4*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(4*xd1 - xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) - 15*G**4*k1**2*(xd1 - xd2)**2*(2*H**3*(31*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(8*Hd1*Qm*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 35*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 7*Qm*(5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 31*xQm - 31) + 217*xQm - 217) + xd1**2*xd2*(47*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 94*Hd2*fd2*xQm*xd2 - 47*Hd2*fd2*xbe*xd2 - 94*Hd2*xQm*xd2 + 47*Hd2*xbe*xd2 + Qm*(-94*Hd1*xd2*(fd1 - 1)*(xQm - 1) + 47*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 1754*xQm - 1754) - 1754*xQm + 1754) + 2*xd1*xd2**2*(-35*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 8*Hd2*fd2*xQm*xd2 + 4*Hd2*fd2*xbe*xd2 + 8*Hd2*xQm*xd2 - 4*Hd2*xbe*xd2 + Qm*(70*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 4*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 877*xQm - 877) - 877*xQm + 877) - 31*xd2**3*(xQm - 1)*(-Hd1*be*xd2*(fd1 - 1) + 2*Qm*(Hd1*xd2*(fd1 - 1) + 7) - 14)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(3*Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(3*Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 3*Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 6*Hd2*fd2*xQm*xd2 - 3*Hd2*fd2*xbe*xd2 - 6*Hd2*xQm*xd2 + 3*Hd2*xbe*xd2 + 16*Qm*(xQm - 1) - 16*xQm + 16) + xd2*(3*Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + 3*Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 6*Hd2*fd2*xQm*xd2 - 3*Hd2*fd2*xbe*xd2 - 6*Hd2*xQm*xd2 + 3*Hd2*xbe*xd2 + 124*Qm*(xQm - 1) - 124*xQm + 124)) + 4*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(31*xd1 - 8*xd2)) - 2*H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(8*xd1 - 31*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(31*xd1 - 8*xd2)) + 3*Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + G**2*k1**4*(xd1 - xd2)**4*(H**3*(32*Hd2*xd1**4*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe) + xd1**3*(-11*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) + 75*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(22*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 75*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 630*xQm + 630) + 630*xQm - 630) - 6*xd1**2*xd2*(-9*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 18*Hd2*fd2*xQm*xd2 + 9*Hd2*fd2*xbe*xd2 + 18*Hd2*xQm*xd2 - 9*Hd2*xbe*xd2 + Qm*(18*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) - 367*xQm + 367) + 367*xQm - 367) + xd1*xd2**2*(-75*Hd1*be*xd2*(fd1 - 1)*(xQm - 1) - 22*Hd2*fd2*xQm*xd2 + 11*Hd2*fd2*xbe*xd2 + 22*Hd2*xQm*xd2 - 11*Hd2*xbe*xd2 + Qm*(150*Hd1*xd2*(fd1 - 1)*(xQm - 1) - 11*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2202*xQm - 2202) - 2202*xQm + 2202) - 2*xd2**3*(xQm - 1)*(-16*Hd1*be*xd2*(fd1 - 1) + Qm*(32*Hd1*xd2*(fd1 - 1) + 315) - 315)) - H**2*(xd1 - xd2)**2*(Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(Qm*(-4*xQm + xbe + 2) + be*(xQm - 1) + 2*xQm - xbe) - 2*xd1*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 11*Qm*(xQm - 1) - 11*xQm + 11) + xd2*(Hd2*Qm*xd2*(fd2 - 1)*(-4*xQm + xbe + 2) + Hd2*be*xd2*(fd2 - 1)*(xQm - 1) + 2*Hd2*fd2*xQm*xd2 - Hd2*fd2*xbe*xd2 - 2*Hd2*xQm*xd2 + Hd2*xbe*xd2 + 64*Qm*(xQm - 1) - 64*xQm + 64)) + 2*Hd2*(Qm - 1)*(fd2 - 1)*(xQm - 1)*(32*xd1 - 11*xd2)) - H*Hp*(xd1 - xd2)**2*(-Hd1*xd2*(fd1 - 1)*(xQm - 1)*(11*xd1 - 32*xd2) + Hd2*xd1*(Qm - 1)*(fd2 - 1)*(32*xd1 - 11*xd2)) + Hd1*Hd2*Hp*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4*(Qm + xQm - 2)) + 4*H**3*k1**6*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**6*(xd1**3 - 2*xd1**2*xd2 - 2*xd1*xd2**2 + xd2**3)) + d**2*(105*G**6*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 5*xQm - 5) - 20*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) - 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 10*xQm - 10) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 90*xQm - 90) - 90*xQm + 90) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 45*xQm - 45) - 30*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 45*xQm - 45) + 15*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 690*xQm - 690) - 1380*xQm + 1380) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) - 10*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 40*xQm - 40) + 5*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 270*xQm - 270) - 540*xQm + 540)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 5*xQm - 5) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 40*xQm - 40)) - 10*Hd2*(Qm - 1)*(fd2 - 1)*(4*xd1 - xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 15*G**4*k1**2*(xd1 - xd2)**2*(H**4*(-3*Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 4*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 8*xQm - 8) + Hd1*be*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 8*xQm - 8) - 31*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 2*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 94*xQm - 94) - 140*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(9*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 94*xQm - 94) + 70*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 434*xQm - 434) - 868*xQm + 868) - 4*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 70*xQm - 70) + 94*Hd2*fd2*xQm*xd2 - 47*Hd2*fd2*xbe*xd2 - 94*Hd2*xQm*xd2 + 47*Hd2*xbe*xd2 + Qm*(2*Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 70*xQm - 70) - 94*Hd2*fd2*xQm*xd2 + 47*Hd2*fd2*xbe*xd2 + 94*Hd2*xQm*xd2 - 47*Hd2*xbe*xd2 + 2188*xQm - 2188) - 2188*xQm + 2188) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 124*xQm - 124) - 32*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(3*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 124*xQm - 124) + 16*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 868*xQm - 868) - 1736*xQm + 1736)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(3*Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(3*Hd2*be*xd2*(fd2 - 1) - 3*Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 16*xQm - 16) + xd2*(3*Hd2*be*xd2*(fd2 - 1) - 3*Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 124*xQm - 124)) - 4*Hd2*(Qm - 1)*(fd2 - 1)*(31*xd1 - 8*xd2)) - 3*Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) + G**2*k1**4*(xd1 - xd2)**4*(H**4*(-Hd1*Hd2*xd1**4*(-2*Qm + be)*(fd1 - 1)*(fd2 - 1)*(-2*xQm + xbe) + 2*xd1**3*(-2*Hd1*Qm*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 11*xQm - 11) + Hd1*be*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 11*xQm - 11) - 32*Hd2*(Qm - 1)*(fd2 - 1)*(-2*xQm + xbe)) + 6*xd1**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 25*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + Qm*(2*Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 18*xQm - 18) + 25*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 210*xQm - 210) - 210*xQm + 210) - 2*xd1*xd2*(-Hd1*be*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 75*xQm - 75) - 54*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 2*Qm*(Hd1*xd2*(fd1 - 1)*(2*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 75*xQm - 75) + 27*Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 1416*xQm - 1416) - 2832*xQm + 2832) + xd2**2*(-Hd1*be*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 64*xQm - 64) + 44*Hd2*fd2*xQm*xd2 - 22*Hd2*fd2*xbe*xd2 - 44*Hd2*xQm*xd2 + 22*Hd2*xbe*xd2 + 2*Qm*(Hd1*xd2*(fd1 - 1)*(Hd2*xd2*(fd2 - 1)*(-2*xQm + xbe) + 64*xQm - 64) - 22*Hd2*fd2*xQm*xd2 + 11*Hd2*fd2*xbe*xd2 + 22*Hd2*xQm*xd2 - 11*Hd2*xbe*xd2 + 630*xQm - 630) - 1260*xQm + 1260)) - H**2*Hp*(xd1 - xd2)**2*(-Hd1*(fd1 - 1)*(Hd2*xd1**2*(fd2 - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*xd1*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 11*xQm - 11) + xd2*(Hd2*be*xd2*(fd2 - 1) - Hd2*xd2*(fd2 - 1)*(2*Qm + 2*xQm - xbe) + 64*xQm - 64)) - 2*Hd2*(Qm - 1)*(fd2 - 1)*(32*xd1 - 11*xd2)) - Hd1*Hd2*Hp**2*(fd1 - 1)*(fd2 - 1)*(xd1 - xd2)**4) - 8*H**4*k1**6*(Qm - 1)*(xQm - 1)*(xd1 - xd2)**6*(xd1**2 - 3*xd1*xd2 + xd2**2)))*np.sin(k1*(-xd1 + xd2)/G))/(H**4*d**2*k1**9*(xd1 - xd2)**9)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 72*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)/(35*G**3*d**2)
    
class TDxTD(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(TDxTD.l0_terms1, TDxTD.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*np.sin(k1*(xd1 - xd2)/G)/(d**2*k1**5*(xd1 - xd2))

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 36*D1d1*D1d2*G*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)/(d**2*k1**4)
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(TDxTD.l1_terms1, TDxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -108*1j*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(G*np.sin(k1*(xd1 - xd2)/G) - k1*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**6*(xd1 - xd2)**2)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(TDxTD.l2_terms1, TDxTD.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 180*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(3*G*k1*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G) + (-3*G**2 + k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**7*(xd1 - xd2)**3)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(TDxTD.l1_terms1, TDxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 252*1j*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(3*G*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G) + k1*(-15*G**2 + k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))/(d**2*k1**8*(xd1 - xd2)**4)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(TDxTD.l1_terms1, TDxTD.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 324*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(Qm - 1)*(xQm - 1)*(-5*G*k1*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G) + (105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4)*np.sin(k1*(xd1 - xd2)/G))/(d**2*k1**9*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
class ISWxISW(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l0_terms1, ISWxISW.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 9*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)*np.sin(k1*(xd1 - xd2)/G)/(H**4*d**2*k1**5*(xd1 - xd2))

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 9*D1d1*D1d2*G*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**4
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l1_terms1, ISWxISW.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -27*1j*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(G*np.sin(k1*(xd1 - xd2)/G) - k1*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/(k1**6*(xd1 - xd2)**2)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l2_terms1, ISWxISW.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -45*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(3*G*k1*(-xd1 + xd2)*np.cos(k1*(xd1 - xd2)/G) + (3*G**2 - k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)/(H**4*d**2*k1**7*(xd1 - xd2)**3)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l3_terms1, ISWxISW.l3_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 63*1j*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(3*G*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G) + k1*(-15*G**2 + k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)/(H**4*d**2*k1**8*(xd1 - xd2)**4)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l4_terms1, ISWxISW.l4_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 81*D1d1*D1d2*G**2*Hd1**3*Hd2**3*OMd1*OMd2*Pk*(fd1 - 1)*(fd2 - 1)*(-5*G*k1*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G) + (105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4)*np.sin(k1*(xd1 - xd2)/G))*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d)/(H**4*d**2*k1**9*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
class TDxISW(BaseInt):
    @staticmethod
    def l0(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l0_terms1, ISWxISW.l0_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l0_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 9*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(H**2*(Hd1*d*(-2*Qm + be)*(fd1 - 1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))*(H**2*(Hd2*d*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd2*(fd2 - 1)*(xQm - 1) - Hd2*Hp*d*(fd2 - 1))*np.sin(k1*(xd1 - xd2)/G)/(H**4*d**2*k1**5*(xd1 - xd2))

    # for when xd1 = xd2
    @staticmethod
    def l0_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return D1d1*D1d2*Pk*(3*G**2*Hd1**3*OMd1*(fd1 - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 6*G**2*Hd1**2*OMd1*(Qm - 1)/(d*k1**2))*(3*G**2*Hd2**3*OMd2*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**2 + 6*G**2*Hd2**2*OMd2*(xQm - 1)/(d*k1**2))/G**3
    
    @staticmethod
    def l1(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l1_terms1, ISWxISW.l1_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l1_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 27*1j*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(G*np.sin(k1*(xd1 - xd2)/G) - k1*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))*(H**2*(-Hd1*be*d*(fd1 - 1) + 2*Qm*(Hd1*d*(fd1 - 1) - 1) + 2) - 2*H*Hd1*(Qm - 1)*(fd1 - 1) + Hd1*Hp*d*(fd1 - 1))*(H**2*(Hd2*d*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd2*(fd2 - 1)*(xQm - 1) - Hd2*Hp*d*(fd2 - 1))/(H**4*d**2*k1**6*(xd1 - xd2)**2)

    # for when xd1 = xd2
    @staticmethod
    def l1_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l2(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l2_terms1, ISWxISW.l2_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l2_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return -45*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(3*G*k1*(-xd1 + xd2)*np.cos(k1*(xd1 - xd2)/G) + (3*G**2 - k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G))*(H**2*(Hd1*d*(-2*Qm + be)*(fd1 - 1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))*(H**2*(Hd2*d*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd2*(fd2 - 1)*(xQm - 1) - Hd2*Hp*d*(fd2 - 1))/(H**4*d**2*k1**7*(xd1 - xd2)**3)

    # for when xd1 = xd2
    @staticmethod
    def l2_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l3(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l3_terms1, ISWxISW.l3_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l3_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 63*1j*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(3*G*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*np.sin(k1*(xd1 - xd2)/G) + k1*(-15*G**2 + k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G))*(H**2*(Hd1*d*(-2*Qm + be)*(fd1 - 1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))*(H**2*(Hd2*d*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd2*(fd2 - 1)*(xQm - 1) - Hd2*Hp*d*(fd2 - 1))/(H**4*d**2*k1**8*(xd1 - xd2)**4)

    # for when xd1 = xd2
    @staticmethod
    def l3_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0
    
    def l4(cosmo_functions, k1, zz=0, t=0, sigma=0, n1=16, n2=16):
        return BaseInt.double_int(ISWxISW.l4_terms1, ISWxISW.l4_terms2, cosmo_functions, k1, zz, t, sigma, n1, n2)

    # for when xd1 != xd2
    @staticmethod
    def l4_terms1(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 81*D1d1*D1d2*G**2*Hd1**2*Hd2**2*OMd1*OMd2*Pk*(-5*G*k1*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)*np.cos(k1*(xd1 - xd2)/G) + (105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4)*np.sin(k1*(xd1 - xd2)/G))*(H**2*(Hd1*d*(-2*Qm + be)*(fd1 - 1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))*(H**2*(Hd2*d*(fd2 - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd2*(fd2 - 1)*(xQm - 1) - Hd2*Hp*d*(fd2 - 1))/(H**4*d**2*k1**9*(xd1 - xd2)**5)

    # for when xd1 = xd2
    @staticmethod
    def l4_terms2(xd1, xd2, cosmo_functions, k1, zz, t=0, sigma=0):
        d, f, D1, b1, xb1, H, Hp, OM, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_functions, zz)
        zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_functions, xd1)
        zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_functions, xd2)

        G = (xd1 + xd2) / (2 * d)
        Pk = BaseInt.pk(cosmo_functions, k1/G)

        return 0