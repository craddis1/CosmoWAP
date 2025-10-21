import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils
from cosmo_wap.lib import integrate

class LxNPP(BaseInt):
    @staticmethod
    def mu_integrand(xd,mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        """2D P(k,mu) power spectra - this returns the integrand - so potentially array (k,mu,xd)"""
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1,zz and mu with xd
        k1,zz,mu = utils.enable_broadcasting(k1,zz,mu,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define cosmo_funcs,zz = args[1,3]G from dirac-delta - could just define q=k1/G
        Pk = baseint.pk(k1/G,zzd1,zz)

        expr = D1*D1d*Pk*(Hd**2*OMd*xd*(3*Qm - 3)*(d - xd)*(1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(f*mu**2 + xb1)*(2*1j*G*mu/(k1*xd) - mu**2 + 1)/d + Hd**2*OMd*xd*(b1 + f*mu**2)*(d - xd)*(3*xQm - 3)*(-1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(-2*1j*G*mu/(k1*xd) - mu**2 + 1)/d)/G**3
        return expr
    
    @staticmethod
    def mu(mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        """2D P(k,mu) power spectra"""
        return BaseInt.single_int(LxNPP.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n, remove_div=remove_div)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128,n_mu=16,fast=False):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(LxNPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n ,n_mu=n_mu,fast=fast)

    ############################ Seperate Multipoles - with analytic mu integration #################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        return BaseInt.single_int(LxNPP.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)

        expr = 2*D1*D1d*Hd**2*OMd*pk*(d - xd)*(-G*(-3*d**3*k1**2*(b1*(xQm - 1) + 3*f*(Qm + xQm - 2) + xb1*(Qm - 1)) + 6*d**2*k1**2*xd*(b1*(xQm - 1) + 2*f*(Qm + xQm - 2) + xb1*(Qm - 1)) + 3*d*(f*(6*G**2 + k1**2*xd**2)*(Qm + xQm - 2) + k1**2*xd**2*(-Qm*xb1 - b1*xQm + b1 + xb1)) - 6*f*xd*(-3*G**2 + k1**2*xd**2)*(Qm + xQm - 2))*np.sin(k1*(-d + xd)/G) - k1*(d - xd)*(18*G**2*f*xd*(Qm + xQm - 2) - 3*d**3*k1**2*(b1*(xQm - 1) + f*(Qm + xQm - 2) + xb1*(Qm - 1)) + 6*d**2*k1**2*xd*(b1*(xQm - 1) + f*(Qm + xQm - 2) + xb1*(Qm - 1)) + 3*d*(f*(6*G**2 - k1**2*xd**2)*(Qm + xQm - 2) + k1**2*xd**2*(-Qm*xb1 - b1*xQm + b1 + xb1)))*np.cos(k1*(-d + xd)/G))/(G*d*k1**5*(-d + xd)**5)

        return expr
    
    @staticmethod
    def l0_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)

        expr = 2*D1**2*H**2*OM*pk*xd*(d - xd)*(5*b1*(Qm - 1) + f*(Qm + xQm - 2) + 5*xb1*(xQm - 1))/(5*G**3*d)

        return expr
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(LxNPP.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l1_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = 6*1j*D1*D1d*Hd**2*OMd*pk*(-3*G*k1*(d - xd)*(b1*k1**2*(d - xd)**2*(2*d + xd)*(xQm - 1) + 2*d**3*k1**2*(-Qm*(2*f + xb1) + 2*f*xQm + xb1) + d**2*k1**2*xd*(5*f*(Qm - xQm) + 3*xb1*(Qm - 1)) + 2*d*f*(12*G**2 + k1**2*xd**2)*(Qm - xQm) - 3*f*xd*(-12*G**2 + k1**2*xd**2)*(Qm - xQm) - k1**2*xb1*xd**3*(Qm - 1))*np.cos(k1*(-d + xd)/G) - (-3*b1*k1**2*(d - xd)**2*(xQm - 1)*(-2*G**2*d - G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2) - f*(3 - 3*Qm)*(60*G**4*xd - 27*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(24*G**4 - 12*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) + f*(3 - 3*xQm)*(60*G**4*xd - 27*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(24*G**4 - 12*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) + 3*k1**2*xb1*(Qm - 1)*(d - xd)**2*(-2*G**2*d - G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2))*np.sin(k1*(-d + xd)/G))/(G*d*k1**6*(d - xd)**5)
        return expr
    
    @staticmethod
    def l1_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
            
        expr = 6*1j*D1**2*H**2*OM*pk*(d - xd)*(5*b1*(Qm - 1) + 3*f*(Qm - xQm) - 5*xb1*(xQm - 1))/(5*G**3*d*k1)

        return expr
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(LxNPP.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l2_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = 10*D1*D1d*Hd**2*OMd*pk*(d - xd)*(G*(b1*k1**2*(3 - 3*xQm)*(d - xd)**2*(18*G**2*xd + 9*G**2*(d - xd) - 7*k1**2*xd*(d - xd)**2 - 4*k1**2*(d - xd)**3) - f*(3 - 3*Qm)*(540*G**4*xd - 246*G**2*k1**2*xd*(d - xd)**2 + 11*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(60*G**4 - 29*G**2*k1**2*(d - xd)**2 + 2*k1**4*(d - xd)**4)) - f*(3 - 3*xQm)*(540*G**4*xd - 246*G**2*k1**2*xd*(d - xd)**2 + 11*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(60*G**4 - 29*G**2*k1**2*(d - xd)**2 + 2*k1**4*(d - xd)**4)) + k1**2*xb1*(3 - 3*Qm)*(d - xd)**2*(18*G**2*xd + 9*G**2*(d - xd) - 7*k1**2*xd*(d - xd)**2 - 4*k1**2*(d - xd)**3))*np.sin(k1*(-d + xd)/G) + k1*(d - xd)*(3*b1*k1**2*(d - xd)**2*(xQm - 1)*(-9*G**2*d - 9*G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2) - f*(3 - 3*Qm)*(540*G**4*xd - 66*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(180*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - f*(3 - 3*xQm)*(540*G**4*xd - 66*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(180*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) + 3*k1**2*xb1*(Qm - 1)*(d - xd)**2*(-9*G**2*d - 9*G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2))*np.cos(k1*(-d + xd)/G))/(G*d*k1**7*(-d + xd)**7)
        return expr
    
    @staticmethod
    def l2_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
            
        expr = -2*D1**2*H**2*OM*pk*xd*(d - xd)*(7*b1*(Qm - 1) - f*(Qm + xQm - 2) + 7*xb1*(xQm - 1))/(7*G**3*d)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(LxNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l3_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
        
        expr = 14*1j*D1*D1d*Hd**2*OMd*pk*(G*k1*(d - xd)*(-3*b1*k1**2*(d - xd)**2*(xQm - 1)*(-90*G**2*xd + 7*d**3*k1**2 - 8*d**2*k1**2*xd - 5*d*(12*G**2 + k1**2*xd**2) + 6*k1**2*xd**3) - f*(3 - 3*Qm)*(6300*G**4*xd - 810*G**2*k1**2*xd*(d - xd)**2 + 17*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(600*G**4 - 88*G**2*k1**2*(d - xd)**2 + 3*k1**4*(d - xd)**4)) + f*(3 - 3*xQm)*(6300*G**4*xd - 810*G**2*k1**2*xd*(d - xd)**2 + 17*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(600*G**4 - 88*G**2*k1**2*(d - xd)**2 + 3*k1**4*(d - xd)**4)) + 3*k1**2*xb1*(Qm - 1)*(d - xd)**2*(-90*G**2*xd + 7*d**3*k1**2 - 8*d**2*k1**2*xd - 5*d*(12*G**2 + k1**2*xd**2) + 6*k1**2*xd**3))*np.cos(k1*(-d + xd)/G) + (-b1*k1**2*(3 - 3*xQm)*(d - xd)**2*(150*G**4*xd - 63*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(60*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - f*(3 - 3*Qm)*(6300*G**6*xd - 2910*G**4*k1**2*xd*(d - xd)**2 + 147*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(1800*G**6 - 864*G**4*k1**2*(d - xd)**2 + 57*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) + f*(3 - 3*xQm)*(6300*G**6*xd - 2910*G**4*k1**2*xd*(d - xd)**2 + 147*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(1800*G**6 - 864*G**4*k1**2*(d - xd)**2 + 57*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) + k1**2*xb1*(3 - 3*Qm)*(d - xd)**2*(150*G**4*xd - 63*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(60*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)))*np.sin(k1*(-d + xd)/G))/(G*d*k1**8*(d - xd)**7)
        return expr
    
    @staticmethod
    def l3_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
        
        expr = 12*1j*D1**2*H**2*OM*pk*f*(Qm - xQm)*(d - xd)/(5*G**3*d*k1)
        return expr
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(LxNPP.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l4_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1,zz)
                    
        expr = 18*D1*D1d*Hd**2*OMd*pk*(d - xd)*(-G*(b1*k1**3*(3 - 3*xQm)*(d - xd)**2*(1575*G**4*xd + 525*G**4*(d - xd) - 690*G**2*k1**2*xd*(d - xd)**2 - 240*G**2*k1**2*(d - xd)**3 + 21*k1**4*xd*(d - xd)**4 + 11*k1**4*(d - xd)**5) - f*(3 - 3*Qm)*(88200*G**6*k1*xd + 22050*G**6*k1*(d - xd) - 41175*G**4*k1**3*xd*(d - xd)**2 - 10575*G**4*k1**3*(d - xd)**3 + 2262*G**2*k1**5*xd*(d - xd)**4 + 696*G**2*k1**5*(d - xd)**5 - 25*k1**7*xd*(d - xd)**6 - 13*k1**7*(d - xd)**7) - f*(3 - 3*xQm)*(88200*G**6*k1*xd + 22050*G**6*k1*(d - xd) - 41175*G**4*k1**3*xd*(d - xd)**2 - 10575*G**4*k1**3*(d - xd)**3 + 2262*G**2*k1**5*xd*(d - xd)**4 + 696*G**2*k1**5*(d - xd)**5 - 25*k1**7*xd*(d - xd)**6 - 13*k1**7*(d - xd)**7) + k1**3*xb1*(3 - 3*Qm)*(d - xd)**2*(1575*G**4*xd + 525*G**4*(d - xd) - 690*G**2*k1**2*xd*(d - xd)**2 - 240*G**2*k1**2*(d - xd)**3 + 21*k1**4*xd*(d - xd)**4 + 11*k1**4*(d - xd)**5))*np.sin(k1*(-d + xd)/G) - k1**2*(d - xd)*(b1*k1**2*(3 - 3*xQm)*(d - xd)**2*(1575*G**4*xd - 165*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(525*G**4 - 65*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - f*(3 - 3*Qm)*(88200*G**6*xd - 11775*G**4*k1**2*xd*(d - xd)**2 + 297*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 3225*G**4*k1**2*(d - xd)**2 + 111*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) - f*(3 - 3*xQm)*(88200*G**6*xd - 11775*G**4*k1**2*xd*(d - xd)**2 + 297*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 3225*G**4*k1**2*(d - xd)**2 + 111*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) + k1**2*xb1*(3 - 3*Qm)*(d - xd)**2*(1575*G**4*xd - 165*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(525*G**4 - 65*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)))*np.cos(k1*(-d + xd)/G))/(G*d*k1**10*(-d + xd)**9)
        return expr
    
    @staticmethod
    def l4_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
                    
        expr = -24*D1**2*H**2*OM*pk*f*xd*(d - xd)*(Qm + xQm - 2)/(35*G**3*d)
        return expr
    
class TDxNPP(BaseInt):
    @staticmethod
    def mu_integrand(xd,mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        """2D P(k,mu) power spectra - this returns the integrand - so potentially array (k,mu,xd)"""
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1,zz and mu with xd
        k1,zz,mu = utils.enable_broadcasting(k1,zz,mu,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define cosmo_funcs,zz = args[1,3]G from dirac-delta - could just define q=k1/G
        Pk = baseint.pk(k1/G,zzd1,zz)

        expr = D1*D1d*Pk*((b1 + f*mu**2)*(-1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd**2*OMd*(2*xQm - 2)/(d*k1**2)) + (1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(f*mu**2 + xb1)*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd**2*OMd*(2*Qm - 2)/(d*k1**2)))/G**3
        return expr
    
    @staticmethod
    def mu(mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        """2D P(k,mu) power spectra"""
        return BaseInt.single_int(TDxNPP.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n, remove_div=remove_div)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128,n_mu=16,fast=False):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(TDxNPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n ,n_mu=n_mu,fast=fast)

    ############################ Seperate Multipoles - with analytic mu integration #################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        return BaseInt.single_int(TDxNPP.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)

        expr = 3*D1*D1d*Hd**2*OMd*pk*(-4*G*f*k1*(d - xd)*(Qm + xQm - 2)*np.cos(k1*(-d + xd)/G) - 2*(f*(2*G**2 - k1**2*(d - xd)**2)*(Qm + xQm - 2) + k1**2*(d - xd)**2*(-Qm*xb1 - b1*xQm + b1 + xb1))*np.sin(k1*(-d + xd)/G))/(d*k1**5*(-d + xd)**3)

        return expr
    
    @staticmethod
    def l0_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)

        expr = 2*D1**2*H**2*OM*pk*(3*b1*(Qm - 1) + f*(Qm + xQm - 2) + 3*xb1*(xQm - 1))/(G**3*d*k1**2)

        return expr
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(TDxNPP.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l1_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = 9*1j*D1*D1d*Hd**2*OMd*pk*(-G*(-2*b1*k1**2*(d - xd)**2*(xQm - 1) - 6*f*(2*G**2 - k1**2*(d - xd)**2)*(Qm - xQm) + 2*k1**2*xb1*(Qm - 1)*(d - xd)**2)*np.sin(k1*(d - xd)/G) + k1*(d - xd)*(-2*b1*k1**2*(d - xd)**2*(xQm - 1) - 2*f*(6*G**2 - k1**2*(d - xd)**2)*(Qm - xQm) + 2*k1**2*xb1*(Qm - 1)*(d - xd)**2)*np.cos(k1*(d - xd)/G))/(d*k1**6*(d - xd)**4)
        return expr
    
    @staticmethod
    def l1_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(TDxNPP.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l2_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = -15*D1*D1d*Hd**2*OMd*pk*(2*G*k1*(d - xd)*(-f*(36*G**2 - 5*k1**2*(d - xd)**2)*(Qm + xQm - 2) + 3*k1**2*(d - xd)**2*(b1*(xQm - 1) + xb1*(Qm - 1)))*np.cos(k1*(-d + xd)/G) - 2*(f*(36*G**4 - 17*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)*(Qm + xQm - 2) + k1**2*(-3*G**2 + k1**2*(d - xd)**2)*(d - xd)**2*(b1*(xQm - 1) + xb1*(Qm - 1)))*np.sin(k1*(-d + xd)/G))/(d*k1**7*(-d + xd)**5)
        return expr
    
    @staticmethod
    def l2_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
            
        expr = 4*D1**2*H**2*OM*pk*f*(Qm + xQm - 2)/(G**3*d*k1**2)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(TDxNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l3_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
        
        expr = -21*1j*D1*D1d*Hd**2*OMd*pk*(G*(6*b1*k1**2*(-5*G**2 + 2*k1**2*(d - xd)**2)*(d - xd)**2*(xQm - 1) - 2*f*(Qm - xQm)*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4) - 6*k1**2*xb1*(-5*G**2 + 2*k1**2*(d - xd)**2)*(Qm - 1)*(d - xd)**2)*np.sin(k1*(-d + xd)/G) + k1*(d - xd)*(2*b1*k1**2*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2*(xQm - 1) - 2*f*(Qm - xQm)*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4) - 2*k1**2*xb1*(-15*G**2 + k1**2*(d - xd)**2)*(Qm - 1)*(d - xd)**2)*np.cos(k1*(-d + xd)/G))/(d*k1**8*(d - xd)**6)
        return expr
    
    @staticmethod
    def l3_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(TDxNPP.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l4_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1,zz)
                    
        expr = 27*D1*D1d*Hd**2*OMd*pk*(-2*G*k1*(d - xd)*(3*f*(10*G**2 - k1**2*(d - xd)**2)*(105*G**2 - 4*k1**2*(d - xd)**2)*(Qm + xQm - 2) + 5*k1**2*(-21*G**2 + 2*k1**2*(d - xd)**2)*(d - xd)**2*(b1*(xQm - 1) + xb1*(Qm - 1)))*np.cos(k1*(-d + xd)/G) - 2*(f*(Qm + xQm - 2)*(3150*G**6 - 1485*G**4*k1**2*(d - xd)**2 + 87*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6) - k1**2*(d - xd)**2*(b1*(xQm - 1) + xb1*(Qm - 1))*(105*G**4 - 45*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4))*np.sin(k1*(-d + xd)/G))/(d*k1**9*(-d + xd)**7)
        return expr
    
    @staticmethod
    def l4_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    
class ISWxNPP(BaseInt):
    @staticmethod
    def mu_integrand(xd,mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        """2D P(k,mu) power spectra - this returns the integrand - so potentially array (k,mu,xd)"""
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1,zz and mu with xd
        k1,zz,mu = utils.enable_broadcasting(k1,zz,mu,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define cosmo_funcs,zz = args[1,3]G from dirac-delta - could just define q=k1/G
        Pk = baseint.pk(k1/G,zzd1,zz)

        expr = D1*D1d*Pk*((b1 + f*mu**2)*(-1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd**2*OMd*(2*xQm - 2)/(d*k1**2)) + (1j*np.sin(k1*mu*(-d + xd)/G) + np.cos(k1*mu*(-d + xd)/G))*(f*mu**2 + xb1)*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd**2*OMd*(2*Qm - 2)/(d*k1**2)))/G**3
        return expr
    
    @staticmethod
    def mu(mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        """2D P(k,mu) power spectra"""
        return BaseInt.single_int(ISWxNPP.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n, remove_div=remove_div)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128,n_mu=16,fast=False):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(ISWxNPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n ,n_mu=n_mu,fast=fast)

    ############################ Seperate Multipoles - with analytic mu integration #################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        return BaseInt.single_int(ISWxNPP.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)

        expr = 3*D1*D1d*Hd**3*OMd*pk*(fd - 1)*(-2*G*f*k1*(d - xd)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d)*np.cos(k1*(-d + xd)/G) + (-f*(2*G**2 - k1**2*(d - xd)**2)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d) + k1**2*(d - xd)**2*(b1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + xb1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.sin(k1*(-d + xd)/G))/(H**2*d*k1**5*(-d + xd)**3)

        return expr
    
    @staticmethod
    def l0_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)

        expr = D1**2*H*OM*pk*(f - 1)*(3*b1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + f*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d) + 3*xb1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d))/(G**3*d*k1**2)

        return expr
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(ISWxNPP.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l1_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = 9*1j*D1*D1d*Hd**3*OMd*pk*(fd - 1)*(G*(6*G**2*f*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d) + k1**2*(b1 + 3*f)*(d - xd)**2*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + (3*f*(-2*G**2 + k1**2*(d - xd)**2) + k1**2*xb1*(d - xd)**2)*(-H**2*d*(-2*Qm + be) - 2*H*(Qm - 1) + Hp*d))*np.sin(k1*(d - xd)/G) + k1*(d - xd)*(6*G**2*f*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + f*(6*G**2 - k1**2*(d - xd)**2)*(-H**2*d*(-2*Qm + be) - 2*H*(Qm - 1) + Hp*d) + k1**2*xb1*(d - xd)**2*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) + k1**2*(b1 + f)*(d - xd)**2*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*np.cos(k1*(d - xd)/G))/(H**2*d*k1**6*(d - xd)**4)
        return expr
    
    @staticmethod
    def l1_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(ISWxNPP.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l2_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
            
        expr = 15*D1*D1d*Hd**3*OMd*pk*(fd - 1)*(G*k1*(d - xd)*(f*(36*G**2 - 5*k1**2*(d - xd)**2)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d) - 3*k1**2*(d - xd)**2*(b1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + xb1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.cos(k1*(-d + xd)/G) + (f*(36*G**4 - 17*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d) + k1**2*(-3*G**2 + k1**2*(d - xd)**2)*(d - xd)**2*(b1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + xb1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.sin(k1*(-d + xd)/G))/(H**2*d*k1**7*(-d + xd)**5)
        return expr
    
    @staticmethod
    def l2_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        OM = cosmo_funcs.Om_m(zz)
        G = 1; xd = d; # if you want to remove these parameters in future
        pk = baseint.pk(k1,zz)
            
        expr = 2*D1**2*H*OM*pk*f*(f - 1)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d)/(G**3*d*k1**2)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(ISWxNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
    @staticmethod
    def l3_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1,zz)
        
        expr = -21*1j*D1*D1d*Hd**3*OMd*pk*(1 - fd)*(G*(-3*b1*k1**2*(5*G**2 - 2*k1**2*(d - xd)**2)*(d - xd)**2*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d) + f*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4) + f*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d) - 3*k1**2*xb1*(5*G**2 - 2*k1**2*(d - xd)**2)*(d - xd)**2*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*np.sin(k1*(-d + xd)/G) + k1*(d - xd)*((H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)*(b1*k1**2*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2 + f*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) + (f*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4) + k1**2*xb1*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*np.cos(k1*(-d + xd)/G))/(H**2*d*k1**8*(d - xd)**6)
        return expr
    
    @staticmethod
    def l3_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(ISWxNPP.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l4_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1,zz)
                    
        expr = 27*D1*D1d*Hd**3*OMd*pk*(fd - 1)*(-G*k1*(d - xd)*(3*f*(10*G**2 - k1**2*(d - xd)**2)*(105*G**2 - 4*k1**2*(d - xd)**2)*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d) + 5*k1**2*(-21*G**2 + 2*k1**2*(d - xd)**2)*(d - xd)**2*(b1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + xb1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)))*np.cos(k1*(-d + xd)/G) + (-f*(H**2*d*(-2*Qm + be - 2*xQm + xbe) + 2*H*(Qm + xQm - 2) - 2*Hp*d)*(3150*G**6 - 1485*G**4*k1**2*(d - xd)**2 + 87*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6) + k1**2*(d - xd)**2*(b1*(H**2*d*(-2*xQm + xbe) + 2*H*(xQm - 1) - Hp*d) + xb1*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(105*G**4 - 45*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4))*np.sin(k1*(-d + xd)/G))/(H**2*d*k1**9*(-d + xd)**7)
        return expr
    
    @staticmethod
    def l4_source(cosmo_funcs, k1, zz=0, t=0, sigma=None):
        return 0
    