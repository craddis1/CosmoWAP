import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils
from cosmo_wap.lib import integrate

class IntNPP(BaseInt):
    @staticmethod
    def mu_integrand(xd,mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        """2D P(k,mu) power spectra - this returns the integrand - so potentially array (k,mu,xd)"""
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1,zz and mu with xd
        k1,zz,mu = utils.enable_broadcasting(k1,zz,mu,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define cosmo_funcs,zz = args[1,3]G from dirac-delta - could just define q=k1/G
        Pk = baseint.pk(k1/G,zzd1)

        expr = Pk*(D1*D1d*(b1 + f*mu**2)*(1j*np.sin(k1*mu*(d - xd)/G) + np.cos(k1*mu*(d - xd)/G))*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 6*G**2*Hd**2*OMd*(Qm - 1)/(d*k1**2) + 3*Hd**2*OMd*xd*(Qm - 1)*(d - xd)*(2*1j*G*mu/(k1*xd) - mu**2 + 1)/d) + D1*D1d*(-1j*np.sin(k1*mu*(d - xd)/G) + np.cos(k1*mu*(d - xd)/G))*(f*mu**2 + xb1)*(3*G**2*Hd**3*OMd*(fd - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**2 + 6*G**2*Hd**2*OMd*(xQm - 1)/(d*k1**2) + 3*Hd**2*OMd*xd*(d - xd)*(xQm - 1)*(-2*1j*G*mu/(k1*xd) - mu**2 + 1)/d))/G**3
        return expr
    
    @staticmethod
    def mu(mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        """2D P(k,mu) power spectra"""
        return BaseInt.single_int(IntNPP.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128,n_mu=16,fast=False):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(IntNPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n ,n_mu=n_mu,fast=fast)

    ############################ Seperate Multipoles - with analytic mu integration #################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l0_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)

        expr = 3*D1*D1d*Hd**2*OMd*pk*(-2*H*f*xd*(2*G**2*(8*H - Hd*fd + Hd) - k1**2*xd**2*(7*H - Hd*fd + Hd))*(Qm + xQm - 2) + 2*H*k1**2*xd**3*(b1*(Qm - 1) + xb1*(xQm - 1))*(H - Hd*fd + Hd) + Hd*d**4*k1**2*(fd - 1)*(-2*H**2*Qm*f + H**2*be*f - 2*H**2*f*xQm + H**2*f*xbe - 2*H**2*xQm*xb1 + H**2*xb1*xbe - 2*Hp*f - Hp*xb1 + b1*(-2*H**2*Qm + H**2*be - Hp)) + d**3*k1**2*(Hd*(fd - 1)*(b1*(-3*H**2*xd*(-2*Qm + be) + 2*H*(Qm - 1) + 3*Hp*xd) + xb1*(-3*H**2*xd*(-2*xQm + xbe) + 2*H*(xQm - 1) + 3*Hp*xd)) + f*(H**2*(-3*Hd*be*xd*(fd - 1) + 6*Hd*fd*xQm*xd - 3*Hd*fd*xbe*xd - 6*Hd*xQm*xd + 3*Hd*xbe*xd + Qm*(6*Hd*xd*(fd - 1) - 4) - 4*xQm + 8) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 6*Hd*Hp*xd*(fd - 1))) + d**2*(-H**2*Hd*be*(fd - 1)*(2*G**2*f - 3*b1*k1**2*xd**2 - 3*f*k1**2*xd**2) + f*(2*G**2*Hd*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) + k1**2*xd*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + Qm*(-6*Hd*xd*(fd - 1) + 22) + 22*xQm - 44) - 6*H*Hd*(fd - 1)*(Qm + xQm - 2) - 6*Hd*Hp*xd*(fd - 1))) + k1**2*xd*(b1*(H**2*(Qm*(-6*Hd*xd*(fd - 1) + 2) - 2) - 6*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) + xb1*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 6*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)))) + d*(f*(-2*G**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Hd*fd*xQm*xd - Hd*fd*xbe*xd - 2*Hd*xQm*xd + Hd*xbe*xd + 2*Qm*(Hd*xd*(fd - 1) - 2) - 4*xQm + 8) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1)) + k1**2*xd**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Hd*fd*xQm*xd - Hd*fd*xbe*xd - 2*Hd*xQm*xd + Hd*xbe*xd + 2*Qm*(Hd*xd*(fd - 1) - 16) - 32*xQm + 64) + 6*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1))) + k1**2*xd**2*(b1*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + xb1*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 4*xQm + 4) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)))))*np.sin(k1*(d - xd)/G)/(H**2*d*k1**5*(d - xd)**4) + 6*D1*D1d*Hd**2*OMd*pk*(H**2*d**3*k1**2*(b1*(Qm - 1) + f*(Qm + xQm - 2) + xb1*(xQm - 1)) - 2*H*xd*(H*k1**2*xd**2*(b1*(Qm - 1) + xb1*(xQm - 1)) - f*(G**2*(8*H - Hd*fd + Hd) - H*k1**2*xd**2)*(Qm + xQm - 2)) + d**2*(G**2*H**2*Hd*be*f*(fd - 1) - G**2*Hd*f*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) - 4*H**2*f*k1**2*xd*(Qm + xQm - 2) + 4*H**2*k1**2*xd*(-Qm*b1 + b1 - xQm*xb1 + xb1)) + d*(5*H**2*k1**2*xd**2*(b1*(Qm - 1) + xb1*(xQm - 1)) + f*(G**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Hd*fd*xQm*xd - Hd*fd*xbe*xd - 2*Hd*xQm*xd + Hd*xbe*xd + 2*Qm*(Hd*xd*(fd - 1) - 2) - 4*xQm + 8) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1)) + 5*H**2*k1**2*xd**2*(Qm + xQm - 2))))*np.cos(k1*(d - xd)/G)/(G*H**2*d*k1**4*(d - xd)**3)

        return expr
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l1_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)
            
        expr = 9*1j*D1*D1d*Hd**2*OMd*pk*(G*H**2*f*k1**4*(Qm - 1)*(d - xd)**4*(2*d - 7*xd)*np.sin(k1*(d - xd)/G) - G*H**2*f*k1**4*(d - xd)**4*(2*d - 7*xd)*(xQm - 1)*np.sin(k1*(d - xd)/G) + G*k1**2*(d - xd)**2*(2*H**2*b1*d**3*k1**2*(Qm - 1) + H*xd*(Qm - 1)*(6*G**2*f*(13*H - Hd*fd + Hd) - 5*H*b1*k1**2*xd**2 + 3*H*f*k1**2*xd**2) + 3*d**2*(G**2*H**2*Hd*be*f*(fd - 1) - G**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) - 3*H**2*b1*k1**2*xd*(Qm - 1) + H**2*f*k1**2*xd*(Qm - 1)) + 3*d*(G**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 3) + 6) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 4*H**2*b1*k1**2*xd**2*(Qm - 1) - 2*H**2*f*k1**2*xd**2*(Qm - 1)))*np.sin(k1*(d - xd)/G) - G*k1**2*(d - xd)**2*(2*H**2*d**3*k1**2*xb1*(xQm - 1) + H*xd*(xQm - 1)*(6*G**2*f*(13*H - Hd*fd + Hd) + 3*H*f*k1**2*xd**2 - 5*H*k1**2*xb1*xd**2) - 3*d**2*(G**2*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - H**2*f*k1**2*xd*(xQm - 1) + 3*H**2*k1**2*xb1*xd*(xQm - 1)) - 3*d*(G**2*f*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 6*xQm - 6) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*H**2*f*k1**2*xd**2*(xQm - 1) - 4*H**2*k1**2*xb1*xd**2*(xQm - 1)))*np.sin(k1*(d - xd)/G) - G*(H*xd*(Qm - 1)*(-b1*k1**2*xd**2*(2*G**2*(4*H - Hd*fd + Hd) + H*k1**2*xd**2) + 6*f*(2*G**4*(13*H - Hd*fd + Hd) + G**2*H*k1**2*xd**2)) + b1*d**4*k1**2*(-G**2*H**2*Hd*be*(fd - 1) + G**2*Hd*(fd - 1)*(2*H**2*Qm + Hp) - H**2*k1**2*xd*(Qm - 1)) + b1*d**3*k1**2*(G**2*(H**2*(3*Hd*be*xd*(fd - 1) + Qm*(-6*Hd*xd*(fd - 1) + 2) - 2) - 2*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) + 4*H**2*k1**2*xd**2*(Qm - 1)) + 3*d**2*(-2*G**4*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) + G**2*H**2*Hd*be*(fd - 1)*(2*G**2*f - b1*k1**2*xd**2) + 2*G**2*H**2*f*k1**2*xd*(Qm - 1) + G**2*b1*k1**2*xd*(2*H**2*(Qm*(Hd*xd*(fd - 1) - 2) + 2) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 2*H**2*b1*k1**4*xd**3*(Qm - 1)) + d*(6*G**4*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 3) + 6) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 12*G**2*H**2*f*k1**2*xd**2*(Qm - 1) + G**2*b1*k1**2*xd**2*(H**2*(Hd*be*xd*(fd - 1) + Qm*(-2*Hd*xd*(fd - 1) + 18) - 18) - 6*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*xd*(fd - 1)) + 4*H**2*b1*k1**4*xd**4*(Qm - 1)))*np.sin(k1*(d - xd)/G) + G*(H*xd*(xQm - 1)*(6*f*(2*G**4*(13*H - Hd*fd + Hd) + G**2*H*k1**2*xd**2) - k1**2*xb1*xd**2*(2*G**2*(4*H - Hd*fd + Hd) + H*k1**2*xd**2)) + d**4*k1**2*xb1*(G**2*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - H**2*k1**2*xd*(xQm - 1)) + d**3*k1**2*xb1*(G**2*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 2*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) + 4*H**2*k1**2*xd**2*(xQm - 1)) - 3*d**2*(2*G**4*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - 2*G**2*H**2*f*k1**2*xd*(xQm - 1) + G**2*k1**2*xb1*xd*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*H**2*k1**4*xb1*xd**3*(xQm - 1)) + d*(-6*G**2*f*(G**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 6*xQm - 6) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*H**2*k1**2*xd**2*(xQm - 1)) + k1**2*xb1*xd**2*(G**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 6*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 4*H**2*k1**2*xd**2*(xQm - 1))))*np.sin(k1*(d - xd)/G) + H**2*f*k1**5*xd*(Qm - 1)*(d - xd)**5*np.cos(k1*(d - xd)/G) - H**2*f*k1**5*xd*(d - xd)**5*(xQm - 1)*np.cos(k1*(d - xd)/G) - k1**3*(d - xd)**3*(H*xd*(Qm - 1)*(2*G**2*f*(13*H - Hd*fd + Hd) - H*b1*k1**2*xd**2 + H*f*k1**2*xd**2) + d**2*(G**2*H**2*Hd*be*f*(fd - 1) - G**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) - H**2*b1*k1**2*xd*(Qm - 1) + H**2*f*k1**2*xd*(Qm - 1)) + d*(G**2*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 3) + 6) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 2*H**2*b1*k1**2*xd**2*(Qm - 1) - 2*H**2*f*k1**2*xd**2*(Qm - 1)))*np.cos(k1*(d - xd)/G) + k1**3*(d - xd)**3*(H*xd*(xQm - 1)*(2*G**2*f*(13*H - Hd*fd + Hd) + H*f*k1**2*xd**2 - H*k1**2*xb1*xd**2) + d**2*(-G**2*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) + H**2*f*k1**2*xd*(xQm - 1) - H**2*k1**2*xb1*xd*(xQm - 1)) + d*(G**2*f*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 6*xQm + 6) + 2*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) - 2*H**2*f*k1**2*xd**2*(xQm - 1) + 2*H**2*k1**2*xb1*xd**2*(xQm - 1)))*np.cos(k1*(d - xd)/G) - k1*(d - xd)*(-H*xd*(Qm - 1)*(-b1*k1**2*xd**2*(2*G**2*(4*H - Hd*fd + Hd) + H*k1**2*xd**2) + 6*f*(2*G**4*(13*H - Hd*fd + Hd) + G**2*H*k1**2*xd**2)) + b1*d**4*k1**2*(G**2*H**2*Hd*be*(fd - 1) - G**2*Hd*(fd - 1)*(2*H**2*Qm + Hp) + H**2*k1**2*xd*(Qm - 1)) + b1*d**3*k1**2*(G**2*(H**2*(-3*Hd*be*xd*(fd - 1) + Qm*(6*Hd*xd*(fd - 1) - 2) + 2) + 2*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1)) - 4*H**2*k1**2*xd**2*(Qm - 1)) - 3*d**2*(-2*G**4*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) + G**2*H**2*Hd*be*(fd - 1)*(2*G**2*f - b1*k1**2*xd**2) + 2*G**2*H**2*f*k1**2*xd*(Qm - 1) + G**2*b1*k1**2*xd*(2*H**2*(Qm*(Hd*xd*(fd - 1) - 2) + 2) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 2*H**2*b1*k1**4*xd**3*(Qm - 1)) + d*(-6*G**4*f*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 3) + 6) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 12*G**2*H**2*f*k1**2*xd**2*(Qm - 1) + G**2*b1*k1**2*xd**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 9) + 18) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 4*H**2*b1*k1**4*xd**4*(Qm - 1)))*np.cos(k1*(d - xd)/G) - k1*(d - xd)*(H*xd*(xQm - 1)*(6*f*(2*G**4*(13*H - Hd*fd + Hd) + G**2*H*k1**2*xd**2) - k1**2*xb1*xd**2*(2*G**2*(4*H - Hd*fd + Hd) + H*k1**2*xd**2)) + d**4*k1**2*xb1*(G**2*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - H**2*k1**2*xd*(xQm - 1)) + d**3*k1**2*xb1*(G**2*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 2*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) + 4*H**2*k1**2*xd**2*(xQm - 1)) - 3*d**2*(2*G**4*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - 2*G**2*H**2*f*k1**2*xd*(xQm - 1) + G**2*k1**2*xb1*xd*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*H**2*k1**4*xb1*xd**3*(xQm - 1)) + d*(-6*G**2*f*(G**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 6*xQm - 6) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*H**2*k1**2*xd**2*(xQm - 1)) + k1**2*xb1*xd**2*(G**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 6*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 4*H**2*k1**2*xd**2*(xQm - 1))))*np.cos(k1*(d - xd)/G))/(G**2*H**2*d*k1**6*(d - xd)**5)
        return expr
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l2_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d_,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)
            
        expr = 3*1j*D1*D1d*Hd**2*OMd*pk*((-1j*np.sin(k1*(d - xd)/G) + np.cos(k1*(d - xd)/G))*(G*d**4*k1**2*(b1 + f)*(G*H**2*Hd*be*(fd - 1) - G*Hd*(fd - 1)*(2*H**2*Qm + Hp) - 2*1j*H**2*k1*(Qm - 1)) + G*d**3*k1*(2*1j*G**2*Hd*f*(fd - 1)*(2*H**2*Qm + Hp) - 1j*G*H**2*Hd*be*f*(2*G - 3*1j*k1*xd)*(fd - 1) + G*Hd*b1*k1*(fd - 1)*(-3*H**2*xd*(-2*Qm + be) + 2*H*(Qm - 1) + 3*Hp*xd) + G*f*k1*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 4) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1)) + 10*1j*H**2*b1*k1**2*xd*(Qm - 1) + 10*1j*H**2*f*k1**2*xd*(Qm - 1)) + G*d**2*(G*H**2*Hd*be*(fd - 1)*(3*b1*k1**2*xd**2 + f*(-2*G**2 + 4*1j*G*k1*xd + 3*k1**2*xd**2)) + b1*k1**2*xd*(G*(H**2*(Qm*(-6*Hd*xd*(fd - 1) + 2) - 2) - 6*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) - 18*1j*H**2*k1*xd*(Qm - 1)) + f*(2*G**3*Hd*(fd - 1)*(2*H**2*Qm + Hp) - 4*1j*G**2*k1*(2*H**2*(Qm*(Hd*xd*(fd - 1) - 1) + 1) + H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + G*k1**2*xd*(H**2*(Qm*(-6*Hd*xd*(fd - 1) + 22) - 22) - 6*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) - 18*1j*H**2*k1**3*xd**2*(Qm - 1))) - H*xd*(Qm - 1)*(-2*G*b1*k1**2*xd**2*(G*(H - Hd*fd + Hd) - 2*1j*H*k1*xd) + 2*G*f*(2*G**3*(8*H - Hd*fd + Hd) - 2*1j*G**2*k1*xd*(8*H - Hd*fd + Hd) - G*k1**2*xd**2*(7*H - Hd*fd + Hd) + 2*1j*H*k1**3*xd**3)) + d*(G*b1*k1**2*xd**2*(G*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 14*1j*H**2*k1*xd*(Qm - 1)) + G*f*(-2*G**3*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 2*1j*G**2*k1*xd*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 10) + 20) + 4*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + G*k1**2*xd**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 16) + 32) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + 14*1j*H**2*k1**3*xd**3*(Qm - 1)))) + (-1j*np.sin(k1*(d - xd)/G) + np.cos(k1*(d - xd)/G))*(-G*d**4*k1**2*(f + xb1)*(G*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) + 2*1j*H**2*k1*(xQm - 1)) + G*d**3*k1*(2*1j*G**2*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) + G*k1*(H**2*(-3*Hd*xd*(f + xb1)*(fd - 1)*(-2*xQm + xbe) - 4*f*(xQm - 1)) + 2*H*Hd*(f + xb1)*(fd - 1)*(xQm - 1) + 3*Hd*Hp*xd*(f + xb1)*(fd - 1)) + 10*1j*H**2*k1**2*xd*(f + xb1)*(xQm - 1)) - H*xd*(xQm - 1)*(2*G*f*(2*G**3*(8*H - Hd*fd + Hd) - 2*1j*G**2*k1*xd*(8*H - Hd*fd + Hd) - G*k1**2*xd**2*(7*H - Hd*fd + Hd) + 2*1j*H*k1**3*xd**3) - 2*G*k1**2*xb1*xd**2*(G*(H - Hd*fd + Hd) - 2*1j*H*k1*xd)) + d**2*(G*f*(2*G**3*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) + 4*1j*G**2*k1*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + H*Hd*(-fd*xQm + fd + xQm - 1) - Hd*Hp*xd*(fd - 1)) + G*k1**2*xd*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 6*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) - 18*1j*H**2*k1**3*xd**2*(xQm - 1)) + G*k1**2*xb1*xd*(G*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 6*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) - 18*1j*H**2*k1*xd*(xQm - 1))) + d*(G*f*(2*G**3*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) - 2*1j*G**2*k1*xd*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 20*xQm - 20) - 4*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + G*k1**2*xd**2*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 32*xQm + 32) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + 14*1j*H**2*k1**3*xd**3*(xQm - 1)) + G*k1**2*xb1*xd**2*(G*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 4*xQm + 4) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + 14*1j*H**2*k1*xd*(xQm - 1)))) - (1j*np.sin(k1*(d - xd)/G) + np.cos(k1*(d - xd)/G))*(G*d**4*k1**2*(b1 + f)*(G*H**2*Hd*be*(fd - 1) - G*Hd*(fd - 1)*(2*H**2*Qm + Hp) + 2*1j*H**2*k1*(Qm - 1)) + G*d**2*(G*H**2*Hd*be*(fd - 1)*(3*b1*k1**2*xd**2 + f*(-2*G**2 - 4*1j*G*k1*xd + 3*k1**2*xd**2)) + b1*k1**2*xd*(G*(H**2*(Qm*(-6*Hd*xd*(fd - 1) + 2) - 2) - 6*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) + 18*1j*H**2*k1*xd*(Qm - 1)) + f*(2*G**3*Hd*(fd - 1)*(2*H**2*Qm + Hp) + 4*1j*G**2*k1*(2*H**2*(Qm*(Hd*xd*(fd - 1) - 1) + 1) + H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + G*k1**2*xd*(H**2*(Qm*(-6*Hd*xd*(fd - 1) + 22) - 22) - 6*H*Hd*(Qm - 1)*(fd - 1) - 3*Hd*Hp*xd*(fd - 1)) + 18*1j*H**2*k1**3*xd**2*(Qm - 1))) - H*xd*(Qm - 1)*(-2*G*b1*k1**2*xd**2*(G*(H - Hd*fd + Hd) + 2*1j*H*k1*xd) + 2*G*f*(2*G**3*(8*H - Hd*fd + Hd) + 2*1j*G**2*k1*xd*(8*H - Hd*fd + Hd) + G*k1**2*xd**2*(-7*H + Hd*(fd - 1)) - 2*1j*H*k1**3*xd**3)) + d**3*k1*(G*b1*k1*(G*Hd*(fd - 1)*(-3*H**2*xd*(-2*Qm + be) + 2*H*(Qm - 1) + 3*Hp*xd) - 10*1j*H**2*k1*xd*(Qm - 1)) + G*f*(-2*1j*G**2*Hd*(fd - 1)*(2*H**2*Qm + Hp) + 1j*G*H**2*Hd*be*(2*G + 3*1j*k1*xd)*(fd - 1) + G*k1*(H**2*(Qm*(6*Hd*xd*(fd - 1) - 4) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + 3*Hd*Hp*xd*(fd - 1)) - 10*1j*H**2*k1**2*xd*(Qm - 1))) + d*(G*b1*k1**2*xd**2*(G*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 14*1j*H**2*k1*xd*(Qm - 1)) + G*f*(-2*G**3*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 2) + 4) + 2*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 2*1j*G**2*k1*xd*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 10) + 20) + 4*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) + G*k1**2*xd**2*(H**2*(-Hd*be*xd*(fd - 1) + 2*Qm*(Hd*xd*(fd - 1) - 16) + 32) + 6*H*Hd*(Qm - 1)*(fd - 1) + Hd*Hp*xd*(fd - 1)) - 14*1j*H**2*k1**3*xd**3*(Qm - 1)))) - (1j*np.sin(k1*(d - xd)/G) + np.cos(k1*(d - xd)/G))*(-G*d**4*k1**2*(f + xb1)*(G*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - 2*1j*H**2*k1*(xQm - 1)) + G*d**3*k1*(-2*1j*G**2*Hd*f*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) + G*k1*(H**2*(-3*Hd*xd*(f + xb1)*(fd - 1)*(-2*xQm + xbe) - 4*f*(xQm - 1)) + 2*H*Hd*(f + xb1)*(fd - 1)*(xQm - 1) + 3*Hd*Hp*xd*(f + xb1)*(fd - 1)) - 10*1j*H**2*k1**2*xd*(f + xb1)*(xQm - 1)) - H*xd*(xQm - 1)*(2*G*f*(2*G**3*(8*H - Hd*fd + Hd) + 2*1j*G**2*k1*xd*(8*H - Hd*fd + Hd) + G*k1**2*xd**2*(-7*H + Hd*(fd - 1)) - 2*1j*H*k1**3*xd**3) - 2*G*k1**2*xb1*xd**2*(G*(H - Hd*fd + Hd) + 2*1j*H*k1*xd)) + d**2*(G*f*(2*G**3*Hd*(fd - 1)*(-H**2*(-2*xQm + xbe) + Hp) - 4*1j*G**2*k1*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + H*Hd*(-fd*xQm + fd + xQm - 1) - Hd*Hp*xd*(fd - 1)) + G*k1**2*xd*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 22*xQm - 22) - 6*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) + 18*1j*H**2*k1**3*xd**2*(xQm - 1)) + G*k1**2*xb1*xd*(G*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) - 6*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) + 18*1j*H**2*k1*xd*(xQm - 1))) + d*(G*f*(2*G**3*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 4*xQm - 4) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 2*1j*G**2*k1*xd*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 20*xQm - 20) - 4*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + G*k1**2*xd**2*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 32*xQm + 32) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) - 14*1j*H**2*k1**3*xd**3*(xQm - 1)) + G*k1**2*xb1*xd**2*(G*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 4*xQm + 4) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) - 14*1j*H**2*k1*xd*(xQm - 1)))))/(2*G**2*H**2*d*k1**5*(d - xd)**4)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l3_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)
        
        expr = 1
        return expr
    
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l4_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)
                    
        expr = 1
        return expr
    
class IntInt(BaseInt):
    @staticmethod
    def mu_integrand(xd1,xd2,mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, fast=True,**kwargs):
        """2D P(k,mu) power spectra - this returns the integrand - so returns array (k,mu,xd1,xd2)"""
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1,zz and mu with xd
        k1,zz,mu = utils.enable_broadcasting(k1,zz,mu,n=1)
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        def int_terms1(xd1, xd2, mu, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta 
            Pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = D1d1*D1d2*Pk*(1j*np.sin(k1*mu*(-xd1 + xd2)/G) + np.cos(k1*mu*(-xd1 + xd2)/G))*(3*G**2*Hd1**3*OMd1*(fd1 - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd1**2*OMd1*(2*Qm - 2)/(d*k1**2) + Hd1**2*OMd1*xd1*(3*Qm - 3)*(d - xd1)*(-2*1j*G*mu/(k1*xd1) - mu**2 + 1)/d)*(3*G**2*Hd2**3*OMd2*(fd2 - 1)*(-2*xQm + xbe + (2*xQm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd2**2*OMd2*(2*xQm - 2)/(d*k1**2) + Hd2**2*OMd2*xd2*(d - xd2)*(3*xQm - 3)*(2*1j*G*mu/(k1*xd2) - mu**2 + 1)/d)/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, mu, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            G = xd1/d # Define G from dirac-delta 
            Pk = baseint.pk(k1/G,zzd1)
            
            expr = D1d1**2*Pk*(3*G**2*Hd1**3*OMd1*(fd1 - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd1**2*OMd1*(2*Qm - 2)/(d*k1**2) + Hd1**2*OMd1*xd1*(3*Qm - 3)*(d - xd1)*(-2*1j*G*mu/(k1*xd1) - mu**2 + 1)/d)*(3*G**2*Hd1**3*OMd1*(fd1 - 1)*(-2*Qm + be + (2*Qm - 2)/(H*d) - Hp/H**2)/k1**2 + 3*G**2*Hd1**2*OMd1*(2*Qm - 2)/(d*k1**2) + Hd1**2*OMd1*xd1*(3*Qm - 3)*(d - xd1)*(2*1j*G*mu/(k1*xd1) - mu**2 + 1)/d)/G**3

            return expr
        
        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2,int_terms1,mu,cosmo_funcs,k1,zz,fast=fast,**kwargs) # parse functions as well
        
    @staticmethod
    def mu(mu, cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, fast=True):
        """2D P(k,mu) power spectra - returns 2D array (k,mu)"""
        return BaseInt.double_int(IntInt.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n, fast=fast)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n_mu=16, fast=False): # fast here has half of mu
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(IntInt.mu,l,cosmo_funcs, k1, zz, t, sigma, n=n ,n_mu=n_mu,fast=fast)
    
    ############################################ Individual Multipoles #############################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None,fast=True):
        return BaseInt.double_int(IntInt.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)
        
    @staticmethod
    def l0_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        
        baseint = BaseInt(cosmo_funcs)

        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params 

        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta 
            pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = D1d1*D1d2*(-6*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(6*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1**2 + 4*xd1*xd2 + xd2**2) + xd1*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)))*np.cos(k1*(-xd1 + xd2)/G)/(H**2*d**2*k1**4*(xd1 - xd2)**4) + 3*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(2*G**2*H**2*xd1*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 2*G**2*H**2*xd2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 3*G**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-G**2*(xd1**2 + 4*xd1*xd2 + xd2**2) + 2*k1**2*xd1*xd2*(xd1 - xd2)**2))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**5*(-xd1 + xd2)**5))*baseint.pk(k1/G,zzd1,zzd2)/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            _, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(45*G**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 60*G**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 10*G**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 10*G**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G,zzd1)/(5*G**3*H**4*d**2*k1**4)
            
            return expr
            
        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None, fast=True):
        return BaseInt.double_int(IntInt.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2,fast=fast)
        
    @staticmethod    
    def l1_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
            pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = -9*1j*D1d1*D1d2*Hd1**2*Hd2**2*OMd1*OMd2*(G*(24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) - 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(G**2*(xd1 + 2*xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(G**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-G**2*(2*xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.sin(k1*(-xd1 + xd2)/G)/(xd1 - xd2)**6 + k1*(2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(xd1 + 2*xd2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(2*xd1 + xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 3*G**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-3*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + k1**2*xd1*xd2*(xd1 - xd2)**2))*np.cos(k1*(-xd1 + xd2)/G)/(-xd1 + xd2)**5)*pk/(H**4*d**2*k1**6)

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            _, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 18*1j*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(5*G**2*(Qm - 1)*(d - xd1)*(H**2*(-Hd2*d*(fd2 - 1)*(-2*xQm + xbe) - 2*xQm + 2) - 2*H*Hd2*(fd2 - 1)*(xQm - 1) + Hd2*Hp*d*(fd2 - 1))/(H**2*k1**2) + 5*G**2*(d - xd2)*(xQm - 1)*(H**2*(Hd1*be*d*(fd1 - 1) + 2*Qm*d*(-Hd1*fd1 + Hd1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))/(H**2*k1**2) + 2*xd1*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 2*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G,zzd1)/(5*G**2*d**2*k1)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    @staticmethod
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None, fast=True):
        return BaseInt.double_int(IntInt.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)
        
    @staticmethod    
    def l2_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
             
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
            pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = D1d1*D1d2*(-15*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-27*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) - 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(3*G**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-9*G**2*(xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**6*(xd1 - xd2)**6) + 15*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G**2*(3*G**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(27*G**4*(xd1**2 + 3*xd1*xd2 + xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(11*xd1**2 + 35*xd1*xd2 + 11*xd2**2) + k1**4*xd1*xd2*(xd1 - xd2)**4))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G,zzd1,zzd2)/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            _, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 2*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(84*G**2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 7*G**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 7*G**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**2*k1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G,zzd1)/(7*G**3*H**2*d**2*k1**2)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    @staticmethod
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None, fast=True):
        return BaseInt.double_int(IntInt.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)
        
    @staticmethod    
    def l3_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
            pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = D1d1*D1d2*(-21*1j*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d))*(30*G**4*(3*xd1 + 2*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4) + (xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(9*G**2*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(3 - 3*xQm)*(d - xd2)*(30*G**4*(2*xd1 + 3*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**8) - 21*1j*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(30*G**2*(3*xd1 + 2*xd2) - k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(30*G**2*(2*xd1 + 3*xd2) - k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G**2*(15*G**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G,zzd1,zzd2)/G**3

            return expr

        # for when xd1 == xd2 
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 2*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(84*G**2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 7*G**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 7*G**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**2*k1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G,zzd1)/(7*G**3*H**2*d**2*k1**2)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
    
    @staticmethod
    def l4(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, n2=None, fast=True):
        return BaseInt.double_int(IntInt.l4_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, n2=n2, fast=fast)
        
    @staticmethod    
    def l4_integrand(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None, fast=True,**kwargs):
        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz)
            
        # for when xd1 != xd2
        def int_terms1(xd1, xd2, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            zzd2, fd2, D1d2, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd2)
            G = (xd1 + xd2) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
            pk = baseint.pk(k1/G,zzd1,zzd2)

            expr = D1d1*D1d2*(-21*1j*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d))*(30*G**4*(3*xd1 + 2*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4) + (xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(9*G**2*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 2*H**2*(3 - 3*xQm)*(d - xd2)*(30*G**4*(2*xd1 + 3*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**8) - 21*1j*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(30*G**2*(3*xd1 + 2*xd2) - k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(30*G**2*(2*xd1 + 3*xd2) - k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G**2*(15*G**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*baseint.pk(k1/G,zzd1,zzd2)/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            _, fd2, _, Hd2, OMd2 = BaseInt.get_integrand_params(cosmo_funcs, xd1) # TODO: should not need to call this function again - all parameters here should be the d1 versions
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 9*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(525*G**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 1260*G**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 70*G**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 70*G**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 136*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*baseint.pk(k1/G,zzd1)/(70*G**3*H**4*d**2*k1**4)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
