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
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
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

        G = (d + xd) / (2 * d) # Define G from dirac-delta
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

        G = (d + xd) / (2 * d) # Define G from dirac-delta
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
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)
            
        expr = 15*D1*D1d*Hd**2*OMd*pk*(G*(2*H*xd*(f*(Qm + xQm - 2)*(36*G**4*(19*H - Hd*fd + Hd) + G**2*k1**2*xd**2*(-316*H + 17*Hd*(fd - 1)) + k1**4*xd**4*(16*H - Hd*fd + Hd)) + k1**2*xd**2*(-3*G**2*(8*H - Hd*fd + Hd) + k1**2*xd**2*(10*H - Hd*fd + Hd))*(b1*(Qm - 1) + xb1*(xQm - 1))) + Hd*d**6*k1**4*(fd - 1)*(-2*H**2*Qm*f + H**2*b1*(-2*Qm + be) + H**2*be*f + H**2*f*xbe - 2*H**2*xQm*(f + xb1) + H**2*xb1*xbe - Hp*b1 - 2*Hp*f - Hp*xb1) + d**5*k1**4*(-2*H*b1*(Qm - 1)*(3*H - Hd*fd + Hd) + 5*Hd*b1*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + f*(5*H**2*(-Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(Hd*xd*(fd - 1) - 1) - 2*xQm + 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 10*Hd*Hp*xd*(fd - 1)) + xb1*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 6*xQm + 6) + 2*H*Hd*(fd - 1)*(xQm - 1) + 5*Hd*Hp*xd*(fd - 1))) + d**4*k1**2*(34*G**2*H**2*Hd*Qm*f*fd - 34*G**2*H**2*Hd*Qm*f - 17*G**2*H**2*Hd*f*fd*xbe + 17*G**2*H**2*Hd*f*xbe - 3*G**2*H**2*Hd*fd*xb1*xbe + 3*G**2*H**2*Hd*xb1*xbe + 34*G**2*Hd*Hp*f*fd - 34*G**2*Hd*Hp*f + 3*G**2*Hd*Hp*fd*xb1 - 3*G**2*Hd*Hp*xb1 - 20*H**2*Hd*Qm*f*fd*k1**2*xd**2 + 20*H**2*Hd*Qm*f*k1**2*xd**2 - H**2*Hd*be*f*(17*G**2 - 10*k1**2*xd**2)*(fd - 1) + 10*H**2*Hd*f*fd*k1**2*xbe*xd**2 - 10*H**2*Hd*f*k1**2*xbe*xd**2 + 10*H**2*Hd*fd*k1**2*xb1*xbe*xd**2 - 10*H**2*Hd*k1**2*xb1*xbe*xd**2 + 72*H**2*Qm*f*k1**2*xd - 144*H**2*f*k1**2*xd - 44*H**2*k1**2*xb1*xd - 10*H*Hd*Qm*f*fd*k1**2*xd + 10*H*Hd*Qm*f*k1**2*xd + 20*H*Hd*f*fd*k1**2*xd - 20*H*Hd*f*k1**2*xd + 10*H*Hd*fd*k1**2*xb1*xd - 10*H*Hd*k1**2*xb1*xd + 2*H*xQm*(G**2*H*Hd*(17*f + 3*xb1)*(fd - 1) - 10*H*Hd*k1**2*xd**2*(f + xb1)*(fd - 1) + k1**2*xd*(36*H*f + 22*H*xb1 - 5*Hd*f*(fd - 1) - 5*Hd*xb1*(fd - 1))) - 20*Hd*Hp*f*fd*k1**2*xd**2 + 20*Hd*Hp*f*k1**2*xd**2 - 10*Hd*Hp*fd*k1**2*xb1*xd**2 + 10*Hd*Hp*k1**2*xb1*xd**2 + b1*(3*G**2*Hd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + 2*H*k1**2*xd*(22*H - 5*Hd*(fd - 1))*(Qm - 1) + 10*Hd*k1**2*xd**2*(fd - 1)*(H**2*(-2*Qm + be) - Hp))) + d**3*k1**2*(G**2*f*(H**2*(51*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-51*Hd*xd*(fd - 1) + 70) + 140*xQm - 280) - 34*H*Hd*(fd - 1)*(Qm + xQm - 2) - 102*Hd*Hp*xd*(fd - 1)) + 3*G**2*xb1*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 4*xQm - 4) + 2*H*Hd*(-fd*xQm + fd + xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) + b1*(3*G**2*(2*H*(Qm - 1)*(2*H - Hd*fd + Hd) - 3*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 2*k1**2*xd**2*(-2*H*(Qm - 1)*(29*H - 5*Hd*fd + 5*Hd) + 5*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp))) + 2*f*k1**2*xd**2*(H**2*(-5*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(5*Hd*xd*(fd - 1) - 47) - 94*xQm + 188) + 10*H*Hd*(fd - 1)*(Qm + xQm - 2) + 10*Hd*Hp*xd*(fd - 1)) + 2*k1**2*xb1*xd**2*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 58*xQm + 58) + 10*H*Hd*(fd - 1)*(xQm - 1) + 5*Hd*Hp*xd*(fd - 1))) + d**2*(H**2*Hd*be*(fd - 1)*(36*G**4*f - 3*G**2*k1**2*xd**2*(3*b1 + 17*f) + 5*k1**4*xd**4*(b1 + f)) + f*(-36*G**4*Hd*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) + 3*G**2*k1**2*xd*(H**2*(34*Hd*Qm*xd*(fd - 1) - 17*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 304*Qm - 304*xQm + 608) + 34*H*Hd*(fd - 1)*(Qm + xQm - 2) + 34*Hd*Hp*xd*(fd - 1)) + k1**4*xd**3*(H**2*(5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(-5*Hd*xd*(fd - 1) + 116) + 232*xQm - 464) - 20*H*Hd*(fd - 1)*(Qm + xQm - 2) - 10*Hd*Hp*xd*(fd - 1))) + k1**2*xd*(9*G**2*b1*(-2*H*(Qm - 1)*(4*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 9*G**2*xb1*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 8*xQm + 8) + 2*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + b1*k1**2*xd**2*(4*H*(36*H - 5*Hd*(fd - 1))*(Qm - 1) - 5*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + k1**2*xb1*xd**2*(H**2*(5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 144*xQm - 144) - 20*H*Hd*(fd - 1)*(xQm - 1) - 5*Hd*Hp*xd*(fd - 1)))) + d*(f*(36*G**4*(H**2*(-Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(Hd*xd*(fd - 1) - 4) - 8*xQm + 16) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1)) + G**2*k1**2*xd**2*(H**2*(17*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-17*Hd*xd*(fd - 1) + 702) + 1404*xQm - 2808) - 102*H*Hd*(fd - 1)*(Qm + xQm - 2) - 34*Hd*Hp*xd*(fd - 1)) + k1**4*xd**4*(H**2*(-Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(Hd*xd*(fd - 1) - 69) - 138*xQm + 276) + 10*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1))) + k1**2*xd**2*(3*G**2*b1*(6*H*(Qm - 1)*(6*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(H**2*(-2*Qm + be) - Hp)) + 3*G**2*xb1*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 36*xQm - 36) - 6*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + b1*k1**2*xd**2*(-2*H*(43*H - 5*Hd*(fd - 1))*(Qm - 1) + Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + k1**2*xb1*xd**2*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 86*xQm + 86) + 10*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)))))*np.sin(k1*(d - xd)/G) + k1*(d - xd)*(2*H**2*d**5*k1**4*(b1*(Qm - 1) + f*(Qm + xQm - 2) + xb1*(xQm - 1)) - 2*H*xd*(f*(Qm + xQm - 2)*(36*G**4*(19*H - Hd*fd + Hd) + G**2*k1**2*xd**2*(-88*H + 5*Hd*(fd - 1)) + 2*H*k1**4*xd**4) + k1**2*xd**2*(-3*G**2*(8*H - Hd*fd + Hd) + 2*H*k1**2*xd**2)*(b1*(Qm - 1) + xb1*(xQm - 1))) + d**4*k1**2*(-10*G**2*H**2*Hd*Qm*f*fd + 10*G**2*H**2*Hd*Qm*f + 5*G**2*H**2*Hd*be*f*(fd - 1) + 5*G**2*H**2*Hd*f*fd*xbe - 5*G**2*H**2*Hd*f*xbe + 3*G**2*H**2*Hd*fd*xb1*xbe - 3*G**2*H**2*Hd*xb1*xbe - 10*G**2*Hd*Hp*f*fd + 10*G**2*Hd*Hp*f - 3*G**2*Hd*Hp*fd*xb1 + 3*G**2*Hd*Hp*xb1 - 12*H**2*Qm*f*k1**2*xd + 24*H**2*f*k1**2*xd + 12*H**2*k1**2*xb1*xd - 2*H**2*xQm*(G**2*Hd*(5*f + 3*xb1)*(fd - 1) + 6*k1**2*xd*(f + xb1)) + 3*b1*(G**2*Hd*(fd - 1)*(H**2*(-2*Qm + be) - Hp) - 4*H**2*k1**2*xd*(Qm - 1))) + d**3*k1**2*(G**2*f*(H**2*(30*Hd*Qm*xd*(fd - 1) - 15*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) - 44*Qm - 44*xQm + 88) + 10*H*Hd*(fd - 1)*(Qm + xQm - 2) + 30*Hd*Hp*xd*(fd - 1)) + 3*G**2*xb1*(H**2*(-3*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 4*xQm + 4) + 2*H*Hd*(fd - 1)*(xQm - 1) + 3*Hd*Hp*xd*(fd - 1)) + 28*H**2*f*k1**2*xd**2*(Qm + xQm - 2) + 28*H**2*k1**2*xb1*xd**2*(xQm - 1) + b1*(3*G**2*(-2*H*(Qm - 1)*(2*H - Hd*fd + Hd) + 3*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 28*H**2*k1**2*xd**2*(Qm - 1))) + d**2*(-3*G**2*H**2*Hd*be*(fd - 1)*(12*G**2*f - k1**2*xd**2*(3*b1 + 5*f)) + f*(36*G**4*Hd*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) + 3*G**2*k1**2*xd*(H**2*(5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(-5*Hd*xd*(fd - 1) + 44) + 88*xQm - 176) - 10*H*Hd*(fd - 1)*(Qm + xQm - 2) - 10*Hd*Hp*xd*(fd - 1)) - 32*H**2*k1**4*xd**3*(Qm + xQm - 2)) + k1**2*xd*(9*G**2*b1*(2*H*(Qm - 1)*(4*H - Hd*fd + Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 9*G**2*xb1*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 8*xQm - 8) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) - 32*H**2*b1*k1**2*xd**2*(Qm - 1) - 32*H**2*k1**2*xb1*xd**2*(xQm - 1))) + d*(f*(36*G**4*(H**2*(Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + Qm*(-2*Hd*xd*(fd - 1) + 8) + 8*xQm - 16) - 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*xd*(fd - 1)) + G**2*k1**2*xd**2*(H**2*(-5*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(5*Hd*xd*(fd - 1) - 198) - 396*xQm + 792) + 30*H*Hd*(fd - 1)*(Qm + xQm - 2) + 10*Hd*Hp*xd*(fd - 1)) + 18*H**2*k1**4*xd**4*(Qm + xQm - 2)) + 3*k1**2*xd**2*(G**2*b1*(-6*H*(Qm - 1)*(6*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + G**2*xb1*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 36*xQm + 36) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + 6*H**2*b1*k1**2*xd**2*(Qm - 1) + 6*H**2*k1**2*xb1*xd**2*(xQm - 1))))*np.cos(k1*(d - xd)/G))/(G*H**2*d*k1**7*(d - xd)**6)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128):
        return BaseInt.single_int(IntNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n)
        
    @staticmethod
    def l3_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)
        
        expr = -21*1j*D1*D1d*Hd**2*OMd*pk*(G*k1*(d - xd)*(-600*G**4*H**2*Hd*Qm*d**2*f*fd + 600*G**4*H**2*Hd*Qm*d**2*f + 600*G**4*H**2*Hd*Qm*d*f*fd*xd - 600*G**4*H**2*Hd*Qm*d*f*xd - 300*G**4*H**2*Hd*d**2*f*fd*xbe + 300*G**4*H**2*Hd*d**2*f*xbe + 300*G**4*H**2*Hd*d*f*fd*xbe*xd - 300*G**4*H**2*Hd*d*f*xbe*xd - 3000*G**4*H**2*Qm*d*f + 15600*G**4*H**2*Qm*f*xd + 600*G**4*H*Hd*Qm*d*f*fd - 600*G**4*H*Hd*Qm*d*f - 600*G**4*H*Hd*Qm*f*fd*xd + 600*G**4*H*Hd*Qm*f*xd + 82*G**2*H**2*Hd*Qm*d**4*f*fd*k1**2 - 82*G**2*H**2*Hd*Qm*d**4*f*k1**2 - 246*G**2*H**2*Hd*Qm*d**3*f*fd*k1**2*xd + 246*G**2*H**2*Hd*Qm*d**3*f*k1**2*xd + 246*G**2*H**2*Hd*Qm*d**2*f*fd*k1**2*xd**2 - 246*G**2*H**2*Hd*Qm*d**2*f*k1**2*xd**2 - 82*G**2*H**2*Hd*Qm*d*f*fd*k1**2*xd**3 + 82*G**2*H**2*Hd*Qm*d*f*k1**2*xd**3 + 41*G**2*H**2*Hd*d**4*f*fd*k1**2*xbe - 41*G**2*H**2*Hd*d**4*f*k1**2*xbe + 15*G**2*H**2*Hd*d**4*fd*k1**2*xb1*xbe - 15*G**2*H**2*Hd*d**4*k1**2*xb1*xbe - 123*G**2*H**2*Hd*d**3*f*fd*k1**2*xbe*xd + 123*G**2*H**2*Hd*d**3*f*k1**2*xbe*xd - 45*G**2*H**2*Hd*d**3*fd*k1**2*xb1*xbe*xd + 45*G**2*H**2*Hd*d**3*k1**2*xb1*xbe*xd + 123*G**2*H**2*Hd*d**2*f*fd*k1**2*xbe*xd**2 - 123*G**2*H**2*Hd*d**2*f*k1**2*xbe*xd**2 + 45*G**2*H**2*Hd*d**2*fd*k1**2*xb1*xbe*xd**2 - 45*G**2*H**2*Hd*d**2*k1**2*xb1*xbe*xd**2 - 41*G**2*H**2*Hd*d*f*fd*k1**2*xbe*xd**3 + 41*G**2*H**2*Hd*d*f*k1**2*xbe*xd**3 - 15*G**2*H**2*Hd*d*fd*k1**2*xb1*xbe*xd**3 + 15*G**2*H**2*Hd*d*k1**2*xb1*xbe*xd**3 + 446*G**2*H**2*Qm*d**3*f*k1**2 - 2958*G**2*H**2*Qm*d**2*f*k1**2*xd + 4578*G**2*H**2*Qm*d*f*k1**2*xd**2 - 2066*G**2*H**2*Qm*f*k1**2*xd**3 + 90*G**2*H**2*d**3*k1**2*xb1 - 570*G**2*H**2*d**2*k1**2*xb1*xd + 870*G**2*H**2*d*k1**2*xb1*xd**2 - 390*G**2*H**2*k1**2*xb1*xd**3 - 82*G**2*H*Hd*Qm*d**3*f*fd*k1**2 + 82*G**2*H*Hd*Qm*d**3*f*k1**2 + 246*G**2*H*Hd*Qm*d**2*f*fd*k1**2*xd - 246*G**2*H*Hd*Qm*d**2*f*k1**2*xd - 246*G**2*H*Hd*Qm*d*f*fd*k1**2*xd**2 + 246*G**2*H*Hd*Qm*d*f*k1**2*xd**2 + 82*G**2*H*Hd*Qm*f*fd*k1**2*xd**3 - 82*G**2*H*Hd*Qm*f*k1**2*xd**3 - 30*G**2*H*Hd*d**3*fd*k1**2*xb1 + 30*G**2*H*Hd*d**3*k1**2*xb1 + 90*G**2*H*Hd*d**2*fd*k1**2*xb1*xd - 90*G**2*H*Hd*d**2*k1**2*xb1*xd - 90*G**2*H*Hd*d*fd*k1**2*xb1*xd**2 + 90*G**2*H*Hd*d*k1**2*xb1*xd**2 + 30*G**2*H*Hd*fd*k1**2*xb1*xd**3 - 30*G**2*H*Hd*k1**2*xb1*xd**3 - 15*G**2*Hd*Hp*d**4*fd*k1**2*xb1 + 15*G**2*Hd*Hp*d**4*k1**2*xb1 + 45*G**2*Hd*Hp*d**3*fd*k1**2*xb1*xd - 45*G**2*Hd*Hp*d**3*k1**2*xb1*xd - 45*G**2*Hd*Hp*d**2*fd*k1**2*xb1*xd**2 + 45*G**2*Hd*Hp*d**2*k1**2*xb1*xd**2 + 15*G**2*Hd*Hp*d*fd*k1**2*xb1*xd**3 - 15*G**2*Hd*Hp*d*k1**2*xb1*xd**3 - 2*H**2*Hd*Qm*d**6*f*fd*k1**4 + 2*H**2*Hd*Qm*d**6*f*k1**4 + 10*H**2*Hd*Qm*d**5*f*fd*k1**4*xd - 10*H**2*Hd*Qm*d**5*f*k1**4*xd - 20*H**2*Hd*Qm*d**4*f*fd*k1**4*xd**2 + 20*H**2*Hd*Qm*d**4*f*k1**4*xd**2 + 20*H**2*Hd*Qm*d**3*f*fd*k1**4*xd**3 - 20*H**2*Hd*Qm*d**3*f*k1**4*xd**3 - 10*H**2*Hd*Qm*d**2*f*fd*k1**4*xd**4 + 10*H**2*Hd*Qm*d**2*f*k1**4*xd**4 + 2*H**2*Hd*Qm*d*f*fd*k1**4*xd**5 - 2*H**2*Hd*Qm*d*f*k1**4*xd**5 + H**2*Hd*be*d*(d - xd)*(fd - 1)*(b1*k1**2*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2 + f*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - H**2*Hd*d**6*f*fd*k1**4*xbe + H**2*Hd*d**6*f*k1**4*xbe - H**2*Hd*d**6*fd*k1**4*xb1*xbe + H**2*Hd*d**6*k1**4*xb1*xbe + 5*H**2*Hd*d**5*f*fd*k1**4*xbe*xd - 5*H**2*Hd*d**5*f*k1**4*xbe*xd + 5*H**2*Hd*d**5*fd*k1**4*xb1*xbe*xd - 5*H**2*Hd*d**5*k1**4*xb1*xbe*xd - 10*H**2*Hd*d**4*f*fd*k1**4*xbe*xd**2 + 10*H**2*Hd*d**4*f*k1**4*xbe*xd**2 - 10*H**2*Hd*d**4*fd*k1**4*xb1*xbe*xd**2 + 10*H**2*Hd*d**4*k1**4*xb1*xbe*xd**2 + 10*H**2*Hd*d**3*f*fd*k1**4*xbe*xd**3 - 10*H**2*Hd*d**3*f*k1**4*xbe*xd**3 + 10*H**2*Hd*d**3*fd*k1**4*xb1*xbe*xd**3 - 10*H**2*Hd*d**3*k1**4*xb1*xbe*xd**3 - 5*H**2*Hd*d**2*f*fd*k1**4*xbe*xd**4 + 5*H**2*Hd*d**2*f*k1**4*xbe*xd**4 - 5*H**2*Hd*d**2*fd*k1**4*xb1*xbe*xd**4 + 5*H**2*Hd*d**2*k1**4*xb1*xbe*xd**4 + H**2*Hd*d*f*fd*k1**4*xbe*xd**5 - H**2*Hd*d*f*k1**4*xbe*xd**5 + H**2*Hd*d*fd*k1**4*xb1*xbe*xd**5 - H**2*Hd*d*k1**4*xb1*xbe*xd**5 - 16*H**2*Qm*d**5*f*k1**4 + 114*H**2*Qm*d**4*f*k1**4*xd - 296*H**2*Qm*d**3*f*k1**4*xd**2 + 364*H**2*Qm*d**2*f*k1**4*xd**3 - 216*H**2*Qm*d*f*k1**4*xd**4 + 50*H**2*Qm*f*k1**4*xd**5 - 12*H**2*d**5*k1**4*xb1 + 86*H**2*d**4*k1**4*xb1*xd - 224*H**2*d**3*k1**4*xb1*xd**2 + 276*H**2*d**2*k1**4*xb1*xd**3 - 164*H**2*d*k1**4*xb1*xd**4 + 38*H**2*k1**4*xb1*xd**5 + 2*H*Hd*Qm*d**5*f*fd*k1**4 - 2*H*Hd*Qm*d**5*f*k1**4 - 10*H*Hd*Qm*d**4*f*fd*k1**4*xd + 10*H*Hd*Qm*d**4*f*k1**4*xd + 20*H*Hd*Qm*d**3*f*fd*k1**4*xd**2 - 20*H*Hd*Qm*d**3*f*k1**4*xd**2 - 20*H*Hd*Qm*d**2*f*fd*k1**4*xd**3 + 20*H*Hd*Qm*d**2*f*k1**4*xd**3 + 10*H*Hd*Qm*d*f*fd*k1**4*xd**4 - 10*H*Hd*Qm*d*f*k1**4*xd**4 - 2*H*Hd*Qm*f*fd*k1**4*xd**5 + 2*H*Hd*Qm*f*k1**4*xd**5 + 2*H*Hd*d**5*fd*k1**4*xb1 - 2*H*Hd*d**5*k1**4*xb1 - 10*H*Hd*d**4*fd*k1**4*xb1*xd + 10*H*Hd*d**4*k1**4*xb1*xd + 20*H*Hd*d**3*fd*k1**4*xb1*xd**2 - 20*H*Hd*d**3*k1**4*xb1*xd**2 - 20*H*Hd*d**2*fd*k1**4*xb1*xd**3 + 20*H*Hd*d**2*k1**4*xb1*xd**3 + 10*H*Hd*d*fd*k1**4*xb1*xd**4 - 10*H*Hd*d*k1**4*xb1*xd**4 - 2*H*Hd*fd*k1**4*xb1*xd**5 + 2*H*Hd*k1**4*xb1*xd**5 + 2*H*xQm*(-300*G**4*f*xd*(26*H - Hd*fd + Hd) + G**2*k1**2*xd**3*(1033*H*f - 41*Hd*f*(fd - 1) + 15*xb1*(13*H - Hd*fd + Hd)) + H*Hd*d**6*k1**4*(f + xb1)*(fd - 1) + d**5*k1**4*(-5*H*Hd*xd*(f + xb1)*(fd - 1) + 8*H*f + 6*H*xb1 - Hd*f*fd + Hd*f - Hd*fd*xb1 + Hd*xb1) + d**4*k1**2*(-G**2*H*Hd*(41*f + 15*xb1)*(fd - 1) + 10*H*Hd*k1**2*xd**2*(f + xb1)*(fd - 1) - k1**2*xd*(57*H*f + 43*H*xb1 - 5*Hd*f*(fd - 1) - 5*Hd*xb1*(fd - 1))) + d**3*k1**2*(3*G**2*H*Hd*xd*(41*f + 15*xb1)*(fd - 1) - G**2*(223*H*f - 41*Hd*f*(fd - 1) + 15*xb1*(3*H - Hd*fd + Hd)) - 10*H*Hd*k1**2*xd**3*(f + xb1)*(fd - 1) + 2*k1**2*xd**2*(74*H*f + 56*H*xb1 - 5*Hd*f*(fd - 1) - 5*Hd*xb1*(fd - 1))) + d**2*(300*G**4*H*Hd*f*(fd - 1) - 3*G**2*H*Hd*k1**2*xd**2*(41*f + 15*xb1)*(fd - 1) + 3*G**2*k1**2*xd*(493*H*f + 95*H*xb1 - 41*Hd*f*(fd - 1) - 15*Hd*xb1*(fd - 1)) + 5*H*Hd*k1**4*xd**4*(f + xb1)*(fd - 1) - 2*k1**4*xd**3*(91*H*f + 69*H*xb1 - 5*Hd*f*(fd - 1) - 5*Hd*xb1*(fd - 1))) + d*(-300*G**4*H*Hd*f*xd*(fd - 1) + 300*G**4*f*(5*H - Hd*fd + Hd) + G**2*H*Hd*k1**2*xd**3*(41*f + 15*xb1)*(fd - 1) - 3*G**2*k1**2*xd**2*(763*H*f - 41*Hd*f*(fd - 1) + 5*xb1*(29*H - 3*Hd*(fd - 1))) - H*Hd*k1**4*xd**5*(f + xb1)*(fd - 1) + k1**4*xd**4*(108*H*f + 82*H*xb1 - 5*Hd*f*(fd - 1) - 5*Hd*xb1*(fd - 1))) - k1**4*xd**5*(f*(25*H - Hd*fd + Hd) + xb1*(19*H - Hd*fd + Hd))) + Hd*Hp*d**6*fd*k1**4*xb1 - Hd*Hp*d**6*k1**4*xb1 - 5*Hd*Hp*d**5*fd*k1**4*xb1*xd + 5*Hd*Hp*d**5*k1**4*xb1*xd + 10*Hd*Hp*d**4*fd*k1**4*xb1*xd**2 - 10*Hd*Hp*d**4*k1**4*xb1*xd**2 - 10*Hd*Hp*d**3*fd*k1**4*xb1*xd**3 + 10*Hd*Hp*d**3*k1**4*xb1*xd**3 + 5*Hd*Hp*d**2*fd*k1**4*xb1*xd**4 - 5*Hd*Hp*d**2*k1**4*xb1*xd**4 - Hd*Hp*d*fd*k1**4*xb1*xd**5 + Hd*Hp*d*k1**4*xb1*xd**5 - b1*k1**2*(d - xd)**2*(2*H*xd*(Qm - 1)*(15*G**2*(13*H - Hd*fd + Hd) - k1**2*xd**2*(19*H - Hd*fd + Hd)) + Hd*d**4*k1**2*(fd - 1)*(2*H**2*Qm + Hp) + d**3*k1**2*(2*H*(Qm - 1)*(6*H - Hd*fd + Hd) - 3*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + d**2*(-15*G**2*Hd*(fd - 1)*(2*H**2*Qm + Hp) + k1**2*xd*(-2*H*(Qm - 1)*(31*H - 3*Hd*fd + 3*Hd) + 3*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp))) + d*(15*G**2*(-2*H*(Qm - 1)*(3*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + k1**2*xd**2*(2*H*(Qm - 1)*(44*H - 3*Hd*fd + 3*Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)))))*np.cos(k1*(d - xd)/G) + (600*G**6*H**2*Hd*Qm*d**2*f*fd - 600*G**6*H**2*Hd*Qm*d**2*f - 600*G**6*H**2*Hd*Qm*d*f*fd*xd + 600*G**6*H**2*Hd*Qm*d*f*xd + 300*G**6*H**2*Hd*d**2*f*fd*xbe - 300*G**6*H**2*Hd*d**2*f*xbe - 300*G**6*H**2*Hd*d*f*fd*xbe*xd + 300*G**6*H**2*Hd*d*f*xbe*xd + 3000*G**6*H**2*Qm*d*f - 15600*G**6*H**2*Qm*f*xd - 600*G**6*H*Hd*Qm*d*f*fd + 600*G**6*H*Hd*Qm*d*f + 600*G**6*H*Hd*Qm*f*fd*xd - 600*G**6*H*Hd*Qm*f*xd - 282*G**4*H**2*Hd*Qm*d**4*f*fd*k1**2 + 282*G**4*H**2*Hd*Qm*d**4*f*k1**2 + 846*G**4*H**2*Hd*Qm*d**3*f*fd*k1**2*xd - 846*G**4*H**2*Hd*Qm*d**3*f*k1**2*xd - 846*G**4*H**2*Hd*Qm*d**2*f*fd*k1**2*xd**2 + 846*G**4*H**2*Hd*Qm*d**2*f*k1**2*xd**2 + 282*G**4*H**2*Hd*Qm*d*f*fd*k1**2*xd**3 - 282*G**4*H**2*Hd*Qm*d*f*k1**2*xd**3 - 141*G**4*H**2*Hd*d**4*f*fd*k1**2*xbe + 141*G**4*H**2*Hd*d**4*f*k1**2*xbe - 15*G**4*H**2*Hd*d**4*fd*k1**2*xb1*xbe + 15*G**4*H**2*Hd*d**4*k1**2*xb1*xbe + 423*G**4*H**2*Hd*d**3*f*fd*k1**2*xbe*xd - 423*G**4*H**2*Hd*d**3*f*k1**2*xbe*xd + 45*G**4*H**2*Hd*d**3*fd*k1**2*xb1*xbe*xd - 45*G**4*H**2*Hd*d**3*k1**2*xb1*xbe*xd - 423*G**4*H**2*Hd*d**2*f*fd*k1**2*xbe*xd**2 + 423*G**4*H**2*Hd*d**2*f*k1**2*xbe*xd**2 - 45*G**4*H**2*Hd*d**2*fd*k1**2*xb1*xbe*xd**2 + 45*G**4*H**2*Hd*d**2*k1**2*xb1*xbe*xd**2 + 141*G**4*H**2*Hd*d*f*fd*k1**2*xbe*xd**3 - 141*G**4*H**2*Hd*d*f*k1**2*xbe*xd**3 + 15*G**4*H**2*Hd*d*fd*k1**2*xb1*xbe*xd**3 - 15*G**4*H**2*Hd*d*k1**2*xb1*xbe*xd**3 - 1446*G**4*H**2*Qm*d**3*f*k1**2 + 10158*G**4*H**2*Qm*d**2*f*k1**2*xd - 15978*G**4*H**2*Qm*d*f*k1**2*xd**2 + 7266*G**4*H**2*Qm*f*k1**2*xd**3 - 90*G**4*H**2*d**3*k1**2*xb1 + 570*G**4*H**2*d**2*k1**2*xb1*xd - 870*G**4*H**2*d*k1**2*xb1*xd**2 + 390*G**4*H**2*k1**2*xb1*xd**3 + 282*G**4*H*Hd*Qm*d**3*f*fd*k1**2 - 282*G**4*H*Hd*Qm*d**3*f*k1**2 - 846*G**4*H*Hd*Qm*d**2*f*fd*k1**2*xd + 846*G**4*H*Hd*Qm*d**2*f*k1**2*xd + 846*G**4*H*Hd*Qm*d*f*fd*k1**2*xd**2 - 846*G**4*H*Hd*Qm*d*f*k1**2*xd**2 - 282*G**4*H*Hd*Qm*f*fd*k1**2*xd**3 + 282*G**4*H*Hd*Qm*f*k1**2*xd**3 + 30*G**4*H*Hd*d**3*fd*k1**2*xb1 - 30*G**4*H*Hd*d**3*k1**2*xb1 - 90*G**4*H*Hd*d**2*fd*k1**2*xb1*xd + 90*G**4*H*Hd*d**2*k1**2*xb1*xd + 90*G**4*H*Hd*d*fd*k1**2*xb1*xd**2 - 90*G**4*H*Hd*d*k1**2*xb1*xd**2 - 30*G**4*H*Hd*fd*k1**2*xb1*xd**3 + 30*G**4*H*Hd*k1**2*xb1*xd**3 + 15*G**4*Hd*Hp*d**4*fd*k1**2*xb1 - 15*G**4*Hd*Hp*d**4*k1**2*xb1 - 45*G**4*Hd*Hp*d**3*fd*k1**2*xb1*xd + 45*G**4*Hd*Hp*d**3*k1**2*xb1*xd + 45*G**4*Hd*Hp*d**2*fd*k1**2*xb1*xd**2 - 45*G**4*Hd*Hp*d**2*k1**2*xb1*xd**2 - 15*G**4*Hd*Hp*d*fd*k1**2*xb1*xd**3 + 15*G**4*Hd*Hp*d*k1**2*xb1*xd**3 + 16*G**2*H**2*Hd*Qm*d**6*f*fd*k1**4 - 16*G**2*H**2*Hd*Qm*d**6*f*k1**4 - 80*G**2*H**2*Hd*Qm*d**5*f*fd*k1**4*xd + 80*G**2*H**2*Hd*Qm*d**5*f*k1**4*xd + 160*G**2*H**2*Hd*Qm*d**4*f*fd*k1**4*xd**2 - 160*G**2*H**2*Hd*Qm*d**4*f*k1**4*xd**2 - 160*G**2*H**2*Hd*Qm*d**3*f*fd*k1**4*xd**3 + 160*G**2*H**2*Hd*Qm*d**3*f*k1**4*xd**3 + 80*G**2*H**2*Hd*Qm*d**2*f*fd*k1**4*xd**4 - 80*G**2*H**2*Hd*Qm*d**2*f*k1**4*xd**4 - 16*G**2*H**2*Hd*Qm*d*f*fd*k1**4*xd**5 + 16*G**2*H**2*Hd*Qm*d*f*k1**4*xd**5 - G**2*H**2*Hd*be*d*(d - xd)*(fd - 1)*(3*b1*k1**2*(-5*G**2 + 2*k1**2*(d - xd)**2)*(d - xd)**2 + f*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4)) + 8*G**2*H**2*Hd*d**6*f*fd*k1**4*xbe - 8*G**2*H**2*Hd*d**6*f*k1**4*xbe + 6*G**2*H**2*Hd*d**6*fd*k1**4*xb1*xbe - 6*G**2*H**2*Hd*d**6*k1**4*xb1*xbe - 40*G**2*H**2*Hd*d**5*f*fd*k1**4*xbe*xd + 40*G**2*H**2*Hd*d**5*f*k1**4*xbe*xd - 30*G**2*H**2*Hd*d**5*fd*k1**4*xb1*xbe*xd + 30*G**2*H**2*Hd*d**5*k1**4*xb1*xbe*xd + 80*G**2*H**2*Hd*d**4*f*fd*k1**4*xbe*xd**2 - 80*G**2*H**2*Hd*d**4*f*k1**4*xbe*xd**2 + 60*G**2*H**2*Hd*d**4*fd*k1**4*xb1*xbe*xd**2 - 60*G**2*H**2*Hd*d**4*k1**4*xb1*xbe*xd**2 - 80*G**2*H**2*Hd*d**3*f*fd*k1**4*xbe*xd**3 + 80*G**2*H**2*Hd*d**3*f*k1**4*xbe*xd**3 - 60*G**2*H**2*Hd*d**3*fd*k1**4*xb1*xbe*xd**3 + 60*G**2*H**2*Hd*d**3*k1**4*xb1*xbe*xd**3 + 40*G**2*H**2*Hd*d**2*f*fd*k1**4*xbe*xd**4 - 40*G**2*H**2*Hd*d**2*f*k1**4*xbe*xd**4 + 30*G**2*H**2*Hd*d**2*fd*k1**4*xb1*xbe*xd**4 - 30*G**2*H**2*Hd*d**2*k1**4*xb1*xbe*xd**4 - 8*G**2*H**2*Hd*d*f*fd*k1**4*xbe*xd**5 + 8*G**2*H**2*Hd*d*f*k1**4*xbe*xd**5 - 6*G**2*H**2*Hd*d*fd*k1**4*xb1*xbe*xd**5 + 6*G**2*H**2*Hd*d*k1**4*xb1*xbe*xd**5 + 98*G**2*H**2*Qm*d**5*f*k1**4 - 784*G**2*H**2*Qm*d**4*f*k1**4*xd + 2156*G**2*H**2*Qm*d**3*f*k1**4*xd**2 - 2744*G**2*H**2*Qm*d**2*f*k1**4*xd**3 + 1666*G**2*H**2*Qm*d*f*k1**4*xd**4 - 392*G**2*H**2*Qm*f*k1**4*xd**5 + 42*G**2*H**2*d**5*k1**4*xb1 - 336*G**2*H**2*d**4*k1**4*xb1*xd + 924*G**2*H**2*d**3*k1**4*xb1*xd**2 - 1176*G**2*H**2*d**2*k1**4*xb1*xd**3 + 714*G**2*H**2*d*k1**4*xb1*xd**4 - 168*G**2*H**2*k1**4*xb1*xd**5 - 16*G**2*H*Hd*Qm*d**5*f*fd*k1**4 + 16*G**2*H*Hd*Qm*d**5*f*k1**4 + 80*G**2*H*Hd*Qm*d**4*f*fd*k1**4*xd - 80*G**2*H*Hd*Qm*d**4*f*k1**4*xd - 160*G**2*H*Hd*Qm*d**3*f*fd*k1**4*xd**2 + 160*G**2*H*Hd*Qm*d**3*f*k1**4*xd**2 + 160*G**2*H*Hd*Qm*d**2*f*fd*k1**4*xd**3 - 160*G**2*H*Hd*Qm*d**2*f*k1**4*xd**3 - 80*G**2*H*Hd*Qm*d*f*fd*k1**4*xd**4 + 80*G**2*H*Hd*Qm*d*f*k1**4*xd**4 + 16*G**2*H*Hd*Qm*f*fd*k1**4*xd**5 - 16*G**2*H*Hd*Qm*f*k1**4*xd**5 - 12*G**2*H*Hd*d**5*fd*k1**4*xb1 + 12*G**2*H*Hd*d**5*k1**4*xb1 + 60*G**2*H*Hd*d**4*fd*k1**4*xb1*xd - 60*G**2*H*Hd*d**4*k1**4*xb1*xd - 120*G**2*H*Hd*d**3*fd*k1**4*xb1*xd**2 + 120*G**2*H*Hd*d**3*k1**4*xb1*xd**2 + 120*G**2*H*Hd*d**2*fd*k1**4*xb1*xd**3 - 120*G**2*H*Hd*d**2*k1**4*xb1*xd**3 - 60*G**2*H*Hd*d*fd*k1**4*xb1*xd**4 + 60*G**2*H*Hd*d*k1**4*xb1*xd**4 + 12*G**2*H*Hd*fd*k1**4*xb1*xd**5 - 12*G**2*H*Hd*k1**4*xb1*xd**5 - 6*G**2*Hd*Hp*d**6*fd*k1**4*xb1 + 6*G**2*Hd*Hp*d**6*k1**4*xb1 + 30*G**2*Hd*Hp*d**5*fd*k1**4*xb1*xd - 30*G**2*Hd*Hp*d**5*k1**4*xb1*xd - 60*G**2*Hd*Hp*d**4*fd*k1**4*xb1*xd**2 + 60*G**2*Hd*Hp*d**4*k1**4*xb1*xd**2 + 60*G**2*Hd*Hp*d**3*fd*k1**4*xb1*xd**3 - 60*G**2*Hd*Hp*d**3*k1**4*xb1*xd**3 - 30*G**2*Hd*Hp*d**2*fd*k1**4*xb1*xd**4 + 30*G**2*Hd*Hp*d**2*k1**4*xb1*xd**4 + 6*G**2*Hd*Hp*d*fd*k1**4*xb1*xd**5 - 6*G**2*Hd*Hp*d*k1**4*xb1*xd**5 - 2*H**2*Qm*d**7*f*k1**6 + 16*H**2*Qm*d**6*f*k1**6*xd - 54*H**2*Qm*d**5*f*k1**6*xd**2 + 100*H**2*Qm*d**4*f*k1**6*xd**3 - 110*H**2*Qm*d**3*f*k1**6*xd**4 + 72*H**2*Qm*d**2*f*k1**6*xd**5 - 26*H**2*Qm*d*f*k1**6*xd**6 + 4*H**2*Qm*f*k1**6*xd**7 - 2*H**2*d**7*k1**6*xb1 + 16*H**2*d**6*k1**6*xb1*xd - 54*H**2*d**5*k1**6*xb1*xd**2 + 100*H**2*d**4*k1**6*xb1*xd**3 - 110*H**2*d**3*k1**6*xb1*xd**4 + 72*H**2*d**2*k1**6*xb1*xd**5 - 26*H**2*d*k1**6*xb1*xd**6 + 4*H**2*k1**6*xb1*xd**7 + 2*H*xQm*(300*G**6*f*xd*(26*H - Hd*fd + Hd) - 3*G**4*k1**2*xd**3*(1211*H*f - 47*Hd*f*(fd - 1) + 5*xb1*(13*H - Hd*fd + Hd)) + 2*G**2*k1**4*xd**5*(98*H*f - 4*Hd*f*(fd - 1) + 3*xb1*(14*H - Hd*fd + Hd)) + H*d**7*k1**6*(f + xb1) + 2*H*d**6*k1**4*(-G**2*Hd*(4*f + 3*xb1)*(fd - 1) - 4*k1**2*xd*(f + xb1)) - 2*H*k1**6*xd**7*(f + xb1) + d**5*k1**4*(10*G**2*H*Hd*xd*(4*f + 3*xb1)*(fd - 1) - G**2*(49*H*f + 21*H*xb1 - 8*Hd*f*(fd - 1) - 6*Hd*xb1*(fd - 1)) + 27*H*k1**2*xd**2*(f + xb1)) + d**4*k1**2*(3*G**4*H*Hd*(47*f + 5*xb1)*(fd - 1) - 20*G**2*H*Hd*k1**2*xd**2*(4*f + 3*xb1)*(fd - 1) + 2*G**2*k1**2*xd*(196*H*f + 84*H*xb1 - 20*Hd*f*(fd - 1) - 15*Hd*xb1*(fd - 1)) - 50*H*k1**4*xd**3*(f + xb1)) + d**3*k1**2*(-9*G**4*H*Hd*xd*(47*f + 5*xb1)*(fd - 1) + 3*G**4*(241*H*f - 47*Hd*f*(fd - 1) + 5*xb1*(3*H - Hd*fd + Hd)) + 20*G**2*H*Hd*k1**2*xd**3*(4*f + 3*xb1)*(fd - 1) - 2*G**2*k1**2*xd**2*(539*H*f + 231*H*xb1 - 40*Hd*f*(fd - 1) - 30*Hd*xb1*(fd - 1)) + 55*H*k1**4*xd**4*(f + xb1)) + d**2*(-300*G**6*H*Hd*f*(fd - 1) + 9*G**4*H*Hd*k1**2*xd**2*(47*f + 5*xb1)*(fd - 1) - 3*G**4*k1**2*xd*(1693*H*f + 95*H*xb1 - 141*Hd*f*(fd - 1) - 15*Hd*xb1*(fd - 1)) - 10*G**2*H*Hd*k1**4*xd**4*(4*f + 3*xb1)*(fd - 1) + 4*G**2*k1**4*xd**3*(343*H*f - 20*Hd*f*(fd - 1) + 3*xb1*(49*H - 5*Hd*(fd - 1))) - 36*H*k1**6*xd**5*(f + xb1)) + d*(300*G**6*H*Hd*f*xd*(fd - 1) - 300*G**6*f*(5*H - Hd*fd + Hd) - 3*G**4*H*Hd*k1**2*xd**3*(47*f + 5*xb1)*(fd - 1) + 3*G**4*k1**2*xd**2*(2663*H*f - 141*Hd*f*(fd - 1) + 5*xb1*(29*H - 3*Hd*(fd - 1))) + 2*G**2*H*Hd*k1**4*xd**5*(4*f + 3*xb1)*(fd - 1) - G**2*k1**4*xd**4*(833*H*f + 357*H*xb1 - 40*Hd*f*(fd - 1) - 30*Hd*xb1*(fd - 1)) + 13*H*k1**6*xd**6*(f + xb1))) - b1*k1**2*(d - xd)**2*(2*H**2*d**5*k1**4*(Qm - 1) + 2*H*xd*(Qm - 1)*(-15*G**4*(13*H - Hd*fd + Hd) + 6*G**2*k1**2*xd**2*(14*H - Hd*fd + Hd) - 2*H*k1**4*xd**4) + 6*d**4*k1**2*(-G**2*Hd*(fd - 1)*(2*H**2*Qm + Hp) - 2*H**2*k1**2*xd*(Qm - 1)) + 2*d**3*k1**2*(3*G**2*(-H*(Qm - 1)*(7*H - 2*Hd*fd + 2*Hd) + 3*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 14*H**2*k1**2*xd**2*(Qm - 1)) + d**2*(15*G**4*Hd*(fd - 1)*(2*H**2*Qm + Hp) + 18*G**2*k1**2*xd*(2*H*(Qm - 1)*(7*H - Hd*fd + Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) - 32*H**2*k1**4*xd**3*(Qm - 1)) + 3*d*(5*G**4*(2*H*(Qm - 1)*(3*H - Hd*fd + Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 2*G**2*k1**2*xd**2*(-3*H*(Qm - 1)*(21*H - 2*Hd*fd + 2*Hd) + Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 6*H**2*k1**4*xd**4*(Qm - 1))))*np.sin(k1*(d - xd)/G))/(G*H**2*d*k1**8*(d - xd)**7)
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
                    
        expr = 27*D1*D1d*Hd**2*OMd*pk*(G*(2*H*xd*(-f*(Qm + xQm - 2)*(3150*G**6*(34*H - Hd*fd + Hd) - 45*G**4*k1**2*xd**2*(1117*H - 33*Hd*(fd - 1)) + 87*G**2*k1**4*xd**4*(33*H - Hd*fd + Hd) - k1**6*xd**6*(37*H - Hd*fd + Hd)) + k1**2*xd**2*(b1*(Qm - 1) + xb1*(xQm - 1))*(105*G**4*(19*H - Hd*fd + Hd) - 15*G**2*k1**2*xd**2*(59*H - 3*Hd*fd + 3*Hd) + k1**4*xd**4*(31*H - Hd*fd + Hd))) + Hd*d**8*k1**6*(fd - 1)*(-2*H**2*Qm*f + H**2*b1*(-2*Qm + be) + H**2*be*f + H**2*f*xbe - 2*H**2*xQm*(f + xb1) + H**2*xb1*xbe - Hp*b1 - 2*Hp*f - Hp*xb1) + d**7*k1**6*(-2*H*b1*(Qm - 1)*(10*H - Hd*fd + Hd) + 7*Hd*b1*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + f*(H**2*(-7*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(7*Hd*xd*(fd - 1) - 12) - 24*xQm + 48) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 14*Hd*Hp*xd*(fd - 1)) + xb1*(H**2*(-7*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 20*xQm + 20) + 2*H*Hd*(fd - 1)*(xQm - 1) + 7*Hd*Hp*xd*(fd - 1))) + d**6*k1**4*(174*G**2*H**2*Hd*Qm*f*fd - 174*G**2*H**2*Hd*Qm*f - 87*G**2*H**2*Hd*f*fd*xbe + 87*G**2*H**2*Hd*f*xbe - 45*G**2*H**2*Hd*fd*xb1*xbe + 45*G**2*H**2*Hd*xb1*xbe + 174*G**2*Hd*Hp*f*fd - 174*G**2*Hd*Hp*f + 45*G**2*Hd*Hp*fd*xb1 - 45*G**2*Hd*Hp*xb1 - 42*H**2*Hd*Qm*f*fd*k1**2*xd**2 + 42*H**2*Hd*Qm*f*k1**2*xd**2 - 3*H**2*Hd*be*f*(29*G**2 - 7*k1**2*xd**2)*(fd - 1) + 21*H**2*Hd*f*fd*k1**2*xbe*xd**2 - 21*H**2*Hd*f*k1**2*xbe*xd**2 + 21*H**2*Hd*fd*k1**2*xb1*xbe*xd**2 - 21*H**2*Hd*k1**2*xb1*xbe*xd**2 + 218*H**2*Qm*f*k1**2*xd - 436*H**2*f*k1**2*xd - 182*H**2*k1**2*xb1*xd - 14*H*Hd*Qm*f*fd*k1**2*xd + 14*H*Hd*Qm*f*k1**2*xd + 28*H*Hd*f*fd*k1**2*xd - 28*H*Hd*f*k1**2*xd + 14*H*Hd*fd*k1**2*xb1*xd - 14*H*Hd*k1**2*xb1*xd + 2*H*xQm*(3*G**2*H*Hd*(29*f + 15*xb1)*(fd - 1) - 21*H*Hd*k1**2*xd**2*(f + xb1)*(fd - 1) + k1**2*xd*(109*H*f - 7*Hd*f*(fd - 1) + 7*xb1*(13*H - Hd*fd + Hd))) - 42*Hd*Hp*f*fd*k1**2*xd**2 + 42*Hd*Hp*f*k1**2*xd**2 - 21*Hd*Hp*fd*k1**2*xb1*xd**2 + 21*Hd*Hp*k1**2*xb1*xd**2 + b1*(45*G**2*Hd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + 14*H*k1**2*xd*(Qm - 1)*(13*H - Hd*fd + Hd) + 21*Hd*k1**2*xd**2*(fd - 1)*(H**2*(-2*Qm + be) - Hp))) + d**5*k1**4*(87*G**2*f*(H**2*(5*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-5*Hd*xd*(fd - 1) + 7) + 14*xQm - 28) - 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 10*Hd*Hp*xd*(fd - 1)) + 15*G**2*xb1*(H**2*(15*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 26*xQm - 26) + 6*H*Hd*(-fd*xQm + fd + xQm - 1) - 15*Hd*Hp*xd*(fd - 1)) + b1*(15*G**2*(2*H*(Qm - 1)*(13*H - 3*Hd*fd + 3*Hd) - 15*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 7*k1**2*xd**2*(-6*H*(Qm - 1)*(16*H - Hd*fd + Hd) + 5*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp))) + f*k1**2*xd**2*(H**2*(70*Hd*Qm*xd*(fd - 1) - 35*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) - 804*Qm - 804*xQm + 1608) + 42*H*Hd*(fd - 1)*(Qm + xQm - 2) + 70*Hd*Hp*xd*(fd - 1)) + 7*k1**2*xb1*xd**2*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 96*xQm + 96) + 6*H*Hd*(fd - 1)*(xQm - 1) + 5*Hd*Hp*xd*(fd - 1))) + d**4*k1**2*(-2970*G**4*H**2*Hd*Qm*f*fd + 2970*G**4*H**2*Hd*Qm*f + 1485*G**4*H**2*Hd*f*fd*xbe - 1485*G**4*H**2*Hd*f*xbe + 105*G**4*H**2*Hd*fd*xb1*xbe - 105*G**4*H**2*Hd*xb1*xbe - 2970*G**4*Hd*Hp*f*fd + 2970*G**4*Hd*Hp*f - 105*G**4*Hd*Hp*fd*xb1 + 105*G**4*Hd*Hp*xb1 + 1740*G**2*H**2*Hd*Qm*f*fd*k1**2*xd**2 - 1740*G**2*H**2*Hd*Qm*f*k1**2*xd**2 - 870*G**2*H**2*Hd*f*fd*k1**2*xbe*xd**2 + 870*G**2*H**2*Hd*f*k1**2*xbe*xd**2 - 450*G**2*H**2*Hd*fd*k1**2*xb1*xbe*xd**2 + 450*G**2*H**2*Hd*k1**2*xb1*xbe*xd**2 - 10614*G**2*H**2*Qm*f*k1**2*xd + 21228*G**2*H**2*f*k1**2*xd + 3330*G**2*H**2*k1**2*xb1*xd + 870*G**2*H*Hd*Qm*f*fd*k1**2*xd - 870*G**2*H*Hd*Qm*f*k1**2*xd - 1740*G**2*H*Hd*f*fd*k1**2*xd + 1740*G**2*H*Hd*f*k1**2*xd - 450*G**2*H*Hd*fd*k1**2*xb1*xd + 450*G**2*H*Hd*k1**2*xb1*xd + 1740*G**2*Hd*Hp*f*fd*k1**2*xd**2 - 1740*G**2*Hd*Hp*f*k1**2*xd**2 + 450*G**2*Hd*Hp*fd*k1**2*xb1*xd**2 - 450*G**2*Hd*Hp*k1**2*xb1*xd**2 - 70*H**2*Hd*Qm*f*fd*k1**4*xd**4 + 70*H**2*Hd*Qm*f*k1**4*xd**4 + 5*H**2*Hd*be*f*(fd - 1)*(297*G**4 - 174*G**2*k1**2*xd**2 + 7*k1**4*xd**4) + 35*H**2*Hd*f*fd*k1**4*xbe*xd**4 - 35*H**2*Hd*f*k1**4*xbe*xd**4 + 35*H**2*Hd*fd*k1**4*xb1*xbe*xd**4 - 35*H**2*Hd*k1**4*xb1*xbe*xd**4 + 1590*H**2*Qm*f*k1**4*xd**3 - 3180*H**2*f*k1**4*xd**3 - 1330*H**2*k1**4*xb1*xd**3 - 70*H*Hd*Qm*f*fd*k1**4*xd**3 + 70*H*Hd*Qm*f*k1**4*xd**3 + 140*H*Hd*f*fd*k1**4*xd**3 - 140*H*Hd*f*k1**4*xd**3 + 70*H*Hd*fd*k1**4*xb1*xd**3 - 70*H*Hd*k1**4*xb1*xd**3 - 2*H*xQm*(15*G**4*H*Hd*(99*f + 7*xb1)*(fd - 1) - 30*G**2*H*Hd*k1**2*xd**2*(29*f + 15*xb1)*(fd - 1) + 3*G**2*k1**2*xd*(29*f*(61*H - 5*Hd*(fd - 1)) + 15*xb1*(37*H - 5*Hd*(fd - 1))) + 35*H*Hd*k1**4*xd**4*(f + xb1)*(fd - 1) - 5*k1**4*xd**3*(159*H*f - 7*Hd*f*(fd - 1) + 7*xb1*(19*H - Hd*fd + Hd))) - 70*Hd*Hp*f*fd*k1**4*xd**4 + 70*Hd*Hp*f*k1**4*xd**4 - 35*Hd*Hp*fd*k1**4*xb1*xd**4 + 35*Hd*Hp*k1**4*xb1*xd**4 + 5*b1*(21*G**4*Hd*(fd - 1)*(H**2*(-2*Qm + be) - Hp) - 18*G**2*H*k1**2*xd*(37*H - 5*Hd*(fd - 1))*(Qm - 1) + 90*G**2*Hd*k1**2*xd**2*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + 14*H*k1**4*xd**3*(Qm - 1)*(19*H - Hd*fd + Hd) + 7*Hd*k1**4*xd**4*(fd - 1)*(H**2*(-2*Qm + be) - Hp))) + d**3*k1**2*(b1*(105*G**4*(-2*H*(Qm - 1)*(4*H - Hd*fd + Hd) + 3*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 30*G**2*k1**2*xd**2*(2*H*(157*H - 15*Hd*(fd - 1))*(Qm - 1) - 15*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 7*k1**4*xd**4*(-10*H*(Qm - 1)*(22*H - Hd*fd + Hd) + 3*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp))) + f*(45*G**4*(H**2*(-99*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(99*Hd*xd*(fd - 1) - 202) - 404*xQm + 808) + 66*H*Hd*(fd - 1)*(Qm + xQm - 2) + 198*Hd*Hp*xd*(fd - 1)) + 174*G**2*k1**2*xd**2*(H**2*(5*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-5*Hd*xd*(fd - 1) + 87) + 174*xQm - 348) - 10*H*Hd*(fd - 1)*(Qm + xQm - 2) - 10*Hd*Hp*xd*(fd - 1)) + k1**4*xd**4*(H**2*(-21*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(21*Hd*xd*(fd - 1) - 920) - 1840*xQm + 3680) + 70*H*Hd*(fd - 1)*(Qm + xQm - 2) + 42*Hd*Hp*xd*(fd - 1))) + xb1*(105*G**4*(H**2*(-3*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 8*xQm + 8) + 2*H*Hd*(fd - 1)*(xQm - 1) + 3*Hd*Hp*xd*(fd - 1)) + 30*G**2*k1**2*xd**2*(H**2*(15*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 314*xQm - 314) - 30*H*Hd*(fd - 1)*(xQm - 1) - 15*Hd*Hp*xd*(fd - 1)) + 7*k1**4*xd**4*(H**2*(-3*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 220*xQm + 220) + 10*H*Hd*(fd - 1)*(xQm - 1) + 3*Hd*Hp*xd*(fd - 1)))) + d**2*(H**2*Hd*be*(fd - 1)*(-3150*G**6*f + 45*G**4*k1**2*xd**2*(7*b1 + 99*f) - 15*G**2*k1**4*xd**4*(15*b1 + 29*f) + 7*k1**6*xd**6*(b1 + f)) + f*(3150*G**6*Hd*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) - 405*G**4*k1**2*xd*(H**2*(22*Hd*Qm*xd*(fd - 1) - 11*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 338*Qm - 338*xQm + 676) + 22*H*Hd*(fd - 1)*(Qm + xQm - 2) + 22*Hd*Hp*xd*(fd - 1)) + 87*G**2*k1**4*xd**3*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(5*Hd*xd*(fd - 1) - 226) - 452*xQm + 904) + 20*H*Hd*(fd - 1)*(Qm + xQm - 2) + 10*Hd*Hp*xd*(fd - 1)) + k1**6*xd**5*(H**2*(7*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(-7*Hd*xd*(fd - 1) + 627) + 1254*xQm - 2508) - 42*H*Hd*(fd - 1)*(Qm + xQm - 2) - 14*Hd*Hp*xd*(fd - 1))) + k1**2*xd*(b1*(315*G**4*(2*H*(Qm - 1)*(9*H - Hd*fd + Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 15*G**2*k1**2*xd**2*(-4*H*(203*H - 15*Hd*(fd - 1))*(Qm - 1) + 15*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 7*k1**4*xd**4*(6*H*(Qm - 1)*(25*H - Hd*fd + Hd) - Hd*xd*(fd - 1)*(2*H**2*Qm + Hp))) + xb1*(315*G**4*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 18*xQm - 18) - 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) - 15*G**2*k1**2*xd**2*(H**2*(15*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 812*xQm - 812) - 60*H*Hd*(fd - 1)*(xQm - 1) - 15*Hd*Hp*xd*(fd - 1)) + 7*k1**4*xd**4*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 150*xQm - 150) - 6*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1))))) + d*(f*(3150*G**6*(H**2*(Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-Hd*fd*xd + Hd*xd + 6) + 12*xQm - 24) - 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*xd*(fd - 1)) + 135*G**4*k1**2*xd**2*(H**2*(-11*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(11*Hd*xd*(fd - 1) - 812) - 1624*xQm + 3248) + 66*H*Hd*(fd - 1)*(Qm + xQm - 2) + 22*Hd*Hp*xd*(fd - 1)) + 87*G**2*k1**4*xd**4*(H**2*(Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + Qm*(-2*Hd*xd*(fd - 1) + 278) + 278*xQm - 556) - 10*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*xd*(fd - 1)) + k1**6*xd**6*(H**2*(-Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(Hd*xd*(fd - 1) - 234) - 468*xQm + 936) + 14*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1))) + k1**2*xd**2*(b1*(105*G**4*(-6*H*(Qm - 1)*(14*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 45*G**2*k1**2*xd**2*(2*H*(Qm - 1)*(83*H - 5*Hd*fd + 5*Hd) + Hd*xd*(fd - 1)*(H**2*(-2*Qm + be) - Hp)) + k1**4*xd**4*(-14*H*(Qm - 1)*(28*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp))) + xb1*(105*G**4*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 84*xQm + 84) + 6*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + 45*G**2*k1**2*xd**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 166*xQm - 166) - 10*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + k1**4*xd**4*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 392*xQm + 392) + 14*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1))))))*np.sin(k1*(d - xd)/G) + k1*(d - xd)*(2*H**2*d**7*k1**6*(b1*(Qm - 1) + f*(Qm + xQm - 2) + xb1*(xQm - 1)) + 2*H*xd*(f*(Qm + xQm - 2)*(3150*G**6*(34*H - Hd*fd + Hd) - 15*G**4*k1**2*xd**2*(971*H - 29*Hd*(fd - 1)) + 12*G**2*k1**4*xd**4*(33*H - Hd*fd + Hd) - 2*H*k1**6*xd**6) - k1**2*xd**2*(b1*(Qm - 1) + xb1*(xQm - 1))*(105*G**4*(19*H - Hd*fd + Hd) - 10*G**2*k1**2*xd**2*(22*H - Hd*fd + Hd) + 2*H*k1**4*xd**4)) - 2*d**6*k1**4*(12*G**2*H**2*Hd*Qm*f*fd - 12*G**2*H**2*Hd*Qm*f - 6*G**2*H**2*Hd*be*f*(fd - 1) - 6*G**2*H**2*Hd*f*fd*xbe + 6*G**2*H**2*Hd*f*xbe - 5*G**2*H**2*Hd*fd*xb1*xbe + 5*G**2*H**2*Hd*xb1*xbe + 12*G**2*Hd*Hp*f*fd - 12*G**2*Hd*Hp*f + 5*G**2*Hd*Hp*fd*xb1 - 5*G**2*Hd*Hp*xb1 + 5*G**2*Hd*b1*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + 8*H**2*Qm*f*k1**2*xd + 8*H**2*b1*k1**2*xd*(Qm - 1) - 16*H**2*f*k1**2*xd - 8*H**2*k1**2*xb1*xd + 2*H**2*xQm*(G**2*Hd*(6*f + 5*xb1)*(fd - 1) + 4*k1**2*xd*(f + xb1))) + 2*d**5*k1**4*(3*G**2*f*(H**2*(-10*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + Qm*(20*Hd*xd*(fd - 1) - 33) - 33*xQm + 66) + 4*H*Hd*(fd - 1)*(Qm + xQm - 2) + 20*Hd*Hp*xd*(fd - 1)) + 5*G**2*xb1*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) - 11*xQm + 11) + 2*H*Hd*(fd - 1)*(xQm - 1) + 5*Hd*Hp*xd*(fd - 1)) + 27*H**2*f*k1**2*xd**2*(Qm + xQm - 2) + 27*H**2*k1**2*xb1*xd**2*(xQm - 1) + b1*(5*G**2*(-H*(Qm - 1)*(11*H - 2*Hd*fd + 2*Hd) + 5*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 27*H**2*k1**2*xd**2*(Qm - 1))) + d**4*k1**2*(870*G**4*H**2*Hd*Qm*f*fd - 870*G**4*H**2*Hd*Qm*f - 435*G**4*H**2*Hd*f*fd*xbe + 435*G**4*H**2*Hd*f*xbe - 105*G**4*H**2*Hd*fd*xb1*xbe + 105*G**4*H**2*Hd*xb1*xbe + 870*G**4*Hd*Hp*f*fd - 870*G**4*Hd*Hp*f + 105*G**4*Hd*Hp*fd*xb1 - 105*G**4*Hd*Hp*xb1 - 240*G**2*H**2*Hd*Qm*f*fd*k1**2*xd**2 + 240*G**2*H**2*Hd*Qm*f*k1**2*xd**2 - 15*G**2*H**2*Hd*be*f*(29*G**2 - 8*k1**2*xd**2)*(fd - 1) + 120*G**2*H**2*Hd*f*fd*k1**2*xbe*xd**2 - 120*G**2*H**2*Hd*f*k1**2*xbe*xd**2 + 100*G**2*H**2*Hd*fd*k1**2*xb1*xbe*xd**2 - 100*G**2*H**2*Hd*k1**2*xb1*xbe*xd**2 + 1584*G**2*H**2*Qm*f*k1**2*xd - 3168*G**2*H**2*f*k1**2*xd - 880*G**2*H**2*k1**2*xb1*xd - 120*G**2*H*Hd*Qm*f*fd*k1**2*xd + 120*G**2*H*Hd*Qm*f*k1**2*xd + 240*G**2*H*Hd*f*fd*k1**2*xd - 240*G**2*H*Hd*f*k1**2*xd + 100*G**2*H*Hd*fd*k1**2*xb1*xd - 100*G**2*H*Hd*k1**2*xb1*xd - 240*G**2*Hd*Hp*f*fd*k1**2*xd**2 + 240*G**2*Hd*Hp*f*k1**2*xd**2 - 100*G**2*Hd*Hp*fd*k1**2*xb1*xd**2 + 100*G**2*Hd*Hp*k1**2*xb1*xd**2 - 100*H**2*Qm*f*k1**4*xd**3 + 200*H**2*f*k1**4*xd**3 + 100*H**2*k1**4*xb1*xd**3 + 2*H*xQm*(15*G**4*H*Hd*(29*f + 7*xb1)*(fd - 1) - 20*G**2*H*Hd*k1**2*xd**2*(6*f + 5*xb1)*(fd - 1) + 2*G**2*k1**2*xd*(6*f*(66*H - 5*Hd*fd + 5*Hd) + 5*xb1*(44*H - 5*Hd*fd + 5*Hd)) - 50*H*k1**4*xd**3*(f + xb1)) + 5*b1*(21*G**4*Hd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp) + 4*G**2*H*k1**2*xd*(44*H - 5*Hd*(fd - 1))*(Qm - 1) + 20*G**2*Hd*k1**2*xd**2*(fd - 1)*(H**2*(-2*Qm + be) - Hp) - 20*H**2*k1**4*xd**3*(Qm - 1))) + d**3*k1**2*(5*b1*(21*G**4*(2*H*(Qm - 1)*(4*H - Hd*fd + Hd) - 3*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 4*G**2*k1**2*xd**2*(-H*(121*H - 10*Hd*(fd - 1))*(Qm - 1) + 5*Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 22*H**2*k1**4*xd**4*(Qm - 1)) + f*(15*G**4*(3*H**2*(29*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(-29*Hd*xd*(fd - 1) + 62) + 124*xQm - 248) - 58*H*Hd*(fd - 1)*(Qm + xQm - 2) - 174*Hd*Hp*xd*(fd - 1)) + 12*G**2*k1**2*xd**2*(H**2*(-10*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + Qm*(20*Hd*xd*(fd - 1) - 363) - 363*xQm + 726) + 20*H*Hd*(fd - 1)*(Qm + xQm - 2) + 20*Hd*Hp*xd*(fd - 1)) + 110*H**2*k1**4*xd**4*(Qm + xQm - 2)) + 5*xb1*(21*G**4*(H**2*(3*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 8*xQm - 8) - 2*H*Hd*(fd - 1)*(xQm - 1) - 3*Hd*Hp*xd*(fd - 1)) - 4*G**2*k1**2*xd**2*(H**2*(5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 121*xQm - 121) - 10*H*Hd*(fd - 1)*(xQm - 1) - 5*Hd*Hp*xd*(fd - 1)) + 22*H**2*k1**4*xd**4*(xQm - 1))) + d**2*(5*G**2*H**2*Hd*be*(fd - 1)*(630*G**4*f - 9*G**2*k1**2*xd**2*(7*b1 + 29*f) + 2*k1**4*xd**4*(5*b1 + 6*f)) - 3*f*(1050*G**6*Hd*(fd - 1)*(H**2*(2*Qm + 2*xQm - xbe) + 2*Hp) - 5*G**4*k1**2*xd*(H**2*(-87*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(87*Hd*xd*(fd - 1) - 1343) - 2686*xQm + 5372) + 174*H*Hd*(fd - 1)*(Qm + xQm - 2) + 174*Hd*Hp*xd*(fd - 1)) + 4*G**2*k1**4*xd**3*(H**2*(-5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 2*Qm*(5*Hd*xd*(fd - 1) - 231) - 462*xQm + 924) + 20*H*Hd*(fd - 1)*(Qm + xQm - 2) + 10*Hd*Hp*xd*(fd - 1)) + 24*H**2*k1**6*xd**5*(Qm + xQm - 2)) + k1**2*xd*(b1*(315*G**4*(-2*H*(Qm - 1)*(9*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) + 10*G**2*k1**2*xd**2*(4*H*(Qm - 1)*(77*H - 5*Hd*fd + 5*Hd) - 5*Hd*xd*(fd - 1)*(2*H**2*Qm + Hp)) - 72*H**2*k1**4*xd**4*(Qm - 1)) + xb1*(315*G**4*(H**2*(-Hd*xd*(fd - 1)*(-2*xQm + xbe) - 18*xQm + 18) + 2*H*Hd*(fd - 1)*(xQm - 1) + Hd*Hp*xd*(fd - 1)) + 10*G**2*k1**2*xd**2*(H**2*(5*Hd*xd*(fd - 1)*(-2*xQm + xbe) + 308*xQm - 308) - 20*H*Hd*(fd - 1)*(xQm - 1) - 5*Hd*Hp*xd*(fd - 1)) - 72*H**2*k1**4*xd**4*(xQm - 1)))) + d*(f*(3150*G**6*(H**2*(-Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 2*Qm*(Hd*xd*(fd - 1) - 6) - 12*xQm + 24) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*xd*(fd - 1)) + 15*G**4*k1**2*xd**2*(H**2*(-58*Hd*Qm*xd*(fd - 1) + 29*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + 4256*Qm + 4256*xQm - 8512) - 174*H*Hd*(fd - 1)*(Qm + xQm - 2) - 58*Hd*Hp*xd*(fd - 1)) + 6*G**2*k1**4*xd**4*(H**2*(-2*Hd*xd*(fd - 1)*(be - 2*xQm + xbe) + Qm*(4*Hd*xd*(fd - 1) - 561) - 561*xQm + 1122) + 20*H*Hd*(fd - 1)*(Qm + xQm - 2) + 4*Hd*Hp*xd*(fd - 1)) + 26*H**2*k1**6*xd**6*(Qm + xQm - 2)) + k1**2*xd**2*(b1*(105*G**4*(6*H*(Qm - 1)*(14*H - Hd*fd + Hd) + Hd*xd*(fd - 1)*(H**2*(-2*Qm + be) - Hp)) + 10*G**2*k1**2*xd**2*(-H*(187*H - 10*Hd*(fd - 1))*(Qm - 1) + Hd*xd*(fd - 1)*(-H**2*(-2*Qm + be) + Hp)) + 26*H**2*k1**4*xd**4*(Qm - 1)) + xb1*(105*G**4*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 84*xQm - 84) - 6*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) - 10*G**2*k1**2*xd**2*(H**2*(Hd*xd*(fd - 1)*(-2*xQm + xbe) + 187*xQm - 187) - 10*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*xd*(fd - 1)) + 26*H**2*k1**4*xd**4*(xQm - 1)))))*np.cos(k1*(d - xd)/G))/(G*H**2*d*k1**9*(d - xd)**8)
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

            expr = D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(45*G**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 60*G**2*H**4*k1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) + 10*G**2*H**2*k1**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(-H**2*d*(-2*xQm + xbe) - 2*H*(xQm - 1) + Hp*d)) + 10*G**2*H**2*k1**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + 24*H**4*k1**4*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*pk/(5*G**3*H**4*d**2*k1**4)
            
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

            expr = 18*1j*D1d1**2*Hd1**2*Hd2**2*OMd1*OMd2*(5*G**2*(Qm - 1)*(d - xd1)*(H**2*(-Hd2*d*(fd2 - 1)*(-2*xQm + xbe) - 2*xQm + 2) - 2*H*Hd2*(fd2 - 1)*(xQm - 1) + Hd2*Hp*d*(fd2 - 1))/(H**2*k1**2) + 5*G**2*(d - xd2)*(xQm - 1)*(H**2*(Hd1*be*d*(fd1 - 1) + 2*Qm*d*(-Hd1*fd1 + Hd1) + 2*Qm - 2) + 2*H*Hd1*(Qm - 1)*(fd1 - 1) - Hd1*Hp*d*(fd1 - 1))/(H**2*k1**2) + 2*xd1*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 2*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*pk/(5*G**2*d**2*k1)
            
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

            expr = D1d1*D1d2*(-15*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(-27*G**2*(xd1**2 + 3*xd1*xd2 + xd2**2) + 2*k1**2*(xd1 - xd2)**2*(xd1**2 + 4*xd1*xd2 + xd2**2)) - 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*xd2*(xd1 - xd2)**2)*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 3*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(3*G**2*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) + 2*H**2*(d - xd2)*(xQm - 1)*(-9*G**2*(xd1 + xd2) + k1**2*xd1*(xd1 - xd2)**2)))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**6*(xd1 - xd2)**6) + 15*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(9*G**2*(xd1 + xd2) - k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G**2*(3*G**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 24*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(27*G**4*(xd1**2 + 3*xd1*xd2 + xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(11*xd1**2 + 35*xd1*xd2 + 11*xd2**2) + k1**4*xd1*xd2*(xd1 - xd2)**4))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*pk/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, fd1, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 2*D1d1**2*Hd1**4*OMd1**2*(84*G**2*H**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1) - 7*G**2*xd1*(3 - 3*Qm)*(d - xd1)*(-2*H**2*(xQm - 1) + Hd1*(fd1 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 7*G**2*xd2*(3 - 3*xQm)*(d - xd2)*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 24*H**2*k1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1))*pk/(7*G**3*H**2*d**2*k1**2)
            
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

            expr = D1d1*D1d2*(-21*1j*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - 3*G**2*k1**2*(xd1 - xd2)**2*(87*xd1**2 + 286*xd1*xd2 + 87*xd2**2) + 7*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(30*G**4*(3*xd1 + 2*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(4*xd1 + 3*xd2) + k1**4*xd2*(xd1 - xd2)**4) + (xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(9*G**2*(5*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) + 2*H**2*(3 - 3*xQm)*(d - xd2)*(30*G**4*(2*xd1 + 3*xd2) - 9*G**2*k1**2*(xd1 - xd2)**2*(3*xd1 + 4*xd2) + k1**4*xd1*(xd1 - xd2)**4)))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**8) - 21*1j*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(-2*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(30*G**2*(3*xd1 + 2*xd2) - k1**2*(xd1 - xd2)**2*(6*xd1 + 7*xd2))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 2*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(30*G**2*(2*xd1 + 3*xd2) - k1**2*(xd1 - xd2)**2*(7*xd1 + 6*xd2))*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - 3*G**2*(15*G**2 - k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - 12*H**4*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(150*G**4*(4*xd1**2 + 13*xd1*xd2 + 4*xd2**2) - G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 208*xd1*xd2 + 61*xd2**2) + 2*k1**4*xd1*xd2*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**7*(-xd1 + xd2)**7))*pk/G**3

            return expr

        # for when xd1 == xd2 
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, _, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = -36*1j*D1d1**2*Hd1**4*OMd1**2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*(xd1 - xd2)*pk/(5*G**2*d**2*k1)
            
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

            expr = D1d1*D1d2*(-9*G**4*Hd1**2*Hd2**2*OMd1*OMd2*(45*G**2*(21*G**2 - 2*k1**2*(xd1 - xd2)**2)*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) + 4*H**4*(3 - 3*Qm)*(3 - 3*xQm)*(d - xd1)*(d - xd2)*(1575*G**4*(5*xd1**2 + 18*xd1*xd2 + 5*xd2**2) - 15*G**2*k1**2*(xd1 - xd2)**2*(61*xd1**2 + 218*xd1*xd2 + 61*xd2**2) + 11*k1**4*(xd1 - xd2)**4*(xd1**2 + 4*xd1*xd2 + xd2**2)) + 6*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(525*G**4*(2*xd1 + xd2) - 5*G**2*k1**2*(xd1 - xd2)**2*(20*xd1 + 13*xd2) + k1**4*xd2*(xd1 - xd2)**4) + 6*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(525*G**4*(xd1 + 2*xd2) - 5*G**2*k1**2*(xd1 - xd2)**2*(13*xd1 + 20*xd2) + k1**4*xd1*(xd1 - xd2)**4))*np.cos(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**8*(xd1 - xd2)**8) + 9*G**3*Hd1**2*Hd2**2*OMd1*OMd2*(6*G**2*H**2*(3 - 3*Qm)*(d - xd1)*(xd1 - xd2)**2*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(525*G**4*(2*xd1 + xd2) - 30*G**2*k1**2*(xd1 - xd2)**2*(15*xd1 + 8*xd2) + k1**4*(xd1 - xd2)**4*(10*xd1 + 11*xd2)) + 6*G**2*H**2*(3 - 3*xQm)*(d - xd2)*(xd1 - xd2)**2*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(525*G**4*(xd1 + 2*xd2) - 30*G**2*k1**2*(xd1 - xd2)**2*(8*xd1 + 15*xd2) + k1**4*(xd1 - xd2)**4*(11*xd1 + 10*xd2)) + 9*G**2*(xd1 - xd2)**4*(-2*H**2*(Qm - 1) - Hd1*(fd1 - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(-2*H**2*(xQm - 1) + Hd2*(fd2 - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(105*G**4 - 45*G**2*k1**2*(xd1 - xd2)**2 + k1**4*(xd1 - xd2)**4) + 4*H**4*(3 - 3*Qm)*(3 - 3*xQm)*(d - xd1)*(d - xd2)*(1575*G**6*(5*xd1**2 + 18*xd1*xd2 + 5*xd2**2) - 60*G**4*k1**2*(xd1 - xd2)**2*(59*xd1**2 + 212*xd1*xd2 + 59*xd2**2) + 3*G**2*k1**4*(xd1 - xd2)**4*(47*xd1**2 + 168*xd1*xd2 + 47*xd2**2) - 2*k1**6*xd1*xd2*(xd1 - xd2)**6))*np.sin(k1*(-xd1 + xd2)/G)/(H**4*d**2*k1**9*(-xd1 + xd2)**9))*pk/G**3

            return expr

        # for when xd1 == xd2
        def int_terms2(xd1, cosmo_funcs, k1, zz, t=0, sigma=None):
            zzd1, _, D1d1, Hd1, OMd1 = BaseInt.get_integrand_params(cosmo_funcs, xd1)
            G = xd1/d 
            pk = baseint.pk(k1/G,zzd1)

            expr = 72*D1d1**2*Hd1**4*OMd1**2*xd1*xd2*(Qm - 1)*(d - xd1)*(d - xd2)*(xQm - 1)*pk/(35*G**3*d**2)
            
            return expr

        return BaseInt.int_2Dgrid(xd1,xd2,int_terms2, int_terms1,cosmo_funcs, k1, zz, fast=fast,**kwargs) # parse functions as well
