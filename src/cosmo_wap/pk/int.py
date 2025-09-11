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
    def mu(mu,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        """2D P(k,mu) power spectra"""
        return BaseInt.single_int(IntNPP.mu_integrand, mu, cosmo_funcs, k1, zz, t, sigma, n=n, remove_div=remove_div)
    
    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128,n_mu=16,fast=False):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return integrate.legendre(IntNPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n ,n_mu=n_mu,fast=fast)

    ############################ Seperate Multipoles - with analytic mu integration #################################

    @staticmethod
    def l0(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=False):
        return BaseInt.single_int(IntNPP.l0_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
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

        expr = D1*D1d*Hd**2*OMd*pk*(d - xd)*(G*(6*H**2*d*k1**2*xb1*(d - xd)**2*(xQm - 1) - 6*H**2*f*(xQm - 1)*(6*G**2*d + 6*G**2*xd - 3*d**3*k1**2 + 4*d**2*k1**2*xd + d*k1**2*xd**2 - 2*k1**2*xd**3) + 2*H**2*(3 - 3*Qm)*(12*G**2*f*xd + 6*G**2*f*(d - xd) - k1**2*xd*(b1 + 5*f)*(d - xd)**2 - k1**2*(b1 + 3*f)*(d - xd)**3) + 3*(d - xd)*(-f*(2*G**2 - k1**2*(d - xd)**2)*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1)) + k1**2*(d - xd)**2*(b1*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)))))*np.sin(k1*(-d + xd)/G) + 2*k1*(d - xd)*(3*G**2*f*(d - xd)*(H**2*(-Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) - 2*Qm - 2*xQm + 4) - 2*H*Hd*(fd - 1)*(Qm + xQm - 2) + 2*Hd*Hp*d*(fd - 1)) + 3*H**2*d*k1**2*xb1*(d - xd)**2*(xQm - 1) + 3*H**2*f*(xQm - 1)*(-6*G**2*d - 6*G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2) + H**2*(3 - 3*Qm)*(6*G**2*d*f + 6*G**2*f*xd - d**3*k1**2*(b1 + f) + 2*d**2*k1**2*xd*(b1 + f) - d*k1**2*xd**2*(b1 + f)))*np.cos(k1*(-d + xd)/G))/(G*H**2*d*k1**5*(-d + xd)**5)

        return expr
    
    @staticmethod
    def l0_same_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)

        expr = D1*D1d*Hd**2*OMd*pk*(5*G**2*f*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1)) + 15*G**2*xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)) + 2*H**2*f*k1**2*xd*(d - xd)*(Qm + xQm - 2) + 10*H**2*k1**2*xb1*xd*(d - xd)*(xQm - 1) + 5*b1*(3*G**2*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + 2*H**2*k1**2*xd*(Qm - 1)*(d - xd)))/(5*G**3*H**2*d*k1**2)

        return expr
    
    @staticmethod
    def l1(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(IntNPP.l1_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
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
            
        expr = -3*1j*D1*D1d*Hd**2*OMd*pk*(-G*k1*(d - xd)*(2*H**2*(3 - 3*Qm)*(-36*G**2*f*xd + 2*d**3*k1**2*(b1 + 2*f) - d**2*k1**2*xd*(3*b1 + 5*f) - 2*d*f*(12*G**2 + k1**2*xd**2) + k1**2*xd**3*(b1 + 3*f)) - 2*H**2*(3 - 3*xQm)*(-36*G**2*f*xd + 2*d**3*k1**2*(2*f + xb1) - d**2*k1**2*xd*(5*f + 3*xb1) - 2*d*f*(12*G**2 + k1**2*xd**2) + k1**2*xd**3*(3*f + xb1)) - 3*(d - xd)*(6*G**2*f*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - f*(6*G**2 - k1**2*(d - xd)**2)*(-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) + k1**2*xb1*(d - xd)**2*(-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d)) - k1**2*(b1 + f)*(d - xd)**2*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))))*np.cos(k1*(-d + xd)/G) + (3*G**2*(d - xd)*(6*G**2*f*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - k1**2*(b1 + 3*f)*(d - xd)**2*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) - (-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(f*(6*G**2 - 3*k1**2*(d - xd)**2) - k1**2*xb1*(d - xd)**2)) + 2*H**2*(3 - 3*Qm)*(60*G**4*f*xd + 24*G**4*f*(d - xd) - 3*G**2*k1**2*xd*(b1 + 9*f)*(d - xd)**2 - 2*G**2*k1**2*(b1 + 6*f)*(d - xd)**3 + k1**4*xd*(b1 + f)*(d - xd)**4 + k1**4*(b1 + f)*(d - xd)**5) + 2*H**2*(3 - 3*xQm)*(-f*(60*G**4*xd - 27*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(24*G**4 - 12*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - k1**2*xb1*(d - xd)**2*(-2*G**2*d - G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2)))*np.sin(k1*(-d + xd)/G))/(G*H**2*d*k1**6*(d - xd)**5)
        return expr
    
    @staticmethod
    def l1_same_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)
            
        expr = -6*1j*D1*D1d*Hd**2*OMd*pk*(d - xd)*(5*b1*(Qm - 1) + 3*f*(Qm - xQm) - 5*xb1*(xQm - 1))/(5*G**2*d*k1)
        return expr
    
    def l2(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(IntNPP.l2_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
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
            
        expr = 5*D1*D1d*Hd**2*OMd*pk*(d - xd)*(G*(2*H**2*(3 - 3*Qm)*(b1*k1**2*(d - xd)**2*(9*G**2*d + 9*G**2*xd - 4*d**3*k1**2 + 5*d**2*k1**2*xd + 2*d*k1**2*xd**2 - 3*k1**2*xd**3) - f*(540*G**4*xd - 246*G**2*k1**2*xd*(d - xd)**2 + 11*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(60*G**4 - 29*G**2*k1**2*(d - xd)**2 + 2*k1**4*(d - xd)**4))) + 2*H**2*(3 - 3*xQm)*(-f*(540*G**4*xd - 246*G**2*k1**2*xd*(d - xd)**2 + 11*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(60*G**4 - 29*G**2*k1**2*(d - xd)**2 + 2*k1**4*(d - xd)**4)) + k1**2*xb1*(d - xd)**2*(9*G**2*d + 9*G**2*xd - 4*d**3*k1**2 + 5*d**2*k1**2*xd + 2*d*k1**2*xd**2 - 3*k1**2*xd**3)) - 3*(d - xd)*(-f*(36*G**4 - 17*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1)) - k1**2*(-3*G**2 + k1**2*(d - xd)**2)*(d - xd)**2*(b1*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)))))*np.sin(k1*(-d + xd)/G) + k1*(d - xd)*(-3*G**2*(d - xd)*(-f*(36*G**2 - 5*k1**2*(d - xd)**2)*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1)) + 3*k1**2*(d - xd)**2*(b1*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)))) + 2*H**2*(3 - 3*Qm)*(-b1*k1**2*(d - xd)**2*(-9*G**2*d - 9*G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2) - f*(540*G**4*xd - 66*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(180*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4))) + 2*H**2*(3 - 3*xQm)*(-f*(540*G**4*xd - 66*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(180*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - k1**2*xb1*(d - xd)**2*(-9*G**2*d - 9*G**2*xd + d**3*k1**2 - 2*d**2*k1**2*xd + d*k1**2*xd**2)))*np.cos(k1*(-d + xd)/G))/(G*H**2*d*k1**7*(-d + xd)**7)
        return expr
    
    @staticmethod
    def l2_same_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)
            
        expr = 2*D1*D1d*Hd**2*OMd*pk*(7*b1*xd + (7*G**2*f*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1))/(H**2*k1**2) - 7*b1*xd*(Qm*d - Qm*xd + xd) + f*xd*(d - xd)*(Qm + xQm - 2) - 7*xb1*xd*(d - xd)*(xQm - 1))/d)/(7*G**3)
        return expr
    
    def l3(cosmo_funcs, k1, zz=0, t=0, sigma=None, n=128, remove_div=True):
        return BaseInt.single_int(IntNPP.l3_integrand, cosmo_funcs, k1, zz, t=t, sigma=sigma, n=n, remove_div=remove_div)
        
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
        
        expr = -7*1j*D1*D1d*Hd**2*OMd*pk*(G*k1*(d - xd)*(2*H**2*(3 - 3*Qm)*(b1*k1**2*(d - xd)**2*(60*G**2*d + 90*G**2*xd - 7*d**3*k1**2 + 8*d**2*k1**2*xd + 5*d*k1**2*xd**2 - 6*k1**2*xd**3) - f*(6300*G**4*xd - 810*G**2*k1**2*xd*(d - xd)**2 + 17*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(600*G**4 - 88*G**2*k1**2*(d - xd)**2 + 3*k1**4*(d - xd)**4))) - 2*H**2*(3 - 3*xQm)*(-f*(6300*G**4*xd - 810*G**2*k1**2*xd*(d - xd)**2 + 17*k1**4*xd*(d - xd)**4 + 3*(d - xd)*(600*G**4 - 88*G**2*k1**2*(d - xd)**2 + 3*k1**4*(d - xd)**4)) + k1**2*xb1*(d - xd)**2*(60*G**2*d + 90*G**2*xd - 7*d**3*k1**2 + 8*d**2*k1**2*xd + 5*d*k1**2*xd**2 - 6*k1**2*xd**3)) - 3*(d - xd)*((-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(b1*k1**2*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2 + f*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - (-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(f*(300*G**4 - 41*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4) + k1**2*xb1*(-15*G**2 + k1**2*(d - xd)**2)*(d - xd)**2)))*np.cos(k1*(-d + xd)/G) + (-3*G**2*(d - xd)*(-3*b1*k1**2*(5*G**2 - 2*k1**2*(d - xd)**2)*(d - xd)**2*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d)) + f*(-2*H**2*(Qm - 1) - Hd*(fd - 1)*(H**2*d*(-2*Qm + be) + 2*H*(Qm - 1) - Hp*d))*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4) - f*(-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))*(300*G**4 - 141*G**2*k1**2*(d - xd)**2 + 8*k1**4*(d - xd)**4) + 3*k1**2*xb1*(5*G**2 - 2*k1**2*(d - xd)**2)*(d - xd)**2*(-2*H**2*(xQm - 1) + Hd*(fd - 1)*(H*(-H*d*(-2*xQm + xbe) - 2*xQm + 2) + Hp*d))) + 2*H**2*(3 - 3*Qm)*(b1*k1**2*(d - xd)**2*(150*G**4*xd - 63*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(60*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - f*(6300*G**6*xd - 2910*G**4*k1**2*xd*(d - xd)**2 + 147*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(1800*G**6 - 864*G**4*k1**2*(d - xd)**2 + 57*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6))) - 2*H**2*(3 - 3*xQm)*(-f*(6300*G**6*xd - 2910*G**4*k1**2*xd*(d - xd)**2 + 147*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(1800*G**6 - 864*G**4*k1**2*(d - xd)**2 + 57*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) + k1**2*xb1*(d - xd)**2*(150*G**4*xd - 63*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(60*G**4 - 27*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4))))*np.sin(k1*(-d + xd)/G))/(G*H**2*d*k1**8*(d - xd)**7)
        return expr
    
    @staticmethod
    def l3_same_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)

        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta
        pk = baseint.pk(k1/G,zzd1)
        
        expr = -12*1j*D1*D1d*Hd**2*OMd*pk*f*(Qm - xQm)*(d - xd)/(5*G**2*d*k1)
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
                    
        expr = 9*D1*D1d*Hd**2*OMd*pk*(d - xd)*(-G*(2*H**2*(3 - 3*Qm)*(b1*k1**2*(d - xd)**2*(3*xd*(525*G**4 - 230*G**2*k1**2*(d - xd)**2 + 7*k1**4*(d - xd)**4) + (d - xd)*(525*G**4 - 240*G**2*k1**2*(d - xd)**2 + 11*k1**4*(d - xd)**4)) - f*(88200*G**6*xd - 41175*G**4*k1**2*xd*(d - xd)**2 + 2262*G**2*k1**4*xd*(d - xd)**4 - 25*k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 10575*G**4*k1**2*(d - xd)**2 + 696*G**2*k1**4*(d - xd)**4 - 13*k1**6*(d - xd)**6))) + 2*H**2*(3 - 3*xQm)*(-f*(88200*G**6*xd - 41175*G**4*k1**2*xd*(d - xd)**2 + 2262*G**2*k1**4*xd*(d - xd)**4 - 25*k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 10575*G**4*k1**2*(d - xd)**2 + 696*G**2*k1**4*(d - xd)**4 - 13*k1**6*(d - xd)**6)) + k1**2*xb1*(d - xd)**2*(3*xd*(525*G**4 - 230*G**2*k1**2*(d - xd)**2 + 7*k1**4*(d - xd)**4) + (d - xd)*(525*G**4 - 240*G**2*k1**2*(d - xd)**2 + 11*k1**4*(d - xd)**4))) - 3*(d - xd)*(-f*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1))*(3150*G**6 - 1485*G**4*k1**2*(d - xd)**2 + 87*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6) + k1**2*(d - xd)**2*(b1*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)))*(105*G**4 - 45*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)))*np.sin(k1*(-d + xd)/G) - k1*(d - xd)*(-3*G**2*(d - xd)*(-3*f*(10*G**2 - k1**2*(d - xd)**2)*(105*G**2 - 4*k1**2*(d - xd)**2)*(H**2*(Hd*d*(fd - 1)*(-2*Qm + be - 2*xQm + xbe) + 2*Qm + 2*xQm - 4) + 2*H*Hd*(fd - 1)*(Qm + xQm - 2) - 2*Hd*Hp*d*(fd - 1)) - 5*k1**2*(-21*G**2 + 2*k1**2*(d - xd)**2)*(d - xd)**2*(b1*(H**2*(Hd*d*(-2*Qm + be)*(fd - 1) + 2*Qm - 2) + 2*H*Hd*(Qm - 1)*(fd - 1) - Hd*Hp*d*(fd - 1)) + xb1*(H**2*(Hd*d*(fd - 1)*(-2*xQm + xbe) + 2*xQm - 2) + 2*H*Hd*(fd - 1)*(xQm - 1) - Hd*Hp*d*(fd - 1)))) + 2*H**2*(3 - 3*Qm)*(b1*k1**2*(d - xd)**2*(1575*G**4*xd - 165*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(525*G**4 - 65*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4)) - f*(88200*G**6*xd - 11775*G**4*k1**2*xd*(d - xd)**2 + 297*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 3225*G**4*k1**2*(d - xd)**2 + 111*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6))) + 2*H**2*(3 - 3*xQm)*(-f*(88200*G**6*xd - 11775*G**4*k1**2*xd*(d - xd)**2 + 297*G**2*k1**4*xd*(d - xd)**4 - k1**6*xd*(d - xd)**6 + (d - xd)*(22050*G**6 - 3225*G**4*k1**2*(d - xd)**2 + 111*G**2*k1**4*(d - xd)**4 - k1**6*(d - xd)**6)) + k1**2*xb1*(d - xd)**2*(1575*G**4*xd - 165*G**2*k1**2*xd*(d - xd)**2 + k1**4*xd*(d - xd)**4 + (d - xd)*(525*G**4 - 65*G**2*k1**2*(d - xd)**2 + k1**4*(d - xd)**4))))*np.cos(k1*(-d + xd)/G))/(G*H**2*d*k1**9*(-d + xd)**9)
        return expr
    
    @staticmethod
    def l4_same_integrand(xd,cosmo_funcs, k1, zz=0, t=0, sigma=None):

        baseint = BaseInt(cosmo_funcs)
        
        # allow broadcasting of k1 and zz with xd
        k1,zz = utils.enable_broadcasting(k1,zz,n=1)
        
        _,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz) # generic power spectrum params
        d, H, Hp, Qm, xQm, be, xbe = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params - could be merged into .unpack_pk
        zzd1, fd, D1d, Hd, OMd = BaseInt.get_integrand_params(cosmo_funcs, xd) # integrand params - arrays in shape (xd)

        G = (d + xd) / (2 * d) # Define G from dirac-delta - could just define q=k1/G
        pk = baseint.pk(k1/G,zzd1)
                    
        expr = -24*D1*D1d*Hd**2*OMd*pk*f*xd*(d - xd)*(Qm + xQm - 2)/(35*G**3*d)
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
