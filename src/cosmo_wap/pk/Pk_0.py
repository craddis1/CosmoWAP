from cosmo_wap.lib.angular_integrate import legendre
from cosmo_wap.lib.utils import add_empty_methods_pk


#1st order terms
@add_empty_methods_pk('l1','l3')
class NPP:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        #unpack all necessary terms
        Pk,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz)
        return D1**2* Pk*(b1 + f*mu**2)*(f*mu**2 + xb1)

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(NPP.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

#1st order terms
@add_empty_methods_pk('l0','l2','l4')
class GR1:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        Pk,f,D1,b1,xb1,gr1,_,xgr1,_ = cosmo_funcs.unpack_pk(k1,zz,GR=True) #unpack all necessary terms
        return 1j*D1**2*Pk*mu*(-b1*xgr1 + f*mu**2*(gr1 - xgr1) + gr1*xb1)/k1

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(GR1.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

#2nd order terms
@add_empty_methods_pk('l1','l3')
class GR2:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        Pk,f,D1,b1,xb1,gr1,gr2,xgr1,xgr2 = cosmo_funcs.unpack_pk(k1,zz,GR=True) #unpack all necessary terms
        return D1**2*Pk*(b1*xgr2 + f*mu**2*(gr2 + xgr2) + gr1*mu**2*xgr1 + gr2*xb1)/k1**2

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(GR2.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

# extra local GR terms (H/k)^3, (H/k)^4 (field only goes up to (H/k)^2)
class GRX:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        Pk,_,D1,_,_,gr1,gr2,xgr1,xgr2 = cosmo_funcs.unpack_pk(k1,zz,GR=True) #unpack all necessary terms
        return D1**2*Pk*(1j*gr1*k1*mu*xgr2 + gr2*(-1j*k1*mu*xgr1 + xgr2))/k1**4

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(GRX.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

