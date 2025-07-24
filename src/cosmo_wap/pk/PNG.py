import numpy as np
from scipy.special import erf  # Error function needed from integral over FoG
from cosmo_wap.lib.utils import add_empty_methods_pk

class BaseFNL:
    """fNL terms for pk are the fungible except for the k^{alpha} - so we can make this short and supremely sweet"""
    @staticmethod
    def _l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        return D1**2*Pk*(5*Mk1*b1*(Mk1*f + 3*Mk1*xb1 + 3*fNL*xb01) + Mk1*f*(3*Mk1*f + 5*Mk1*xb1 + 5*fNL*xb01) + 5*b01*fNL*(Mk1*f + 3*Mk1*xb1 + 3*fNL*xb01))/(15*Mk1**2)
    
    @staticmethod
    def _l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        expr =D1**2*Pk*fNL*(Mk1*xb01*(3*b1 + f) + b01*(Mk1*f + 3*Mk1*xb1 + 3*fNL*xb01))/(3*Mk1**2)
        
        if sigma != None:
            expr = D1**2*Pk*fNL*(-2*Mk1*f*k1*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(Mk1*b01*f + Mk1*xb01*(b1*k1**2*sigma**2 + f) + b01*k1**2*sigma**2*(Mk1*xb1 + fNL*xb01))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*Mk1**2*k1**3*sigma**3)
        return expr
    
    @staticmethod
    def _l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        expr = 2*D1**2*Pk*f*fNL*(b01 + xb01)/(3*Mk1)
        
        if sigma != None:
            expr = -5*D1**2*Pk*fNL*(2*k1*sigma*(Mk1*b01*f*(2*k1**2*sigma**2 + 9) + 3*Mk1*b1*k1**2*sigma**2*xb01 + Mk1*f*xb01*(2*k1**2*sigma**2 + 9) + 3*b01*k1**2*sigma**2*(Mk1*xb1 + fNL*xb01)) + np.sqrt(2)*np.sqrt(np.pi)*(Mk1*b01*f*(k1**2*sigma**2 - 9) + Mk1*b1*k1**2*sigma**2*xb01*(k1**2*sigma**2 - 3) + Mk1*f*xb01*(k1**2*sigma**2 - 9) + b01*k1**2*sigma**2*(Mk1*xb1 + fNL*xb01)*(k1**2*sigma**2 - 3))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*Mk1**2*k1**5*sigma**5)
        
        return expr


@add_empty_methods_pk('l1','l3','l4')
class Loc:
    @staticmethod
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        return BaseFNL._l(mu,cosmo_funcs,k1,zz=zz,t=t,fNL=fNL,fNL_type='Loc')
        
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l0(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Loc')
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l2(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Loc')


@add_empty_methods_pk('l1','l3','l4')
class Eq:
    @staticmethod
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        return BaseFNL._l(mu,cosmo_funcs,k1,zz=zz,t=t,fNL= k1**2 *fNL,fNL_type='Eq')
        
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l0(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=k1**2 *fNL,fNL_type='Eq')
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l2(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=k1**2 *fNL,fNL_type='Eq')


@add_empty_methods_pk('l1','l3','l4')
class Orth:
    @staticmethod
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        return BaseFNL._l(mu,cosmo_funcs,k1,zz=zz,t=t,fNL=k1 *fNL,fNL_type='Orth')
        
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l0(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=k1 *fNL,fNL_type='Orth')
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return BaseFNL._l2(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=k1 *fNL,fNL_type='Orth')