import numpy as np
from scipy.special import erf  # Error function needed from integral over FoG

class BaseFNL:
    """fNL terms for pk are the fungible except for the k^{alpha} - so we can make this short and supremely sweet"""
    @staticmethod
    def _l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        return D1**2*Pk*fNL*(b01*(f*mu**2 + xb1) + xb01*(b1 + f*mu**2))/Mk1
    
    @staticmethod
    def _l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        expr = D1**2*Pk*fNL*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        if sigma != None:
            expr = D1**2*Pk*fNL*(-2*f*k1*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + k1**2*sigma**2*xb1) + xb01*(b1*k1**2*sigma**2 + f))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*Mk1*k1**3*sigma**3)
        
        return expr
    
    @staticmethod
    def _l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1, fNL_type='Loc'):
        Pk,f,D1,b1,xb1,b01,Mk1,xb01 = cosmo_funcs.unpack_pk(k1,zz,fNL_type=fNL_type)
        
        expr = 2*D1**2*Pk*f*fNL*(b01 + xb01)/(3*Mk1)
        
        if sigma != None:
            expr = -5*D1**2*Pk*fNL*(2*k1*sigma*(b01*f*(2*k1**2*sigma**2 + 9) + 3*b01*k1**2*sigma**2*xb1 + 3*b1*k1**2*sigma**2*xb01 + f*xb01*(2*k1**2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(k1**2*sigma**2 - 9) + b01*k1**2*sigma**2*xb1*(k1**2*sigma**2 - 3) + b1*k1**2*sigma**2*xb01*(k1**2*sigma**2 - 3) + f*xb01*(k1**2*sigma**2 - 9))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*Mk1*k1**5*sigma**5)
        
        return expr

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
    
class Eq:
    @staticmethod
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        return k1**2 * BaseFNL._l(mu,cosmo_funcs,k1,zz=zz,t=t,fNL=fNL,fNL_type='Eq')
        
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return k1**2 * BaseFNL._l0(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Eq')
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return k1**2 * BaseFNL._l2(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Eq')
    
class Orth:
    @staticmethod
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        return k1 * BaseFNL._l(mu,cosmo_funcs,k1,zz=zz,t=t,fNL=fNL,fNL_type='Orth')
        
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return k1 * BaseFNL._l0(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Orth')
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        return k1 * BaseFNL._l2(cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL,fNL_type='Orth')