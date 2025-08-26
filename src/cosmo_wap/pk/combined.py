#create composite function which can be called for convenience
import cosmo_wap.pk as pk
import numpy as np
from cosmo_wap.lib.utils import add_empty_methods_pk

#so we want create a general power spectrum to have same format as bispectrum bk_func 
def pk_func(term,l,cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1,n=None):
    """Convenience function to call power spectrum terms in a standardised format. Wrapper function."""
    if isinstance(term, list):# so we can pass term as a list of contribtuions
        # then call recursively for each term
        tot = []
        for x in term:
             tot.append(pk_func(x,l,cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,fNL=fNL))
        
        return np.sum(tot,axis=0)
    
    if isinstance(term, str):
        pk_class = getattr(pk,term)
    else:
        pk_class = term
    
    if term in ['Loc','Eq','Orth']:
        args = (cosmo_funcs, k1, zz, t, sigma, fNL)
    else:
        if 'Int' in term:
            if n is None:
                n = cosmo_funcs.n
            args = (cosmo_funcs, k1, zz, t, sigma, n)
        else:
            args = (cosmo_funcs, k1, zz, t, sigma)
    return getattr(pk_class, f'l{l}')(*args)

@add_empty_methods_pk('l4')
class WS:#for all wide separation
    """
    Pure wide separation terms, RR + WA
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA2.l0(cosmo_funcs,k1,zz,t,sigma)+pk.WARR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.RR2.l0(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA1.l1(cosmo_funcs,k1,zz,t,sigma) + pk.RR1.l1(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA2.l2(cosmo_funcs,k1,zz,t,sigma)+pk.WARR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.RR2.l2(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA1.l3(cosmo_funcs,k1,zz,t,sigma) + pk.RR1.l3(cosmo_funcs,k1,zz,t,sigma)

@add_empty_methods_pk('l4')
class WSGR:
    """
    Full wide separation effects including relativistic mixing
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l0(cosmo_funcs,k1,zz,t,sigma)+pk.WAGR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.RRGR.l0(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l1(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l2(cosmo_funcs,k1,zz,t,sigma)+pk.WAGR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.RRGR.l2(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l3(cosmo_funcs,k1,zz,t,sigma)

@add_empty_methods_pk('l4')
class Full:
    """
    WS + GR
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WSGR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.GR2.l0(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l1(cosmo_funcs,k1,zz,t,sigma)+pk.GR1.l1(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WSGR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.GR2.l2(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l3(cosmo_funcs,k1,zz,t,sigma)+pk.GR1.l3(cosmo_funcs,k1,zz,t,sigma)
