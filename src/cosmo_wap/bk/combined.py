#create composite function which can be called for convenience
import cosmo_wap.bk as bk
import cosmo_wap.lib.integrate as integrate
import numpy as np

#so we want create a general bispectrum function like COV.cov()
def bk_func(term,l,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,m=0,sigma=None,fNL=None):
    """Convenience function to call bispectrum terms in a standardised format including FoG."""

    if isinstance(term, list):# so we can pass term as a list of contribtuions
        # then call recursively for each term
        tot = []
        for x in term:
             tot.append(bk_func(x,l,cosmo_funcs,k1,k2,k3,theta,zz=zz,r=r,s=s,m=m,sigma=sigma,fNL=fNL))
        
        return np.sum(tot,axis=0)
    

    if isinstance(term, str):# if it's string get the class from bk
        bk_class = getattr(bk,term)
    else:
        bk_class = term
    
    kwargs = {}
    if fNL is not None: # if fNL is provided, then pass it as an keyword argument
        kwargs['fNL'] = fNL

    if sigma is None:
        return getattr(bk_class, f'l{l}')(cosmo_funcs,k1,k2,k3,theta,zz,r,s,**kwargs)
    else:
        return bk.ylm(l,m,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma=sigma,**kwargs)

def ylm_picker(l,m,terms,*args,**kwargs):
    """Sums the contribution using ylm method from terms provided.
    Is a bit of a hack to allow for different terms to be used in the same way.
    Also is more efficient e.g. only calls odd term for odd multipoles ...
    Args:
        l (int): Multipole order.
        m (int): Multipole order.
        terms (list): List of terms to include in the sum.
        *args: arguments of bk methods, e.g. cosmo_funcs, k1, k2, k3, theta, zz, r, s.
        **kwargs: see above.
        Returns:
        contribution (array): Sum of contributions from the terms."""
    subterms = [term for term in terms if '1' in term]
    if l % 2 == 1:
        subterms = [term for term in terms if '1' in term]
    else:
        subterms = [term for term in terms if '1' not in term]
    contribution = None
    for term in subterms:
        bk_class = getattr(bk,term)
        result = bk_class.ylm(l,m,*args,**kwargs)
        if contribution is None:
            contribution = result
        else:
            contribution += result
    return contribution

class WS:#for all wide separation
    """
    Pure wide separation terms, RR + WA
    """
    @staticmethod
    def ylm(l,m,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,sigma=None):
        return ylm_picker(l,m,['WA1','WA2','WARR','RR1','RR2'],cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma)
    @staticmethod
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WARR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s) + bk.RR1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WARR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RR2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
class WSGR:
    """
    Full wide separation effects including relativistic mixing
    """
    @staticmethod
    def ylm(l,m,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,sigma=None):
        return ylm_picker(l,m,['WA1','WA2','WARR','RR1','RR2','WAGR','RRGR'],cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma)
    @staticmethod
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WAGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RRGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WAGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RRGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
class Full:
    """
    WS + GR
    """
    @staticmethod
    def ylm(l,m,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,sigma=None):
        return ylm_picker(l,m,['WA1','WA2','WARR','RR1','RR2','WAGR','RRGR','GR1','GR2'],cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma)
    @staticmethod
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WSGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    @staticmethod
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WSGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)