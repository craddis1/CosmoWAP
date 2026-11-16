# A space for creating composite function which can be called for convenience 
# Also contains pk_func which is a wrapper function to call power spectrum terms in a standardised format

import numpy as np

import cosmo_wap.pk as pk
from cosmo_wap.lib.utils import composite as _composite
from cosmo_wap.pk import Pk_0, Pk_WS, Pk_WSGR


#so we want create a general power spectrum to have same format as bispectrum bk_func
def pk_func(term,l,cosmo_funcs,k1,zz=0,t=0,sigma=None,n=None,kernels=None,mu_grid=None,window=None,**kwargs):
    """Convenience function to call power spectrum terms in a standardised format. Wrapper function.
    Kwargs currently just deals with fNL shit, fNL_loc,fNL_orth etc
    l can be a list of multipoles; kernels e.g. ['N','I'] add the numeric-mu signal computed once across l
    window = (dr, k_f) puts the kernels' integrated terms through the bin's radial window - see numeric_mu.pk.get_int_window"""

    if not np.isscalar(l):# list of multipoles - analytic per l, kernels computed once
        with cosmo_funcs.unpack_cache():# all l and terms unpack the same k grid - do it once
            tot = np.array([pk_func(term,li,cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,n=n,**kwargs) for li in l]) if term else 0
            if kernels:
                tot = tot + pk_kernel_multipoles(kernels,l,cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,mu_grid=mu_grid,window=window,**kwargs)
        return tot

    if kernels:# single multipole - add numeric-mu kernels onto analytic terms
        tot = pk_func(term,l,cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,n=n,**kwargs) if term else 0
        return tot + pk_kernel_multipoles(kernels,[l],cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,mu_grid=mu_grid,window=window,**kwargs)[0]

    if isinstance(term, list):# so we can pass term as a list of contribtuions
        # then call recursively for each term
        tot = []
        for x in term:
             tot.append(pk_func(x,l,cosmo_funcs,k1,zz=zz,t=t,sigma=sigma,**kwargs))

        return np.sum(tot,axis=0)

    if sigma is not None:# the closed forms damp P by exp(-(k mu sigma)^2/2) - one field's - so sqrt(2) for both, as numeric_mu.pk
        sigma = np.sqrt(2)*sigma

    if isinstance(term, str):
        pk_class = getattr(pk,term)
        name = term
    else:
        pk_class = term
        name = pk_class.__name__# the checks below are on the name

    if name in ['Loc','Eq','Orth']:
        return getattr(pk_class, f'l{l}')(cosmo_funcs, k1, zz, t, sigma, **kwargs)

    if 'Int' in name:
        if n is None:
            n = cosmo_funcs.n
        args = (cosmo_funcs, k1, zz, t, sigma, n)
    else:
        args = (cosmo_funcs, k1, zz, t, sigma)
    return getattr(pk_class, f'l{l}')(*args)


def pk_kernel_multipoles(kernels, ln, cosmo_funcs, k1, zz=0, t=0, sigma=None, mu_grid=None, window=None, **kwargs):
    """numeric-mu signal multipoles for kernels e.g. ['N','LP','I'] - get P(k,mu) once then project each l
    kwargs (fNL, shape, ...) reach the kernels themselves - see numeric_mu.kernels.K1"""
    from cosmo_wap.numeric_mu import pk as numeric_mu_pk
    if mu_grid is None:# use the numeric pk defaults
        return numeric_mu_pk.get_multipoles(kernels, kernels, list(ln), cosmo_funcs, k1, zz, sigma=sigma, window=window, **kwargs)
    # [n_mu, GL, los_n, deg] with an optional 5th entry n_p - a None entry keeps the get_multipoles default
    grid = {k:v for k,v in zip(('n_mu','GL','n','deg','n_p'), mu_grid) if v is not None}
    return numeric_mu_pk.get_multipoles(kernels, kernels, list(ln), cosmo_funcs, k1, zz,
                                        sigma=sigma, window=window, **grid, **kwargs)


# composites of parts with one parity each, so each multipole sums the parts it can have - see lib.utils.composite
_GR = [(Pk_0.GR1, 1), (Pk_0.GR2, 0)]
_WS = [(Pk_WS.WA1, 1), (Pk_WS.RR1, 1), (Pk_WS.WA2, 0), (Pk_WS.WARR, 0), (Pk_WS.RR2, 0)]
_WS_GR = [(Pk_WSGR.WAGR, 0), (Pk_WSGR.RRGR, 0)]

GRL = _composite("GRL", "Local relativistic terms, GR1 + GR2", _GR, module=__name__)

class GRI: #for all Integrated GR
    """
    integrated relativistic terms, IxI + IxS
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.IntInt.l0(cosmo_funcs,k1,zz,t,sigma,n)+pk.IntNPP.l0(cosmo_funcs,k1,zz,t,sigma,n)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.IntInt.l1(cosmo_funcs,k1,zz,t,sigma,n)+pk.IntNPP.l1(cosmo_funcs,k1,zz,t,sigma,n)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.IntInt.l2(cosmo_funcs,k1,zz,t,sigma,n)+pk.IntNPP.l2(cosmo_funcs,k1,zz,t,sigma,n)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.IntInt.l3(cosmo_funcs,k1,zz,t,sigma,n)+pk.IntNPP.l3(cosmo_funcs,k1,zz,t,sigma,n)
    @staticmethod
    def l4(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.IntInt.l4(cosmo_funcs,k1,zz,t,sigma,n)+pk.IntNPP.l4(cosmo_funcs,k1,zz,t,sigma,n)

class GR: #for all Integrated GR
    """
    All relativistic terms, GRI + GRL
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.GRI.l0(cosmo_funcs,k1,zz,t,sigma,n)+pk.GRL.l0(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.GRI.l1(cosmo_funcs,k1,zz,t,sigma,n)+pk.GRL.l1(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.GRI.l2(cosmo_funcs,k1,zz,t,sigma,n)+pk.GRL.l2(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.GRI.l3(cosmo_funcs,k1,zz,t,sigma,n)+pk.GRL.l3(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l4(cosmo_funcs,k1,zz=0,t=0,sigma=None,n=128):
        return pk.GRI.l4(cosmo_funcs,k1,zz,t,sigma,n)+pk.GRL.l4(cosmo_funcs,k1,zz,t,sigma)

WS = _composite("WS", "Pure wide separation terms, RR + WA", _WS, module=__name__)
WS_GR = _composite("WS_GR", "just wide separation effects including relativistic mixing", _WS_GR, module=__name__)
WSGR = _composite("WSGR", "Full wide separation effects including relativistic mixing", _WS + _WS_GR, module=__name__)
Full = _composite("Full", "WS + GR", _WS + _WS_GR + _GR, module=__name__)


class ISW:#for all ISW
    """
    ISW
    """
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.ISWxISW.l0(cosmo_funcs,k1,zz,t,sigma)+pk.LxISW.l0(cosmo_funcs,k1,zz,t,sigma)+pk.TDxISW.l0(cosmo_funcs,k1,zz,t,sigma)+pk.ISWxNPP.l0(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.ISWxISW.l1(cosmo_funcs,k1,zz,t,sigma)+pk.LxISW.l1(cosmo_funcs,k1,zz,t,sigma)+pk.TDxISW.l1(cosmo_funcs,k1,zz,t,sigma)+pk.ISWxNPP.l1(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.ISWxISW.l2(cosmo_funcs,k1,zz,t,sigma)+pk.LxISW.l2(cosmo_funcs,k1,zz,t,sigma)+pk.TDxISW.l2(cosmo_funcs,k1,zz,t,sigma)+pk.ISWxNPP.l2(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.ISWxISW.l3(cosmo_funcs,k1,zz,t,sigma)+pk.LxISW.l3(cosmo_funcs,k1,zz,t,sigma)+pk.TDxISW.l3(cosmo_funcs,k1,zz,t,sigma)+pk.ISWxNPP.l3(cosmo_funcs,k1,zz,t,sigma)
    @staticmethod
    def l4(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.ISWxISW.l4(cosmo_funcs,k1,zz,t,sigma)+pk.LxISW.l4(cosmo_funcs,k1,zz,t,sigma)+pk.TDxISW.l4(cosmo_funcs,k1,zz,t,sigma)+pk.ISWxNPP.l4(cosmo_funcs,k1,zz,t,sigma)

