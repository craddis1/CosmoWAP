#create composite function which can be called for convenience
#import cosmo_wap.lib.angular_integrate as angular_integrate
import numpy as np

import cosmo_wap.bk as bk
import cosmo_wap.bk_mt as bk_mt
from cosmo_wap.lib.utils import composite as _composite


#so we want create a general bispectrum function like COV.cov()
def bk_func(term,l,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,m=0,sigma=None,kernels=None,mu_grid=None,**kwargs):
    """Convenience function to call bispectrum terms in a standardised format including FoG.
    l can be a list of multipoles; kernels e.g. ['N','LP'] add the numeric-mu signal computed once across l - see bk_kernel_multipoles"""

    if not np.isscalar(l):# list of multipoles - analytic per l, kernels computed once. Entries l or (l, m)
        lm = [tuple(li) if isinstance(li,(tuple,list)) else (li,m) for li in l]
        if term and any(mi for _,mi in lm):
            raise NotImplementedError("m != 0 in a list of multipoles is bk_kernels only - pass the analytic terms as kernels")
        with cosmo_funcs.unpack_cache():# all l and terms unpack the same triangles - do it once
            tot = np.array([bk_func(term,li,cosmo_funcs,k1,k2,k3,theta,zz=zz,r=r,s=s,m=m,sigma=sigma,**kwargs) for li,_ in lm]) if term else 0
            if kernels:
                tot = tot + bk_kernel_multipoles(kernels,lm,cosmo_funcs,k1,k2,k3,theta,zz=zz,sigma=sigma,mu_grid=mu_grid,**kwargs)
        return tot

    if kernels:# single multipole - add numeric-mu kernels onto analytic terms
        tot = bk_func(term,l,cosmo_funcs,k1,k2,k3,theta,zz=zz,r=r,s=s,m=m,sigma=sigma,**kwargs) if term else 0
        return tot + bk_kernel_multipoles(kernels,[(l,m)],cosmo_funcs,k1,k2,k3,theta,zz=zz,sigma=sigma,mu_grid=mu_grid,**kwargs)[0]

    if isinstance(term, list):# so we can pass term as a list of contribtuions
        # then call recursively for each term
        tot = []
        with cosmo_funcs.unpack_cache():# every term unpacks the same triangles - do it once
            for x in term:
                 tot.append(bk_func(x,l,cosmo_funcs,k1,k2,k3,theta,zz=zz,r=r,s=s,m=m,sigma=sigma,**kwargs))

        return np.sum(tot,axis=0)

    if cosmo_funcs.multi_tracer: # if multi-tracer use the multi-tracer compliant terms.
        bkclass = bk_mt
    else:
        bkclass = bk

    if isinstance(term, str):# if it's string get the class from bk
        bk_class = getattr(bkclass,term)
    else:
        bk_class = term

    wargs = {}
    if term in ['Loc','Eq','Orth']: # then pass fNL as keyword argument
        wargs = kwargs

    if sigma is None:
        return getattr(bk_class, f'l{l}')(cosmo_funcs,k1,k2,k3,theta,zz,r,s,**wargs)

    # FoG acts inside the angular integral, so it needs the term's raw B(mu,phi) through its
    # ylm method - see _needs_ylm for the terms that have one
    _needs_ylm(bk_class, cosmo_funcs)
    return bk_class.ylm(l,m,cosmo_funcs,k1,k2,k3,theta,zz,r,s,sigma=sigma,**wargs)


def bk_kernel_multipoles(kernels,lm,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,sigma=None,mu_grid=None,**kwargs):
    """numeric-mu signal multipoles for kernels e.g. ['N','LP'] on every field - get B(mu,phi) once then project each (l,m)
    kwargs (n_mu, n_phi, fNL, ...) reach numeric_mu.bk.get_multipoles and the kernels"""
    from cosmo_wap.numeric_mu import bk as numeric_mu_bk
    # [n_mu, n_phi] - a None entry keeps the get_multipoles default, as pk_kernel_multipoles
    grid = {k:v for k,v in zip(('n_mu','n_phi'), mu_grid) if v is not None} if mu_grid is not None else {}
    return numeric_mu_bk.get_multipoles(kernels,kernels,kernels,lm,cosmo_funcs,k1,k2,k3,theta,zz=zz,sigma=sigma,**grid,**kwargs)


def _needs_ylm(bk_class, cosmo_funcs=None):
    """Raise unless bk_class can be integrated numerically over the LOS orientation.

    bk_mt's GR, NPP and PNG terms have no ylm, so FoG is unavailable for those (and for its Full)."""
    if not hasattr(bk_class, 'ylm'):
        name = getattr(bk_class, '__name__', bk_class)
        mt = ' in multi-tracer' if cosmo_funcs is not None and cosmo_funcs.multi_tracer else ''
        raise NotImplementedError(f"FoG (sigma) is not implemented for '{name}'{mt}")


# the composites, as [(part, parity)] - built by _install_composites at the end of cosmo_wap.bk, once every part has its
# C kernels, tables and WS_ multipoles - see lib.utils.composite
_WS = [("WA1", 1), ("RR1", 1), ("WA2", 0), ("WARR", 0), ("RR2", 0)]
_WS_GR = [("WAGR", 0), ("RRGR", 0)]
_COMPOSITES = {
    "WS": ("Pure wide separation terms, RR + WA", _WS),
    "WSGR": ("Full wide separation effects including relativistic mixing", _WS + _WS_GR),
    "Full": ("WS + GR", _WS + _WS_GR + [("GR1", 1), ("GR2", 0)]),
}


def _install_composites(namespace):
    for name, (doc, parts) in _COMPOSITES.items():
        cls = _composite(name, doc, [(namespace[p], parity) for p, parity in parts], bk=True, module=__name__)
        namespace[name] = globals()[name] = cls
