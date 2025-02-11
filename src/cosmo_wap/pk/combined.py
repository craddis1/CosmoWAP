#create composite function which can be called for convenience
import cosmo_wap.pk as pk

class WS:#for all wide separation
    """
    Pure wide separation terms, RR + WA
    """
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA2.l0(cosmo_funcs,k1,zz,t,sigma)+pk.WARR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.RR2.l0(cosmo_funcs,k1,zz,t,sigma)
    
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA1.l1(cosmo_funcs,k1,zz,t,sigma) + pk.RR1.l1(cosmo_funcs,k1,zz,t,sigma)
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return pk.WA2.l2(cosmo_funcs,k1,zz,t,sigma)+pk.WARR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.RR2.l2(cosmo_funcs,k1,zz,t,sigma)

class WSGR:
    """
    Full wide separation effects including relativistic mixing
    """
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l0(cosmo_funcs,k1,zz,t,sigma)+pk.WAGR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.RRGR.l0(cosmo_funcs,k1,zz,t,sigma)
    
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l1(cosmo_funcs,k1,zz,t,sigma)
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l2(cosmo_funcs,k1,zz,t,sigma)+pk.WAGR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.RRGR.l2(cosmo_funcs,k1,zz,t,sigma)
    
class Full:
    """
    WS + GR
    """
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WSGR.l0(cosmo_funcs,k1,zz,t,sigma)+pk.GR2.l0(cosmo_funcs,k1,zz,t,sigma)
    
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WS.l1(cosmo_funcs,k1,zz,t,sigma)+pk.GR1.l1(cosmo_funcs,k1,zz,t,sigma)
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        return WSGR.l2(cosmo_funcs,k1,zz,t,sigma)+pk.GR2.l2(cosmo_funcs,k1,zz,t,sigma)
