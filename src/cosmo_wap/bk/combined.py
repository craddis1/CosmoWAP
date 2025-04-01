#create composite function which can be called for convenience
import cosmo_wap.bk as bk


class WS:#for all wide separation
    """
    Pure wide separation terms, RR + WA
    """
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WARR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s) + bk.RR1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return bk.WA2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WARR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RR2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
class WSGR:
    """
    Full wide separation effects including relativistic mixing
    """
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WAGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RRGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.WAGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.RRGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
class Full:
    """
    WS + GR
    """
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WSGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l1(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WS.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)
    
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        return WSGR.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)+bk.GR2.l2(cosmo_funcs,k1,k2,k3,theta,zz,r,s)