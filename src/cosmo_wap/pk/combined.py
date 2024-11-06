#creating composite functions is also useful
class WS:#for all wide separation
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return pk.WA2.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.WARR.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.RR2.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)
    
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return pk.WA1.l1(cosmo_funcs,k1,zz,t,sigma,nonlin) + pk.RR1.l1(cosmo_funcs,k1,zz,t,sigma,nonlin)
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return pk.WA2.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.WARR.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.RR2.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)
    
class allWSGR:
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return WS.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.WAGR.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.RRGR.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.GR2.l0(cosmo_funcs,k1,zz,t,sigma,nonlin)
    
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return WS.l1(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.GR1.l1(cosmo_funcs,k1,zz,t,sigma,nonlin)
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=0,nonlin=False):
        return WS.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.WAGR.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.RRGR.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)+pk.GR2.l2(cosmo_funcs,k1,zz,t,sigma,nonlin)
