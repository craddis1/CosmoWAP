import numpy as np

class cov_NPP:
    @staticmethod
    def l00(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11 = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = D1**2*Pk*(2*D1**2*Pk*(315*b1*b11*b12*b13 + 35*f**4 + 45*f**3*(b1 + b11 + b12 + b13) + 63*f**2*(b1*(b11 + b12 + b13) + b11*(b12 + b13) + b12*b13) + 105*f*(b1*b11*b12 + b1*b13*(b11 + b12) + b11*b12*b13)) + 105*b11*(3*b12*n14 + 3*b13*n13 + f*(n13 + n14)) + 105*b13*(3*b1*n23 + f*(n13 + n23)) + 21*f*(5*b1*n23 + 5*b12*n14 + 3*f*(n13 + n14 + n23)) + 21*n24*(15*b1*b12 + 3*f**2 + 5*f*(b1 + b12)))/315 + n13*n24 + n14*n23
        return expr

class cov_WA2:
    @staticmethod
    def l00(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,Pkd,Pkdd,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,WS=True)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = -2*D1**2*f*(10*D1**2*f**3*(41*Pk**2 - 2*Pk*k1*(93*Pkd + 67*Pkdd*k1) - 134*Pkd**2*k1**2) - 11*D1**2*f**2*(4*b1*(3*Pk**2 + 2*Pk*k1*(41*Pkd + 19*Pkdd*k1) + 38*Pkd**2*k1**2) + 3*b11*(-12*Pk**2 + 8*Pk*k1*(-Pkd + Pkdd*k1) + 8*Pkd**2*k1**2) + 3*b12*(3*Pk**2 + 2*Pk*k1*(61*Pkd + 29*Pkdd*k1) + 58*Pkd**2*k1**2)) + 231*b1*(3*Pk + k1*(5*Pkd + Pkdd*k1))*(n23 + n24) + 231*b12*(D1**2*b1*b11*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + n14*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + n24*(3*Pk + k1*(5*Pkd + Pkdd*k1))) + 33*b13*(D1**2*(b1*(7*b11*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + 14*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) - f*(23*Pk**2 + 2*Pk*k1*(85*Pkd + 31*Pkdd*k1) + 62*Pkd**2*k1**2)) + b11*(14*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(17*Pk**2 + 2*Pk*k1*(19*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2)) - f*(4*b12*(11*Pk**2 + 2*Pk*k1*(37*Pkd + 13*Pkdd*k1) + 26*Pkd**2*k1**2) + f*(3*Pk**2 + 2*Pk*k1*(61*Pkd + 29*Pkdd*k1) + 58*Pkd**2*k1**2))) + 7*n13*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 7*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1))) - 33*f*(2*D1**2*Pk**2*b1*b11 + 23*D1**2*Pk**2*b1*b12 - 17*D1**2*Pk**2*b11*b12 + 44*D1**2*Pk*Pkd*b1*b11*k1 + 170*D1**2*Pk*Pkd*b1*b12*k1 - 38*D1**2*Pk*Pkd*b11*b12*k1 + 20*D1**2*Pk*Pkdd*b1*b11*k1**2 + 62*D1**2*Pk*Pkdd*b1*b12*k1**2 - 2*D1**2*Pk*Pkdd*b11*b12*k1**2 + 20*D1**2*Pkd**2*b1*b11*k1**2 + 62*D1**2*Pkd**2*b1*b12*k1**2 - 2*D1**2*Pkd**2*b11*b12*k1**2 - 18*Pk*n23 - 18*Pk*n24 - 30*Pkd*k1*n23 - 30*Pkd*k1*n24 - 6*Pkdd*k1**2*n23 - 6*Pkdd*k1**2*n24 + n13*(Pk + k1*(11*Pkd + 5*Pkdd*k1)) + n14*(Pk + 11*Pkd*k1 + 5*Pkdd*k1**2)))/(3465*d**2*k1**2)
        return expr
    
    @staticmethod
    def l11(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,Pkd,Pkdd,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,WS=True)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = (D1**2*(-13*b1*(10*D1**2*f**3*(-6*Pk**2 + Pk*k1*(46*Pkd + 29*Pkdd*k1) + 29*Pkd**2*k1**2) - 22*D1**2*f**2*(b11*(3*Pk**2 - Pk*k1*(8*Pkd + 7*Pkdd*k1) - 7*Pkd**2*k1**2) - 3*b12*(6*Pk**2 + Pk*k1*(74*Pkd + 31*Pkdd*k1) + 31*Pkd**2*k1**2)) - 231*b12*n24*(4*Pk + k1*(10*Pkd + 3*Pkdd*k1)) + 22*b13*(D1**2*(21*b11*b12*(9*Pk**2 + Pk*k1*(40*Pkd + 11*Pkdd*k1) + 11*Pkd**2*k1**2) - 9*b11*f*(Pk**2 + Pk*k1*(8*Pkd + 3*Pkdd*k1) + 3*Pkd**2*k1**2) + 9*b12*f*(6*Pk**2 + Pk*k1*(34*Pkd + 11*Pkdd*k1) + 11*Pkd**2*k1**2) - f**2*(6*Pk**2 + Pk*k1*(74*Pkd + 31*Pkdd*k1) + 31*Pkd**2*k1**2)) + 21*n23*(11*Pk + k1*(25*Pkd + 7*Pkdd*k1))) + 33*f*(18*D1**2*b11*b12*(Pk**2 + Pk*k1*(8*Pkd + 3*Pkdd*k1) + 3*Pkd**2*k1**2) + 12*n23*(2*Pk + k1*(8*Pkd + 3*Pkdd*k1)) - 7*n24*(4*Pk + k1*(14*Pkd + 5*Pkdd*k1)))) + f*(70*D1**2*f**3*(17*Pk**2 - Pk*k1*(4*Pkd + 19*Pkdd*k1) - 19*Pkd**2*k1**2) + 130*D1**2*f**2*(b11*(10*Pk**2 + Pk*k1*(26*Pkd + 3*Pkdd*k1) + 3*Pkd**2*k1**2) + 21*b12*(Pk**2 - Pk*k1*(4*Pkd + 3*Pkdd*k1) - 3*Pkd**2*k1**2)) + 3003*b12*n24*(4*Pk + k1*(14*Pkd + 5*Pkdd*k1)) - 26*b13*(D1**2*(-33*b11*b12*(2*Pk**2 + Pk*k1*(2*Pkd - Pkdd*k1) - Pkd**2*k1**2) + 11*b11*f*(6*Pk**2 + Pk*k1*(14*Pkd + Pkdd*k1) + Pkd**2*k1**2) + 11*b12*f*(3*Pk**2 + Pk*k1*(52*Pkd + 23*Pkdd*k1) + 23*Pkd**2*k1**2) + 35*f**2*(Pk**2 - Pk*k1*(4*Pkd + 3*Pkdd*k1) - 3*Pkd**2*k1**2)) + 198*n23*(2*Pk + k1*(8*Pkd + 3*Pkdd*k1))) + 143*f*(6*D1**2*b11*b12*(6*Pk**2 + Pk*k1*(14*Pkd + Pkdd*k1) + Pkd**2*k1**2) + 10*n23*(3*Pk - k1*(7*Pkd + 5*Pkdd*k1)) + n24*(-12*Pk + k1*(106*Pkd + 59*Pkdd*k1))))) + 429*n13*(D1**2*(7*b11*b13*(4*Pk + k1*(10*Pkd + 3*Pkdd*k1)) + 3*b11*f*(2*Pk + k1*(8*Pkd + 3*Pkdd*k1)) - 3*b13*f*(4*Pk + k1*(2*Pkd - Pkdd*k1)) + f**2*(-6*Pk + k1*(8*Pkd + 7*Pkdd*k1))) + 28*n24) - 858*n14*(D1**2*(7*b11*b12*(11*Pk + k1*(25*Pkd + 7*Pkdd*k1)) + b11*f*(11*Pk + k1*(37*Pkd + 13*Pkdd*k1)) + 6*b12*f*(2*Pk + k1*(8*Pkd + 3*Pkdd*k1)) + 10*f**2*k1*(2*Pkd + Pkdd*k1)) + 77*n23))/(5005*d**2*k1**2)
        return expr
    