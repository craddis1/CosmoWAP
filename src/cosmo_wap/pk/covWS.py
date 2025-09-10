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
        
        expr = -4*D1**4*Pk**2*(315*b1*b11*b12*b13 + 35*f**4 + 45*f**3*(b1 + b11 + b12 + b13) + 63*f**2*(b1*(b11 + b12 + b13) + b11*(b12 + b13) + b12*b13) + 105*f*(b1*b11*b12 + b1*b13*(b11 + b12) + b11*b12*b13))/315 + 2*D1**2*Pk*(15*b1*b12*n24 + 5*b11*n13*(3*b13 + f) + f*(5*b13*n13 + 3*f*(n13 + n24) + 5*n24*(b1 + b12)))/15 + 2*D1**2*Pk*(15*b1*b13*n23 + 5*b11*n14*(3*b12 + f) + f*(5*b12*n14 + 3*f*(n14 + n23) + 5*n23*(b1 + b13)))/15 + 2*n13*n24 + 2*n14*n23
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
        
        expr = -4*D1**2*f*(10*D1**2*f**3*(19*Pk**2 - 2*Pk*k1*(-17*Pkd + Pkdd*k1) - 2*Pkd**2*k1**2) + 11*D1**2*f**2*(4*b1*(3*Pk**2 + 2*Pk*k1*(Pkd - Pkdd*k1) - 2*Pkd**2*k1**2) + 4*b11*(3*Pk**2 + 2*Pk*k1*(Pkd - Pkdd*k1) - 2*Pkd**2*k1**2) + 3*b12*(7*Pk**2 + 2*Pk*k1*(9*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2)) + 77*b1*(3*Pk + k1*(5*Pkd + Pkdd*k1))*(n23 + n24) + 77*b12*(D1**2*b1*b11*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + n14*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + n24*(3*Pk + k1*(5*Pkd + Pkdd*k1))) + 11*b13*(D1**2*(b1*(7*b11*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + 14*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(17*Pk**2 + 2*Pk*k1*(19*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2)) + b11*(14*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(17*Pk**2 + 2*Pk*k1*(19*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2)) + 3*f*(4*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(7*Pk**2 + 2*Pk*k1*(9*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2))) + 7*n13*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 7*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1))) + 11*f*(-2*D1**2*Pk**2*b1*b11 + 17*D1**2*Pk**2*b1*b12 + 17*D1**2*Pk**2*b11*b12 - 44*D1**2*Pk*Pkd*b1*b11*k1 + 38*D1**2*Pk*Pkd*b1*b12*k1 + 38*D1**2*Pk*Pkd*b11*b12*k1 - 20*D1**2*Pk*Pkdd*b1*b11*k1**2 + 2*D1**2*Pk*Pkdd*b1*b12*k1**2 + 2*D1**2*Pk*Pkdd*b11*b12*k1**2 - 20*D1**2*Pkd**2*b1*b11*k1**2 + 2*D1**2*Pkd**2*b1*b12*k1**2 + 2*D1**2*Pkd**2*b11*b12*k1**2 + 18*Pk*n23 + 18*Pk*n24 + 30*Pkd*k1*n23 + 30*Pkd*k1*n24 + 6*Pkdd*k1**2*n23 + 6*Pkdd*k1**2*n24 - n13*(Pk + k1*(11*Pkd + 5*Pkdd*k1)) - n14*(Pk + 11*Pkd*k1 + 5*Pkdd*k1**2)))/(1155*d**2*k1**2)
        return expr
    
    @staticmethod
    def l11(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,Pkd,Pkdd,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,WS=True)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = (-2*D1**2*(13*b1*(10*D1**2*f**3*(12*Pk**2 + Pk*k1*(62*Pkd + 19*Pkdd*k1) + 19*Pkd**2*k1**2) + 22*D1**2*f**2*(9*b11*(Pk**2 + Pk*k1*(4*Pkd + Pkdd*k1) + Pkd**2*k1**2) - b12*(12*Pk**2 + Pk*k1*(58*Pkd + 17*Pkdd*k1) + 17*Pkd**2*k1**2)) + 462*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 33*b13*(2*D1**2*(7*b11*(5*b12 + 3*f)*(Pk**2 + Pk*k1*(4*Pkd + Pkdd*k1) + Pkd**2*k1**2) + f*(3*b12*(4*Pk**2 + Pk*k1*(18*Pkd + 5*Pkdd*k1) + 5*Pkd**2*k1**2) + f*(12*Pk**2 + Pk*k1*(58*Pkd + 17*Pkdd*k1) + 17*Pkd**2*k1**2))) + 7*n23*(4*Pk + k1*(10*Pkd + 3*Pkdd*k1))) - 33*f*(14*D1**2*b11*b12*(Pk**2 + Pk*k1*(4*Pkd + Pkdd*k1) + Pkd**2*k1**2) + 12*Pk*n23 - 3*k1*n23*(-2*Pkd + Pkdd*k1) - 12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)))) + f*(70*D1**2*f**3*(11*Pk**2 + Pk*k1*(80*Pkd + 29*Pkdd*k1) + 29*Pkd**2*k1**2) + 130*D1**2*f**2*(b11*(12*Pk**2 + Pk*k1*(62*Pkd + 19*Pkdd*k1) + 19*Pkd**2*k1**2) - b12*(13*Pk**2 + Pk*k1*(80*Pkd + 27*Pkdd*k1) + 27*Pkd**2*k1**2)) + 5148*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) - 13*b13*(-2*D1**2*(33*b11*(3*b12*(4*Pk**2 + Pk*k1*(18*Pkd + 5*Pkdd*k1) + 5*Pkd**2*k1**2) + f*(12*Pk**2 + Pk*k1*(58*Pkd + 17*Pkdd*k1) + 17*Pkd**2*k1**2)) + 5*f*(11*b12*(3*Pk**2 + Pk*k1*(16*Pkd + 5*Pkdd*k1) + 5*Pkd**2*k1**2) + 3*f*(13*Pk**2 + Pk*k1*(80*Pkd + 27*Pkdd*k1) + 27*Pkd**2*k1**2))) + 99*n23*(4*Pk + k1*(2*Pkd - Pkdd*k1))) - 143*f*(2*D1**2*b11*b12*(12*Pk**2 + Pk*k1*(58*Pkd + 17*Pkdd*k1) + 17*Pkd**2*k1**2) + 5*n23*(12*Pk + k1*(14*Pkd + Pkdd*k1)) - 30*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1))))) - 1716*n13*(D1**2*(b11*(7*b13*(3*Pk + 5*Pkd*k1 + Pkdd*k1**2) - f*(Pk + 11*Pkd*k1 + 5*Pkdd*k1**2)) + 2*f*(9*Pk*b13 + 3*Pk*f + 3*b13*k1*(5*Pkd + Pkdd*k1) + f*k1*(Pkd - Pkdd*k1))) + 21*n24) - 858*n14*(D1**2*(7*b11*b12*(4*Pk + k1*(10*Pkd + 3*Pkdd*k1)) + 3*b11*f*(2*Pk + k1*(8*Pkd + 3*Pkdd*k1)) - 3*b12*f*(4*Pk + k1*(2*Pkd - Pkdd*k1)) + f**2*(-6*Pk + k1*(8*Pkd + 7*Pkdd*k1))) + 28*n23))/(5005*d**2*k1**2)
        return expr
    
    @staticmethod
    def l22(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,Pkd,Pkdd,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,WS=True)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = (-10*D1**2*(b1*(2*D1**2*f**3*(2584*Pk**2 + Pk*k1*(4566*Pkd - 301*Pkdd*k1) - 301*Pkd**2*k1**2) + 26*D1**2*f**2*(b11*(179*Pk**2 + Pk*k1*(252*Pkd - 53*Pkdd*k1) - 53*Pkd**2*k1**2) + b12*(202*Pk**2 + Pk*k1*(358*Pkd - 23*Pkdd*k1) - 23*Pkd**2*k1**2)) + 858*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) - 13*b13*(2*D1**2*(11*b11*(9*b12*(Pk**2 + Pk*k1*(8*Pkd + 3*Pkdd*k1) + 3*Pkd**2*k1**2) + f*(-15*Pk**2 + Pk*k1*(-28*Pkd + Pkdd*k1) + Pkd**2*k1**2)) - f*(11*b12*(18*Pk**2 + Pk*k1*(38*Pkd + Pkdd*k1) + Pkd**2*k1**2) + f*(184*Pk**2 + Pk*k1*(386*Pkd + 9*Pkdd*k1) + 9*Pkd**2*k1**2))) + 33*n23*(12*Pk + k1*(34*Pkd + 11*Pkdd*k1))) + 143*f*(2*D1**2*b11*b12*(21*Pk**2 + Pk*k1*(40*Pkd - Pkdd*k1) - Pkd**2*k1**2) + n23*(12*Pk - k1*(2*Pkd + 7*Pkdd*k1)) + 8*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)))) + f*(130*D1**2*f**3*(41*Pk**2 + Pk*k1*(80*Pkd - Pkdd*k1) - Pkd**2*k1**2) + 2*D1**2*f**2*(b11*(2584*Pk**2 + Pk*k1*(4566*Pkd - 301*Pkdd*k1) - 301*Pkd**2*k1**2) + b12*(2697*Pk**2 + Pk*k1*(5428*Pkd + 17*Pkdd*k1) + 17*Pkd**2*k1**2)) + 1144*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + b13*(2*D1**2*(13*b11*(11*b12*(18*Pk**2 + Pk*k1*(38*Pkd + Pkdd*k1) + Pkd**2*k1**2) + f*(184*Pk**2 + Pk*k1*(386*Pkd + 9*Pkdd*k1) + 9*Pkd**2*k1**2)) + f*(39*b12*(69*Pk**2 + Pk*k1*(164*Pkd + 13*Pkdd*k1) + 13*Pkd**2*k1**2) + f*(2527*Pk**2 + Pk*k1*(5728*Pkd + 337*Pkdd*k1) + 337*Pkd**2*k1**2))) + 143*n23*(12*Pk - k1*(2*Pkd + 7*Pkdd*k1))) + 13*f*(2*D1**2*b11*b12*(202*Pk**2 + Pk*k1*(358*Pkd - 23*Pkdd*k1) - 23*Pkd**2*k1**2) + 3*n23*(60*Pk + k1*(34*Pkd - 13*Pkdd*k1)) + 78*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1))))) - 260*n13*(D1**2*(33*b11*(3*Pk*b13 + 5*Pk*f + b13*k1*(5*Pkd + Pkdd*k1) + f*k1*(7*Pkd + Pkdd*k1)) + 4*f*(33*Pk*b13 + 28*Pk*f + 11*b13*k1*(5*Pkd + Pkdd*k1) + 2*f*k1*(16*Pkd + Pkdd*k1))) + 99*n24) + 130*n14*(D1**2*(33*b11*(12*Pk*b12 - 2*Pk*f + b12*k1*(34*Pkd + 11*Pkdd*k1) + f*k1*(4*Pkd + 3*Pkdd*k1)) + f*(-132*Pk*b12 - 134*Pk*f + 11*b12*k1*(2*Pkd + 7*Pkdd*k1) + f*k1*(4*Pkd + 69*Pkdd*k1))) + 396*n23))/(3003*d**2*k1**2)
        return expr
    
    @staticmethod
    def l20(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,Pkd,Pkdd,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,WS=True)
        _,_,_,b12,b13 = cosmo_funcs_list[t3][t4].unpack_pk(k1,zz)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = (-2*D1**2*(13*b1*(2*D1**2*f**3*(219*Pk**2 + 2*Pk*k1*(233*Pkd + 7*Pkdd*k1) + 14*Pkd**2*k1**2) + 22*D1**2*f**2*(b11*(19*Pk**2 - 2*Pk*k1*(-17*Pkd + Pkdd*k1) - 2*Pkd**2*k1**2) + b12*(23*Pk**2 + 2*Pk*k1*(31*Pkd + 4*Pkdd*k1) + 8*Pkd**2*k1**2)) + 231*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 11*b13*(2*D1**2*(21*b11*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + b11*f*(15*Pk**2 + 22*Pk*Pkd*k1 - 4*Pk*Pkdd*k1**2 - 4*Pkd**2*k1**2) + 11*b12*f*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f**2*(23*Pk**2 + 2*Pk*k1*(31*Pkd + 4*Pkdd*k1) + 8*Pkd**2*k1**2)) + 21*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1))) + 11*f*(2*D1**2*b11*b12*(15*Pk**2 + 22*Pk*Pkd*k1 - 4*Pk*Pkdd*k1**2 - 4*Pkd**2*k1**2) + 11*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 11*n24*(3*Pk + 5*Pkd*k1 + Pkdd*k1**2))) + f*(10*D1**2*f**3*(565*Pk**2 + 2*Pk*k1*(647*Pkd + 41*Pkdd*k1) + 82*Pkd**2*k1**2) + 26*D1**2*f**2*(b11*(219*Pk**2 + 2*Pk*k1*(233*Pkd + 7*Pkdd*k1) + 14*Pkd**2*k1**2) + b12*(237*Pk**2 + 2*Pk*k1*(329*Pkd + 46*Pkdd*k1) + 92*Pkd**2*k1**2)) + 1573*b12*n24*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 13*b13*(2*D1**2*(11*b11*(11*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(23*Pk**2 + 2*Pk*k1*(31*Pkd + 4*Pkdd*k1) + 8*Pkd**2*k1**2)) + f*(99*b12*(3*Pk**2 + 2*Pk*k1*(5*Pkd + Pkdd*k1) + 2*Pkd**2*k1**2) + f*(237*Pk**2 + 2*Pk*k1*(329*Pkd + 46*Pkdd*k1) + 92*Pkd**2*k1**2))) + 121*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1))) + 143*f*(2*D1**2*b11*b12*(23*Pk**2 + 62*Pk*Pkd*k1 + 8*Pk*Pkdd*k1**2 + 8*Pkd**2*k1**2) + 9*n23*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + 9*n24*(3*Pk + 5*Pkd*k1 + Pkdd*k1**2)))) - 286*n13*(D1**2*(21*b11*b13*(3*Pk + k1*(5*Pkd + Pkdd*k1)) - 3*b11*f*(Pk + k1*(11*Pkd + 5*Pkdd*k1)) + 11*b13*f*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + f**2*(19*Pk + k1*(17*Pkd - Pkdd*k1))) + 63*n24) - 286*n14*(D1**2*(21*b11*b12*(3*Pk + k1*(5*Pkd + Pkdd*k1)) - 3*b11*f*(Pk + k1*(11*Pkd + 5*Pkdd*k1)) + 11*b12*f*(3*Pk + k1*(5*Pkd + Pkdd*k1)) + f**2*(19*Pk + k1*(17*Pkd - Pkdd*k1))) + 63*n23))/(3003*d**2*k1**2)
        return expr