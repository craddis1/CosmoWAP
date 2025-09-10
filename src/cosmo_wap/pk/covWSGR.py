import numpy as np

class cov_WAGR:
    @staticmethod
    def l00(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,gr1,_,gr11,_,Pkd,_,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True,WS=True)
        _,_,_,b12,b13,gr12,_,gr13,_ = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = 4*D1**2*(-5*D1**2*Pk*f**3*(Pk + 2*Pkd*k1)*(2*gr1 - 6*gr11 + gr12 + gr13) - 9*D1**2*Pk*f**2*(Pk + 2*Pkd*k1)*(b1*(-2*gr11 + gr12 + gr13) + b11*(6*gr1 + gr12 + gr13) - b12*(4*gr11 + gr13)) + 105*b1*(Pk + Pkd*k1)*(gr12*n24 + gr13*n23) - 105*b12*(-D1**2*Pk*b1*b11*gr13*(Pk + 2*Pkd*k1) + gr1*n24*(Pk + Pkd*k1) + gr11*n14*(Pk + Pkd*k1)) - 3*b13*(-D1**2*Pk*gr12*(Pk + 2*Pkd*k1)*(7*b1*(5*b11 + f) + f*(7*b11 + 3*f)) + 7*gr1*(2*D1**2*Pk*(Pk + 2*Pkd*k1)*(5*b11*b12 + 2*b11*f - b12*f) + 5*n23*(Pk + Pkd*k1)) + gr11*(2*D1**2*Pk*(Pk + 2*Pkd*k1)*(35*b1*b12 - f*(7*b12 + 6*f)) + 35*n13*(Pk + Pkd*k1))) - 21*f*(D1**2*Pk**2*b1*b11*gr12 + 4*D1**2*Pk**2*b11*b12*gr1 + 2*D1**2*Pk*Pkd*b1*b11*gr12*k1 + 8*D1**2*Pk*Pkd*b11*b12*gr1*k1 + 3*Pk*gr1*n23 + 3*Pk*gr1*n24 - 2*Pk*gr12*n14 - 3*Pk*gr12*n24 + 3*Pkd*gr1*k1*n23 + 3*Pkd*gr1*k1*n24 - 2*Pkd*gr12*k1*n14 - 3*Pkd*gr12*k1*n24 + gr11*(Pk + Pkd*k1)*(n13 + n14) + gr13*(D1**2*Pk*(Pk + 2*Pkd*k1)*(b1*(b11 - b12) - b11*b12) - 2*n13*(Pk + Pkd*k1) - 3*n23*(Pk + Pkd*k1))))/(315*d*k1**2)
        return expr
    
    @staticmethod
    def l11(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,gr1,_,gr11,_,Pkd,_,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True,WS=True)
        _,_,_,b12,b13,gr12,_,gr13,_ = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = 4*D1**2*(-35*D1**2*Pk*f**3*(Pk + 2*Pkd*k1)*(gr1 - gr11 - 6*gr12 + 6*gr13) - 55*D1**2*Pk*f**2*(Pk + 2*Pkd*k1)*(-b1*(gr11 + 2*gr12 - 2*gr13) + b11*(gr1 - 10*gr12 + 10*gr13) + b12*(-gr1 + gr11 + 4*gr13)) - 231*b11*(4*D1**2*Pk*b1*b12*gr13*(Pk + 2*Pkd*k1) - gr12*n14*(Pk + Pkd*k1) + 2*gr13*n13*(Pk + Pkd*k1)) + 11*b13*(-3*D1**2*Pk*gr1*(Pk + 2*Pkd*k1)*(7*b11*b12 + 9*b11*f + 3*b12*f + 5*f**2) + 4*D1**2*Pk*gr12*(Pk + 2*Pkd*k1)*(21*b1*b11 + f*(18*b11 + 5*f)) + 42*gr1*n23*(Pk + Pkd*k1) + 3*gr11*(D1**2*Pk*(Pk + 2*Pkd*k1)*(7*b1*b12 + 9*b1*f + 3*b12*f + 5*f**2) + 7*n13*(Pk + Pkd*k1))) + 99*f*(-6*D1**2*Pk**2*b1*b11*gr13 + D1**2*Pk**2*b11*b12*gr1 - 8*D1**2*Pk**2*b11*b12*gr13 - 12*D1**2*Pk*Pkd*b1*b11*gr13*k1 + 2*D1**2*Pk*Pkd*b11*b12*gr1*k1 - 16*D1**2*Pk*Pkd*b11*b12*gr13*k1 + 6*D1**2*Pk*b1*b11*gr12*(Pk + 2*Pkd*k1) - D1**2*Pk*b1*b12*gr11*(Pk + 2*Pkd*k1) + 4*Pk*gr1*n23 - 5*Pk*gr1*n24 - 4*Pk*gr13*n23 + 4*Pkd*gr1*k1*n23 - 5*Pkd*gr1*k1*n24 - 4*Pkd*gr13*k1*n23 + gr11*n13*(Pk + Pkd*k1) - gr12*n14*(Pk + Pkd*k1) + 5*gr12*n24*(Pk + Pkd*k1)) - 231*(Pk + Pkd*k1)*(-3*b1*gr12*n24 + 2*b1*gr13*n23 + 3*b12*gr1*n24))/(385*d*k1**2)
        return expr
    
    @staticmethod
    def l22(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,gr1,_,gr11,_,Pkd,_,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True,WS=True)
        _,_,_,b12,b13,gr12,_,gr13,_ = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = 20*D1**2*(-5*D1**2*Pk*f**3*(Pk + 2*Pkd*k1)*(523*gr1 - 687*gr11 - 106*gr12 + 188*gr13) - 39*D1**2*Pk*f**2*(Pk + 2*Pkd*k1)*(b1*(-89*gr11 - 18*gr12 + 32*gr13) + b11*(117*gr1 - 18*gr12 + 32*gr13) + b12*(75*gr1 - 103*gr11 + 18*gr13)) - 429*b11*(-2*D1**2*Pk*b1*b12*gr13*(Pk + 2*Pkd*k1) + 3*gr12*n14*(Pk + Pkd*k1) + 6*gr13*n13*(Pk + Pkd*k1)) + 39*b13*(4*D1**2*Pk*gr12*(Pk + 2*Pkd*k1)*(11*b1*(2*b11 + f) + f*(11*b11 + 8*f)) - gr1*(D1**2*Pk*(Pk + 2*Pkd*k1)*(209*b11*b12 + 143*b11*f + 77*b12*f + 75*f**2) + 88*n23*(Pk + Pkd*k1)) + gr11*(-D1**2*Pk*(Pk + 2*Pkd*k1)*(11*b1*(b12 - 9*f) - f*(121*b12 + 103*f)) + 11*n13*(Pk + Pkd*k1))) - 429*f*(-2*D1**2*Pk**2*b1*b11*gr12 + 13*D1**2*Pk**2*b11*b12*gr1 - 4*D1**2*Pk*Pkd*b1*b11*gr12*k1 + 26*D1**2*Pk*Pkd*b11*b12*gr1*k1 + 6*Pk*gr1*n23 + 9*Pk*gr1*n24 + Pk*gr12*n14 - 9*Pk*gr12*n24 + 6*Pkd*gr1*k1*n23 + 9*Pkd*gr1*k1*n24 + Pkd*gr12*k1*n14 - 9*Pkd*gr12*k1*n24 - gr11*(9*D1**2*Pk*b1*b12*(Pk + 2*Pkd*k1) + 5*n13*(Pk + Pkd*k1) + 2*n14*(Pk + Pkd*k1)) + 2*gr13*(D1**2*Pk*(Pk + 2*Pkd*k1)*(b1*(2*b11 + b12) + b11*b12) + 2*n13*(Pk + Pkd*k1) - 3*n23*(Pk + Pkd*k1))) - 429*(Pk + Pkd*k1)*(-11*b1*gr12*n24 - 8*b1*gr13*n23 + 11*b12*gr1*n24 + 2*b12*gr11*n14))/(9009*d*k1**2)
        return expr
    
    @staticmethod
    def l20(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,gr1,_,gr11,_,Pkd,_,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True,WS=True)
        _,_,_,b12,b13,gr12,_,gr13,_ = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = -4*D1**2*(5*D1**2*Pk*f**3*(Pk + 2*Pkd*k1)*(52*gr1 - 72*gr11 + 5*gr12 + 5*gr13) + 33*D1**2*Pk*f**2*(Pk + 2*Pkd*k1)*(b1*(-12*gr11 + gr12 + gr13) + b11*(16*gr1 + gr12 + gr13) + b12*(10*gr1 - 14*gr11 - gr13)) + 231*b11*(D1**2*Pk*b1*b12*gr13*(Pk + 2*Pkd*k1) + 3*gr12*n14*(Pk + Pkd*k1) + 3*gr13*n13*(Pk + Pkd*k1)) - 33*b13*(-D1**2*Pk*gr12*(Pk + 2*Pkd*k1)*(b1*(7*b11 - f) - f*(b11 + f)) - 2*gr1*(D1**2*Pk*(Pk + 2*Pkd*k1)*(14*b11*b12 + 11*b11*f + 8*b12*f + 5*f**2) + 7*n23*(Pk + Pkd*k1)) + 2*gr11*(D1**2*Pk*(Pk + 2*Pkd*k1)*(28*b1*b12 + 9*b1*f + 10*b12*f + 7*f**2) + 14*n13*(Pk + Pkd*k1))) - 33*f*(-D1**2*Pk**2*b1*b11*gr12 - 22*D1**2*Pk**2*b11*b12*gr1 - 2*D1**2*Pk*Pkd*b1*b11*gr12*k1 - 44*D1**2*Pk*Pkd*b11*b12*gr1*k1 - 12*Pk*gr1*n23 - 12*Pk*gr1*n24 - 7*Pk*gr12*n14 + 12*Pk*gr12*n24 - 12*Pkd*gr1*k1*n23 - 12*Pkd*gr1*k1*n24 - 7*Pkd*gr12*k1*n14 + 12*Pkd*gr12*k1*n24 + 2*gr11*(9*D1**2*Pk*b1*b12*(Pk + 2*Pkd*k1) + 4*n13*(Pk + Pkd*k1) + 4*n14*(Pk + Pkd*k1)) + gr13*(-D1**2*Pk*(Pk + 2*Pkd*k1)*(b1*b11 - b1*b12 - b11*b12) - 7*n13*(Pk + Pkd*k1) + 12*n23*(Pk + Pkd*k1))) - 462*(Pk + Pkd*k1)*(b1*gr12*n24 + b1*gr13*n23 - b12*gr1*n24 + 2*b12*gr11*n14))/(693*d*k1**2)
        return expr
    
    @staticmethod
    def l02(cosmo_funcs_list,t1,t2,t3,t4,k1,zz=0):
        Pk,f,D1,b1,b11,gr1,_,gr11,_,Pkd,_,d = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True,WS=True)
        _,_,_,b12,b13,gr12,_,gr13,_ = cosmo_funcs_list[t1][t2].unpack_pk(k1,zz,GR=True)

        n13 = 1/cosmo_funcs_list[t1][t3].n_g(zz)
        n14 = 1/cosmo_funcs_list[t1][t4].n_g(zz)
        n23 = 1/cosmo_funcs_list[t2][t3].n_g(zz)
        n24 = 1/cosmo_funcs_list[t2][t4].n_g(zz)
        
        expr = 4*D1**2*(-5*D1**2*Pk*f**3*(Pk + 2*Pkd*k1)*(31*gr1 - 51*gr11 - 16*gr12 + 26*gr13) - 33*D1**2*Pk*f**2*(Pk + 2*Pkd*k1)*(b1*(-7*gr11 - 4*gr12 + 6*gr13) + b11*(11*gr1 - 4*gr12 + 6*gr13) + b12*(5*gr1 - 9*gr11 + 4*gr13)) - 231*b11*gr13*(4*D1**2*Pk*b1*b12*(Pk + 2*Pkd*k1) + 3*n13*(Pk + Pkd*k1)) + 33*b13*(-D1**2*Pk*gr1*(Pk + 2*Pkd*k1)*(7*b11*b12 + 13*b11*f + 7*b12*f + 5*f**2) + D1**2*Pk*gr11*(Pk + 2*Pkd*k1)*(35*b1*b12 + 9*b1*f + 11*b12*f + 9*f**2) + 2*D1**2*Pk*gr12*(Pk + 2*Pkd*k1)*(7*b1*b11 + 5*b1*f + 5*b11*f + 3*f**2) + 7*gr1*n23*(Pk + Pkd*k1) + 28*gr11*n13*(Pk + Pkd*k1)) - 33*f*(-8*D1**2*Pk**2*b1*b11*gr12 + 13*D1**2*Pk**2*b11*b12*gr1 - 16*D1**2*Pk*Pkd*b1*b11*gr12*k1 + 26*D1**2*Pk*Pkd*b11*b12*gr1*k1 + 3*Pk*gr1*n23 + 12*Pk*gr1*n24 - 2*Pk*gr12*n14 - 12*Pk*gr12*n24 + 3*Pkd*gr1*k1*n23 + 12*Pkd*gr1*k1*n24 - 2*Pkd*gr12*k1*n14 - 12*Pkd*gr12*k1*n24 + gr11*(-9*D1**2*Pk*b1*b12*(Pk + 2*Pkd*k1) - 8*n13*(Pk + Pkd*k1) + n14*(Pk + Pkd*k1)) + gr13*(2*D1**2*Pk*(Pk + 2*Pkd*k1)*(5*b1*b11 + 4*b1*b12 + 4*b11*b12) + 7*n13*(Pk + Pkd*k1) - 3*n23*(Pk + Pkd*k1))) + 231*(Pk + Pkd*k1)*(2*b1*gr12*n24 - b1*gr13*n23 - 2*b12*gr1*n24 + b12*gr11*n14))/(693*d*k1**2)
        return expr