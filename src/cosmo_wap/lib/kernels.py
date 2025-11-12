
import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

# store kernels
class K1:
    @staticmethod
    def KN(mu,cosmo_funcs,k1,zz=0): #kaiser
        #unpack all necessary terms
        _,f,D1,b1,_ = cosmo_funcs.unpack_pk(k1,zz)
        return D1*(b1 + f *mu^2)
    
    def KTD(mu,cosmo_funcs,k1,zz=0,r=None):

        d, H, Hp, Qm, _, be, _ = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        z_r, f_r, D1_r, H_r, OM_r = BaseInt.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        return 6*D1_r*(Qm-1)*OM_r*H_r**2/(k1**2 *d)
    

    def KTD1(mu,cosmo_funcs,k1,zz=0,r=None):

        d, H, Hp, Qm, _, be, _ = BaseInt.get_int_params(cosmo_funcs, zz) # source integrated params
        z_r, f_r, D1_r, H_r, OM_r = BaseInt.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        return 6*D1_r*(Qm-1)*OM_r*H_r**2/(d)   # k1**2 *