#import numpy as np
#from cosmo_wap.integrated import BaseInt
#from cosmo_wap.lib import utils
#import numpy as np

class Unpack:
    @staticmethod
    def common(cosmo_funcs,zz,k1,tracer=0): #kaiser
        """get base things unpacked"""

        f = cosmo_funcs.f(zz)
        D1 = cosmo_funcs.D(zz)
        b1 = cosmo_funcs.survey[tracer].b_1(zz)

        return D1,f,b1
    
    @staticmethod
    def get_int_params(cosmo_funcs, zz, tracer=0):
        """Get Source quatities for integrated power spectra"""
        d = cosmo_funcs.comoving_dist(zz)
        H = cosmo_funcs.H_c(zz)
        Hp = -(1+zz)*H*cosmo_funcs.dH_c(zz) # dH_dt - deriv wrt to conformal time! - equivalently: (1-(3/2)*cosmo_funcs.Om_m(zz))*H**2
        #OM = cosmo_funcs.Om_m(zz)
        Qm = cosmo_funcs.survey[tracer].Q(zz)
        be = cosmo_funcs.survey[tracer].be(zz)

        return d, H, Hp, Qm, be
    
    @staticmethod
    def get_integrand_params(cosmo_funcs, xd):
        """Get parameters that are funcs of xd"""
        # convert comoving distance to redshift
        zzd = cosmo_funcs.d_to_z(xd)
        # get interpolated values
        fd = cosmo_funcs.f(zzd)
        D1d = cosmo_funcs.D(zzd)
        Hd = cosmo_funcs.H_c(zzd)
        OMd = cosmo_funcs.Om_m(zzd)
        return zzd, fd, D1d, Hd, OMd

# store kernels
class K1:
    @staticmethod
    def N(cosmo_funcs,zz,mu,k1,tracer=0): #kaiser
        #unpack all necessary terms
        D1,f,b1 = Unpack.common(cosmo_funcs,zz,k1,tracer=tracer)
        return D1*(b1 + f *mu**2)
    
    @staticmethod
    def LP(cosmo_funcs,zz,mu,k1,tracer=0): # local projection effects
        #unpack all necessary terms
        D1,_,_ = Unpack.common(cosmo_funcs,zz,k1,tracer=tracer)
        gr1,gr2   = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey[tracer])[:2]
        return D1*(1j*mu*gr1/k1 + gr2/k1**2)
    
    def L(r,cosmo_funcs,zz=0,mu=None,k1=None,tracer=0):

        d, _, _, Qm, _ = Unpack.get_int_params(cosmo_funcs, zz, tracer=tracer) # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        tmp_arr = 3*D1_r*(Qm-1)*OM_r*H_r**2*(d-r)*r/d   # [1-mu**2+2i mu/r*q] *

        if k1 is None: # get k,mu seperated dict
            tmp_dict = {} # so store in dictionary {mu: {k: something}}
            tmp_dict[0] = {}
            tmp_dict[2] = {}
            tmp_dict[1] = {}
            tmp_dict[0][0] = tmp_arr
            tmp_dict[2][0] = -tmp_arr
            tmp_dict[1][-1] = tmp_arr*(2j/r)
            return tmp_dict
        
        return tmp_arr*(1-mu**2 + 2j*mu/(r*k1))
    
    def TD(r,cosmo_funcs,zz=0,mu=None,k1=None,tracer=0):

        d, _, _, Qm, _ = Unpack.get_int_params(cosmo_funcs, zz, tracer=tracer) # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        tmp_arr = 6*D1_r*(Qm-1)*OM_r*H_r**2/(d) # k1**2 *

        if k1 is None: # get k,mu seperated dict
            tmp_dict = {} # so store in dictionary {mu: {k: something}}
            tmp_dict[0] = {}
            tmp_dict[0][-2] = tmp_arr
            return tmp_dict
        
        return tmp_arr/(k1**2)
    
    def ISW(r,cosmo_funcs,zz=0,mu=None,k1=None,tracer=0):

        d, H, Hp, Qm, be = Unpack.get_int_params(cosmo_funcs, zz, tracer=tracer) # source integrated params
        _, f_r, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        tmp_arr = 3*D1_r*(be-2*Qm+2*(Qm-1)/(d*H)-Hp/H**2)*OM_r*H_r**3*(f_r-1)   # k1**2 *

        if k1 is None: # get k,mu seperated dict
            tmp_dict = {} # so store in dictionary {mu: {k: something}}
            tmp_dict[0] = {}
            tmp_dict[0][-2] = tmp_arr
            return tmp_dict
        
        return tmp_arr/(k1**2)
    
    def I(r,cosmo_funcs,zz=0,mu=None,k1=None,tracer=0):
        """Combined (L+TD+ISW) integrated 1st order kernel"""
        d, H, Hp, Qm, be = Unpack.get_int_params(cosmo_funcs, zz, tracer=tracer) # source integrated params
        _, f_r, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        tmp_arr_L = 3*D1_r*(Qm-1)*OM_r*H_r**2*(d-r)*r/d
        tmp_arr_TD = 6*D1_r*(Qm-1)*OM_r*H_r**2/(d)
        tmp_arr_ISW = 3*D1_r*(be-2*Qm+2*(Qm-1)/(d*H)-Hp/H**2)*OM_r*H_r**3*(f_r-1)

        if k1 is None: # get k,mu seperated dict
            tmp_dict = {} # so store in dictionary {mu: {k: something}}
            tmp_dict[0] = {}
            tmp_dict[2] = {}
            tmp_dict[1] = {}
            tmp_dict[0][0] = tmp_arr_L
            tmp_dict[2][0] = -tmp_arr_L
            tmp_dict[1][-1] = tmp_arr_L*(2j/r)
        
            tmp_dict[0][-2] = tmp_arr_TD + tmp_arr_ISW
            return tmp_dict

        return tmp_arr_L*(1-mu**2 + 2j* mu/(r*k1)) + tmp_arr_TD/(k1**2) + tmp_arr_ISW/(k1**2)

    @staticmethod
    def kappa_g(r,cosmo_funcs,zz,mu,k1,tracer=0): #kaiser
        d, _, _, Qm, _ = Unpack.get_int_params(cosmo_funcs, zz, tracer=tracer) # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r) # integrand params - arrays in shape (xd)

        tmp_arr = (3/2)*D1_r*OM_r*H_r**2*(d-r)*r/d   # [1-mu**2+2i mu/r*q] *

        if k1 is None: # get k,mu seperated dict
            tmp_dict = {} # so store in dictionary {mu: {k: something}}
            tmp_dict[0] = {}
            tmp_dict[2] = {}
            tmp_dict[1] = {}
            tmp_dict[0][0] = tmp_arr
            tmp_dict[2][0] = -tmp_arr
            tmp_dict[1][-1] = tmp_arr*(2j/r)
            return tmp_dict
        
        return tmp_arr*(1-mu**2 + 2j*mu/(r*k1))
