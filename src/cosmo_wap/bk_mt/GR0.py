import numpy as np
from numpy import cos
import cosmo_wap.bk as bk
import cosmo_wap
   
#Netwonian, plane parallel constant redshift limit
class NPP:   
    @staticmethod
    def l0(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):

        #get generic cosmology parameters
        k1,k2,k3,theta,Pk1,Pk2,Pk3,f,D1,K,C,b1,xb1,yb1,b2,xb2,yb2,g2,xg2,yg2 = cosmo_funcs.unpack_bk(k1,k2,k3,theta,zz)
        
        return 1
    
    @staticmethod
    def l2(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        
        #get generic cosmology parameters
        k1,k2,k3,theta,Pk1,Pk2,Pk3,f,D1,K,C,b1,xb1,yb1,b2,xb2,yb2,g2,xg2,yg2 = cosmo_funcs.unpack_bk(k1,k2,k3,theta,zz)
        
        return 1
        
    @staticmethod
    def l4(cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
        
        #get generic cosmology parameters
        k1,k2,k3,theta,Pk1,Pk2,Pk3,f,D1,K,C,b1,xb1,yb1,b2,xb2,yb2,g2,xg2,yg2 = cosmo_funcs.unpack_bk(k1,k2,k3,theta,zz)

        return 1