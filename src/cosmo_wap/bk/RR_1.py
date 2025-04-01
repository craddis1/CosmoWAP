import numpy as np

def RR_1(mu,phi,cosmo_funcs,k1,k2,k3=None,theta=None,zz=0,r=0,s=0):
    
    #get generic cosmology parameters
    k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2 = cosmo_funcs.get_params(k1,k2,k3,theta,zz)
    
    fd,Dd,gd2,bd2,bd1,fdd,Ddd,gdd2,bdd2,bdd1 = cosmo_funcs.get_derivs(zz)
    
    mu2 = mu*np.cos(theta)+(1-mu)**2 *np.sin(theta)*np.cos(phi)
    st = np.sin(theta)
    ct= np.cos(theta)

    perm12 = 1
    perm13 = 1
    perm23 = 1
    return (perm12+perm13+perm23)