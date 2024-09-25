import numpy as np

def RR_1(params,derivs,r,s,mu,phi):
    k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2 = params
    fd,Dd,gd2,bd2,bd1,fdd,Ddd,gdd2,bdd2,bdd1 = derivs

    mu2 = mu*np.cos(theta)+(1-mu)**2 *np.sin(theta)*np.cos(phi)
    st = np.sin(theta)
    ct= np.cos(theta)

    perm12 = 1
    perm13 = 1
    perm23 = 1
    return (perm12+perm13+perm23)