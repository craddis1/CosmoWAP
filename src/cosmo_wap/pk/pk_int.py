import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

from cosmo_wap.lib.kernels import K1


def Ir1(p,mu,cosmo_funcs,zz,kernel):
    # shape (mu,p) # p is some combination of r2,k and mu

    func = K1.getattr(kernel)

    r1 = np.linspace(0,cosmo_funcs.comoving_dist(zz),128)

    arr = func(mu,cosmo_funcs,zz=zz,r=r1)

    abc = np.fft.fft(term)
    
    return 1