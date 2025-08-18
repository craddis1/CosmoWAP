"""
So following https://arxiv.org/pdf/1909.12069 - Semi-analytic models for linear bias for different tracers.


"""

import numpy as np


class SAM:
    """Semi-anlaytic model with free parameters from table 2 in 1909.12069
    (∫_x^inf \phi(x) b_1(x) dx)/(∫_x^inf \phi(x) dx)
    """
    def __init__(self, cosmo=None):
        """
        H-alpha Luminosity Function calculator
        Luminsotiy in Units h^3 Mpc^-3
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7,2,1000)
    
    def Ha(self):
        a = 0.844
        b = 0.116
        c = 42.623
        d = 1.186
        e = 1.765
        return a,b,c,d,e

    def b_1(self,x,zz):
        #Ha
        a,b,c,d,e = self.Ha()

        return a+b*(1+zz)**e *(1+np.exp((x-c)*d))
    
    def get_b_1(self,LF):


        return 1



