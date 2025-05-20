import scipy
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
import cosmo_wap as cw
import cosmo_wap.pk as pk

# import and define cosmlogy in class
from classy import Class

class BaseInt:
    """Base Integral class with defined integration funcs and parameter getter funcs"""
    @staticmethod
    def get_int_params(cosmo_funcs, zz=0):
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        H = cosmo_funcs.H_c(zz)
        OM = cosmo_funcs.Om(zz)
        Qm = cosmo_funcs.survey.Q_survey(zz)
        xQm = cosmo_funcs.survey1.Q_survey(zz)

        be = cosmo_funcs.survey.be_survey(zz)
        xbe = cosmo_funcs.survey1.be_survey(zz)

        return b1, xb1, H, OM, Qm, xQm, be, xbe

    @staticmethod
    def get_integrand_params(cosmo_funcs, xd):
        """Get parameters that are funcs of xd"""
        # convert comoving distance to redshift
        zzd = cosmo_funcs.d_to_z(xd)
        # get interpolated values
        fd = cosmo_funcs.f_intp(zzd)
        D1d = cosmo_funcs.D_intp(zzd)
        Hd = cosmo_funcs.H_c(zzd)
        OMd = cosmo_funcs.Om(zzd)
        return zzd, fd, D1d, Hd, OMd

    @staticmethod
    def single_int(func, cosmo_funcs, k1, zz=0, t=0, sigma=None, n=16):
        """Do single integral for RSDxIntegrated term"""

        nodes, weights = np.polynomial.legendre.leggauss(n)  # legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)

        # so limits [0,d]
        d = cosmo_funcs.comoving_dist(zz)

        # define nodes in comoving distance: for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
        xd_nodes = (d) * (nodes + 1) / 2.0  # sample phi range [0,d]

        # call term func
        int_grid = func(xd_nodes, cosmo_funcs, k1, zz, t, sigma)

        # (x1-x0)/2
        return (d) / 2.0 * np.sum(weights * int_grid, axis=(-1))  # sum over last

    @staticmethod
    def double_int(func, cosmo_funcs, k1, zz=0, t=0, sigma=None, n1=16, n2=None):
        """Do double integral for IntegratedxIntegrated term"""
        
        if n2 is None:
            n2 = n1

        # legendre gauss - get nodes and weights for given n
        nodes1, weights1 = np.polynomial.legendre.leggauss(n1)
        nodes2, weights2 = np.polynomial.legendre.leggauss(n2)
        nodes1 = np.real(nodes1)
        nodes2 = np.real(nodes2)

        # so limits [0,d]
        d = cosmo_funcs.comoving_dist(zz)

        #  define nodes in comoving distance for limits [x0,x1]:(x1)*(nodes+1)/2.0 - x0
        xd_nodes1 = (d) * (nodes1 + 1) / 2.0  # sample phi range [0,d]
        xd_nodes2 = (d) * (nodes2 + 1) / 2.0

        # so for last two axis we need to define the nodes and weights on the grid
        xd1 = xd_nodes1[:, np.newaxis]
        xd2 = xd_nodes2[np.newaxis, :]

        weights1 = weights1[:, np.newaxis]
        weights2 = weights2[np.newaxis, :]
        
        int_grid = func(xd1,xd2,cosmo_funcs, k1, zz)# get back 2D grid

        # (x1-x0)/2
        # sum over last 2 axis
        return ((d) / 2.0) ** 2 * np.sum(weights1 * weights2 * int_grid, axis=(1, 2))


