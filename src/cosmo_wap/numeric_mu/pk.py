import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from .integration import compute_robust_integral, filon_integrate
from .kernels import K1
from .utils import merge_kernel_dicts, split_kernels


def get_int_K1(kernel, cosmo_funcs, zz, deg=8, n_p=5000):
    """Get values - dont interpolate as very oscillatory - so just call as is very quick"""
    p_arr = np.concatenate((-np.logspace(-6, 4, n_p)[::-1], np.logspace(-6, 4, n_p)))
    d = cosmo_funcs.comoving_dist(zz)
    r1 = np.linspace(1e-3, d, 1000)

    arr_dict = {}
    for kern in kernel:
        func = getattr(K1, kern)

        tmp_dict = func(r1, cosmo_funcs, zz=zz, ti=0)
        merge_kernel_dicts(arr_dict, tmp_dict)

    # now get interpolated function in p for all kernels
    for i in arr_dict:
        for j in arr_dict[i]:
            I_arr, _ = compute_robust_integral(d, p_arr, r1, arr_dict[i][j], deg=deg)
            arr_dict[i][j] = [CubicSpline(p_arr, I_arr.real), CubicSpline(p_arr, I_arr.imag)]

    return arr_dict


def get_int_r1(p, kernel, cosmo_funcs, zz, deg=8, n_p=2000):
    """Get values - dont interpolate as very oscillatory - so just call as is very quick"""
    p_arr = p.flatten()
    d = cosmo_funcs.comoving_dist(zz)
    r1 = np.linspace(1e-3, d - 1e-3, n_p)

    # could precompute these few lines
    arr_dict = {}
    for kern in kernel:
        func = getattr(K1, kern)

        tmp_dict = func(r1, cosmo_funcs, zz=zz, ti=0)
        merge_kernel_dicts(arr_dict, tmp_dict)

    # now get interpolated function in p for all kernels
    for i in arr_dict:
        for j in arr_dict[i]:
            I_arr, _ = compute_robust_integral(d, p_arr, r1, arr_dict[i][j], deg=deg)
            arr_dict[i][j] = I_arr.reshape(p.shape)

    return arr_dict


def get_int_K2(kernel, r2, cosmo_funcs, zz, mu, kk):
    """Get kernel 2 array"""
    d = cosmo_funcs.comoving_dist(zz)
    mu, kk = utils.enable_broadcasting(mu, kk, n=1)

    G = r2 / d
    qq = kk / G
    k2_arr = np.zeros((kk.shape[0], mu.shape[0], r2.shape[0]), dtype=np.complex128)

    for kern in kernel:
        func = getattr(K1, kern)
        k2_arr += func(r2, cosmo_funcs, zz, -mu, qq, ti=1)

    return k2_arr


def I1_sum(arr_dict, r2_arr, mu, kk, cosmo_funcs, zz, n=128, I2=False):
    baseint = BaseInt(cosmo_funcs)
    d = cosmo_funcs.comoving_dist(zz)
    if I2:  # II
        nodes, weights = np.polynomial.legendre.leggauss(n)  # legendre gauss - get nodes and weights for given
        r2 = (d) * (nodes + 1) / 2.0  # sample r range [0,d]
        mu, kk = utils.enable_broadcasting(mu, kk, n=1)
        G = r2 / d
        qq = kk / G
    else:
        # only IS
        qq = kk

    tot_arr = np.zeros((len(kk), len(mu)), dtype=np.complex128)  # shape (mu,kk)
    for i in arr_dict.keys():
        for j in arr_dict[i].keys():
            coef = qq**j * mu**i  # this is the mu from first order field

            r1_arr = arr_dict[i][j][0](qq * mu) + 1j * arr_dict[i][j][1](qq * mu)  # get complex array

            tmp_arr = np.exp(-1j * kk * mu * d) * coef * r2_arr * r1_arr
            if I2:
                tot_arr += ((d) / 2.0) * np.sum(G ** (-3) * baseint.pk(qq, zz) * weights * tmp_arr, axis=-1)
            else:
                tot_arr += baseint.pk(qq, zz) * tmp_arr

    return tot_arr


def s1_sum(s_k1, r2_arr, mu, kk, cosmo_funcs, zz, n=128, I2=False):
    """Now for case where S if first field"""
    baseint = BaseInt(cosmo_funcs)
    if I2:  # SI
        # so this integral can be a little oscillatory near the source - we use filon integration
        d = cosmo_funcs.comoving_dist(zz)

        nodes, _ = np.polynomial.legendre.leggauss(n)  # legendre gauss - get nodes and weights for given
        r2 = (d) * (nodes + 1) / 2.0  # sample r range [0,d]

        # u = d / r = 1 / G
        u = d / r2  # Note: u_grid now goes from 1.0 to (d/r_min)

        # (Recall: dr = -d/u^2 du, and G^-3 = u^3, so net is d*u)
        Jac = d * u  # ignore the negative as we also switch integration directions

        mu, kk = utils.enable_broadcasting(mu, kk, n=1)
        qq = kk * u

        # Full Integrand
        s1_arr = get_K(s_k1, cosmo_funcs, zz, mu, qq, ti=0)
        integrand = Jac * baseint.pk(qq, zz) * r2_arr * s1_arr

        # u_grid is now strictly increasing, so Filon works perfectly.
        tot_arr = np.exp(-1j * kk[..., 0] * mu[..., 0] * d) * filon_integrate(u[::-1], kk, mu, integrand[..., ::-1], d)

    else:
        # only SS
        s1_arr = get_K(s_k1, cosmo_funcs, zz, mu, kk, ti=0)
        tot_arr = baseint.pk(kk, zz) * r2_arr * s1_arr

    return tot_arr


def get_K(kernels, cosmo_funcs, zz, mu, kk, ti=0):
    tot_arr = np.zeros(np.broadcast_shapes(kk.shape, mu.shape), dtype=np.complex128)
    for kern in kernels:
        func = getattr(K1, kern)
        tot_arr += func(cosmo_funcs, zz, mu, kk, ti=ti)
    return tot_arr


def get_mu(mu, kernels1, kernels2, cosmo_funcs, kk, zz, n=16, deg=8, nr=2000):
    """Collect power spectrum contribution"""

    d = cosmo_funcs.comoving_dist(zz)
    nodes, _ = np.polynomial.legendre.leggauss(n)  # legendre gauss - get nodes and weights for given
    r2 = (d) * (nodes + 1) / 2.0  # sample r range [0,d]

    # lets split kernels into integrated and not!
    int_k1, s_k1 = split_kernels(kernels1)
    int_k2, s_k2 = split_kernels(kernels2)

    # precompute ------------------------------------------------------------------------
    if s_k2:
        s2_arr = get_K(s_k2, cosmo_funcs, zz, -mu, kk, ti=1)
    if int_k1:
        arr_dict = get_int_K1(int_k1, cosmo_funcs, zz, deg=deg)  # is dict
    if int_k2:
        r2_arr = get_int_K2(int_k2, r2, cosmo_funcs, zz, mu, kk)  # is array

    # sum stuff -----------------------------------------------------------------------------
    tot_arr = np.zeros(np.broadcast_shapes(kk.shape, mu.shape), dtype=np.complex128)  # shape (mu,kk)
    if int_k1:
        if int_k2:  # II
            tot_arr += I1_sum(arr_dict, r2_arr, mu, kk, cosmo_funcs, zz, n=n, I2=True)
        if s_k2:  # IS
            tot_arr += I1_sum(arr_dict, s2_arr, mu, kk, cosmo_funcs, zz, I2=False)

    if s_k1:
        if int_k2:  # SI
            tot_arr += s1_sum(s_k1, r2_arr, mu, kk, cosmo_funcs, zz, n=n, I2=True)
        if s_k2:  # SS
            tot_arr += s1_sum(s_k1, s2_arr, mu, kk, cosmo_funcs, zz, I2=False)

    return tot_arr


def get_multipole(
    kernel1, kernel2, l, cosmo_funcs, kk, zz, sigma=None, n=32, n_mu=256, nr=2000, deg=8, delta=0.1, GL=False
):
    if GL:
        mu, weights = np.polynomial.legendre.leggauss(n_mu)  # legendre gauss - get nodes and weights for given n
    else:
        N_fine = n_mu // 2  # High density for the central region
        N_coarse = n_mu // 4  # Low density for the wings

        # 1. Generate Non-Uniform Grid (Vectorized)
        mu = np.unique(
            np.concatenate(
                [np.linspace(-1, -delta, N_coarse), np.linspace(-delta, delta, N_fine), np.linspace(delta, 1, N_coarse)]
            )
        )

    arr = get_mu(mu, kernel1, kernel2, cosmo_funcs, kk[:, np.newaxis], zz, n=n, deg=deg, nr=nr)

    # get legendre
    leg = eval_legendre(l, mu)

    if sigma is None:  # no FOG
        dfog_val = 1
    else:
        dfog_val = np.exp(-(1 / 2) * ((kk * mu) ** 2) * sigma**2)  # exponential damping

    if GL:
        return ((2 * l + 1) / 2) * np.sum(weights * leg * dfog_val * arr, axis=-1)
    return ((2 * l + 1) / 2) * utils.trapezoid(leg * dfog_val * arr, x=mu, axis=-1)
