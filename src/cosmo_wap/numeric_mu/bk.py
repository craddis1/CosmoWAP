"""Numeric-mu bispectrum - B(k1,k2,k3,mu,phi) built from the kernels and projected onto the Ylm multipoles.

Plane-parallel, as numeric_mu.pk. mu = k1.x and phi is the azimuth of k2 about k1, as in the analytic
(mu, phi) expressions (e.g. bk.Bk_0), so the multipoles are the same int dOmega conj(Ylm) B."""

import itertools

import numpy as np
from scipy.special import sph_harm_y

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from .kernels import K1, K2, M_tail

# kernels that carry fNL, and the primordial bispectrum shapes each brings along
PNG_SHAPES = {"Loc": ["Loc"], "Eq": ["Eq"], "Orth": ["Orth"], "PNG": ["Loc", "Eq", "Orth"]}


def check_kernels(kernels):
    """Kernel names for one field - each needs its first and second order part (K1 and K2)"""
    if not isinstance(kernels, list):
        kernels = [kernels]
    missing = [kern for kern in kernels if not hasattr(K2, kern)]
    if missing:
        raise NotImplementedError(f"no second order kernel for {missing} in the numeric bispectrum")
    return kernels


def los_cosines(mu, phi, k1, k2, k3, theta):
    """LOS cosines of the three sides - k3 = -(k1 + k2)"""
    mu2 = mu * np.cos(theta) + np.sqrt(1 - mu**2) * np.sin(theta) * np.cos(phi)
    mu3 = -(k1 * mu + k2 * mu2) / k3
    return mu, mu2, mu3


def get_Z1(kernels, cosmo_funcs, zz, mu, kk, ti=0, **kwargs):
    return sum(getattr(K1, kern)(cosmo_funcs, zz, mu, kk, ti=ti, **kwargs) for kern in kernels)


def get_Z2(kernels, cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
    return sum(getattr(K2, kern)(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=ti, **kwargs) for kern in kernels)


def get_B_prim(shapes, cosmo_funcs, k1, k2, k3, zz, fNL=1, **kwargs):
    """M(k1)M(k2)M(k3) B_Phi(k1,k2,k3) / D**3 summed over shapes, each with its own fNL - the primordial
    bispectrum templates as in MathWAP. D**3 because the three first order kernels it meets carry D each."""
    P1, P2, P3 = cosmo_funcs.Pk_phi(k1), cosmo_funcs.Pk_phi(k2), cosmo_funcs.Pk_phi(k3)
    pairs = P1 * P2 + P1 * P3 + P2 * P3
    cube = (P1 * P2 * P3) ** (2 / 3)
    mixed = sum(a ** (1 / 3) * b ** (2 / 3) * c for a, b, c in itertools.permutations((P1, P2, P3)))
    templates = {
        "Loc": 2 * pairs,
        "Eq": 6 * (-pairs - 2 * cube + mixed),
        "Orth": 6 * (-3 * pairs - 8 * cube + 3 * mixed),
    }

    B_Phi = 0
    for shape in shapes:
        shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # as K1._PNG
        B_Phi = B_Phi + (fNL if shape_fNL is None else shape_fNL) * templates[shape]
    M = M_tail(cosmo_funcs, k1, zz) * M_tail(cosmo_funcs, k2, zz) * M_tail(cosmo_funcs, k3, zz)
    return M * B_Phi / cosmo_funcs.D(zz) ** 3


def get_mu_phi(mu, phi, kernels1, kernels2, kernels3, cosmo_funcs, k1, k2, k3, theta, zz, **kwargs):
    """B(k1,k2,k3,mu,phi) = 2*Z1(k1)Z1(k2)Z2(-k1,-k2)P(k1)P(k2) + 2 perms - 2407.00168 eq (2.22),
    plus Z1(k1)Z1(k2)Z1(k3) M1 M2 M3 B_Phi for each PNG shape named, kept linear in fNL.

    Field i sits at k_i with kernelsi and tracer ti = i-1 (as bk_mt), so each field's Z1 and Z2 are
    computed once and shared by the perms. mu and phi are 1D and span the last two axes, so k1, k2,
    k3 and theta need two trailing axes."""
    mu1, mu2, mu3 = los_cosines(mu[:, None], phi[None, :], k1, k2, k3, theta)
    cos12 = np.cos(theta)
    cos13 = -(k1 + k2 * cos12) / k3
    cos23 = -(k2 + k1 * cos12) / k3

    kernels = [check_kernels(kern) for kern in (kernels1, kernels2, kernels3)]
    gauss = [[kern for kern in kerns if kern not in PNG_SHAPES] for kerns in kernels]
    png = [[kern for kern in kerns if kern in PNG_SHAPES] for kerns in kernels]
    args = (cosmo_funcs, zz)

    # second order field sourced by the other two sides - contracting with the first order fields at k_a, k_b
    # puts Z2 at (-k_a, -k_b), as K2 at -mu in numeric_mu.pk. Flips the LOS cosines, not the cosine between them.
    # 2011.13660 eq (3.13) and the analytic GR1/GR2 (from MathWAP) use (k_a, k_b), which flips beta14-beta18
    first = [(mu1, k1), (mu2, k2), (mu3, k3)]
    second = [(-mu2, -mu3, k2, k3, cos23), (-mu1, -mu3, k1, k3, cos13), (-mu1, -mu2, k1, k2, cos12)]

    def Z(kerns):
        """(Z1, Z2) of each field"""
        return (
            [get_Z1(kerns[i], *args, *first[i], ti=i, **kwargs) for i in range(3)],
            [get_Z2(kerns[i], *args, *second[i], ti=i, **kwargs) for i in range(3)],
        )

    baseint = BaseInt(cosmo_funcs)
    Pk1, Pk2, Pk3 = baseint.pk(k1, zz), baseint.pk(k2, zz), baseint.pk(k3, zz)

    def tree(Z1, Z2):
        return 2 * (
            Z1[0] * Z1[1] * Z2[2] * Pk1 * Pk2 + Z1[0] * Z2[1] * Z1[2] * Pk1 * Pk3 + Z2[0] * Z1[1] * Z1[2] * Pk2 * Pk3
        )

    Z1, Z2 = Z(gauss)
    B = tree(Z1, Z2)
    if not any(png):
        return B

    # each tree term has every field once, as Z1 or Z2, so its part linear in fNL is the sum over fields of the
    # term with that one field non-Gaussian
    Z1_ng, Z2_ng = Z(png)
    for i in range(3):
        B = B + tree(
            [Z1_ng[j] if j == i else Z1[j] for j in range(3)], [Z2_ng[j] if j == i else Z2[j] for j in range(3)]
        )

    shapes = sorted({shape for kerns in png for kern in kerns for shape in PNG_SHAPES[kern]})
    return B + Z1[0] * Z1[1] * Z1[2] * get_B_prim(shapes, cosmo_funcs, k1, k2, k3, zz, **kwargs)


def get_mu_phi_sym(mu, phi, kernels1, kernels2, kernels3, cosmo_funcs, k1, k2, k3, theta, zz, **kwargs):
    """get_mu_phi exploiting B(-mu, phi) = B(mu, pi - phi)* - flipping x flips every LOS cosine, and
    phi -> pi - phi with mu -> -mu does exactly that. For a mu grid symmetric about 0 and a phi grid
    symmetric about pi/2 only mu >= 0 is computed and the negative half is mirrored - see get_mu_sym."""
    if not (np.allclose(mu, -mu[::-1]) and np.allclose(phi, np.pi - phi[::-1])):
        return get_mu_phi(mu, phi, kernels1, kernels2, kernels3, cosmo_funcs, k1, k2, k3, theta, zz, **kwargs)

    M = len(mu)
    half = (M + 1) // 2  # upper half - includes mu=0 if M is odd
    arr_hi = get_mu_phi(mu[M - half :], phi, kernels1, kernels2, kernels3, cosmo_funcs, k1, k2, k3, theta, zz, **kwargs)

    arr = np.empty((*arr_hi.shape[:-2], M, len(phi)), dtype=np.complex128)
    arr[..., M - half :, :] = arr_hi
    arr[..., : M - half, :] = arr_hi[..., ::-1, ::-1][..., : M - half, :].conj()
    return arr


def get_mu_phi_grid(n_mu=16, n_phi=16):
    """(mu, phi) nodes and the product weights for the dOmega quadrature.

    Gauss-Legendre in mu on [-1, 1]. B depends on phi only through cos(phi) (los_cosines) so phi only
    spans [0, pi] - the other half is its mirror, see project_multipole. It is periodic, so phi uses the
    midpoint rule, which is exact for cos(j*phi) with j < 2*n_phi where Gauss-Legendre is not."""
    mu, w_mu = utils.leggauss(n_mu)
    phi = np.pi * (np.arange(n_phi) + 0.5) / n_phi
    return mu, phi, w_mu[:, None] * (np.pi / n_phi)


def project_multipole(arr, mu, phi, weights, l, m, k1, k2, k3, theta, sigma=None):
    """Project a precomputed B(mu, phi) onto the (l, m) multipole: int dOmega conj(Ylm) B.
    phi in [pi, 2pi] mirrors phi in [0, pi], where conj(Ylm) has exp(+i*m*phi) instead - together 2*Re(Ylm)."""
    ylm = np.real(sph_harm_y(l, m, np.arccos(mu)[:, None], phi[None, :]))

    if sigma is None:  # no FOG
        dfog_val = 1
    else:
        mu1, mu2, mu3 = los_cosines(mu[:, None], phi[None, :], k1, k2, k3, theta)
        dfog_val = np.exp(-(1 / 2) * ((k1 * mu1) ** 2 + (k2 * mu2) ** 2 + (k3 * mu3) ** 2) * sigma**2)

    return 2 * np.sum(weights * ylm * dfog_val * arr, axis=(-2, -1))


def get_multipoles(
    kernels1,
    kernels2,
    kernels3,
    lm,
    cosmo_funcs,
    k1,
    k2,
    k3=None,
    theta=None,
    zz=0,
    sigma=None,
    n_mu=16,
    n_phi=16,
    **kwargs,
):
    """(l, m) multipoles for a list of (l, m) pairs - computes B(mu, phi) once and projects each.
    Returns array of shape (len(lm), *triangle shape)."""
    k3, theta = utils.get_theta_k3(k1, k2, k3, theta)
    k1, k2, k3, theta = utils.enable_broadcasting(k1, k2, k3, theta)

    mu, phi, weights = get_mu_phi_grid(n_mu, n_phi)
    arr = get_mu_phi_sym(mu, phi, kernels1, kernels2, kernels3, cosmo_funcs, k1, k2, k3, theta, zz, **kwargs)
    return np.array([project_multipole(arr, mu, phi, weights, l, m, k1, k2, k3, theta, sigma) for l, m in lm])


def pair_response(kernels, cf_i, cf_j, zz, mu_i, mu_j, k_i, k_j, cos_ij, k_ij=None, **kwargs):
    """R_ij = h(i, j) + h(j, i), h(i, j) = Z2_i(-k_j, k_i + k_j) Z1_j(k_j) P(k_j): response of the pair to the mode
    k_i + k_j they exchange. With the third field's Z1 P it is the rest of the tree-level bispectrum -
    B_ijc = 2 Z1_c(k_c) P(k_c) R_ij + 2 Z1_i Z1_j Z2_c(-k_i, -k_j) P_i P_j, k_c = -(k_i + k_j).
    cf_i and cf_j are single-tracer views. Where k_i + k_j = 0 it is not defined - see get_T_exchange. Near there
    the cosine loses |k_i + k_j|, so pass it in if known."""
    if k_ij is None:
        k_ij = np.sqrt(np.maximum(k_i**2 + k_j**2 + 2 * k_i * k_j * cos_ij, 0))
    mu_ij = (k_i * mu_i + k_j * mu_j) / k_ij
    pk = BaseInt(cf_i).pk

    def h(cf_a, mu_a, k_a, cf_b, mu_b, k_b):
        cos = -(k_a * k_b * cos_ij + k_b**2) / (k_b * k_ij)  # between -k_b and k_a + k_b
        Z2 = get_Z2(kernels, cf_a, zz, -mu_b, mu_ij, k_b, k_ij, cos, **kwargs)
        return Z2 * get_Z1(kernels, cf_b, zz, mu_b, k_b, **kwargs) * pk(k_b, zz)

    return h(cf_i, mu_i, k_i, cf_j, mu_j, k_j) + h(cf_j, mu_j, k_j, cf_i, mu_i, k_i)


def get_T_exchange(kernels, cfs, zz, vecs, channels="stu", **kwargs):
    """Tree-level exchange part (T2211) of the trispectrum of four fields at the 3D wavevectors vecs (4, ..., 3),
    summing to zero with the LOS along z - field i with the single-tracer view cfs[i]. Omits the Z3 (T3111) terms.

    T = 4 sum over channels (ij|kl) of P(|k_i + k_j|) R_ij R_kl - see pair_response. s, t and u are (01|23),
    (02|13) and (03|12). A channel whose exchanged mode is exactly zero gives nothing, as P(0) = 0 - its 1/q
    terms cancel in R, so it goes to zero continuously."""
    k = np.linalg.norm(vecs, axis=-1)
    mu = vecs[..., 2] / k
    pk = BaseInt(cfs[0]).pk

    def R(i, j, k_ij):
        cos = np.sum(vecs[i] * vecs[j], axis=-1) / (k[i] * k[j])
        return pair_response(kernels, cfs[i], cfs[j], zz, mu[i], mu[j], k[i], k[j], cos, k_ij=k_ij, **kwargs)

    pairs = {"s": ((0, 1), (2, 3)), "t": ((0, 2), (1, 3)), "u": ((0, 3), (1, 2))}
    T = 0
    for (i, j), (a, b) in (pairs[c] for c in channels):
        k_ij = np.linalg.norm(vecs[i] + vecs[j], axis=-1)
        exchanged = k_ij > 1e-10 * np.minimum(k[i], k[j])
        k_ij = np.where(exchanged, k_ij, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            T = T + np.where(exchanged, 4 * pk(k_ij, zz) * R(i, j, k_ij) * R(a, b, k_ij), 0)
    return T
