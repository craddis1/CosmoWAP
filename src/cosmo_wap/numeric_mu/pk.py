import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from . import accel
from .integration import compute_robust_integral
from .kernels import K1, IntK1, eval_terms, survey_scalars, term_weight
from .utils import merge_terms, split_kernels


def _int_K1_basis(kernel, cosmo_funcs, zz, deg=8, n_p=2000):
    """Survey-independent basis for the first-order integrated kernel LOS integrals.

    Returns (entries, spline): entries is a list of (mu_pow, q_pow, frozen_weights) and
    spline is a vector-valued complex CubicSpline with one component per entry. Everything
    here depends only on the cosmology - the survey enters the kernels only through the
    scalar weight dicts, applied per term in accel.kernel_sum - so the result is cached on
    cosmo_funcs: a new cosmology is a new instance, and the in-place gamma override goes
    through compute_derivs_cosmo which resets the cache."""
    key = (tuple(kernel), float(zz), deg, n_p)
    cache = cosmo_funcs.__dict__.setdefault("_int_K1_basis_cache", {})
    if key in cache:
        return cache[key]

    p_arr = np.concatenate((-np.logspace(-6, 4, n_p)[::-1], np.logspace(-6, 4, n_p)))
    d = cosmo_funcs.comoving_dist(zz)
    r1 = np.linspace(1e-3, d, 1000)

    terms = []
    for kern in kernel:
        terms += getattr(IntK1, kern)(r1, cosmo_funcs, zz=zz, ti=0)

    # line-of-sight integrals - all kernel terms in one call, stacked into one complex spline
    merged = merge_terms(terms)
    entries = list(merged)
    I_arrs, _ = compute_robust_integral(d, p_arr, r1, np.stack(list(merged.values()), axis=-1), deg=deg)

    cache[key] = (entries, CubicSpline(p_arr, I_arrs))
    return cache[key]


def get_int_K1(kernel, cosmo_funcs, zz, deg=8, n_p=2000):
    """First-order integrated kernel line-of-sight integrals I_ij(p).

    Returns (powers, weights, spline): powers[m] is the (mu_power, q_power) of kernel term m,
    weights[m] its survey scalar, and spline is a single complex vector-valued CubicSpline
    over p, so spline(p)[..., m] is I(p) for term m. Stacking all kernel terms into one
    spline means the (expensive) breakpoint search in I1_sum is done once for every term.

    The expensive part (the LOS integrals, cosmology-only) comes from the cached
    _int_K1_basis; all that is left here is evaluating the weight dicts against the current
    survey scalars (Q, be), so those can change every call (e.g. sampled in an MCMC) without
    redoing the integrals. They multiply their own term in accel.kernel_sum rather than being
    folded into the spline coefficients: that array runs to hundreds of kB while a call only
    samples a few thousand points off it, so folding cost more than the dedup it bought.

    Note: I(p) oscillates with period ~2*pi/d so the log-spaced p grid only resolves it for
    |p| below a few tenths - fine because P(q) ~ q^-3 suppresses larger |p| = |q mu| in the
    sums (so n_p matters more than deg for accuracy). Beyond |p| = 1e4 (only reached for
    n >~ 512 line-of-sight nodes) the spline silently extrapolates - also P(q) suppressed."""
    entries, spline = _int_K1_basis(kernel, cosmo_funcs, zz, deg=deg, n_p=n_p)

    scal = survey_scalars(cosmo_funcs, zz, ti=0)
    powers = [(i, j) for i, j, _ in entries]
    weights = np.array([term_weight(dict(wt), scal) for _, _, wt in entries])

    return powers, weights, spline


def get_int_r1(p, kernel, cosmo_funcs, zz, deg=8, n_p=2000):
    """Get values - dont interpolate as very oscillatory - so just call as is very quick"""
    p_arr = p.flatten()
    d = cosmo_funcs.comoving_dist(zz)
    r1 = np.linspace(1e-3, d - 1e-3, n_p)

    # could precompute these few lines
    terms = []
    for kern in kernel:
        terms += getattr(IntK1, kern)(r1, cosmo_funcs, zz=zz, ti=0)

    # now get integral values in p for all kernel terms - survey weights applied after
    scal = survey_scalars(cosmo_funcs, zz, ti=0)
    merged = merge_terms(terms)
    I_arrs, _ = compute_robust_integral(d, p_arr, r1, np.stack(list(merged.values()), axis=-1), deg=deg)

    arr_dict = {}
    for m, (i, j, wt) in enumerate(merged):
        I_arr = term_weight(dict(wt), scal) * I_arrs[:, m].reshape(p.shape)
        arr_dict[(i, j)] = arr_dict[(i, j)] + I_arr if (i, j) in arr_dict else I_arr

    return arr_dict


def get_int_K2(kernel, r2, cosmo_funcs, zz, mu, kk):
    """Get kernel 2 array"""
    d = cosmo_funcs.comoving_dist(zz)
    mu, kk = utils.enable_broadcasting(mu, kk, n=1)

    G = r2 / d
    qq = kk / G
    k2_arr = np.zeros(np.broadcast_shapes(mu.shape, qq.shape), dtype=np.complex128)

    for kern in kernel:
        k2_arr += eval_terms(getattr(IntK1, kern)(r2, cosmo_funcs, zz, ti=1), -mu, qq, cosmo_funcs, zz, ti=1)

    return k2_arr


def I1_sum(int_K1, r2_arr, mu, kk, cosmo_funcs, zz, r2=None, weights=None, I2=False):
    """Do first integral sum - if I2 is True then we are doing II, otherwise IS
    r2 and weights are the Gauss-Legendre nodes and weights from get_mu (II only)"""
    powers, term_weights, spline = int_K1  # weights here are the GL ones - see the signature
    baseint = BaseInt(cosmo_funcs)
    d = cosmo_funcs.comoving_dist(zz)
    if I2:  # II
        mu, kk = utils.enable_broadcasting(mu, kk, n=1)
        G = r2 / d
        qq = kk / G
    else:
        # only IS
        qq = kk

    # sum the kernel terms - only I(q*mu) and qq**j * mu**i (the mu from the first order
    # field) vary between them, so the common factors below are applied once after
    kernel_sum = accel.kernel_sum(spline, powers, term_weights, qq, mu)

    if I2:
        # phase is independent of r2 so sits outside the integral
        phase = np.exp(-1j * kk[..., 0] * mu[..., 0] * d)
        return ((d) / 2.0) * phase * np.sum(G ** (-3) * baseint.pk(qq, zz) * weights * r2_arr * kernel_sum, axis=-1)

    return baseint.pk(qq, zz) * np.exp(-1j * kk * mu * d) * r2_arr * kernel_sum


def s1_sum(s_k1, r2_arr, mu, kk, cosmo_funcs, zz, r2=None, I2=False, **kwargs):
    """Now for case where S if first field
    r2 are the Gauss-Legendre nodes from get_mu (SI only)"""
    baseint = BaseInt(cosmo_funcs)
    if I2:  # SI
        # so this integral can be a little oscillatory near the source - we use filon integration
        d = cosmo_funcs.comoving_dist(zz)

        # u = d / r = 1 / G
        u = d / r2  # Note: u_grid now goes from 1.0 to (d/r_min)

        # (Recall: dr = -d/u^2 du, and G^-3 = u^3, so net is d*u)
        Jac = d * u  # ignore the negative as we also switch integration directions

        mu, kk = utils.enable_broadcasting(mu, kk, n=1)
        qq = kk * u

        # Full Integrand
        s1_arr = get_K(s_k1, cosmo_funcs, zz, mu, qq, ti=0, **kwargs)
        integrand = Jac * baseint.pk(qq, zz) * r2_arr * s1_arr

        # u_grid is now strictly increasing, so Filon works perfectly.
        tot_arr = np.exp(-1j * kk[..., 0] * mu[..., 0] * d) * accel.filon_integrate(
            u[::-1], kk, mu, integrand[..., ::-1], d
        )

    else:
        # only SS
        s1_arr = get_K(s_k1, cosmo_funcs, zz, mu, kk, ti=0, **kwargs)
        tot_arr = baseint.pk(kk, zz) * r2_arr * s1_arr

    return tot_arr


def get_K(kernels, cosmo_funcs, zz, mu, kk, ti=0, **kwargs):
    tot_arr = np.zeros(np.broadcast_shapes(kk.shape, mu.shape), dtype=np.complex128)
    for kern in kernels:
        func = getattr(K1, kern)
        tot_arr += func(cosmo_funcs, zz, mu, kk, ti=ti, **kwargs)
    return tot_arr


def get_mu(mu, kernels1, kernels2, cosmo_funcs, kk, zz, n=8, deg=8, n_p=2000, **kwargs):
    """Collect power spectrum contribution P(k,mu) = <K1(mu,k) K2(-mu,k)>.
    Kernels are split into integrated (I) and source (S) parts and each combination
    (II, IS, SI, SS) is summed. n is the number of Gauss-Legendre nodes for the
    line-of-sight integrals (II and SI).
    Note: for the integrated terms mu=0 exposes the near-source divergence (no
    oscillatory suppression) so exactly mu=0 does not converge with n."""

    d = cosmo_funcs.comoving_dist(zz)
    nodes, weights = utils.leggauss(n)  # legendre gauss - get nodes and weights for given n
    r2 = (d) * (nodes + 1) / 2.0  # sample r range [0,d] - shared by II (I1_sum), SI (s1_sum) and get_int_K2

    # lets split kernels into integrated and not!
    int_k1, s_k1 = split_kernels(kernels1)
    int_k2, s_k2 = split_kernels(kernels2)

    # precompute ------------------------------------------------------------------------
    if s_k2:
        s2_arr = get_K(s_k2, cosmo_funcs, zz, -mu, kk, ti=1, **kwargs)
    if int_k1:
        int_K1 = get_int_K1(int_k1, cosmo_funcs, zz, deg=deg, n_p=n_p)  # (powers, weights, spline)
    if int_k2:
        r2_arr = get_int_K2(int_k2, r2, cosmo_funcs, zz, mu, kk)  # is array

    # sum stuff -----------------------------------------------------------------------------
    tot_arr = np.zeros(np.broadcast_shapes(kk.shape, mu.shape), dtype=np.complex128)  # shape (mu,kk)
    if int_k1:
        if int_k2:  # II
            tot_arr += I1_sum(int_K1, r2_arr, mu, kk, cosmo_funcs, zz, r2=r2, weights=weights, I2=True)
        if s_k2:  # IS
            tot_arr += I1_sum(int_K1, s2_arr, mu, kk, cosmo_funcs, zz, I2=False)

    if s_k1:
        if int_k2:  # SI
            tot_arr += s1_sum(s_k1, r2_arr, mu, kk, cosmo_funcs, zz, r2=r2, I2=True, **kwargs)
        if s_k2:  # SS
            tot_arr += s1_sum(s_k1, s2_arr, mu, kk, cosmo_funcs, zz, I2=False, **kwargs)

    return tot_arr


def get_mu_sym(mu, kernels1, kernels2, cosmo_funcs, kk, zz, **kwargs):
    """get_mu exploiting P(k,-mu) = P(k,mu)* (correlations are real in configuration space):
    for a symmetric mu grid only mu >= 0 is computed and the negative half is mirrored."""
    mu = np.asarray(mu)
    if mu.ndim != 1 or not np.allclose(mu, -mu[::-1]):  # need a symmetric 1D grid to mirror
        return get_mu(mu, kernels1, kernels2, cosmo_funcs, kk, zz, **kwargs)

    M = len(mu)
    half = (M + 1) // 2  # upper half - includes mu=0 if M is odd
    arr_hi = get_mu(mu[M - half :], kernels1, kernels2, cosmo_funcs, kk, zz, **kwargs)

    arr = np.empty((*arr_hi.shape[:-1], M), dtype=arr_hi.dtype)
    arr[..., M - half :] = arr_hi
    arr[..., : M - half] = arr_hi[..., ::-1][..., : M - half].conj()  # mu < 0 from the conjugate
    return arr


def get_mu_grid(n_mu, delta=0.1, GL=True):
    """mu nodes (and weights for GL) used to project P(k,mu) onto multipoles.

    GL is composite Gauss-Legendre over the panels [-1,-delta], [-delta,0], [0,delta], [delta,1],
    n_mu//4 nodes each. A single GL rule over [-1,1] cannot resolve the narrow feature the
    integrated kernels leave at mu=0, but each panel is smooth so each converges spectrally -
    n_mu=48 is more accurate than the trapezoid grid below at n_mu=256, for a quarter of the
    nodes. The panels are mirror images so get_mu_sym still halves the work, and no node lands
    on mu=0 itself (where the integrated terms do not converge with n - see get_mu)."""
    if GL:
        edges = [-1.0, -delta, 0.0, delta, 1.0]
        panels = list(zip(edges[:-1], edges[1:]))
        nodes, wts = utils.leggauss(max(n_mu // 4, 2))  # same order in each panel
        # leggauss nodes are ascending so the concatenation is too - and stays symmetric about 0
        mu = np.concatenate([(b - a) / 2 * nodes + (a + b) / 2 for a, b in panels])
        return mu, np.concatenate([(b - a) / 2 * wts for a, b in panels])
    # non-uniform grid: dense in the central region, coarse in the wings
    N_fine = n_mu // 2
    N_coarse = n_mu // 4
    mu = np.unique(
        np.concatenate(
            [np.linspace(-1, -delta, N_coarse), np.linspace(-delta, delta, N_fine), np.linspace(delta, 1, N_coarse)]
        )
    )
    return mu, None


def project_multipole(arr, mu, weights, l, kk, sigma=None, GL=True):
    """Project a precomputed P(k,mu) array onto the lth multipole."""
    leg = eval_legendre(l, mu)

    if sigma is None:  # no FOG
        dfog_val = 1
    else:
        dfog_val = np.exp(-(1 / 2) * ((kk[:, None] * mu) ** 2) * sigma**2)  # exponential damping (k,mu)

    if GL:
        return ((2 * l + 1) / 2) * np.sum(weights * leg * dfog_val * arr, axis=-1)
    return ((2 * l + 1) / 2) * utils.trapezoid(leg * dfog_val * arr, x=mu, axis=-1)


def get_multipole(
    kernel1, kernel2, l, cosmo_funcs, kk, zz, sigma=None, n=8, n_mu=48, deg=8, n_p=2000, delta=0.1, GL=True, **kwargs
):
    mu, weights = get_mu_grid(n_mu, delta, GL)
    arr = get_mu_sym(mu, kernel1, kernel2, cosmo_funcs, kk[:, np.newaxis], zz, n=n, deg=deg, n_p=n_p, **kwargs)
    return project_multipole(arr, mu, weights, l, kk, sigma, GL)


def get_multipoles(
    kernel1, kernel2, ln, cosmo_funcs, kk, zz, sigma=None, n=8, n_mu=48, deg=8, n_p=2000, delta=0.1, GL=True, **kwargs
):
    """Like get_multipole but for a list of multipoles - computes P(k,mu) once and projects each l.
    Returns array of shape (len(ln), len(kk))."""
    mu, weights = get_mu_grid(n_mu, delta, GL)
    arr = get_mu_sym(mu, kernel1, kernel2, cosmo_funcs, kk[:, np.newaxis], zz, n=n, deg=deg, n_p=n_p, **kwargs)
    return np.array([project_multipole(arr, mu, weights, l, kk, sigma, GL) for l in ln])
