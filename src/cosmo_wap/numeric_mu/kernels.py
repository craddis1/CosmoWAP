import numpy as np

from cosmo_wap.lib import utils


def M_tail(cosmo_funcs, k1, zz):
    """M(k,z) using the k**-3 power law past K_MAX, as BaseInt.pk does for P(k).

    ClassWAP.M reads the raw P(k) spline, which extrapolates negative past K_MAX and gives
    nan through the sqrt. Only the kernels hit this: in the line-of-sight branches the source
    kernel is evaluated at a rescaled q = k*d/r, which runs well past K_MAX."""
    K_MAX = cosmo_funcs.K_MAX
    Pk = np.where(k1 > K_MAX, cosmo_funcs.Pk(K_MAX) * utils.cube(K_MAX / k1), cosmo_funcs.Pk(np.minimum(k1, K_MAX)))
    return np.sqrt(cosmo_funcs.D(zz) ** 2 * Pk / cosmo_funcs.Pk_phi(k1))


class Unpack:
    @staticmethod
    def common(cosmo_funcs, zz, k1, ti=0):  # kaiser
        """get base things unpacked"""

        f = cosmo_funcs.f(zz)
        D1 = cosmo_funcs.D(zz)
        b1 = cosmo_funcs.survey[ti].b_1(zz)

        return D1, f, b1

    @staticmethod
    def get_int_params(cosmo_funcs, zz, ti=0):
        """Get Source quatities for integrated power spectra"""
        d = cosmo_funcs.comoving_dist(zz)
        H = cosmo_funcs.H_c(zz)
        Hp = (
            -(1 + zz) * H * cosmo_funcs.dH_c(zz)
        )  # dH_dt - deriv wrt to conformal time! - equivalently: (1-(3/2)*cosmo_funcs.Om_m(zz))*H**2
        # OM = cosmo_funcs.Om_m(zz)
        Qm = cosmo_funcs.survey[ti].Q(zz)
        be = cosmo_funcs.survey[ti].be(zz)

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


# store source (non-integrated) kernels
class K1:
    # k-scaling of the effective fNL for each PNG shape - as in pk/PNG.py
    PNG_ALPHA = {"Loc": 0, "Orth": 1, "Eq": 2}

    @staticmethod
    def N(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):  # kaiser
        """D1*(b1 + f*mu**2)"""
        # unpack all necessary terms
        D1, f, b1 = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        return D1 * (b1 + f * mu**2)

    @staticmethod
    def LP(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):  # local projection effects # GR1 and GR2
        """D1*(1j*mu*gr1/k1 + gr2/k1**2)"""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        gr1, gr2 = cosmo_funcs.get_beta_funcs(zz, ti=ti)[:2]
        return D1 * (1j * mu * gr1 / k1 + gr2 / k1**2)

    @staticmethod
    def PNG(cosmo_funcs, zz, mu, k1, ti=0, fNL=1, shape="Loc", **kwargs):  # scale-dependent bias
        """D1*fNL*k1**alpha*b_01/M(k1) - see 2511.09466 eq (2.21).
        shape picks the PNG bias and its k-scaling; b_01 carries no fNL, as in pk/PNG.py"""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # per-shape override, as the analytic classes do
        if shape_fNL is not None:
            fNL = shape_fNL
        b01, _ = cosmo_funcs.get_PNG_bias(zz, ti, shape)
        return D1 * fNL * k1 ** K1.PNG_ALPHA[shape] * b01 / M_tail(cosmo_funcs, k1, zz)


# store integrated kernels as term lists - each formula lives in one place for both the
# line-of-sight integrals (II/IS) and the explicit evaluation at the second field (SI/II)
class IntK1:
    """First-order integrated kernels: each returns a list of (mu_pow, q_pow, radial_arr, weights)
    terms so the full kernel is sum(weight * mu**i * q**j * arr) - see eval_terms.

    radial_arr must be survey-independent (cosmology only): the survey dependence lives
    entirely in `weights`, a dict of scalar coefficients on survey scalars at the source
    redshift - {1: c0, 'Q': cQ, 'be': cbe} means c0 + cQ*Q(zz) + cbe*be(zz) (tuple keys
    multiply, e.g. ('Q','be')). This split lets the expensive line-of-sight integrals be
    cached per cosmology while Q/be amplitudes vary freely (e.g. sampled in an MCMC) -
    the weights are applied after the integral, which is linear in radial_arr."""

    @staticmethod
    def _params(r, cosmo_funcs, zz, ti, src, intg):
        """The two parameter sets every kernel below opens with, unless already supplied.

        Both are spline evaluations and together are most of a kernel's cost (~31 us of L's
        ~37 us), so IntK1.I evaluates them once and hands them to L, TD and ISW instead of
        letting each redo them. Passing them stays optional: the kernels are also reached one
        at a time, by name, from numeric_mu.pk."""
        if src is None:
            src = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)  # source integrated params
        if intg is None:
            intg = Unpack.get_integrand_params(cosmo_funcs, r)  # arrays in shape (xd)
        return src, intg

    @staticmethod
    def L(r, cosmo_funcs, zz=0, ti=0, src=None, intg=None):  # lensing
        """3*D1_r*(Qm - 1)*OM_r*H_r**2*(d - r)*r/d * (1 - mu**2 + 2j*mu/(r*q))"""
        src, intg = IntK1._params(r, cosmo_funcs, zz, ti, src, intg)
        d, _, _, _, _ = src
        _, _, D1_r, H_r, OM_r = intg

        tmp_arr = 3 * D1_r * OM_r * H_r**2 * (d - r) * r / d  # [1-mu**2+2i mu/r*q] *
        wt = {1: -1.0, "Q": 1.0}  # (Qm - 1)

        return [(0, 0, tmp_arr, wt), (2, 0, -tmp_arr, wt), (1, -1, 2j * tmp_arr / r, wt)]

    @staticmethod
    def TD(r, cosmo_funcs, zz=0, ti=0, src=None, intg=None):  # time delay
        """6*D1_r*(Qm - 1)*OM_r*H_r**2/d * 1/q**2"""
        src, intg = IntK1._params(r, cosmo_funcs, zz, ti, src, intg)
        d, _, _, _, _ = src
        _, _, D1_r, H_r, OM_r = intg

        tmp_arr = 6 * D1_r * OM_r * H_r**2 / (d)  # k1**2 *
        wt = {1: -1.0, "Q": 1.0}  # (Qm - 1)

        return [(0, -2, tmp_arr, wt)]

    @staticmethod
    def ISW(r, cosmo_funcs, zz=0, ti=0, src=None, intg=None):  # integrated Sachs-Wolfe
        """3*D1_r*(be - 2*Qm + 2*(Qm - 1)/(d*H) - Hp/H**2)*OM_r*H_r**3*(f_r - 1) * 1/q**2"""
        src, intg = IntK1._params(r, cosmo_funcs, zz, ti, src, intg)
        d, H, Hp, _, _ = src
        _, f_r, D1_r, H_r, OM_r = intg

        tmp_arr = 3 * D1_r * OM_r * H_r**3 * (f_r - 1)  # k1**2 *
        wt = {1: -2 / (d * H) - Hp / H**2, "Q": -2 + 2 / (d * H), "be": 1.0}  # be - 2*Qm + 2*(Qm-1)/(d*H) - Hp/H**2

        return [(0, -2, tmp_arr, wt)]

    @staticmethod
    def I(r, cosmo_funcs, zz=0, ti=0, src=None, intg=None):
        """Combined (L+TD+ISW) integrated 1st order kernel"""
        src, intg = IntK1._params(r, cosmo_funcs, zz, ti, src, intg)  # shared by all three - tiny speed up by hey ho
        args = (r, cosmo_funcs, zz)
        kw = {"ti": ti, "src": src, "intg": intg}
        return IntK1.L(*args, **kw) + IntK1.TD(*args, **kw) + IntK1.ISW(*args, **kw)

    @staticmethod
    def kappa_g(r, cosmo_funcs, zz=0, ti=0, src=None, intg=None):
        """(3/2)*D1_r*OM_r*H_r**2*(d - r)*r/d * (1 - mu**2 + 2j*mu/(r*q))"""
        src, intg = IntK1._params(r, cosmo_funcs, zz, ti, src, intg)
        d, _, _, _, _ = src
        _, _, D1_r, H_r, OM_r = intg

        tmp_arr = (3 / 2) * D1_r * OM_r * H_r**2 * (d - r) * r / d  # [1-mu**2+2i mu/r*q] *

        return [(0, 0, tmp_arr, {1: 1.0}), (2, 0, -tmp_arr, {1: 1.0}), (1, -1, 2j * tmp_arr / r, {1: 1.0})]


def survey_scalars(cosmo_funcs, zz, ti=0):
    """The survey scalars (at the source redshift) that integrated-kernel weights may reference."""
    survey = cosmo_funcs.survey[ti]
    return {"Q": survey.Q(zz), "be": survey.be(zz)}


def term_weight(weights, scalars):
    """Collapse a term's weight dict to a number given the survey scalars (tuple keys multiply)."""
    tot = 0
    for key, coeff in weights.items():
        if key == 1:
            tot = tot + coeff
        elif isinstance(key, tuple):
            val = coeff
            for name in key:
                val = val * scalars[name]
            tot = tot + val
        else:
            tot = tot + coeff * scalars[key]
    return tot


def eval_terms(terms, mu, qq, cosmo_funcs, zz, ti=0):
    """Evaluate an integrated kernel term list at explicit (mu, q)"""
    scal = survey_scalars(cosmo_funcs, zz, ti=ti)
    # only mu**i and qq**j vary between terms and most are trivial - cache the powers and
    # skip the zero ones, as I1_sum does with the same (i, j)
    q_pows, mu_pows = {}, {}
    tot = 0
    for i, j, arr, wt in terms:
        term = term_weight(wt, scal) * arr
        if j:
            if j not in q_pows:
                q_pows[j] = qq**j
            term = term * q_pows[j]
        if i:
            if i not in mu_pows:
                mu_pows[i] = mu**i
            term = term * mu_pows[i]
        tot = tot + term
    return tot
