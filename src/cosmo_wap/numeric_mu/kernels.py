import numpy as np

from cosmo_wap.lib import utils

from .integration import integrate_polynomial


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
    def second_order(cosmo_funcs, zz, ti=0):
        """K, C and the second order biases for the K2 kernels - K and C as in get_params"""
        K = C = 3 / 7  # from einstein-de-sitter
        if cosmo_funcs.growth2:
            cosmo_funcs.solve_second_order_KC()
            K, C = cosmo_funcs.K_intp(zz), cosmo_funcs.C_intp(zz)

        tracer = cosmo_funcs.survey[ti]
        return K, C, tracer.b_2(zz), tracer.g_2(zz)

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
    def _PNG(shapes, cosmo_funcs, zz, mu, k1, ti=0, fNL=1, **kwargs):  # scale-dependent bias
        """sum over shapes of D1*fNL*k1**alpha*b_01/M(k1) - see 2511.09466 eq (2.21).
        Each shape picks the PNG bias and its k-scaling; b_01 carries no fNL, as in pk/PNG.py.
        The shapes share M(k1), so summing them here evaluates it once - see PNG."""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        amp = 0
        for shape in shapes:
            shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # per-shape override, as the analytic classes do
            b01, _ = cosmo_funcs.get_PNG_bias(zz, ti, shape)
            amp = amp + (fNL if shape_fNL is None else shape_fNL) * k1 ** K1.PNG_ALPHA[shape] * b01
        return D1 * amp / M_tail(cosmo_funcs, k1, zz)

    # one kernel per shape, named as the analytic classes in pk/PNG.py - several can share a
    # kernel list, where they sum into the one kernel so the square keeps their cross terms
    @staticmethod
    def Loc(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):
        return K1._PNG(["Loc"], cosmo_funcs, zz, mu, k1, ti=ti, **kwargs)

    @staticmethod
    def Eq(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):
        return K1._PNG(["Eq"], cosmo_funcs, zz, mu, k1, ti=ti, **kwargs)

    @staticmethod
    def Orth(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):
        return K1._PNG(["Orth"], cosmo_funcs, zz, mu, k1, ti=ti, **kwargs)

    @staticmethod
    def PNG(cosmo_funcs, zz, mu, k1, ti=0, **kwargs):
        """Loc + Eq + Orth in one kernel - as listing all three, but M(k1) is evaluated once"""
        return K1._PNG(["Loc", "Eq", "Orth"], cosmo_funcs, zz, mu, k1, ti=ti, **kwargs)


def F2(q1, q2, cos12, K):
    """Second order density kernel - 2407.00168 eq (2.12a)"""
    return (1 + K) / 2 + cos12 / 2 * (q1 / q2 + q2 / q1) + (1 - K) / 2 * cos12**2


def G2(q1, q2, cos12, C):
    """Second order velocity kernel - 2407.00168 eq (2.12b)"""
    return C + cos12 / 2 * (q1 / q2 + q2 / q1) + (1 - C) * cos12**2


# store second order source kernels - Z2(q1, q2) for the pair of first order modes that source it, each with
# its LOS cosine mu_i = q_i.x, and cos12 = q1.q2/(q1*q2). Named as their first order parts in K1
class K2:
    @staticmethod
    def N(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):  # kaiser
        """D1**2*(b1*F2 + f*(q12.x)**2/q12**2*G2 + b2/2 + g2*S2
        + f*(q12.x)/2*(b1*(mu1/q1 + mu2/q2) + f*mu1*mu2*(q12.x)/(q1*q2))) - 2407.00168 eq (2.16)

        Evaluated as triangle-only coefficients times powers of p_i = q_i.x, so only the last line runs on
        the full (mu, phi) grid - as LP below."""
        # unpack all necessary terms
        D1, f, b1 = Unpack.common(cosmo_funcs, zz, q1, ti=ti)
        K, C, b2, g2 = Unpack.second_order(cosmo_funcs, zz, ti=ti)

        fG = f * G2(q1, q2, cos12, C) / (q1**2 + q2**2 + 2 * q1 * q2 * cos12)  # f G2 / q12**2
        c0 = b1 * F2(q1, q2, cos12, K) + b2 / 2 + g2 * (cos12**2 - 1)
        c1, c2 = fG + f * b1 / (2 * q1**2), fG + f * b1 / (2 * q2**2)  # p1**2, p2**2
        c12 = 2 * fG + f * b1 / 2 * (1 / q1**2 + 1 / q2**2)  # p1 p2
        c4 = f**2 / (2 * q1**2 * q2**2)  # p1 p2 (p1 + p2)**2

        p1, p2 = q1 * mu1, q2 * mu2
        p12 = p1 * p2
        return D1**2 * (c0 + c1 * p1**2 + c2 * p2**2 + p12 * (c12 + c4 * (p1 + p2) ** 2))

    @staticmethod
    def LP(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):  # local projection effects
        """D1**2 * 2011.13660 eq (3.24)/2 without beta1-beta5 (unless cosmo_funcs.full_betas), with q3 = q1 + q2 the mode it sources.

        (3.24) is for the field Delta = Delta^(1) + Delta^(2)/2, so it is twice Z2: its F2 and G2 are twice
        ours, which leaves beta6, beta7 and beta19 whole and halves the rest. 2407.00168 eq (2.18b) misses that
        1/2 and misprints the beta16 term. Like Z2 in (3.15) this is in the modes of the two first order fields,
        so the bispectrum evaluates it at (-k1, -k2) - see numeric_mu.bk.get_mu_phi.

        With p_i = q_i.x and p3 = p1 + p2 the real part is c0 + c12 p1 p2 + c2 (p1**2 + p2**2) and the imaginary
        part a1 p1 + a2 p2 + b17/2 (p1**3 + p2**3) + b18/2 p1 p2 (p1 + p2), all over q1**2 q2**2: the betas and
        F2, G2, E2 fold into coefficients that depend only on the triangle, and only the last line runs on the
        full (mu, phi) grid."""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, q1, ti=ti)
        K, C, _, _ = Unpack.second_order(cosmo_funcs, zz, ti=ti)
        _, _, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19 = cosmo_funcs.get_beta_funcs(zz, ti=ti)

        q3_sq = q1**2 + q2**2 + 2 * q1 * q2 * cos12
        qq = (q1 * q2) ** 2
        q1q2 = q1 * q2 * cos12  # q1.q2
        G2_ = G2(q1, q2, cos12, C)
        E2 = qq / q3_sq**2 * (3 + 2 * cos12 * (q2 / q1 + q1 / q2) + cos12**2)  # (3.25)

        norm = D1**2 / qq
        b3 = (b9 + E2 * b10) / 2  # of p3**2 = p1**2 + 2 p1 p2 + p2**2
        c0 = norm * (qq / q3_sq * (F2(q1, q2, cos12, K) * b6 + G2_ * b7) + (q1q2 * b11 + (q1**2 + q2**2) * b12) / 2)
        c12, c2 = norm * (b8 / 2 + 2 * b3), norm * (b3 + b13 / 2)
        a0 = qq / q3_sq * G2_ * b19  # of p3
        a1 = norm * (a0 + (q1**2 * b14 + q1q2 * b15 + q2**2 * b16) / 2)  # q1 q2 (mu1 q2 + mu2 q1) = p1 q2**2 + p2 q1**2
        a2 = norm * (a0 + (q2**2 * b14 + q1q2 * b15 + q1**2 * b16) / 2)
        a3, a21 = norm * b17 / 2, norm * b18 / 2

        if cosmo_funcs.full_betas:  # (H/k)^4 beta1 + E2 beta2 and i(H/k)^3 p3 (beta3 + beta4 + E2 beta5), p3 = p1 + p2
            beta1, beta2, beta3, beta4, beta5 = cosmo_funcs.get_beta_HO(zz, ti=ti)
            c0 = c0 + norm * (beta1 + E2 * beta2) / 2
            a_HO = norm * (beta3 + beta4 + E2 * beta5) / 2
            a1, a2 = a1 + a_HO, a2 + a_HO

        p1, p2 = q1 * mu1, q2 * mu2
        p12, sq = p1 * p2, p1**2 + p2**2
        # p1**3 + p2**3 = (p1 + p2) (p1**2 - p1 p2 + p2**2)
        return c0 + c12 * p12 + c2 * sq + 1j * (a1 * p1 + a2 * p2 + (p1 + p2) * (a3 * (sq - p12) + a21 * p12))

    @staticmethod
    def _PNG(shapes, cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, fNL=1, **kwargs):  # scale-dependent bias
        """sum over shapes of D1**2*fNL*[b01/2*(q1.q2)*(phi2/q1**2 + phi1/q2**2) + b11/2*(phi1 + phi2)
        + f*b01/2*(q12.x)*(mu1/q1*phi2 + mu2/q2*phi1)], phi_i = q_i**alpha/M(q_i) - MathWAP's KN2PNG.
        2407.00168 eq (D.2) prints the (q1.q2) term with a minus: the displacement -Psi.grad(phi) gives +."""
        # unpack all necessary terms
        D1, f, _ = Unpack.common(cosmo_funcs, zz, q1, ti=ti)
        M1, M2 = M_tail(cosmo_funcs, q1, zz), M_tail(cosmo_funcs, q2, zz)
        p12 = q1 * mu1 + q2 * mu2  # q12.x

        amp = 0
        for shape in shapes:
            shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # as K1._PNG
            b01, b11 = cosmo_funcs.get_PNG_bias(zz, ti, shape)
            phi1, phi2 = q1 ** K1.PNG_ALPHA[shape] / M1, q2 ** K1.PNG_ALPHA[shape] / M2
            amp = amp + (fNL if shape_fNL is None else shape_fNL) * (
                b01 / 2 * q1 * q2 * cos12 * (phi2 / q1**2 + phi1 / q2**2)
                + b11 / 2 * (phi1 + phi2)
                + f * b01 / 2 * p12 * (mu1 / q1 * phi2 + mu2 / q2 * phi1)
            )
        return D1**2 * amp

    @staticmethod
    def _PNG2(shapes, cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, fNL=1, **kwargs):  # fNL^2 phi_p^2 bias
        """sum over shapes of D1**2*fNL**2*b02/2*phi1*phi2, phi_i as _PNG - the b02 phi_p^2 term of 2011.13660
        (2.17), with (2.21) over fNL^2 as b_02. Only shapes whose holder has b_02, i.e. Loc: for the others the
        response to phi_p^2 needs the primordial trispectrum. Not a kernel of its own: numeric_mu.bk adds it to a
        field's Z2 with its PNG kernels."""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, q1, ti=ti)

        amp = 0
        for shape in shapes:
            b02 = getattr(getattr(cosmo_funcs.survey[ti], shape.lower(), None), "b_02", None)
            if b02 is None:
                continue
            shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # as K1._PNG
            phi1 = q1 ** K1.PNG_ALPHA[shape] / M_tail(cosmo_funcs, q1, zz)
            phi2 = q2 ** K1.PNG_ALPHA[shape] / M_tail(cosmo_funcs, q2, zz)
            amp = amp + (fNL if shape_fNL is None else shape_fNL) ** 2 * b02(zz) / 2 * phi1 * phi2
        return D1**2 * amp

    @staticmethod
    def _LP_PNG(shapes, cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, fNL=1, **kwargs):  # b_01 x GR projection
        """sum over shapes of D1**2*fNL/2*[U3*(phi2/q1**2 + phi1/q2**2)
        + 1j*(U4*(p1*phi1/q2**2 + p2*phi2/q1**2) + U5*(mu1/q1*phi2 + mu2/q2*phi1))] - 2011.13660 eq (3.26)
        lines 3-4 without Upsilon1, Upsilon2, halved as LP, phi_i as _PNG and U the Upsilon/fNL of
        betas.interpolate_upsilon. Not a kernel of its own: numeric_mu.bk adds it to a field's PNG Z2 when that
        field has LP and cosmo_funcs.full_betas is set. In internal modes, as LP."""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, q1, ti=ti)
        M1, M2 = M_tail(cosmo_funcs, q1, zz), M_tail(cosmo_funcs, q2, zz)
        p1, p2 = q1 * mu1, q2 * mu2

        amp = 0
        for shape in shapes:
            shape_fNL = kwargs.get(f"fNL_{shape.lower()}")  # as K1._PNG
            U3, U4, U5 = cosmo_funcs.get_upsilon(zz, ti, shape)
            phi1, phi2 = q1 ** K1.PNG_ALPHA[shape] / M1, q2 ** K1.PNG_ALPHA[shape] / M2
            amp = amp + (fNL if shape_fNL is None else shape_fNL) * (
                U3 * (phi2 / q1**2 + phi1 / q2**2)
                + 1j * (U4 * (p1 * phi1 / q2**2 + p2 * phi2 / q1**2) + U5 * (mu1 / q1 * phi2 + mu2 / q2 * phi1))
            )
        return D1**2 * amp / 2

    @staticmethod
    def Loc(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
        return K2._PNG(["Loc"], cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=ti, **kwargs)

    @staticmethod
    def Eq(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
        return K2._PNG(["Eq"], cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=ti, **kwargs)

    @staticmethod
    def Orth(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
        return K2._PNG(["Orth"], cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=ti, **kwargs)

    @staticmethod
    def PNG(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, **kwargs):
        """Loc + Eq + Orth in one kernel, as K1.PNG"""
        return K2._PNG(["Loc", "Eq", "Orth"], cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=ti, **kwargs)


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


# second order integrated kernels - evaluated rather than term lists, as a term can hold line-of-sight integrals
# over both first order fields, which no single radial array times mu**i * q**j can carry
class IntK2:
    """kappa^(2), the second order convergence, as a Z2: in the modes of the two first order fields with q3 = q1 + q2
    the mode it sources, as K2. The integrated bispectrum notes' (3.6a-e) and (4.7)-(4.12), corrected:
    - in our convention Delta = Delta^(1) + Delta^(2), as K2.N. The source of (3.6) writes Delta^(2)/2 (its -2|gamma|^2
      and +theta_ij theta^ij), so its products of first order fields are halved here while Phi^(2) keeps F2 - the
      same factor as 2011.13660, see K2.LP
    - grad_perp^2 = grad^2 - d_par^2 - (2/r) d_par, so 1 - mu**2 + 2i*mu/(q*r) as IntK1.L - (3.10) has +2/r
    - (4.9) counts both orderings twice and drops the r/d of (3.6b), (4.10) has the wrong sign, (4.11) drops the r
      of (3.6d) and every term needs its growth factors

    Each first order field on the line of sight enters through J_f(p) = int_0^d dr e^{ip(r-d)} f(r), p = q.x: its
    own e^{iq.x} comes out, so the kernel multiplies e^{i(q1+q2).x} like any source Z2 - exact with the line of sight
    on this field, leading (wide-angle) order otherwise. The f(r) depend only on the cosmology. Dlna = -Phi + d_par v
    at the source, without the ISW part 2I^(1) (as the notes). phi_r = 3/2*OM_r*H_r**2*D1_r, so Phi = -phi_r*delta/q**2,
    and A = (i q1_perp).(i q2_perp)."""

    # the IntK2 counterpart of each IntK1 kernel - I = L + TD + ISW has only its L part at second order, TD and ISW none yet
    SECOND_ORDER = {"L": "L", "kappa_g": "kappa2", "I": "L"}
    PARTIAL = {"I", "TD", "ISW"}

    # the second order field off the line of sight, by where its two modes sit: I1 both at one point, I2 + I3 one
    # integrated and the other at the source, I4 + I5 at two points. "I" is the field on the line of sight, where IntK2
    # integrates itself
    LAYOUTS = {"same": ("_I1",), "mix_a": ("_I2", "_I3"), "mix_b": ("_I2", "_I3"), "two": ("_I4", "_I5")}

    # and the products of first order fields that IntK2.L carries on top of 2(Q - 1) kappa^(2): I6 + I7 one lensing and
    # the source field, I8 two lensings
    PRODUCTS = {"mix_a": ("_I6", "_I7"), "mix_b": ("_I6", "_I7"), "two": ("_I8",)}

    # the power of q_j each IntK2 term's radial function on side j comes with, by term and key: a spectator scales q_j as
    # 1/lambda, so the kernel at lambda = 1 times lambda**-p is the kernel at lambda (_integrand_split). _I1 has both modes
    # at r_c, where it is the power of their common scale
    Q_POWERS = {
        "_I1": {"1a": 0, "1b": -1},
        "_I2": {"2a": 0, "2b": -1},
        "_I3": {"3": -1},
        "_I4": {"4a": 1, "4b": 0, "4c": -1},
        "_I5": {"4a": 0, "5c": 0},
        "_I6": {"4c": -1},
        "_I7": {"4b": 0, "4c": -1},
        "_I8": {"4b": 0, "4c": -1},
    }

    @staticmethod
    def radial(cosmo_funcs, zz, r):
        """The radial functions f(r) the terms below integrate along the line of sight - cosmology only. _los turns
        them into J_f(p); numeric_mu.int_bk evaluates them at quadrature nodes when this field is off the line of sight"""
        d = cosmo_funcs.comoving_dist(zz)
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)
        phi_r = 1.5 * OM_r * H_r**2 * D1_r
        u = r / d
        return {
            "1a": (d - r) * u * phi_r * D1_r,  # (1 - mu12**2) *
            "1b": (1 - u) * phi_r * D1_r,  # 2i*mu12/q12 * - the 1/r taken in
            "2a": u * phi_r,  # (1 - mu**2) *
            "2b": phi_r / d,  # 2i*mu/q *
            "3": phi_r,
            "4a": (d - r) * r * u * phi_r,  # (1 - mu**2) * - also I5 at r
            "4b": (d - r) * u * phi_r,  # 2i*mu/q *
            "4c": (1 - u) * phi_r,  # I4 at r'
            "5c": u * (u - 1) * phi_r,  # I5 at r'
        }

    @staticmethod
    def source(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=0):
        """The source scalars every term opens with: H, Q and Dlna of each mode - and for the products of first order
        fields (I6 - I8) the source field K1.N of each mode, D*b1' and Q' (luminosity derivatives at the cut, 0 for a
        tracer without them)"""
        _, H, _, Qm, _ = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)
        phi_x = 1.5 * cosmo_funcs.Om_m(zz) * H**2 * cosmo_funcs.D(zz)
        fHD = cosmo_funcs.f(zz) * H * cosmo_funcs.D(zz)
        dlna = [phi_x / q**2 + 1j * fHD * mu / q for q, mu in ((q1, mu1), (q2, mu2))]  # -Phi + d_par v
        tracer = cosmo_funcs.survey[ti]
        dQ, db1 = (getattr(tracer, name)(zz) if hasattr(tracer, name) else 0.0 for name in ("dQ_dlnL", "db1_dlnL"))
        ZN = [K1.N(cosmo_funcs, zz, mu, q, ti=ti) for q, mu in ((q1, mu1), (q2, mu2))]
        return {"H": H, "Q": Qm, "dlna": dlna, "ZN": ZN, "Db1p": cosmo_funcs.D(zz) * db1, "Qp": dQ}

    @staticmethod
    def _los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=0, deg=8):
        """What every term below opens with: the source scalars and J_f at p1, p2 and p1 + p2, with every radial
        function fitted once and evaluated at all three frequency groups. kappa2 hands it to all five,
        as IntK1.I does its unpack."""
        d = cosmo_funcs.comoving_dist(zz)
        r = np.linspace(1e-3, d, 1000)
        radial = IntK2.radial(cosmo_funcs, zz, r)
        arr = np.stack(list(radial.values()), axis=-1)
        coeffs_u = np.polyfit(r / d, arr, deg=deg)

        def J(p):
            p = np.asarray(p)
            I_arrs = integrate_polynomial(d, p.ravel(), coeffs_u)
            I_arrs = I_arrs * np.exp(-1j * p.ravel() * d)[:, None]
            return {key: I_arrs[:, m].reshape(p.shape) for m, key in enumerate(radial)}

        p1, p2 = q1 * mu1, q2 * mu2
        los = IntK2.source(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti)
        los.update(J=[J(p1), J(p2)], J12=J(p1 + p2))
        return los

    @staticmethod
    def _I1(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # lensing by Phi^(2)
        """(d - r)*r/d*phi_r*D1_r*F2*(1 - mu12**2 + 2i*mu12/(q12*r)) - (4.8), Phi^(2) = -phi_r*D1_r*F2*delta*delta/q12**2.
        EdS F2 - K(z) would vary along the line of sight."""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        J = los["J12"]
        q12 = np.sqrt(q1**2 + q2**2 + 2 * q1 * q2 * cos12)
        mu12 = (q1 * mu1 + q2 * mu2) / q12
        return F2(q1, q2, cos12, 3 / 7) * (J["1a"] * (1 - mu12**2) + J["1b"] * 2j * mu12 / q12)

    @staticmethod
    def _I2(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # source shift x lensing
        """-1/(2H)*Dlna_1*r/d*phi_r*(1 - mu2**2 + 2i*mu2/(q2*r)) + (1<->2) - (4.9)"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        qs, mus = (q1, q2), (mu1, mu2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            Jb = los["J"][b]
            tot = tot - los["dlna"][a] * (Jb["2a"] * (1 - mus[b] ** 2) + Jb["2b"] * 2j * mus[b] / qs[b])
        return tot / (2 * los["H"])

    @staticmethod
    def _I3(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # deflection x source shift
        """1/(2H)*A*phi_r/q1**2*Dlna_2 + (1<->2) - (4.10), S_perp.grad_perp Dlna/H"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        A = -q1 * q2 * (cos12 - mu1 * mu2)
        qs = (q1, q2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            tot = tot + A / qs[a] ** 2 * los["J"][a]["3"] * los["dlna"][b]
        return tot / (2 * los["H"])

    @staticmethod
    def _I4(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # post-Born
        """(d - r)*r**2/d*A*phi_r*(1 - mu1**2 + 2i*mu1/(q1*r)) * (1 - r'/d)*phi_r'/q2**2 + (1<->2) - (4.11), the
        r grad_perp grad_perp^2 Phi.(2S - grad_perp T) of (3.6d): the r and r' integrals factorise"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        A = -q1 * q2 * (cos12 - mu1 * mu2)
        qs, mus = (q1, q2), (mu1, mu2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            Ja, Jb = los["J"][a], los["J"][b]
            tot = tot + A / qs[b] ** 2 * (Ja["4a"] * (1 - mus[a] ** 2) + Ja["4b"] * 2j * mus[a] / qs[a]) * Jb["4c"]
        return tot

    @staticmethod
    def _I5(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # post-Born
        """(d - r)*r**2/d*phi_r * (r'/d)*(r'/d - 1)*phi_r' * A**2/(q1*q2)**2 + (1<->2) - (4.12), (q1_perp.q2_perp)**2 = A**2"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        A = -q1 * q2 * (cos12 - mu1 * mu2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            tot = tot + los["J"][a]["4a"] * los["J"][b]["5c"]
        return A**2 / (q1 * q2) ** 2 * tot

    # the products of first order fields in Delta^(2) that come with lensing - observed counts are the source field at
    # its true position, times the area and the flux cut moved by the magnification mu = 1/det A,
    # 1 + Delta = [1 + Delta_N(x + Dx_perp)] det A exp(Q ln mu - Q' (ln mu)**2 / 2), b1 -> b1 - b1' ln mu. Past
    # 2(Q - 1) kappa^(2) that leaves I6 - I8 (Nielsen & Durrer 1606.02113 eqs 2.14, 2.17, 2.19 at Q = 0). The 2i*mu/(q*r)
    # of grad_perp^2 is a pure trace (the sky's Hessian of f(r n) is r^2 grad_perp grad_perp f - r delta_ij d_r f), so
    # it is in kappa but not in the shear or the deflection

    @staticmethod
    def _I6(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # deflection of the source
        """(d - r)*phi_r*A/q1**2*Z_N(q2) + (1<->2): Dx_perp.grad_perp Delta_N, Dx_perp = -2 int dr (d - r) grad_perp Phi
        from observed to true position. No Q - the source moves with or without a flux cut"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        d = cosmo_funcs.comoving_dist(zz)
        A = -q1 * q2 * (cos12 - mu1 * mu2)
        qs = (q1, q2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            tot = tot + A / qs[a] ** 2 * d * los["J"][a]["4c"] * los["ZN"][b]
        return tot

    @staticmethod
    def _I7(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # magnification x source
        """kappa(q1)*[(Q - 1)*Z_N(q2) - D*b1'] + (1<->2): 2 kappa [(Q - 1) Delta_N - b1' D delta], kappa as
        IntK1.kappa_g"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        qs, mus = (q1, q2), (mu1, mu2)
        tot = 0
        for a, b in ((0, 1), (1, 0)):
            Ja = los["J"][a]
            kappa = Ja["4b"] * (1 - mus[a] ** 2) + Ja["4c"] * 2j * mus[a] / qs[a]
            tot = tot + kappa * ((los["Q"] - 1) * los["ZN"][b] - los["Db1p"])
        return tot

    @staticmethod
    def _I8(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):  # area and flux cut
        """[(1 - Q)(1 - 2Q) - 2Q'] kappa^2 + (Q - 1)|gamma|^2, the two lensings at r and r'. gamma from the flat Hessian
        psi_ij = 2 G q_i q_j / q**2 (G the (d - r) r/d phi_r integral): gamma_1.gamma_2 = 2 G1 G2 A**2/(q1 q2)**2 -
        G1 G2 (1 - mu1**2)(1 - mu2**2)"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        J1, J2 = los["J"]
        kappa = [J["4b"] * (1 - mu**2) + J["4c"] * 2j * mu / q for J, q, mu in ((J1, q1, mu1), (J2, q2, mu2))]
        A = -q1 * q2 * (cos12 - mu1 * mu2)
        gamma2 = J1["4b"] * J2["4b"] * (2 * A**2 / (q1 * q2) ** 2 - (1 - mu1**2) * (1 - mu2**2))
        Qm = los["Q"]
        return ((1 - Qm) * (1 - 2 * Qm) - 2 * los["Qp"]) * kappa[0] * kappa[1] + (Qm - 1) * gamma2

    @staticmethod
    def kappa2(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, los=None, **kwargs):
        """kappa^(2) = I1 + ... + I5, one _los for all five"""
        if los is None:
            los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        args = (cosmo_funcs, zz, mu1, mu2, q1, q2, cos12)
        terms = (IntK2._I1, IntK2._I2, IntK2._I3, IntK2._I4, IntK2._I5)
        return sum(term(*args, ti=ti, los=los) for term in terms)

    @staticmethod
    def L(cosmo_funcs, zz, mu1, mu2, q1, q2, cos12, ti=0, deg=8, **kwargs):  # lensing
        """2*(Qm - 1)*kappa^(2) + I6 + I7 + I8, the second order part of IntK1.L: what is linear in kappa^(2) and the
        products of first order lensing with the source field and itself. One _los for all eight"""
        los = IntK2._los(cosmo_funcs, zz, mu1, mu2, q1, q2, ti=ti, deg=deg)
        args = (cosmo_funcs, zz, mu1, mu2, q1, q2, cos12)
        products = sum(term(*args, ti=ti, los=los) for term in (IntK2._I6, IntK2._I7, IntK2._I8))
        return 2 * (los["Q"] - 1) * IntK2.kappa2(*args, ti=ti, los=los) + products
