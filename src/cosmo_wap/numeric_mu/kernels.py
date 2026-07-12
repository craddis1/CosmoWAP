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
    @staticmethod
    def N(cosmo_funcs, zz, mu, k1, ti=0):  # kaiser
        """D1*(b1 + f*mu**2)"""
        # unpack all necessary terms
        D1, f, b1 = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        return D1 * (b1 + f * mu**2)

    @staticmethod
    def LP(cosmo_funcs, zz, mu, k1, ti=0):  # local projection effects # GR1 and GR2
        """D1*(1j*mu*gr1/k1 + gr2/k1**2)"""
        # unpack all necessary terms
        D1, _, _ = Unpack.common(cosmo_funcs, zz, k1, ti=ti)
        gr1, gr2 = cosmo_funcs.get_beta_funcs(zz, ti=ti)[:2]
        return D1 * (1j * mu * gr1 / k1 + gr2 / k1**2)


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
    def L(r, cosmo_funcs, zz=0, ti=0):  # lensing
        """3*D1_r*(Qm - 1)*OM_r*H_r**2*(d - r)*r/d * (1 - mu**2 + 2j*mu/(r*q))"""
        d, _, _, _, _ = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)  # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)  # integrand params - arrays in shape (xd)

        tmp_arr = 3 * D1_r * OM_r * H_r**2 * (d - r) * r / d  # [1-mu**2+2i mu/r*q] *
        wt = {1: -1.0, "Q": 1.0}  # (Qm - 1)

        return [(0, 0, tmp_arr, wt), (2, 0, -tmp_arr, wt), (1, -1, 2j * tmp_arr / r, wt)]

    @staticmethod
    def TD(r, cosmo_funcs, zz=0, ti=0):  # time delay
        """6*D1_r*(Qm - 1)*OM_r*H_r**2/d * 1/q**2"""
        d, _, _, _, _ = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)  # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)  # integrand params - arrays in shape (xd)

        tmp_arr = 6 * D1_r * OM_r * H_r**2 / (d)  # k1**2 *
        wt = {1: -1.0, "Q": 1.0}  # (Qm - 1)

        return [(0, -2, tmp_arr, wt)]

    @staticmethod
    def ISW(r, cosmo_funcs, zz=0, ti=0):  # integrated Sachs-Wolfe
        """3*D1_r*(be - 2*Qm + 2*(Qm - 1)/(d*H) - Hp/H**2)*OM_r*H_r**3*(f_r - 1) * 1/q**2"""
        d, H, Hp, _, _ = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)  # source integrated params
        _, f_r, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)  # integrand params - arrays in shape (xd)

        tmp_arr = 3 * D1_r * OM_r * H_r**3 * (f_r - 1)  # k1**2 *
        wt = {1: -2 / (d * H) - Hp / H**2, "Q": -2 + 2 / (d * H), "be": 1.0}  # be - 2*Qm + 2*(Qm-1)/(d*H) - Hp/H**2

        return [(0, -2, tmp_arr, wt)]

    @staticmethod
    def I(r, cosmo_funcs, zz=0, ti=0):
        """Combined (L+TD+ISW) integrated 1st order kernel"""
        args = (r, cosmo_funcs, zz)
        return IntK1.L(*args, ti=ti) + IntK1.TD(*args, ti=ti) + IntK1.ISW(*args, ti=ti)

    @staticmethod
    def kappa_g(r, cosmo_funcs, zz=0, ti=0):
        """(3/2)*D1_r*OM_r*H_r**2*(d - r)*r/d * (1 - mu**2 + 2j*mu/(r*q))"""
        d, _, _, _, _ = Unpack.get_int_params(cosmo_funcs, zz, ti=ti)  # source integrated params
        _, _, D1_r, H_r, OM_r = Unpack.get_integrand_params(cosmo_funcs, r)  # integrand params - arrays in shape (xd)

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
    return sum(term_weight(wt, scal) * mu**i * qq**j * arr for i, j, arr, wt in terms)
