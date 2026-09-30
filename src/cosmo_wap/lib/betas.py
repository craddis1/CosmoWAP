import numpy as np

from cosmo_wap.lib import utils


def dy_dz(y, zz):
    """d/dz on the uniform z_survey grid (see ClassWAP._process_survey).

    Same second-order stencil as np.gradient(y, zz, edge_order=2) - agrees with it to 1e-13 -
    but np.gradient's generic path costs a fifth of interpolate_beta_funcs on a 100 point grid,
    and a sampler rebuilds those per redshift bin. The uniform spacing is the whole saving, so
    a grid without it goes back to np.gradient rather than being approximated here - the same
    line lib.accel draws for its kernels."""
    dz = zz[1] - zz[0]
    if np.ptp(np.diff(zz)) > 1e-10 * abs(dz):
        return np.gradient(y, zz, edge_order=2)

    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dz)
    dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dz)  # one-sided at the edges
    dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dz)
    return dy


def cosmo_terms(cf, zz):
    """The survey-independent inputs the beta coefficients need on a z grid.

    Split out of interpolate_beta_funcs so it can be cached per cosmology
    (ClassWAP.compute_derivs_cosmo): a changed bias resets the betas but leaves these alone,
    so a sampler was rebuilding them once per redshift bin. zz comes back with the terms so
    the reader can tell the cache was built on the grid it is about to use them on - the two
    pick it up independently, off the same tracer. Conformal time derivatives as in
    interpolate_beta_funcs below."""
    H_c = cf.H_c(zz)
    dH_c = cf.dH_c(zz)
    dH_dt = -(1 + zz) * H_c * dH_c
    dH_dt2 = (1 + zz) ** 2 * H_c**2 * cf.ddH_c(zz) + H_c * (1 + zz) * (H_c + (1 + zz) * dH_c) * dH_c
    # dH22 = np.gradient(dH_dt(cf.z_cl),cf.conf_time(cf.z_cl)) # can also do numerically

    return zz, H_c, cf.f(zz), cf.Om_m(zz), cf.comoving_dist(zz), dH_dt, dH_dt2


def survey_terms(cf, ti=0):
    """The tracer half of the beta inputs on its z grid, with the cached cosmology half (cosmo_terms) -
    shared by interpolate_beta_funcs, interpolate_beta_HO and interpolate_upsilon.

    Returns zz, Q, b_e, b_1, dQ/dlnL, db1/dlnL, dlnLc/dz, the fixed-L conformal time derivatives
    Q', b_e', b_1', and the cosmo_terms tuple (H_c, f, Om, xi, dH_dt, dH_dt2)"""
    tracer = cf.survey[ti]

    # Remove nested interpolators!
    # ok so lets change things so we only interpolate at the end and work normally with arrays in redshift till then

    # these derivs here are wrt to conformal time so we convert to derivs wrt to z
    # d/dt = da/dt d/da = a H dz/da d/dz =  -(1+z) H d/dz # everything here is conformal both t and H
    # d^2/d^2 t = (1+z)^2 H^2 d^2/d z^2 + H(1+z)(H+(1+z)H')d/dz

    zz = tracer.z_survey

    Q = tracer.Q(zz)
    b_e = tracer.be(zz)
    b_1 = tracer.b_1(zz)

    # luminosity derivatives at the cut (2011.13660 A.29-A.31) - zero without a luminosity function
    partdQ = tracer.dQ_dlnL(zz)
    partdb1 = tracer.db1_dlnL(zz)
    dlnLc_dz = tracer.dlnLc_dz(zz)

    # derivatives wrt redshift at fixed luminosity - Q, b_e and b_1 above follow a cut that moves by
    # dlnLc_dz, so take that off; for b_e with d b_e/d ln L = d Q/d ln(1+z) at fixed L
    dQ_dz = dy_dz(Q, zz) - partdQ * dlnLc_dz
    dbe_dz = dy_dz(b_e, zz) - (1 + zz) * dQ_dz * dlnLc_dz
    db1_dz = dy_dz(b_1, zz) - partdb1 * dlnLc_dz

    # reduce intepolations - the cosmology half is cached, only the survey half above rebuilds
    zz_cached, *cosmo = cf.beta_cosmo[ti]
    if not np.array_equal(zz_cached, zz):  # a regridded tracer would otherwise mix two grids silently
        raise ValueError(f"beta_cosmo[{ti}] was built on another grid - run cf.compute_derivs_cosmo()")
    H_c = cosmo[0]

    # derivatives wrt conformal time
    dQ_dt = -(1 + zz) * H_c * dQ_dz
    dbe_dt = -(1 + zz) * H_c * dbe_dz
    db1_dt = -(1 + zz) * H_c * db1_dz

    return zz, Q, b_e, b_1, partdQ, partdb1, dlnLc_dz, dQ_dt, dbe_dt, db1_dt, cosmo


def interpolate_beta_funcs(cf, ti=0):
    """
    Function that relies on biases and functions defined in ClassWAP to return beta coefficients
    (beta expressions adapted from Eline de Weerd GitHub) from paper 1711.01812v4 - updated to match: 2011.13660

    Calculate and return beta coefficients values for given redshift and tracer

    Parameters:
    -----------
    cf : ClassWAP instance
        The cosmology and function class instance
    tracer : object
        Survey tracer object

    Returns:
    --------
    SplineStack of the beta values - calling it gives them all for a given redshift
    """
    tracer = cf.survey[ti]
    zz, Q, b_e, b_1, partdQ, partdb1, _, dQ_dt, dbe_dt, db1_dt, cosmo = survey_terms(cf, ti)
    H_c, f, Om, xi, dH_dt, dH_dt2 = cosmo

    # build every beta as an array over zz and spline them in one go - one solve, not one per beta
    # these recur in nearly every beta below - one evaluation each
    H2 = H_c**2
    f2 = f**2
    b3e = 3 - b_e
    Om32 = (3 / 2) * Om
    inv_xH = 1 / (xi * H_c)
    dH_H2 = dH_dt / H2

    gr1 = H_c * f * (b_e - 2 * Q - 2 * (1 - Q) * inv_xH - dH_H2)

    beta = [
        # for 1st order petrubation theory
        gr1,
        H2 * (f * b3e + Om32 * (2 + b_e - f - 4 * Q - 2 * (1 - Q) * inv_xH - dH_H2)),  # gr2
        # for second order pertubration theory
        H2 * (Om32 * (2 - 2 * f + b_e - 4 * Q - 2 * (1 - Q) * inv_xH - dH_H2)),  # 6
        H2 * (f * b3e),  # 7
        H2
        * (
            3 * Om * f * (2 - f - 2 * Q)
            + f2
            * (
                4
                + b_e
                - b_e**2
                + 4 * b_e * Q
                - 6 * Q
                - 4 * Q**2
                + 4 * partdQ
                + 4 * (dQ_dt / H_c)
                - (dbe_dt / H_c)
                - 2 * inv_xH**2 * (1 - Q + 2 * Q**2 - 2 * partdQ)
                - 2
                * inv_xH
                * (3 - 2 * b_e + 2 * b_e * Q - Q - 4 * Q**2 + 3 * dH_H2 * (1 - Q) + 4 * partdQ + 2 * (dQ_dt / H_c))
                - dH_H2 * (3 - 2 * b_e + 4 * Q + 3 * dH_H2)
                + (dH_dt2 / H_c**3)
            )
        ),  # 8
        H2 * (-(9 / 2) * Om * f),  # 9
        H2 * (3 * Om * f),  # 10
        H2
        * (Om32 * (1 + 2 * f / (3 * Om)) + 3 * Om * f - f2 * (-1 + b_e - 2 * Q - 2 * (1 - Q) * inv_xH - dH_H2)),  # 11
        H2
        * (
            -3 * Om * (1 + 2 * f / (3 * Om))
            - f * (b_1 * (f - 3 + b_e) + (db1_dt / H_c))
            + Om32
            * (b_1 * (2 + b_e - 4 * Q - 2 * (1 - Q) * inv_xH - dH_H2) + db1_dt / H_c + 2 * (2 - inv_xH) * partdb1)
        ),  # 12
        H2
        * (
            ((9 / 4) * Om**2 + Om32 * f * (1 - (2 * f) + 2 * b_e - 6 * Q - 4 * (1 - Q) * inv_xH - 3 * dH_H2))
            + (f2 * b3e)
        ),  # 13
        H_c * (-Om32 * b_1),  # 14
        H_c * 2 * f2,  # 15
        H_c
        * (
            f * (b_1 * (f + b_e - 2 * Q - 2 * (1 - Q) * inv_xH - dH_H2) + (db1_dt / H_c) + 2 * (1 - inv_xH) * partdb1)
        ),  # 16
        H_c * (-Om32 * f),  # 17
        H_c * (Om32 * f - f2 * (3 - 2 * b_e + 4 * Q + 4 * (1 - Q) * inv_xH + 3 * dH_H2)),
        # 18
        gr1,  # 19 - the same expression as gr1
    ]

    beta = utils.SplineStack(zz, beta)
    tracer.gr1, tracer.gr2 = beta[0], beta[1]

    return beta


def interpolate_beta_HO(cf, ti=0):
    """beta1-beta5 of 2011.13660 (A.1)-(A.5): the (H/k)^4 (beta1, beta2) and i(H/k)^3 (beta3-beta5) terms of
    the second order GR kernel (3.24). Only used with cf.full_betas - see numeric_mu.kernels.K2.LP.

    A SplineStack of their own, as the analytic bk terms unpack exactly the 16 of interpolate_beta_funcs.
    None of them depend on b_1."""
    zz, Q, b_e, _, partdQ, _, _, dQ_dt, dbe_dt, _, cosmo = survey_terms(cf, ti)
    H_c, f, Om, xi, dH_dt, dH_dt2 = cosmo

    # everything over the powers of H it comes with, as printed
    Qp, bep = dQ_dt / H_c, dbe_dt / H_c
    fp = -(1 + zz) * dy_dz(f, zz)  # f'/H - cosmology only, but nothing else needs it
    dH, ddH = dH_dt / H_c**2, dH_dt2 / H_c**3
    x = 1 / (xi * H_c)
    Om2, f2, be2, Q2 = Om**2, f**2, b_e**2, Q**2
    gr = b_e - 2 * Q - 2 * (1 - Q) * x - dH  # gamma1/(f H)

    beta = [
        H_c**4
        * (
            (9 / 4)
            * Om2
            * (
                6
                - 2 * f * (2 * b_e - 4 * Q - 4 * (1 - Q) * x - 2 * dH)
                - 2 * fp
                + be2
                + 5 * b_e
                - 8 * b_e * Q
                + 4 * Q
                + 16 * Q2
                - 16 * partdQ
                - 8 * Qp
                + bep
                + 2 * x**2 * (1 - Q + 2 * Q2 - 2 * partdQ)
                - 2 * x * (3 + 2 * b_e - 2 * b_e * Q - 3 * Q + 8 * Q2 - 3 * dH * (1 - Q) - 8 * partdQ - 2 * Qp)
                + dH * (-7 - 2 * b_e + 8 * Q + 3 * dH)
                - ddH
            )
            + (3 / 2)
            * Om
            * f
            * (
                5
                - 2 * f * (4 - b_e)
                + 2 * fp
                + 2 * b_e * (5 + 2 * (1 - Q) * x)
                - 2 * bep
                - 2 * be2
                + 8 * b_e * Q
                - 28 * Q
                - 14 * (1 - Q) * x
                - 3 * dH
                + 4 * (2 - x) * Qp
            )
            + (3 / 2) * Om * f2 * (-2 + 2 * f - b_e + 4 * Q + 2 * (1 - Q) * x + 3 * dH)
            + f2 * (12 - 7 * b_e + be2 + bep + (b_e - 3) * dH)
            - (3 / 2) * Om * fp
        ),  # 1
        H_c**4
        * (
            (9 / 2) * Om2 * (gr - 1)
            + 3 * Om * f * (-1 + 2 * f - b_e + 4 * Q + 2 * (1 - Q) * x + 3 * dH)
            + 3 * Om * f2 * (gr - 1)
            + 3 * Om * fp
        ),  # 2
        H_c**3
        * (
            (9 / 4) * Om2 * (f - 2 + 2 * Q)
            + (3 / 2)
            * Om
            * f
            * (
                -2
                - f * (-3 + f + 2 * b_e - 3 * Q - 4 * (1 - Q) * x - 2 * dH)
                - fp
                + 3 * b_e
                + be2
                - 6 * b_e * Q
                + 4 * Q
                + 8 * Q2
                - 8 * partdQ
                - 6 * Qp
                + bep
                + 2 * x**2 * (1 - Q + 2 * Q2 - 2 * partdQ)
                + 2 * x * (-1 - 2 * b_e + 2 * b_e * Q + Q - 6 * Q2 + 3 * dH * (1 - Q) + 6 * partdQ + 2 * Qp)
                - dH * (3 + 2 * b_e - 6 * Q - 3 * dH)
                - ddH
            )
            + f2
            * (-3 + 2 * b_e * (2 + (1 - Q) * x) - be2 + 2 * b_e * Q - 6 * Q - bep - 6 * (1 - Q) * x + 2 * (1 - x) * Qp)
        ),  # 3
        H_c**3 * (-(9 / 2) * Om * f * gr),  # 4
        H_c**3 * (3 * Om * f * gr),  # 5 - so beta4 + E2 beta5 = Om H^2 gr1 (3 E2 - 9/2), as beta9, beta10
    ]

    return utils.SplineStack(zz, beta)


def interpolate_upsilon(cf, ti=0, shape="Loc"):
    """Upsilon3-Upsilon5 of 2011.13660 (3.26) over fNL: the scale-dependent bias b_01 phi_p coupled to first
    order GR projection - (H/k)^4 (Upsilon3) and i(H/k)^3 (Upsilon4, Upsilon5). Only used with cf.full_betas -
    see numeric_mu.kernels.K2._LP_PNG.

    From the TERM x COEFFICIENT rows of Table 3, which give -R in Upsilon3 where (B.3) prints +R - as beta12,
    its b_10 counterpart. Upsilon1, Upsilon2 are 2 fNL gamma2, 2 fNL gamma1 (the first for LCDM growth), so are
    the full Z1 on the primordial bispectrum, which numeric_mu.bk has already.

    The paper is local only, with universality for the b_01 derivatives; here each shape's own b_01(z), whose
    scale dependence does not enter - the couplings are to the first order galaxy field, and phi_p has no
    time dependence. d b_01/d ln L is the holder's db_01_dlnL, zero if it has none (as compute_bias zeroes db1)."""
    zz, Q, b_e, _, _, _, dlnLc_dz, _, _, _, cosmo = survey_terms(cf, ti)
    H_c, f, Om, xi, dH_dt, _ = cosmo

    holder = getattr(cf.survey[ti], shape.lower())
    b01, _ = cf.get_PNG_bias(zz, ti, shape)
    db01_L = holder.db_01_dlnL(zz) if hasattr(holder, "db_01_dlnL") else 0 * zz
    db01_lna = -(1 + zz) * (dy_dz(b01, zz) - db01_L * dlnLc_dz)  # d b_01/d ln a at fixed L

    x = 1 / (xi * H_c)
    R = 2 * (1 - Q) * x + dH_dt / H_c**2  # (A.28)
    lum = 2 * (2 - x) * db01_L  # S/(b_10 - 1) b_01/4 under universality - (A.29)

    ups = [
        H_c**2 * ((3 / 2) * Om * (b01 * (2 + b_e - 4 * Q - R) + lum) + f * (b01 * (3 - f - b_e) + db01_lna / 2)),  # 3
        H_c * (-(3 / 2) * Om * b01),  # 4
        H_c * f * (b01 * (f + b_e - 2 * Q - R) + lum),  # 5
    ]

    return utils.SplineStack(zz, ups)
