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

    # Remove nested interpolators!
    # ok so lets change things so we only interpolate at the end and work normally with arrays in redshift till then

    # these derivs here are wrt to conformal time so we convert to derivs wrt to z
    # d/dt = da/dt d/da = a H dz/da d/dz =  -(1+z) H d/dz # everything here is conformal both t and H
    # d^2/d^2 t = (1+z)^2 H^2 d^2/d z^2 + H(1+z)(H+(1+z)H')d/dz

    zz = tracer.z_survey

    Q = tracer.Q(zz)
    b_e = tracer.be(zz)
    b_1 = tracer.b_1(zz)

    # derivatives wrt redshift
    dQ_dz = dy_dz(Q, zz)
    dbe_dz = dy_dz(b_e, zz)
    db1_dz = dy_dz(b_1, zz)

    # reduce intepolations - the cosmology half is cached, only the survey half above rebuilds
    zz_cached, H_c, f, Om, xi, dH_dt, dH_dt2 = cf.beta_cosmo[ti]
    if not np.array_equal(zz_cached, zz):  # a regridded tracer would otherwise mix two grids silently
        raise ValueError(f"beta_cosmo[{ti}] was built on another grid - run cf.compute_derivs_cosmo()")

    # derivatives wrt conformal time
    dQ_dt = -(1 + zz) * H_c * dQ_dz
    dbe_dt = -(1 + zz) * H_c * dbe_dz
    db1_dt = -(1 + zz) * H_c * db1_dz

    # generally set these partial derivatives to 0
    partdQ = 0
    partdb1 = 0

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
