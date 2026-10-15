"""Integrated bispectrum - B(k1,k2,k3,mu,phi) with line-of-sight integrated kernels (IntK1, IntK2) on any of the three
fields, at leading order in the separations over d. The line of sight is the endpoint d = x3 unless los_leg says
otherwise - the integrated terms depend on it at this order (2511.09466 sec 2.2.1), so it has to match the estimator.

Each tree-level term 2 Z1_a(q_a) Z1_b(q_b) Z2_c(-q_a, -q_b) P(q_a) P(q_b) has second order field c and first order
fields a < b, as numeric_mu.bk.get_mu_phi. An integrated part puts its field at r = y d along the line of sight, so with
x_i = d + x_iL (x_LL = 0) the separation x_iL it is Fourier transformed against comes in scaled by y, and the transverse
delta functions read
    k_i = sum_j M_ij q_j,   M_ij = y_j [field of j is i] - y_cj [c = i],   i != L, j = a, b
where y_cj is where the second order field picks up mode j. So q = M^-1 k with Jacobian |det M|^-3, and what is left is
the phase exp(i sum_j p_j (R_j - R_cj)), p_j = q_j.x, between the radial positions R of the two ends of each mode.

A point on the line of sight field never enters M, only the phase, so its integral is a J_f(p) spline - IntK1 through
numeric_mu.pk._int_K1_basis, IntK2 as it is. Every other integrated point is a quadrature node and moves the modes: with
c = L only a rescaling q_j = k_j / y_j, otherwise the second order field's points reshape the triangle too.

The phase is linear along every node variable - in u = 1/y for those in det M, in y for the rest - so the innermost one
is Filon (as numeric_mu.pk's SI term). Where that phase stops oscillating, near the observer the integrand is largest,
so each term has a narrow spike on the great circle of directions x perpendicular to the vector its phase varies along -
which only for c = L and the rescaled field on k1 is the equator of numeric_mu.bk's (mu, phi) grid. get_multipoles_int
therefore projects each term on its own sphere grid, pole along that vector and composite Gauss-Legendre clustered at
the spike; get_mu_phi_int only evaluates B on given nodes. The spike is no narrow bump but ~Lorentzian, flat below
mu ~ 1e-3 and falling over decades, so the panels are graded geometrically (numeric_mu.pk.get_mu_grid's single panel
pair at delta = 0.1 is ~3% on it).

This is the brute-force reference - each term costs ~n_y**dim local bispectra per angular node, dim the number of
integrated points off the line of sight (<= 3). The rescaled modes run out to q = uk ~ K_MAX, where the integrand still
falls only as a power of u, so u is log-spaced. Keep n_y >= 256: below that the Filon error moves erratically with the
direction and the angular sum never settles (at kd ~ 10, n_y = 64 swings a monopole by 2x between n_mu = 80 and 320;
at 256 most terms settle to ~0.3%, but see below).

get_multipoles_int_fast turns that around: at fixed nodes a term is a polynomial in x times e^{iW.x}, so with the pole
along W its angular integral is exact - Legendre series in the cosine against 2 i^n j_n(|W|) - and only radial
integrals are left, Filon on the e^{+-i|W|} of j_n. Nothing angular is sampled, so neither the spike nor the
e^{-ik.d} observer-end oscillation (which the reference's graded panels resolve only while kd is small) costs anything,
and the 0D and 1D terms do not get dearer with kd. Against it the reference's rescalings scatter by ~3% between grids at
kd ~ 10, and at kd ~ 100 its defaults are off by 2x or more: use it for the y integrals and as a check at small kd.
The reference does not settle at all on IIS (orders=(2,)), which is only in the fast path in practice.
"""

import functools
import itertools
import math

import numpy as np
from scipy.special import eval_legendre, lpmv

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from ._int_bk_common import U_MAX as U_MAX
from ._int_bk_common import _det_vars as _det_vars
from ._int_bk_common import _fields as _fields
from ._int_bk_common import _first as _first
from ._int_bk_common import _gram as _gram
from ._int_bk_common import _Grid as _Grid
from ._int_bk_common import _integrand as _integrand
from ._int_bk_common import _kvec as _kvec
from ._int_bk_common import _layout_terms as _layout_terms
from ._int_bk_common import _mode_geometry as _mode_geometry
from ._int_bk_common import _nodes as _nodes
from ._int_bk_common import _positions as _positions
from ._int_bk_common import _second as _second
from ._int_bk_common import _set_y as _set_y
from ._int_bk_common import _Spectator as _Spectator
from ._int_bk_common import _sphere as _sphere
from ._int_bk_common import _Term as _Term
from ._int_bk_common import _terms as _terms
from ._int_bk_common import solve_modes as solve_modes
from ._int_bk_radial import _G as _G
from ._int_bk_radial import SPEC_EFOLDS as SPEC_EFOLDS
from ._int_bk_radial import SPEC_PANEL as SPEC_PANEL
from ._int_bk_radial import _caps as _caps
from ._int_bk_radial import _corner as _corner
from ._int_bk_radial import _gl_log as _gl_log
from ._int_bk_radial import _graded as _graded
from ._int_bk_radial import _grid_0d as _grid_0d
from ._int_bk_radial import _grid_edge as _grid_edge
from ._int_bk_radial import _grid_lines as _grid_lines
from ._int_bk_radial import _grid_rays as _grid_rays
from ._int_bk_radial import _grids as _grids
from ._int_bk_radial import _jn_direct as _jn_direct
from ._int_bk_radial import _jn_filon as _jn_filon
from ._int_bk_radial import _node_map as _node_map
from ._int_bk_radial import _P_along as _P_along
from ._int_bk_radial import _polygon as _polygon
from ._int_bk_radial import _radial as _radial
from ._int_bk_radial import _reach as _reach
from ._int_bk_radial import _region as _region
from ._int_bk_radial import _rho as _rho
from ._int_bk_radial import _sides as _sides
from ._int_bk_radial import _spectate as _spectate
from ._int_bk_radial import _spectator as _spectator
from ._int_bk_radial import _spectator_work as _spectator_work
from ._int_bk_radial import _spline_map as _spline_map
from ._int_bk_reference import _filon as _filon
from ._int_bk_reference import _filon_weights as _filon_weights
from ._int_bk_reference import _outer as _outer
from ._int_bk_reference import _phase_vector as _phase_vector
from ._int_bk_reference import _pole as _pole
from ._int_bk_reference import _pole_grid as _pole_grid
from ._int_bk_reference import _setup as _setup
from ._int_bk_reference import _term as _term
from ._int_bk_reference import get_mu_phi_int as get_mu_phi_int
from ._int_bk_reference import get_multipoles_int as get_multipoles_int
from .kernels import IntK1, IntK2, survey_scalars, term_weight

# keep the existing metadata imports available; IntK2 owns the definitions
SECOND_ORDER = IntK2.SECOND_ORDER
PARTIAL = IntK2.PARTIAL
LAYOUTS = IntK2.LAYOUTS
PRODUCTS = IntK2.PRODUCTS
Q_POWERS = IntK2.Q_POWERS


################################################ the fast path: angles first ################################################


@functools.cache
def _legendre(n_c):
    """Gauss-Legendre nodes c in the cosine to the pole and the matrix (2n + 1)/2 w P_n(c) that projects a polynomial of
    degree < n_c on the Legendre polynomials. Shared, so read-only"""
    c, w = utils.leggauss(n_c)
    n = np.arange(n_c)[:, None]
    proj = (2 * n + 1) / 2 * w * eval_legendre(n, c)
    proj.flags.writeable = False
    return c, proj


def _re_ylm(l, m, mu, phi):
    """Re Y_lm(arccos mu, phi) as sph_harm_y's, through lpmv - half the cost"""
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
    return norm * lpmv(m, l, mu) * np.cos(m * phi)


def _integrand_split(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, y, spec, n_s):
    """_integrand with its spectators summed out (_node_map): a spectator lambda moves q_j = Q_j / lambda, the radial
    functions at r_j and r_cj and P(q_j), and nothing angular. So each factor it moves splits into a coefficient at
    lambda = 1 (IntK1 term by term, IntK2 term by term and key by key - Q_POWERS - with that radial function set to 1)
    and a function of lambda, and the sum over lambda (_spectator, n_s nodes) is done per node of the grid, not per
    angle. y at lambda = 1; returns the amplitude"""
    c, a, b, pa, pb, pc = term
    d = cosmo_funcs.comoving_dist(zz)
    q, _, mu_q, cos_ab, det = _mode_geometry(c, ks, mus, gram, los_leg, y)
    baseint = BaseInt(cosmo_funcs)
    group = {j: g for g, spectator in enumerate(spec) for j in spectator.sides}
    lam, w = zip(*(_spectator(spectator.top, n_s) for spectator in spec))
    # lambda**k of the Jacobian and |det M|**-3, with one factor of det M a side
    w = [wg * lg ** (spectator.power - 3 * len(spectator.sides)) for wg, lg, spectator in zip(w, lam, spec)]

    full = 2 * np.abs(det) ** -3.0  # the factors no spectator moves
    F, F_lam, P_lam = {}, {}, {}
    for j, i, part in (("a", a, pa), ("b", b, pb)):
        if j not in group:
            Fj = _first(part, i, q[j], mu_q[j], y[j] * d, False, fields, cosmo_funcs, zz)
            full = full * baseint.pk(q[j], zz) * Fj
            continue
        lg = lam[group[j]]
        terms = [
            t for kern in fields[i][1] for t in getattr(IntK1, kern)(y[j][..., None] * lg * d, cosmo_funcs, zz, ti=i)
        ]
        scal = survey_scalars(cosmo_funcs, zz, ti=i)
        F[j] = np.stack([term_weight(wt, scal) * mu_q[j] ** mi * q[j] ** qi for mi, qi, _, wt in terms], axis=-1)
        F_lam[j] = np.stack([arr * lg**-qi for _, qi, arr, _ in terms], axis=-2)
        P_lam[j] = baseint.pk(q[j][..., None] / lg, zz)

    # the second order kernel, IntK2 term by term with each spectator's radial functions as a basis on trailing axes
    split = [j for j in "ab" if j in group]
    nax = 1 if pc == "same" else len(split)
    ex = (...,) + (None,) * nax
    args = (cosmo_funcs, zz, -mu_q["a"][ex], -mu_q["b"][ex], q["a"][ex], q["b"][ex], cos_ab[ex])
    los = IntK2.source(*args[:6], ti=c)
    if pc == "same":  # both modes at y_c = lambda
        radial = {"ab": IntK2.radial(cosmo_funcs, zz, y["ca"][..., None] * lam[0] * d)}
    else:
        radial = {j: IntK2.radial(cosmo_funcs, zz, y["c" + j][..., None] * lam[group[j]] * d) for j in split}
    Fc, Fc_lam = [], [[] for _ in spec]
    for name, wt in _layout_terms(pc, fields[c][2], cosmo_funcs.survey[c].Q(zz)):
        keys, pw = list(Q_POWERS[name]), Q_POWERS[name]
        if pc == "same":
            Fc.append(wt * getattr(IntK2, name)(*args, ti=c, los=dict(los, J12=dict(zip(keys, np.eye(len(keys)))))))
            Fc_lam[0] += [radial["ab"][key] * lam[0] ** -pw[key] for key in keys]
            continue
        J, combos = [], []
        for j in "ab":
            if pc not in ("two", "mix_" + j):  # this mode at the source
                J.append(dict.fromkeys(keys, 0))
                combos.append([None])
            elif j in group:
                shape = (-1,) + (1,) * (nax - 1 - split.index(j))
                J.append({key: np.eye(len(keys))[n].reshape(shape) for n, key in enumerate(keys)})
                combos.append(keys)
            else:
                J.append({key: val[ex] for key, val in IntK2.radial(cosmo_funcs, zz, y["c" + j] * d).items()})
                combos.append([None])
        out = getattr(IntK2, name)(*args, ti=c, los=dict(los, J=J))
        out = np.broadcast_to(out, np.broadcast_shapes(np.shape(out), (len(keys),) * nax))
        Fc.append(wt * out.reshape(out.shape[: out.ndim - nax] + (-1,)))
        for combo in itertools.product(*combos):
            for j, key in zip("ab", combo):
                if key is not None:
                    Fc_lam[group[j]].append(radial[j][key] * lam[group[j]] ** -pw[key])
    Fc = np.concatenate(Fc, axis=-1)
    Fc_lam = [np.stack(fl, axis=-2) for fl in Fc_lam]

    # sum over lambda per node of the grid, then the coefficients a pair at a time
    if pc == "same":  # one spectator moves both sides
        S = np.einsum(
            "...l,...il,...jl,...kl->...ijk", w[0] * P_lam["a"] * P_lam["b"], F_lam["a"], F_lam["b"], Fc_lam[0]
        )
        S = np.einsum("...j,...jk->...k", F["b"], np.einsum("...i,...ijk->...jk", F["a"], S))
        return full * np.sum(Fc * S, axis=-1)
    for g, j in enumerate(split):  # group g is side j's
        S = np.einsum("...l,...il,...kl->...ik", w[g] * P_lam[j], F_lam[j], Fc_lam[g])
        Fc = Fc * np.einsum("...i,...ik->...k", F[j], S)
    return full * np.sum(Fc, axis=-1)


def _project(term, fields, cosmo_funcs, zz, tri, lm, grid, los_leg, n_c, n_az, sigma, n_s, split, max_size):
    """(l, m) multipoles of one term from a grid: on each ray the cosine to the pole and the azimuth are exact for
    polynomials (n_c and n_az nodes), and the radial kernel integrates e^{iXc} against the Legendre series in c. Only
    c > 0 is evaluated - B(-x) = B(x)* term by term and Re Y_lm(-x) = (-1)^l Re Y_lm(x), up to the reflection in the
    plane that both are even under. Spectators do not move W, so they are summed first - by _integrand_split, or on a
    trailing axis of y"""
    alpha, X, K, y, jac, spec, w_out = grid
    c, proj = _legendre(n_c)
    pos = slice(n_c // 2, None)  # n_c is even
    proj = proj[: K.shape[-1]]
    k1, k2, k3, theta = tri  # 4 trailing axes - rays, cosine, azimuth, X
    gram = _gram(k1, k2, k3, theta)
    n_y = _spectator_work(len(spec), n_s, split)  # the size of the spectator axis, or what stands in for it
    rows = max(1, max_size // (np.size(k1) * n_c // 2 * n_az * X.shape[-1] * n_y))
    out = 0
    for start in range(0, alpha.shape[-2], rows):
        sl = (..., slice(start, start + rows), slice(None))
        # the radial kernel times the Legendre projector at c and at -c
        Ks = K[sl + (slice(None),)]
        M = [np.einsum("...xn,nc->...cx", Ks, P[:, pos])[..., None, :] for P in (proj, proj[:, ::-1])]
        mus, (mu, phi), _ = _sphere(alpha[sl][..., None, None], k1, k2, k3, theta, c[pos], proj[0, pos], n_az)
        mus, mu, phi = [np.moveaxis(arr, -3, -1) for arr in mus], np.moveaxis(mu, -3, -1), np.moveaxis(phi, -3, -1)
        ys = {j: (val[sl] if np.ndim(val) else val) for j, val in y.items()}
        sp = [spectator._replace(top=spectator.top[sl]) for spectator in spec]
        if split and sp:
            ys = {j: (val[..., None, None, :] if np.ndim(val) else val) for j, val in ys.items()}
            sp = [spectator._replace(top=spectator.top[..., None, None, :]) for spectator in sp]
            args = (term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, ys, sp, n_s)
            val = _integrand_split(*args) * jac[sl][..., None, None, :]
        else:
            ys, jac_s = _spectate(ys, jac[sl], sp, n_s)
            ys = {j: (val[..., None, None, :, :] if np.ndim(val) else val) for j, val in ys.items()}
            ks, mus_s = [k[..., None] for k in (k1, k2, k3)], [m[..., None] for m in mus]
            gram_s = {key: val[..., None] for key, val in gram.items()}
            val, _ = _integrand(term, fields, cosmo_funcs, zz, ks, mus_s, gram_s, los_leg, False, ys)
            val = np.sum(val * jac_s[..., None, None, :, :], axis=-1)
        if sigma is not None:
            val = val * np.exp(-(1 / 2) * ((k1 * mus[0]) ** 2 + (k2 * mus[1]) ** 2 + (k3 * mus[2]) ** 2) * sigma**2)
        val = [2 * np.pi / n_az * val * M[0], 2 * np.pi / n_az * np.conj(val) * M[1]]
        if alpha.shape[-1] == 1:  # the pole is fixed along the ray, so Y_lm is too
            val = [v.sum(axis=-1, keepdims=True) for v in val]
        res = [np.sum(_re_ylm(l, m, mu, phi) * (val[0] + (-1) ** l * val[1]), axis=(-3, -2, -1)) for l, m in lm]
        out = out + np.sum(w_out[sl[:-1]] * np.array(res), axis=-1)
    return out


def get_multipoles_int_fast(
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
    los_leg=2,
    orders=(1,),
    perms=(0, 1, 2),
    allow_partial=False,
    n_c=None,
    n_az=None,
    n_d=32,
    n_f=64,
    n_psi=None,
    x_s=None,
    deg=8,
    n_p=2000,
    n_s=48,
    split=True,
    max_size=2**21,
):
    """(l, m) multipoles of the integrated terms as get_multipoles_int, angles first: at fixed nodes a term is a
    polynomial in x times e^{iW.x}, so on a sphere grid with the pole along W its angular integral is exact - azimuth
    by the midpoint rule, the cosine c by its Legendre series, int P_n(c) e^{iXc} dc = 2 i^n j_n(X), X = |W|. What is
    left is radial - in X, Gauss-Legendre up to x_s and a Filon on the e^{+-iX} of j_n past it - so neither the near
    observer spike nor the e^{-ik.d} oscillation of the reference's angular grids is ever sampled, and the cost does
    not grow with kd but for the 2D terms.

    Every integral is on nodes, the line of sight field's too, and W/d = alpha k_o + beta k_l over the term's region
    of (alpha, beta) (_region): a ray along a side for one node variable, else a convex polygon, cut as seen from W = 0
    into triangles to its finite edges, half-strips under its unbounded ones and the cone of directions that never
    leave it (_polygon) - rays to the edges and caps, lines across the strips, n_d + n_f nodes in X on each. A far edge
    the integrand does not vanish on puts e^{iX} on each ray's or line's end, so n_psi - by default ~ kd/4 edge rays and
    ~ kd/2 strip lines, kd of the largest side passed, and 16 cone rays, which have no such edge - grows with kd; an int
    sets all three. Where W fixes only the ratio of a mode's two ends (orders 2 and 3), their scale is a spectator, on
    n_s nodes in ln y that _integrand_split sums at each node of the grid rather than at each angle - split=False sums
    the whole integrand over them instead, for checks.

    n_c (even) and 2 n_az must exceed the degree in x of kernels x Y_lm, ~9 + l for N + LP + L: by default
    10 + l_max and n_c / 2. FoG damping (sigma) is not a polynomial, so it holds while k sigma is not much past 1.
    The second order lensing field is IntK2.L: 2(Q - 1) kappa^(2) and the products of first order lensing with the
    source field and itself (deflection, magnification, kappa^2 and |gamma|^2). Radial shifts by the time delay and
    ISW, and the second order time delay and ISW themselves, are not in.

    deg and n_p are accepted for compatibility; this path evaluates every integral on nodes.
    max_size targets elements per work array: triangles are batched before grids are built, pieces are streamed and
    rays are chunked before expanding spectators. At least one triangle and one ray are evaluated at a time; it is
    not a byte limit. Automatic quadrature settings are fixed for the whole batch."""
    k3, theta = utils.get_theta_k3(k1, k2, k3, theta)
    k1, k2, k3, theta = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (k1, k2, k3, theta)))
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    d = cosmo_funcs.comoving_dist(zz)
    n_c = n_c or 10 + max(l for l, _ in lm)
    n_c += n_c % 2
    n_az = n_az or n_c // 2
    x_s = x_s or max(30.0, n_c + 10.0)
    if n_psi is None:
        kd = d * np.max((k1, k2, k3))
        n_psi = (max(16, int(np.ceil(kd / 4))), 16, max(16, int(np.ceil(kd / 2))))
    elif np.ndim(n_psi) == 0:
        n_psi = (n_psi,) * 3

    terms = list(_terms(fields, los_leg, orders, perms, False))
    if not terms:
        return np.zeros((len(lm),) + k1.shape)

    count = 0
    for term in terms:
        _, _, _, I, C = _sides(term, los_leg)
        spectators = int(I["a"] and I["b"]) if term.part_c == "same" else sum(I[j] and C[j] for j in "ab")
        count = max(count, spectators)
    n_x = n_d + 2 * n_f  # a strip has a direct segment and two Filon branches
    projection_work = n_c // 2 * n_az * n_x * _spectator_work(count, n_s, split)
    grid_work = max(n_psi) * max(n_x, 4 * (n_f - 1) + 1) * n_c  # Filon's fine grid has refine=4
    step = max(1, max_size // max(projection_work, grid_work))

    shape = k1.shape
    flat = [x.reshape(-1) for x in (k1, k2, k3, theta)]
    out = np.zeros((len(lm), k1.size), dtype=complex)
    for start in range(0, k1.size, step):
        sl = slice(start, start + step)
        sides = [x[sl] for x in flat]
        kvec = _kvec(sides[0], sides[1], sides[3])
        tri = [x[..., None, None, None, None] for x in sides]
        for term in terms:
            if _nodes(term, los_leg, False):
                grids = _grids(term, los_leg, d, kvec, x_s, n_d, n_f, n_c, n_psi, cosmo_funcs, zz)
            else:
                grids = [_grid_0d()]
            for grid in grids:
                args = (los_leg, n_c, n_az, sigma, n_s, split, max_size)
                out[:, sl] += _project(term, fields, cosmo_funcs, zz, tri, lm, grid, *args)
    return out.reshape((len(lm),) + shape)
