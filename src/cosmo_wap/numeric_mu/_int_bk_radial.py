"""Radial grids and spectator nodes for the angular-first integrated bispectrum solver."""

import functools
import itertools

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from ._int_bk_common import U_MAX, _Grid, _nodes, _Spectator, solve_modes
from .integration import filon_weights as _filon_weights

# a spectator's range in ln y below its top - its integrand falls as y**2 or faster there (P(q) at q = Q/y past the
# turnover), so the rest is < 1e-8 - in panels of SPEC_PANEL Gauss-Legendre nodes, which resolve the BAO in P(Q/y)
SPEC_EFOLDS = 9
SPEC_PANEL = 16


@functools.cache
def _spline_map(n_f, refine):
    """The cubic spline through n_f equally spaced values, at refine times as many points - as a matrix. Read-only"""
    s = np.linspace(0, 1, n_f)
    S = CubicSpline(s, np.eye(n_f), axis=0)(np.linspace(0, 1, (n_f - 1) * refine + 1))
    S.flags.writeable = False
    return S


def _jn_direct(X, w, n_c):
    """Radial kernel of nodes X with weights w that take the angle-integrated plane wave as it is: 2 i^n j_n(X)"""
    n = np.arange(n_c)
    return w[..., None] * 2 * 1j**n * spherical_jn(n, X[..., None])


def _G(X, n_c):
    """G_n = h_n(X) e^{-iX}, n < n_c, on a trailing axis - by the upward recurrence, stable for h_n"""
    G = [-1j / X, -1 / X - 1j / X**2]
    for n in range(1, n_c - 1):
        G.append((2 * n + 1) / X * G[n] - G[n - 1])
    return np.stack(G[:n_c], axis=-1)


def _jn_filon(X0, X_lo, X_hi, n_f, n_c, weight=None, refine=4):
    """Nodes on [X_lo, X_hi], geometric in X - X0, and their radial kernel i^n (W+ G_n + W- G_n*): j_n is e^{+-iX}
    times G_n = h_n(X) e^{-iX}, so the amplitude over weight(X) is a cubic spline in the node index, and G_n, the weight
    and the phase are exact on refine times as many nodes (_filon_weights there) - near X ~ n^2 G_n moves faster than
    the nodes do, and the weight takes P(q) along the ray, BAO and turnover, out of the spline. Zero where X_hi <= X_lo,
    with the nodes at X_hi - still on the ray or line, where the node positions make sense"""
    on = X_hi > X_lo
    X0 = np.asarray(X0)[..., None]
    with np.errstate(divide="ignore"):  # X_hi = X0 puts the nodes at X0
        lo = np.log(np.where(on, X_lo, X_hi) - X0[..., 0])[..., None]
        hi = np.log(X_hi - X0[..., 0])[..., None]
    X = X0 + np.exp(lo + (hi - lo) * np.linspace(0, 1, n_f))
    X_fine = X0 + np.exp(lo + (hi - lo) * np.linspace(0, 1, (n_f - 1) * refine + 1))
    G = _G(X_fine, n_c)
    WG = _filon_weights(X_fine, 1.0)[..., None] * G + _filon_weights(X_fine, -1.0)[..., None] * np.conj(G)
    if weight is not None:
        WG = WG * weight(X_fine)[..., None]
    St = np.swapaxes(_spline_map(n_f, refine), 0, 1)
    K = St @ WG.real + 1j * (St @ WG.imag)
    if weight is not None:
        K = K / weight(X)[..., None]
    return X, np.where(on[..., None, None], 1j ** np.arange(n_c) * K, 0)


def _gl_log(lo, hi, n, scale=None):
    """Gauss-Legendre nodes and weights on [lo, hi] (trailing axis), in ln(scale + x) if a scale is given - for the
    rescalings, whose integrand moves on the scale of u - 1, scale = what X or beta is at u = 2"""
    x, w = utils.leggauss(n)
    lo, hi = lo[..., None], hi[..., None]
    if scale is None:
        return lo + (x + 1) / 2 * (hi - lo), w / 2 * (hi - lo)
    scale = scale[..., None]
    v_lo, v_hi = np.log(scale + lo), np.log(scale + hi)
    v = v_lo + (x + 1) / 2 * (v_hi - v_lo)
    return np.exp(v) - scale, w / 2 * (v_hi - v_lo) * np.exp(v)


def _radial(X_max, x_s, n_d, n_f, n_c, scale=None, weight=None):
    """Nodes and radial kernel for int_0^X_max dX: Gauss-Legendre up to x_s (_gl_log), _jn_filon past it"""
    X_d, w_d = _gl_log(np.zeros_like(X_max), np.minimum(x_s, X_max), n_d, scale)
    X_f, K_f = _jn_filon(0.0, np.full_like(X_max, x_s), X_max, n_f, n_c, weight)
    return np.concatenate((X_d, X_f), axis=-1), np.concatenate((_jn_direct(X_d, w_d, n_c), K_f), axis=-2)


def _sides(term, los_leg):
    """The two first order sides j = a, b of a term as (o, l), with W/d = alpha k_o + beta k_l - l the field on the line
    of sight when c is off it. Returns (aligned, o, l, I, C): I[j] whether side j's first order field is integrated,
    C[j] whether the second order field picks up mode j at an integrated point"""
    c, a, b, pa, pb, pc = term
    o, l = ("a", "b") if c == los_leg or b == los_leg else ("b", "a")
    I = {"a": pa == "I", "b": pb == "I"}
    C = {j: pc in ("same", "two", "mix_" + j) for j in "ab"}
    return c == los_leg, o, l, I, C


def _rho(aligned, al, be):
    """The ratios rho_j = y_cj / y_j at (alpha, beta) and d(rho_o, rho_l)/d(alpha, beta): the phase only knows where
    each mode's two ends sit relative to each other, W/d = (1 - rho_o) k_o + (1 - rho_l) k_l with c on the line of sight
    and the same over rho_l with it off (q_l = ((1 - rho_o) k_o + k_l) / y_cl)"""
    if aligned:
        return (1 - al, 1 - be), ((-1, 0), (0, -1))
    rl = 1 / (1 + be)
    return (1 - al * rl, rl), ((-rl, al * rl**2), (0, -(rl**2)))


def _region(term, los_leg):
    """The term's region of (alpha, beta), from the range of each rho_j: [1, inf) with only y_j integrated, (0, 1] with
    only y_cj, (0, inf) with both and 1 with neither - and the I1 layout's shared point orders the two when only one
    first order field is integrated. Returns half-planes (n, h), n.p <= h (h >= 0: W = 0 is in it), the direction e of
    the line p = t e it lies on if it is 1D, and the normals of the lines through W = 0 where a spectator's range has a
    kink (_node_map: rho_j = 1, and rho_o = rho_l for the shared one) - the grids are not to cross them"""
    aligned, o, l, I, C = _sides(term, los_leg)
    planes, fixed, kinks = [], [], []
    for j, x in ((o, (1.0, 0.0)), (l, (0.0, 1.0))):
        neg = (-x[0], -x[1])
        # rho_j > 0 - with c off the line of sight rho_l -> inf at beta = -1 too, the observer end of y_l
        bound = (x, 1.0) if aligned else ((1.0, -1.0), 1.0) if j == o else (neg, 1.0)
        if I[j] and not C[j]:  # rho_j >= 1
            planes += [(x, 0.0)] + ([] if aligned or j == o else [bound])
        elif C[j] and not I[j]:  # rho_j <= 1
            planes += [(neg, 0.0)] + ([bound] if aligned or j == o else [])
        elif I[j]:
            planes.append(bound)
            kinks.append(x)  # rho_j = 1
        else:
            fixed.append(j)
    if term.part_c == "same":
        if I[o] != I[l]:  # y_j <= 1 for the integrated one: rho_j >= rho_k
            planes.append(((1.0, -1.0) if I[o] else (-1.0, 1.0), 0.0))
            kinks = []
        elif I[o]:
            kinks.append((1.0, -1.0))  # rho_o = rho_l
        else:
            return planes, (1.0, 1.0), []
    if fixed:
        return planes, (0.0, 1.0) if fixed == [o] else (1.0, 0.0), kinks
    return planes, None, kinks


def _caps(term, los_leg):
    """Where a rescaling stops, as half-planes the rays and lines end on (the decomposition does not see them): rho_j <=
    U_MAX for an integrated first order field, so its y (or its spectator's top) stays >= 1/U_MAX as in the reference,
    and with c off the line of sight y_cl = rho_l >= 1/U_MAX. Off them the modes run to q ~ U_MAX**2 k, where IntK2's
    |q_a + q_b| is lost to rounding"""
    aligned, o, l, I, C = _sides(term, los_leg)
    caps = []
    for j, x in ((o, (1.0, 0.0)), (l, (0.0, 1.0))):
        if I[j] and aligned:  # 1 - x_j <= U_MAX
            caps.append(((-x[0], -x[1]), U_MAX - 1))
        elif I[j]:  # alpha >= (1 - U_MAX)(1 + beta), or 1 + beta >= 1/U_MAX
            caps.append(((-1.0, 1 - U_MAX), U_MAX - 1) if j == o else ((0.0, -1.0), 1 - 1 / U_MAX))
        if C[j] and not aligned and j == l:
            caps.append(((0.0, 1.0), U_MAX - 1))
    return caps


def _reach(P, D, caps):
    """How far P + s D goes (s >= 0) before it crosses a cap - inf if it never does"""
    s = np.inf
    for n, h in caps:
        nD = n[0] * D[0] + n[1] * D[1]
        s = np.where(nD > 0, np.minimum(s, (h - n[0] * P[0] - n[1] * P[1]) / np.where(nD > 0, nD, 1)), s)
    return s


def _polygon(planes):
    """The convex region n.p <= h as seen from W = 0, which it contains: a ray from W = 0 leaves it through one edge, so
    it is a triangle to each finite edge, a half-strip {g V + s D: g in [0, 1], s >= 0} under each unbounded one V + s D,
    and the cone of directions that never leave. Returns [("edge", V0, V1), ("strip", V, D), ("cone", E0, E1)], E0 to
    E1 anticlockwise"""
    planes = [(np.array(n), h) for n, h in planes]
    pieces = []
    for i, (n, h) in enumerate(planes):
        if h == 0:  # through W = 0, where the region's angular range ends
            continue
        t, p0 = np.array([-n[1], n[0]]), h * n / (n @ n)
        lo, hi = -np.inf, np.inf  # the edge p0 + s t
        for j, (m, hm) in enumerate(planes):
            mt, slack = m @ t, hm - m @ p0
            if j == i or (abs(mt) < 1e-12 and slack >= 0):
                continue
            if abs(mt) < 1e-12:
                lo = np.inf
            elif mt > 0:
                hi = min(hi, slack / mt)
            else:
                lo = max(lo, slack / mt)
        if not hi > lo + 1e-12:
            continue
        if np.isfinite(lo) and np.isfinite(hi):
            pieces.append(("edge", p0 + lo * t, p0 + hi * t))
        elif np.isfinite(lo) or np.isfinite(hi):
            pieces.append(("strip", p0 + lo * t, t) if np.isfinite(lo) else ("strip", p0 + hi * t, -t))
        else:
            pieces += [("strip", p0, t), ("strip", p0, -t)]

    # the cone: directions e with n.e <= 0 for every plane, an arc between two of their edges' angles
    cand = sorted({(np.arctan2(n[1], n[0]) + s * np.pi / 2) % (2 * np.pi) for n, _ in planes for s in (1, -1)})
    cand.append(cand[0] + 2 * np.pi)
    ok = [all(n @ (np.cos(m), np.sin(m)) <= 0 for n, _ in planes) for m in (np.array(cand[:-1]) + cand[1:]) / 2]
    if any(ok):
        k = ok.index(False)  # roll so the arc does not wrap
        ok, cand = ok[k:] + ok[:k], cand[k:-1] + [c + 2 * np.pi for c in cand[: k + 1]]
        first, last = ok.index(True), len(ok) - 1 - ok[::-1].index(True)
        pieces.append(("cone", *(np.array([np.cos(c), np.sin(c)]) for c in (cand[first], cand[last + 1]))))
    return pieces


def _corner(aligned, V):
    """Whether the vertex V of (alpha, beta) is the image of a whole edge of the y's - rho_l infinite with c off the line
    of sight (y_l -> 0), or both rho_j zero with it on (y_c -> 0 at a fixed y_c / y_j). The integrand goes as one over
    the distance to it there, so the parameters along an edge or across a strip are graded towards it"""
    return bool(V[1] <= -1 + 1e-12) if not aligned else bool(min(V) >= 1 - 1e-12)


def _graded(n, at0, at1):
    """Gauss-Legendre nodes and weights on [0, 1], quadratically clustered at the ends flagged"""
    v, w = utils.leggauss(n)
    v, w = (v + 1) / 2, w / 2
    if at0 and at1:
        return v**2 * (3 - 2 * v), w * 6 * v * (1 - v)
    if at0:
        return v**2, w * 2 * v
    if at1:
        return 1 - (1 - v) ** 2, w * 2 * (1 - v)
    return v, w


def _grid_edge(V0, V1, aligned, ko, kl, d, x_s, n_d, n_f, n_c, n_ray, caps, weight, cone=False):
    """The triangle from W = 0 to the edge V0 V1: rays to points B spaced evenly along it, so a thin triangle needs no
    more of them than a fat one, dalpha dbeta = |V0 x V1| lambda dlambda dt - each short of the edge (where a y can be 0
    or rho_l infinite) and of the caps, and graded towards a _corner. With cone, the directions E0 = V0 to E1 = V1 that
    never leave the region - spaced along the chord in (alpha, beta), as even angles bunch where k_o ~ -k_l - and each
    ray goes on to its cap, Gauss-Legendre in ln(scale + X) near W = 0 (a rescaling, scale the X where max(|alpha|,
    |beta|) = 1)"""
    t, w = _graded(n_ray, *(not cone and _corner(aligned, V) for V in (V0, V1)))
    B = [V0[i] + t * (V1[i] - V0[i]) for i in (0, 1)]
    Bp = B[0] * ko + B[1] * kl
    Bn = d * np.hypot(Bp[0], Bp[1])  # X at lambda = 1
    reach = np.maximum(_reach((0, 0), B, caps), 0)

    def wX(X):  # the weight (_P_along) along each ray
        return weight(X / Bn[..., None] * B[0][:, None], X / Bn[..., None] * B[1][:, None])

    if cone:
        top = np.maximum(np.abs(B[0]), np.abs(B[1]))
        X, K = _radial(Bn * np.minimum(reach, (U_MAX - 1) / top), x_s, n_d, n_f, n_c, Bn / top, wX)
    else:
        X, K = _radial(Bn * np.minimum(reach, 1 - 1e-9), x_s, n_d, n_f, n_c, weight=wX)
    lam = X / Bn[..., None]
    dA_dX = abs(V0[0] * V1[1] - V0[1] * V1[0]) * lam / Bn[..., None]
    return np.arctan2(Bp[1], Bp[0])[..., None], X, K, lam * B[0][:, None], lam * B[1][:, None], dA_dX, w


def _grid_lines(V, D, aligned, ko, kl, d, x_s, n_d, n_f, n_c, n_line, caps, weight):
    """The half-strip {g V + s D} under an unbounded edge, on lines of fixed g in (0, 1] parallel to it, where the
    integrand does not vanish - polar rays ending on it pick up its phase e^{iX} at an X without bound. X along a line is
    hyperbolic with at most one stationary point s_s (obtuse triangles): Gauss-Legendre while X - X_min < x_s around it,
    in ln(scale + s) with scale the s over which a ratio rho_j moves by itself from the line's start, and _jn_filon in X
    on the branches either side, out to the caps. The lines are graded towards V if it is a _corner"""
    g, w_g = _graded(n_line, False, _corner(aligned, V))
    g_cap = min([1.0] + [h / (n[0] * V[0] + n[1] * V[1]) for n, h in caps if n[0] * V[0] + n[1] * V[1] > 0])
    g, w_g = g * g_cap, w_g * g_cap  # lines that start past a cap are not there
    Vp, Dp = (u[0] * ko + u[1] * kl for u in (V, D))
    Dn = np.hypot(Dp[0], Dp[1])
    P = Vp * g  # W/d = P + s D
    s_s = -(P[0] * Dp[0] + P[1] * Dp[1]) / Dn**2
    X_min = d * np.abs(P[0] * Dp[1] - P[1] * Dp[0]) / Dn
    kD = d * Dn
    reach = np.minimum(_reach((g * V[0], g * V[1]), D, caps), (U_MAX - 1) / np.max(np.abs(D)))
    s0, s1 = -s_s, reach - s_s  # in s - s_s
    rho, drho = _rho(aligned, g * V[0], g * V[1])
    scale = 1 / np.maximum(*(np.abs((dr[0] * D[0] + dr[1] * D[1]) / r) for r, dr in zip(rho, drho)))

    def X_at(s, ax=()):  # ax: the node axis, if s has one
        return np.sqrt(X_min[(...,) + ax] ** 2 + (kD[(...,) + ax] * s) ** 2)

    sd = np.sqrt((X_min + x_s) ** 2 - X_min**2) / kD
    lo, hi = np.maximum(s0, -sd), np.maximum(np.minimum(s1, sd), np.maximum(s0, -sd))
    s_dir, w_dir = _gl_log(lo + s_s, hi + s_s, n_d, scale + 0 * lo)
    s_dir = s_dir - s_s[..., None]
    X_dir = X_at(s_dir, (None,))
    K_dir = _jn_direct(X_dir, w_dir, n_c)
    branches = [(-1, X_min + x_s, np.where(s0 < -sd, X_at(s0), X_min + x_s)), (1, X_at(np.maximum(s0, sd)), X_at(s1))]
    s, X, K = [s_dir], [X_dir], [K_dir]

    def wX(X, sign):  # the weight (_P_along) along each branch
        s = np.maximum(sign * np.sqrt(np.maximum(X**2 - X_min[..., None] ** 2, 0)) / kD[..., None] + s_s[..., None], 0)
        return weight(g[:, None] * V[0] + s * D[0], g[:, None] * V[1] + s * D[1])

    for sign, X_lo, X_hi in branches:
        X_b, K_b = _jn_filon(X_min, X_lo, X_hi, n_f, n_c, functools.partial(wX, sign=sign))
        root = np.sqrt(X_b**2 - X_min[..., None] ** 2)
        s.append(sign * root / kD[..., None])
        X.append(X_b)
        K.append(K_b * (X_b / (kD[..., None] * root))[..., None])  # ds/dX
    s, X, K = np.concatenate(s, axis=-1), np.concatenate(X, axis=-1), np.concatenate(K, axis=-2)
    s = np.maximum(s + s_s[..., None], 0)  # empty segments sit at s <= 0 with zero weight
    W = P[..., None] + s * Dp[..., None]
    al, be = g[:, None] * V[0] + s * D[0], g[:, None] * V[1] + s * D[1]
    return np.arctan2(W[1], W[0]), X, K, al, be, abs(V[0] * D[1] - V[1] * D[0]) + 0 * X, w_g


def _grid_rays(planes, e, ko, kl, d, x_s, n_d, n_f, n_c, weight):
    """A region on the line p = t e: a ray from W = 0 each way it goes, to t = U_MAX - 1 (a rescaling, Gauss-Legendre in
    ln(1 + t) near W = 0) or its end, where y > 0 keeps 1/r finite"""
    t_lo, t_hi = -np.inf, np.inf
    for n, h in planes:
        ne = n[0] * e[0] + n[1] * e[1]
        if ne > 0:
            t_hi = min(t_hi, h / ne)
        elif ne < 0:
            t_lo = max(t_lo, h / ne)
    E = e[0] * ko + e[1] * kl
    kappa = d * np.hypot(E[0], E[1])  # X at t = 1, one ray
    for sign, t_end in ((1, t_hi), (-1, -t_lo)):
        if t_end <= 0:
            continue
        rescale = np.isinf(t_end)
        X_max = kappa * (U_MAX - 1 if rescale else t_end * (1 - 1e-9))

        def wX(X, sign=sign):  # the weight (_P_along) along the ray
            return weight(sign * X / kappa[..., None] * e[0], sign * X / kappa[..., None] * e[1])

        X, K = _radial(X_max, x_s, n_d, n_f, n_c, kappa if rescale else None, wX)
        t = sign * X / kappa[..., None]
        alpha = np.arctan2(sign * E[1], sign * E[0])[..., None]  # one pole along the ray
        yield alpha, X, K, t * e[0], t * e[1], 1 / kappa[..., None] + 0 * X, np.ones(X.shape[:-1])


def _spectator(lam_max, n_s):
    """n_s composite Gauss-Legendre nodes (SPEC_PANEL a panel) in ln y on the last SPEC_EFOLDS e-folds below lam_max (a
    trailing axis), with their weights in y"""
    x, w = utils.leggauss(SPEC_PANEL)
    n_pan = max(n_s // SPEC_PANEL, 1)
    v = (np.arange(n_pan)[:, None] + (x + 1) / 2) / n_pan - 1  # ln(y / lam_max) / SPEC_EFOLDS, in [-1, 0]
    lam = lam_max[..., None] * np.exp(SPEC_EFOLDS * v.ravel())
    return lam, lam * np.tile(w, n_pan) * SPEC_EFOLDS / (2 * n_pan)


def _spectator_work(count, n_s, split):
    """Spectator work per radial node: separate sums when split, a product grid otherwise."""
    size = SPEC_PANEL * max(n_s // SPEC_PANEL, 1)
    return max(size * count, 1) if split else size**count


def _node_map(term, los_leg, d, al, be, e=None):
    """y of every node variable at (alpha, beta) and the Jacobian d^n (dr/dalpha dbeta, or dr/dt along e if the region
    is the line p = t e). W fixes only the ratio of a mode's two ends, so where both are integrated their scale lambda
    is a spectator - y_j = lambda in (0, min(1, 1/rho_j)], or the I1 layout's shared y_c in (0, min(1, rho_o, rho_l)].
    For each, y comes at lambda = 1 (the points it scales go as lambda) and the Jacobian without its lambda**k: returns
    y, the Jacobian and [(sides, the y it scales, its top, k)]"""
    aligned, o, l, I, C = _sides(term, los_leg)
    (ro, rl), D = _rho(aligned, al, be)
    if e is None:
        jac = np.abs(D[0][0] * D[1][1] - D[0][1] * D[1][0]) + 0 * al
    else:
        k = 1 if e == (0.0, 1.0) else 0  # the ratio that moves
        jac = np.abs(D[k][0] * e[0] + D[k][1] * e[1]) + 0 * al
    jac = jac * d ** len(_nodes(term, los_leg, False))

    rho = {o: ro, l: rl}
    y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
    spec = []
    if term.part_c == "same":
        if I[o] and I[l]:  # y_c = lambda, y_j = lambda / rho_j
            y["ca"] = y["cb"] = 1 + 0 * ro
            y[o], y[l] = 1 / ro, 1 / rl
            jac = jac / (ro * rl) ** 2
            spec.append(_Spectator("ab", ("a", "b", "ca", "cb"), np.minimum(1, np.minimum(ro, rl)), 2))
        elif I[o] or I[l]:
            j, k = (o, l) if I[o] else (l, o)
            y["ca"] = y["cb"] = rho[k]
            y[j] = rho[k] / rho[j]
            jac = jac * rho[k] / rho[j] ** 2
        else:  # on the diagonal, rho_o = rho_l
            y["ca"] = y["cb"] = ro
    else:
        for j in (o, l):
            if I[j] and C[j]:  # y_j = lambda, y_cj = rho_j lambda
                y[j], y["c" + j] = 1 + 0 * rho[j], rho[j]
                spec.append(_Spectator(j, (j, "c" + j), np.minimum(1, 1 / rho[j]), 1))
            elif I[j]:
                y[j] = 1 / rho[j]
                jac = jac / rho[j] ** 2
            elif C[j]:
                y["c" + j] = rho[j]
    return y, jac, sorted(spec)


def _spectate(y, jac, spec, n_s):
    """The spectators on trailing axes of y and the Jacobian (two flattened into one, none a size 1 axis) - for the
    brute sum over them"""
    ax = (...,) + (None,) * max(len(spec), 1)
    y = {j: v[ax] if np.ndim(v) else v for j, v in y.items()}
    jac = jac[ax]
    for i, spectator in enumerate(spec):
        lam, w = _spectator(spectator.top, n_s)
        if len(spec) == 2:
            lam, w = (v[..., :, None] if i == 0 else v[..., None, :] for v in (lam, w))
        for name in spectator.names:
            y[name] = y[name] * lam
        jac = jac * w * lam**spectator.power
    if len(spec) == 2:
        shape = np.broadcast_shapes(*(np.shape(v) for v in [*y.values(), jac]))
        y = {j: np.broadcast_to(v, shape).reshape(shape[:-2] + (-1,)) if np.ndim(v) else v for j, v in y.items()}
        jac = np.broadcast_to(jac, shape).reshape(shape[:-2] + (-1,))
    return y, jac


def _P_along(term, los_leg, d, kvec, cosmo_funcs, zz):
    """P(q_a) P(q_b) at (alpha, beta), of the modes no spectator moves (its own sum takes P exactly): the weight the
    Filon takes exactly, leaving its spline the rest of the integrand, which is smooth"""
    baseint = BaseInt(cosmo_funcs)

    def weight(al, be):
        y, _, spec = _node_map(term, los_leg, d, al, be)
        (r1, r2), coef, _ = solve_modes(los_leg, term.c, y)
        moved = {j for spectator in spec for j in spectator.sides}
        out = 1.0
        for j in "ab":
            if j not in moved:
                q = coef[j][0] * kvec[r1][..., None, None] + coef[j][1] * kvec[r2][..., None, None]
                out = out * baseint.pk(np.hypot(q[0], q[1]), zz)
        return np.broadcast_to(out, np.broadcast_shapes(np.shape(out), np.shape(al)))

    return weight


def _grids(term, los_leg, d, kvec, x_s, n_d, n_f, n_c, n_psi, cosmo_funcs, zz):
    """The term's grids as _project takes them - (pole angle, X, radial kernel, y, Jacobian, spectators, outer weight) -
    over its region (_region, _polygon) with the node positions of _node_map. Yield one piece at a time, leaving its
    spectators unexpanded for _project. n_psi: (edge rays, cone rays, strip lines)"""
    a, b = term.a, term.b
    aligned, o, l, _, _ = _sides(term, los_leg)
    ko, kl = (kvec[{"a": a, "b": b}[j]][..., None] for j in (o, l))
    planes, e, kinks = _region(term, los_leg)
    caps = _caps(term, los_leg)
    weight = _P_along(term, los_leg, d, kvec, cosmo_funcs, zz)
    args = (ko, kl, d, x_s, n_d, n_f, n_c)

    def pieces():
        if e is not None:  # a kink on the line is at W = 0, where its rays start anyway
            yield from _grid_rays(planes, e, *args, weight)
            return
        wedges = itertools.product(*([(n, 0.0), ((-n[0], -n[1]), 0.0)] for n in kinks))
        for kind, A, B in (piece for wedge in wedges for piece in _polygon(planes + list(wedge))):
            if kind == "edge":
                yield _grid_edge(A, B, aligned, *args, n_psi[0], caps, weight)
            elif kind == "cone":
                yield _grid_edge(A, B, aligned, *args, n_psi[1], caps, weight, cone=True)
            else:
                yield _grid_lines(A, B, aligned, *args, n_psi[2], caps, weight)

    for alpha, X, K, al, be, dA_dX, w_out in pieces():
        y, jac, spec = _node_map(term, los_leg, d, al, be, e)
        yield _Grid(alpha, X, K, y, jac * dA_dX, spec, w_out)


def _grid_0d():
    """No node: X = 0, the plain angular integral"""
    X, one = np.zeros((1, 1)), np.ones((1, 1))
    return _Grid(X, X, _jn_direct(X, one, 1), {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}, one, [], np.ones(1))
