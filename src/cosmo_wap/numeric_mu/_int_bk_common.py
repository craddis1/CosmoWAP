"""Shared term geometry and kernel evaluation for the integrated bispectrum solvers."""

from typing import NamedTuple

import numpy as np

from cosmo_wap.lib.integrated import BaseInt

from . import accel
from .bk import PNG_SHAPES, get_Z1, get_Z2
from .kernels import K2, IntK1, IntK2, eval_terms, survey_scalars, term_weight
from .pk import _int_K1_basis
from .utils import split_kernels

# largest u = d / r on a node - q = u k is then far past K_MAX, where P(q) falls as q**-3
U_MAX = 1e6


class _Term(NamedTuple):
    """Second order field c, first order fields a < b, and the part on each."""

    c: int
    a: int
    b: int
    part_a: str
    part_b: str
    part_c: str


class _Spectator(NamedTuple):
    """The sides and positions scaled by lambda, its top and its power in the Jacobian."""

    sides: str
    names: tuple[str, ...]
    top: np.ndarray
    power: int


class _Grid(NamedTuple):
    """A term's radial grid, with the pole, node positions and weights for _project."""

    alpha: np.ndarray
    X: np.ndarray
    kernel: np.ndarray
    y: dict[str, np.ndarray | float]
    jac: np.ndarray
    spectators: list[_Spectator]
    outer_weights: np.ndarray


def _fields(kernels, allow_partial=False):
    """(local, first order integrated, second order integrated) kernel names of each field"""
    out = []
    for kerns in kernels:
        intg, local = split_kernels(kerns)
        png = [kern for kern in local if kern in PNG_SHAPES]
        if png:
            raise NotImplementedError(f"PNG kernels {png} are not in the integrated bispectrum yet")
        missing = [kern for kern in local if not hasattr(K2, kern)]
        if missing:
            raise NotImplementedError(f"no second order kernel for {missing} in the numeric bispectrum")
        partial = [kern for kern in intg if kern in IntK2.PARTIAL]
        if partial and not allow_partial:
            raise NotImplementedError(
                f"{partial} have only part (or none) of their second order kernel - "
                "allow_partial=True to use what there is"
            )
        out.append((local, intg, [IntK2.SECOND_ORDER[kern] for kern in intg if kern in IntK2.SECOND_ORDER]))
    return out


def _terms(fields, los_leg, orders, perms, los_mode):
    """(c, a, b, part_a, part_b, part_c) of every term with a number of integrated parts in orders - part S is local, I
    integrated (and, for c, a key of LAYOUTS)"""
    for c in perms:
        a, b = (i for i in range(3) if i != c)
        first = {i: [part for part, kerns in (("S", fields[i][0]), ("I", fields[i][1])) if kerns] for i in (a, b)}
        second = ["S"] if fields[c][0] else []
        if fields[c][2]:
            second += ["I"] if (c == los_leg and los_mode) else list(IntK2.LAYOUTS)
        for pa in first[a]:
            for pb in first[b]:
                for pc in second:
                    if (pa == "I") + (pb == "I") + (pc != "S") in orders:
                        yield _Term(c, a, b, pa, pb, pc)


def _nodes(term, los_leg, los_mode):
    """The quadrature node variables of a term - c is the one point both modes share in the I1 layout. Those in det M
    come last, so the innermost (Filon) one is a rescaling whenever there is one"""
    c, a, b, pa, pb, pc = term
    names = [j for j, i, part in (("a", a, pa), ("b", b, pb)) if part == "I" and not (i == los_leg and los_mode)]
    names += {"same": ["c"], "mix_a": ["ca"], "mix_b": ["cb"], "two": ["ca", "cb"]}.get(pc, [])
    return sorted(names, key=lambda name: name in _det_vars(los_leg, c))


def _det_vars(los_leg, c):
    """The node variables in det M - the phase is linear in 1/y for these and in y for the rest"""
    a, b = (i for i in range(3) if i != c)
    if c == los_leg:
        return {"a", "b"}
    return {"a", "cb", "c"} if los_leg == b else {"b", "ca", "c"}


def _set_y(y, name, val):
    if name == "c":
        y["ca"] = y["cb"] = val
    else:
        y[name] = val


def solve_modes(los_leg, c, y):
    """The modes from the transverse delta functions: q_j = alpha_j k_r1 + beta_j k_r2 over the two fields r1 < r2 off
    the line of sight. Returns ((r1, r2), {j: (alpha_j, beta_j)}, det M), y as in _setup"""
    a, b = (i for i in range(3) if i != c)

    def row(i):  # (M_ia, M_ib)
        if i == a:
            return y["a"], 0
        if i == b:
            return 0, y["b"]
        return -y["ca"], -y["cb"]

    r1, r2 = (i for i in range(3) if i != los_leg)
    (m11, m12), (m21, m22) = row(r1), row(r2)
    det = m11 * m22 - m12 * m21
    return (r1, r2), {"a": (m22 / det, -m12 / det), "b": (-m21 / det, m11 / det)}, det


def _mode_geometry(c, ks, mus, gram, los_leg, y):
    """Magnitudes, LOS projections and cosines of the solved modes, with det M."""
    (r1, r2), coef, det = solve_modes(los_leg, c, y)

    def dot(u, v):
        return u[0] * v[0] * gram[r1, r1] + u[1] * v[1] * gram[r2, r2] + (u[0] * v[1] + u[1] * v[0]) * gram[r1, r2]

    q = {j: np.sqrt(dot(coef[j], coef[j])) for j in "ab"}
    p = {j: coef[j][0] * ks[r1] * mus[r1] + coef[j][1] * ks[r2] * mus[r2] for j in "ab"}  # q_j.x
    mu_q = {j: p[j] / q[j] for j in "ab"}
    cos_ab = dot(coef["a"], coef["b"]) / (q["a"] * q["b"])
    return q, p, mu_q, cos_ab, det


def _positions(term, y, d, los_leg, los_mode):
    """Radial positions R of each mode's two ends in the phase. An integral on the line of sight comes with its own
    e^{ipr} (IntK1, so R = 0) or e^{ip(r-d)} (IntK2, R = d - which its y = 1 already gives)"""
    c, a, b, pa, pb, pc = term
    R = {j: d * yj for j, yj in y.items()}
    for j, i, part in (("a", a, pa), ("b", b, pb)):
        if part == "I" and i == los_leg and los_mode:
            R[j] = 0.0
    return R


def _first(part, i, q, mu, r, on_los, fields, cosmo_funcs, zz, *, deg=8, n_p=2000):
    """First order kernel of field i at its mode - an integrated one on the line of sight already integrated over r"""
    local, intg, _ = fields[i]
    if part == "S":
        return get_Z1(local, cosmo_funcs, zz, mu, q, ti=i)
    if on_los:  # int_0^d dr e^{ipr} K - numeric_mu.pk's IS spline
        entries, spline = _int_K1_basis(intg, cosmo_funcs, zz, deg=deg, n_p=n_p)
        scal = survey_scalars(cosmo_funcs, zz, ti=i)
        weights = np.array([term_weight(dict(wt), scal) for _, _, wt in entries])
        return accel.kernel_sum(spline, [(m, n) for m, n, _ in entries], weights, q, mu)
    return sum(
        eval_terms(getattr(IntK1, kern)(r, cosmo_funcs, zz, ti=i), mu, q, cosmo_funcs, zz, ti=i) for kern in intg
    )


def _layout_terms(part, second, Qm):
    """The IntK2 terms of a layout, each with its weight in the field's second order kernels: the kappa^(2) ones
    2(Q - 1) for IntK2.L and 1 for kappa2, the products of first order fields (PRODUCTS) 1 for IntK2.L alone"""
    terms = [(name, sum(2 * (Qm - 1) if kern == "L" else 1 for kern in second)) for name in IntK2.LAYOUTS[part]]
    return terms + [(name, second.count("L")) for name in IntK2.PRODUCTS.get(part, ()) if "L" in second]


def _second(part, c, qa, qb, mua, mub, cos_ab, r, fields, cosmo_funcs, zz, deg):
    """Second order kernel of field c at (-q_a, -q_b). Off the line of sight IntK2's terms get the radial functions at
    the nodes in place of their J_f(p) - zero for a mode at the source, so I2, I3, I6 and I7 keep the ordering
    integrated here"""
    local, _, second = fields[c]
    args = (cosmo_funcs, zz, -mua, -mub, qa, qb, cos_ab)
    if part == "S":
        return get_Z2(local, *args, ti=c)
    if part == "I":
        return sum(getattr(IntK2, name)(*args, ti=c, deg=deg) for name in second)

    los = IntK2.source(cosmo_funcs, zz, -mua, -mub, qa, qb, ti=c)
    if part == "same":
        los["J12"] = IntK2.radial(cosmo_funcs, zz, r["ca"])
    else:
        Ja = IntK2.radial(cosmo_funcs, zz, r["ca"]) if part in ("mix_a", "two") else None
        Jb = IntK2.radial(cosmo_funcs, zz, r["cb"]) if part in ("mix_b", "two") else None
        zero = dict.fromkeys(Ja if Ja is not None else Jb, 0)
        los["J"] = [zero if Ja is None else Ja, zero if Jb is None else Jb]
    terms = _layout_terms(part, second, cosmo_funcs.survey[c].Q(zz))
    return sum(wt * getattr(IntK2, name)(*args, ti=c, los=los) for name, wt in terms)


def _integrand(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, y, deg=8, n_p=2000):
    """One term at the line of sight cosines mus = (mu1, mu2, mu3) and node positions y, which broadcast against each
    other - without the node weights. Returns the amplitude and the phase it multiplies"""
    c, a, b, pa, pb, pc = term
    d = cosmo_funcs.comoving_dist(zz)
    r = {j: d * yj for j, yj in y.items()}
    q, p, mu_q, cos_ab, det = _mode_geometry(c, ks, mus, gram, los_leg, y)

    common = (fields, cosmo_funcs, zz)
    nodes = _nodes(term, los_leg, los_mode)
    Fa = _first(pa, a, q["a"], mu_q["a"], r["a"], "a" not in nodes, *common, deg=deg, n_p=n_p)
    Fb = _first(pb, b, q["b"], mu_q["b"], r["b"], "b" not in nodes, *common, deg=deg, n_p=n_p)
    Fc = _second(pc, c, q["a"], q["b"], mu_q["a"], mu_q["b"], cos_ab, r, *common, deg)

    baseint = BaseInt(cosmo_funcs)
    R = _positions(term, y, d, los_leg, los_mode)
    phase = p["a"] * (R["a"] - R["ca"]) + p["b"] * (R["b"] - R["cb"])
    val = 2 * np.abs(det) ** -3.0 * baseint.pk(q["a"], zz) * baseint.pk(q["b"], zz) * Fa * Fb * Fc
    return val, phase


def _gram(k1, k2, k3, theta):
    """k_i.k_j"""
    k12 = k1 * k2 * np.cos(theta)
    gram = {(0, 0): k1**2, (1, 1): k2**2, (2, 2): k3**2, (0, 1): k12, (0, 2): -(k1**2) - k12, (1, 2): -(k2**2) - k12}
    gram.update({(j, i): val for (i, j), val in list(gram.items())})
    return gram


def _kvec(k1, k2, theta):
    """The three sides as in-plane vectors (x along k1) - components on the first axis"""
    kvec = [np.stack(np.broadcast_arrays(k * np.cos(ang), k * np.sin(ang))) for k, ang in ((k1, 0.0), (k2, theta))]
    return kvec + [-(kvec[0] + kvec[1])]


def _sphere(alpha, k1, k2, k3, theta, c_pole, w_pole, n_phi):
    """Nodes on the half sphere with the pole at angle alpha in the triangle plane: c_pole (weights w_pole) in its
    cosine and the midpoint rule over the azimuth from the plane in [0, pi] - B is even about the plane. Returns the
    line of sight cosines, numeric_mu.bk's (mu, phi) of each node and the weights"""
    az = np.pi * (np.arange(n_phi) + 0.5) / n_phi
    cp, sp = c_pole[:, None], np.sqrt(1 - c_pole[:, None] ** 2)
    # x along k1, y in the plane, z the normal
    dx = cp * np.cos(alpha) - sp * np.cos(az) * np.sin(alpha)
    dy = cp * np.sin(alpha) + sp * np.cos(az) * np.cos(alpha)
    dz = sp * np.sin(az)
    mu1 = dx
    mu2 = dx * np.cos(theta) + dy * np.sin(theta)
    mu3 = -(k1 * mu1 + k2 * mu2) / k3
    return (mu1, mu2, mu3), (dx, np.arctan2(dz, dy)), w_pole[:, None] * (np.pi / n_phi)
