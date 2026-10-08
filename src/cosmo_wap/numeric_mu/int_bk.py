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
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre, lpmv, sph_harm_y, spherical_jn

from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt

from . import accel
from .bk import PNG_SHAPES, get_Z1, get_Z2, los_cosines
from .kernels import K2, IntK1, IntK2, eval_terms, survey_scalars, term_weight
from .pk import _int_K1_basis
from .utils import split_kernels

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

# largest u = d / r on a node - q = u k is then far past K_MAX, where P(q) falls as q**-3
U_MAX = 1e6

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

# a spectator's range in ln y below its top - its integrand falls as y**2 or faster there (P(q) at q = Q/y past the
# turnover), so the rest is < 1e-8 - in panels of SPEC_PANEL Gauss-Legendre nodes, which resolve the BAO in P(Q/y)
SPEC_EFOLDS = 9
SPEC_PANEL = 16


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
        partial = [kern for kern in intg if kern in PARTIAL]
        if partial and not allow_partial:
            raise NotImplementedError(
                f"{partial} have only part (or none) of their second order kernel - "
                "allow_partial=True to use what there is"
            )
        out.append((local, intg, [SECOND_ORDER[kern] for kern in intg if kern in SECOND_ORDER]))
    return out


def _terms(fields, los_leg, orders, perms, los_mode):
    """(c, a, b, part_a, part_b, part_c) of every term with a number of integrated parts in orders - part S is local, I
    integrated (and, for c, a key of LAYOUTS)"""
    for c in perms:
        a, b = (i for i in range(3) if i != c)
        first = {i: [part for part, kerns in (("S", fields[i][0]), ("I", fields[i][1])) if kerns] for i in (a, b)}
        second = ["S"] if fields[c][0] else []
        if fields[c][2]:
            second += ["I"] if (c == los_leg and los_mode) else list(LAYOUTS)
        for pa in first[a]:
            for pb in first[b]:
                for pc in second:
                    if (pa == "I") + (pb == "I") + (pc != "S") in orders:
                        yield c, a, b, pa, pb, pc


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


def _positions(term, y, d, los_leg, los_mode):
    """Radial positions R of each mode's two ends in the phase. An integral on the line of sight comes with its own
    e^{ipr} (IntK1, so R = 0) or e^{ip(r-d)} (IntK2, R = d - which its y = 1 already gives)"""
    c, a, b, pa, pb, pc = term
    R = {j: d * yj for j, yj in y.items()}
    for j, i, part in (("a", a, pa), ("b", b, pb)):
        if part == "I" and i == los_leg and los_mode:
            R[j] = 0.0
    return R


def _outer(name, term, d, los_leg, n_y):
    """Gauss-Legendre nodes y and weights in r of an outer node variable - in y, or in ln u for those in det M, whose
    integrand falls only as a power of u out to u ~ K_MAX / k"""
    x, w = utils.leggauss(n_y)
    if name in _det_vars(los_leg, term[0]):
        y = np.exp(-(x + 1) / 2 * np.log(U_MAX))
        return y, d * y * w / 2 * np.log(U_MAX)  # dr = d dy = d e^{-s} ds, s = ln u
    return (x + 1) / 2, d * w / 2


def _setup(term, d, los_leg, los_mode, n_y, fixed=None):
    """The node variables not in fixed ({name: y}), each on its own trailing axis: returns their names, y, the
    Gauss-Legendre weights of all but the last (_outer), and the last one's Filon variable t with dr/dt (None if no
    nodes) - u on a log-spaced grid, or y on Gauss-Legendre nodes"""
    fixed = fixed or {}
    names = [name for name in _nodes(term, los_leg, los_mode) if name not in fixed]
    nv = len(names)

    y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
    for name, val in fixed.items():
        _set_y(y, name, val)
    weight, filon = 1.0, None
    for v, name in enumerate(names):
        shape = (n_y,) + (1,) * (nv - 1 - v)
        if v < nv - 1:
            yv, wv = _outer(name, term, d, los_leg, n_y)
            weight = weight * wv.reshape(shape)
        elif name in _det_vars(los_leg, term[0]):  # dr = d du / u**2
            t = np.geomspace(1, U_MAX, n_y)
            yv, filon = 1 / t, (t, d / t**2)
        else:
            yv = (utils.leggauss(n_y)[0] + 1) / 2
            filon = (yv, np.full(n_y, d))
        _set_y(y, name, yv.reshape(shape))
    return names, y, weight, filon


def _filon_weights(t, omega):
    """Weights W with sum(W * f, -1) = int f(t) e^{i omega t} dt, f linear between the nodes t (increasing along the
    last axis) and the phase exact - those of integration.filon_integrate, t and omega[..., None] broadcasting"""
    omega = np.asarray(omega)[..., None]
    dt = np.diff(t, axis=-1)
    E = np.exp(1j * omega * t)
    E_left, E_right = E[..., :-1], E[..., 1:]
    theta, dt = np.broadcast_arrays(omega * dt, dt)
    small = np.abs(theta) < 1e-3

    with np.errstate(divide="ignore", invalid="ignore"):  # omega = 0 entries are overwritten below
        inv_iw = 1.0 / (1j * omega)
        v0 = (E_right - E_left) * inv_iw
        v1 = (dt * E_right - v0) * inv_iw
        W_right = v1 / dt
        W_left = v0 - W_right

    if np.any(small):  # series in theta - leading order is the trapezoid weight dt/2
        th = theta[small]
        dt_E = dt[small] * E_left[small]
        W_right[small] = dt_E * (0.5 + 1j * th / 3 - th**2 / 8)
        W_left[small] = dt_E * (0.5 + 1j * th / 6 - th**2 / 24)

    pad = [(0, 0)] * (W_left.ndim - 1)
    return np.pad(W_left, pad + [(0, 1)]) + np.pad(W_right, pad + [(1, 0)])


def _filon(t, omega, f):
    """int f(t) e^{i omega t} dt over the last axis - see _filon_weights, omega broadcasting against f[..., 0]"""
    return np.sum(_filon_weights(t, np.broadcast_to(omega, f.shape[:-1])) * f, axis=-1)


def _first(part, i, q, mu, r, on_los, fields, cosmo_funcs, zz, deg, n_p):
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
    terms = [(name, sum(2 * (Qm - 1) if kern == "L" else 1 for kern in second)) for name in LAYOUTS[part]]
    return terms + [(name, second.count("L")) for name in PRODUCTS.get(part, ()) if "L" in second]


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


def _integrand(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, y, deg, n_p):
    """One term at the line of sight cosines mus = (mu1, mu2, mu3) and node positions y, which broadcast against each
    other - without the node weights. Returns the amplitude and the phase it multiplies"""
    c, a, b, pa, pb, pc = term
    d = cosmo_funcs.comoving_dist(zz)
    r = {j: d * yj for j, yj in y.items()}
    (r1, r2), coef, det = solve_modes(los_leg, c, y)

    def dot(u, v):
        return u[0] * v[0] * gram[r1, r1] + u[1] * v[1] * gram[r2, r2] + (u[0] * v[1] + u[1] * v[0]) * gram[r1, r2]

    q = {j: np.sqrt(dot(coef[j], coef[j])) for j in "ab"}
    p = {j: coef[j][0] * ks[r1] * mus[r1] + coef[j][1] * ks[r2] * mus[r2] for j in "ab"}  # q_j.x
    mu_q = {j: p[j] / q[j] for j in "ab"}
    cos_ab = dot(coef["a"], coef["b"]) / (q["a"] * q["b"])

    common = (fields, cosmo_funcs, zz)
    Fa = _first(pa, a, q["a"], mu_q["a"], r["a"], "a" not in _nodes(term, los_leg, los_mode), *common, deg, n_p)
    Fb = _first(pb, b, q["b"], mu_q["b"], r["b"], "b" not in _nodes(term, los_leg, los_mode), *common, deg, n_p)
    Fc = _second(pc, c, q["a"], q["b"], mu_q["a"], mu_q["b"], cos_ab, r, *common, deg)

    baseint = BaseInt(cosmo_funcs)
    R = _positions(term, y, d, los_leg, los_mode)
    phase = p["a"] * (R["a"] - R["ca"]) + p["b"] * (R["b"] - R["cb"])
    val = 2 * np.abs(det) ** -3.0 * baseint.pk(q["a"], zz) * baseint.pk(q["b"], zz) * Fa * Fb * Fc
    return val, phase


def _term(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, n_y, deg, n_p, fixed=None):
    """One term at the line of sight cosines mus = (mu1, mu2, mu3), summed over its node variables not in fixed - the
    last by Filon, the others by Gauss-Legendre in y, which only converges while their phase is resolved"""
    d = cosmo_funcs.comoving_dist(zz)
    names, y, weight, filon = _setup(term, d, los_leg, los_mode, n_y, fixed)
    nv = len(names)

    def ex(arr):
        return np.asarray(arr)[(...,) + (None,) * nv]

    ks, mus = [ex(k) for k in ks], [ex(mu) for mu in mus]
    gram = {key: ex(val) for key, val in gram.items()}
    val, phase = _integrand(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, los_mode, y, deg, n_p)
    val = val * weight
    if not nv:
        return val * np.exp(1j * phase)

    # Filon on the last axis: the phase is linear along it, so take out its slope and keep the rest in the amplitude
    t, dr_dt = filon
    phase = np.broadcast_to(phase, val.shape)
    omega = (phase[..., -1] - phase[..., 0]) / (t[-1] - t[0])
    val = _filon(t, omega, val * dr_dt * np.exp(1j * (phase - omega[..., None] * t)))
    return val.sum(axis=tuple(range(-(nv - 1), 0))) if nv > 1 else val


def _gram(k1, k2, k3, theta):
    """k_i.k_j"""
    k12 = k1 * k2 * np.cos(theta)
    gram = {(0, 0): k1**2, (1, 1): k2**2, (2, 2): k3**2, (0, 1): k12, (0, 2): -(k1**2) - k12, (1, 2): -(k2**2) - k12}
    gram.update({(j, i): val for (i, j), val in list(gram.items())})
    return gram


def get_mu_phi_int(
    mu,
    phi,
    kernels1,
    kernels2,
    kernels3,
    cosmo_funcs,
    k1,
    k2,
    k3,
    theta,
    zz,
    los_leg=2,
    orders=(1,),
    perms=(0, 1, 2),
    n_y=256,
    los_mode=True,
    allow_partial=False,
    deg=8,
    n_p=2000,
    max_size=2**22,
):
    """B(k1,k2,k3,mu,phi) from the terms with integrated parts, on the grid of numeric_mu.bk.get_mu_phi (same
    conventions: field i at k_i with kernelsi and tracer ti = i-1, mu and phi 1D on the last two axes). Pointwise only -
    its multipoles need get_multipoles_int's grids, see the module docstring.

    orders: which numbers of integrated parts to keep - 1 is ISS (with SS|I, kappa^(2) on the second order field),
        2 IIS, 3 III, and 0 the local tree (for checks).
    perms: which fields to take as the second order one.
    n_y: nodes per integral off the line of sight - see _setup.
    los_mode: False puts the line of sight field's integrals on nodes too rather than J_f(p) splines (for checks).
    max_size: elements per chunk - the mu axis is split so a term's (mu, phi, nodes) arrays stay below it."""
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    gram = _gram(k1, k2, k3, theta)

    tri_size = np.broadcast(k1, k2, k3, theta).size
    out = np.zeros(np.broadcast_shapes(*(np.shape(x) for x in (k1, k2, k3, theta)), (len(mu), len(phi))), dtype=complex)
    for term in _terms(fields, los_leg, orders, perms, los_mode):
        rows = max(1, max_size // (tri_size * len(phi) * n_y ** len(_nodes(term, los_leg, los_mode))))
        for start in range(0, len(mu), rows):
            sl = slice(start, start + rows)
            mus = los_cosines(mu[sl, None], phi[None, :], k1, k2, k3, theta)
            out[..., sl, :] += _term(
                term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, los_mode, n_y, deg, n_p
            )
    return out


def _kvec(k1, k2, theta):
    """The three sides as in-plane vectors (x along k1) - components on the first axis"""
    kvec = [np.stack(np.broadcast_arrays(k * np.cos(ang), k * np.sin(ang))) for k, ang in ((k1, 0.0), (k2, theta))]
    return kvec + [-(kvec[0] + kvec[1])]


def _phase_vector(term, y, d, los_leg, los_mode, kvec):
    """W with the phase of _integrand W.x, at scalar node positions y: sum_j (R_j - R_cj) q_j"""
    (r1, r2), coef, _ = solve_modes(los_leg, term[0], y)
    R = _positions(term, y, d, los_leg, los_mode)
    A, B = (sum(coef[j][m] * (R[j] - R["c" + j]) for j in "ab") for m in (0, 1))
    return A * kvec[r1] + B * kvec[r2]


def _pole(term, d, k1, k2, k3, theta, los_leg, los_mode, n_y, fixed):
    """In-plane angle from k1 of the vector the term's phase x.W varies along as its Filon variable does - the spike
    sits on the great circle perpendicular to it. With no node it is the line of sight field's own side"""
    kvec = _kvec(k1, k2, theta)
    names, _, _, _ = _setup(term, d, los_leg, los_mode, n_y, fixed)
    if not names:
        vec = kvec[los_leg]
    else:
        W = []
        for val in (1.0, 0.5):  # W is linear along the variable, so two points give its direction
            y = {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}
            for name, fval in (fixed or {}).items():
                _set_y(y, name, fval)
            _set_y(y, names[-1], val)
            W.append(_phase_vector(term, y, d, los_leg, los_mode, kvec))
        vec = W[0] - W[1]
    return np.arctan2(vec[1], vec[0])


def _pole_grid(n_mu, edges):
    """Composite Gauss-Legendre on [-1, 1], the same n_mu // (2 * (len(edges) + 1)) nodes on every panel between 0, the
    edges and 1 (and their mirror images) - so graded edges cluster the nodes at 0 over decades"""
    edges = np.concatenate(([0.0], np.asarray(edges, dtype=float), [1.0]))
    x, w = utils.leggauss(max(n_mu // (2 * (len(edges) - 1)), 2))
    half = [(b - a) / 2 * x + (a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    half_w = [(b - a) / 2 * w for a, b in zip(edges[:-1], edges[1:])]
    nodes, wts = np.concatenate(half), np.concatenate(half_w)
    return np.concatenate((-nodes[::-1], nodes)), np.concatenate((wts[::-1], wts))


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


def get_multipoles_int(
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
    n_mu=160,
    n_phi=16,
    edges=(1e-4, 1e-3, 1e-2, 0.1),
    los_leg=2,
    orders=(1,),
    perms=(0, 1, 2),
    n_y=256,
    los_mode=True,
    allow_partial=False,
    deg=8,
    n_p=2000,
):
    """(l, m) multipoles of the integrated terms, as numeric_mu.bk.get_multipoles: int dOmega conj(Ylm) B, each term on
    its own sphere grid (_sphere: n_mu x n_phi nodes, the cosine panels between edges). A term with two node variables
    is a loop over the outer one, its pole moving with it. Returns array of shape (len(lm), *triangle shape) - see
    get_mu_phi_int for the rest."""
    k3, theta = utils.get_theta_k3(k1, k2, k3, theta)
    k1, k2, k3, theta = utils.enable_broadcasting(k1, k2, k3, theta)
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    gram = _gram(k1, k2, k3, theta)
    d = cosmo_funcs.comoving_dist(zz)

    out = 0
    for term in _terms(fields, los_leg, orders, perms, los_mode):
        outer = _nodes(term, los_leg, los_mode)[:-1]
        grids = [_outer(name, term, d, los_leg, n_y) for name in outer]
        for idx in itertools.product(range(n_y), repeat=len(outer)):
            fixed = {name: grid[0][i] for name, grid, i in zip(outer, grids, idx)}
            wt = np.prod([grid[1][i] for grid, i in zip(grids, idx)])
            alpha = _pole(term, d, k1, k2, k3, theta, los_leg, los_mode, n_y, fixed)
            mus, (mu, phi), weights = _sphere(alpha, k1, k2, k3, theta, *_pole_grid(n_mu, edges), n_phi)
            arr = wt * _term(
                term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, los_mode, n_y, deg, n_p, fixed
            )
            if sigma is not None:
                arr = arr * np.exp(-(1 / 2) * ((k1 * mus[0]) ** 2 + (k2 * mus[1]) ** 2 + (k3 * mus[2]) ** 2) * sigma**2)
            out = out + np.array(
                [
                    2 * np.sum(weights * np.real(sph_harm_y(l, m, np.arccos(mu), phi)) * arr, axis=(-2, -1))
                    for l, m in lm
                ]
            )
    return out


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
    if term[5] == "same":
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
    out = []
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
        out.append((alpha, X, K, t * e[0], t * e[1], 1 / kappa[..., None] + 0 * X, np.ones(X.shape[:-1])))
    return out


def _spectator(lam_max, n_s):
    """n_s composite Gauss-Legendre nodes (SPEC_PANEL a panel) in ln y on the last SPEC_EFOLDS e-folds below lam_max (a
    trailing axis), with their weights in y"""
    x, w = utils.leggauss(SPEC_PANEL)
    n_pan = max(n_s // SPEC_PANEL, 1)
    v = (np.arange(n_pan)[:, None] + (x + 1) / 2) / n_pan - 1  # ln(y / lam_max) / SPEC_EFOLDS, in [-1, 0]
    lam = lam_max[..., None] * np.exp(SPEC_EFOLDS * v.ravel())
    return lam, lam * np.tile(w, n_pan) * SPEC_EFOLDS / (2 * n_pan)


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
    if term[5] == "same":
        if I[o] and I[l]:  # y_c = lambda, y_j = lambda / rho_j
            y["ca"] = y["cb"] = 1 + 0 * ro
            y[o], y[l] = 1 / ro, 1 / rl
            jac = jac / (ro * rl) ** 2
            spec.append(("ab", ("a", "b", "ca", "cb"), np.minimum(1, np.minimum(ro, rl)), 2))
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
                spec.append((j, (j, "c" + j), np.minimum(1, 1 / rho[j]), 1))
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
    for i, (_, names, top, k) in enumerate(spec):
        lam, w = _spectator(top, n_s)
        if len(spec) == 2:
            lam, w = (v[..., :, None] if i == 0 else v[..., None, :] for v in (lam, w))
        for name in names:
            y[name] = y[name] * lam
        jac = jac * w * lam**k
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
        (r1, r2), coef, _ = solve_modes(los_leg, term[0], y)
        moved = {j for sides, *_ in spec for j in sides}
        out = 1.0
        for j in "ab":
            if j not in moved:
                q = coef[j][0] * kvec[r1][..., None, None] + coef[j][1] * kvec[r2][..., None, None]
                out = out * baseint.pk(np.hypot(q[0], q[1]), zz)
        return np.broadcast_to(out, np.broadcast_shapes(np.shape(out), np.shape(al)))

    return weight


def _grids(term, los_leg, d, kvec, x_s, n_d, n_f, n_c, n_psi, n_s, split, cosmo_funcs, zz):
    """The term's grids as _project takes them - (pole angle, X, radial kernel, y, Jacobian, spectators, outer weight) -
    over its region (_region, _polygon) with the node positions of _node_map, its spectators on n_s nodes each: left to
    _integrand_split, or (split=False) on axes of y for _integrand. n_psi: (edge rays, cone rays, strip lines)"""
    a, b = term[1:3]
    aligned, o, l, _, _ = _sides(term, los_leg)
    ko, kl = (kvec[{"a": a, "b": b}[j]][..., None] for j in (o, l))
    planes, e, kinks = _region(term, los_leg)
    caps = _caps(term, los_leg)
    weight = _P_along(term, los_leg, d, kvec, cosmo_funcs, zz)
    args = (ko, kl, d, x_s, n_d, n_f, n_c)
    if e is not None:  # a kink on the line is at W = 0, where its rays start anyway
        pieces = _grid_rays(planes, e, *args, weight)
    else:
        pieces = []
        wedges = itertools.product(*([(n, 0.0), ((-n[0], -n[1]), 0.0)] for n in kinks))
        for kind, A, B in (piece for wedge in wedges for piece in _polygon(planes + list(wedge))):
            if kind == "edge":
                pieces.append(_grid_edge(A, B, aligned, *args, n_psi[0], caps, weight))
            elif kind == "cone":
                pieces.append(_grid_edge(A, B, aligned, *args, n_psi[1], caps, weight, cone=True))
            else:
                pieces.append(_grid_lines(A, B, aligned, *args, n_psi[2], caps, weight))
    out = []
    for alpha, X, K, al, be, dA_dX, w_out in pieces:
        y, jac, spec = _node_map(term, los_leg, d, al, be, e)
        if not (split and spec):
            (y, jac), spec = _spectate(y, jac, spec, n_s), []
        out.append((alpha, X, K, y, jac * (dA_dX if spec else dA_dX[..., None]), spec, w_out))
    return out


def _grid_0d():
    """No node: X = 0, the plain angular integral"""
    X, one = np.zeros((1, 1)), np.ones((1, 1, 1))
    return X, X, _jn_direct(X, one[..., 0], 1), {"a": 1.0, "b": 1.0, "ca": 1.0, "cb": 1.0}, one, [], np.ones(1)


def _re_ylm(l, m, mu, phi):
    """Re Y_lm(arccos mu, phi) as sph_harm_y's, through lpmv - half the cost"""
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
    return norm * lpmv(m, l, mu) * np.cos(m * phi)


def _integrand_split(term, fields, cosmo_funcs, zz, ks, mus, gram, los_leg, y, spec, n_s, deg, n_p):
    """_integrand with its spectators summed out (_node_map): a spectator lambda moves q_j = Q_j / lambda, the radial
    functions at r_j and r_cj and P(q_j), and nothing angular. So each factor it moves splits into a coefficient at
    lambda = 1 (IntK1 term by term, IntK2 term by term and key by key - Q_POWERS - with that radial function set to 1)
    and a function of lambda, and the sum over lambda (_spectator, n_s nodes) is done per node of the grid, not per
    angle. y at lambda = 1; returns the amplitude"""
    c, a, b, pa, pb, pc = term
    d = cosmo_funcs.comoving_dist(zz)
    (r1, r2), coef, det = solve_modes(los_leg, c, y)

    def dot(u, v):
        return u[0] * v[0] * gram[r1, r1] + u[1] * v[1] * gram[r2, r2] + (u[0] * v[1] + u[1] * v[0]) * gram[r1, r2]

    q = {j: np.sqrt(dot(coef[j], coef[j])) for j in "ab"}
    p = {j: coef[j][0] * ks[r1] * mus[r1] + coef[j][1] * ks[r2] * mus[r2] for j in "ab"}
    mu_q = {j: p[j] / q[j] for j in "ab"}
    cos_ab = dot(coef["a"], coef["b"]) / (q["a"] * q["b"])
    baseint = BaseInt(cosmo_funcs)
    group = {j: g for g, (sides, *_) in enumerate(spec) for j in sides}
    lam, w = zip(*(_spectator(top, n_s) for _, _, top, _ in spec))
    # lambda**k of the Jacobian and |det M|**-3, with one factor of det M a side
    w = [wg * lg ** (k - 3 * len(sides)) for wg, lg, (sides, _, _, k) in zip(w, lam, spec)]

    full = 2 * np.abs(det) ** -3.0  # the factors no spectator moves
    F, F_lam, P_lam = {}, {}, {}
    for j, i, part in (("a", a, pa), ("b", b, pb)):
        if j not in group:
            on_los = j not in _nodes(term, los_leg, False)
            Fj = _first(part, i, q[j], mu_q[j], y[j] * d, on_los, fields, cosmo_funcs, zz, deg, n_p)
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


def _project(term, fields, cosmo_funcs, zz, tri, lm, grid, los_leg, n_c, n_az, sigma, n_s, deg, n_p, max_size):
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
    n_y = n_s * len(spec) if spec else jac.shape[-1]  # the size of the spectator axis, or what stands in for it
    rows = max(1, max_size // (np.size(k1) * n_c // 2 * n_az * X.shape[-1] * n_y))
    out = 0
    for start in range(0, alpha.shape[-2], rows):
        sl = (..., slice(start, start + rows), slice(None))
        # the radial kernel times the Legendre projector at c and at -c
        Ks = K[sl + (slice(None),)]
        M = [np.einsum("...xn,nc->...cx", Ks, P[:, pos])[..., None, :] for P in (proj, proj[:, ::-1])]
        mus, (mu, phi), _ = _sphere(alpha[sl][..., None, None], k1, k2, k3, theta, c[pos], proj[0, pos], n_az)
        mus, mu, phi = [np.moveaxis(arr, -3, -1) for arr in mus], np.moveaxis(mu, -3, -1), np.moveaxis(phi, -3, -1)
        if spec:
            ys = {j: (val[sl][..., None, None, :] if np.ndim(val) else val) for j, val in y.items()}
            sp = [(sides, names, top[sl][..., None, None, :], k) for sides, names, top, k in spec]
            args = (term, fields, cosmo_funcs, zz, (k1, k2, k3), mus, gram, los_leg, ys, sp, n_s, deg, n_p)
            val = _integrand_split(*args) * jac[sl][..., None, None, :]
        else:
            ys = {j: (val[sl + (slice(None),)][..., None, None, :, :] if np.ndim(val) else val) for j, val in y.items()}
            ks, mus_s = [k[..., None] for k in (k1, k2, k3)], [m[..., None] for m in mus]
            gram_s = {key: val[..., None] for key, val in gram.items()}
            val, _ = _integrand(term, fields, cosmo_funcs, zz, ks, mus_s, gram_s, los_leg, False, ys, deg, n_p)
            val = np.sum(val * jac[sl + (slice(None),)][..., None, None, :, :], axis=-1)
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
    ISW, and the second order time delay and ISW themselves, are not in."""
    k3, theta = utils.get_theta_k3(k1, k2, k3, theta)
    k1, k2, k3, theta = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (k1, k2, k3, theta)))
    fields = _fields((kernels1, kernels2, kernels3), allow_partial)
    d = cosmo_funcs.comoving_dist(zz)
    kvec = _kvec(k1, k2, theta)
    tri = [x[..., None, None, None, None] for x in (k1, k2, k3, theta)]
    n_c = n_c or 10 + max(l for l, _ in lm)
    n_c += n_c % 2
    n_az = n_az or n_c // 2
    x_s = x_s or max(30.0, n_c + 10.0)
    if n_psi is None:
        kd = d * np.max((k1, k2, k3))
        n_psi = (max(16, int(np.ceil(kd / 4))), 16, max(16, int(np.ceil(kd / 2))))
    elif np.ndim(n_psi) == 0:
        n_psi = (n_psi,) * 3

    out = 0
    for term in _terms(fields, los_leg, orders, perms, False):
        if _nodes(term, los_leg, False):
            grids = _grids(term, los_leg, d, kvec, x_s, n_d, n_f, n_c, n_psi, n_s, split, cosmo_funcs, zz)
        else:
            grids = [_grid_0d()]
        for grid in grids:
            args = (los_leg, n_c, n_az, sigma, n_s, deg, n_p, max_size)
            out = out + _project(term, fields, cosmo_funcs, zz, tri, lm, grid, *args)
    return out if np.ndim(out) else np.zeros((len(lm),) + k1.shape)
