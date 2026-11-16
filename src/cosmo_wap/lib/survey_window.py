"""Survey window convolution of the power spectrum multipoles, as one matrix on the P_l' of a model: P̂ = W P.

In configuration space the window multiplies. For its multipoles Q_L(s) (from the randoms' pair counts, see
window_multipoles)

    xî_l(s) = sum_{l', L} (2l+1) (l l' L; 0 0 0)^2 xi_l'(s) Q_L(s),

and P̂_l(k) = 4 pi (-i)^l int ds s^2 xî_l(s) j_l(ks) (Wilson et al. 2017, arXiv:1511.07799; Beutler et al. 2017,
arXiv:1607.03150). Every step is linear in P, so they compose into a matrix on P_l' at the nodes k_in (Beutler &
McDonald 2021, arXiv:2106.06324) - built once per window, then a matvec. Q_L vanishes beyond the survey, so the s
integral is finite and needs none of lib/hankel.py's tapers.

Wide-angle orders (Beutler, Castorina & Zhang 2019, arXiv:1810.05051): xi_l(s, d) = sum_n (s/d)^n xi_l^(n)(s), and
order n sees its own window Q_L^(n), the pair counts weighted by (s/d)^n. The wide-separation terms carry an explicit
1/d^n, so the order-n part of P at d gives xi^(n) = (d/s)^n xi^[n]. An order with no Q^(n) of its own uses Q^(0) -
the effective-distance approximation, the same as adding it to order 0.

Integral constraints (de Mattia & Ruhlmann-Kleider 2019, arXiv:1904.08851 - dMR19): with the mean density (global IC)
or n(z) (radial IC) taken from the data itself, the survey cannot see the modes those fix,

    xî^cic_l = xî_l - IC^{d,ic}_l - IC^{ic,d}_l + IC^{ic,ic}_l,    IC^{i,j}_l(s) = int D^2 dD sum_p 4pi/(2p+1) xi_p(D) W^{i,j}_lp(s, D)

(their eqs 2.17-2.20), with W^{i,j} 3- and 4-point windows of the randoms. For the global IC W^{ic,ic} is separable
and needs only Q_L (their eq 2.22); keeping only -IC^{ic,ic} is the usual P̂_l(k) - P̂_0(0) W_l(k)/W_0(0) of Beutler
et al. 2017, which dMR19 find accurate for both the global and radial ICs. Their other windows took ~1500 CPU hours
from randoms, so they come in as arrays. A constant in xi_0 - what the k^-4 (potential) terms make IR divergent, see
hankel's k_ir - windows to const x Q_l(s), which any IC removes exactly, so with one P̂ does not depend on k_in[0].
"""

from math import factorial

import numpy as np
from numpy.polynomial import legendre
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from cosmo_wap.lib import hankel


def coupling(ell, ell_in, L):
    """(2l+1) (l l' L; 0 0 0)^2 - so L_l'(mu) L_L(mu) = sum_l coupling(l, l', L) L_l(mu)"""
    J = ell + ell_in + L
    if J % 2 or L > ell + ell_in or L < abs(ell - ell_in):
        return 0.0
    g = J // 2
    ratio = factorial(J - 2 * ell) * factorial(J - 2 * ell_in) * factorial(J - 2 * L) / factorial(J + 1)
    return (
        (2 * ell + 1) * ratio * (factorial(g) // (factorial(g - ell) * factorial(g - ell_in) * factorial(g - L))) ** 2
    )


def window_multipoles(s_edges, mu_edges, RR, norm, Ls=(0, 2, 4, 6, 8)):
    """Q_L(s) at the centres of s_edges from the randoms' pair counts RR, shape (len(s_edges)-1, len(mu_edges)-1) -
    mu against the estimator's line of sight, and the pairs weighted by (s/d)^n for wide-angle order n.

    norm: the estimator's normalisation A = alpha^2 sum_i w_i n_i (dMR19 eq 3.3) in RR's units, so Q_0(s -> 0) ~ 1;
          the limit itself is ill-defined with finite randoms (their section 3.2). The same for every order.
    mu_edges from 0 fold mu and -mu together - even L only."""
    s_edges, mu_edges, RR = (np.asarray(a, dtype=float) for a in (s_edges, mu_edges, RR))
    if RR.shape != (len(s_edges) - 1, len(mu_edges) - 1):
        raise ValueError(f"RR has shape {RR.shape} - expected (len(s_edges)-1, len(mu_edges)-1).")
    if mu_edges[0] >= 0 and any(L % 2 for L in Ls):
        raise ValueError("mu_edges from 0 fold the pairs - odd L need mu over [-1, 1].")
    shell = 4 * np.pi / 3 * np.diff(s_edges**3)
    Q = {}
    for L in Ls:
        bin_avg = np.diff(legendre.legval(mu_edges, legendre.legint(np.eye(L + 1)[L]))) / np.diff(mu_edges)
        Q[L] = (2 * L + 1) * (RR @ bin_avg) / (shell * norm)
    return (s_edges[1:] + s_edges[:-1]) / 2, Q


def windowed_xi(xi, Q, ells, s=None, d=None):
    """xî_l(s), shape (len(ells), len(s)), from xi = {l': xi_l'(s)} and Q = {L: Q_L(s)} on the same s - or keyed
    (l', n) and (L, n) by wide-angle order, which needs s and d. No integral constraint - see SurveyWindow.xi."""
    xi, Q = _by_order(xi), _by_order(Q)
    Qn = _order_windows(Q, None if s is None else np.asarray(s, dtype=float), d, {n for _, n in xi})
    zero = np.zeros_like(next(iter(xi.values())))
    return np.array(
        [
            sum((coupling(ell, lp, L) * x * q for (lp, n), x in xi.items() for L, q in Qn[n].items()), zero)
            for ell in _as_list(ells)
        ]
    )


class SurveyWindow:
    """The window as a matrix, built once: i^l P̂_l(k) = matrix @ (i^l' P_l'(k_in)) - real, flattened (ells, k) by
    (orders, ells_in, k_in). Call it on P from a model; Model.pk(..., window=) does that.

    s, Q:      the window multipoles on s, {L: Q_L(s)}, or {(L, n): Q_L^(n)(s)} by wide-angle order - normalised as
               window_multipoles. s from ~0 to past where Q vanishes; Q is 0 beyond s[-1]
    ells:      the output multipoles
    k, k_edges: the output wavenumbers - or bin edges, for k^2-weighted shell averages
    ells_in:   the model's multipoles - default every l' that couples to ells through Q or the integral constraint
    k_in:      the nodes the model is computed on, splined between in ln k - default 256 log-spaced from 1e-4 to
               k[-1] + 0.25 h/Mpc
    d:         the effective comoving distance [Mpc/h], for wide-angle orders
    ic:        the integral constraint. 'global' (default): -IC^{ic,ic} from Q alone. None for none. Or a dict of
               dMR19's windows {(l, p): W^{i,j}_lp(s, D)}, normalised by A as Q, on the grid 's' (both axes): 'ic_ic'
               (default the global one from Q), 'd_ic' and 'ic_d' (default d_ic - the same for even l with the
               midpoint line of sight). Without d_ic only -IC^{ic,ic}, their approximation. The IC acts on xi_l'
               summed over orders
    ds, dk:    quadrature steps in s and k' - default pi/(2 (k[-1] + k_in[-1])) and pi/(4 s[-1])
    n_gl:      Gauss-Legendre nodes per k bin - default enough for the phase of j_l(ks) across a bin at s[-1]"""

    def __init__(
        self, s, Q, ells=(0, 2, 4), k=None, k_edges=None, ells_in=None, k_in=None, d=None, ic="global",
        ds=None, dk=None, n_gl=None,
    ):  # fmt: skip
        self.s = np.asarray(s, dtype=float)
        self.Q = _by_order(Q)
        if (0, 0) not in self.Q:
            raise ValueError("Q needs Q_0 of order 0 - the window's monopole.")
        if (k is None) == (k_edges is None):
            raise ValueError("Pass one of k (wavenumbers) or k_edges (bin edges).")
        self.ells = _as_list(ells)
        self.k_edges = None if k_edges is None else np.asarray(k_edges, dtype=float)
        self.k = np.asarray(k, dtype=float) if k_edges is None else (self.k_edges[1:] + self.k_edges[:-1]) / 2
        Ls = {L for L, _ in self.Q}
        self.ells_in = (
            sorted({lp for l in self.ells for L in Ls for lp in range(abs(l - L), l + L + 1) if coupling(l, lp, L)})
            if ells_in is None
            else _as_list(ells_in)
        )
        if ells_in is None and ic is not None:
            needed = set(self.ells_in)
            if not isinstance(ic, dict) or "ic_ic" not in ic:
                needed.update(Ls)
            if isinstance(ic, dict):
                for name in ("ic_ic", "d_ic", "ic_d"):
                    needed.update(p for l, p in ic.get(name, {}) if l in self.ells)
            self.ells_in = sorted(needed)
        k_top = self.k[-1] if k_edges is None else self.k_edges[-1]
        self.k_in = np.geomspace(1e-4, k_top + 0.25, 256) if k_in is None else np.asarray(k_in, dtype=float)
        self.d = d
        self.orders = sorted({n for _, n in self.Q})
        self.ic = ic

        s_max, k_hi = self.s[-1], self.k_in[-1]
        sf = np.linspace(0, s_max, int(np.ceil(s_max / (ds or np.pi / (2 * (k_top + k_hi))))) + 1)
        kf = np.linspace(self.k_in[0], k_hi, int(np.ceil((k_hi - self.k_in[0]) / (dk or np.pi / (4 * s_max)))) + 1)
        ws = _trapezoid_weights(sf) * sf**2
        spline = CubicSpline(np.log(self.k_in), np.eye(len(self.k_in)))(np.log(kf))  # P(kf) = spline @ P(k_in)
        wk = (_trapezoid_weights(kf) * kf**2 / (2 * np.pi**2))[:, None] * spline
        self._hankel = kf, wk
        F = {lp: _bessel_dot(lp, sf, kf, wk) for lp in self.ells_in}  # xi_l'(sf) = F @ (i^l' P_l'(k_in))
        if k_edges is None:
            B = {l: 4 * np.pi * spherical_jn(l, np.outer(self.k, sf)) * ws for l in self.ells}
        else:
            n_gl = n_gl or 16 + int(np.diff(self.k_edges).max() * s_max)
            B = {l: 4 * np.pi * hankel.bessel_kernel(l, sf, r_edges=self.k_edges, n_gl=n_gl) * ws for l in self.ells}

        on_sf = lambda q: np.interp(sf, self.s, q, right=0)
        Qn = {
            n: {L: on_sf(q) for L, q in Qs.items()} for n, Qs in _order_windows(self.Q, self.s, d, self.orders).items()
        }
        shape_in = (len(self.orders), len(self.ells_in), len(self.k_in))
        M = np.zeros((len(self.ells), len(self.k)) + shape_in)
        for i, l in enumerate(self.ells):
            for o, n in enumerate(self.orders):
                for j, lp in enumerate(self.ells_in):
                    q = sum(coupling(l, lp, L) * QL for L, QL in Qn[n].items())
                    if np.any(q):
                        M[i, :, o, j] = B[l] @ (q[:, None] * F[lp])
        M = M.reshape(len(self.ells), len(self.k), -1)

        sn = np.array([np.full(len(self.k), float(l == 0)) for l in self.ells])
        self._ic = None if ic is None else self._ic_terms(ic, sf, ws, Qn, F, shape_in)
        if self._ic is not None:
            glo, s_ic, G = self._ic["global"], self._ic["s"], self._ic["G"]
            if glo:
                t = np.array([B[l] @ Qn[0].get(l, 0 * sf) for l in self.ells]) / self._ic["V"]
                M -= glo * t[:, :, None] * self._ic["r0"]
            if G:
                interp = _interp_matrix(sf, s_ic)
                for i, l in enumerate(self.ells):
                    if l in G:
                        M[i] -= (interp.T @ B[l].T).T @ G[l]
            sn = sn - t if "ic_ic" not in self._ic["kernels"] else None  # dMR19 section 3.3, unweighted
        self.matrix = M.reshape(len(self.ells) * len(self.k), -1)
        self.shot_noise = None if sn is None else (-1j) ** np.array(self.ells)[:, None] * sn

    @classmethod
    def from_matrix(cls, matrix, k_in, ells_in, k, ells, shot_noise=None):
        """A window matrix from elsewhere (pypower, a data release, ...) acting on P_l' directly: P̂ = matrix @ P,
        flattened (ells, k) by (ells_in, k_in). One wide-angle order and no Q, so no xi."""
        self = cls.__new__(cls)
        self.ells, self.ells_in = _as_list(ells), _as_list(ells_in)
        self.k, self.k_edges, self.k_in = np.asarray(k, dtype=float), None, np.asarray(k_in, dtype=float)
        self.s = self.Q = self.d = self.ic = self._ic = None
        self.orders = [0]
        phase = np.outer(
            np.repeat(1j ** np.array(self.ells), len(self.k)),
            np.repeat((-1j) ** np.array(self.ells_in), len(self.k_in)),
        )
        M = phase * np.asarray(matrix)
        if np.abs(M.imag).max() > 1e-12 * np.abs(M).max():
            raise ValueError("matrix mixes even and odd multipoles with the wrong phase - see the class docstring.")
        self.matrix = M.real
        self.shot_noise = shot_noise
        return self

    def __call__(self, P):
        """P̂_l on k, shape (len(ells), len(k)), complex as pk_func. P: P_l' on k_in, shape (len(ells_in), len(k_in)),
        or {n: P^[n]} by wide-angle order - orders the window has no Q^(n) for go in with order 0"""
        y = (self.matrix @ self._x(P)).reshape(len(self.ells), len(self.k))
        return (-1j) ** np.array(self.ells)[:, None] * y

    def xi(self, xi=None, r=None, P=None, k_split=None):
        """xî_l(r), shape (len(ells), len(r)) - the windowed correlation function, P̂'s Hankel pair, r default s.
        xi: xi_l'(r) for ells_in, by order as P in __call__. P: P_l' on k_in, for the integral constraint - its terms
        integrate xi over the whole window, so they come from P through the matrix's own maps.
        With xi omitted, P supplies xi too, using the matrix's k quadrature - the same IR convention as the IC, but cut
        sharply at k_in[-1], which rings through xi at every r (10-30% for the default k_in at 10-100 Mpc/h).
        k_split: xi is only P's part above k_split, P (1 - S) with S = hankel.window(k, k_split, 2 k_split), from a
        converged transform, and P S goes in through the matrix's quadrature - so the IR constants still cancel with
        the IC (Model.xi(..., window=))."""
        if self.Q is None:
            raise ValueError("No Q - a window from from_matrix has no xi.")
        r = self.s if r is None else np.atleast_1d(np.asarray(r, dtype=float))
        parts = [] if xi is None else list(xi.items()) if isinstance(xi, dict) else [(0, xi)]
        if xi is None or k_split is not None:
            if P is None:
                raise ValueError("Pass xi, or P on k_in to compute xi through the window's transform.")
            kf, wk = self._hankel
            if k_split is not None:
                if 2 * k_split > self.k_in[-1]:
                    raise ValueError(f"k_split = {k_split} needs k_in up to 2 k_split - it stops at {self.k_in[-1]}.")
                wk = hankel.window(kf, k_split, 2 * k_split)[:, None] * wk
            F = {lp: _bessel_dot(lp, r, kf, wk) for lp in self.ells_in}
            x = self._x(P).reshape(len(self.orders), len(self.ells_in), len(self.k_in))
            parts += [
                (n, np.array([F[lp] @ xn[j] for j, lp in enumerate(self.ells_in)])) for n, xn in zip(self.orders, x)
            ]
        merged = {}
        for n, x in parts:
            m = n if n in self.orders else 0
            merged[m] = merged.get(m, 0) + np.asarray(x)
        Qr = {key: np.interp(r, self.s, q, right=0) for key, q in self.Q.items()}
        Qn = {
            n: {L: np.interp(r, self.s, q, right=0) for L, q in Qs.items()}
            for n, Qs in _order_windows(self.Q, self.s, self.d, merged).items()
        }
        out = sum(windowed_xi(dict(zip(self.ells_in, x)), Qn[n], self.ells) for n, x in merged.items())
        if self._ic is None:
            return out
        if P is None:
            raise ValueError("The integral constraint needs P on k_in - pass P, as for calling the window.")
        x = self._x(P)
        glo, s_ic, G = self._ic["global"], self._ic["s"], self._ic["G"]
        for i, l in enumerate(self.ells):
            if glo and (l, 0) in Qr:
                out[i] -= glo * (self._ic["r0"] @ x) / self._ic["V"] * Qr[(l, 0)]
            if l in G:
                out[i] -= np.interp(r, s_ic, G[l] @ x, right=0)
        return out

    def _x(self, P):
        """the real input vector: i^l' P_l', by order"""
        x = np.zeros((len(self.orders), len(self.ells_in), len(self.k_in)))
        for n, Pn in P.items() if isinstance(P, dict) else [(0, P)]:
            Pn = np.asarray(Pn)
            if Pn.shape != x.shape[1:]:
                raise ValueError(f"P has shape {Pn.shape} - expected (len(ells_in), len(k_in)) = {x.shape[1:]}.")
            x[self.orders.index(n) if n in self.orders else 0] += np.real(1j ** np.array(self.ells_in)[:, None] * Pn)
        return x.ravel()

    def _ic_terms(self, ic, sf, ws, Qn, F, shape_in):
        """The IC in configuration space, IC_l(s) = glo Q_l(s) (r0 @ x)/V + interp(s_ic -> s) G_l @ x: the global
        -IC^{ic,ic} as a rank one (r0 @ x = P̂_0(0), V = 4 pi int s^2 Q_0), and the kernels' sum on their grid"""
        if ic == "global":
            ic = {}
        elif not isinstance(ic, dict):
            raise ValueError(f"ic must be 'global', None or a dict of windows - not {ic!r}.")
        kernels = {name: ic[name] for name in ("ic_ic", "d_ic", "ic_d") if name in ic}
        if "ic_d" in kernels and "d_ic" not in kernels:
            raise ValueError("ic_d without d_ic - pass both cross terms, or d_ic alone for even l.")
        exact = "d_ic" in kernels
        if exact:
            kernels.setdefault("ic_d", kernels["d_ic"])
            # the global ic_ic flips sign - an l short of a term is worse than none
            for name, windows in kernels.items():
                missing = sorted(set(self.ells) - {l for l, _ in windows})
                if missing:
                    raise ValueError(f"The exact IC needs {name} windows for every output l - none for l = {missing}.")
        out = dict(kernels=kernels, s=None, G={})
        out["global"] = 0 if "ic_ic" in kernels else (-1 if exact else 1)
        if out["global"]:
            out["V"] = 4 * np.pi * ws @ Qn[0][0]
            r0 = np.zeros(shape_in)
            for o, n in enumerate(self.orders):
                for j, lp in enumerate(self.ells_in):
                    if lp in Qn[n]:  # xî_0 = sum_l' xi_l' Q_l' / (2l'+1)
                        r0[o, j] = 4 * np.pi * ws @ (Qn[n][lp][:, None] * F[lp]) / (2 * lp + 1)
            out["r0"] = r0.ravel()
        if kernels:
            if "s" not in ic:
                raise ValueError("The IC windows need their grid, ic['s'].")
            s_ic = out["s"] = np.asarray(ic["s"], dtype=float)
            sign = {"ic_ic": -1 if exact else 1, "d_ic": 1, "ic_d": 1}
            interp_T = _interp_matrix(sf, s_ic).T
            R = {p: interp_T @ (4 * np.pi / (2 * p + 1) * ws[:, None] * F[p]) for p in self.ells_in}
            for name, windows in kernels.items():
                for (l, p), K in windows.items():
                    if l not in self.ells:
                        continue
                    if p not in R:
                        raise ValueError(f"The {name} window ({l}, {p}) needs l' = {p} in ells_in.")
                    block = np.zeros((len(s_ic),) + shape_in)
                    block[:, :, self.ells_in.index(p)] = (np.asarray(K) @ R[p])[:, None]
                    out["G"][l] = out["G"].get(l, 0) + sign[name] * block.reshape(len(s_ic), -1)
        return out


def _bessel_dot(ell, s, k, w, chunk=2**22):
    """j_l(s k) @ w, in chunks of k so the (s, k) array stays small"""
    step = max(1, chunk // len(s))
    return sum(spherical_jn(ell, np.outer(s, k[i : i + step])) @ w[i : i + step] for i in range(0, len(k), step))


def _trapezoid_weights(x):
    w = np.zeros_like(x)
    w[1:] += np.diff(x) / 2
    w[:-1] += np.diff(x) / 2
    return w


def _interp_matrix(x_new, x):
    """linear interpolation from x onto x_new as a sparse matrix - constant below x[0], 0 above x[-1]"""
    i = np.clip(np.searchsorted(x, x_new) - 1, 0, len(x) - 2)
    t = np.clip((x_new - x[i]) / (x[i + 1] - x[i]), 0, 1)
    inside = x_new <= x[-1]
    rows = np.arange(len(x_new))
    return sparse.csr_matrix(
        (np.concatenate([(1 - t) * inside, t * inside]), (np.concatenate([rows, rows]), np.concatenate([i, i + 1]))),
        shape=(len(x_new), len(x)),
    )


def _by_order(windows):
    """{l: a} or {(l, n): a} -> {(l, n): a}"""
    return {(key if isinstance(key, tuple) else (key, 0)): np.asarray(a, dtype=float) for key, a in windows.items()}


def _order_windows(Q, s, d, orders):
    """{n: {L: Q_L^(n) (d/s)^n}} - or Q^(0) for an order with no windows of its own"""
    out = {}
    for n in sorted(orders):
        own = {L: q for (L, m), q in Q.items() if m == n}
        if n == 0 or not own:
            out[n] = {L: q for (L, m), q in Q.items() if m == 0}
            continue
        if d is None or s is None:
            raise ValueError(f"Q has wide-angle order {n} - pass s and d, the effective comoving distance.")
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n] = {L: np.where(s > 0, q * (d / s) ** n, Q.get((L, 0), 0 * q)) for L, q in own.items()}
    return out


def _as_list(l):
    return [int(l)] if np.isscalar(l) else [int(x) for x in l]
