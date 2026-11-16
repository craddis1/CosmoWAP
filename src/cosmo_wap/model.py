"""Model - set the signal once (the kernels of each field, and the analytic wide-separation terms the kernels do
not have), then get its power spectrum multipoles, its P(k, mu), or its correlation function multipoles.

    m = cw.Model(cosmo_funcs, kernels=["N", "LP", "I"], wide_separation=True)
    P = m.pk(k, l=[0, 2], zz=1.0)
    xi = m.xi(r, l=[0, 1, 2], zz=1.0)

    IS = cw.Model(cosmo_funcs, kernels="I", kernels2=["N", "LP"])   # integrated (field 1) x source (field 2)

The numeric kernels are those of numeric_mu/kernels.py; the wide-separation terms are the pk classes' full
angular polynomials, projected on the same mu grid as the kernels.
"""

import numpy as np
from scipy.interpolate import CubicSpline

import cosmo_wap.pk as pk
from cosmo_wap.forecast.amplitudes import KERNEL_NAMES, as_list
from cosmo_wap.lib import hankel
from cosmo_wap.numeric_mu import pk as numeric_pk
from cosmo_wap.numeric_mu.utils import split_kernels

# the analytic wide-separation terms: first order in r/d (odd l), second order (even l), and their mixing with the
# local relativistic terms (even l)
WS_BLOCKS = ("WA1", "RR1", "WA2", "RR2", "WARR", "WAGR", "RRGR")
WS_GROUPS = {
    "WA": ("WA1", "WA2"),
    "RR": ("RR1", "RR2"),
    "WS1": ("WA1", "RR1"),
    "WS2": ("WA2", "RR2", "WARR"),
    "WS": ("WA1", "RR1", "WA2", "RR2", "WARR"),  # WARR is in neither WA nor RR
    "WSGR": ("WA1", "RR1", "WA2", "RR2", "WARR", "WAGR", "RRGR"),
}
GR_MIXING = ("WAGR", "RRGR")
# the power of 1/d each block carries - its wide-angle order, which a survey window can weight on its own
WS_ORDER = {"WA1": 1, "RR1": 1, "WAGR": 1, "RRGR": 1, "WA2": 2, "RR2": 2, "WARR": 2}


class Model:
    """The signal of two fields, field 1 at x1 and field 2 at x2.

    cosmo_funcs:     ClassWAP - its tracers are the fields' (survey[0] and survey[1]; the same one for one survey)
    kernels:         field 1's kernels (numeric_mu/kernels.py), default ['N']
    kernels2:        field 2's, default the same as kernels. Field 1 sits at x1, so swapping kernels and kernels2
                     flips the odd multipoles (and the tracers, for two) - kernels='I', kernels2=['N','LP'] is the
                     IS piece, SI the reverse
    wide_separation: analytic wide-separation terms on top of the kernels. False (default) for none; True for all
                     that fit the kernels - WA1, RR1, WA2, RR2, WARR, and WAGR, RRGR once 'LP' is in; or any of the
                     blocks WS_BLOCKS and groups WS_GROUPS ('WA', 'RR', 'WS1', 'WS2', 'WS', 'WSGR'), as a list to
                     combine. Needs the same kernels on both fields (the analytic terms are not split by field)
    t:               line of sight of the wide-separation terms, d = (1-t) x1 + t x2: 0.5 the midpoint (default),
                     0 at x1, 1 at x2. The kernels are leading order in r/d, so t does not change them
    sigma:           FoG damping, as pk_func
    n_mu, n_los, deg, n_p: precision of the kernels' multipoles (n_los, deg, n_p None for get_multipoles' defaults) -
                     n_los is its n, the line-of-sight nodes of the integrated kernels
    **kwargs:        to the kernels (fNL, ...) - GL and delta also set pk's mu grid, as get_multipoles"""

    def __init__(
        self,
        cosmo_funcs,
        kernels=None,
        kernels2=None,
        wide_separation=False,
        t=0.5,
        sigma=None,
        n_mu=48,
        n_los=None,
        deg=None,
        n_p=None,
        **kwargs,
    ):
        self.cosmo_funcs = cosmo_funcs
        self.kernels = _check_kernels(["N"] if kernels is None else kernels, "kernels")
        self.kernels2 = list(self.kernels) if kernels2 is None else _check_kernels(kernels2, "kernels2")
        self.wide_separation = _resolve_ws(wide_separation, self.kernels, self.kernels2)
        self.t = t
        self.sigma = sigma
        self.n_mu = n_mu
        self.los = {k: v for k, v in dict(n=n_los, deg=deg, n_p=n_p).items() if v is not None}
        self.kwargs = kwargs
        self._bessel = {}  # Bessel kernels of xi, kept across calls

    def __repr__(self):
        k2 = "" if self.kernels2 == self.kernels else f", kernels2={self.kernels2}"
        ws = f", wide_separation={list(self.wide_separation)}" if self.wide_separation else ""
        return f"Model(kernels={self.kernels}{k2}{ws}, t={self.t})"

    def pk(self, k, l=0, zz=0, by_order=False, window=None):
        """P_l(k), shape (len(l), len(k)) - or (len(k),) for a single l. Complex, odd l imaginary, as pk_func.
        by_order: {n: P^[n]} split by wide-angle order instead - 0 the kernels, WS_ORDER the wide-separation blocks
        window:   a lib.survey_window.SurveyWindow - P̂_l through it instead, so k and l must be its k and ells (P is
                  computed on window.k_in). The orders go in by the Q^(n) the window has; by_order splits P̂ the same
                  way"""
        if window is not None:
            self._check_window(window, zz, l, k)
            P = self.pk(window.k_in, window.ells_in, zz, by_order=True)
            parts = {n: window({n: Pn}) for n, Pn in P.items()} if by_order else {0: window(P)}
            return _order_out(parts, l, by_order)
        ells = _ells(l)
        k = np.atleast_1d(np.asarray(k, dtype=float))
        GL = self.kwargs.get("GL", True)
        mu, weights = numeric_pk.get_mu_grid(self.n_mu, self.kwargs.get("delta", 0.1), GL)
        P = {
            n: np.array([numeric_pk.project_multipole(Pmu, mu, weights, ell, k, GL=GL) for ell in ells])
            for n, Pmu in self._pk_mu_orders(k, mu, zz).items()
        }
        return _order_out(P, l, by_order)

    def pk_mu(self, k, mu, zz=0):
        """P(k, mu), shape (len(k), len(mu)). The wide-separation terms use their full angular polynomials and
        FoG damps the total. With integrated kernels avoid mu = 0 itself - see numeric_mu.pk.get_mu"""
        return sum(self._pk_mu_orders(k, mu, zz).values())

    def _pk_mu_orders(self, k, mu, zz):
        k = np.atleast_1d(np.asarray(k, dtype=float))
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        kk = k[:, None]
        with self.cosmo_funcs.unpack_cache():
            P = {
                0: numeric_pk.get_mu_window(
                    mu, self.kernels, self.kernels2, self.cosmo_funcs, kk, zz, **self.los, **self.kwargs
                )
            }
            for block in self.wide_separation:
                n = WS_ORDER[block]
                P[n] = P.get(n, 0) + getattr(pk, block).mu(mu, self.cosmo_funcs, kk, zz, t=self.t)
        if self.sigma is not None:  # exp(-(k mu sigma)^2/2) on each field, as numeric_mu.pk.project_multipole
            fog = np.exp(-((kk * mu) ** 2) * self.sigma**2)
            P = {n: Pn * fog for n, Pn in P.items()}
        return P

    def xi(
        self, r=None, l=0, zz=0, r_edges=None, n_k=2**14, n_kc=2000, k_min=1e-5, kr_taper=200.0, k_ir=1e-4, n_gl=16,
        by_order=False, window=None, k_split=0.02,
    ):  # fmt: skip
        """xi_l(r), shape (len(l), len(r)) - or with r_edges the r^2-weighted average over each bin. Real.

        The Hankel transform of pk (see lib/hankel.py) - so the same basis: line of sight t, field 1 at x1.
        P_l is computed on n_kc log-spaced k from k_min to cosmo_funcs.K_MAX and splined onto n_k for the transform
        (n_kc=None computes it on all n_k). kr_taper, k_ir and n_gl as hankel.pk_to_xi. xi at r below
        ~600/K_MAX is limited by K_MAX itself. by_order as pk.

        window: a lib.survey_window.SurveyWindow - xî_l(r) through it instead, integral constraint and all, so l must
        be its ells, and r points. The IC comes from P on window.k_in, and so does xi below k_split, through the
        window's k quadrature, so the IR constants cancel exactly; above k_split xi is this transform (see
        SurveyWindow.xi)."""
        opts = dict(n_k=n_k, n_kc=n_kc, k_min=k_min, kr_taper=kr_taper, k_ir=k_ir, n_gl=n_gl)
        if window is None:
            return _order_out(self._xi_orders(r, _ells(l), zz, r_edges, **opts), l, by_order)
        if r_edges is not None:
            raise ValueError("With a window xi takes points r - the window's terms are evaluated at points.")
        self._check_window(window, zz, l)
        P = self.pk(window.k_in, window.ells_in, zz, by_order=True)
        xi = self._xi_orders(r, window.ells_in, zz, None, k_split=k_split, **opts)
        if by_order:
            parts = {n: window.xi({n: x}, r, {n: P[n]}, k_split) for n, x in xi.items()}
        else:
            parts = {0: window.xi(xi, r, P, k_split)}
        return _order_out(parts, l, by_order)

    def _xi_orders(self, r, ells, zz, r_edges, n_k, n_kc, k_min, kr_taper, k_ir, n_gl, k_split=None):
        """xi by order, {n: (len(ells), len(r))} - with k_split only P's part above it,
        P (1 - hankel.window(k, k_split, 2 k_split)), for SurveyWindow.xi to add the rest"""
        k_max = self.cosmo_funcs.K_MAX
        k = np.geomspace(k_min, k_max, n_k)
        if n_kc is None:
            P = self.pk(k, ells, zz, by_order=True)
        else:
            kc = np.geomspace(k_min, k_max, n_kc)
            lnk, lnkc = np.log(k), np.log(kc)
            P = {
                n: CubicSpline(lnkc, Pc.real, axis=1)(lnk) + 1j * CubicSpline(lnkc, Pc.imag, axis=1)(lnk)
                for n, Pc in self.pk(kc, ells, zz, by_order=True).items()
            }
        if k_split is not None:
            P = {n: Pn * (1 - hankel.window(k, k_split, 2 * k_split)) for n, Pn in P.items()}

        kernels = [self._bessel_kernel(ell, k, r, r_edges, n_gl) for ell in ells]
        out = {}
        for n, Pn in P.items():
            out[n] = np.array(
                [
                    hankel.pk_to_xi(k, p, ell, r, r_edges, kr_taper, k_ir, n_gl, kernel=K)
                    for p, ell, K in zip(Pn, ells, kernels)
                ]
            )
        return out

    def _check_window(self, window, zz, l, k=None):
        if _ells(l) != window.ells:
            raise ValueError(f"l = {l} but the window gives ells = {window.ells} - pass those.")
        if k is not None and (np.size(k) != len(window.k) or not np.allclose(k, window.k, rtol=1e-10, atol=0)):
            raise ValueError("k must be the window's output k - window.k, the bin centres for k_edges.")
        if max(window.orders) > 0 and not np.isclose(window.d, self.cosmo_funcs.comoving_dist(zz), rtol=1e-3):
            raise ValueError(
                f"window.d = {window.d:.1f} Mpc/h but zz = {zz} is at {self.cosmo_funcs.comoving_dist(zz):.1f} - the"
                " window's Q^(n) carry (s/d)^n at its own d."
            )
        integrated = split_kernels(self.kernels)[0] or split_kernels(self.kernels2)[0]
        if integrated and self.kwargs.get("window") is not None and self.kwargs.get("box") is None:
            raise ValueError(
                "window=(dr, k_f) on its own is the bin's pair counts, which the survey window's Q already has - pass"
                " box=1 too, see numeric_mu.pk.get_int_window."
            )

    def _bessel_kernel(self, ell, k, r, r_edges, n_gl):
        key = (
            ell,
            k[0],
            k[-1],
            len(k),
            None if r is None else np.asarray(r, dtype=float).tobytes(),
            None if r_edges is None else np.asarray(r_edges, dtype=float).tobytes(),
            n_gl,
        )
        if key not in self._bessel:
            if len(self._bessel) >= 32:
                self._bessel.clear()
            self._bessel[key] = hankel.bessel_kernel(ell, k, r, r_edges, n_gl)
        return self._bessel[key]


def _ells(l):
    return [l] if np.isscalar(l) else list(l)


def _order_out(parts, l, by_order):
    """{n: (len(l), ...)} -> the sum over orders, or the dict - each squeezed for a single l"""
    out = parts if by_order else {0: sum(parts.values())}
    out = {n: (x[0] if np.isscalar(l) else x) for n, x in out.items()}
    return out if by_order else out[0]


def _check_kernels(value, name):
    names = list(dict.fromkeys(as_list(value)))  # drop repeats - each kernel once
    if not names:
        raise ValueError(f"{name} is empty - a field needs at least one kernel, e.g. ['N'].")
    unknown = [n for n in names if n not in KERNEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown {name} {unknown} - the kernels are {sorted(KERNEL_NAMES)}.")
    return names


def _resolve_ws(wide_separation, kernels, kernels2):
    """wide_separation -> the blocks it adds, in WS_BLOCKS order (each once, so 'WS' with 'WA1' is not WA1 twice)"""
    if not wide_separation:
        return ()
    if isinstance(wide_separation, (bool, np.bool_)):
        blocks = set(WS_GROUPS["WSGR"] if "LP" in kernels else WS_GROUPS["WS"])
    else:
        blocks = set()
        for name in as_list(wide_separation):
            if name in WS_GROUPS:
                blocks |= set(WS_GROUPS[name])
            elif name in WS_BLOCKS:
                blocks.add(name)
            else:
                raise ValueError(
                    f"Unknown wide_separation entry '{name}' - use True, the blocks {list(WS_BLOCKS)} or the groups"
                    f" {list(WS_GROUPS)}."
                )
    if set(kernels2) != set(kernels):
        raise ValueError(
            "wide_separation needs the same kernels on both fields - the analytic terms are not split by field."
        )
    if "N" not in kernels:
        raise ValueError(
            "wide_separation needs 'N' in kernels - the wide-separation terms correct the Newtonian (Kaiser) term."
        )
    mixing = sorted(blocks & set(GR_MIXING))
    if mixing and "LP" not in kernels:
        raise ValueError(f"{mixing} mix wide separation with the local relativistic terms - they need 'LP' in kernels.")
    return tuple(b for b in WS_BLOCKS if b in blocks)
