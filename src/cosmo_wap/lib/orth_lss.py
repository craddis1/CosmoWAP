"""The orthogonal PNG template is orthogonal-LSS - Senatore, Smith & Zaldarriaga 2010 (0905.3746) App. B.

The standard orthogonal template (what MathWAP exported as bk.Orth) goes as k_L/k_S in the squeezed limit,
which gives it a k^-1 scale-dependent bias (alpha = 1). That limit is an artefact of the factorisable
template: the orthogonal shape of single-field inflation goes as (k_L/k_S)^2, as the equilateral one does.
The orthogonal-LSS template has that limit, the same value at equilateral triangles, and is orthogonal to
the equilateral template. It is the one used for LSS (e.g. the Euclid PNG forecasts, Cabass+ 2022), and
'Orth' means it throughout:

- the primordial bispectrum is B_phi below (numeric_mu.bk.get_B_prim);
- the scale-dependent bias goes as k^2/M(k), as for Eq (pk.Orth, numeric_mu.kernels.K1.PNG_ALPHA), and
  the PBS biases take alpha = 2 with the amplitude A below (HOD.PBBias.Orth).

The bk/bk_mt 'Orth' exports are of the standard template, so install() replaces their multipoles (and
bk.Orth_, which bk.Orth.ylm integrates): each returns Eq evaluated with the orthogonal biases (the
alpha = 2 kernels), less Eq with no PNG bias (its primordial term alone), plus the orthogonal-LSS
primordial term, taken as Loc with no PNG bias (B_I^loc times the Gaussian kernels) rescaled by
B_Phi^orth-LSS/B_Phi^loc. That is exact at linear order in fNL, which is all a Fisher derivative at
fNL = 0 sees; bk_mt's exports also keep fNL^2 terms, and in those the primordial term's PNG-bias cross
terms carry the Eq template. The generated standard-template code is left in place (editing it would
make the PNG C kernels stale) until Orth is regenerated from MathWAP, and is no longer tabulated.
"""

from __future__ import annotations

import inspect
import weakref

import numpy as np

from cosmo_wap.lib import utils

# SSZ10 (B.2), p there - 0905.3746 App. B
Q = 27 / (-21 + 743 / (7 * (20 * np.pi**2 - 193)))  # ~8.526

# Squeezed limit B_Phi -> 4 A fNL P(k_L) P(k_S) (k_L/k_S)^alpha, taken at k1 = k2 (mu = 0, the long mode
# perpendicular to the short ones) - as A = 3 for Eq, and A = -3 (alpha = 1) for the standard orthogonal
# template, whose k_L/k_S term this template cancels. For alpha = 2 the limit depends on that angle: as
# (1 - mu^2) for Eq, and as 2(1 + Q)(1 - mu^2) - 64 Q/27 here. Averaged over mu, which is what the PBS
# response of the local variance picks up, A would be 2 for Eq and 2 - 14 Q/9 ~ -11.3 here.
# A below is derived from 0905.3746 App. B at k1 = k2, using the same squeezed approximation
# as Eq (1008.0638 sec. V, Eq. (58)), which drops the angular dependence.
A = 3 - 5 * Q / 9  # ~ -1.737
ALPHA = 2


def B_phi(P1, P2, P3, k1, k2, k3):
    """B_Phi / fNL - 6 (P1 P2 P3)^(2/3) [(1 + Q) Delta/(k1 k2 k3) - Q Gamma^3/(k1 k2 k3)^2], SSZ10 (B.1)"""
    kt = k1 + k2 + k3
    delta = (kt - 2 * k1) * (kt - 2 * k2) * (kt - 2 * k3)
    gamma = 2 / 3 * (k1 * k2 + k2 * k3 + k3 * k1) - 1 / 3 * (k1**2 + k2**2 + k3**2)
    kkk = k1 * k2 * k3
    return 6 * (P1 * P2 * P3) ** (2 / 3) * ((1 + Q) * delta / kkk - Q * gamma**3 / kkk**2)


def loc_ratio(cosmo_funcs, k1, k2, k3=None, theta=None):
    """B_Phi^orth-LSS / B_Phi^loc on the triangle - transfer functions and growth cancel"""
    k3, _ = utils.get_theta_k3(k1, k2, k3, theta)
    P1, P2, P3 = (cosmo_funcs.Pk_phi(k) for k in (k1, k2, k3))
    return B_phi(P1, P2, P3, k1, k2, k3) / (2 * (P1 * P2 + P1 * P3 + P2 * P3))


class _OrthBiasView:
    """cosmo_funcs whose PNG biases are the orthogonal ones, or zero, whatever shape is asked for - so the
    Eq and Loc expressions can be evaluated with them. Everything else is the wrapped object's.

    Holds a weak proxy: _views keeps one pair per cosmo_funcs (the bk tables key their cache on the
    object passed in), and a strong reference back would keep every cosmo_funcs alive."""

    def __init__(self, cosmo_funcs, zero=False):
        self._cf = weakref.proxy(cosmo_funcs)
        self._zero = zero

    def __getattr__(self, name):
        return getattr(self._cf, name)

    def get_PNG_bias(self, zz, ti=0, shape="Loc"):
        if self._zero:
            zero = 0 * np.asarray(zz, dtype=float)
            return zero, zero
        return self._cf.get_PNG_bias(zz, ti, "Orth")

    def get_PNGparams(self, zz, k1, k2, k3, ti=0, shape="Loc"):
        _, _, Mk1, Mk2, Mk3 = self._cf.get_PNGparams(zz, k1, k2, k3, ti=ti, shape="Loc")  # M(k) is shape-free
        b01, b11 = self.get_PNG_bias(zz, ti)
        return b01, b11, Mk1, Mk2, Mk3


_VIEWS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _views(cosmo_funcs):
    """(orthogonal biases, zero biases) views of cosmo_funcs - the same pair on every call, so the bk tables'
    cache (keyed on the object) hits. Keyed by identity, so a utils.copy gets its own."""
    pair = _VIEWS.get(cosmo_funcs)
    if pair is None:
        pair = _VIEWS[cosmo_funcs] = (_OrthBiasView(cosmo_funcs), _OrthBiasView(cosmo_funcs, zero=True))
    return pair


def _compose(sig, eq, loc, cf_name="cosmo_funcs"):
    """One orthogonal-LSS method with the signature sig of the export it replaces - see the module docstring.
    eq and loc return the same method of Eq and Loc (or bk.Eq_ and bk.Loc_), looked up at call time so later
    patches apply."""
    # the generated methods take **kwargs, a table's wrapper **kw (bk.table._wrap) - fNL can be in either
    var_kw = next((n for n, p in sig.parameters.items() if p.kind is p.VAR_KEYWORD), None)

    def method(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = dict(bound.arguments)
        a.update(a.pop(var_kw, {}) if var_kw else {})
        fNL_orth, fNL = a.pop("fNL_orth", None), a.pop("fNL", 1)
        # no fNL_loc/fNL_eq - they would override fNL in the Eq and Loc expressions
        a = {k: v for k, v in a.items() if not k.startswith("fNL")}
        a["fNL"] = fNL if fNL_orth is None else fNL_orth
        cf = a[cf_name]
        with_bias, no_bias = _views(cf)

        def call(func, view):
            return func(**{**a, cf_name: view})

        ratio = loc_ratio(cf, a["k1"], a["k2"], a.get("k3"), a.get("theta"))
        return call(eq(), with_bias) - call(eq(), no_bias) + ratio * call(loc(), no_bias)

    method.__doc__ = "Orthogonal-LSS template, built from Eq and Loc - see cosmo_wap.lib.orth_lss."
    method.orth_lss = True
    return method


def install(namespace):
    """Replace a package's Orth multipoles (l0, l1, ...) and, if present, its (mu, phi) expression Orth_
    (bk.Orth.ylm integrates bk.Orth_) with the orthogonal-LSS ones. Called last from bk/__init__ and
    bk_mt/__init__, after the C kernels and tables are patched on - so the Eq and Loc pieces use them."""
    Orth, Eq, Loc = (namespace[name] for name in ("Orth", "Eq", "Loc"))
    for meth in [m for m in vars(Orth) if m[:1] == "l" and m[1:].isdigit()]:
        export = getattr(Orth, meth)
        # the zero stand-ins (utils.add_empty_methods_bk) are zero for either template
        if getattr(export, "orth_lss", False) or export.__name__ == "empty_array_func":
            continue
        lss = _compose(
            inspect.signature(export), lambda meth=meth: getattr(Eq, meth), lambda meth=meth: getattr(Loc, meth)
        )
        setattr(Orth, meth, staticmethod(lss))
    if "Orth_" in namespace and not getattr(namespace["Orth_"], "orth_lss", False):
        namespace["Orth_"] = _compose(
            inspect.signature(namespace["Orth_"]), lambda: namespace["Eq_"], lambda: namespace["Loc_"]
        )
