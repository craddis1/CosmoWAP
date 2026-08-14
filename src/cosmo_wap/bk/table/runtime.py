"""Runtime for the redshift-monomial tables: cache the coefficients, contract per call.

The table path is deliberately hard to enter by accident. It runs only when

  * COSMOWAP_BK_TABLE=1 is set, and
  * a sampler has called set_version() to say which slow (cosmology) state is current.

Anything that does not register a version - Fisher forecasts, notebooks, one-off calls -
never takes this branch and keeps the ordinary compiled kernel, so it cannot be made
slower by a table it would only build once and throw away.
"""
import os
import warnings
from collections import OrderedDict

import numpy as np

ENABLED = os.environ.get('COSMOWAP_BK_TABLE') == '1'

# The cache is bounded by bytes, not by entry count: one entry exists per
# (class, method, cosmology, triangle set), so a run with several redshift bins, both
# multipoles and more than one term module needs dozens of them, while their sizes differ
# by more than an order of magnitude between modules (WA2 0.8 MB, RR2 tens of MB).
# Sizing by count either thrashes - rebuilding every call, which is slower than having no
# table at all - or pins far more memory than intended. Per process, so N MPI chains want
# N times this.
_CACHE_MB = float(os.environ.get('COSMOWAP_BK_TABLE_MB', 256))
_THRASH_WARN = 16  # evictions before we say the budget looks too small

_version = None
_cache = OrderedDict()   # key -> (table, *inputs kept alive); least-recently-used first
_cache_bytes = 0
_evictions = 0
_warned = False


def set_version(v):
    """Tell the runtime which cosmology is current; None disables the table path.

    The coefficients depend on cosmology only through Pk, so anything that rebuilds the
    cosmology should change v. The cache also keys on the cosmo_funcs object itself, so
    a forgotten bump costs a rebuild rather than returning a table for the wrong
    cosmology - but a cosmo_funcs mutated in place would defeat both.
    """
    global _version
    _version = v


def active():
    return ENABLED and _version is not None


def reset():
    global _cache_bytes, _evictions, _warned
    _cache.clear()
    _cache_bytes = 0
    _evictions = 0
    _warned = False


def cache_stats():
    """(entries, bytes held, evictions) - for diagnosing a table that is not paying off."""
    return len(_cache), _cache_bytes, _evictions


_exp_cache = {}


def _exponents(monomial):
    """(names, exponent matrix) for a monomial list, built once per list.

    Looping the monomials in python costs 0.42 ms against 0.012 ms for the contraction
    itself, so the exponents are laid out as a matrix and the basis evaluated in one
    vectorised power. The lists are module-level constants, so id() is a stable key.
    """
    hit = _exp_cache.get(id(monomial))
    if hit is None:
        names = sorted({n for mono in monomial for n, _ in mono})
        pos = {n: i for i, n in enumerate(names)}
        exp = np.zeros((len(monomial), len(names)), dtype=np.int64)
        for m, mono in enumerate(monomial):
            for n, e in mono:
                exp[m, pos[n]] = e
        hit = _exp_cache[id(monomial)] = (names, exp, monomial)
    return hit[0], hit[1]


def _basis(monomial, zvals):
    """M_m = prod(sym**exp): one number per monomial, evaluated per call."""
    names, exp = _exponents(monomial)
    # float() on the values matters: np.float64 ** np.int64 is fine, but a sympy or
    # 0-d-object operand silently turns the whole array into dtype=object
    zvec = np.array([float(zvals[n]) for n in names], dtype=np.float64)
    return np.prod(zvec ** exp, axis=1)


def table(coeff_fn, cls, meth, cosmo_funcs, k1, k2, k3, theta, zz):
    """Coefficients for one method, cached on (cosmology version, triangle set).

    zz is passed through to get_params but the coefficients do not depend on it: the
    growth sits in D1, which is a monomial variable, and Pk is redshift-independent. One
    table therefore serves every redshift bin.
    """
    # cosmo_funcs is keyed on as well as the version: the sampler builds one object per
    # cosmology, so this catches a caller that forgot to bump the version, which would
    # otherwise return a table for the wrong cosmology with no error at all
    global _cache_bytes, _evictions, _warned

    key = (cls, meth, _version, id(cosmo_funcs), id(k1), np.shape(k1), id(k3), id(theta))
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit[0]
    # dtype comes from the coefficients, not imposed: the odd multipoles (GR1.l1, l3) are
    # imaginary, and the imaginary unit is a number rather than a redshift symbol, so it
    # rides along in the coefficient. The monomial basis stays real either way.
    tab = np.ascontiguousarray(coeff_fn(cosmo_funcs, k1, k2, k3, theta, zz))
    # the inputs are kept alive alongside the table: the key holds their id(), which the
    # allocator would otherwise be free to hand to a different object
    _cache[key] = (tab, cosmo_funcs, k1, k3, theta)
    _cache_bytes += tab.nbytes

    budget = _CACHE_MB * 1e6
    while len(_cache) > 1 and _cache_bytes > budget:
        _, old = _cache.popitem(last=False)   # least recently used
        _cache_bytes -= old[0].nbytes
        _evictions += 1
    if _evictions >= _THRASH_WARN and not _warned:
        _warned = True
        warnings.warn(
            f'cosmo_wap: bk coefficient tables are being evicted repeatedly '
            f'({_evictions} so far) at COSMOWAP_BK_TABLE_MB={_CACHE_MB:g}. Rebuilding a '
            f'table costs several kernel calls, so this is likely slower than running '
            f'without COSMOWAP_BK_TABLE. Raise the budget or reduce the term list.')
    return tab


def usable(zz):
    """The basis is one number per monomial, so the table path needs a scalar redshift."""
    return np.ndim(zz) == 0


def evaluate(coeff_fn, monomial, zvals_fn, cls, meth, cosmo_funcs,
             k1, k2, k3=None, theta=None, zz=0, r=0, s=0, **kw):
    """B for one method as basis . table.

    zvals_fn is the generated module's own zvals: it reuses that module's preamble, so
    nothing here needs to know which of get_params/get_derivs a module unpacks. It is
    called on a single triangle, since none of the scalars depends on k and the power
    spectrum splines in the preamble are the expensive part.

    kw goes to zvals only (fNL and friends - see _wrap): anything the convert step did not
    put in KSYM ends up in the monomial, so the coefficients cannot depend on it.
    """
    tab = table(coeff_fn, cls, meth, cosmo_funcs, k1, k2, k3, theta, zz)
    one = [np.asarray(a).ravel()[:1] if a is not None else None
           for a in (k1, k2, k3, theta)]
    zvals = zvals_fn(cosmo_funcs, *one, zz, r, s, **kw)
    return (_basis(monomial, zvals) @ tab).reshape(np.shape(k1))
