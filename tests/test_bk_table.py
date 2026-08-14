"""Tests for the bk redshift-monomial coefficient tables (cosmo_wap.bk.table).

The table path is normally gated behind COSMOWAP_BK_TABLE=1 plus a registered cosmology
version, so simply running the suite with the flag set exercises only the fallback.
These tests install the dispatch explicitly instead, so they cover the real path whether
or not the flag is set, and uninstall it again afterwards.

Each generated module is covered; one that has not been generated is skipped
(python -m cosmo_wap.bk.table.convert WA2,RR2).
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.bk.table import TABLE_MODULES
from cosmo_wap.lib import utils

# the table is an expanded-sum rearrangement, so it is close to the kernel but never
# bit-identical; measured ~3e-7 relative to max|B| with a median around 1e-12
RTOL_MAX = 1e-5

R, S = 0.3, 0.2


@pytest.fixture(scope="module", params=TABLE_MODULES)
def mod_name(request):
    return request.param


@pytest.fixture(scope="module")
def tab_mod(mod_name):
    return pytest.importorskip(
        f"cosmo_wap.bk.table.{mod_name}_tab",
        reason=f"run: python -m cosmo_wap.bk.table.convert {mod_name}",
    )


@pytest.fixture(scope="module")
def dispatch(tab_mod, mod_name):
    """Install table dispatch onto this module's bk class, then put it back."""
    from cosmo_wap.bk import table as tbl
    from cosmo_wap.bk.table import runtime

    bk_cls = getattr(cw.bk, mod_name)
    saved = {m: getattr(bk_cls, m) for m in tab_mod.MONOMIALS}
    was_enabled = runtime.ENABLED
    runtime.ENABLED = True
    installed = tbl.install(vars(cw.bk), modules=(mod_name,))
    if not installed:
        runtime.ENABLED = was_enabled
        pytest.skip("table dispatch did not install")

    yield runtime

    for meth, fn in saved.items():
        setattr(bk_cls, meth, staticmethod(fn))
    runtime.ENABLED = was_enabled
    runtime.set_version(None)
    runtime.reset()


@pytest.fixture
def bk_cls(mod_name, dispatch):
    return getattr(cw.bk, mod_name)


@pytest.fixture(autouse=True)
def clean_runtime(dispatch):
    """No test may leak a version or a cached table into the next."""
    dispatch.set_version(None)
    dispatch.reset()
    yield
    dispatch.set_version(None)
    dispatch.reset()


@pytest.fixture(scope="module")
def tri(bk_bin):
    """(cosmo_funcs, k1, k2, k3, theta, zz) for one bin."""
    return bk_bin.args


@pytest.fixture(scope="module")
def cosmo_funcs_alt():
    """A second, genuinely different cosmology."""
    cosmo = utils.get_cosmo(h=0.67, Omega_m=0.28, k_max=1.0, z_max=4.0)
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), verbose=False)


@pytest.fixture(scope="module")
def meths(tab_mod):
    """The multipoles this module tabulates - GR1 has l1/l3, every other class l0/l2."""
    return list(tab_mod.MONOMIALS)


@pytest.fixture(scope="module")
def meth(meths):
    """The one used wherever a test only needs a single multipole."""
    return meths[0]


def _call(bk_cls, cf, args, meth, **kw):
    _, k1, k2, k3, theta, zz = args
    # no dtype= here: the odd multipoles (GR1.l1, l3) are imaginary
    return np.asarray(getattr(bk_cls, meth)(cf, k1, k2, k3, theta, zz, R, S, **kw))


def _kernel(dispatch, bk_cls, cf, args, meth, **kw):
    """Force the ordinary kernel, whatever dispatch is installed."""
    dispatch.set_version(None)
    return _call(bk_cls, cf, args, meth, **kw)


def _rel(got, ref):
    return np.abs(np.asarray(got) - np.asarray(ref)).max() / np.abs(ref).max()


def test_M_is_separable_in_redshift(cosmo_funcs):
    """M(k,z)/D(z) must not depend on z, or the PNG tables are silently wrong.

    convert._m_subs rewrites Mk_i as D1*mk_i so the growth joins the monomial, and
    runtime.table then reuses one set of coefficients for every redshift bin. Both rest
    on this identity, which holds for M = sqrt(D**2 * Pk/Pk_phi) but not for any M."""
    kk = np.logspace(-3, -0.5, 50)
    ref = cosmo_funcs.M(kk, 0.5) / cosmo_funcs.D(0.5)
    for z in (0.0, 1.0, 2.5, 3.9):
        np.testing.assert_allclose(cosmo_funcs.M(kk, z) / cosmo_funcs.D(z), ref, rtol=1e-12)


# ── the table reproduces the kernel ───────────────────────────────────────────


def test_table_matches_kernel(dispatch, bk_cls, tri, meths):
    cf = tri[0]
    for m in meths:
        ref = _kernel(dispatch, bk_cls, cf, tri, m)

        dispatch.set_version(("fiducial",))
        got = _call(bk_cls, cf, tri, m)

        # without this the test passes just as happily if dispatch silently fell back
        assert dispatch._cache, f"{m}: the table path was not taken"
        assert got.shape == ref.shape
        assert got.dtype == ref.dtype, f"{m}: table lost the imaginary part"
        assert _rel(got, ref) < RTOL_MAX, m


def test_table_is_reused_not_rebuilt(dispatch, bk_cls, tri, meth):
    """A second call at the same cosmology must hit the cache rather than rebuild."""
    cf = tri[0]
    dispatch.set_version(("fiducial",))
    _call(bk_cls, cf, tri, meth)
    n_after_first = len(dispatch._cache)
    assert n_after_first == 1
    _call(bk_cls, cf, tri, meth)
    assert len(dispatch._cache) == n_after_first


# ── invalidation: the dangerous failure is a stale table ──────────────────────


def test_new_cosmology_rebuilds(dispatch, bk_cls, tri, cosmo_funcs_alt, meth):
    """Same triangles, different cosmology: each must match its own kernel."""
    cf = tri[0]
    alt = (cosmo_funcs_alt,) + tri[1:]

    ref_a = _kernel(dispatch, bk_cls, cf, tri, meth)
    ref_b = _kernel(dispatch, bk_cls, cosmo_funcs_alt, alt, meth)
    assert _rel(ref_a, ref_b) > 1e-3, "the two cosmologies should differ appreciably"

    dispatch.set_version(("a",))
    got_a = _call(bk_cls, cf, tri, meth)
    dispatch.set_version(("b",))
    got_b = _call(bk_cls, cosmo_funcs_alt, alt, meth)
    dispatch.set_version(("a",))
    again_a = _call(bk_cls, cf, tri, meth)

    assert len(dispatch._cache) == 2, "each cosmology needs its own cached table"
    assert _rel(got_a, ref_a) < RTOL_MAX
    assert _rel(got_b, ref_b) < RTOL_MAX
    assert _rel(again_a, ref_a) < RTOL_MAX


def test_cosmology_identity_guards_a_forgotten_version_bump(dispatch, bk_cls, tri, cosmo_funcs_alt, meth):
    """The cache keys on cosmo_funcs too, so a stale version costs a rebuild, not a wrong answer."""
    cf = tri[0]
    alt = (cosmo_funcs_alt,) + tri[1:]
    ref_b = _kernel(dispatch, bk_cls, cosmo_funcs_alt, alt, meth)

    dispatch.set_version(("never bumped",))
    _call(bk_cls, cf, tri, meth)
    got = _call(bk_cls, cosmo_funcs_alt, alt, meth)

    assert _rel(got, ref_b) < RTOL_MAX


# ── the fallbacks must be exact, not approximate ──────────────────────────────


def test_no_version_is_the_kernel_exactly(dispatch, bk_cls, tri, meth):
    cf = tri[0]
    dispatch.set_version(None)
    a = _call(bk_cls, cf, tri, meth)
    b = _call(bk_cls, cf, tri, meth)
    assert np.array_equal(a, b)


def test_array_redshift_falls_back(dispatch, bk_cls, tri, meth):
    """An array zz has no single monomial basis, so it must take the kernel."""
    cf, k1, k2, k3, theta, zz = tri
    zz_arr = np.full(np.shape(k1), float(np.asarray(zz).ravel()[0]))
    args = (cf, k1, k2, k3, theta, zz_arr)

    ref = _kernel(dispatch, bk_cls, cf, args, meth)
    dispatch.set_version(("fiducial",))
    got = _call(bk_cls, cf, args, meth)

    assert np.array_equal(got, ref)
    assert not dispatch._cache, "the fallback must not populate the table cache"


def test_extra_kwargs_fall_back(dispatch, bk_cls, tri, tab_mod, mod_name, meth):
    """The table was not generated under nonlin/growth2, so those must take the kernel."""
    import inspect

    # not every module offers them - the WSGR classes take no such options at all
    coeff_cls = getattr(tab_mod, f"{mod_name}_tab")
    if "growth2" not in inspect.signature(getattr(coeff_cls, meth)).parameters:
        pytest.skip(f"{mod_name} has no growth2 option")

    cf = tri[0]
    ref = _kernel(dispatch, bk_cls, cf, tri, meth, growth2=False)
    dispatch.set_version(("fiducial",))
    got = _call(bk_cls, cf, tri, meth, growth2=False)

    assert np.array_equal(got, ref)
    assert not dispatch._cache


def test_install_is_a_noop_when_disabled(tab_mod):
    from cosmo_wap.bk import table as tbl
    from cosmo_wap.bk.table import runtime

    was = runtime.ENABLED
    runtime.ENABLED = False
    try:
        assert tbl.install({"WA2": object()}) == []
    finally:
        runtime.ENABLED = was


# ── cache sizing ──────────────────────────────────────────────────────────────


def _fill(dispatch, bk_cls, tri, meth, versions):
    for v in versions:
        dispatch.set_version((v,))
        _call(bk_cls, tri[0], tri, meth)


def test_cache_holds_several_tables_within_budget(dispatch, bk_cls, tri, meth, monkeypatch):
    """A run with several bins/methods needs more than a couple of entries resident."""
    monkeypatch.setattr(dispatch, "_CACHE_MB", 4096)
    _fill(dispatch, bk_cls, tri, meth, ["a", "b", "c", "d", "e"])

    entries, nbytes, evictions = dispatch.cache_stats()
    assert entries == 5
    assert evictions == 0
    assert nbytes > 0


def test_cache_evicts_least_recently_used_over_budget(dispatch, bk_cls, tri, meth, monkeypatch):
    """Bounded by bytes: a budget below one table keeps exactly the newest."""
    monkeypatch.setattr(dispatch, "_CACHE_MB", 1e-3)
    _fill(dispatch, bk_cls, tri, meth, ["a", "b", "c"])

    entries, nbytes, evictions = dispatch.cache_stats()
    assert entries == 1, "an over-budget cache must not grow without bound"
    assert evictions == 2
    assert nbytes > 0, "the entry just built is kept even when it alone exceeds budget"


def test_cache_reports_bytes_it_holds(dispatch, bk_cls, tri, meth, monkeypatch):
    monkeypatch.setattr(dispatch, "_CACHE_MB", 4096)
    _fill(dispatch, bk_cls, tri, meth, ["a"])

    entries, nbytes, _ = dispatch.cache_stats()
    assert entries == 1
    tab = next(iter(dispatch._cache.values()))[0]
    assert nbytes == tab.nbytes


# ── the basis itself ──────────────────────────────────────────────────────────


def test_basis_matches_naive_product(dispatch, tab_mod):
    """The vectorised basis must agree with the obvious per-monomial product."""
    monomial = next(iter(tab_mod.MONOMIALS.values()))
    names = sorted({n for mono in monomial for n, _ in mono})
    rng = np.random.default_rng(0)
    zvals = {n: float(v) for n, v in zip(names, rng.uniform(0.5, 1.5, len(names)))}

    got = dispatch._basis(monomial, zvals)
    naive = np.array([np.prod([zvals[n] ** int(e) for n, e in mono]) if mono else 1.0 for mono in monomial])

    assert got.dtype == np.float64
    np.testing.assert_allclose(got, naive, rtol=1e-12)


def test_monomials_match_table_rows(dispatch, tri, tab_mod, mod_name):
    """One coefficient row per monomial, or the contraction is silently misaligned."""
    cf, k1, k2, k3, theta, zz = tri
    coeff_cls = getattr(tab_mod, f"{mod_name}_tab")
    for meth, monomial in tab_mod.MONOMIALS.items():
        rows = np.asarray(getattr(coeff_cls, meth)(cf, k1, k2, k3, theta, zz))
        assert rows.shape[0] == len(monomial)
