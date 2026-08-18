"""Tests for the multi-tracer coefficient tables (cosmo_wap.bk_mt.table).

The decomposition, gating and caching are cosmo_wap.bk.table's and are covered by
test_bk_table.py. What is specific here is that there are now two expression packages
defining classes of the same name (NPP, GR1, GR2, Loc), so this file concentrates on the
table reproducing the multi-tracer kernel and on the two packages staying apart - in the
runtime cache, and in which compiled kernel gets patched onto which class.
"""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.bk_mt.table import TABLE_MODULES

# an expanded-sum rearrangement, so close to the kernel but never bit-identical. The
# multi-tracer expressions are small (600-2300 terms against RR2's 605k), so they land
# near machine precision rather than the ~1e-6 the wide-separation classes reach.
RTOL_MAX = 1e-9

R, S = 0.3, 0.2


@pytest.fixture(scope="module", params=TABLE_MODULES)
def mod_name(request):
    return request.param


@pytest.fixture(scope="module")
def tab_mod(mod_name):
    return pytest.importorskip(
        f"cosmo_wap.bk_mt.table.{mod_name}_tab",
        reason="run: python -m cosmo_wap.bk.table.convert --pkg bk_mt GR0,GR1,GR2,PNG",
    )


@pytest.fixture(scope="module")
def dispatch(tab_mod, mod_name):
    """Install table dispatch onto this module's bk_mt class, then put it back."""
    from cosmo_wap.bk.table import runtime
    from cosmo_wap.bk_mt import table as tbl

    cls = getattr(cw.bk_mt, mod_name)
    saved = {m: getattr(cls, m) for m in tab_mod.MONOMIALS}
    was_enabled = runtime.ENABLED
    runtime.ENABLED = True
    if not tbl.install(vars(cw.bk_mt), modules=(mod_name,)):
        runtime.ENABLED = was_enabled
        pytest.skip("table dispatch did not install")

    yield runtime

    for meth, fn in saved.items():
        setattr(cls, meth, staticmethod(fn))
    runtime.ENABLED = was_enabled
    runtime.set_version(None)
    runtime.reset()


@pytest.fixture
def bk_cls(mod_name, dispatch):
    return getattr(cw.bk_mt, mod_name)


@pytest.fixture(autouse=True)
def clean_runtime(dispatch):
    """No test may leak a version or a cached table into the next."""
    dispatch.set_version(None)
    dispatch.reset()
    yield
    dispatch.set_version(None)
    dispatch.reset()


@pytest.fixture(scope="module")
def tri(forecast_mt):
    """(cosmo_funcs, k1, k2, k3, theta, zz) for one multi-tracer bin."""
    return forecast_mt.get_bk_bin(0).args


def _call(cls, cf, args, meth, **kw):
    _, k1, k2, k3, theta, zz = args
    # no dtype= here: the odd multipoles (GR1.l1, l3) are imaginary
    return np.asarray(getattr(cls, meth)(cf, k1, k2, k3, theta, zz, R, S, **kw))


def _kernel(dispatch, cls, cf, args, meth, **kw):
    """Force the ordinary kernel, whatever dispatch is installed."""
    dispatch.set_version(None)
    return _call(cls, cf, args, meth, **kw)


def _rel(got, ref):
    return np.abs(np.asarray(got) - np.asarray(ref)).max() / np.abs(ref).max()


def test_the_bin_really_is_multi_tracer(tri):
    """Otherwise every assertion below would be testing the single-tracer expressions."""
    assert tri[0].multi_tracer


def test_table_matches_kernel(dispatch, bk_cls, tri, tab_mod):
    cf = tri[0]
    for meth in tab_mod.MONOMIALS:
        ref = _kernel(dispatch, bk_cls, cf, tri, meth)

        dispatch.set_version(("fiducial",))
        got = _call(bk_cls, cf, tri, meth)

        # without this the test passes just as happily if dispatch silently fell back
        assert dispatch._cache, f"{meth}: the table path was not taken"
        assert got.shape == ref.shape
        assert got.dtype == ref.dtype, f"{meth}: table lost the imaginary part"
        assert _rel(got, ref) < RTOL_MAX, meth


def test_monomials_match_table_rows(dispatch, tri, tab_mod, mod_name):
    """One coefficient row per monomial, or the contraction is silently misaligned."""
    cf, k1, k2, k3, theta, zz = tri
    coeff_cls = getattr(tab_mod, f"{mod_name}_tab")
    for meth, monomial in tab_mod.MONOMIALS.items():
        rows = np.asarray(getattr(coeff_cls, meth)(cf, k1, k2, k3, theta, zz))
        assert rows.shape[0] == len(monomial)


def test_per_tracer_scalars_reach_the_monomial(tab_mod):
    """The second and third tracer's biases must be monomial symbols, not coefficients.

    They are what makes these tables multi-tracer; had they been left in the coefficients
    convert.py would have refused, but a silently single-tracer monomial list would not
    show up in any comparison against the kernel at a fixed cosmology.
    """
    names = {n for keys in tab_mod.MONOMIALS.values() for k in keys for n, _ in k}
    assert names & {"xb1", "yb1"}, names


def test_array_redshift_falls_back(dispatch, bk_cls, tri, tab_mod):
    """An array zz has no single monomial basis, so it must take the kernel."""
    meth = next(iter(tab_mod.MONOMIALS))
    cf, k1, k2, k3, theta, zz = tri
    zz_arr = np.full(np.shape(k1), float(np.asarray(zz).ravel()[0]))
    args = (cf, k1, k2, k3, theta, zz_arr)

    ref = _kernel(dispatch, bk_cls, cf, args, meth)
    dispatch.set_version(("fiducial",))
    got = _call(bk_cls, cf, args, meth)

    assert np.array_equal(got, ref)
    assert not dispatch._cache, "the fallback must not populate the table cache"


# ── the two packages must not be confused for one another ────────────────────


def test_cache_key_separates_the_two_packages(dispatch, bk_cls, tri, tab_mod, mod_name, cosmo_funcs):
    """bk.GR1.l0 and bk_mt.GR1.l0 share a class name; they must not share a cache entry.

    id(cosmo_funcs) would usually save this, since a given object is one or the other -
    but that is a coincidence of how the sampler builds them, not a guarantee.
    """
    from cosmo_wap.bk.table import install as install_bk

    if not hasattr(cw.bk, mod_name):
        pytest.skip(f"bk has no {mod_name}")
    single = getattr(cw.bk, mod_name)
    saved = {m: getattr(single, m) for m in tab_mod.MONOMIALS if hasattr(single, m)}
    if not install_bk(vars(cw.bk), modules=(mod_name,)):
        pytest.skip("single-tracer table not generated")
    try:
        meth = next(iter(saved))
        st_args = (cosmo_funcs,) + tri[1:]
        ref_st = _kernel(dispatch, single, cosmo_funcs, st_args, meth)
        ref_mt = _kernel(dispatch, bk_cls, tri[0], tri, meth)

        dispatch.set_version(("fiducial",))
        got_mt = _call(bk_cls, tri[0], tri, meth)
        got_st = _call(single, cosmo_funcs, st_args, meth)

        assert len(dispatch._cache) == 2, "one cache entry per package, not one shared"
        assert _rel(got_mt, ref_mt) < RTOL_MAX
        assert _rel(got_st, ref_st) < RTOL_MAX
    finally:
        for meth, fn in saved.items():
            setattr(single, meth, staticmethod(fn))


def test_compiled_kernels_stay_in_their_own_package():
    """Each package's classes must carry their own compiled kernel, not the other's.

    The manifest is flat and keys the multi-tracer modules by an mt_ prefix; dropping the
    filter in _load_c_kernels would patch bk_mt's GR1 onto bk's GR1, which is a wrong
    answer with no error anywhere.
    """
    from cosmo_wap.bk import c_compile

    for name in ("NPP", "GR1", "GR2", "Loc"):
        for pkg, want in ((cw.bk, "bk"), (cw.bk_mt, "bk_mt")):
            cls = getattr(pkg, name, None)
            if cls is None:
                continue
            for meth in ("l0", "l1", "l2", "l3", "l4"):
                fn = getattr(cls, meth, None)
                mod = getattr(fn, "__module__", "") or ""
                if fn is None or "c_lib" not in mod:
                    continue  # numpy or the empty-method stand-in
                key = mod.rsplit(".", 1)[-1].removesuffix("_c")
                is_mt = key.startswith(c_compile.MT_PREFIX)
                assert is_mt == (want == "bk_mt"), f"{want}.{name}.{meth} loaded {key}"


def test_compiled_coefficient_kernel_matches_numpy(tri, tab_mod, mod_name):
    """If the table kernel has been built, it must agree with the numpy table module.

    The manifest key and the class inside the wrapper are different strings here
    (mt_NPP_tab names the build, NPP_tab the class), which is exactly what _compiled has
    to keep straight - getting it wrong is an AttributeError at import, not a fallback.
    """
    from cosmo_wap.bk.table import _compiled

    compiled = _compiled(f"mt_{mod_name}_tab", f"{mod_name}_tab")
    if compiled is None:
        pytest.skip(f"run: python -m cosmo_wap.bk.c_compile mt_{mod_name}_tab")

    cf, k1, k2, k3, theta, zz = tri
    numpy_cls = getattr(tab_mod, f"{mod_name}_tab")
    for meth in tab_mod.MONOMIALS:
        ref = np.asarray(getattr(numpy_cls, meth)(cf, k1, k2, k3, theta, zz))
        got = np.asarray(getattr(compiled, meth)(cf, k1, k2, k3, theta, zz))
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12, meth
