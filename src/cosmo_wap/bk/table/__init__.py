"""Redshift-monomial coefficient tables for the bispectrum.

convert.py explains the decomposition, runtime.py caches and contracts it. install()
points the ordinary bk classes at a table when one has been generated and enabled.

Nothing here changes behaviour unless COSMOWAP_BK_TABLE=1 *and* a sampler has registered
a cosmology version (runtime.set_version), so Fisher forecasts, notebooks and one-off
calls keep the ordinary compiled kernel.
"""
import importlib
import importlib.util
import json
import os

TABLE_MODULES = ('WA2', 'RR2')


def _compiled(mod):
    """The compiled coefficient kernel for `mod`, or None to fall back to numpy."""
    from cosmo_wap.bk import c_compile

    manifest = os.path.join(c_compile.C_LIB, 'manifest.json')
    if os.environ.get('COSMOWAP_DISABLE_C') or not os.path.exists(manifest):
        return None
    info = json.load(open(manifest)).get(mod)
    if not info or c_compile._src_hash(mod) != info['sha256']:
        return None
    spec = importlib.util.spec_from_file_location(
        f'cosmo_wap.bk.c_lib.{mod}_c', os.path.join(c_compile.C_LIB, f'{mod}_c.py'))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return getattr(wrapper, mod)


def _wrap(coeff_fn, monomial, zvals_fn, cls, meth, orig):
    from . import runtime

    def bk_method(cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, r=0, s=0, **kw):
        # kw is nonlin/growth2, which the table was not generated under; an array zz has
        # no single monomial basis. Either way the ordinary kernel answers.
        if not kw and runtime.active() and runtime.usable(zz):
            return runtime.evaluate(coeff_fn, monomial, zvals_fn, cls, meth,
                                    cosmo_funcs, k1, k2, k3, theta, zz, r, s)
        return orig(cosmo_funcs, k1, k2, k3, theta, zz, r, s, **kw)

    bk_method.__name__ = meth
    bk_method.__doc__ = f'{cls}.{meth} via redshift-monomial table (falls back to kernel).'
    return bk_method


def install(namespace, modules=TABLE_MODULES):
    """Patch table dispatch onto the bk classes in `namespace`. Returns what was patched."""
    from . import runtime

    if not runtime.ENABLED:
        return []
    installed = []
    for mod in modules:
        try:
            tab = importlib.import_module(f'{__name__}.{mod}_tab')
        except ModuleNotFoundError:
            continue  # not generated for this module yet
        target = namespace.get(mod)
        if target is None:
            continue
        coeff_cls = _compiled(f'{mod}_tab') or getattr(tab, f'{mod}_tab')
        for meth, monomial in tab.MONOMIALS.items():
            orig = getattr(target, meth)
            setattr(target, meth, staticmethod(_wrap(
                getattr(coeff_cls, meth), monomial, tab.zvals, mod, meth, orig)))
            installed.append(f'{mod}.{meth}')
    return installed
