"""Optional C fast path for the generated bispectrum expression files.

Even after the sympy.cse rewrite, evaluating the WS-family terms (WA2, WARR, RR2,
WSGR, ...) is dominated by numpy ufunc-dispatch overhead: one Python/C round trip per
arithmetic op on arrays of only ~100 triangles. This module compiles those expressions
to a C shared library (one plain loop per method, every CSE temporary a scalar double),
which is another ~10x on top of the CSE gain.

The C path is strictly opt-in. Build it once per machine with:

    python -m cosmo_wap.bk.c_compile              # default WS modules (~45 min, mostly RR2)
    python -m cosmo_wap.bk.c_compile WA2,RR2      # or a specific list

This writes kernels, wrappers and a source-hash manifest into bk/c_lib/ (gitignored).
On import, cosmo_wap.bk silently patches the compiled methods onto the numpy classes
whenever c_lib/ exists and the hashes still match the expression files - so results
are identical (to float precision, validated ~1e-9) and nothing changes for users who
never run the build. Delete bk/c_lib/ or set COSMOWAP_DISABLE_C=1 to go back to numpy.

Each kernel is emitted split into sub-functions of ~2000 statements (temporaries live
in one scratch array) so gcc -O2 stays fast and memory-bounded even for RR2, whose
single-function form needs more RAM than most machines have.

Only real-valued l* methods are compiled; complex ones (odd multipoles with 1j) and
the ylm/m-variant path stay in numpy. sympy is imported lazily so this file is inert
when cosmo_wap.bk auto-imports it.
"""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
C_LIB = os.path.join(HERE, 'c_lib')

DEFAULT_MODULES = ['WSGR', 'WA2', 'WARR', 'RR2']
CHUNK = 500  # statements per C sub-function (gcc -O2 time grows superlinearly with size)

IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
STMT = re.compile(r'^\s+(x\d+|perm\d+) = (.+)$')
DEF = re.compile(r'^    def (l\d+)(\(.*)$')
CLS = re.compile(r'^class (\w+)')
RET = re.compile(r'^\s+return (.+)$')


def _sympify_rhs(sp, rhs, symtab, funcs):
    rhs = (rhs.replace('np.cos', 'cos').replace('np.sin', 'sin')
              .replace('np.sqrt', 'sqrt').replace('np.pi', 'pi'))
    loc = dict(funcs)
    for n in set(IDENT.findall(rhs)):
        if n not in loc:
            loc[n] = symtab.setdefault(n, sp.Symbol(n, real=True))
    return sp.sympify(rhs, locals=loc)


def _gen_method(sp, cls, meth, body_lines, ret_rhs, funcs):
    """Emit one method as chunked C sub-functions + driver. Returns (c_src, free, preamble)."""
    symtab, assigned, stmts, preamble = {}, [], [], []
    for ln in body_lines:
        m = STMT.match(ln)
        if m:
            name, rhs = m.group(1), m.group(2)
            stmts.append((name, _sympify_rhs(sp, rhs, symtab, funcs)))
            if name not in assigned:
                assigned.append(name)
        else:
            preamble.append(ln)
    ret_expr = _sympify_rhs(sp, ret_rhs, symtab, funcs)

    free = sorted(n for n in symtab if n not in assigned and n != 'pi')
    # every variable (input or temporary) gets a slot in one scratch array, so the
    # kernel can be split into small sub-functions that gcc -O2 handles cheaply
    idx = {n: i for i, n in enumerate(free + assigned)}
    nv = len(idx)

    def to_c(expr):
        code = sp.ccode(expr)
        return IDENT.sub(lambda m: f'v[{idx[m.group(0)]}]' if m.group(0) in idx else m.group(0), code)

    fname = f'{cls.lower()}_{meth}'
    out = []
    n_parts = (len(stmts) + CHUNK - 1) // CHUNK
    for p in range(n_parts):
        out.append(f'static void {fname}_p{p}(double* restrict v)\n{{')
        for name, expr in stmts[p * CHUNK:(p + 1) * CHUNK]:
            out.append(f'  v[{idx[name]}] = {to_c(expr)};')
        out.append('}\n')
    sig = ', '.join(f'const double* restrict a{i}' for i in range(len(free)))
    out.append(f'void {fname}(long n, double* restrict out, {sig})\n{{\n  double v[{nv}];')
    out.append('  for (long i = 0; i < n; i++) {')
    for i in range(len(free)):
        out.append(f'    v[{i}] = a{i}[i];')
    for p in range(n_parts):
        out.append(f'    {fname}_p{p}(v);')
    out.append(f'    out[i] = {to_c(ret_expr)};')
    out.append('  }\n}\n')
    return '\n'.join(out), free, preamble


def _parse_module(path):
    """Split a generated expression file into (cls, meth, def_line, body, ret) tuples."""
    methods, cls, meth, def_line, body = [], None, None, None, []
    for ln in open(path).read().split('\n'):
        m = CLS.match(ln)
        if m:
            cls = m.group(1)
            continue
        m = DEF.match(ln)
        if m:
            meth, def_line, body = m.group(1), f'    def {m.group(1)}{m.group(2)}', []
            continue
        if meth is None:
            continue
        m = RET.match(ln)
        if m:
            methods.append((cls, meth, def_line, body, m.group(1)))
            meth = None
        else:
            body.append(ln)
    return methods


def _build_module(mod, verbose=True):
    """Generate, compile and wrap one expression module. Returns method names built."""
    import sympy as sp
    sys.setrecursionlimit(100000)
    # abs only ever wraps wavenumbers (k1, k2), which are positive - drop it
    funcs = {'cos': sp.cos, 'sin': sp.sin, 'sqrt': sp.sqrt, 'pi': sp.pi, 'abs': sp.Id}

    src_path = os.path.join(HERE, f'{mod}.py')
    os.makedirs(C_LIB, exist_ok=True)
    c_src = ['#include <math.h>', '#define M_PI 3.14159265358979323846', '']
    py = [f'"""Generated by cosmo_wap.bk.c_compile from {mod}.py - do not edit."""',
          'import os', 'import ctypes', 'import numpy as np',
          'from numpy.ctypeslib import ndpointer', '',
          f"_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), '{mod}_kernels.so'))",
          '_arr = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")', '', '',
          'def _call(fn, free_vals):',
          '    args = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in free_vals])',
          '    shape = args[0].shape',
          '    flat = [np.ascontiguousarray(a).ravel() for a in args]',
          '    out = np.empty(flat[0].size)',
          '    fn(flat[0].size, out, *flat)',
          '    return out.reshape(shape)', '', '']
    decls, defs, built = [], [], []
    cur_cls = None
    for cls, meth, def_line, body, ret in _parse_module(src_path):
        if '1j' in ret or any('1j' in ln for ln in body):
            continue  # complex-valued (odd multipoles): stays numpy
        t0 = time.time()
        c_fn, free, preamble = _gen_method(sp, cls, meth, body, ret, funcs)
        c_src.append(c_fn)
        fname = f'{cls.lower()}_{meth}'
        decls.append(f'_lib.{fname}.restype = None')
        decls.append(f'_lib.{fname}.argtypes = [ctypes.c_long, _arr] + [_arr]*{len(free)}')
        if cls != cur_cls:
            defs.append(f'\nclass {cls}:')
            cur_cls = cls
        defs.append(def_line)
        defs.extend(ln for ln in preamble if ln.strip())
        defs.append(f'        return _call(_lib.{fname}, [{", ".join(free)}])')
        defs.append('')
        built.append(f'{cls}.{meth}')
        if verbose:
            print(f'  {mod} {cls}.{meth}: {len(body)} stmts, {len(free)} inputs '
                  f'[{time.time()-t0:.0f}s]', flush=True)
    py += decls + defs

    c_path = os.path.join(C_LIB, f'{mod}_kernels.c')
    open(c_path, 'w').write('\n'.join(c_src))
    t0 = time.time()
    subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-o',
                    os.path.join(C_LIB, f'{mod}_kernels.so'), c_path, '-lm'], check=True)
    if verbose:
        print(f'  {mod}: gcc -O2 in {time.time()-t0:.0f}s', flush=True)
    open(os.path.join(C_LIB, f'{mod}_c.py'), 'w').write('\n'.join(py))
    return built


def _src_hash(mod):
    with open(os.path.join(HERE, f'{mod}.py'), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_c_kernels(modules=None, verbose=True):
    """Build the C fast path for the given expression modules (default WS family)."""
    modules = modules or DEFAULT_MODULES
    manifest_path = os.path.join(C_LIB, 'manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    for mod in modules:
        t0 = time.time()
        if verbose:
            print(f'building {mod}...', flush=True)
        methods = _build_module(mod, verbose=verbose)
        manifest[mod] = {'sha256': _src_hash(mod), 'methods': methods}
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
        if verbose:
            print(f'  {mod} done in {time.time()-t0:.0f}s', flush=True)
    if verbose:
        print('C kernels built - they are picked up automatically on the next import.')


def _load_c_kernels(namespace):
    """Patch compiled methods onto the numpy classes in `namespace` (bk globals).

    Called from cosmo_wap.bk.__init__; a no-op unless c_lib/ exists. Kernels whose
    source expression file changed since compilation are skipped with a warning.
    """
    manifest_path = os.path.join(C_LIB, 'manifest.json')
    if os.environ.get('COSMOWAP_DISABLE_C') or not os.path.exists(manifest_path):
        return
    manifest = json.load(open(manifest_path))
    for mod, info in manifest.items():
        if _src_hash(mod) != info['sha256']:
            import warnings
            warnings.warn(f'cosmo_wap: stale C kernels for bk.{mod} (expression file changed) - '
                          f'using numpy; rerun python -m cosmo_wap.bk.c_compile')
            continue
        spec = importlib.util.spec_from_file_location(
            f'cosmo_wap.bk.c_lib.{mod}_c', os.path.join(C_LIB, f'{mod}_c.py'))
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        for name in info['methods']:
            cls_name, meth = name.split('.')
            setattr(namespace[cls_name], meth,
                    staticmethod(getattr(getattr(wrapper, cls_name), meth)))


if __name__ == '__main__':
    build_c_kernels(sys.argv[1].split(',') if len(sys.argv) > 1 else None)
