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

Each kernel is emitted split into sub-functions of ~500 statements (temporaries live in
one scratch array), and those sub-functions are spread over many small translation units
compiled in parallel, so gcc -O2 stays fast and memory-bounded even for RR2, whose
single-file form needs more RAM and time than most machines have.

The real-valued kernels are generated WIDTH triangles at a time using GCC's vector
extensions, defaulting to 4 (set COSMOWAP_SIMD_WIDTH to override; 1 restores the plain
scalar codegen). These expressions are >99% plain +,-,* on the scratch array, so each
statement becomes the same operation on WIDTH lanes; pow/sqrt are evaluated a lane at a
time and nothing is reassociated, so results are bit-identical to the scalar kernel at
any width. The win is instruction fetch: a kernel streams megabytes of machine code past
a 32 KB L1i once per triangle, and at WIDTH=4 that happens once per four triangles
instead - measured ~3.4x on RR2.l0 with a single chain running.

Wider is not always better. The scratch array grows with WIDTH too, so N concurrent
chains need N x WIDTH x (temporaries) of cache between them; past the shared L3 the win
inverts. On a 6-core/16 MB machine WIDTH=8 measured 5.2x with one chain and 0.69x with
six, while WIDTH=4 held 1.9-3.6x across the range. Raise it only when a chain has a
socket largely to itself. The odd multipoles (complex temporaries) stay scalar.

Within one call the redshift is a single number, so `f`, `D1`, `b1`, `r`, `s` and the rest
of the z-dependent inputs hold the same value for every triangle - yet ~90% of RR2's CSE
temporaries (40% of its arithmetic) are built from those alone, and were being recomputed
per triangle, WIDTH identical lanes at a time. Those statements are taint-separated into a
prologue that runs once per thread into a scalar scratch array, leaving only the genuinely
per-triangle statements in the loop; measured ~1.5x on WA2 beyond the vector win, and it
shrinks the hot working set several-fold, which matters most when chains contend for L3.
Set COSMOWAP_HOIST=0 to disable. Z_SCALARS below is the list of inputs treated as constant
across a call - the wrapper checks at runtime and falls back to grouping the triangles by
redshift if one of them turns out to vary, so an array zz still gives correct results.

The triangle loop is OpenMP-enabled but runs serial unless OMP_NUM_THREADS is set, so
MPI/multi-chain jobs that saturate their cores with processes are never oversubscribed.
Set OMP_NUM_THREADS to the cores available per chain to thread each likelihood call
(results are bitwise identical at any thread count - triangles are independent).

Complex-valued methods (odd multipoles, e.g. WA1.l1/RR1.l1) are supported: temporaries
are taint-tracked so only statements touching the imaginary unit use C99 double complex
(via tgmath.h), the rest stay real doubles. The ylm/m-variant path stays in numpy.
sympy is imported lazily so this file is inert when cosmo_wap.bk auto-imports it.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
C_LIB = os.path.join(HERE, 'c_lib')

DEFAULT_MODULES = ['WSGR', 'WA2', 'WARR', 'RR2', 'WA1', 'RR1', 'GR0', 'GR1', 'GR2', 'PNG']
CHUNK = 500  # statements per C sub-function (gcc -O2 time grows superlinearly with size)

# triangles evaluated per pass. Each kernel streams its whole machine code (RR2.l0 ~6 MB)
# past a 32 KB L1i once per triangle, so the cost is instruction fetch, not arithmetic;
# at width W that fetch is amortised W-fold. 4 is the default because it is the width that
# survives contention: the scratch array grows W-fold too (RR2 2 MB/chain at W=4, 4 MB at
# W=8), and once N chains x scratch exceeds the shared L3 the win collapses - measured on a
# 6-core/16 MB box, W=8 goes from 5.2x at one chain to 0.69x at six, while W=4 holds 1.9-3.6x
# throughout. W=8 is worth setting only when a chain has a socket largely to itself.
# W=1 restores the original scalar codegen exactly.
WIDTH = int(os.environ.get('COSMOWAP_SIMD_WIDTH', 4))
SUBS_PER_TU = 2       # sub-functions per translation unit (see _build_module)
JOBS = int(os.environ.get('COSMOWAP_BUILD_JOBS', min(4, os.cpu_count() or 1)))

# Inputs that get_params/get_derivs/get_PNGparams/get_beta_funcs evaluate at zz alone. At
# scalar zz they hold the same value for every triangle, so every temporary built only from
# them is loop-invariant and is computed once in a prologue rather than n times (and, at
# WIDTH>1, in WIDTH identical lanes). Measured ~1.5x on WA2 on top of the vector win.
# A name missing from this list is treated as per-triangle: the kernel is still correct,
# it just loses the saving, so erring towards omission is safe. Mk1/Mk2/Mk3 (PNG) belong
# with the k-dependent inputs - only bE01/bE11 out of get_PNGparams are z-only.
Z_SCALARS = {'d', 'K', 'C', 'f', 'D1', 'b1', 'b2', 'g2', 'r', 's', 'zz',
             'fd', 'Dd', 'gd2', 'bd2', 'bd1', 'fdd', 'Ddd', 'gdd2', 'bdd2', 'bdd1',
             'fNL', 'b01', 'b11', 'gr1', 'gr2'} | {f'beta{i}' for i in range(6, 20)}
HOIST = os.environ.get('COSMOWAP_HOIST', '1') == '1' and WIDTH > 1

# Two ways to feed hoisted values back into the triangle loop.
# 'scalar': hot statements read them straight out of the prologue's double array. Smallest
#   working set and the fastest kernel, but gcc must broadcast at each use, and in the
#   million-character statements cse_convert emits that doubles its peak memory: RR2's worst
#   unit needs 6.5 GB against 3.3 GB unhoisted, which no 8 GB machine will build.
# 'bcast' : splat them into vector slots at the end of the prologue, so every hot statement
#   is pure-vector as before. Same arithmetic saving, larger working set, and it compiles in
#   the same memory as the unhoisted build.
# The blowup tracks the number of distinct invariants the hot statements read, so that is
# what selects the mode (WARR 12.7k reads builds fine scalar; RR2 47k does not).
HOIST_MODE = os.environ.get('COSMOWAP_HOIST_MODE', 'auto')
BCAST_ABOVE = int(os.environ.get('COSMOWAP_BCAST_ABOVE', 20000))

# maths sympy's C printer can emit. Only the ones actually used get a per-lane wrapper;
# anything outside this list raises rather than reaching gcc, because a missed name is a
# type error in a 2 MB line and the diagnostic is unreadable (cbrt, from the PNG shape
# templates' cube roots, is how this list got written).
VEC_BINARY = ('pow', 'atan2')
VEC_UNARY = ('sqrt', 'cbrt', 'exp', 'log', 'fabs', 'cos', 'sin', 'tan',
             'acos', 'asin', 'atan', 'cosh', 'sinh', 'tanh')
SCALAR_FUNCS = VEC_BINARY + VEC_UNARY
CALL = re.compile(r'(?<![A-Za-z_0-9])([A-Za-z_][A-Za-z_0-9]*)\(')

# -ffp-contract=off keeps gcc from fusing a*b+c into an FMA. That is what makes the vector
# build bit-identical to the scalar one, and it is also what makes -mavx2/-mfma safe here:
# these expressions carry real cancellation, and contraction alone shifted results by 1.2e-7.
CFLAGS = ['-O2', '-fopenmp', '-fPIC', '-ffp-contract=off']
if WIDTH > 1:
    # AVX2 exists on every x86-64 part since ~2013, so a c_lib built on a login node still
    # runs on the compute nodes; -march=native would risk SIGILL on a heterogeneous cluster.
    CFLAGS += ['-mavx2', '-mfma']

IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
STMT = re.compile(r'^\s+(x\d+|perm\d+|tmp_expr\w*|cse_out) = (.+)$')
DEF = re.compile(r'^    def (l\d+)(\(.*)$')
CLS = re.compile(r'^class (\w+)')
RET = re.compile(r'^\s+return (.+)$')


def _sympify_rhs(sp, rhs, symtab, funcs):
    rhs = (rhs.replace('np.cos', 'cos').replace('np.sin', 'sin')
              .replace('np.sqrt', 'sqrt').replace('np.pi', 'pi')
              .replace('1j', 'I'))  # complex literal -> sympy I (ccode prints it as C99 I)
    loc = dict(funcs)
    for n in set(IDENT.findall(rhs)):
        if n not in loc:
            loc[n] = symtab.setdefault(n, sp.Symbol(n, real=True))
    return sp.sympify(rhs, locals=loc)


def _split_ret(rhs):
    """Return-statement right-hand side -> list of expression strings.

    `return [a, b, c]` is a coefficient table; anything else is a single expression.
    Splitting is depth-aware so commas inside pow(x, 2) do not split a term.
    """
    rhs = rhs.strip()
    if not rhs.startswith('['):
        return [rhs]
    parts, depth, cur = [], 0, []
    for ch in rhs[1:-1]:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if ''.join(cur).strip():
        parts.append(''.join(cur))
    return [p.strip() for p in parts]


def _vec_calls(s, used, tag='vr['):
    """Point maths calls at the per-lane helpers, but only where the argument is a vector.

    Most statements touch the scratch array, but constant subexpressions do not - the
    return prefactor calls sqrt on M_PI, and those have to stay scalar or the types clash.
    Runs to a fixed point so nested calls are reached; the rewritten v-names are not
    re-matched because the character before them fails the word-boundary test. Records
    which functions were vectorised in `used` so only those get a wrapper emitted.
    """
    for name in SCALAR_FUNCS:
        vname = 'v' + name
        while True:
            out, i, hit = [], 0, False
            while i < len(s):
                k = s.find(name + '(', i)
                if k < 0:
                    out.append(s[i:])
                    break
                if k and (s[k - 1].isalnum() or s[k - 1] == '_'):
                    out.append(s[i:k + len(name) + 1])
                    i = k + len(name) + 1
                    continue
                depth, j = 0, k + len(name)
                while j < len(s):
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                arg = s[k + len(name) + 1:j]
                out.append(s[i:k])
                if tag in arg:
                    out.append(f'{vname}({arg})')
                    used.add(name)
                    hit = True
                else:
                    out.append(f'{name}({arg})')
                i = j + 1
            s = ''.join(out)
            if not hit:
                break
    # a maths call this module does not know how to widen would otherwise surface as a
    # type error inside a multi-megabyte statement, so fail here with the name instead
    unknown = {c for c in CALL.findall(s)
               if c not in SCALAR_FUNCS and c.lstrip('v') not in SCALAR_FUNCS}
    if unknown:
        raise ValueError(f'cannot vectorise call(s) {sorted(unknown)}: add them to '
                         f'VEC_UNARY/VEC_BINARY in c_compile.py, or build with '
                         f'COSMOWAP_SIMD_WIDTH=1')
    return s


def _gen_method(sp, cls, meth, body_lines, ret_rhs, funcs):
    """Emit one method as C sub-functions + driver.
    Returns (subs, driver, free, preamble, out_complex, used_funcs, n_inv_free, n_out)."""
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
    # an ordinary expression module returns one bispectrum value; a generated coefficient
    # table (bk/table/) returns the list of monomial coefficients instead
    ret_exprs = [_sympify_rhs(sp, r, symtab, funcs) for r in _split_ret(ret_rhs)]
    n_out = len(ret_exprs)

    # taint-track complexity: a temporary is complex iff its expression references I or a
    # complex temp (to fixpoint). Real temps keep plain double so tgmath dispatches the
    # fast real pow/sqrt - only the (few) tainted statements pay for complex arithmetic.
    cplx_names = set()
    changed = True
    while changed:
        changed = False
        for name, expr in stmts:
            if name not in cplx_names and (
                    expr.has(sp.I) or any(str(s) in cplx_names for s in expr.free_symbols)):
                cplx_names.add(name)
                changed = True
    out_complex = any(e.has(sp.I) or any(str(s) in cplx_names for s in e.free_symbols)
                      for e in ret_exprs)

    free = sorted(n for n in symtab if n not in assigned and n not in ('pi', 'I'))
    # every variable (input or temporary) gets a slot in a scratch array (real or complex),
    # so the kernel can be split into small sub-functions that gcc -O2 handles cheaply
    reals = free + [n for n in assigned if n not in cplx_names]
    cplx = [n for n in assigned if n in cplx_names]

    # the odd multipoles carry complex temporaries, which GCC's vector extensions do not
    # support - they keep the original scalar codegen (they are also the cheap kernels)
    vector = WIDTH > 1 and not cplx and not out_complex
    hoist = HOIST and vector

    n_inv_free = 0
    if hoist:
        # taint from the per-triangle inputs; whatever none of them reaches is loop-invariant.
        # cse emits its temporaries in dependency order, so one forward pass is a fixpoint.
        varying = {n for n in free if n not in Z_SCALARS}
        for name, expr in stmts:
            if any(str(sym) in varying for sym in expr.free_symbols):
                varying.add(name)
        inv = [n for n in reals if n not in varying]
        idx = {n: f'vi[{i}]' for i, n in enumerate(inv)}                     # prologue slots
        # invariants the per-triangle statements actually read; the rest never leave the
        # prologue, so they cost nothing in the loop whichever mode is chosen
        read = {str(sym) for e in ret_exprs for sym in e.free_symbols} - varying
        for name, expr in stmts:
            if name in varying:
                read |= {str(sym) for sym in expr.free_symbols} - varying
        live = [n for n in inv if n in read]
        mode = HOIST_MODE if HOIST_MODE in ('scalar', 'bcast') else (
            'bcast' if len(live) > BCAST_ABOVE else 'scalar')
        var = [n for n in reals if n in varying]
        if mode == 'bcast':
            hot = {n: f'vv[{i}]' for i, n in enumerate(live)}
            hot.update({n: f'vv[{i}]' for i, n in enumerate(var, start=len(live))})
            # the splats go in as a table plus a loop, not one statement each: RR2 has 47k
            # of them per method, and emitting them as code put 92k statements in the single
            # driver function, which gcc was still chewing on after half an hour
            pos = {n: i for i, n in enumerate(inv)}
            bcast = [pos[n] for n in live]
        else:
            hot = dict(idx)
            hot.update({n: f'vv[{i}]' for i, n in enumerate(var)})
            bcast = []
        # invariant inputs come first so the wrapper can pass them as scalars
        free = [n for n in free if n not in varying] + [n for n in free if n in varying]
        n_inv_free = sum(1 for n in free if n not in varying)
    else:
        idx = {n: f'vr[{i}]' for i, n in enumerate(reals)}
        idx.update({n: f'vc[{i}]' for i, n in enumerate(cplx)})
        hot = idx

    used = set()
    tag = 'vv[' if hoist else 'vr['

    def to_c(expr, mp=None, want_vec=False):
        s = IDENT.sub(lambda m: (mp or idx).get(m.group(0), m.group(0)), sp.ccode(expr))
        if not vector:
            return s
        s = _vec_calls(s, used, tag)
        # a name can be assigned twice (cse_convert chunks long sums as `perm = perm + ...`),
        # so a slot on the varying side can still take a wholly invariant right-hand side.
        # GCC does not broadcast a scalar into a vector on assignment, so splat it.
        return f'vbc({s})' if want_vec and tag not in s else s

    def emit_out(pad, mp=None, want_vec=False, vec=True):
        """Write the return values into out, laid out coefficient-major.

        With n_out > 1 the result is an (n_out, n) table, which is the shape the runtime
        contracts a monomial basis against - and keeps each coefficient contiguous.
        """
        ls = []
        if not vec:
            return [f'{pad}out[{m}*n + i] = {to_c(e, mp=mp)};' if n_out > 1 else
                    f'{pad}out[i] = {to_c(e, mp=mp)};' for m, e in enumerate(ret_exprs)]
        for m, e in enumerate(ret_exprs):
            ls.append(f'{pad}const vdf _r{m} = {to_c(e, mp=mp, want_vec=want_vec)};')
        ls.append(f'{pad}for (int j = 0; j < {WIDTH} && i0 + j < n; j++) {{')
        for m in range(n_out):
            tgt = f'out[{m}*n + i0 + j]' if n_out > 1 else 'out[i0 + j]'
            ls.append(f'{pad}  {tgt} = _r{m}[j];')
        ls.append(f'{pad}}}')
        return ls

    fname = f'{cls.lower()}_{meth}'
    vtype = 'vdf' if vector else 'double'
    if hoist:
        args_decl, call_args = 'const double* restrict vi, vdf* restrict vv', 'vi, vv'
    else:
        args_decl = f'{vtype}* restrict vr' + (', double complex* restrict vc' if cplx else '')
        call_args = 'vr' + (', vc' if cplx else '')
    # sub-functions have external linkage so they can be split across translation units
    subs = []
    if hoist:
        i_stmts = [(n, e) for n, e in stmts if n not in varying]
        v_stmts = [(n, e) for n, e in stmts if n in varying]
    else:
        i_stmts, v_stmts = [], stmts
    n_iparts = (len(i_stmts) + CHUNK - 1) // CHUNK
    for p in range(n_iparts):
        blk = [f'void {fname}_i{p}(double* restrict vi)', '{']
        for name, expr in i_stmts[p * CHUNK:(p + 1) * CHUNK]:
            blk.append(f'  {idx[name]} = {to_c(expr)};')
        blk.append('}')
        subs.append('\n'.join(blk))
    n_parts = (len(v_stmts) + CHUNK - 1) // CHUNK
    for p in range(n_parts):
        blk = [f'void {fname}_p{p}({args_decl})', '{']
        for name, expr in v_stmts[p * CHUNK:(p + 1) * CHUNK]:
            blk.append(f'  {hot[name]} = {to_c(expr, mp=hot, want_vec=hoist)};')
        blk.append('}')
        subs.append('\n'.join(blk))

    sig = ', '.join(f'const double* restrict a{i}' for i in range(len(free)))
    out_type = 'double complex' if out_complex else 'double'
    out = [f'void {fname}(long n, {out_type}* restrict out, {sig})\n{{']
    # triangles are independent (scratch is per-iteration), so the loop threads cleanly;
    # serial unless OMP_NUM_THREADS is explicitly set so cluster jobs are never oversubscribed
    out.append('  const int use_omp = getenv("OMP_NUM_THREADS") != NULL;')
    if hoist:
        out.append(f'  const long nb = (n + {WIDTH} - 1) / {WIDTH};')
        out.append('  #pragma omp parallel if(use_omp)')
        out.append('  {')
        # thread-local rather than shared: a worker must never read another thread's
        # half-built prologue. It is written once per thread, then read-only.
        out.append(f'    static _Thread_local double vi[{max(len(inv), 1)}];')
        out.append(f'    static _Thread_local vdf vv[{max(len(var) + len(bcast), 1)}];')
        for i, n_ in enumerate(free[:n_inv_free]):
            out.append(f'    {idx[n_]} = a{i}[0];')
        for p in range(n_iparts):
            out.append(f'    {fname}_i{p}(vi);')
        if bcast:
            out.append(f'    static const int bmap[{len(bcast)}] = {{'
                       + ','.join(str(i) for i in bcast) + '};')
            out.append(f'    for (int i = 0; i < {len(bcast)}; i++) vv[i] = vbc(vi[bmap[i]]);')
        out.append('    #pragma omp for')
        out.append('    for (long b = 0; b < nb; b++) {')
        out.append(f'      const long i0 = b * {WIDTH};')
        out.append(f'      for (int j = 0; j < {WIDTH}; j++) {{')
        out.append('        const long ii = (i0 + j < n) ? i0 + j : n - 1;  /* pad tail */')
        for i, n_ in enumerate(free[n_inv_free:], start=n_inv_free):
            out.append(f'        {hot[n_]}[j] = a{i}[ii];')
        out.append('      }')
        for p in range(n_parts):
            out.append(f'      {fname}_p{p}({call_args});')
        out += emit_out('      ', mp=hot, want_vec=True)
        out.append('    }')
        out.append('  }\n}\n')
        return subs, '\n'.join(out), free, preamble, out_complex, used, n_inv_free, n_out
    if vector:
        # every statement becomes the same IEEE operation on WIDTH lanes, in the same order -
        # no reassociation - so the result is bit-identical to the scalar kernel
        out.append(f'  const long nb = (n + {WIDTH} - 1) / {WIDTH};')
        out.append('  #pragma omp parallel for if(use_omp)')
        out.append('  for (long b = 0; b < nb; b++) {')
        # thread-local rather than automatic: at WIDTH=4 this is up to 2 MB (RR2), too much
        # to put on an OpenMP worker's stack
        out.append(f'    static _Thread_local vdf vr[{len(reals)}];')
        out.append(f'    const long i0 = b * {WIDTH};')
        out.append(f'    for (int j = 0; j < {WIDTH}; j++) {{')
        out.append('      const long ii = (i0 + j < n) ? i0 + j : n - 1;  /* pad tail */')
        for i in range(len(free)):
            out.append(f'      vr[{i}][j] = a{i}[ii];')
        out.append('    }')
        for p in range(n_parts):
            out.append(f'    {fname}_p{p}({call_args});')
        out += emit_out('    ')
        out.append('  }\n}\n')
    else:
        out.append('  #pragma omp parallel for if(use_omp)')
        out.append('  for (long i = 0; i < n; i++) {')
        out.append(f'    double vr[{len(reals)}];')
        if cplx:
            out.append(f'    double complex vc[{len(cplx)}];')
        for i in range(len(free)):
            out.append(f'    vr[{i}] = a{i}[i];')
        for p in range(n_parts):
            out.append(f'    {fname}_p{p}({call_args});')
        out += emit_out('    ', vec=False)
        out.append('  }\n}\n')
    return subs, '\n'.join(out), free, preamble, out_complex, used, 0, n_out


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
    funcs = {'cos': sp.cos, 'sin': sp.sin, 'sqrt': sp.sqrt, 'pi': sp.pi, 'abs': sp.Id, 'I': sp.I}

    src_path = _src_path(mod)
    os.makedirs(C_LIB, exist_ok=True)
    # tgmath.h makes sqrt/pow/cos/sin type-generic so the complex-tainted statements
    # dispatch to csqrt/cpow etc. while real statements keep the fast real versions
    header = ['#include <math.h>', '#include <complex.h>', '#include <tgmath.h>',
              '#include <stdlib.h>', '#define M_PI 3.14159265358979323846', '']
    py = [f'"""Generated by cosmo_wap.bk.c_compile from {mod}.py - do not edit."""',
          'import os', 'import ctypes', 'import numpy as np',
          'from numpy.ctypeslib import ndpointer', '',
          f"_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), '{mod}_kernels.so'))",
          '_arr = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")',
          '_arrc = ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")', '', '',
          'def _call(fn, free_vals, n_inv=0, out_dtype=np.float64, n_out=1):',
          '    """n_inv leading values are hoisted out of the triangle loop, so the kernel',
          '    takes them as scalars. Should they vary across the call (an array zz), the',
          '    triangles are grouped by their redshift values and the kernel is run once per',
          '    group - correct, and no slower than one call per distinct redshift.',
          '',
          '    n_out > 1 is a coefficient table: the kernel writes n_out values per triangle',
          '    and the result is (n_out,) + shape. Such a kernel has no hoisted scalars by',
          '    construction (its coefficients carry no redshift dependence), so it never',
          '    reaches the grouping branch above."""',
          '    vals = [np.asarray(v, dtype=np.float64) for v in free_vals]',
          '    if n_out > 1 and n_inv:',
          '        raise ValueError("n_out > 1 with hoisted scalars is not supported")',
          '    if n_inv and any(a.size != 1 for a in vals[:n_inv]):',
          '        full = np.broadcast_arrays(*vals)',
          '        shape = full[0].shape',
          '        flat = [np.ascontiguousarray(a).ravel() for a in full]',
          '        out = np.empty(flat[0].size, dtype=out_dtype)',
          '        key = np.stack([f for f in flat[:n_inv]])',
          '        _, first, inverse = np.unique(key, axis=1, return_index=True,',
          '                                      return_inverse=True)',
          '        for j in range(first.size):',
          '            m = np.flatnonzero(inverse.ravel() == j)',
          '            sub = np.empty(m.size, dtype=out_dtype)',
          '            fn(m.size, sub,',
          '               *[np.ascontiguousarray(f[first[j]:first[j] + 1]) for f in flat[:n_inv]],',
          '               *[np.ascontiguousarray(f[m]) for f in flat[n_inv:]])',
          '            out[m] = sub',
          '        return out.reshape(shape)',
          '    inv = [np.ascontiguousarray(a, dtype=np.float64).ravel() for a in vals[:n_inv]]',
          '    args = np.broadcast_arrays(*vals[n_inv:])',
          '    shape = args[0].shape',
          '    flat = [np.ascontiguousarray(a).ravel() for a in args]',
          '    out = np.empty(flat[0].size * n_out, dtype=out_dtype)',
          '    fn(flat[0].size, out, *inv, *flat)',
          '    return out.reshape((n_out,) + shape if n_out > 1 else shape)', '', '']
    decls, defs, built = [], [], []
    all_subs, drivers, used_funcs = [], [], set()
    cur_cls = None
    for cls, meth, def_line, body, ret in _parse_module(src_path):
        t0 = time.time()
        subs, driver, free, preamble, out_complex, used, n_inv, n_out = _gen_method(
            sp, cls, meth, body, ret, funcs)
        used_funcs |= used
        all_subs.extend(subs)
        drivers.append(driver)
        fname = f'{cls.lower()}_{meth}'
        out_arr = '_arrc' if out_complex else '_arr'
        decls.append(f'_lib.{fname}.restype = None')
        decls.append(f'_lib.{fname}.argtypes = [ctypes.c_long, {out_arr}] + [_arr]*{len(free)}')
        if cls != cur_cls:
            defs.append(f'\nclass {cls}:')
            cur_cls = cls
        defs.append(def_line)
        defs.extend(ln for ln in preamble if ln.strip())
        dtype_kw = ', out_dtype=np.complex128' if out_complex else ''
        n_out_kw = f', n_out={n_out}' if n_out > 1 else ''
        defs.append(f'        return _call(_lib.{fname}, [{", ".join(free)}], '
                    f'{n_inv}{dtype_kw}{n_out_kw})')
        defs.append('')
        built.append(f'{cls}.{meth}')
        if verbose:
            print(f'  {mod} {cls}.{meth}: {len(body)} stmts, {len(free)} inputs '
                  f'[{time.time()-t0:.0f}s]', flush=True)
    py += decls + defs

    # one translation unit per couple of sub-functions rather than one per module: gcc's
    # cost is dominated by the largest unit, and the units are independent, so this both
    # bounds peak memory and lets them compile in parallel. RR2 as a single file did not
    # finish in an hour; split it builds in ~11 min, nearly all of that one unit holding
    # the chunked `perm = perm + (...)` sums.
    if WIDTH > 1:
        # these have no vector form, but they are a fraction of a percent of the statements
        # (56 of 19003 in WARR.l0), so doing them a lane at a time through the scalar libm
        # costs nothing and is what keeps the answer bit-identical to the scalar kernel
        header.append(f'typedef double vdf __attribute__((vector_size({8 * WIDTH})));')
        if HOIST:
            header += ['static inline vdf vbc(double x)',
                       f'{{ vdf r; for (int j = 0; j < {WIDTH}; j++) r[j] = x; return r; }}']
        for name in sorted(used_funcs):
            arg2 = ', double e' if name in VEC_BINARY else ''
            call2 = ', e' if name in VEC_BINARY else ''
            header += [f'static inline vdf v{name}(vdf x{arg2})',
                       f'{{ vdf r; for (int j = 0; j < {WIDTH}; j++) '
                       f'r[j] = {name}(x[j]{call2}); return r; }}']
        header.append('')

    src_dir = os.path.join(C_LIB, f'{mod}_src')
    shutil.rmtree(src_dir, ignore_errors=True)
    os.makedirs(src_dir)
    protos = [s.split('\n', 1)[0] + ';' for s in all_subs]
    open(os.path.join(src_dir, 'kern.h'), 'w').write(
        '#pragma once\n' + '\n'.join(header + protos) + '\n')
    units = []
    for i in range(0, len(all_subs), SUBS_PER_TU):
        unit = f'part{i // SUBS_PER_TU:04d}.c'
        open(os.path.join(src_dir, unit), 'w').write(
            '#include "kern.h"\n\n' + '\n\n'.join(all_subs[i:i + SUBS_PER_TU]) + '\n')
        units.append(unit)
    open(os.path.join(src_dir, 'driver.c'), 'w').write(
        '#include "kern.h"\n\n' + '\n\n'.join(drivers) + '\n')
    units.append('driver.c')

    t0 = time.time()

    def _cc(unit):
        subprocess.run(['gcc'] + CFLAGS + ['-c', unit, '-o', unit + '.o'],
                       cwd=src_dir, check=True)
        return unit + '.o'

    with ThreadPoolExecutor(JOBS) as ex:
        objs = list(ex.map(_cc, units))
    subprocess.run(['gcc', '-shared', '-fopenmp', '-o',
                    os.path.join(C_LIB, f'{mod}_kernels.so')] + objs + ['-lm'],
                   cwd=src_dir, check=True)
    if verbose:
        print(f'  {mod}: gcc -O2 W={WIDTH}, {len(units)} units on {JOBS} jobs '
              f'in {time.time()-t0:.0f}s', flush=True)
    open(os.path.join(C_LIB, f'{mod}_c.py'), 'w').write('\n'.join(py))
    return built


def _src_path(mod):
    """Expression modules sit in bk/, the generated coefficient tables in bk/table/."""
    if mod.endswith('_tab'):
        return os.path.join(HERE, 'table', f'{mod}.py')
    return os.path.join(HERE, f'{mod}.py')


def _src_hash(mod):
    with open(_src_path(mod), 'rb') as f:
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
        manifest[mod] = {'sha256': _src_hash(mod), 'methods': methods, 'width': WIDTH,
                         'hoist': HOIST}
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
            # coefficient tables (bk/table/) have no class in the bk namespace - they are
            # loaded by their own runtime - and a stale manifest entry should not break import
            if cls_name not in namespace:
                continue
            setattr(namespace[cls_name], meth,
                    staticmethod(getattr(getattr(wrapper, cls_name), meth)))


if __name__ == '__main__':
    build_c_kernels(sys.argv[1].split(',') if len(sys.argv) > 1 else None)
