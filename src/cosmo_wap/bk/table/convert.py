"""Rewrite a generated bk expression module as a redshift-monomial coefficient table.

Every term of the bispectrum factorises into a product of redshift scalars and a factor
built only from the triangle geometry and the power spectrum:

    B = sum_m  C_m(k1,k2,k3,ct,st,Pk...) * M_m(d,K,C,f,D1,b1,b2,g2,r,s)

C_m moves only when cosmology moves, M_m is one number per monomial per call. Under
cobaya's fast/slow dragging the slow block is exactly cosmology, so C_m can be built once
per slow step and every fast step in the drag is a dot product against the cached table.

The input is the shipped CSE'd module (WA2.py, ...), not the pre-CSE source: the cse
temporaries are inlined back before expanding, so this needs nothing that is not in-tree.

Usage (one-time per module, ~80 s per method for WA2):
    python -m cosmo_wap.bk.table.convert WA2

Writes <mod>_tab.py next to this file. The result is a valid numpy module in its own
right, and is also the input c_compile consumes when building the compiled table kernel.
"""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BK = os.path.dirname(HERE)

# what a coefficient is allowed to depend on. Anything else lands in the monomial, which
# is the safe direction: an over-large table is slow, a stale one is wrong.
# mk1..3 are the redshift-free part of the PNG transfer function, introduced by _m_subs
KSYM = {'k1', 'k2', 'k3', 'ct', 'st', 'theta',
        'Pk1', 'Pk2', 'Pk3', 'Pkd1', 'Pkd2', 'Pkd3', 'Pkdd1', 'Pkdd2', 'Pkdd3',
        'mk1', 'mk2', 'mk3'}

IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
MSYM = re.compile(r'^Mk\d$')
STMT = re.compile(r'^\s+(x\d+|perm\d+|tmp_expr\w*|cse_out) = (.+)$')
DEF = re.compile(r'^    def (l\d+)(\(.*)$')
CLS = re.compile(r'^class (\w+)')
RET = re.compile(r'^\s+return (.+)$')


def _sympify(sp, expr_str):
    """Like cse_convert._sympify_expr, but with c_compile's function table.

    abs only ever wraps wavenumbers, which are positive, so it maps to the identity -
    without it `abs(k2)` parses as a call on a Symbol and raises.
    """
    expr_str = (expr_str.replace('np.cos', 'cos').replace('np.sin', 'sin')
                        .replace('np.sqrt', 'sqrt').replace('np.pi', 'pi'))
    loc = {'cos': sp.cos, 'sin': sp.sin, 'sqrt': sp.sqrt, 'pi': sp.pi, 'abs': sp.Id}
    for n in set(IDENT.findall(expr_str)):
        if n not in loc:
            loc[n] = sp.Symbol(n, real=True)
    return sp.sympify(expr_str, locals=loc)


def _split_method(lines, start, end):
    """-> (preamble lines, {name: rhs} in assignment order, return rhs)"""
    pre, stmts, ret = [], [], None
    for ln in lines[start:end]:
        m = RET.match(ln)
        if m:
            ret = m.group(1)
            break
        m = STMT.match(ln)
        if m:
            stmts.append((m.group(1), m.group(2)))
        else:
            pre.append(ln)
    return pre, stmts, ret


def _inline(sp, sympify, stmts, ret_rhs):
    """Substitute every cse temporary back into the return expression."""
    env, order = {}, []
    for name, rhs in stmts:
        e = sympify(sp, rhs)
        if name in env:
            # cse_convert chunks long sums as `perm = perm + (...)`, so a name is
            # assigned repeatedly and the rhs refers to its own previous value
            e = e.xreplace({sp.Symbol(name, real=True): env[name]})
        env[name] = e
        if name not in order:
            order.append(name)
    # assignment order is dependency order, so one forward pass resolves everything.
    # xreplace keys must be Symbols - a dict keyed by strings silently does nothing.
    resolved = {}
    for n in order:
        resolved[sp.Symbol(n, real=True)] = env[n].xreplace(resolved)
    return sympify(sp, ret_rhs).xreplace(resolved)


def _split_term(sp, term):
    """-> (monomial key, coefficient factor). Numbers and cos/sin stay in the coefficient."""
    zf, kf = [], []
    for fac in sp.Mul.make_args(term):
        b, e = fac.as_base_exp()
        nm = getattr(b, 'name', None)
        if b.is_number or nm is None or nm in KSYM:
            kf.append(fac)
        else:
            zf.append((nm, int(e)))
    return tuple(sorted(zf)), sp.Mul(*kf)


def _m_subs(sp, expr):
    """{Mk_i: D1*mk_i} for the PNG transfer functions in `expr`, empty if it has none.

    M(k,z) = sqrt(D(z)**2 * Pk(k)/Pk_phi(k)) is exactly separable into D(z) times a
    function of k alone. That matters here because runtime.table keys the coefficients on
    the cosmology and not on z - one table serves every redshift bin - so an Mk left whole
    would bake one bin's growth into a table the other bins reuse. Splitting it sends the
    growth to D1, which is already a monomial symbol, and leaves mk_i as pure k.
    """
    D1 = sp.Symbol('D1', real=True)
    return {s: D1 * sp.Symbol(s.name.lower(), real=True)
            for s in expr.free_symbols if MSYM.match(s.name)}


def convert(mod, verbose=True):
    """Convert one expression module. Emits one <class>_tab.py per class it defines."""
    import sympy as sp

    sys.setrecursionlimit(100000)

    src = os.path.join(BK, f'{mod}.py')
    lines = open(src).read().split('\n')
    # one output module per class, rather than one per source module: WSGR defines both
    # WAGR and RRGR, and keeping a single class per generated file lets MONOMIALS stay
    # flat and lets the runtime look each one up by its bk class name
    bounds = [(i, CLS.match(ln).group(1)) for i, ln in enumerate(lines) if CLS.match(ln)]
    bounds.append((len(lines), None))
    return [_convert_class(sp, mod, cls, lines, start, bounds[i + 1][0], verbose)
            for i, (start, cls) in enumerate(bounds[:-1])]


def _convert_class(sp, mod, cls, lines, cls_start, cls_end, verbose=True):
    from cosmo_wap.bk.cse_convert import _emit, _fix_pycode

    starts = [i for i in range(cls_start, cls_end) if DEF.match(lines[i])]
    starts.append(cls_end)

    out = [f'"""Generated by cosmo_wap.bk.table.convert from {mod}.py - do not edit."""',
           'import numpy as np', 'from numpy import cos, sin, sqrt, pi', '']
    mono_repr, bodies = {}, []

    for j, start in enumerate(starts[:-1]):
        meth = DEF.match(lines[start]).group(1)
        t0 = time.time()
        pre, stmts, ret_rhs = _split_method(lines, start + 1, starts[j + 1])
        expr = _inline(sp, _sympify, stmts, ret_rhs)
        msubs = _m_subs(sp, expr)
        # the preamble is reused verbatim below, so it has to define whatever _m_subs
        # introduced - Mk_i/D1 rather than a fresh sqrt, so this tracks ClassWAP.M
        pre = pre + [f'        {k.name.lower()} = {k.name}/D1' for k in sorted(msubs, key=str)]
        total = sp.expand(expr.xreplace(msubs))
        t1 = time.time()

        groups = {}
        for term in sp.Add.make_args(total):
            key, coeff = _split_term(sp, term)
            groups.setdefault(key, []).append(coeff)
        keys = sorted(groups)
        exprs = [sp.Add(*groups[k]) for k in keys]
        mono_repr[meth] = keys

        # _split_term can only see a redshift symbol when it is a bare Mul factor, so one
        # inside an Add or under a fractional power (PNG's Eq/Orth cube-root shapes) stays
        # in the coefficient. That is not the safe direction it is for a k symbol: the
        # coefficients are cached across redshift bins, so this would freeze one bin's
        # growth into all of them. Refuse rather than emit a table that is wrong per bin.
        stray = sorted({s.name for e in exprs for s in e.free_symbols} - KSYM)
        if stray:
            raise ValueError(
                f'{mod}.{cls}.{meth}: {", ".join(stray)} survived into the coefficients '
                f'rather than the monomial, so they cannot be cached across redshift bins. '
                f'This method is not tabulatable by this decomposition.')

        # one cse across all coefficients: they come from the same expression and share
        # heavily, and c_compile wants exactly this shape (x.. temporaries + named outputs)
        repl, reduced = sp.cse(exprs, optimizations='basic')
        body = []
        for s_, e in repl:
            _emit(sp, e, str(s_), body, '        ')
        for i, e in enumerate(reduced):
            _emit(sp, e, f'tmp_expr{i}', body, '        ')
        ret = '[' + ', '.join(f'tmp_expr{i}' for i in range(len(reduced))) + ']'

        bodies.append((meth, lines[start], pre, body, ret))
        if verbose:
            print(f'  {mod}.{cls}.{meth}: {len(sp.Add.make_args(total))} terms -> '
                  f'{len(keys)} monomials, {len(body)} stmts [inline+expand {t1-t0:.0f}s, '
                  f'cse {time.time()-t1:.0f}s]', flush=True)

    # The redshift scalars come from the source preamble reused verbatim, rather than
    # from anything this package knows about get_params: modules differ in what they
    # unpack (RR2 also calls get_derivs), and the signature carries r/s/nonlin/growth2.
    znames = sorted({n for keys in mono_repr.values() for k in keys for n, _ in k})
    zdef, zpre = bodies[0][1], bodies[0][2]
    out += [re.sub(r'^\s*def l\d+', 'def zvals', zdef),
            '    """The redshift scalars of MONOMIALS, by name.',
            '',
            '    None of them depends on k, so the caller may pass a single triangle and',
            '    skip the power spectrum splines, which are the expensive part of the',
            '    preamble and belong to the cached coefficients instead."""']
    out += [ln[4:] if ln.startswith('    ') else ln for ln in zpre]
    out += ['    return {' + ', '.join(f'{n!r}: {n}' for n in znames) + '}', '',
            '# B = sum_m COEFF[m] * prod(sym**exp for sym, exp in MONOMIALS[meth][m])',
            'MONOMIALS = {']
    for meth, keys in mono_repr.items():
        out.append(f'    {meth!r}: [')
        out += [f'        {k!r},' for k in keys]
        out.append('    ],')
    out += ['}', '', f'class {cls}_tab:']
    for meth, def_line, pre, body, ret in bodies:
        # keep the original signature and unpacking preamble verbatim so the generated
        # module stays a drop-in numpy implementation as well as c_compile input
        out.append(re.sub(r'^    def l', '    def l', def_line))
        out += [ln for ln in pre]
        out += [_fix_pycode(b) for b in body]
        out.append(f'        return {ret}')
        out.append('')

    dst = os.path.join(HERE, f'{cls}_tab.py')
    open(dst, 'w').write('\n'.join(out))
    if verbose:
        print(f'  wrote {dst} ({os.path.getsize(dst)/1e6:.1f} MB)', flush=True)
    return dst


if __name__ == '__main__':
    for m in (sys.argv[1] if len(sys.argv) > 1 else 'WA2').split(','):
        print(f'{m}:', flush=True)
        convert(m)
