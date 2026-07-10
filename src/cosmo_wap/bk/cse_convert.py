"""Convert a generated cosmo_wap expression module into a CSE'd drop-in replacement.

The machine-generated term files (WA2.py, RR2.py, ...) are fully expanded sympy
expressions printed as single numpy megalines; evaluating them is dominated by numpy
ufunc-dispatch overhead (one call per arithmetic op). This script rewrites each
`perm.. = <huge expr>` assignment as sympy.cse temporaries + chunked sums, which cuts
the op count (and runtime) by ~3-9x while leaving results identical to float precision.
Every other line - signatures, unpacking preambles, return lines/prefactors - is kept
verbatim, so the output is a drop-in replacement.

Usage (one-time per file, minutes for small files, ~15 min for RR2):
    python cse_convert.py WA2,WARR,RR2 [out_dir]

Writes converted modules to out_dir (default: ./cse_out next to this script); inspect
and validate against the originals before copying them over.

sympy is imported lazily so this file stays inert when cosmo_wap.bk auto-imports it.
"""
import re
import sys
import time

IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')


def _sympify_expr(sp, expr_str):
    expr_str = (expr_str.replace('np.cos', 'cos').replace('np.sin', 'sin')
                        .replace('np.sqrt', 'sqrt').replace('np.pi', 'pi'))
    loc = {'cos': sp.cos, 'sin': sp.sin, 'sqrt': sp.sqrt, 'pi': sp.pi}
    for n in set(IDENT.findall(expr_str)):
        if n not in loc:
            loc[n] = sp.Symbol(n, real=True)
    return sp.sympify(expr_str, locals=loc)


def _emit(sp, e, target, body, indent, chunk=40):
    """emit `target = e`, splitting long sums into accumulated chunks (keeps the AST shallow)"""
    if isinstance(e, sp.Add) and len(e.args) > chunk:
        args = e.args
        body.append(f"{indent}{target} = " + sp.pycode(sp.Add(*args[:chunk], evaluate=False)))
        for i in range(chunk, len(args), chunk):
            body.append(f"{indent}{target} = {target} + ("
                        + sp.pycode(sp.Add(*args[i:i + chunk], evaluate=False)) + ")")
    else:
        body.append(f"{indent}{target} = {sp.pycode(e)}")


def _fix_pycode(line):
    # sp.pycode emits math.* for scalar-typed subexpressions; the module namespace
    # provides the numpy versions instead (see header injection below)
    return (line.replace('math.cos', 'cos').replace('math.sin', 'sin')
                .replace('math.sqrt', 'sqrt').replace('math.pi', 'pi'))


def _convert_module(path, out_path, min_len=2000, verbose=True):
    """Rewrite perm assignments longer than min_len chars using CSE.
    Consecutive perm lines (the permutations of one function body) go through a single
    cse() call so subexpressions are shared across permutations."""
    import sympy as sp
    sys.setrecursionlimit(100000)  # sympify recurses on the megaline ASTs

    src = open(path).read()
    lines = src.split('\n')
    out_lines = []
    i = 0
    while i < len(lines):
        m_ret = re.match(r'(\s+)return (.+)$', lines[i])
        if m_ret and len(lines[i]) > min_len:
            # bk_mt style: whole expression inline in the return statement
            indent = m_ret.group(1)
            t0 = time.time()
            expr = _sympify_expr(sp, m_ret.group(2))
            t1 = time.time()
            repl, reduced = sp.cse([expr], optimizations='basic')
            t2 = time.time()
            body = []
            for s_, e in repl:
                _emit(sp, e, str(s_), body, indent)
            _emit(sp, reduced[0], 'cse_out', body, indent)
            body.append(f"{indent}return cse_out")
            out_lines.extend(_fix_pycode(b) for b in body)
            if verbose:
                print(f"  {path.split('/')[-1]}: return expr "
                      f"(src {len(m_ret.group(2))/1e6:.2f}MB) -> {len(body)} stmts "
                      f"[sympify {t1-t0:.0f}s, cse {t2-t1:.0f}s]", flush=True)
            i += 1
            continue
        m = re.match(r'(\s+)(perm\d+|tmp_expr\w*) = (.+)$', lines[i])
        if m and len(lines[i]) > min_len:
            indent = m.group(1)
            group = []  # consecutive perm lines
            while i < len(lines):
                m2 = re.match(r'(\s+)(perm\d+|tmp_expr\w*) = (.+)$', lines[i])
                if not m2:
                    break
                group.append((m2.group(2), m2.group(3)))
                i += 1
            t0 = time.time()
            exprs = [_sympify_expr(sp, e) for _, e in group]
            t1 = time.time()
            repl, reduced = sp.cse(exprs, optimizations='basic')
            t2 = time.time()
            body = []
            for s_, e in repl:
                _emit(sp, e, str(s_), body, indent)
            for (name, _), e in zip(group, reduced):
                _emit(sp, e, name, body, indent)
            out_lines.extend(_fix_pycode(b) for b in body)
            if verbose:
                print(f"  {path.split('/')[-1]}: perm group of {len(group)} "
                      f"(src {sum(len(e) for _, e in group)/1e6:.2f}MB) -> {len(body)} stmts "
                      f"[sympify {t1-t0:.0f}s, cse {t2-t1:.0f}s]", flush=True)
        else:
            out_lines.append(lines[i])
            i += 1
    out = '\n'.join(out_lines)
    # the CSE'd statements use bare cos/sin/sqrt/pi (numpy) - make sure they resolve
    m_imp = re.search(r'^from numpy import ([\w, ]+)$', out, flags=re.M)
    if m_imp:
        have = {n.strip() for n in m_imp.group(1).split(',')}
        need = ['cos', 'sin', 'sqrt', 'pi']
        merged = 'from numpy import ' + ', '.join(sorted(have.union(need)))
        out = out.replace(m_imp.group(0), merged, 1)
    elif 'import numpy as np' in out:
        out = out.replace('import numpy as np',
                          'import numpy as np\nfrom numpy import cos, sin, sqrt, pi', 1)
    open(out_path, 'w').write(out)
    return out_path


if __name__ == '__main__':
    import os
    # usage: python cse_convert.py MOD1,MOD2 [out_dir] [src_dir]
    src_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src_dir, 'cse_out')
    os.makedirs(out_dir, exist_ok=True)
    for mod in sys.argv[1].split(','):
        t0 = time.time()
        print(f"converting {mod}...", flush=True)
        _convert_module(os.path.join(src_dir, f'{mod}.py'), os.path.join(out_dir, f'{mod}.py'))
        print(f"  {mod} done in {time.time()-t0:.0f}s", flush=True)
