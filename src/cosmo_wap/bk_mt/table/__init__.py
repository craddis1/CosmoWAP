"""Redshift-monomial coefficient tables for the multi-tracer bispectrum.

Same decomposition, gating and caching as cosmo_wap.bk.table - only the generated
modules differ - so everything is reused from there and this package holds the
<class>_tab.py files plus the module list.

Regenerate with:
    python -m cosmo_wap.bk.table.convert --pkg bk_mt GR0,GR1,GR2,PNG

The multi-tracer expressions carry three copies of the bias and beta scalars
(b1/xb1/yb1, beta6..19/xbeta6..19/ybeta6..19, ...), which roughly triples the monomial
count against the single-tracer classes - NPP.l0 goes from 20 to 54 - but they are small
expressions to begin with, so the largest here (GR2.l0) is 294 monomials against RR2's
1,254 and the whole set converts in seconds.
"""

# bk_mt only implements the Newtonian and relativistic terms plus local PNG; everything
# in combined.py is a zero stub, so there is nothing else to tabulate. PNG's Eq/Orth are
# absent for the same reason as in bk: their cube-root shape functions keep D1 in the
# coefficients, which convert.py refuses to emit.
TABLE_MODULES = ('NPP', 'GR1', 'GR2', 'Loc')


def install(namespace, modules=TABLE_MODULES):
    """Patch table dispatch onto the bk_mt classes in `namespace`."""
    from cosmo_wap.bk.table import install as _install

    return _install(namespace, modules=modules, pkg='bk_mt')
