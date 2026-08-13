import glob
import importlib
import os

# Get all .py files in the bk directory (excluding __init__.py)
module_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
module_names = [os.path.basename(f)[:-3] for f in module_files if f != "__init__.py"]

# Import each class and function dynamically from the modules
for module in module_names:
    mod = importlib.import_module(f'.{module}', package='cosmo_wap.bk')

    for attr in dir(mod):
        # Get the attribute - either class or function
        item = getattr(mod, attr)

        # Import if it's a class or function, skip if it's a private attribute
        if isinstance(item, (type, type(lambda: None))) and not attr.startswith("_"):
            # Add the class or function to the current namespace
            globals()[attr] = item

# optional C fast path: if compiled kernels exist (python -m cosmo_wap.bk.c_compile),
# patch them onto the expression classes; otherwise this is a no-op and numpy is used
from cosmo_wap.bk.c_compile import _load_c_kernels
_load_c_kernels(globals())

# the generated WS expression classes are parity-specific: the 1st order terms (WA1, RR1) define
# only the odd multipoles and the 2nd order ones (WA2, WARR, RR2) only the even ones, so a flat
# term list spanning both parities hits AttributeError in bk_func. Fill the absent multipoles with
# the zero-array stand-in, as the hand-written classes do via the decorator. Applied here rather
# than in the expression files because those are generated and hashed by c_compile - editing them
# would flag the compiled kernels as stale and silently fall back to numpy.
from cosmo_wap.lib.utils import add_empty_methods_bk

for _cls in (WA1, WA2, WARR, RR1, RR2):
    add_empty_methods_bk('l0', 'l1', 'l2', 'l3')(_cls)

# optional redshift-monomial tables (COSMOWAP_BK_TABLE=1), for MCMC fast/slow dragging.
# Applied last so the kernel patched above is what a table falls back to.
from cosmo_wap.bk.table import install as _install_tables
_install_tables(globals())
