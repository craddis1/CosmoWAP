import glob
import importlib
import os

# Get all .py files in the bk directory (excluding __init__.py)
module_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
module_names = [os.path.basename(f)[:-3] for f in module_files if f != "__init__.py"]

# Import each class and function dynamically from the modules
for module in module_names:
    mod = importlib.import_module(f'.{module}', package='cosmo_wap.bk_mt')

    for attr in dir(mod):
        # Get the attribute - either class or function
        item = getattr(mod, attr)

        # Import if it's a class or function, skip if it's a private attribute
        if isinstance(item, (type, type(lambda: None))) and not attr.startswith("_"):
            # Add the class or function to the current namespace
            globals()[attr] = item

# optional C fast path, as in cosmo_wap.bk. The prefix picks this package's entries out of
# the shared manifest - both packages define classes of the same name, so an unfiltered
# load would patch bk_mt's kernel onto bk's class.
from cosmo_wap.bk.c_compile import _load_c_kernels
_load_c_kernels(globals(), prefix='mt_')

# optional redshift-monomial tables (COSMOWAP_BK_TABLE=1), for MCMC fast/slow dragging.
# Applied last so the kernel patched above is what a table falls back to.
from cosmo_wap.bk_mt.table import install as _install_tables
_install_tables(globals())
