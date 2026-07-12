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
