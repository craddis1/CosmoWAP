import glob
import importlib
import os

# Get all .py files in the pk directory (excluding __init__.py)
module_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py")))  # glob order is arbitrary
module_names = [os.path.basename(f)[:-3] for f in module_files
                if os.path.basename(f) != "__init__.py"]

# Import each class and function dynamically from the modules
_namespace = {}
for module in module_names:
    mod = importlib.import_module(f'.{module}', package='cosmo_wap.pk')

    for attr in dir(mod):
        # Get the attribute - either class or function
        item = getattr(mod, attr)

        # Import if it's a class or function, skip if it's a private attribute
        if isinstance(item, (type, type(lambda: None))) and not attr.startswith("_"):
            # Collect the class or function - applied to the namespace after the loop
            _namespace[attr] = item

# applied after the loop so classes win over same-named modules, as in cosmo_wap.bk
globals().update(_namespace)
