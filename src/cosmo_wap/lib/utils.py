import numpy as np
from classy import Class
import os
import copy

def get_cosmo(h = 0.6766,Omega_m = 0.30964144,Omega_b = 0.04897,A_s = 2.105e-9,n_s = 0.9665,Omega_cdm=None,k_max=10,z_max = 6,sigma8=None,method_nl='halofit',emulator=False):
    """ calls class for some set of parameters and returns the cosmology - base cosmology is planck 2018
    Omega_i is defined without h**2 dependence
    
    
    So we work normally with Omega_m and Omega_b but aloow for Omega_cdm and Omega_b - easiest for MCMC samples."""
    
    if Omega_cdm:
        Omega_m = Omega_b+Omega_cdm # so we always use Omega_b

    #Create a params dictionary
    params = {'Omega_b':Omega_b,
                 'Omega_m': Omega_m,
                 'h':h,
                 'n_s':n_s
    }
    if sigma8 is None:# if sigma8 define with sigma8 not A_s
        params['A_s'] = A_s
    else:
        params['sigma8'] = sigma8

    if not emulator: # then we use class powerspectrum!
        params['output'] = 'mPk'
        params['non linear'] = method_nl
        params['P_k_max_1/Mpc'] = k_max
        params['z_max_pk'] = z_max
        
    #Initialize the cosmology and compute everything
    cosmo = Class()
    cosmo.set(params)
    if not emulator:
        cosmo.compute()
        return cosmo
    return cosmo, params # - A_s is tricky to get out of cosmo so this is needed for speedup with emulator

def get_b_params(cosmo):
    """Get params for bacco from cosmo"""
    params = {
        'omega_cold'    :  cosmo.Omega_m(),
        'sigma8_cold'   :  cosmo.sigma8(), # if A_s is not specified
        'omega_baryon'  :  cosmo.Omega_b(),
        'ns'            :  cosmo.n_s(),
        'hubble'        :  cosmo.h(),
        'neutrino_mass' :  0.0,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  1   # a - set z=0 for now but can call in vectorised format for nonlinear pk later
    }
    return params

class Emulator:
    """
    A nested class to encapsulate all Cosmopower emulator functionality.
    It loads the pre-trained neural network models for P(k).
    """
    def __init__(self):
        import cosmopower as cp
        
        # Define the path to the data file relative to the script location
        # Note: __file__ refers to the location of the file this code is in.
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for interactive environments like Jupyter notebooks
            module_dir = os.getcwd()

        # Load pre-trained NN models and k-modes
        self.Pk = cp.cosmopower_NN(restore=True, restore_filename=os.path.join(module_dir,'../data_library/PKLIN_NN'))
        self.Pk_NL = cp.cosmopower_NN(restore=True, restore_filename=os.path.join(module_dir,'../data_library/PKNLBOOST_NN'))
        self.k = np.loadtxt(os.path.join(module_dir,'../data_library/k_modes.txt'))

###################################################

# useful for defining the triangle (just cosine rule)
def get_theta(k1,k2,k3):
    """
    get theta for given triangle - being careful with rounding
    """
    cos_theta = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    cos_theta = np.where(np.isclose(np.abs(cos_theta), 1), np.sign(cos_theta), cos_theta)
    return np.arccos(cos_theta)

def get_k3(theta,k1,k2):
    """
    get k3 for given triangle
    """
    k3 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(theta))
    return np.where(k3==0,1e-4,k3)

def get_theta_k3(k1,k2,k3,theta):
    if theta is None:
        if k3 is None:
            raise  ValueError('Define either theta or k3')
        else:
            theta = get_theta(k1,k2,k3) #from utils
    else:
        if k3 is None:
            k3 = get_k3(theta,k1,k2)
    return k3, theta

def enable_broadcasting(*args,n=2):
    """Make last n axes size 1 if arrays, to allow numpy broadcasting"""
    result = []
    
    for var in args:
        if isinstance(var, np.ndarray):
            # Create a tuple of n trailing None dimensions
            new_axes = (None,) * n
            result.append(var[(...,) + new_axes])
        else:
            result.append(var)
           
    return tuple(result)

#################################################################### Misc
def create_copy(self):
    """
    Create a deep copy of the object, preserving the cosmo and emu references
    by instructing `deepcopy` on how to handle them.
    """
    # 1. Create the 'memo' dictionary for deepcopy.
    memo = {}

    # 2. Pre-populate the memo with objects that should not be copied.
    # We map the object's ID to the object's own reference.
    # This tells deepcopy: "When you see this object, its 'copy' is just itself."
    for key in ['cosmo', 'emu']:
        if hasattr(self, key):
            # Get the actual object reference (e.g., the cosmo instance)
            obj_ref = getattr(self, key)
            # Add its id and reference to the memo
            memo[id(obj_ref)] = obj_ref

    # 3. Create the new, empty object instance
    new_self = self.__class__.__new__(self.__class__)

    # 4. Now, perform the deepcopy using our custom memo.
    # When deepcopy encounters `cosmo` or `emu` (at any level of nesting),
    # it will find their ID in the memo and use the provided reference
    # instead of attempting to copy them, thus avoiding the error.
    new_self.__dict__ = copy.deepcopy(self.__dict__, memo)

    return new_self

def modify_func(parent, func_name, modifier):
    """Apply a modifier function to an existing function-
    Useful when computing derivatives of stuff with respect to a change in a function"""
    new_parent = create_copy(parent)
    current_func = getattr(new_parent, func_name)
    
    # Preserve original function signature
    def wrapped_func(*args, **kwargs):
        return modifier(current_func(*args, **kwargs))
    
    setattr(new_parent, func_name, wrapped_func)
    return new_parent

def add_empty_methods_pk(*method_names):
    """
    A class decorator factory that adds empty static methods to a class.
    Basically just defines multipoles for terms which are zero (so we dont have errors in forecast)
    Just implemented for pk for now!
    Each new method will return an empty list [].
    """
    def decorator(cls):
        # This returns a zero array of correct size
        def empty_array_func(cosmo_funcs,k1,zz=0,*args, **kwargs):
            return np.zeros(*(k1*zz).shape)

        # Loop through the desired method names
        for name in method_names:
            # Check if the method already exists to avoid overwriting
            if not hasattr(cls, name):
                # Add the function as a staticmethod to the class
                setattr(cls, name, staticmethod(empty_array_func))
        return cls
    return decorator

import cProfile
import pstats
from pstats import SortKey
#import sys
import io

def profile_code(code_to_run, global_vars, local_vars, num_results=20, sort_by_time=False):
    """
    Profiles the execution of the provided code using the specified global and local scope.
    
    So example usage:
    utils.profile_code("... code ...", globals(), locals())

    Args:
        code_to_run (str): The code to be executed and profiled.
        global_vars (dict): The global namespace from the calling environment (use globals()).
        local_vars (dict): The local namespace from the calling environment (use locals()).
        num_results (int): The number of top results to print. Defaults to 20.
        sort_by_time (bool): If True, also prints results sorted by total time.
                             Defaults to False.
    """
    profiler = cProfile.Profile()
    output_stream = io.StringIO()
    
    try:
        profiler.enable()
        # Execute the code with the provided scope
        exec(code_to_run, global_vars, local_vars)
        profiler.disable()

        # Create stats object writing to our stream
        stats = pstats.Stats(profiler, stream=output_stream).sort_stats(SortKey.CUMULATIVE)
        
        # Print the stats sorted by cumulative time
        output_stream.write("--- Profiling results sorted by cumulative time ---\n")
        stats.print_stats(num_results)

        if sort_by_time:
            # Optionally, print stats sorted by total time
            output_stream.write("\n--- Profiling results sorted by total time ---\n")
            stats.sort_stats(SortKey.TIME).print_stats(num_results)

    finally:
        profiler.disable()
        # Print the collected output
        print(output_stream.getvalue())

