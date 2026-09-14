import copy as _copy
import functools
import os
import types

import numpy as np
from classy import Class
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PPoly

from cosmo_wap.lib import accel

trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


@functools.lru_cache(maxsize=None)
def leggauss(n):
    """Gauss-Legendre nodes and weights for order n, cached.

    They depend only on n, but numpy re-solves them on every call (0.1 ms at n=12) - and a
    sampler asks for the same two or three orders per redshift bin per likelihood call, which
    is ~8% of a likelihood there. Shared, so handed out read-only: as with array_memo below,
    callers must treat the result as immutable.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def cube(x):
    """x**3 without a libm pow per element.

    numpy special-cases ** only for exponents 2, 1, 0.5, 0 and -1, so spelling the k**-3
    power-law tails as (K_MAX/k)**3 costs a pow per element - 2.4x slower on the numeric-mu
    grids than the three multiplies."""
    return x * x * x


class CachedSpline(CubicSpline):
    """CubicSpline that remembers its value at a scalar argument.

    The cosmology and bias splines are asked for the same handful of redshifts over and over
    - one per bin, re-evaluated for every term, multipole and stencil point - and scipy spends
    2.3 us of its 3.7 us per call on input handling rather than on the evaluation. 74% of the
    spline calls in a Fisher iteration are at a scalar z and 98.7% of those repeat, so
    remembering them is worth ~13% of the iteration.

    A cached value can never go stale: no spline is mutated in place anywhere: a changed bias
    builds a new object (interpolate_beta_funcs reassigns) or a modify_func wrapper, which is
    a plain closure and so is not cached at all. Subclassing rather than wrapping means
    derivative() keeps the caching, since PPoly builds its result through construct_fast(cls).
    """

    _MAX = 512  # a forecast asks for ~30 distinct redshifts; a scalar sweep must not grow forever
    # every scalar redshift the forecast passes is one of these; np.ndim would answer for a
    # 0-d array too, but it reaches that answer through asarray, which costs more than the
    # cache lookup it guards. A 0-d array simply takes the uncached path.
    _SCALARS = (float, int, np.floating, np.integer)

    def __call__(self, x, nu=0, extrapolate=None):
        if not isinstance(x, self._SCALARS):  # an array is used once - nothing to remember
            return super().__call__(x, nu, extrapolate)
        cache = self.__dict__.get("_at")  # construct_fast skips __init__, so build it lazily
        if cache is None:
            cache = self.__dict__["_at"] = {}
        # float() keeps the key type-stable; a spline value is never None, so a miss is
        # unambiguous without a second lookup
        key = (float(x), nu, extrapolate)
        val = cache.get(key)
        if val is None:
            if len(cache) >= self._MAX:
                cache.clear()
            val = cache[key] = super().__call__(x, nu, extrapolate)
        return val

    def __getstate__(self):
        # PPoly holds its coefficients in __slots__, so the default state is (instance dict,
        # slots dict) - the memo is in the first and rebuilds on demand, so it need not travel
        state, slots = super().__getstate__()
        return ({k: v for k, v in state.items() if k != "_at"} if state else state), slots


def cached(spl):
    """Re-type an already-built spline so it memoises; anything else passes through.

    For splines that arrive built - accel.spline_stack's, and the views SplineStack cuts out
    of it - where swapping the constructor is not an option."""
    if isinstance(spl, PPoly) and not isinstance(spl, CachedSpline):
        return CachedSpline.construct_fast(spl.c, spl.x, spl.extrapolate, spl.axis)
    return spl


class SplineStack:
    """Curves (n_curve,len(zz)) splined together over a shared z grid - one solve for all of them.

    Calling it evaluates every curve in one go, indexing or iterating gives the individual
    splines as views on that solve. Identical to building each CubicSpline separately, but
    without paying the scipy per-spline overhead n times - which dominates when these are
    rebuilt and evaluated per redshift bin"""

    def __init__(self, zz, arr):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"SplineStack takes a (n_curve, len(zz)) array, got shape {arr.shape}")
        # jitted coefficient build where numba is available, else scipy - see lib.accel
        self.spl = cached(accel.spline_stack(zz, arr))

    def __call__(self, zz):
        return self.spl(zz).T  # shape (n_curve,)+shape(zz)

    def __len__(self):
        return self.spl.c.shape[-1]

    def __getitem__(self, i):  # out of range raises IndexError, so iteration/unpacking works
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return CachedSpline.construct_fast(self.spl.c[..., i].copy(), self.spl.x, extrapolate=self.spl.extrapolate)


# __all__ = ['get_cosmo', 'get_b_params','Emulator']


def get_cosmo(
    h=0.6766,
    Omega_m=0.30964144,
    Omega_b=0.04897,
    A_s=2.105e-9,
    n_s=0.9665,
    Omega_cdm=None,
    k_max=10,
    z_max=6,
    sigma8=None,
    w0=None,
    wa=None,
    method_nl="halofit",
    emulator=False,
    ln_A_s=None,
):
    """Calls class for some set of parameters and returns the cosmology - base cosmology is planck 2018
    Omega_i is defined without h**2 dependence

    Default is Planck 2018 cosmology.

    So we work normally with Omega_m and Omega_b but allow for Omega_cdm and Omega_b - easiest for MCMC samples."""

    if Omega_cdm is not None:
        Omega_m = Omega_b + Omega_cdm  # so we always use Omega_b

    if ln_A_s is not None:  # sample/differentiate in the well-conditioned ln(10^10 A_s); convert to A_s here
        A_s = 1e-10 * np.exp(ln_A_s)

    # Create a params dictionary
    params = {"Omega_b": Omega_b, "Omega_m": Omega_m, "h": h, "n_s": n_s}
    if sigma8 is None:  # if sigma8 define with sigma8 not A_s
        params["A_s"] = A_s
    else:
        params["sigma8"] = sigma8

    if not emulator:  # then we use class powerspectrum!
        params["output"] = "mPk"
        if method_nl:  # could be HMcode or halofit; None skips the non-linear computation
            params["non linear"] = method_nl
        params["P_k_max_1/Mpc"] = k_max
        params["z_max_pk"] = z_max

        if w0 is not None or wa is not None:
            # for w0 wa - default the unspecified one to its fiducial
            params["Omega_Lambda"] = 0
            params["w0_fld"] = w0 if w0 is not None else -1.0
            params["wa_fld"] = wa if wa is not None else 0.0

    # Initialize the cosmology and compute everything
    cosmo = Class()
    cosmo.set(params)
    if not emulator:
        cosmo.compute()
        return cosmo
    return cosmo, params  # - A_s is tricky to get out of cosmo so this is needed for speedup with emulator


# get_cosmo parameters a forecast can step or a sampler can sample - one list so the two agree:
# one the fisher steps but update_cosmo_funcs drops gives the chain a flat likelihood in it.
COSMO_PARAMS = ["Omega_m", "Omega_cdm", "Omega_b", "A_s", "ln_A_s", "sigma8", "n_s", "h", "w0", "wa"]


def fiducial_cosmo_kwargs(cf):
    """get_cosmo kwargs reproducing cf's cosmology - the base a stepped/sampled value overrides.

    Without it get_cosmo fills what it is not handed from its own (Planck 2018) defaults, so a step
    off any other fiducial lands somewhere else entirely. Only the parameterisation cf was built in
    is returned: Omega_cdm, sigma8 and ln_A_s would override Omega_m/A_s, and w0/wa switch CLASS to
    a fluid dark energy, so those are left to the caller.
    """
    kwargs = {p: getattr(cf, p) for p in ("h", "Omega_m", "Omega_b", "A_s", "n_s")}
    w0, wa = getattr(cf, "w0", -1.0), getattr(cf, "wa", 0.0)
    if (w0, wa) != (-1.0, 0.0):
        kwargs["w0"], kwargs["wa"] = w0, wa
    return kwargs


def get_b_params(cosmo):
    """Get params for bacco from cosmo"""
    params = {
        "omega_cold": cosmo.Omega_m(),
        "sigma8_cold": cosmo.sigma8(),  # if A_s is not specified
        "omega_baryon": cosmo.Omega_b(),
        "ns": cosmo.n_s(),
        "hubble": cosmo.h(),
        "neutrino_mass": 0.0,
        "w0": -1.0,
        "wa": 0.0,
        "expfactor": 1,  # a - set z=0 for now but can call in vectorised format for nonlinear pk later
    }
    return params


def t_lookback(cosmo, zz, n=1024):
    """Lookback time [Gyr] to redshift zz - t_L(z) = int_0^z dz'/((1+z')H(z'))

    Integrated on a grid uniform in ln(1+z), so a single call spans both survey redshifts
    and the z ~ 1100 upper limit used to normalise the WISE luminosity function's age
    (see lib.luminosity_funcs.WISELuminosityFunction)."""
    zz = np.asarray(zz, dtype=float)

    u = np.linspace(0, np.log1p(zz.max()), n)  # u = ln(1+z), so dz/(1+z) = du
    t = cumulative_trapezoid(1 / cosmo.Hubble(np.expm1(u)), u, initial=0)  # [Mpc], H is 1/Mpc

    return np.interp(np.log1p(zz), u, t) * 3.2616e-3  # Mpc/c in Gyr


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
        self.Pk = cp.cosmopower_NN(restore=True, restore_filename=os.path.join(module_dir, "../data_library/PKLIN_NN"))
        self.Pk_NL = cp.cosmopower_NN(
            restore=True, restore_filename=os.path.join(module_dir, "../data_library/PKNLBOOST_NN")
        )
        self.k = np.loadtxt(os.path.join(module_dir, "../data_library/k_modes.txt"))


###################################################


# useful for defining the triangle (just cosine rule)
def get_theta(k1, k2, k3):
    """
    get theta for given triangle - being careful with rounding
    """
    cos_theta = (k3**2 - k1**2 - k2**2) / (2 * k1 * k2)
    cos_theta = np.where(np.isclose(np.abs(cos_theta), 1), np.sign(cos_theta), cos_theta)
    return np.arccos(cos_theta)


def get_k3(theta, k1, k2):
    """
    get k3 for given triangle
    """
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))
    return np.where(k3 == 0, 1e-4, k3)


def get_theta_k3(k1, k2, k3, theta):
    if theta is None:
        if k3 is None:
            raise ValueError("Define either theta or k3")
        else:
            theta = get_theta(k1, k2, k3)  # from utils
    else:
        if k3 is None:
            k3 = get_k3(theta, k1, k2)
    return k3, theta


def enable_broadcasting(*args, n=2):
    """Make last n axes size 1 if arrays, to allow numpy broadcasting
    Careful with only one arg as always returns a tuple!"""
    result = []

    for var in args:
        if isinstance(var, np.ndarray):
            # Create a tuple of n trailing None dimensions
            new_axes = (None,) * n
            result.append(var[(...,) + new_axes])
        else:
            result.append(var)

    return tuple(result)


def get_faint_bias(zz, n_T, n_B, b_T, b_B):
    """Get faint bias from total and bright - uses number density weighting"""
    return CachedSpline(zz, (n_T * b_T - n_B * b_B) / (n_T - n_B))


#################################################################### Misc

# RULE: on a copy, only the objects `copy()` freshens below may be *mutated in place* -
# a tracer in `survey`, one of its `_TRACER_HOLDERS`, and its `deriv` dict. Everything
# else (the large immutable cosmology splines Pk, D, f, H_c, ...; survey_params; cosmo;
# emu; and anything deeper inside a tracer - hod, lf, eulbias, the bias splines) is
# shared by reference, so on a copy it must ONLY be *reassigned* (`cf.attr = new`, which
# rebinds the copy's own __dict__ slot), never mutated in place - otherwise the change
# leaks back into the original and every other copy. Widen the freshened set here if you
# add an edit that needs it (see tests/test_utils.py::TestCopy).
#
# The bias-derivative code obeys this: every edit it makes is a setattr on a tracer
# (`modify_func(..., do_copy=False)`, `shift_linked_bias`) or on that tracer's loc/eq/orth
# holder, so two levels of freshening is all it needs. Sharing the rest by reference makes
# a multi-tracer ClassWAP copy ~50x cheaper than deep-copying `survey` (166 us -> 3 us),
# which matters because the five-point stencil makes one per parameter, per bin, per point.
#
# Two consequences of sharing by reference:
# - A top-level attribute that also lives inside a tracer stays shared (desired for e.g.
#   `self.n_g = self.survey[0].n_g`, which aliases the spline), but it is *not* updated by
#   an edit to the tracer - the alias still points at the unshifted function.
# - On objects with no `survey` attribute (tracers, luminosity functions, PNG bias
#   holders), `copy()` shares *everything*: the copy is only safe to modify by
#   reassigning attributes.
_TRACER_HOLDERS = ("loc", "eq", "orth")  # nested PNG bias holders the derivative code setattrs on

# Nothing here deep-copies any more, but a scipy CubicSpline carries a module (`_xp`) in
# its reduce state that can't be deep-copied (`cannot pickle 'module' object`) - notably
# on Python 3.14, whose deepcopy atomic-type rework changed how such C-extension objects
# reduce. Kept as a process-wide safety net for any caller that still deep-copies an
# object holding one: a module is a singleton, so copying it is never meaningful and
# always raises. `setdefault` defers to any handler another library installed first.
_copy._deepcopy_dispatch.setdefault(types.ModuleType, lambda x, memo: x)


def _copy_tracer(tracer):
    """A tracer that can be edited without the edit reaching the original.

    Shallow, plus the two things the derivative code writes through: the loc/eq/orth
    holders (`modify_func(cf.survey[t].loc, 'b_01', ...)`) and the `deriv` cache, which
    unpack.py fills in place (`tracer.deriv['beta'] = ...`) with values built from the
    tracer's biases - shared, a shifted copy would poison the fiducial tracer's cache.
    """
    new = tracer.__class__.__new__(tracer.__class__)
    attrs = dict(tracer.__dict__)
    for name in _TRACER_HOLDERS:
        holder = attrs.get(name)
        if holder is not None:
            new_holder = holder.__class__.__new__(holder.__class__)
            new_holder.__dict__ = dict(holder.__dict__)
            attrs[name] = new_holder
    if "deriv" in attrs:
        attrs["deriv"] = dict(attrs["deriv"])
    new.__dict__ = attrs
    return new


def copy(self):
    """Fast, independent copy for the forecast/derivative machinery.

    Freshens the tracers in `survey` (see `_copy_tracer`) and shares everything else by
    reference, under the RULE above. On an object with no `survey` this is a plain shallow
    copy - which is what the deep-copying version it replaced also did there.
    """
    survey = self.__dict__.get("survey")
    new_self = self.__class__.__new__(self.__class__)
    new_self.__dict__ = attrs = dict(self.__dict__)
    if survey is not None:
        # a tracer repeated in the list (survey is e.g. [X, Y, X]) must stay one object:
        # callers edit `set(cf.survey)` and expect the edit to reach every slot
        copies = {}
        new_survey = []
        for tracer in survey:
            if tracer is None:
                new_survey.append(None)
                continue
            if id(tracer) not in copies:
                copies[id(tracer)] = _copy_tracer(tracer)
            new_survey.append(copies[id(tracer)])
        attrs["survey"] = new_survey
    return new_self


def array_memo(limit=4):
    """Memoise a method on its (positional) array arguments, keyed by their values.

    For pure functions of a few large arrays that get asked the same question repeatedly -
    HMF.n_h and YP.HOD are each called ~159 times per bias build over a couple of distinct
    inputs. Arrays are unhashable, so the key is their bytes; by value rather than by id()
    so an equal but rebuilt array still hits and a freed one cannot be recycled into a
    wrong answer. The cache is per instance - a new cosmology builds new objects - and is
    dropped wholesale past `limit`, which suits a handful of arguments, not a long tail.

    Only safe where the result is treated as read-only, since callers share one array.
    """

    def decorate(func):
        attr = f"_{func.__name__}_memo"

        @functools.wraps(func)
        def wrapper(self, *args):
            cache = self.__dict__.setdefault(attr, {})
            key = tuple((a.shape, a.dtype.str, a.tobytes()) for a in map(np.asarray, args))
            hit = cache.get(key)
            if hit is None:
                if len(cache) >= limit:
                    cache.clear()
                hit = cache[key] = func(self, *args)
            return hit

        return wrapper

    return decorate


def modify_func(parent, func_name, modifier, do_copy=True):
    """Apply a modifier function to an existing function-
    Useful when computing derivatives of stuff with respect to a change in a function"""
    if do_copy:
        new_parent = copy(parent)
    else:
        new_parent = parent

    current_func = getattr(new_parent, func_name)

    # Preserve original function signature
    def wrapped_func(*args, **kwargs):
        return modifier(current_func(*args, **kwargs))

    setattr(new_parent, func_name, wrapped_func)

    # resets cache of compute betas and derivs
    if hasattr(new_parent, "reset_cache"):
        new_parent.reset_cache()

    return new_parent


# Linked-bias params - nuisances with no survey attribute of their own, mapped here to the
# tracer attributes they shift. So 'b_phi' is just the local PNG bias survey.loc.b_01, while
# 'b_phi_e' moves that and the evolution bias b_e together (b_e with an f(z)/2 weight, below).
LINKED_BIAS_TARGETS = {
    "b_phi": (("loc", "b_01"),),
    "b_phi_e": (("loc", "b_01"), (None, "be")),
}


def linked_bias_fid(tracer, param):
    """The function a linked-bias param is normalised on - b_phi for every entry."""
    return tracer.loc.b_01


def shift_linked_bias(tracer, param, delta, f=None):
    """Additively shift the survey functions behind a linked-bias param, in place.

    delta is a callable of redshift, or a scalar for a flat offset. b_phi takes the full shift;
    b_e takes f(z)/2 of it, so raising 'b_phi_e' by 1 raises b_phi by 1 and b_e by f/2 - f is the
    growth rate cosmo_funcs.f, required whenever param moves b_e. Returns
    [(obj, attr, original_func),...] so the sampler can undo it: it edits the cached (shared)
    survey objects rather than a copy.
    """
    shift = delta if callable(delta) else (lambda *args, **kwargs: delta)

    restore = []
    for holder, attr in LINKED_BIAS_TARGETS[param]:
        obj = tracer if holder is None else getattr(tracer, holder)
        func = getattr(obj, attr)
        restore.append((obj, attr, func))
        weight = (lambda zz: f(zz) / 2) if attr == "be" else (lambda zz: 1.0)
        setattr(
            obj,
            attr,
            lambda zz, *args, _f=func, _w=weight, **kwargs: _f(zz, *args, **kwargs)
            + _w(zz) * shift(zz, *args, **kwargs),
        )

    # b_e feeds the cached betas/derivs - the nested PNG holder has no cache of its own
    if hasattr(tracer, "reset_cache"):
        tracer.reset_cache()
    return restore


def add_empty_methods_pk(*method_names):
    """
    A class decorator factory that adds empty static methods to a class.
    Basically just defines multipoles for terms which are zero (so we dont have errors in forecast)
    Each new method will return an empty list [] - powerspectrum.
    """

    def decorator(cls):
        # This returns a zero array of correct size
        def empty_array_func(cosmo_funcs, k1, zz=0, *args, **kwargs):
            return np.zeros_like(k1)

        # Loop through the desired method names
        for name in method_names:
            # Check if the method already exists to avoid overwriting
            if not hasattr(cls, name):
                # Add the function as a staticmethod to the class
                setattr(cls, name, staticmethod(empty_array_func))
        return cls

    return decorator


def add_empty_methods_bk(*method_names):
    """
    A class decorator factory that adds empty static methods to a class
    Basically just defines multipoles for terms which are zero (so we dont have errors in forecast)
    Each new method will return an empty list [] - bispectrum
    """

    def decorator(cls):
        # This returns a zero array of correct size
        def empty_array_func(cosmo_funcs, k1, k2, k3=None, theta=None, zz=0, *args, **kwargs):
            third = k3 if k3 is not None else theta  # the triangle is closed by either one
            return np.zeros(np.broadcast_shapes(np.shape(k1), np.shape(k2), np.shape(third)))

        # Loop through the desired method names
        for name in method_names:
            # Check if the method already exists to avoid overwriting
            if not hasattr(cls, name):
                # Add the function as a staticmethod to the class
                setattr(cls, name, staticmethod(empty_array_func))
        return cls

    return decorator


def solve_preconditioned(F, precondition=True):
    """Return F^{-1}, optionally with diagonal preconditioning for ill-conditioned matrices.

    Scales by 1/sqrt(diag(F)) before inversion and unscales after — identical to
    np.linalg.inv(F) but the intermediate matrix has unit diagonal, greatly reducing
    the condition number when constraints span many orders of magnitude.
    """

    if not precondition:
        return np.linalg.inv(F)
    s = np.sqrt(np.abs(np.diag(F)))
    s[s == 0] = 1.0
    F_scaled = F / np.outer(s, s)
    return np.linalg.inv(F_scaled) / np.outer(s, s)


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
    import cProfile
    import io
    import pstats
    from pstats import SortKey

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
