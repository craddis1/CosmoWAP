"""One place that decides whether the jitted kernels are used.

Both accel modules (``lib.accel`` and ``numeric_mu.accel``) bind their kernels through this,
so the numba import, the ``COSMOWAP_DISABLE_NUMBA`` switch and the no-op decorator that keeps
those modules importable without numba are written once rather than once each.
"""

import os

try:
    from numba import njit

    _NUMBA_IMPORTED = True
except ImportError:  # the kernels are still defined below it - they are just never bound
    _NUMBA_IMPORTED = False

    def njit(**kwargs):
        """Stand-in for numba.njit that hands the function back untouched."""
        return lambda func: func


_OFF = {"", "0", "false", "no"}  # so COSMOWAP_DISABLE_NUMBA=0 leaves numba *on*


def have_numba():
    """Whether kernels should bind to their jitted versions rather than their fallbacks.

    A function rather than a constant so that reloading an accel module re-reads the
    environment, which is how the tests exercise the fallback path in-process.
    """
    return _NUMBA_IMPORTED and os.environ.get("COSMOWAP_DISABLE_NUMBA", "").strip().lower() in _OFF
