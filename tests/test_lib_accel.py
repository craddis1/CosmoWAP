"""The jitted kernels in lib.accel must reproduce scipy's CubicSpline and np.gradient.

Every beta coefficient and every lnd_derivatives set in a sampler goes through both (via
lib.utils.SplineStack and ClassWAP.lnd_derivatives), so the whole bias model depends on them
agreeing - and the fallbacks (no numba, short grid, non-float64) have to stay wired to
scipy/numpy rather than approximating them.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from cosmo_wap.lib import accel, utils

numba = pytest.importorskip("numba", reason="numba is optional - accel falls back to scipy")


def _grids():
    rng = np.random.default_rng(0)
    yield "uniform", np.linspace(0.9, 1.8, 100)
    yield "survey-like", np.linspace(0.0, 4.0, 61)
    x = np.sort(rng.uniform(0.9, 1.8, 100))  # non-uniform: the not-a-knot rows depend on dx
    x[0], x[-1] = 0.9, 1.8
    yield "non-uniform", x
    yield "minimum length", np.array([0.1, 0.4, 0.42, 1.7])  # n=4, the smallest not-a-knot fit


@pytest.mark.parametrize("name,x", list(_grids()), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("n_curve", [1, 3, 7, 16])
def test_coeffs_match_scipy(name, x, n_curve):
    rng = np.random.default_rng(abs(hash((name, n_curve))) % 2**32)
    arr = rng.standard_normal((n_curve, x.size))

    ref = CubicSpline(x, arr.T, axis=0)
    got = accel.spline_stack(x, arr)

    assert got.c.shape == ref.c.shape
    assert np.max(np.abs(got.c - ref.c)) / np.max(np.abs(ref.c)) < 1e-11

    # what actually matters downstream: the values, inside the grid and extrapolated past it
    ze = np.linspace(x[0] - 0.3, x[-1] + 0.3, 257)
    assert np.max(np.abs(got(ze) - ref(ze))) / np.max(np.abs(ref(ze))) < 1e-12


def test_smooth_data_matches_to_round_off():
    """Random data exercises the solver; the real inputs are smooth, where it agrees far better."""
    x = np.linspace(0.9, 1.8, 100)
    arr = np.stack([np.sin(3 * x), np.exp(-x), 1 / (1 + x) ** 2, np.ones_like(x)])

    ref = CubicSpline(x, arr.T, axis=0)
    got = accel.spline_stack(x, arr)
    ze = np.linspace(0.9, 1.8, 401)
    assert np.max(np.abs(got(ze) - ref(ze))) < 1e-14


def test_splinestack_uses_the_kernel_and_still_indexes():
    """SplineStack's own API (call, len, getitem, unpacking) over the jitted build."""
    x = np.linspace(0.9, 1.8, 100)
    arr = np.stack([np.sin(3 * x), np.cos(x), x**2])

    stack = utils.SplineStack(x, arr)
    ref = CubicSpline(x, arr.T, axis=0)

    assert len(stack) == 3
    ze = np.linspace(0.95, 1.75, 37)
    assert np.allclose(stack(ze), ref(ze).T, rtol=0, atol=1e-13)
    for i, spl in enumerate(stack):  # __getitem__ views onto the one solve
        assert np.allclose(spl(ze), ref(ze).T[i], rtol=0, atol=1e-13)
    a, b, c = stack
    assert np.allclose(c(ze), ref(ze).T[2], rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "x,arr",
    [
        (np.array([0.1, 0.5, 1.0]), np.ones((2, 3))),  # n < 4: scipy does not solve not-a-knot
        (np.linspace(0.9, 1.8, 10), np.ones((2, 10), dtype=np.complex128)),  # non-float64
        (np.linspace(0.9, 1.8, 10, dtype=np.float32), np.ones((2, 10), dtype=np.float32)),
    ],
)
def test_unsupported_inputs_fall_back_to_scipy(x, arr):
    assert not accel._jittable(np.asarray(x), np.asarray(arr), min_points=4)
    ref = CubicSpline(x, np.asarray(arr).T, axis=0)
    got = accel.spline_stack(x, arr)
    assert np.allclose(got.c, ref.c, rtol=0, atol=0)


def test_non_increasing_x_raises():
    """scipy rejects unsorted x; silently fitting it would be worse than the fallback."""
    x = np.array([0.1, 0.4, 0.3, 0.9, 1.2])
    with pytest.raises(ValueError):
        accel.spline_stack(x, np.ones((2, 5)))


def test_disable_env_var_is_honoured():
    """COSMOWAP_DISABLE_NUMBA has to reach the guard, not just the import."""
    import importlib

    import cosmo_wap.lib.accel as mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("COSMOWAP_DISABLE_NUMBA", "1")
        reloaded = importlib.reload(mod)
        try:
            assert not reloaded.HAVE_NUMBA
            x = np.linspace(0.9, 1.8, 100)
            arr = np.stack([np.sin(3 * x), np.cos(x)])
            assert not reloaded._jittable(x, arr, min_points=4)
            assert isinstance(reloaded.spline_stack(x, arr), CubicSpline)
        finally:
            importlib.reload(mod)  # leave the module as the rest of the suite found it


# --- gradient ------------------------------------------------------------------------------


@pytest.mark.parametrize("name,x", list(_grids()), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("n_curve", [1, 2, 3, 7])
def test_gradient_is_bit_identical_to_numpy(name, x, n_curve):
    """Not merely close: numpy's own operation order is reproduced, so this is exact."""
    rng = np.random.default_rng(abs(hash((name, n_curve))) % 2**32)
    y = rng.standard_normal((n_curve, x.size))

    got = accel.gradient(y, x)
    ref = np.gradient(y, x, axis=-1)

    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_gradient_on_a_two_point_grid():
    """n=2 is all edge and no interior - numpy allows it, so the kernel has to as well."""
    x = np.array([0.5, 1.25])
    y = np.array([[1.0, 3.0], [-2.0, 0.5]])
    assert np.array_equal(accel.gradient(y, x), np.gradient(y, x, axis=-1))


@pytest.mark.parametrize(
    "y,x",
    [
        (np.ones((2, 10), dtype=np.complex128), np.linspace(0.9, 1.8, 10)),  # non-float64
        (np.ones(10), np.linspace(0.9, 1.8, 10)),  # 1D, not a curve stack
    ],
)
def test_gradient_unsupported_inputs_fall_back_to_numpy(y, x):
    assert np.array_equal(accel.gradient(y, x), np.gradient(y, x, axis=-1))


def test_lnd_derivatives_matches_numpy(cosmo_funcs):
    """End to end through the caller that matters."""
    tracer = cosmo_funcs.survey[0]
    zz = tracer.z_survey
    funcs = [tracer.b_1, tracer.b_2, tracer.g_2]

    values = np.array([f(zz) for f in funcs])
    ref = np.gradient(values, cosmo_funcs.lnd_survey[0], axis=-1)
    assert np.array_equal(accel.gradient(values, cosmo_funcs.lnd_survey[0]), ref)

    got = cosmo_funcs.lnd_derivatives(funcs)
    ze = np.linspace(zz[0], zz[-1], 41)
    assert np.allclose(got(ze), utils.SplineStack(zz, ref)(ze), rtol=0, atol=1e-13)


def test_env_var_is_read_as_a_flag():
    """A bare truthiness test on the variable would let COSMOWAP_DISABLE_NUMBA=0 switch it off."""
    from cosmo_wap.lib import jit

    with pytest.MonkeyPatch.context() as mp:
        for off in ("1", "true", "yes", "TRUE"):
            mp.setenv("COSMOWAP_DISABLE_NUMBA", off)
            assert not jit.have_numba(), off
        for on in ("", "0", "false", "no", " 0 "):
            mp.setenv("COSMOWAP_DISABLE_NUMBA", on)
            assert jit.have_numba(), repr(on)
        mp.delenv("COSMOWAP_DISABLE_NUMBA")
        assert jit.have_numba()
