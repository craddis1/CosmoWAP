"""Tests for cosmo_wap.lib.utils — geometry, broadcasting, decorators, get_cosmo."""

import pickle

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import cosmo_wap as cw
from cosmo_wap.lib import utils

# ── Triangle geometry ─────────────────────────────────────────────────────────


class TestGetTheta:
    def test_equilateral(self):
        """Equilateral triangle → θ = 2π/3 (angle opposite k3 via cosine rule)."""
        theta = utils.get_theta(0.1, 0.1, 0.1)
        assert theta == pytest.approx(2 * np.pi / 3, abs=1e-10)

    def test_degenerate_folded(self):
        """k3 = k1 + k2 → θ = 0 (folded / collinear)."""
        theta = utils.get_theta(0.1, 0.1, 0.2)
        assert theta == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_squeezed(self):
        """k3 ≈ |k1 − k2| → θ ≈ π (squeezed / anti-collinear)."""
        theta = utils.get_theta(0.1, 0.05, 0.05)
        assert theta == pytest.approx(np.pi, abs=1e-10)

    def test_vectorised(self):
        """Should broadcast over arrays."""
        k1 = np.array([0.1, 0.2])
        k2 = np.array([0.1, 0.2])
        k3 = np.array([0.1, 0.2])
        theta = utils.get_theta(k1, k2, k3)
        np.testing.assert_allclose(theta, 2 * np.pi / 3, atol=1e-10)


class TestGetK3:
    def test_round_trip(self):
        """get_k3(get_theta(k1,k2,k3), k1, k2) ≈ k3."""
        k1, k2, k3_in = 0.1, 0.15, 0.12
        theta = utils.get_theta(k1, k2, k3_in)
        k3_out = utils.get_k3(theta, k1, k2)
        assert k3_out == pytest.approx(k3_in, rel=1e-10)

    def test_zero_k3_clamped(self):
        """When θ = π and k1 = k2, k3 → 0 is clamped to 1e-4."""
        k3 = utils.get_k3(np.pi, 0.1, 0.1)
        assert k3 == pytest.approx(1e-4)


class TestGetThetaK3:
    def test_theta_from_k3(self):
        theta_expected = utils.get_theta(0.1, 0.1, 0.1)
        k3, theta = utils.get_theta_k3(0.1, 0.1, 0.1, None)
        assert theta == pytest.approx(theta_expected)
        assert k3 == pytest.approx(0.1)

    def test_k3_from_theta(self):
        k3, theta = utils.get_theta_k3(0.1, 0.1, None, 2 * np.pi / 3)
        assert k3 == pytest.approx(0.1, rel=1e-5)

    def test_raises_without_either(self):
        with pytest.raises(ValueError):
            utils.get_theta_k3(0.1, 0.1, None, None)


# ── Broadcasting helper ──────────────────────────────────────────────────────


class TestEnableBroadcasting:
    def test_adds_axes(self):
        a = np.array([1.0, 2.0])
        (b,) = utils.enable_broadcasting(a, n=2)
        assert b.shape == (2, 1, 1)

    def test_scalars_unchanged(self):
        (s,) = utils.enable_broadcasting(3.14, n=2)
        assert s == 3.14

    def test_multiple_args(self):
        a = np.ones(3)
        b, c = utils.enable_broadcasting(a, 5.0, n=3)
        assert b.shape == (3, 1, 1, 1)
        assert c == 5.0


# ── Decorators ────────────────────────────────────────────────────────────────


class TestAddEmptyMethods:
    def test_pk_decorator(self):
        @utils.add_empty_methods_pk("l3", "l4")
        class DummyPk:
            pass

        k = np.ones(5)
        assert np.all(DummyPk.l3(None, k) == 0)
        assert np.all(DummyPk.l4(None, k) == 0)

    def test_bk_decorator(self):
        @utils.add_empty_methods_bk("l3", "l4")
        class DummyBk:
            pass

        k = np.ones(5)
        result = DummyBk.l3(None, k, k, k)
        assert result.shape == (5,)
        assert np.all(result == 0)

    def test_does_not_overwrite(self):
        @utils.add_empty_methods_pk("l0")
        class HasL0:
            @staticmethod
            def l0(cosmo_funcs, k1, zz=0):
                return np.ones_like(k1)

        assert np.all(HasL0.l0(None, np.ones(3)) == 1)


# ── CachedSpline ──────────────────────────────────────────────────────────────


class TestCachedSpline:
    """CachedSpline must be a CubicSpline that happens to be faster.

    It memoises scalar evaluations, which is only sound because no spline is ever mutated
    in place - see the class docstring. These pin that it answers identically, that the
    caching follows derivative(), and that one spline's cache never reaches another's.
    """

    @pytest.fixture
    def curves(self):
        zz = np.linspace(0, 5, 200)
        return utils.CachedSpline(zz, np.sin(zz)), CubicSpline(zz, np.sin(zz))

    @pytest.mark.parametrize("x", [1.234, 0.0, 5.0, np.float64(2.5), np.array(3.7)])
    def test_scalar_matches_plain_spline(self, curves, x):
        cached_spl, plain = curves
        assert cached_spl(x) == plain(x)
        assert cached_spl(x) == plain(x)  # again, now off the cache

    def test_array_and_derivative_order_match(self, curves):
        cached_spl, plain = curves
        zz = np.linspace(0.1, 4.9, 37)
        np.testing.assert_array_equal(cached_spl(zz), plain(zz))
        for nu in (1, 2):
            assert cached_spl(1.234, nu) == plain(1.234, nu)

    def test_derivative_keeps_caching(self, curves):
        """PPoly builds derivative() through construct_fast(cls), so the subclass survives -
        which is what gets Pk_d/Pk_dd and dH_c cached without touching where they are built."""
        cached_spl, plain = curves
        assert isinstance(cached_spl.derivative(nu=1), utils.CachedSpline)
        assert cached_spl.derivative(nu=1)(1.234) == plain.derivative(nu=1)(1.234)

    def test_is_a_cubic_spline(self, curves):
        """It replaces CubicSpline on public attributes (cosmo_funcs.D, .Pk, the biases),
        so anything type-checking those must still be satisfied."""
        assert isinstance(curves[0], CubicSpline)

    def test_cache_is_per_spline(self):
        """The cache lives on the spline, so two splines asked for the same x must not share."""
        zz = np.linspace(0, 5, 200)
        sin_spl, cos_spl = utils.CachedSpline(zz, np.sin(zz)), utils.CachedSpline(zz, np.cos(zz))
        sin_spl(1.234)
        assert cos_spl(1.234) == CubicSpline(zz, np.cos(zz))(1.234)

    def test_cache_is_bounded(self, curves):
        cached_spl, plain = curves
        for x in np.linspace(0.01, 4.99, cached_spl._MAX + 20):
            cached_spl(x)
        assert len(cached_spl.__dict__["_at"]) <= cached_spl._MAX
        assert cached_spl(1.234) == plain(1.234)  # still correct after the clear

    def test_cached_retypes_and_passes_others_through(self, curves):
        """`cached` is for splines that arrive built - the SplineStack views and accel's."""
        _, plain = curves
        retyped = utils.cached(plain)
        assert isinstance(retyped, utils.CachedSpline)
        assert retyped(1.234) == plain(1.234)
        assert utils.cached(retyped) is retyped
        sentinel = lambda zz: zz  # noqa: E731 - a lambda bias is not a spline
        assert utils.cached(sentinel) is sentinel

    def test_pickles_without_its_cache(self, curves):
        cached_spl, plain = curves
        cached_spl(1.234)
        restored = pickle.loads(pickle.dumps(cached_spl))
        assert "_at" not in restored.__dict__  # before evaluating it, which would refill it
        assert restored(1.234) == plain(1.234)

    def test_spline_stack_views_are_cached(self):
        """f_d/D_d and the beta curves reach callers as SplineStack views."""
        zz = np.linspace(0, 5, 200)
        stack = utils.SplineStack(zz, np.vstack([np.sin(zz), np.cos(zz)]))
        assert isinstance(stack[0], utils.CachedSpline)
        assert stack[0](1.234) == pytest.approx(np.sin(1.234), abs=1e-8)


# ── copy ──────────────────────────────────────────────────────────────────────


class TestCopy:
    """utils.copy must produce a *deeply independent* object.

    The forecast/derivative machinery mutates copies in place - e.g.
    ``modify_func(cf.survey[t], param, ..., do_copy=False)`` does
    ``setattr`` on a tracer, and ``cf.survey[i] = ...`` rebinds the list -
    so a shallow copy (which shares ``survey`` and its tracers) would corrupt
    the original. These tests pin that independence, and guard the Python-3.14
    regression where deep-copying scipy CubicSpline biases raised
    ``cannot pickle 'module' object``.
    """

    def test_copy_does_not_raise_with_scipy_biases(self, cosmo_funcs):
        """Tracers hold scipy CubicSpline biases (with a module in their state);
        copy must not choke on them (regression: py3.14 'cannot pickle module')."""
        cf_copy = utils.copy(cosmo_funcs)
        assert cf_copy is not cosmo_funcs

    def test_survey_list_and_tracers_independent(self, cosmo_funcs):
        """The survey list and its tracer objects must be fresh, not shared."""
        cf_copy = utils.copy(cosmo_funcs)
        assert cf_copy.survey is not cosmo_funcs.survey
        for orig, new in zip(cosmo_funcs.survey, cf_copy.survey):
            if orig is not None:
                assert new is not orig

    def test_cosmo_shared_by_reference(self, cosmo_funcs):
        """cosmo (and emu if present) are heavy/unpicklable singletons - shared, not cloned.
        survey_params is immutable reference data - shared to skip the costliest copy branch."""
        cf_copy = utils.copy(cosmo_funcs)
        assert cf_copy.cosmo is cosmo_funcs.cosmo
        assert cf_copy.survey_params is cosmo_funcs.survey_params
        if getattr(cosmo_funcs, "emu", None) is not None:
            assert cf_copy.emu is cosmo_funcs.emu

    def test_cosmology_splines_shared_but_reassign_isolated(self, cosmo_funcs):
        """Large immutable cosmology splines are shared by reference (the speed win),
        yet *reassigning* one on the copy must not affect the original."""
        cf_copy = utils.copy(cosmo_funcs)
        assert cf_copy.Pk is cosmo_funcs.Pk  # shared, not deep-copied
        sentinel = object()
        cf_copy.Pk = sentinel  # reassignment rebinds the copy's slot only
        assert cosmo_funcs.Pk is not sentinel

    def test_inplace_edit_of_copy_does_not_touch_original(self, cosmo_funcs):
        """Editing a tracer bias on the copy in place must not affect the original."""
        cf_copy = utils.copy(cosmo_funcs)
        zz = 1.0
        orig_val = cosmo_funcs.survey[0].b_1(zz)
        # same in-place edit the Fisher derivative code performs
        utils.modify_func(cf_copy.survey[0], "b_1", lambda f: f + 100.0, do_copy=False)
        assert cf_copy.survey[0].b_1(zz) == pytest.approx(orig_val + 100.0)
        assert cosmo_funcs.survey[0].b_1(zz) == pytest.approx(orig_val)

    def test_png_holder_edit_of_copy_does_not_touch_original(self, cosmo_funcs):
        """The loc/eq/orth holders are the second level the derivative code edits in place
        (the png_amp_bias params, e.g. loc_b_01) - they must be fresh on the copy too."""
        cf_copy = utils.copy(cosmo_funcs)
        zz = 1.0
        orig_val = cosmo_funcs.survey[0].loc.b_01(zz)
        assert cf_copy.survey[0].loc is not cosmo_funcs.survey[0].loc
        utils.modify_func(cf_copy.survey[0].loc, "b_01", lambda f: f * 2.0, do_copy=False)
        assert cf_copy.survey[0].loc.b_01(zz) == pytest.approx(2 * orig_val)
        assert cosmo_funcs.survey[0].loc.b_01(zz) == pytest.approx(orig_val)

    def test_deriv_cache_of_copy_is_not_shared(self, cosmo_funcs):
        """`deriv` is filled in place (unpack.py: `tracer.deriv['beta'] = ...`) with values
        built from the tracer's biases, so a shifted copy sharing the dict would poison the
        fiducial tracer's cache - a wrong answer, not a crash."""
        cf_copy = utils.copy(cosmo_funcs)
        assert cf_copy.survey[0].deriv is not cosmo_funcs.survey[0].deriv
        cf_copy.survey[0].deriv["beta"] = "sentinel"
        assert "beta" not in cosmo_funcs.survey[0].deriv

    def test_repeated_tracer_stays_one_object(self, cosmo_funcs):
        """survey is e.g. [X, Y, X]; callers edit `set(cf.survey)` once per unique tracer,
        so a repeat must map to the same copy or the edit would miss a slot."""
        cf_copy = utils.copy(cosmo_funcs)
        for i, first in enumerate(cosmo_funcs.survey):
            for j, second in enumerate(cosmo_funcs.survey):
                assert (cf_copy.survey[i] is cf_copy.survey[j]) == (first is second)

    def test_survey_bf_split_returns_distinct_tracers(self, cosmo):
        """BF_split (which calls utils.copy on a survey holding scipy splines) yields
        two independent tracer objects."""
        bright, faint = cw.SurveyParams.Euclid(cosmo).BF_split(6e-16)
        assert bright is not faint
        assert bright.n_g is not faint.n_g


# ── get_cosmo ─────────────────────────────────────────────────────────────────


class TestGetCosmo:
    def test_returns_class_object(self, cosmo):
        from classy import Class

        assert isinstance(cosmo, Class)

    def test_h_value(self, cosmo):
        assert cosmo.h() == pytest.approx(0.67, rel=1e-5)

    def test_emulator_returns_tuple(self):
        c, params = utils.get_cosmo(emulator=True)
        assert isinstance(params, dict)
        assert "h" in params
