"""IntK1.I shares one unpack across L, TD and ISW - that has to stay a pure hoist.

The kernels are also reached one at a time by name from numeric_mu.pk, so each has to give
the same answer whether it unpacks its own parameters or is handed them.
"""

import numpy as np
import pytest

from cosmo_wap.lib import utils
from cosmo_wap.numeric_mu.kernels import IntK1, Unpack

KERNELS = ["L", "TD", "ISW", "I", "kappa_g"]


def _terms_equal(a, b):
    """Term lists are (mu_pow, q_pow, radial_arr, weight_dict) - compare all four."""
    assert len(a) == len(b)
    for (i1, j1, arr1, wt1), (i2, j2, arr2, wt2) in zip(a, b):
        assert (i1, j1) == (i2, j2)
        assert wt1 == wt2
        np.testing.assert_array_equal(arr1, arr2)


@pytest.fixture(scope="module")
def grid(cosmo_funcs):
    zz = 1.4
    d = cosmo_funcs.comoving_dist(zz)
    nodes, _ = utils.leggauss(8)
    return zz, d * (nodes + 1) / 2.0


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("ti", [0, 1])
def test_supplied_params_match_self_unpacked(cosmo_funcs, grid, name, ti):
    zz, r = grid
    kern = getattr(IntK1, name)

    own = kern(r, cosmo_funcs, zz=zz, ti=ti)
    given = kern(
        r,
        cosmo_funcs,
        zz=zz,
        ti=ti,
        src=Unpack.get_int_params(cosmo_funcs, zz, ti=ti),
        intg=Unpack.get_integrand_params(cosmo_funcs, r),
    )
    _terms_equal(own, given)


@pytest.mark.parametrize("ti", [0, 1])
def test_I_is_exactly_L_plus_TD_plus_ISW(cosmo_funcs, grid, ti):
    """The shared unpack must not change what I returns, term for term."""
    zz, r = grid
    args = dict(zz=zz, ti=ti)
    separate = IntK1.L(r, cosmo_funcs, **args) + IntK1.TD(r, cosmo_funcs, **args) + IntK1.ISW(r, cosmo_funcs, **args)
    _terms_equal(IntK1.I(r, cosmo_funcs, **args), separate)


def test_unpack_is_called_once_per_I(cosmo_funcs, grid, monkeypatch):
    """The point of the hoist: three kernels, one set of spline evaluations."""
    zz, r = grid
    counts = {"src": 0, "intg": 0}
    for key, name in (("src", "get_int_params"), ("intg", "get_integrand_params")):
        f = getattr(Unpack, name)

        def counted(*a, _f=f, _k=key, **kw):
            counts[_k] += 1
            return _f(*a, **kw)

        monkeypatch.setattr(Unpack, name, staticmethod(counted))

    IntK1.I(r, cosmo_funcs, zz=zz, ti=0)
    assert counts == {"src": 1, "intg": 1}
