"""The numba kernels in numeric_mu.accel must agree with the numpy ones they replace.

Two implementations of the same maths only stay in step if something checks, so these run
both paths in one process - the jitted functions are compiled whenever numba is installed,
independently of which one accel bound at import.
"""

import numpy as np
import pytest

from cosmo_wap.lib import utils
from cosmo_wap.numeric_mu import accel
from cosmo_wap.numeric_mu import pk as npk

numba = pytest.importorskip("numba", reason="numba is optional - accel falls back to numpy")


@pytest.fixture(scope="module")
def blocks(cosmo_funcs):
    """The inputs I1_sum and s1_sum hand to the two blocks, at a sampler-sized grid."""
    zz = 1.0
    kernels = ["N", "LP", "I"]
    kk = np.logspace(-2.5, np.log10(0.15), 12)[:, np.newaxis]
    mu, _ = npk.get_mu_grid(48, 0.1, True)
    mu = mu[len(mu) // 2 :]  # get_mu_sym only ever evaluates the mu >= 0 half

    d = cosmo_funcs.comoving_dist(zz)
    nodes, _ = utils.leggauss(8)
    r2 = d * (nodes + 1) / 2.0

    powers, term_weights, spline = npk.get_int_K1(["I"], cosmo_funcs, zz)
    r2_arr = npk.get_int_K2(["I"], r2, cosmo_funcs, zz, mu, kk)
    mu_b, kk_b = utils.enable_broadcasting(mu, kk, n=1)

    u = d / r2
    integrand = npk.get_K(["N", "LP"], cosmo_funcs, zz, mu_b, kk_b * u, ti=0) * r2_arr
    return dict(
        kernels=kernels,
        zz=zz,
        kk=kk,
        mu=mu,
        powers=powers,
        term_weights=term_weights,
        spline=spline,
        qq_II=kk_b / (r2 / d),  # II: q rescaled onto each line-of-sight node
        qq_IS=kk,  # IS: the second field is a source kernel, so q = k
        mu_b=mu_b,
        kk_b=kk_b,
        u=u,
        integrand=integrand,
        d=d,
    )


@pytest.mark.parametrize("case", ["II", "IS"])
def test_kernel_sum_matches_numpy(blocks, case):
    qq = blocks["qq_II"] if case == "II" else blocks["qq_IS"]
    mu = blocks["mu_b"] if case == "II" else blocks["mu"]
    args = (blocks["spline"], blocks["powers"], blocks["term_weights"], qq, mu)

    ref = accel._kernel_sum_np(*args)
    got = accel._kernel_sum_nb(*args)

    assert got.shape == np.asarray(ref).shape
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-13


def test_filon_matches_numpy(blocks):
    # as s1_sum calls it: the u grid is reversed so it increases
    args = (blocks["u"][::-1], blocks["kk_b"], blocks["mu_b"], blocks["integrand"][..., ::-1], blocks["d"])

    ref = accel._filon_np(*args)
    got = accel._filon_nb(*args)

    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-13


def test_multipoles_match_numpy(monkeypatch, cosmo_funcs, blocks):
    """End to end: the whole numeric-mu path down either binding."""
    kernels, kk, zz = blocks["kernels"], blocks["kk"][:, 0], blocks["zz"]

    def multipoles():
        return npk.get_multipoles(kernels, kernels, [0, 2, 4], cosmo_funcs, kk, zz, fNL=2.0)

    monkeypatch.setattr(accel, "kernel_sum", accel._kernel_sum_np)
    monkeypatch.setattr(accel, "filon_integrate", accel._filon_np)
    ref = multipoles()

    monkeypatch.setattr(accel, "kernel_sum", accel._kernel_sum_nb)
    monkeypatch.setattr(accel, "filon_integrate", accel._filon_nb)
    got = multipoles()

    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


def test_disable_env_var_is_honoured():
    """COSMOWAP_DISABLE_NUMBA has to reach the binding, not just the import."""
    import importlib

    import cosmo_wap.numeric_mu.accel as mod

    assert mod.kernel_sum is (mod._kernel_sum_nb if mod.HAVE_NUMBA else mod._kernel_sum_np)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("COSMOWAP_DISABLE_NUMBA", "1")
        reloaded = importlib.reload(mod)
        try:
            assert not reloaded.HAVE_NUMBA
            assert reloaded.kernel_sum is reloaded._kernel_sum_np
            assert reloaded.filon_integrate is reloaded._filon_np
        finally:
            importlib.reload(mod)  # leave the module as the rest of the suite found it
