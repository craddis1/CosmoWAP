"""Numerical spectrum amplitudes, including grouped parameters and sampler/Fisher agreement."""

import numpy as np
import pytest

from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import contract
from cosmo_wap.forecast.sampler import Sampler
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import pk as npk

MU_GRID = [16, True, 4, 6, 200]


@pytest.fixture
def small_fc(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.035, s_k=3, N_bins=1)


@pytest.fixture
def small_mt(forecast_mt):
    return FullForecast(forecast_mt.cosmo_funcs, kmax_func=0.035, s_k=3, N_bins=1)


@pytest.mark.parametrize("kernel", ["N", "LP", "I", "L", "TD", "ISW", "kappa_g", "Loc"])
def test_pk_kernel_difference(small_fc, kernel):
    fc = small_fc.get_pk_bin(0)
    kernels = ["N", kernel] if kernel != "N" else ["N", "LP"]
    kw = dict(mu_grid=MU_GRID, fNL=2.0)
    full = fc.get_data_vector(None, [0, 2], kernels=kernels, **kw)
    reduced = fc.get_data_vector(None, [0, 2], kernels=[k for k in kernels if k != kernel], **kw)
    actual = fc.get_data_vector("GR2", [0, 2], param=kernel, kernels=kernels, **kw)
    np.testing.assert_allclose(actual, full - reduced, rtol=1e-12)


@pytest.mark.parametrize("fc_name", ["small_fc", "small_mt"])
@pytest.mark.parametrize("kernel", ["N", "LP", "Loc"])
def test_bk_kernel_difference_with_fog(fc_name, kernel, request):
    fc = request.getfixturevalue(fc_name).get_bk_bin(0, all_tracer=fc_name == "small_mt")
    kernels = ["N", "LP", "Loc"]
    kw = dict(sigma=5.0, fNL=2.0)
    full = fc.get_data_vector(None, [0, 1, 2], kernels=kernels, **kw)
    reduced = fc.get_data_vector(None, [0, 1, 2], kernels=[k for k in kernels if k != kernel], **kw)
    actual = fc.get_data_vector(None, [0, 1, 2], param=kernel, kernels=kernels, **kw)
    np.testing.assert_allclose(actual, full - reduced, rtol=1e-12)


@pytest.mark.parametrize("probe", ["pk", "bk"])
def test_removing_only_kernel_gives_full_signal(small_fc, probe):
    fc = getattr(small_fc, f"get_{probe}_bin")(0)
    full = fc.get_data_vector(None, [0, 2], kernels="N")
    derivative = fc.get_data_vector(None, [0, 2], kernels="N", param="N")
    np.testing.assert_array_equal(derivative, full)


def test_composite_is_sum_and_shares_full_evaluation(small_fc, monkeypatch):
    fc = small_fc.get_pk_bin(0)
    kw = dict(kernels=["N", "LP", "I"], mu_grid=MU_GRID)
    expected = sum(fc.get_data_vector(None, [0, 2], param=p, **kw) for p in ["LP", "I"])
    calls = []
    original = npk.get_multipoles

    def counted(k1, k2, *args, **kwargs):
        calls.append(tuple(k1))
        return original(k1, k2, *args, **kwargs)

    monkeypatch.setattr(npk, "get_multipoles", counted)
    actual = fc.get_data_vector(None, [0, 2], param=["LP", "I"], **kw)
    np.testing.assert_array_equal(actual, expected)
    assert calls == [("N", "LP", "I"), ("N", "I"), ("N", "LP")]


def test_grouped_fisher_joint_probes_and_fiducial(small_fc):
    fish = small_fc.get_fish(
        [["LP", "I"]],
        terms=None,
        pkln=[0, 2],
        bkln=[0, 1, 2],
        kernels=["N", "LP", "I"],
        bk_kernels=["N", "LP"],
        mu_grid=MU_GRID,
        verbose=False,
    )
    pkfc, bkfc = small_fc.get_pk_bin(0), small_fc.get_bk_bin(0)
    pkd = sum(
        pkfc.get_data_vector(None, [0, 2], param=p, kernels=["N", "LP", "I"], mu_grid=MU_GRID) for p in ["LP", "I"]
    )
    bkd = bkfc.get_data_vector(None, [0, 1, 2], kernels=["N", "LP"], param="LP")
    np.testing.assert_array_equal(bkfc.get_data_vector(None, [0, 1, 2], kernels=["N", "LP"], param="I"), 0)
    expected = contract(pkd, pkfc.get_inv_cov([0, 2]), pkd) + contract(bkd, bkfc.get_inv_cov([0, 1, 2]), bkd)
    np.testing.assert_allclose(fish.fisher_matrix[0, 0], expected.real, rtol=1e-10)
    assert fish.param_list == ["LP_I"]
    assert fish.fiducial["LP_I"] == 1


def test_separate_fisher_amplitudes_share_full_evaluation(small_fc, monkeypatch):
    calls = []
    original = nbk.get_multipoles

    def counted(k1, *args, **kwargs):
        calls.append(tuple(k1))
        return original(k1, *args, **kwargs)

    monkeypatch.setattr(nbk, "get_multipoles", counted)
    fish = small_fc.get_fish(["N", "LP"], terms=None, bkln=[0, 1, 2], bk_kernels=["N", "LP"], verbose=False)
    assert np.all(np.isfinite(fish.errors))
    assert calls == [("N", "LP"), ("LP",), ("N",)]


def test_grouped_fisher_metadata_survives_prior_and_full_bin_matrix(small_fc):
    args = dict(terms=None, bkln=[0, 1, 2], bk_kernels=["N", "LP"], verbose=False)
    fish = small_fc.get_fish([["N", "LP"]], bias_list="GR2", **args)
    assert "N_LP" in fish.bias[0]
    assert fish.add_gaussian_priors({"N_LP": 1}).fiducial["N_LP"] == 1
    full = small_fc.get_fish([["N", "LP"]], per_bin_params=["b_1"], marginalize_per_bin=False, **args)
    assert full.fiducial["N_LP"] == 1


@pytest.mark.parametrize("factory", ["fisher", "sampler"])
def test_missing_kernel_parameter_raises(small_fc, factory):
    with pytest.raises(ValueError, match="Kernel amplitude 'I'"):
        if factory == "fisher":
            small_fc.get_fish(["I"], terms=None, bkln=[0], bk_kernels=["N"], verbose=False)
        else:
            Sampler(small_fc, ["I"], terms=None, bkln=[0], bk_kernels=["N"], fisher_covmat=False)


def test_unimplemented_bk_kernel_still_raises(small_fc):
    with pytest.raises(NotImplementedError, match="second order"):
        small_fc.get_fish(["I"], terms=None, bkln=[0], bk_kernels=["N", "I"], verbose=False)


@pytest.mark.parametrize("fc_name", ["small_fc", "small_mt"])
def test_sampler_unit_zero_and_shift_match_fisher(fc_name, request):
    fc = request.getfixturevalue(fc_name)
    mt = fc_name == "small_mt"
    sampler = Sampler(
        fc, ["LP"], terms=None, bkln=[0, 1, 2], bk_kernels=["N", "LP"], all_tracer=mt, fisher_covmat=False, drag=False
    )
    b = fc.get_bk_bin(0, all_tracer=mt)
    grid = sampler.bk_mu_grid  # the sampler's bk (mu, phi) grid, not bk_func's
    full = b.get_data_vector(None, [0, 1, 2], kernels=["N", "LP"], mu_grid=grid)
    reduced = b.get_data_vector(None, [0, 1, 2], kernels=["N"], mu_grid=grid)
    derivative = b.get_data_vector(None, [0, 1, 2], kernels=["N", "LP"], param="LP", mu_grid=grid)
    np.testing.assert_array_equal(sampler.data[0][0]["bk"], full)
    np.testing.assert_array_equal(sampler.get_theory([1])[0]["bk"], full)
    np.testing.assert_allclose(sampler.get_theory([0])[0]["bk"], reduced, atol=1e-7)
    np.testing.assert_allclose(sampler.get_theory([1.2])[0]["bk"], full + 0.2 * derivative)
    assert sampler.fiducial["LP"] == 1
    assert sampler.get_likelihood(LP=1) == pytest.approx(0, abs=1e-12)


def test_sampler_grouped_joint_probes_and_proposal(small_fc):
    sampler = Sampler(
        small_fc,
        [["LP", "I"]],
        terms=None,
        pkln=[0, 2],
        bkln=[0, 1, 2],
        kernels=["N", "LP", "I"],
        bk_kernels=["N", "LP"],
        mu_grid=MU_GRID,
        drag=False,
    )
    full = sampler.get_theory([1])[0]
    moved = sampler.get_theory([1.25])[0]
    for probe, ln, kernels in [("pk", [0, 2], ["N", "LP", "I"]), ("bk", [0, 1, 2], ["N", "LP"])]:
        b = getattr(small_fc, f"get_{probe}_bin")(0)
        kwargs = {"mu_grid": MU_GRID} if probe == "pk" else {"mu_grid": sampler.bk_mu_grid}
        derivative = b.get_data_vector(None, ln, param=["LP", "I"], kernels=kernels, **kwargs)
        np.testing.assert_allclose(moved[probe], full[probe] + 0.25 * derivative)
    assert sampler.param_list == ["LP_I"]
    assert sampler.fiducial["LP_I"] == 1
    assert sampler.get_likelihood(LP_I=1) == pytest.approx(0, abs=1e-12)
    assert sampler.info["sampler"]["mcmc"]["covmat_params"] == ["LP_I"]


def test_sampler_amplitude_cache_rebuilds_for_bias(small_fc, monkeypatch):
    sampler = Sampler(
        small_fc,
        ["LP", "A_b_1"],
        terms=None,
        bkln=[0, 2],
        bk_kernels=["N", "LP"],
        per_bin_params=["b_1"],
        fisher_covmat=False,
        drag=False,
    )
    calls = []
    original = nbk.get_multipoles

    def counted(k1, *args, **kwargs):
        calls.append(tuple(k1))
        return original(k1, *args, **kwargs)

    monkeypatch.setattr(nbk, "get_multipoles", counted)
    base = sampler.get_theory([1, 1, 1])[0]["bk"]
    sampler.get_theory([1.1, 1, 1])
    sampler.get_theory([0, 1, 1])
    assert calls == [("N", "LP"), ("N",)]
    shifted = sampler.get_theory([1, 1.1, 1.2])[0]["bk"]
    assert len(calls) == 4
    assert not np.allclose(base, shifted)
    # Both the baseline and template must carry the changed global and per-bin bias.
    vals = [1.3, 1.1, 1.2]
    cf = sampler.update_cosmo_funcs(vals)
    with sampler._amplitude_bias(cf, vals), sampler._per_bin_bias(cf, vals, 0):
        full = sampler.get_bk_d1(0, None, [0, 2], [cf])
        reduced = sampler.get_bk_d1(0, None, [0, 2], [cf], amplitude="LP")
    np.testing.assert_allclose(sampler.get_theory(vals)[0]["bk"], full + 0.3 * reduced)


def test_sampler_shared_png_name_uses_kernel(small_fc):
    sampler = Sampler(
        small_fc,
        ["Loc", "fNL"],
        terms=None,
        bkln=[0, 2],
        bk_kernels=["N", "LP", "Loc"],
        fisher_covmat=False,
        drag=False,
    )
    b = small_fc.get_bk_bin(0)
    kw = dict(kernels=["N", "LP", "Loc"], fNL=3)
    full = b.get_data_vector(None, [0, 2], **kw)
    derivative = b.get_data_vector(None, [0, 2], param="Loc", **kw)
    np.testing.assert_allclose(sampler.get_theory([1.2, 3])[0]["bk"], full + 0.2 * derivative)
    np.testing.assert_allclose(
        sampler.get_theory([1.2, 0])[0]["bk"], b.get_data_vector(None, [0, 2], kernels=["N", "LP"])
    )


def test_sampler_grouped_save_load(small_fc, tmp_path):
    sampler = Sampler(
        small_fc, [["N", "LP"]], terms=None, bkln=[0, 2], bk_kernels=["N", "LP"], fisher_covmat=False, drag=False
    )
    sampler.samples_df = None
    path = tmp_path / "kernel_sampler.pkl"
    sampler.save(path)
    loaded = Sampler.load(path, small_fc)
    assert loaded.global_param_list == [["N", "LP"]]
    assert loaded.fiducial["N_LP"] == 1
    np.testing.assert_allclose(loaded.get_theory([0.7])[0]["bk"], sampler.get_theory([0.7])[0]["bk"])
