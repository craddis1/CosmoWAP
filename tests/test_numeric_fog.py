"""FoG inside numerical projections and throughout forecast signal/covariance plumbing."""

import numpy as np
import pytest
from scipy.special import eval_legendre, sph_harm_y

from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import contract, joint_inv_cov
from cosmo_wap.forecast.covariances import BBCovBk, FullCovBk, FullCovPk, PBCov
from cosmo_wap.forecast.sampler import Sampler
from cosmo_wap.lib import utils
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import pk as npk

SIGMA = 8.0
KERNELS = ["N", "LP"]


@pytest.fixture(params=[False, True], ids=["single", "multi"])
def fog_forecast(request, cosmo_funcs, forecast_mt):
    mt = request.param
    cf = forecast_mt.cosmo_funcs if mt else cosmo_funcs
    return FullForecast(cf, kmax_func=0.035, s_k=3, N_bins=1), mt


@pytest.mark.parametrize("kernels", [["N"], ["N", "LP"], ["N", "LP", "I"]])
def test_pk_fog_matches_full_mu_integral(fog_forecast, kernels):
    forecast, _ = fog_forecast
    cf = forecast.cosmo_funcs
    kk, z = np.array([0.01, 0.08, 0.2]), 1.2
    ln = [0, 1, 2, 3, 4]
    grid = dict(n_mu=64, n=8, deg=8, n_p=200)
    mu, w = npk.get_mu_grid(64)
    raw = npk.get_mu(mu, kernels, kernels, cf, kk[:, None], z, n=8, deg=8, n_p=200)
    fog = np.exp(-0.5 * (kk[:, None] * mu * SIGMA) ** 2)
    ref = np.array([(2 * l + 1) / 2 * np.sum(w * eval_legendre(l, mu) * fog * raw, axis=-1) for l in ln])
    actual = npk.get_multipoles(kernels, kernels, ln, cf, kk, z, sigma=SIGMA, **grid)
    np.testing.assert_allclose(actual, ref, rtol=1e-11, atol=1e-10)
    undamped = npk.get_multipoles(kernels, kernels, ln, cf, kk, z, **grid)
    np.testing.assert_array_equal(npk.get_multipoles(kernels, kernels, ln, cf, kk, z, sigma=0, **grid), undamped)
    assert not np.allclose(actual, undamped)


def test_bk_fog_matches_full_sphere_integral(fog_forecast):
    forecast, _ = fog_forecast
    cf = forecast.cosmo_funcs
    k1, k2, theta = np.array([0.02, 0.1]), np.array([0.035, 0.08]), np.array([2.0, 2.2])
    k3, theta = utils.get_theta_k3(k1, k2, None, theta)
    mu, w = utils.leggauss(48)
    phi = 2 * np.pi * (np.arange(64) + 0.5) / 64
    args = tuple(k[:, None, None] for k in (k1, k2, k3, theta))
    raw = nbk.get_mu_phi(mu, phi, KERNELS, KERNELS, KERNELS, cf, *args, 1.2)
    mu1, mu2, mu3 = nbk.los_cosines(mu[:, None], phi, *args)
    fog = np.exp(-0.5 * SIGMA**2 * sum((k * m) ** 2 for k, m in zip(args[:3], (mu1, mu2, mu3))))
    lm = [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)]
    ref = np.array(
        [
            2
            * np.pi
            / len(phi)
            * np.sum(w[:, None] * sph_harm_y(l, m, np.arccos(mu)[:, None], phi).conj() * fog * raw, axis=(-2, -1))
            for l, m in lm
        ]
    )
    actual = nbk.get_multipoles(KERNELS, KERNELS, KERNELS, lm, cf, k1, k2, k3, theta, zz=1.2, sigma=SIGMA)
    np.testing.assert_allclose(actual, ref, rtol=1e-10, atol=1e-7)
    undamped = nbk.get_multipoles(KERNELS, KERNELS, KERNELS, lm, cf, k1, k2, k3, theta, zz=1.2)
    np.testing.assert_array_equal(
        nbk.get_multipoles(KERNELS, KERNELS, KERNELS, lm, cf, k1, k2, k3, theta, zz=1.2, sigma=0), undamped
    )
    assert np.all(actual[0].real < undamped[0].real)


def test_pk_covariance_damps_clustering_before_shot_noise(fog_forecast):
    forecast, mt = fog_forecast
    fc = forecast.get_pk_bin(0, all_tracer=mt)
    cov = FullCovPk(fc, fc.cf_mat, KERNELS, sigma=SIGMA)
    fog = np.exp(-0.5 * (cov.args[1] * cov.mu * SIGMA) ** 2)

    def P(a, b):
        return cov.pk_cache[a][b] * fog + 1 / cov.cf_mat[a][b].n_g(cov.zz)

    pairs = [(0, 0), (0, 1), (1, 1)] if mt else [(0, 0)]
    ref = np.array(
        [
            [np.sum(cov.weights * (P(a, c) * P(b, d).conj() + P(a, d) * P(b, c).conj()), axis=-1) / 2 for c, d in pairs]
            for a, b in pairs
        ]
    )
    np.testing.assert_allclose(cov.get_cov([0]), ref, rtol=1e-12)
    # Omitted sigma preserves the constructor setting; explicit None disables it.
    undamped = cov.get_cov([0], sigma=None)
    assert not np.allclose(ref, undamped)
    np.testing.assert_array_equal(cov.get_cov([0], sigma=0), undamped)


def test_bk_covariance_damps_each_leg_before_shot_noise(cosmo_funcs):
    forecast = FullForecast(cosmo_funcs, kmax_func=0.035, s_k=3, N_bins=1)
    fc = forecast.get_bk_bin(0)
    cov = FullCovBk(fc, fc.cf_mat, KERNELS, sigma=SIGMA)
    P = [
        cov.pk_cache[i][0][0] * np.exp(-0.5 * (cov.ks[i] * cov.mus[i] * SIGMA) ** 2) + 1 / cosmo_funcs.n_g(cov.zz)
        for i in range(3)
    ]
    ref = fc.s123 * np.pi * np.sum(cov.weights * P[0] * P[1] * P[2], axis=(-2, -1))
    np.testing.assert_allclose(cov.get_cov([0])[0, 0], ref, rtol=1e-12)


def test_ng_shared_power_spectrum_is_damped(cosmo_funcs):
    forecast = FullForecast(cosmo_funcs, kmax_func=0.035, s_k=3, N_bins=1)
    pkfc, bkfc = forecast.get_pk_bin(0), forecast.get_bk_bin(0)
    bb = BBCovBk(bkfc, ["N"], [0], sigma=SIGMA, n_mu=8)
    pb = PBCov(pkfc, bb, [0])
    k = pkfc.args[1][:, None]
    z = pkfc.z_mid
    P = npk.get_mu(bb.mu, ["N"], ["N"], cosmo_funcs, k, z) * np.exp(-0.5 * (k * bb.mu * SIGMA) ** 2)
    expected = 2 / k * (P + 1 / cosmo_funcs.n_g(z))
    np.testing.assert_allclose(pb.V[:, 0, 0, : len(bb.mu)], expected, rtol=1e-12)
    # The PT shot-noise column also divides by the damped clustering P on the shared leg.
    k = bkfc.args[1][:, None]
    P = npk.get_mu(bb.mu, ["N"], ["N"], cosmo_funcs, k, z) * np.exp(-0.5 * (k * bb.mu * SIGMA) ** 2)
    expected = bb.U[:, 0, 0, : len(bb.mu)] / np.sqrt(cosmo_funcs.n_g(z) * P)
    np.testing.assert_allclose(bb.U[:, 0, 0, len(bb.mu) :], expected, rtol=1e-12)


def test_ssc_response_fog_in_isotropic_limit(cosmo_funcs):
    """Without RSD the response is isotropic; FoG alone generates its higher multipoles."""
    cf = utils.copy(cosmo_funcs)
    cf.f = lambda z: 0 * z
    fc = FullForecast(cf, kmax_func=0.15, s_k=3, N_bins=1).get_pk_bin(0)
    ln = [0, 2, 4]
    monopole = fc.get_data_vector(None, [0], param="delta_b", kernels=["N"])[0]
    mu, w = utils.leggauss(64)
    fog = np.exp(-0.5 * (fc.args[1][:, None] * mu * SIGMA) ** 2)
    ref = np.array([(2 * l + 1) / 2 * np.sum(w * eval_legendre(l, mu) * fog, axis=-1) * monopole for l in ln])
    actual = fc.get_data_vector(None, ln, param="delta_b", kernels=["N"], sigma=SIGMA)
    np.testing.assert_allclose(actual, ref, rtol=1e-7, atol=1e-9 * np.max(np.abs(ref)))
    assert not np.allclose(actual[0], monopole)


@pytest.mark.parametrize("cov_ng", [False, True])
def test_fisher_and_snr_forward_numeric_fog(fog_forecast, cov_ng):
    forecast, mt = fog_forecast
    forecast.cov_ng = cov_ng
    forecast.ng_kwargs = dict(n_mu=8, n_psi=8)
    kw = dict(
        terms=None,
        pkln=[0, 2],
        bkln=[0, 1, 2],
        kernels=KERNELS,
        bk_kernels=KERNELS,
        sigma=SIGMA,
        all_tracer=mt,
        verbose=False,
    )
    fish = forecast.get_fish(["LP"], **kw)
    expected = 0
    derivatives, bins = [], []
    for probe, ln in [("pk", kw["pkln"]), ("bk", kw["bkln"])]:
        fc = getattr(forecast, f"get_{probe}_bin")(0, all_tracer=mt)
        derivative = fc.get_data_vector(None, ln, param="LP", kernels=KERNELS, sigma=SIGMA)
        derivatives.append(derivative)
        bins.append(fc)
        np.testing.assert_array_equal(forecast.derivs[0][0][probe], derivative)
        value = contract(derivative, fc.get_inv_cov(ln, sigma=SIGMA), derivative).real
        expected += value
        opts = {f"{probe}ln": ln, "kernels" if probe == "pk" else "bk_kernels": KERNELS}
        snr = getattr(forecast, f"{probe}_SNR")(None, param="LP", sigma=SIGMA, all_tracer=mt, verbose=False, **opts)
        np.testing.assert_allclose(snr[0], value, rtol=1e-6)
    if cov_ng:
        inv_cov = joint_inv_cov(*bins, kw["pkln"], kw["bkln"], sigma=SIGMA)
        expected = contract(tuple(derivatives), inv_cov, tuple(derivatives)).real
    np.testing.assert_allclose(fish.fisher_matrix[0, 0], expected, rtol=1e-10)
    kw["term"] = kw.pop("terms")
    np.testing.assert_allclose(forecast.combined_SNR(param="LP", **kw)[0], expected, rtol=1e-6)


def test_sampler_bias_derivative_matches_damped_fisher(fog_forecast):
    forecast, mt = fog_forecast
    kw = dict(
        terms=None,
        pkln=[0, 2],
        bkln=[0, 1, 2],
        kernels=KERNELS,
        bk_kernels=KERNELS,
        sigma=SIGMA,
        all_tracer=mt,
        verbose=False,
    )
    forecast.get_fish(["A_b_1"], **kw)
    sampler = forecast.sampler(["LP", "A_b_1"], fisher_covmat=False, drag=False, **kw)
    h = 1e-4
    above, below = sampler.get_theory([1, 1 + h]), sampler.get_theory([1, 1 - h])
    for probe in ("pk", "bk"):
        derivative = (above[0][probe] - below[0][probe]) / (2 * h)
        ref = forecast.derivs[0][0][probe]
        np.testing.assert_allclose(derivative, ref, rtol=1e-6, atol=1e-9 * np.max(np.abs(ref)))


@pytest.mark.parametrize("cov_ng", [False, True])
def test_sampler_fog_data_theory_covariance_proposal_and_save(fog_forecast, cov_ng, monkeypatch, tmp_path):
    forecast, mt = fog_forecast
    forecast.cov_ng = cov_ng
    forecast.ng_kwargs = dict(n_mu=8, n_psi=8)
    calls = []
    original = forecast.get_fish

    def recorded(*args, **kwargs):
        calls.append(kwargs.get("sigma"))
        return original(*args, **kwargs)

    monkeypatch.setattr(forecast, "get_fish", recorded)
    s = forecast.sampler(
        ["LP"],
        terms=None,
        pkln=[0, 2],
        bkln=[0, 1, 2],
        kernels=KERNELS,
        bk_kernels=KERNELS,
        sigma=SIGMA,
        all_tracer=mt,
        drag=False,
        data_cosmo_funcs=forecast.cosmo_funcs,
    )
    assert calls == [SIGMA]
    assert s.sigma == SIGMA
    bins = {p: getattr(forecast, f"get_{p}_bin")(0, all_tracer=mt) for p in ("pk", "bk")}
    for probe, ln in [("pk", s.pkln), ("bk", s.bkln)]:
        fc = bins[probe]
        grid = {"mu_grid": s.bk_mu_grid} if probe == "bk" else {}  # the sampler's bk (mu, phi) grid, not bk_func's
        full = fc.get_data_vector(None, ln, kernels=KERNELS, sigma=SIGMA, **grid)
        reduced = fc.get_data_vector(None, ln, kernels=["N"], sigma=SIGMA, **grid)
        np.testing.assert_allclose(s.data[0][0][probe], full, rtol=1e-12, atol=1e-7)
        np.testing.assert_allclose(s.get_theory([1])[0][probe], full, rtol=1e-12, atol=1e-7)
        np.testing.assert_allclose(s.get_theory([0])[0][probe], reduced, rtol=1e-12, atol=1e-7)
        assert not np.allclose(full, fc.get_data_vector(None, ln, kernels=KERNELS, **grid))
        if not cov_ng:
            np.testing.assert_allclose(s.inv_covs[0][probe], fc.get_inv_cov(ln, sigma=SIGMA), rtol=1e-9, atol=1e-20)
    if cov_ng:
        ref = joint_inv_cov(
            bins["pk"],
            bins["bk"],
            s.pkln,
            s.bkln,
            sigma=SIGMA,
            n_mu_pk=forecast.n_mu,
            n_mu=forecast.n_mu,
            n_phi=forecast.n_phi,
        )
        data = tuple(s.data[0][0][p] for p in ("pk", "bk"))
        np.testing.assert_allclose(contract(data, s.inv_covs[0]["pkbk"], data), contract(data, ref, data), rtol=1e-10)
    assert abs(s.get_likelihood(LP=1)) < 1e-10
    s.samples_df = None
    path = tmp_path / "fog.pkl"
    s.save(path)
    loaded = Sampler.load(path, forecast)
    assert loaded.sigma == SIGMA
    for probe in ("pk", "bk"):
        np.testing.assert_allclose(
            loaded.get_theory([0.7])[0][probe], s.get_theory([0.7])[0][probe], rtol=1e-12, atol=1e-7
        )
