"""Super-sample covariance as the per-bin nuisance 'delta_b' - its response (core.PkForecast.ssc_response,
numeric_mu.bk.get_P_response), its variance (core.Forecast.sigma_b2) and the prior that marginalises it."""

import numpy as np
import pytest

from cosmo_wap.forecast import FullForecast
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import bk as nbk


def test_response_matches_real_space(cosmo_funcs):
    """f = 0: P_g's response to the linear mode is D^3 P0 [b1^2 C21 + 2 b1 (b2 - 4 g2/3)], C21 = 68/21 -
    dln(k^3 P)/dlnk / 3 - growth and dilation (1910.02914 eqs 29-32). The error is O(eps^2)"""
    zz = 1.0
    real = utils.copy(cosmo_funcs)
    real.f = lambda z: 0 * z
    k = np.array([0.01, 0.03, 0.07, 0.12, 0.2])[:, None]
    mu = np.array([0.1, 0.5, 0.9])
    s = real.survey[0]
    b1, b2, g2 = s.b_1(zz), s.b_2(zz), s.g_2(zz)
    pk = BaseInt(real).pk
    dlnP = (np.log(pk(k * np.exp(1e-4), zz)) - np.log(pk(k * np.exp(-1e-4), zz))) / 2e-4
    C21 = 68 / 21 - (3 + dlnP) / 3
    ref = real.D(zz) ** 3 * pk(k, zz) * (b1**2 * C21 + 2 * b1 * (b2 - 4 * g2 / 3))
    np.testing.assert_allclose(nbk.get_P_response(real, real, zz, mu, k), np.broadcast_to(ref, (5, 3)), rtol=1e-5)


def test_sigma_b2_full_shell(cosmo_funcs, monkeypatch):
    """Full sky only the monopole of the cap survives: V^-2 int d^3k/(2pi)^3 P |W|^2, W = 4pi int r^2 D j0(kr) dr -
    here on an independent trapezoid grid, linear in k to resolve W^2's oscillation at 2 chi"""
    fc = FullForecast(cosmo_funcs, kmax_func=0.1, s_k=2, N_bins=9).get_pk_bin(0)
    monkeypatch.setattr(cosmo_funcs, "f_sky", 1.0)
    chi1, chi2 = cosmo_funcs.comoving_dist(fc.z_bin[0]), cosmo_funcs.comoving_dist(fc.z_bin[1])
    r = np.linspace(chi1, chi2, 2001)
    k = np.concatenate([np.geomspace(1e-5, 1e-3, 100, endpoint=False), np.arange(1e-3, 0.5, 1e-4)])
    W = 4 * np.pi * np.trapezoid(r**2 * cosmo_funcs.D(cosmo_funcs.d_to_z(r)) * np.sinc(k[:, None] * r / np.pi), r)
    V = 4 * np.pi * (chi2**3 - chi1**3) / 3
    ref = np.trapezoid(k**2 * cosmo_funcs.Pk(k) * W**2, k) / (2 * np.pi**2) / V**2
    assert fc.sigma_b2() == pytest.approx(ref, rel=1e-3)


def test_delta_b_marginal_is_ssc_covariance(cosmo_funcs):
    """Marginalising delta_b with its prior 1/sigma_b^2 is the covariance plus sigma_b^2 r r^dagger, r = dd/d delta_b -
    Sherman-Morrison, bin by bin"""
    F = FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2)
    ln, params = [0, 2], ["NPP", "GR2"]
    fish = F.get_fish(params, terms="NPP", pkln=ln, per_bin_params=["delta_b"], pinv_rtol=None, verbose=False)
    ref = 0
    for i in range(F.N_bins):
        fc = F.get_pk_bin(i)
        C = fc.get_cov_mat(ln, n_mu=F.n_mu)
        n, _, N = C.shape
        dense = np.zeros((n * N, n * N), dtype=np.complex128)
        for t in range(N):
            dense[t * n : (t + 1) * n, t * n : (t + 1) * n] = C[:, :, t]
        r = fc.get_data_vector("NPP", ln, param="delta_b").T.ravel()
        dense += fc.sigma_b2() * np.outer(r, r.conj())
        d = [fc.get_data_vector("NPP", ln, param=p).T.ravel() for p in params]
        ref = ref + np.array([[np.real(a.conj() @ np.linalg.solve(dense, b)) for b in d] for a in d])
    np.testing.assert_allclose(fish.fisher_matrix, ref, rtol=1e-8)


def test_bk_has_no_response(bk_bin):
    """B's super-sample response is left out - its SSC is an order of magnitude below P's"""
    assert not np.any(bk_bin.get_data_vector("NPP", [0, 2], param="delta_b"))
