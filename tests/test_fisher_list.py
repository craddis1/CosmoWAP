"""Tests for cosmo_wap.forecast.fisher_list — get_fish_list grids and FisherList access/plotting."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast

PARAMS = ["A_b_1"]
FISH_KW = dict(terms="NPP", pkln=[0], bkln=None)
CUTS = [2e-16, 3e-16]
SPLITS = [2.5e-16, 4e-16, 6e-16]


def bf_split(cosmo, cut, split):
    return cw.SurveyParams.Euclid(cosmo, cut=cut).BF_split(split) if split > cut else None


@pytest.fixture(scope="module")
def kmax_list(forecast):
    return forecast.get_fish_list(PARAMS, grid={"kmax_func": [0.08, 0.1]}, verbose=False, **FISH_KW)


@pytest.fixture(scope="module")
def cut_split_list(forecast):
    return forecast.get_fish_list(
        PARAMS, grid={"cut": CUTS, "split": SPLITS}, survey_func=bf_split, verbose=False, **FISH_KW
    )


class TestGrid:
    def test_kmax_axis_matches_get_fish(self, forecast, kmax_list):
        """A FullForecast argument as an axis - no survey_func needed."""
        assert kmax_list.fish.shape == (2,)
        ff = FullForecast(forecast.cosmo_funcs, kmax_func=0.08, s_k=forecast.s_k, N_bins=forecast.N_bins)
        ref = ff.get_fish(PARAMS, verbose=False, **FISH_KW)
        np.testing.assert_allclose(kmax_list[0].fisher_matrix, ref.fisher_matrix, rtol=1e-12)
        assert kmax_list.get_error("A_b_1")[0] > kmax_list.get_error("A_b_1")[1]  # more modes, smaller error

    def test_fish_axis(self, forecast):
        fl = forecast.get_fish_list(PARAMS, grid={"pkln": [[0], [0, 2]]}, terms="NPP", bkln=None, verbose=False)
        err = fl.get_error("A_b_1")
        assert err[1] < err[0]

    def test_survey_axes_match_direct_build(self, cosmo, forecast, cut_split_list):
        survey = cw.SurveyParams.Euclid(cosmo, cut=3e-16).BF_split(6e-16)
        cf = cw.ClassWAP(cosmo, survey, verbose=False)
        ff = FullForecast(cf, kmax_func=forecast.kmax_func, s_k=forecast.s_k, N_bins=forecast.N_bins)
        ref = ff.get_fish(PARAMS, verbose=False, **FISH_KW)
        np.testing.assert_allclose(
            cut_split_list.at(cut=3e-16, split=6e-16).fisher_matrix, ref.fisher_matrix, rtol=1e-10
        )

    def test_skipped_points(self, cut_split_list):
        assert cut_split_list.fish.shape == (2, 3)
        assert cut_split_list[1, 0] is None  # split < cut
        err = cut_split_list.get_error("A_b_1")
        assert np.isnan(err[1, 0])
        assert np.all(np.isfinite(err[~np.isnan(err)]))

    def test_forecast_settings_carried(self, forecast):
        ff = FullForecast(forecast.cosmo_funcs, kmax_func=0.1, s_k=2, N_bins=2, WS_cut=False, n_mu=12)
        fl = ff.get_fish_list(PARAMS, grid={"kmax_func": [0.1]}, verbose=False, **FISH_KW)
        assert fl[0].forecast.WS_cut is False
        assert fl[0].forecast.n_mu == 12

    def test_unknown_axis_needs_survey_func(self, forecast):
        with pytest.raises(ValueError, match="survey_func"):
            forecast.get_fish_list(PARAMS, grid={"cut": CUTS}, verbose=False, **FISH_KW)


class TestAccess:
    def test_at_partial(self, cut_split_list):
        row = cut_split_list.at(cut=2e-16)
        assert row.shape == (3,)

    def test_at_missing_value(self, cut_split_list):
        with pytest.raises(ValueError):
            cut_split_list.at(cut=5e-16, split=6e-16)

    def test_best(self, cut_split_list):
        err = cut_split_list.get_error("A_b_1")
        best = cut_split_list.best("A_b_1")
        assert cut_split_list.at(**best).get_error("A_b_1") == np.nanmin(err)

    def test_map(self, cut_split_list):
        fl = cut_split_list.map(lambda f: f.add_gaussian_priors({"A_b_1": 0.01}))
        assert fl[1, 0] is None
        assert np.nanmax(fl.get_error("A_b_1") - cut_split_list.get_error("A_b_1")) < 0


class TestPlot:
    def test_1d(self, kmax_list):
        fig, ax = kmax_list.plot()
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_2d_smooth_and_raw(self, cut_split_list, tmp_path):
        for smooth in (True, False):
            fig, ax = cut_split_list.plot("A_b_1", smooth=smooth, save=tmp_path / f"plot_{smooth}.png")
            assert len(ax.images) == 1
            assert (tmp_path / f"plot_{smooth}.png").exists()
            plt.close(fig)

    def test_fixed_slice(self, cut_split_list):
        fig, ax = cut_split_list.plot(fixed={"cut": 2e-16})
        assert len(ax.lines) == 1
        plt.close(fig)
