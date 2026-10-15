"""Fixed-fraction bright/faint split - SurveyBase.BF_split(fraction=...).

The bright sample is the brightest fraction f of the survey at every redshift, so its cut moves with z,
drifting against a fixed flux by D = d ln(L_B/L_T)/d ln(1+z). Checked against the defining identities,
the Doppler contrast of the note (s_Q_selection_note), and a brute force holding both luminosity cuts fixed.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

import cosmo_wap as cw
from cosmo_wap.lib.betas import interpolate_beta_funcs, survey_terms

FRACTION = 0.3


@pytest.fixture(scope="module")
def split_params(cosmo):
    return cw.SurveyParams.Euclid(cosmo).BF_split(fraction=FRACTION)


@pytest.fixture(scope="module")
def cf_split(cosmo, split_params):
    return cw.ClassWAP(cosmo, split_params, verbose=False)


@pytest.fixture(scope="module")
def cf_total(cosmo):
    return cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), verbose=False)


@pytest.fixture(scope="module")
def forecast_ff(cf_split):
    return cw.forecast.FullForecast(cf_split, kmax_func=0.1, N_bins=2)


def _interior(tr):
    return tr.z_survey[3:-3]


class TestConstruction:
    def test_bright_fraction_is_fixed(self, cf_split, cf_total):
        B, T = cf_split.survey[0], cf_total.survey[0]
        zz = _interior(B)
        np.testing.assert_allclose(B.n_g(zz) / T.n_g(zz), FRACTION, rtol=1e-10)

    def test_n_evol_is_shared(self, cf_split, cf_total):
        """n_B/n_T fixed, so the count slope is the same for total, bright and faint."""
        B, F, T = cf_split.survey[0], cf_split.survey[1], cf_total.survey[0]
        zz = _interior(B)
        np.testing.assert_allclose(B.n_evol(zz), T.n_evol(zz), atol=1e-5)
        np.testing.assert_allclose(F.n_evol(zz), T.n_evol(zz), atol=1e-5)

    def test_b_e_is_number_weighted(self, cf_split, cf_total):
        """n_T b_eT = n_B b_eB + n_F b_eF - b_e is minus the fixed-L slope of the counts."""
        B, F, T = cf_split.survey[0], cf_split.survey[1], cf_total.survey[0]
        zz = _interior(B)
        total = T.n_g(zz) * T.be(zz)
        np.testing.assert_allclose(B.n_g(zz) * B.be(zz) + F.n_g(zz) * F.be(zz), total, rtol=1e-8)

    def test_doppler_contrast(self, cf_split, cf_total):
        """gamma_B - gamma_F = -[(4 + K')(Q_B - Q_T) + Q_B D]/(1 - f), the note's Section 5 with m = 1/f."""
        B, T = cf_split.survey[0], cf_total.survey[0]
        zz = _interior(B)

        def gamma(cf, ti):
            return interpolate_beta_funcs(cf, ti)[0](zz) / (cf.H_c(zz) * cf.f(zz))

        Kp = T.c_Q(zz) - 2 * (1 + 1 / (cf_split.comoving_dist(zz) * cf_split.H_c(zz)))
        expected = -((4 + Kp) * (B.Q(zz) - T.Q(zz)) + B.Q(zz) * B.dlnF_cut(zz)) / (1 - FRACTION)
        np.testing.assert_allclose(gamma(cf_split, 0) - gamma(cf_split, 1), expected, atol=1e-5)

    def test_bright_cut_moves(self, cf_split):
        """A fixed fraction is not a fixed flux - D is what separates the two."""
        B, F = cf_split.survey[0], cf_split.survey[1]
        zz = _interior(B)
        assert np.all(B.dlnF_cut(zz) < 0)  # the bright cut gets fainter in flux more slowly than ~d_L^-2
        np.testing.assert_allclose(F.w_split(zz), FRACTION / (1 - FRACTION) * B.dlnF_cut(zz))

    def test_one_of_split_or_fraction(self, cosmo):
        sp = cw.SurveyParams.Euclid(cosmo)
        with pytest.raises(ValueError):
            sp.BF_split()
        with pytest.raises(ValueError):
            sp.BF_split(3e-16, fraction=0.5)
        with pytest.raises(ValueError):
            sp.BF_split(fraction=1.2)


class TestMegaMapper:
    """The LBG luminosity function exists only at its five fitted redshifts - the bright cut must be solved at
    each, not at z = 2.0, which gave b_B too high and b_F < 0 at large fractions."""

    @pytest.fixture(scope="class")
    def survey(self):
        from cosmo_wap.lib import utils

        return cw.SurveyParams.MegaMapper(utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=6.0))

    def test_bright_Q_at_its_own_redshift(self, survey):
        """Q_B is the LF's Q at the cut that keeps n_B = f n_T at that same redshift - solved on the whole grid,
        taking redshift i's column, so this does not rely on single-redshift calls as _get_fraction does."""
        LF, zz = survey.LF, survey.zz
        n_T = LF.number_density(survey.cut, zz)
        B, _ = survey.BF_split(fraction=0.2)
        expected = []
        for i, n in enumerate(n_T):
            n_B = lambda d: LF.number_density(LF.shift_cut(survey.cut, d), zz)[i] - 0.2 * n  # noqa: E731
            expected.append(LF.get_Q(LF.shift_cut(survey.cut, brentq(n_B, 0, 5)), zz)[i])
        np.testing.assert_allclose(B.Q(zz), expected, rtol=1e-6)

    @pytest.mark.parametrize("fraction", [0.2, 0.7])
    def test_bias_ordering(self, survey, fraction):
        """b_B > b_T > b_F > 0 - the brightest galaxies are the most biased."""
        B, F = survey.BF_split(fraction=fraction)
        zz = survey.zz
        b_B, b_T, b_F = B.b_1(zz), survey.b_1(zz), F.b_1(zz)
        assert np.all(b_B > b_T) and np.all(b_T > b_F) and np.all(b_F > 0)
        assert np.all(F.Q(zz) > 0)


class TestFixedLDerivatives:
    """The faint sits between two cuts moving at different rates - its fixed-L z-derivatives
    (lib.betas.survey_terms) against finite differences holding both luminosity cuts fixed."""

    @pytest.fixture(scope="class")
    def brute(self, split_params, cf_split):
        LF, F_T = split_params[0].LF, split_params[0].cut
        za = lambda z: np.array([z])  # noqa: E731

        def F_B(z):  # the bright flux cut, n_B = f n_T, as _get_fraction
            n_T = LF.number_density(F_T, za(z))[0]
            dlnL = brentq(lambda d: LF.number_density(F_T * np.exp(d), za(z))[0] - FRACTION * n_T, 0, 5)
            return F_T * np.exp(dlnL)

        def faint(z, L_T, L_B):  # n_F, Q_F, b_1F at z with the cuts held at L_T, L_B (L_c is linear in flux)
            g = LF.L_c(1.0, za(z))[0]
            n_T, n_B = LF.number_density(L_T / g, za(z))[0], LF.number_density(L_B / g, za(z))[0]
            Q_T, Q_B = LF.get_Q(L_T / g, za(z))[0], LF.get_Q(L_B / g, za(z))[0]
            b_T, b_B = LF.get_b_1(L_T / g, za(z))[0], LF.get_b_1(L_B / g, za(z))[0]
            n_F = n_T - n_B
            return n_F, (n_T * Q_T - n_B * Q_B) / n_F, (n_T * b_T - n_B * b_B) / n_F

        zz = cf_split.survey[1].z_survey
        out = {}
        for i in np.searchsorted(zz, [1.1, 1.35, 1.6]):
            z0 = zz[i]
            g0 = LF.L_c(1.0, za(z0))[0]
            eps = 2e-3
            vals = np.array([faint(z0 + eps * j, F_T * g0, F_B(z0) * g0) for j in (-1, 0, 1)])
            lnn = np.log(vals[:, 0])
            dlnn, d2lnn = (lnn[2] - lnn[0]) / (2 * eps), (lnn[2] - 2 * lnn[1] + lnn[0]) / eps**2
            out[i] = {
                "be": -(1 + z0) * dlnn,
                "Q": (vals[2, 1] - vals[0, 1]) / (2 * eps),
                "b_1": (vals[2, 2] - vals[0, 2]) / (2 * eps),
                "dbe": -dlnn - (1 + z0) * d2lnn,  # b_e = -(1+z) dln n/dz
            }
        return out

    def test_faint_b_e(self, cf_split, brute):
        F = cf_split.survey[1]
        for i, b in brute.items():
            assert F.be(F.z_survey[i]) == pytest.approx(b["be"], rel=1e-4)

    def test_fixed_L_derivatives(self, cf_split, brute):
        zz = cf_split.survey[1].z_survey
        _, _, _, _, _, _, _, dQ_dt, dbe_dt, db1_dt, cosmo_t = survey_terms(cf_split, 1)
        to_dz = -1 / ((1 + zz) * cosmo_t[0])  # d/dt = -(1+z) H d/dz
        corr = {k: v(zz) for k, v in cf_split.survey[1].split_corr.items()}
        for i, b in brute.items():
            for name, d_dt in [("Q", dQ_dt), ("b_1", db1_dt), ("dbe", dbe_dt)]:
                assert to_dz[i] * d_dt[i] == pytest.approx(b[name], rel=1e-3), name
            # and the bright-cut term is what makes them agree - several times that tolerance
            assert abs(corr["Q"][i]) > 3e-3 * abs(b["Q"])
            assert abs(corr["be"][i]) > 3e-3 * abs(b["dbe"])


class TestSecondCutTerms:
    """The second-cut terms beyond Q, b_e and b_1: b_01 for the full_betas Upsilons, the b_e term at another
    cosmology, and none for b_1 once compute_bias replaces it."""

    def test_b01_term_in_upsilon(self, cf_split):
        """Upsilon3 carries f d b_01/d ln a / 2 - its second-cut term is 2 delta_c that of b_1 (b_01 = 2 delta_c (b_1 - p))."""
        from cosmo_wap.lib.betas import interpolate_upsilon

        F = cf_split.survey[1]
        zz = F.z_survey
        corr = F.loc.split_corr_b01(zz)
        np.testing.assert_allclose(corr, 2 * 1.686 * F.split_corr["b_1"](zz))
        with_corr = interpolate_upsilon(cf_split, 1)[0](zz)
        saved = F.loc.split_corr_b01
        del F.loc.split_corr_b01
        try:
            without = interpolate_upsilon(cf_split, 1)[0](zz)
        finally:
            F.loc.split_corr_b01 = saved
        expected = cf_split.H_c(zz) ** 2 * cf_split.f(zz) * (1 + zz) * corr / 2
        np.testing.assert_allclose(with_corr - without, expected, rtol=1e-8, atol=1e-12 * np.abs(expected).max())
        assert not hasattr(cf_split.survey[0].loc, "split_corr_b01")  # the bright has one cut

    def test_b_e_term_follows_cosmology(self, cf_split):
        """Rebuilt on an adopted copy from that cosmology's c_T - against the defining formula with its b_e."""
        from cosmo_wap.lib.utils import get_cosmo

        cf_h = cw.ClassWAP(get_cosmo(Omega_m=0.35), fast=True, verbose=False).adopt_survey(cf_split)
        B, F = cf_h.survey[0], cf_h.survey[1]
        zz = F.z_survey
        D, Q_B, Q_F, c_T = B.dlnF_cut(zz), B.Q(zz), F.Q(zz), F.c_Q(zz)
        dbe_B = (1 + zz) * np.gradient(Q_B, zz) - B.dQ_dlnL(zz) * (c_T + D)
        k = D / (1 + zz) * FRACTION / (1 - FRACTION)
        expected = k * (Q_B * (B.be(zz) - F.be(zz)) - dbe_B)
        got = F.split_corr["be"](zz)
        np.testing.assert_allclose(got, expected, rtol=1e-4)
        moved = np.abs(got - cf_split.survey[1].split_corr["be"](zz)).max()
        assert moved > 100 * 1e-4 * np.abs(expected).max()  # and the cosmology change is resolved by that tolerance

    def test_compute_bias_drops_the_b1_term(self, cosmo):
        """An HMF/HOD b_1 has no luminosity derivatives, so no second-cut term either - as the HOD split."""
        cf = cw.ClassWAP(
            cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(fraction=FRACTION), compute_bias=True, verbose=False
        )
        F = cf.survey[1]
        assert "b_1" not in F.split_corr and {"Q", "be"} <= set(F.split_corr)
        assert not hasattr(F.loc, "split_corr_b01")


class TestCrossTracer:
    """XQ moves the bright Q at fixed n_evol - and, through Q_split, the faint b_e by w_split of it."""

    TERMS = ["NPP", "GR2"]

    def derivs(self, forecast, params):
        pk_bin = forecast.get_pk_bin(0, all_tracer=True)
        forecast.set_selection_basis(params)
        try:
            # monopole only: [XX, XY, YY]
            return {p: pk_bin.get_data_vector(self.TERMS, [0], param=p) for p in params}
        finally:
            forecast.set_selection_basis([])

    def test_XQ_reaches_the_faint_auto(self, forecast_ff):
        d = self.derivs(forecast_ff, ["XQ", "Yn_evol"])
        w = forecast_ff.cosmo_funcs.survey[1].w_split(forecast_ff.get_pk_bin(0).z_mid)
        YY = 2
        assert np.abs(d["XQ"][YY]).max() > 0
        np.testing.assert_allclose(
            d["XQ"][YY], -w * d["Yn_evol"][YY], rtol=1e-6, atol=1e-10 * np.abs(d["XQ"][YY]).max()
        )

    def test_fixed_flux_has_no_cross_term(self, forecast_mt):
        d = self.derivs(forecast_mt, ["XQ"])
        assert np.abs(d["XQ"][2]).max() < 1e-10 * np.abs(d["XQ"][0]).max()  # round-off, not a response

    def test_fisher_runs(self, forecast_ff):
        fish = forecast_ff.get_fish(
            ["fNL"],
            terms=["NPP", "GR2", "Loc"],
            pkln=[0, 2],
            all_tracer=True,
            per_bin_params=["n_evol", "XQ", "YQ"],
            fNL=5.0,
            verbose=False,
        )
        forecast_ff.set_selection_basis([])
        assert np.isfinite(fish.errors[0]) and fish.errors[0] > 0


class TestHOD:
    def test_bgs_hod_fraction(self, cosmo):
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.BGS(cosmo).BF_split(fraction=0.5), verbose=False)
        total = cw.ClassWAP(cosmo, cw.SurveyParams.BGS(cosmo), verbose=False).survey[0]
        B, F = cf.survey[0], cf.survey[1]
        zz = _interior(B)
        np.testing.assert_allclose(B.n_g(zz) / total.n_g(zz), 0.5, rtol=1e-6)
        np.testing.assert_allclose(B.n_evol(zz), total.n_evol(zz), atol=1e-3)
        assert B.dlnF_cut is not None and F.w_split is not None and "be" in F.split_corr


class TestHODCaches:
    def test_bright_and_faint_do_not_share_derivs(self, cosmo, monkeypatch):
        """The HOD split copies one tracer twice - each copy needs its own deriv cache, else the faint reads the
        bright's radial derivatives. set_selection also resets it, so take that out to test the split itself."""
        from cosmo_wap.survey_params import SetSurveyFunctions

        monkeypatch.setattr(SetSurveyFunctions, "set_selection", lambda self, cf: None)
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.BGS(cosmo).BF_split(19.5), verbose=False)
        assert cf.survey[0].deriv is not cf.survey[1].deriv


class TestSampler:
    @pytest.fixture(scope="class")
    def sampler_ff(self, forecast_ff):
        pytest.importorskip("cobaya")
        from cosmo_wap.forecast.sampler import Sampler

        sampler = Sampler(
            forecast_ff,
            ["fNL"],
            terms=["NPP", "GR2"],
            pkln=[0, 2],
            per_bin_params=["n_evol", "XQ", "YQ"],
            fisher_covmat=False,
            drag=False,
        )
        yield sampler
        forecast_ff.set_selection_basis([])

    def test_XQ_moves_the_faint_b_e(self, sampler_ff, forecast_ff):
        B, F = forecast_ff.cosmo_funcs.survey[0], forecast_ff.cosmo_funcs.survey[1]
        z = 1.3
        QB0, beF0, QF0 = B.Q(z), F.be(z), F.Q(z)
        vals = [1.1 if p == "XQ_0" else sampler_ff.fiducial[p] for p in sampler_ff.param_list]
        with sampler_ff._per_bin_bias(forecast_ff.cosmo_funcs, vals, 0):
            assert B.Q(z) == pytest.approx(1.1 * QB0)
            assert F.Q(z) == pytest.approx(QF0)
            assert F.be(z) == pytest.approx(beF0 + F.w_split(z) * 0.1 * QB0)
        assert (B.Q(z), F.be(z)) == (QB0, beF0)
