"""Tests for the luminosity derivatives at the cut and how lib.betas uses them (2011.13660 A.8, A.12, A.16)."""

import numpy as np
import pytest

import cosmo_wap as cw
from cosmo_wap.lib import betas, utils
from cosmo_wap.lib.luminosity_funcs import BGSLuminosityFunction, Model3LuminosityFunction, lum_derivs

zero = lambda xx: 0 * xx  # noqa: E731


def _ln_L_c(LF, cut, zz):
    if hasattr(LF, "L_c"):
        return np.log(LF.L_c(cut, zz))
    return -0.4 * np.log(10) * LF.M_UV(cut, zz)


def _at_fixed_L(LF, cut, z0, zp):
    """The cut at redshift zp that selects the limiting luminosity `cut` selects at z0."""
    return LF.shift_cut(cut, _ln_L_c(LF, cut, np.array([z0]))[0] - _ln_L_c(LF, cut, np.array([zp]))[0])


@pytest.fixture(scope="module", params=["model3", "bgs"])
def lf_case(request, cosmo):
    """Flux-limited without K-correction, and magnitude-limited with one."""
    if request.param == "model3":
        return Model3LuminosityFunction(cosmo), 2e-16, np.linspace(0.9, 1.8, 100)
    return BGSLuminosityFunction(cosmo), 20.175, np.linspace(0.05, 0.5, 100)


class TestLumDerivs:
    def test_be_matches_definition(self, lf_case):
        """get_be is -dln n/dln(1+z) at fixed luminosity (2107.13401 2.9, 3.4) - K term counted once."""
        LF, cut, zz = lf_case
        be = LF.get_be(cut, zz)

        d = 1e-3
        for i in (25, 50, 75):
            z0 = zz[i]
            ln_n = [
                np.log(LF.number_density(_at_fixed_L(LF, cut, z0, zp), np.array([zp]))[0]) for zp in (z0 + d, z0 - d)
            ]
            assert be[i] == pytest.approx(-(ln_n[0] - ln_n[1]) / np.log((1 + z0 + d) / (1 + z0 - d)), rel=2e-3)

    def test_dQ_matches_lf_slope(self, cosmo):
        """Flux-limited: dQ/dlnL = Q (1 + Q + dln g/dln y) at the cut."""
        LF, F_c, zz = Model3LuminosityFunction(cosmo), 2e-16, np.linspace(0.9, 1.8, 20)
        dQ, _, _ = lum_derivs(LF, F_c, zz)

        Q = LF.get_Q(F_c, zz)
        y = LF.get_y(LF.L_c(F_c, zz), zz)
        eps = 1e-5
        dlng = (np.log(LF.g(y * np.exp(eps))) - np.log(LF.g(y * np.exp(-eps)))) / (2 * eps)
        np.testing.assert_allclose(dQ, Q * (1 + Q + dlng), rtol=1e-3)

    def test_fixed_L_dQ_dz(self, lf_case):
        """dQ/dz - dQ/dlnL dlnL_c/dz is dQ/dz at fixed luminosity."""
        LF, cut, zz = lf_case
        dQ, _, dlnLc = lum_derivs(LF, cut, zz)
        chain = betas.dy_dz(LF.get_Q(cut, zz), zz) - dQ * dlnLc

        dz = 1e-3
        for i in (25, 50, 75):
            z0 = zz[i]
            Qp, Qm = (LF.get_Q(_at_fixed_L(LF, cut, z0, zp), np.array([zp]))[0] for zp in (z0 + dz, z0 - dz))
            assert chain[i] == pytest.approx((Qp - Qm) / (2 * dz), rel=5e-3)

    def test_fixed_L_dbe_dz(self, cosmo):
        """d b_e/d ln L = dQ/d ln(1+z) at fixed L - the b_e correction lib.betas makes."""
        LF, F_c, zz = Model3LuminosityFunction(cosmo), 2e-16, np.linspace(0.9, 1.8, 100)
        dQ, _, dlnLc = lum_derivs(LF, F_c, zz)
        dQ_dz = betas.dy_dz(LF.get_Q(F_c, zz), zz) - dQ * dlnLc
        chain = betas.dy_dz(LF.get_be(F_c, zz), zz) - (1 + zz) * dQ_dz * dlnLc

        def be_at(z, z0, d=1e-3):
            """-dln n/dln(1+z) at z, for the luminosity F_c selects at z0"""
            ln_n = [np.log(LF.number_density(_at_fixed_L(LF, F_c, z0, zp), np.array([zp]))[0]) for zp in (z + d, z - d)]
            return -(ln_n[0] - ln_n[1]) / np.log((1 + z + d) / (1 + z - d))

        D = 5e-3
        for i in (25, 50, 75):
            z0 = zz[i]
            assert chain[i] == pytest.approx((be_at(z0 + D, z0) - be_at(z0 - D, z0)) / (2 * D), rel=5e-3)

    def test_faint_combination(self, cosmo):
        """get_faint_lum_deriv matches shifting both cuts of the faint sample together."""
        LF, zz, cut, split, h = Model3LuminosityFunction(cosmo), np.linspace(0.9, 1.8, 20), 2e-16, 6e-16, 0.01

        def faint(s):
            """faint Q and b_1 with both cuts shifted by s in ln L"""
            (n_T, Q_T), (n_B, Q_B) = (LF.get_nQ(LF.shift_cut(c, s), zz) for c in (cut, split))
            b_T, b_B = (LF.get_b_1(LF.shift_cut(c, s), zz) for c in (cut, split))
            return utils.get_faint_bias(zz, n_T, n_B, Q_T, Q_B)(zz), utils.get_faint_bias(zz, n_T, n_B, b_T, b_B)(zz)

        (Qp, bp), (Qm, bm) = faint(h), faint(-h)

        (n_T, Q_T), (n_B, Q_B) = LF.get_nQ(cut, zz), LF.get_nQ(split, zz)
        b_T, b_B = LF.get_b_1(cut, zz), LF.get_b_1(split, zz)
        (dQ_T, db_T, _), (dQ_B, db_B, _) = lum_derivs(LF, cut, zz), lum_derivs(LF, split, zz)
        dQ_F = utils.get_faint_lum_deriv(zz, n_T, n_B, Q_T, Q_B, Q_T, Q_B, dQ_T, dQ_B)(zz)
        db_F = utils.get_faint_lum_deriv(zz, n_T, n_B, Q_T, Q_B, b_T, b_B, db_T, db_B)(zz)

        np.testing.assert_allclose(dQ_F, (Qp - Qm) / (2 * h), rtol=1e-3)
        np.testing.assert_allclose(db_F, (bp - bm) / (2 * h), rtol=1e-3)


class TestTracerDefaults:
    def test_no_luminosity_function_gives_zero(self, cosmo):
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo, fitting=True), verbose=False)
        tr = cf.survey[0]
        for name in ("dQ_dlnL", "db1_dlnL", "dlnLc_dz"):
            assert np.all(getattr(tr, name)(tr.z_survey) == 0)

    def test_compute_bias_drops_db1(self, cosmo):
        """db_1/dlnL belongs to the luminosity function's b_1, which the HMF/HOD one replaces."""
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), compute_bias=True, verbose=False)
        tr = cf.survey[0]
        assert np.all(tr.db1_dlnL(tr.z_survey) == 0)
        assert np.all(tr.dQ_dlnL(tr.z_survey) > 0)


class TestBetas:
    def test_luminosity_terms_enter_as_in_paper(self, cosmo):
        """With dlnLc_dz off, only beta8/12/16 move, by the dQ/dlnL and db1/dlnL terms of A.8, A.12, A.16."""
        cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo), verbose=False)
        tr = cf.survey[0]
        zz = tr.z_survey
        dQ, db1 = tr.dQ_dlnL(zz), tr.db1_dlnL(zz)
        assert np.all(dQ > 0) and np.all(db1 > 0)

        def beta(dQ_dlnL, db1_dlnL, dlnLc_dz):
            tr.dQ_dlnL, tr.db1_dlnL, tr.dlnLc_dz = dQ_dlnL, db1_dlnL, dlnLc_dz
            return np.asarray(betas.interpolate_beta_funcs(cf, 0)(zz))

        off = beta(zero, zero, zero)
        part = beta(utils.CachedSpline(zz, dQ), utils.CachedSpline(zz, db1), zero)

        _, H_c, f, Om, xi, _, _ = cf.beta_cosmo[0]
        inv = 1 / (xi * H_c)
        expect = np.zeros_like(off)
        expect[4] = H_c**2 * 4 * f**2 * (1 - inv) ** 2 * dQ  # beta8
        expect[8] = H_c**2 * 3 * Om * (2 - inv) * db1  # beta12
        expect[12] = H_c * 2 * f * (1 - inv) * db1  # beta16
        for i in range(len(off)):
            np.testing.assert_allclose(part[i] - off[i], expect[i], rtol=1e-8, atol=1e-10 * np.abs(off[i]).max())
