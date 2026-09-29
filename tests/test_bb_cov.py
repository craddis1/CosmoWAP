"""BB term (and its collapsed PT partner) of the bispectrum covariance - covariances.BBCovBk and WoodburyInvCov.

Checked against a brute-force sum over explicit 3D triangle orientations (B in its original labelling),
a lattice count of triangle pairs sharing a Fourier mode, and the dense inverse.
"""

import pickle

import numpy as np
import pytest
from scipy.special import eval_legendre, sph_harm_y

import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import contract, joint_inv_cov
from cosmo_wap.forecast.covariances import BBCovBk, PBCov, PTTreeCovBk, WoodburyInvCov
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import pk as npk

LN = [0, 1, 2, 3]
LN_M = [0, 1, 2]  # all_m: 6 rows per tracer
TERMS = ["N", "LP"]


def dense_bb(bb):
    """U Lambda U^dagger as a dense (N_tri*n_rows)^2 matrix, triangle-major"""
    N, n, _, b = bb.U.shape
    U = np.zeros((N, n, bb.n_shell, b), dtype=np.complex128)
    for w in range(3):
        U[np.arange(N), :, bb.shell[w], :] += bb.U[:, :, w]
    U = U.reshape(N * n, -1)
    return U @ np.kron(np.eye(bb.n_shell), bb.lam) @ U.conj().T


def dense_block_diag(D):
    n, _, N = D.shape
    out = np.zeros((N * n, N * n), dtype=np.complex128)
    for t in range(N):
        out[t * n : (t + 1) * n, t * n : (t + 1) * n] = D[:, :, t]
    return out


def bb_element(bb, t1, r1, t2, r2):
    """C_BB between row r1 of triangle t1 and row r2 of triangle t2, from the factors"""
    tot = 0
    for w1 in range(3):
        for w2 in range(3):
            if bb.shell[w1, t1] == bb.shell[w2, t2]:
                tot += bb.U[t1, r1, w1] @ bb.lam @ bb.U[t2, r2, w2].conj()
    return tot


@pytest.fixture(scope="module")
def small_st_forecast(cosmo_funcs):
    return FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2)


@pytest.fixture(scope="module")
def small_mt_forecast(cosmo):
    cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(6e-16), verbose=False)
    return FullForecast(cf, kmax_func=0.035, s_k=2, N_bins=2)


@pytest.fixture(scope="module")
def small_mt_m(small_mt_forecast):
    """all_m - every m of each l, as FullForecast(all_m=True)"""
    F = FullForecast(small_mt_forecast.cosmo_funcs, kmax_func=0.035, s_k=2, N_bins=2, all_m=True)
    return F.get_bk_bin(0, cov_terms=TERMS, all_tracer=True)


@pytest.fixture(scope="module")
def small_st_m(small_st_forecast):
    F = FullForecast(small_st_forecast.cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, all_m=True)
    return F.get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def small_st(small_st_forecast):
    return small_st_forecast.get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def small_mt(small_mt_forecast):
    return small_mt_forecast.get_bk_bin(0, cov_terms=TERMS, all_tracer=True)


@pytest.fixture(scope="module")
def small_mt8(small_mt_forecast):
    """all_tracer=8 - every placement, see bk_placements"""
    return small_mt_forecast.get_bk_bin(0, cov_terms=TERMS, all_tracer=8)


@pytest.fixture(scope="module")
def small_mt8_pk(small_mt_forecast):
    return small_mt_forecast.get_pk_bin(0, cov_terms=TERMS, all_tracer=8)


@pytest.fixture(scope="module")
def small_st_pk(small_st_forecast):
    return small_st_forecast.get_pk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def small_mt_pk(small_mt_forecast):
    return small_mt_forecast.get_pk_bin(0, cov_terms=TERMS, all_tracer=True)


class TestGeometry:
    """Each triangle built as vectors - shared side +-k_c e, rotated about it - and B evaluated at its own
    (mu_1, phi), with shot noise written out in the original labelling. Independent of the relabelling,
    the -k on the second triangle and the tracer swaps in BBCovBk."""

    @staticmethod
    def B_N(fc, tracers, kk, mu1, cphi, c, swap=None):
        """B without shot noise, or with tracer swap on side c: B^(N) = (Z1^swap/Z1) B + the shot noise of swap's
        galaxy merged with each other side"""
        k1, k2, k3 = kk
        cf = fc.cf_mat_bk[tracers[0]][tracers[1]][tracers[2]]
        zz = fc.z_mid
        theta = utils.get_theta(k1, k2, k3)
        phi = np.arccos(np.clip(cphi, -1, 1))
        B = np.diagonal(nbk.get_mu_phi(mu1, phi, TERMS, TERMS, TERMS, cf, k1, k2, k3, theta, zz))
        if swap is None:
            return B
        mus = nbk.los_cosines(mu1, phi, k1, k2, k3, theta)
        tr = list(tracers)
        tr[c] = swap
        cf2 = fc.cf_mat_bk[tr[0]][tr[1]][tr[2]]
        B = B * nbk.get_Z1(TERMS, cf2, zz, mus[c], kk[c], ti=c) / nbk.get_Z1(TERMS, cf, zz, mus[c], kk[c], ti=c)
        for i, j, m in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
            if c in (i, j) and tr[i] == tr[j]:
                Z = nbk.get_Z1(TERMS, cf2, zz, -mus[m], kk[m], ti=i) * nbk.get_Z1(TERMS, cf2, zz, mus[m], kk[m], ti=m)
                B = B + Z * BaseInt(cf2).pk(kk[m], zz) / cf2.survey[i].n_g(zz)
        return B

    def g(self, fc, kk, c, sign, tracers, l, mu_e, swap=None, n_psi=32):
        """<sqrt(4pi(2l+1)) L_l(mu_1) B> over rotations about side c, whose vector is sign*k_c*e, e.z = mu_e -
        B as B_N. l can be (l, m): then 4pi Re Y_lm(mu_1, phi), phi the LOS azimuth about k1 from k2"""
        l, m = l if isinstance(l, tuple) else (l, 0)
        a, b = [i for i in range(3) if i != c]
        e = np.array([np.sqrt(1 - mu_e**2), 0, mu_e])
        e1 = np.array([mu_e, 0, -np.sqrt(1 - mu_e**2)])
        e2 = np.cross(e, e1)
        cos_a = (kk[b] ** 2 - kk[a] ** 2 - kk[c] ** 2) / (2 * kk[a] * kk[c])
        psi = 2 * np.pi * (np.arange(n_psi) + 0.5) / n_psi
        rot = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
        v = [None] * 3
        v[c] = np.broadcast_to(sign * kk[c] * e, rot.shape)
        v[a] = kk[a] * (cos_a * sign * e + np.sqrt(1 - cos_a**2) * rot)
        v[b] = -(v[a] + v[c])
        mu = [vi[:, 2] / kk[i] for i, vi in enumerate(v)]
        cos12 = np.sum(v[0] * v[1], axis=1) / (kk[0] * kk[1])
        cphi = (mu[1] - mu[0] * cos12) / np.sqrt((1 - mu[0] ** 2) * (1 - cos12**2))
        B = self.B_N(fc, tracers, kk, mu[0], cphi, c, swap)
        if m:
            w = 4 * np.pi * np.real(sph_harm_y(l, m, np.arccos(mu[0]), np.arccos(np.clip(cphi, -1, 1))))
        else:
            w = np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu[0])
        return np.mean(w * B)

    def brute(self, fc, bb, t1, ci1, l1, t2, ci2, l2, swap, sign, shot=False, n_mu=16):
        """sum over shared pairs of (1/N_k) int dOmega_k/(4pi) <(4pi)^2 Y_l1 Y_l2 B_T1 conj(B_T2)>, T2 on sign*k.
        B^(N), with swap the other's tracer on each shared side (BB, -k), else its own (PT, +k), and shot the shot
        noise of PT's pairing P - x (1 + 1/(n P_gal)) where the shared sides have the same tracer"""
        ks = np.array(fc.args[1:4])
        mu_e, w = np.polynomial.legendre.leggauss(n_mu)
        dk = fc.forecast.s_k * fc.k_f
        tot = 0
        for c1 in range(3):
            for c2 in range(3):
                if bb.shell[c1, t1] != bb.shell[c2, t2]:
                    continue
                tr1 = list(fc.placements[ci1])
                tr2 = list(fc.placements[ci2])
                sw1, sw2 = (tr2[c2], tr1[c1]) if swap else (tr1[c1], tr2[c2])
                k = ks[c1, t1]

                def pair(m, t=tr1[c1]):
                    if not (shot and t == tr2[c2]):
                        return 1
                    view = fc.cf_mat_bk[t][t][t]
                    P_gal = npk.get_mu(np.array([m]), TERMS, TERMS, view, np.array([[k]]), fc.z_mid)[0, 0].real
                    return 1 + 1 / (view.survey[0].n_g(fc.z_mid) * P_gal)

                A = 0.5 * sum(
                    wi
                    * pair(m)
                    * self.g(fc, ks[:, t1], c1, 1, tr1, l1, m, sw1)
                    * np.conj(self.g(fc, ks[:, t2], c2, sign, tr2, l2, m, sw2))
                    for m, wi in zip(mu_e, w)
                )
                tot += A / (4 * np.pi * k**2 * dk / fc.k_f**3)
        return tot

    @pytest.mark.parametrize("pt", [False, True])
    @pytest.mark.parametrize("fixture", ["small_mt", "small_mt8"])
    def test_matches_brute_force(self, fixture, pt, request):
        fc = request.getfixturevalue(fixture)
        nt = len(fc.placements)
        bb = BBCovBk(fc, TERMS, LN, pt=pt)
        ks = np.array(fc.args[1:4])
        closed = (
            ks[1] + ks[2] - ks[0] >= 1.5 * fc.forecast.s_k * fc.k_f
        )  # all closed across the bin - brute counts uniformly
        rng = np.random.default_rng(3)
        pairs = []
        while len(pairs) < 3:
            t1, t2 = rng.choice(np.flatnonzero(closed), 2)
            if set(bb.shell[:, t1]) & set(bb.shell[:, t2]):
                pairs.append((t1, t2))
        pairs.append((pairs[0][0], pairs[0][0]))

        for t1, t2 in pairs:
            ci1, ci2 = rng.integers(0, nt, 2)
            got, ref = [], []
            for l1, l2 in [(0, 0), (1, 2), (3, 1)]:
                got.append(bb_element(bb, t1, LN.index(l1) * nt + ci1, t2, LN.index(l2) * nt + ci2))
                args = (fc, bb, t1, ci1, l1, t2, ci2, l2)
                ref.append(self.brute(*args, swap=True, sign=-1))
                if pt:
                    ref[-1] += self.brute(*args, swap=False, sign=1, shot=True)
            # relative to the pair's largest element - some vanish by symmetry. 1/P_gal is not polynomial in mu, so
            # PT's shot noise is not exact at n_mu=12
            assert np.max(np.abs(np.array(got) - ref)) <= (1e-7 if pt else 1e-10) * np.max(np.abs(ref))

    @pytest.mark.parametrize("pt", [False, True])
    def test_m_matches_brute_force(self, small_mt_m, pt):
        """all_m rows - Re Y_lm with the LOS azimuth about k1, see FullCovBk.re_ylm"""
        fc = small_mt_m
        bb = BBCovBk(fc, TERMS, LN_M, pt=pt)
        lm = fc.multipoles(LN_M)
        ks = np.array(fc.args[1:4])
        closed = ks[1] + ks[2] - ks[0] >= 1.5 * fc.forecast.s_k * fc.k_f
        rng = np.random.default_rng(4)
        pairs = []
        while len(pairs) < 3:
            t1, t2 = rng.choice(np.flatnonzero(closed), 2)
            if set(bb.shell[:, t1]) & set(bb.shell[:, t2]):
                pairs.append((t1, t2))
        pairs.append((pairs[0][0], pairs[0][0]))

        for t1, t2 in pairs:
            ci1, ci2 = rng.integers(0, len(fc.placements), 2)
            got, ref = [], []
            for l1, l2 in [((1, 1), (1, 1)), ((2, 1), (1, 0)), ((2, 2), (2, 1)), ((0, 0), (2, 2))]:
                got.append(bb_element(bb, t1, lm.index(l1) * 4 + ci1, t2, lm.index(l2) * 4 + ci2))
                args = (fc, bb, t1, ci1, l1, t2, ci2, l2)
                ref.append(self.brute(*args, swap=True, sign=-1))
                if pt:
                    ref[-1] += self.brute(*args, swap=False, sign=1, shot=True)
            assert np.max(np.abs(np.array(got) - ref)) <= (1e-7 if pt else 1e-10) * np.max(np.abs(ref))


def lattice_pairs(T1, T2, dk):
    """(pairs of triangles sharing a mode)/(N1 N2) on the k_f = 1 lattice, for bins centred on T1, T2"""
    K = int(max(T1 + T2) + dk)
    g = np.arange(-K, K + 1)
    Q = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(float)
    R = np.linalg.norm(Q, axis=1)

    def in_shell(r, k):
        return (r >= k - dk / 2) & (r < k + dk / 2)

    def n_side(T, c):
        """# triangles of T with side c on each lattice vector of shell c"""
        a, b = [i for i in range(3) if i != c]
        kc, qa = Q[in_shell(R, T[c])], Q[in_shell(R, T[a])]
        return kc, np.sum(in_shell(np.linalg.norm(kc[:, None] + qa[None], axis=-1), T[b]), axis=1)

    n1 = [n_side(T1, c) for c in range(3)]
    n2 = [n_side(T2, c) for c in range(3)]
    pairs = 0
    for c1 in range(3):
        for c2 in range(3):
            if T1[c1] == T2[c2]:
                (kv, a1), (kv2, a2) = n1[c1], n2[c2]
                idx = {tuple(v): i for i, v in enumerate(kv2)}
                pairs += np.sum(a1 * a2[[idx[tuple(-v)] for v in kv]])  # second triangle on -k
    return pairs / (n1[0][1].sum() * n2[0][1].sum())


class ConstB(BBCovBk):
    def bk_clust(self, tracers, legs, theta, mus):
        return np.ones_like(mus[1], dtype=np.complex128)  # mus[0] is the bare mu nodes

    def bk_shot(self, tracers, legs, mus):
        return 0


@pytest.mark.parametrize(
    "T1, T2, rel",
    [
        ((12, 10, 6), (10, 8, 6), 0.03),
        ((12, 12, 12), (12, 12, 12), 0.03),
        ((12, 8, 4), (12, 8, 4), 0.1),  # flattened - the closure fraction varies across the shared shell
        ((12, 8, 4), (8, 6, 2), 0.1),  # flattened, sharing the long side of one and the middle of the other
    ],
)
def test_mode_count_matches_lattice(bk_bin, T1, T2, rel, monkeypatch):
    """B = 1 is the real-space B_T B_T'/N_k per shared pair - the thin-shell count, with the closure fraction
    resolved across the shell, is within lattice noise (larger for flattened bins, with fewer modes). Without it the
    flattened pairs are off by 15-30%. And without shot noise PT is BB (2BB)"""
    fc = bk_bin
    bb = ConstB(fc, TERMS, [0], pt=False, n_delta=4)
    ks = np.array(fc.args[1:4]) / fc.k_f
    t1, t2 = (np.flatnonzero(np.all(np.isclose(ks, np.array(T)[:, None]), axis=0))[0] for T in (T1, T2))
    got = bb_element(bb, t1, 0, t2, 0) / (4 * np.pi)  # B_00 = sqrt(4pi) B
    ref = lattice_pairs(T1, T2, fc.forecast.s_k)
    assert got == pytest.approx(ref, rel=rel)
    monkeypatch.setattr(fc.cf_mat_bk[0][0][0].survey[0], "n_g", lambda zz: 1e30 + 0 * zz)
    assert bb_element(ConstB(fc, TERMS, [0], n_delta=4), t1, 0, t2, 0) / (4 * np.pi) == pytest.approx(
        2 * got, rel=1e-12
    )


@pytest.mark.parametrize("fixture", ["small_st", "small_mt"])
def test_total_positive_definite(fixture, request):
    """BB alone is indefinite (odd l) - with PT one tracer is a covariance by construction, and multi-tracer (BB's
    exact shot noise swap, ~-3% here) is once whitened by the Gaussian part (min eigenvalue ~0.99)"""
    fc = request.getfixturevalue(fixture)
    ev = np.linalg.eigvalsh(dense_bb(BBCovBk(fc, TERMS, LN, pt=False)))
    assert ev.min() < -0.1 * ev.max()
    C_ng = dense_bb(BBCovBk(fc, TERMS, LN))
    if fc.all_tracer:
        Wh = whitener(fc.get_cov_mat(LN, n_mu=16, n_phi=16))
        assert np.linalg.eigvalsh(np.eye(Wh.shape[1]) + Wh.conj().T @ C_ng @ Wh).min() > 0
    else:
        ev = np.linalg.eigvalsh(C_ng)
        assert ev.min() > -1e-12 * ev.max()


@pytest.mark.parametrize("fixture", ["small_st_m", "small_mt_m"])
def test_total_positive_definite_all_m(fixture, request):
    """as test_total_positive_definite with every m - whitened by the Gaussian part on its kept subspace, which drops
    the rows that vanish on equal sides"""
    fc = request.getfixturevalue(fixture)
    C_ng = dense_bb(BBCovBk(fc, TERMS, LN_M))
    Wh = whitener(fc.get_cov_mat(LN_M, n_mu=16, n_phi=16))
    assert np.linalg.eigvalsh(np.eye(Wh.shape[1]) + Wh.conj().T @ C_ng @ Wh).min() > 0


@pytest.mark.parametrize("fixture", ["small_st", "small_mt"])
def test_finite_without_clustering(fixture, request, monkeypatch):
    """P -> eps P at fixed n: the catalogue-subtracted terms vanish, led by PT's shot noise (P/n)^2/(n P) ~ eps - B
    with its 1/n^2 over n P_gal in PT diverged as 1/eps"""
    fc = request.getfixturevalue(fixture)
    pk = BaseInt.pk
    C = {}
    for eps in (1e-5, 1e-6):
        monkeypatch.setattr(BaseInt, "pk", lambda self, x, zz, zz2=None, eps=eps: eps * pk(self, x, zz, zz2))
        C[eps] = dense_bb(BBCovBk(fc, TERMS, LN)) / eps
    assert np.abs(C[1e-6]).max() > 0
    np.testing.assert_allclose(C[1e-6], C[1e-5], atol=1e-3 * np.abs(C[1e-5]).max())


def test_mt_bb_clustering_is_own_tracer(small_mt, monkeypatch):
    """Without shot noise the squeezed-limit swaps cancel in each BB pair (Z1(-mu)* = Z1(mu)) - BB is the own-tracer
    one, the (n, T, T) columns paired with (-n, T', T')"""
    for t in range(2):
        monkeypatch.setattr(small_mt.cf_mat_bk[t][t][t].survey[0], "n_g", lambda zz: 1e30 + 0 * zz)
    bb = BBCovBk(small_mt, TERMS, LN, pt=False)
    n_mu, n_t = len(bb.mu), bb.n_t
    own = np.arange(n_mu * n_t * n_t).reshape(n_mu, n_t, n_t)[:, range(n_t), range(n_t)]
    C_bb = dense_bb(bb)
    bb.lam = np.zeros_like(bb.lam)
    bb.lam[own[:, :, None], own[::-1, None, :]] = (
        small_mt.k_f**2 / (8 * np.pi * small_mt.forecast.s_k) * utils.leggauss(n_mu)[1][:, None, None]
    )
    ref = dense_bb(bb)
    assert np.max(np.abs(C_bb - ref)) <= 1e-10 * np.max(np.abs(ref))


@pytest.mark.parametrize("fixture", ["small_st", "small_mt"])
def test_woodbury_matches_dense(fixture, request):
    fc = request.getfixturevalue(fixture)
    D = fc.get_cov_mat(LN, n_mu=16, n_phi=16)
    bb = BBCovBk(fc, TERMS, LN)
    inv = WoodburyInvCov([(fc.invert_matrix(D, None), bb.U, bb.shell)], bb.lam, bb.n_shell)

    C_bb = dense_bb(bb)
    assert np.max(np.abs(C_bb - C_bb.conj().T)) <= 1e-14 * np.max(np.abs(C_bb))
    C = dense_block_diag(D) + C_bb

    n, N = D.shape[0], D.shape[-1]
    rng = np.random.default_rng(0)
    d1, d2 = (rng.normal(size=(n, N)) + 1j * rng.normal(size=(n, N)) for _ in range(2))
    ref = d1.T.ravel().conj() @ np.linalg.solve(C, d2.T.ravel())
    assert inv.contract(d1, d2) == pytest.approx(ref, rel=1e-9)

    again = pickle.loads(pickle.dumps(inv))  # the Sampler saves inv_covs
    assert again.contract(d1, d2) == inv.contract(d1, d2)


def test_mt_auto_blocks_match_single_tracer(cosmo, cosmo_funcs):
    """Two copies of one survey: the all-X and all-Y blocks are the single-tracer term. The mixed blocks are
    not - distinct samples share no shot noise"""
    survey = cw.SurveyParams.Euclid(cosmo)
    cf2 = cw.ClassWAP(cosmo, [survey, cw.SurveyParams.Euclid(cosmo)], verbose=False)
    st = FullForecast(cosmo_funcs, kmax_func=0.035, s_k=2, N_bins=2).get_bk_bin(0, cov_terms=TERMS)
    mt = FullForecast(cf2, kmax_func=0.035, s_k=2, N_bins=2).get_bk_bin(0, cov_terms=TERMS, all_tracer=True)

    ref = dense_bb(BBCovBk(st, TERMS, LN))
    N, n = len(st.args[1]), len(LN)
    C = dense_bb(BBCovBk(mt, TERMS, LN)).reshape(N, n, 4, N, n, 4)
    for ci in (0, 3):
        np.testing.assert_allclose(
            C[:, :, ci, :, :, ci].reshape(N * n, -1), ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max()
        )


def test_bb_off_unchanged(forecast):
    """Without cov_ng the inverse covariance and the contraction are the previous ones"""
    assert not forecast.cov_ng
    fc = forecast.get_bk_bin(0)
    inv = fc.get_inv_cov(LN, n_mu=16, n_phi=16)
    assert isinstance(inv, np.ndarray)
    np.testing.assert_array_equal(inv, fc.invert_matrix(fc.get_cov_mat(LN, n_mu=16, n_phi=16)))

    rng = np.random.default_rng(1)
    d1, d2 = (rng.normal(size=inv.shape[1:]) + 1j * rng.normal(size=inv.shape[1:]) for _ in range(2))
    assert contract(d1, inv, d2) == np.sum(np.einsum("ik,ijk,jk->k", np.conjugate(d1), inv, d2))


def test_bb_flag_reaches_fisher(cosmo_funcs):
    """cov_ng switches the precompute to WoodburyInvCov, and adding the non-Gaussian term lowers the SNR"""
    kw = dict(kmax_func=0.05, s_k=2, N_bins=2)
    fish = {}
    for flag in (False, True):
        F = FullForecast(cosmo_funcs, cov_ng=flag, **kw)
        _, inv_covs = F._precompute_derivatives_and_covariances(
            [None], terms="NPP", bk_terms="NPP", bkln=[0, 2], verbose=False
        )
        assert isinstance(inv_covs[0]["bk"], WoodburyInvCov) == flag
        fish[flag] = F.bk_SNR("NPP", [0, 2], verbose=False)
    assert np.all(np.isfinite(fish[True]))
    assert np.all(fish[True].real < fish[False].real)


def test_ng_kwargs_reach_bb(cosmo_funcs):
    """FullForecast(ng_kwargs) sets BBCovBk's quadrature - one tracer has a BB + PT column per mu node, rank one in
    each (n, -n) pair, and one of PT's shot noise: 3/2 n_mu once WoodburyInvCov compresses Lambda"""
    F = FullForecast(cosmo_funcs, kmax_func=0.05, s_k=2, N_bins=2, cov_ng=True, ng_kwargs=dict(n_mu=8))
    assert F.get_bk_bin(0).get_inv_cov([0, 2], n_mu=16, n_phi=16).lam.shape == (12, 12)


# ---- tree-level PT (exchange trispectrum) ----


def test_pair_response_reproduces_bispectrum(cosmo_funcs):
    """B_123 = 2 Z1(k3) P(k3) R_12 + 2 Z1 Z1 Z2(-k1, -k2) P1 P2 - the trispectrum's R is the bispectrum's"""
    zz = 1.2
    rng = np.random.default_rng(0)
    k1, k2 = rng.uniform(0.01, 0.1, (2, 50))
    th = rng.uniform(0.1, 3.0, 50)
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(th))
    tri = (k[:, None, None] for k in (k1, k2, k3, th))
    k1_, k2_, k3_, th_ = tri
    mu, phi = np.array([0.3]), np.array([1.1])
    pk = BaseInt(cosmo_funcs).pk
    for kern in (["N"], TERMS):
        B = nbk.get_mu_phi(mu, phi, kern, kern, kern, cosmo_funcs, k1_, k2_, k3_, th_, 1.2)[:, 0, 0]
        m1, m2, m3 = (m[:, 0, 0] for m in np.broadcast_arrays(*nbk.los_cosines(mu[:, None], phi, k1_, k2_, k3_, th_)))
        R = nbk.pair_response(kern, cosmo_funcs, cosmo_funcs, zz, m1, m2, k1, k2, np.cos(th))
        Z1 = [nbk.get_Z1(kern, cosmo_funcs, zz, m, k) for m, k in ((m1, k1), (m2, k2), (m3, k3))]
        rest = (
            2
            * Z1[0]
            * Z1[1]
            * nbk.get_Z2(kern, cosmo_funcs, zz, -m1, -m2, k1, k2, np.cos(th))
            * pk(k1, zz)
            * pk(k2, zz)
        )
        np.testing.assert_allclose(2 * Z1[2] * pk(k3, zz) * R + rest, B, rtol=1e-12)


def test_T_exchange_symmetries(forecast_mt):
    """T of four tracers is symmetric under permuting the legs with their tracers, and real (T(-k) = conj T)"""
    views = [forecast_mt.cf_mat_bk[t][t][t] for t in (0, 1)]
    rng = np.random.default_rng(1)
    v = rng.normal(scale=0.03, size=(3, 20, 3))
    vecs = np.concatenate([v, -v.sum(axis=0, keepdims=True)])
    tracers = (0, 1, 1, 0)
    T = nbk.get_T_exchange(TERMS, [views[t] for t in tracers], 1.2, vecs)
    for perm in [(1, 0, 2, 3), (2, 3, 0, 1), (3, 1, 2, 0)]:
        Tp = nbk.get_T_exchange(TERMS, [views[tracers[p]] for p in perm], 1.2, vecs[list(perm)])
        np.testing.assert_allclose(Tp, T, rtol=1e-10)
    np.testing.assert_allclose(nbk.get_T_exchange(TERMS, [views[t] for t in tracers], 1.2, -vecs), T.conj(), rtol=1e-10)


def test_F3_G3_match_P13():
    """6 r^2 int dx F3(k, q, -q), q = r k, against the closed-form P13 kernel of the density - and G3 of the velocity"""
    x, w = np.polynomial.legendre.leggauss(64)
    k = np.array([0.0, 0.0, 1.0])
    for r in (0.1, 0.5, 1.3, 3.0):
        q = r * np.stack([np.sqrt(1 - x**2), 0 * x, x], -1)
        F3, G3 = nbk._FG3(np.broadcast_to(k, q.shape), q, -q)
        L = np.log(abs((1 + r) / (1 - r)))
        ref_F = (12 / r**2 - 158 + 100 * r**2 - 42 * r**4 + 3 / r**3 * (r**2 - 1) ** 3 * (7 * r**2 + 2) * L) / 252
        ref_G = (12 / r**2 - 82 + 4 * r**2 - 6 * r**4 + 3 / r**3 * (r**2 - 1) ** 3 * (r**2 + 2) * L) / 84
        assert 6 * r**2 * np.sum(w * F3) == pytest.approx(ref_F, rel=1e-9)
        assert 6 * r**2 * np.sum(w * G3) == pytest.approx(ref_G, rel=1e-9)


def test_Z3_matches_redshift_space_mapping(cosmo_funcs):
    """The exact mapping delta_s(k) = <(1 + delta_g) exp(-i k.s)>_x, s = x + f (i k_z/k^2 theta) z-hat, on a periodic
    grid (k_f = 1): delta and theta to third order from delta_L = sum_a eps_a 2 cos(q_a.x), delta_g = b1 delta +
    b2/2 delta^2 + g2 G2. The eps1 eps2 eps3 part at q1 + q2 + q3, by a mixed difference to O(h^2), is 6 Z3/D^3"""
    import itertools

    zz, n = 1.2, 16
    D, f = cosmo_funcs.D(zz), cosmo_funcs.f(zz)
    b1, b2, g2 = (getattr(cosmo_funcs.survey[0], b)(zz) for b in ("b_1", "b_2", "g_2"))
    g = np.arange(n) * 2 * np.pi / n
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
    K = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1 / n)] * 3, indexing="ij"), -1)
    K2 = np.sum(K**2, -1)
    K2[0, 0, 0] = 1

    def delta_s(eps, qs):
        modes = [(e, s * q) for e, q in zip(eps, qs) for s in (1, -1)]
        d, th = sum(e * np.exp(1j * X @ q) for e, q in modes), sum(e * np.exp(1j * X @ q) for e, q in modes)
        for order in (2, 3):
            for ms in itertools.product(modes, repeat=order):
                p = sum(q for _, q in ms)
                if np.allclose(p, 0):
                    continue
                FG = nbk._FG2(*(q for _, q in ms)) if order == 2 else nbk._FG3(*(q for _, q in ms))
                w = np.prod([e for e, _ in ms]) * np.exp(1j * X @ p)
                d, th = d + FG[0] * w, th + FG[1] * w
        d, th = d.real, th.real
        dk = np.fft.fftn(d)
        tidal = sum(np.fft.ifftn(K[..., i] * K[..., j] / K2 * dk).real ** 2 for i in range(3) for j in range(3)) - d**2
        s_z = np.fft.ifftn(f * 1j * K[..., 2] / K2 * np.fft.fftn(th)).real
        k = sum(qs)
        return np.mean((1 + b1 * d + b2 / 2 * d**2 + g2 * tidal) * np.exp(-1j * (X @ k + k[2] * s_z)))

    h = 1e-3
    for qs in [([0, 0, 1], [2, -2, -2], [2, 2, -1]), ([-1, 2, 0], [-1, 2, -1], [0, 1, 0])]:
        qs = [np.array(q, dtype=float) for q in qs]
        got = sum(
            s1 * s2 * s3 * delta_s((s1 * h, s2 * h, s3 * h), qs) for s1, s2, s3 in itertools.product((1, -1), repeat=3)
        ) / (8 * h**3)
        assert got == pytest.approx(6 * nbk.get_Z3(cosmo_funcs, zz, *qs) / D**3, rel=1e-5)


def test_Z3_IR_limit(cosmo_funcs):
    """3 Z3(k, q, -q) -> -1/2 [(k.q + f k_z q_z)/q^2]^2 Z1(k) as q -> 0 - the displacement of the redshift-space mode"""
    zz = 1.2
    D, f = cosmo_funcs.D(zz), cosmo_funcs.f(zz)
    rng = np.random.default_rng(0)
    k, e = rng.normal(scale=0.05, size=(2, 3))
    Z1 = nbk.get_Z1(["N"], cosmo_funcs, zz, k[2] / np.linalg.norm(k), np.linalg.norm(k)) / D
    for eps in (1e-5, 1e-6):
        q = eps * e
        ref = -0.5 * ((k @ q + f * k[2] * q[2]) / (q @ q)) ** 2 * Z1
        assert 3 * nbk.get_Z3(cosmo_funcs, zz, k, q, -q) / D**3 == pytest.approx(ref, rel=1e-6)


def test_B_vec_matches_mu_phi(forecast_mt):
    """The 3D-vector B used by get_T_shot is get_mu_phi's, multi-tracer"""
    zz = 1.2
    views = [forecast_mt.cf_mat_bk[t][t][t] for t in (0, 1, 1)]
    rng = np.random.default_rng(2)
    v = rng.normal(scale=0.03, size=(2, 20, 3))
    vecs = np.concatenate([v, -v.sum(axis=0, keepdims=True)])
    k1, k2, k3 = np.linalg.norm(vecs, axis=-1)
    mu1, mu2 = vecs[0, :, 2] / k1, vecs[1, :, 2] / k2
    theta = np.arccos(np.sum(vecs[0] * vecs[1], -1) / (k1 * k2))
    phi = np.arccos(np.clip((mu2 - mu1 * np.cos(theta)) / (np.sqrt(1 - mu1**2) * np.sin(theta)), -1, 1))
    ref = np.diagonal(
        nbk.get_mu_phi(mu1, phi, ["N"], ["N"], ["N"], forecast_mt.cf_mat_bk[0][1][1], k1, k2, k3, theta, zz)
    )
    np.testing.assert_allclose(nbk._B_vec(views, zz, vecs), ref, rtol=1e-12)


@pytest.mark.parametrize("shot", [False, True])
def test_T_tree_symmetries(forecast_mt, shot):
    """The tree-level T is symmetric under permuting the legs with their tracers - its shot noise under those keeping
    the two estimators (0, 1) and (2, 3) apart - and real"""
    views = [forecast_mt.cf_mat_bk[t][t][t] for t in (0, 1)]
    rng = np.random.default_rng(1)
    v = rng.normal(scale=0.03, size=(3, 20, 3))
    vecs = np.concatenate([v, -v.sum(axis=0, keepdims=True)])
    tracers = (0, 1, 1, 0)
    T = nbk.get_T_tree([views[t] for t in tracers], 1.2, vecs, shot=shot)
    perms = [(1, 0, 2, 3), (2, 3, 0, 1), (0, 1, 3, 2)] if shot else [(1, 0, 2, 3), (2, 3, 0, 1), (3, 1, 2, 0)]
    for perm in perms:
        Tp = nbk.get_T_tree([views[tracers[p]] for p in perm], 1.2, vecs[list(perm)], shot=shot)
        np.testing.assert_allclose(Tp, T, rtol=1e-10)
    Tm = nbk.get_T_tree([views[t] for t in tracers], 1.2, -vecs, shot=shot)
    np.testing.assert_allclose(Tm, T.conj(), rtol=1e-10)


def test_T_shot_limit_at_coincidence(cosmo_funcs):
    """Where the two estimators' fields coincide - every rotation of a flattened triangle - get_T_shot takes B's limit:
    just inside the cut it agrees with just outside, where B converges as P(|k_0 + k_2|)"""
    rng = np.random.default_rng(4)
    q_a, q_b = rng.normal(scale=0.03, size=(2, 3))
    e = rng.normal(size=3)
    e /= np.linalg.norm(e)
    T = []
    for eps in (5e-6, 2e-5):
        d = eps * np.linalg.norm(q_a) * e
        T.append(nbk.get_T_shot([cosmo_funcs] * 4, 1.2, np.stack([q_a, q_b, -(q_a + d), -(q_b - d)])))
    assert T[0] == pytest.approx(T[1], rel=1e-2)


class PairResponseB(BBCovBk):
    """B -> 2 Z1(k_w) P(k_w) R_xy: the part carrying P of the shared side, which the s channel factorises into"""

    def bk_clust(self, tracers, legs, theta, mus):
        views = [self.cf_mat_bk[t][t][t] for t in tracers]
        kw, kx, ky = legs
        cos_xy = (kw**2 - kx**2 - ky**2) / (2 * kx * ky)
        R = nbk.pair_response(["N"], views[1], views[2], self.zz, mus[1], mus[2], kx, ky, cos_xy)
        return 2 * nbk.get_Z1(self.terms, views[0], self.zz, mus[0], kw) * BaseInt(views[0]).pk(kw, self.zz) * R


@pytest.mark.parametrize("fixture", ["small_st", "small_mt"])
def test_pt_s_channel_matches_separable(fixture, request, monkeypatch):
    """The s channel is P(k) R R, so it is PT' of BBCovBk with B -> 2 Z1 P R - checks the 3D geometry, the pairing
    P and the mode count of PTTreeCovBk against the verified separable path. Shot noise off - the separable form
    only has P's Z1 Z1 part"""
    fc = request.getfixturevalue(fixture)
    for t in range(len(fc.cf_mat_bk)):
        monkeypatch.setattr(fc.cf_mat_bk[t][t][t].survey[0], "n_g", lambda zz: 1e30 + 0 * zz)
    pt = PTTreeCovBk(fc, TERMS, LN, channels="s")

    bb = PairResponseB(fc, TERMS, LN)  # n_delta=1 - PTTreeCovBk uses the bin-averaged closure too
    n_mu = len(bb.mu)
    n_t = bb.n_t
    own = np.arange(n_mu * n_t * n_t).reshape(n_mu, n_t, n_t)[:, range(n_t), range(n_t)]
    bb.lam = np.zeros_like(bb.lam)
    bb.lam[own[:, :, None], own[:, None, :]] = (
        fc.k_f**2 / (8 * np.pi * fc.forecast.s_k) * utils.leggauss(n_mu)[1][:, None, None]
    )
    ref = dense_bb(bb)
    assert np.max(np.abs(pt.cov - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_pt_hermitian_by_construction(small_mt):
    """blocks with the two triangles swapped is the conjugate transpose - PTTreeCovBk only computes t1 <= t2.
    The t and u channels (and T3111, T's shot noise) only to quadrature error: swapping puts each triangle on the
    other's psi grid"""
    pt = PTTreeCovBk(small_mt, TERMS, LN, n_psi=4, channels="tu")  # cheap - only blocks() is used, on a finer grid
    pt.channels = "tu3"
    pt.psi1 = np.pi * (np.arange(16) + 0.5) / 16
    pt.psi2 = 2 * np.pi * np.arange(32) / 32
    t1, t2 = np.nonzero((pt.shell[0][:, None] == pt.shell[2]) & ~np.eye(pt.shell.shape[1], dtype=bool))
    t1, t2 = t1[:6], t2[:6]
    fwd, bwd = pt.blocks(t1, 0, t2, 2), pt.blocks(t2, 2, t1, 0)
    np.testing.assert_allclose(bwd, fwd.conj().transpose(0, 3, 4, 1, 2), atol=1e-3 * np.abs(fwd).max())


# ---- power spectrum - bispectrum cross-covariance ----


def pk_rows(pk_fc, ln):
    """(l, (a, b)) of each row of PkForecast.get_data_vector"""
    if not pk_fc.all_tracer:
        return [(l, (0, 0)) for l in ln]
    return [(l, pair) for l in ln for pair in ([(0, 0), (0, 1), (1, 1)] if l % 2 == 0 else [(0, 1)])]


def pb_element(pb, bb, kb, rP, t, rB):
    """C_PB between row rP of pk bin kb and row rB of triangle t, from the factors"""
    return sum(
        pb.V[kb, rP, 0] @ (pb.lam * bb.U[t, rB, w].conj()) for w in range(3) if bb.shell[w, t] == pb.shell[0, kb]
    )


def dense_joint(blocks, lam, n_shell):
    """D + W Xi W^dagger as one dense matrix - the blocks' rows in order, each bin-major"""
    widths = [U.shape[-1] for _, U, _ in blocks]
    offsets = np.cumsum([0] + widths)
    Ws, Ds = [], []
    for (D, U, shell), lo in zip(blocks, offsets[:-1]):
        N, n = U.shape[:2]
        W = np.zeros((N, n, n_shell, offsets[-1]), dtype=np.complex128)
        for w in range(len(shell)):
            W[np.arange(N), :, shell[w], lo : lo + U.shape[-1]] += U[:, :, w]
        Ws.append(W.reshape(N * n, -1))
        Ds.append(dense_block_diag(D))
    W = np.vstack(Ws)
    D = np.zeros((len(W), len(W)), dtype=np.complex128)
    at = 0
    for Db in Ds:
        D[at : at + len(Db), at : at + len(Db)] = Db
        at += len(Db)
    return D + W @ np.kron(np.eye(n_shell), lam) @ W.conj().T


def whitener(D):
    """D^-1/2 bin by bin on its kept eigen-subspace (as invert_matrix) - rows as dense_block_diag, a column per kept direction"""
    n, _, N = D.shape
    cols = []
    for t in range(N):
        w, v = np.linalg.eigh((D[:, :, t] + D[:, :, t].conj().T) / 2)
        keep = w > 1e-10 * w.max()
        col = np.zeros((N * n, keep.sum()), dtype=np.complex128)
        col[t * n : (t + 1) * n] = v[:, keep] / np.sqrt(w[keep])
        cols.append(col)
    return np.hstack(cols)


def joint_blocks(pk_fc, bk_fc, ln_pk, ln_bk):
    """(D, U, shell) of each data vector and Xi - as joint_inv_cov, with the covariances not their inverses"""
    D_pk = pk_fc.get_cov_mat(ln_pk, n_mu=32)
    D_bk = bk_fc.get_cov_mat(ln_bk, n_mu=16, n_phi=16)
    bb = BBCovBk(bk_fc, TERMS, ln_bk)
    pb = PBCov(pk_fc, bb, ln_pk)
    cross = np.diag(pb.lam)
    lam = np.block([[np.zeros_like(cross), cross], [cross, bb.lam]])
    return [(D_pk, pb.V, pb.shell), (D_bk, bb.U, bb.shell)], lam, bb, pb


@pytest.mark.parametrize("fixtures", [("small_mt", "small_mt_pk"), ("small_mt8", "small_mt8_pk")])
def test_pb_matches_brute_force(fixtures, request):
    """Independent of PBCov's algebra: in the frame of the P mode q, the triangle sits on +q (P^{a t}, t -> b) or
    on -q (P^{t b}(q), t -> a) - PBCov writes both on the triangle's side, with (-1)^l. The swap t -> b is B^(N)'s,
    as BB. all_tracer=8 on the bispectrum is the usual all_tracer for pk"""
    fc, pk_fc = (request.getfixturevalue(f) for f in fixtures)
    bb = BBCovBk(fc, TERMS, LN)
    ln_pk = [0, 1, 2]
    pb = PBCov(pk_fc, bb, ln_pk)
    rows = pk_rows(pk_fc, ln_pk)
    geom = TestGeometry()
    ks = np.array(fc.args[1:4])
    _, kk, zz = pk_fc.args
    dk = fc.forecast.s_k * fc.k_f
    mu_q, w_q = np.polynomial.legendre.leggauss(16)
    closed = ks[1] + ks[2] - ks[0] > 1e-8
    rng = np.random.default_rng(4)

    def P(x, y, mu, k):
        return npk.get_mu(np.array([mu]), TERMS, TERMS, pk_fc.cf_mat[x][y], np.array([[k]]), zz)[0, 0] + (
            x == y
        ) / pk_fc.cf_mat[x][x].n_g(zz)

    checked = 0
    for t in rng.permutation(np.flatnonzero(closed)):
        sides = [i for i in range(3) if bb.shell[i, t] < bb.n_shell]
        kb = np.flatnonzero(pb.shell[0] == bb.shell[sides[0], t])
        if checked == 4 or not len(kb):
            continue
        kb = kb[0]
        k = kk[kb]
        for rP, (l, (a, b)) in enumerate(rows):
            ci, l2 = rng.integers(0, len(fc.placements)), LN[rng.integers(0, len(LN))]
            ref = 0
            for i in range(3):
                if bb.shell[i, t] != pb.shell[0, kb]:
                    continue
                own = list(fc.placements[ci])
                ti = own[i]
                for m, wm in zip(mu_q, w_q):
                    g_plus = geom.g(fc, ks[:, t], i, 1, own, l2, m, swap=b)  # shared side on +q
                    g_minus = geom.g(fc, ks[:, t], i, -1, own, l2, m, swap=a)  # on -q
                    leg = (2 * l + 1) * eval_legendre(l, m)
                    ref += 0.5 * wm * leg * (P(a, ti, m, k) * np.conj(g_plus) + P(ti, b, m, k) * np.conj(g_minus))
            ref /= 4 * np.pi * k**2 * dk / fc.k_f**3
            got = pb_element(pb, bb, kb, rP, t, LN.index(l2) * len(fc.placements) + ci)
            assert np.abs(got - ref) <= 1e-7 * np.abs(
                ref
            )  # the Z1 ratio is not polynomial in mu, so not exact at n_mu=12
        checked += 1
    assert checked == 4


@pytest.mark.parametrize("fixtures", [("small_st", "small_st_pk"), ("small_mt", "small_mt_pk")])
def test_joint_woodbury_matches_dense(fixtures, request):
    """joint_inv_cov's Woodbury against the dense joint inverse, and the joint covariance is positive definite.
    The non-Gaussian part is hermitian by construction - FullCovBk's diagonal carries quadrature-level imaginary parts"""
    fc, pk_fc = (request.getfixturevalue(f) for f in fixtures)
    ln_pk = [0, 1, 2] if pk_fc.all_tracer else [0, 2]
    blocks, lam, bb, _ = joint_blocks(pk_fc, fc, ln_pk, LN)
    C = dense_joint(blocks, lam, bb.n_shell)
    C_ng = dense_joint([(np.zeros_like(D), U, sh) for D, U, sh in blocks], lam, bb.n_shell)
    assert np.max(np.abs(C_ng - C_ng.conj().T)) <= 1e-12 * np.max(np.abs(C_ng))
    # whiten by the Gaussian part bin by bin, on its kept subspace as invert_matrix (pk and bk differ by ~1e15)
    Wh = [whitener(D) for D, _, _ in blocks]
    Wh = np.block([[Wh[0], np.zeros((len(Wh[0]), Wh[1].shape[1]))], [np.zeros((len(Wh[1]), Wh[0].shape[1])), Wh[1]]])
    assert np.linalg.eigvalsh(np.eye(Wh.shape[1]) + Wh.conj().T @ C_ng @ Wh).min() > 0

    inv = WoodburyInvCov(
        [(np.linalg.inv(np.moveaxis(D, -1, 0)).transpose(1, 2, 0), U, sh) for D, U, sh in blocks], lam, bb.n_shell
    )
    assert len(inv.lam) < len(lam)  # PBCov's zero padding is dropped
    rng = np.random.default_rng(2)
    d1, d2 = (
        [rng.normal(size=U.shape[:2][::-1]) + 1j * rng.normal(size=U.shape[:2][::-1]) for _, U, _ in blocks]
        for _ in range(2)
    )
    ref = np.concatenate([d.T.ravel() for d in d1]).conj() @ np.linalg.solve(
        C, np.concatenate([d.T.ravel() for d in d2])
    )
    assert inv.contract(d1, d2) == pytest.approx(ref, rel=1e-8)


def test_joint_without_cross_is_the_sum(small_mt, small_mt_pk):
    """Xi with no pk-bk block gives the pk and bk contractions separately"""
    ln_pk = [0, 1, 2]
    blocks, lam, bb, _ = joint_blocks(small_mt_pk, small_mt, ln_pk, LN)
    b = bb.lam.shape[0]
    lam[:b, b:] = lam[b:, :b] = 0
    D_inv = [small_mt.invert_matrix(D, None) for D, _, _ in blocks]
    joint = WoodburyInvCov([(Di, U, sh) for Di, (_, U, sh) in zip(D_inv, blocks)], lam, bb.n_shell)
    bk_only = WoodburyInvCov([(D_inv[1], bb.U, bb.shell)], bb.lam, bb.n_shell)
    rng = np.random.default_rng(3)
    d_pk, d_bk = (rng.normal(size=U.shape[:2][::-1]) + 1j * rng.normal(size=U.shape[:2][::-1]) for _, U, _ in blocks)
    sep = contract(d_pk, D_inv[0], d_pk) + bk_only.contract(d_bk, d_bk)
    assert joint.contract((d_pk, d_bk), (d_pk, d_bk)) == pytest.approx(sep, rel=1e-10)


def test_cov_ng_joint_entry_points(small_mt_forecast):
    """cov_ng with pk and bk: one joint inverse per bin, used by the Fisher and combined_SNR"""
    F = FullForecast(small_mt_forecast.cosmo_funcs, kmax_func=0.035, s_k=2, N_bins=2, cov_ng=True)
    kw = dict(terms="NPP", bk_terms="NPP", pkln=[0, 2], bkln=[0, 2], all_tracer=True, cov_terms=TERMS, verbose=False)
    derivs, inv_covs = F._precompute_derivatives_and_covariances(["b_1"], **kw)
    assert set(inv_covs[0]) == {"pkbk"} and isinstance(inv_covs[0]["pkbk"], WoodburyInvCov)
    assert F._fisher_element(derivs, inv_covs, 0, 0, 0, [0, 2], [0, 2]) > 0
    snr = F.combined_SNR("NPP", [0, 2], [0, 2], verbose=False, all_tracer=True, cov_terms=TERMS)
    assert np.all(np.isfinite(snr)) and np.all(snr.real > 0)


def test_joint_positive_definite_mt_kmax01(cosmo, monkeypatch):
    """Whitened by the Gaussian part the joint covariance is I + W Xi W^dagger, whose non-unit eigenvalues are those of
    M = I + G Xi that WoodburyInvCov factorises. With the exact tracer swap in PBCov its minimum here was -0.08"""
    import cosmo_wap.forecast.covariances as covariances

    captured = {}
    lu_factor = covariances.lu_factor
    monkeypatch.setattr(covariances, "lu_factor", lambda M: (captured.__setitem__("M", M), lu_factor(M))[1])

    cf = cw.ClassWAP(cosmo, cw.SurveyParams.Euclid(cosmo).BF_split(3e-16), verbose=False)
    F = FullForecast(cf, kmax_func=0.1, s_k=2, N_bins=4)
    joint_inv_cov(
        F.get_pk_bin(1, all_tracer=True), F.get_bk_bin(1, all_tracer=True), [0, 1, 2], LN, n_mu_pk=16, n_mu=16, n_phi=16
    )
    assert np.linalg.eigvals(captured["M"]).real.min() > 0


def test_delta_nodes_keep_closed_bins_and_joint(small_st, small_st_pk):
    """Resolving the shell only changes (nearly) flattened bins, and the joint inverse still matches its dense form"""
    one, four = BBCovBk(small_st, TERMS, LN), BBCovBk(small_st, TERMS, LN, n_delta=4)
    ks = np.array(small_st.args[1:4])
    closed = np.flatnonzero(ks[1] + ks[2] - ks[0] >= 1.5 * small_st.forecast.s_k * small_st.k_f)
    C1, C4 = (dense_bb(bb).reshape(len(ks[0]), len(LN), len(ks[0]), len(LN)) for bb in (one, four))
    block = np.ix_(closed, range(len(LN)), closed, range(len(LN)))
    np.testing.assert_allclose(C4[block], C1[block], rtol=1e-10, atol=1e-10 * np.abs(C1).max())
    assert not np.allclose(C4, C1, rtol=1e-3)  # the flattened ones move

    pb = PBCov(small_st_pk, four, [0, 2])
    D = [small_st_pk.get_cov_mat([0, 2], n_mu=32), small_st.get_cov_mat(LN, n_mu=16, n_phi=16)]
    cross = np.diag(pb.lam)
    lam = np.block([[np.zeros_like(cross), cross], [cross, four.lam]])
    blocks = [(D[0], pb.V, pb.shell), (D[1], four.U, four.shell)]
    C = dense_joint(blocks, lam, four.n_shell)
    inv = WoodburyInvCov(
        [(np.linalg.inv(np.moveaxis(Db, -1, 0)).transpose(1, 2, 0), U, sh) for Db, U, sh in blocks], lam, four.n_shell
    )
    rng = np.random.default_rng(5)
    d = [rng.normal(size=U.shape[:2][::-1]) + 1j * rng.normal(size=U.shape[:2][::-1]) for _, U, _ in blocks]
    ref = np.concatenate([x.T.ravel() for x in d]).conj() @ np.linalg.solve(C, np.concatenate([x.T.ravel() for x in d]))
    assert inv.contract(d, d) == pytest.approx(ref, rel=1e-8)
