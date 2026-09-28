"""BB term (and its collapsed PT partner) of the bispectrum covariance - covariances.BBCovBk and WoodburyInvCov.

Checked against a brute-force sum over explicit 3D triangle orientations (B in its original labelling),
a lattice count of triangle pairs sharing a Fourier mode, and the dense inverse.
"""

import pickle

import numpy as np
import pytest
from scipy.special import eval_legendre

import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast
from cosmo_wap.forecast.core import contract, joint_inv_cov
from cosmo_wap.forecast.covariances import BBCovBk, PBCov, PTTreeCovBk, WoodburyInvCov
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import bk as nbk
from cosmo_wap.numeric_mu import pk as npk

LN = [0, 1, 2, 3]
TERMS = ["N", "LP"]
COMBOS = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]


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
def small_st(small_st_forecast):
    return small_st_forecast.get_bk_bin(0, cov_terms=TERMS)


@pytest.fixture(scope="module")
def small_mt(small_mt_forecast):
    return small_mt_forecast.get_bk_bin(0, cov_terms=TERMS, all_tracer=True)


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
    def B_tot(fc, tracers, kk, mu1, cphi):
        k1, k2, k3 = kk
        cf = fc.cf_mat_bk[tracers[0]][tracers[1]][tracers[2]]
        zz = fc.z_mid
        theta = utils.get_theta(k1, k2, k3)
        phi = np.arccos(np.clip(cphi, -1, 1))
        B = np.diagonal(nbk.get_mu_phi(mu1, phi, TERMS, TERMS, TERMS, cf, k1, k2, k3, theta, zz))
        mus = nbk.los_cosines(mu1, phi, k1, k2, k3, theta)
        for i, j, c in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
            if tracers[i] == tracers[j]:
                Z = nbk.get_Z1(TERMS, cf, zz, -mus[c], kk[c], ti=i) * nbk.get_Z1(TERMS, cf, zz, mus[c], kk[c], ti=c)
                B = B + Z * BaseInt(cf).pk(kk[c], zz) / cf.survey[i].n_g(zz)
        if tracers[0] == tracers[1] == tracers[2]:
            B = B + 1 / cf.survey[0].n_g(zz) ** 2
        return B

    def g(self, fc, kk, c, sign, tracers, l, mu_e, n_psi=32):
        """<sqrt(4pi(2l+1)) L_l(mu_1) B_tot> over rotations about side c, whose vector is sign*k_c*e, e.z = mu_e"""
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
        B = self.B_tot(fc, tracers, kk, mu[0], cphi)
        return np.mean(np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu[0]) * B)

    def brute(self, fc, bb, t1, ci1, l1, t2, ci2, l2, swap, sign, shot=False, n_mu=16):
        """sum over shared pairs of (1/N_k) int dOmega_k/(4pi) <(4pi)^2 Y_l1 Y_l2 B_T1 conj(B_T2)>, T2 on sign*k.
        Exact BB: -k and the tracers on the shared sides swapped. With PT: BB and PT (+k) on own tracers, and shot
        the shot noise of PT's pairing P - x (1 + 1/(n P_gal)) where the shared sides have the same tracer"""
        ks = np.array(fc.args[1:4])
        mu_e, w = np.polynomial.legendre.leggauss(n_mu)
        dk = fc.forecast.s_k * fc.k_f
        tot = 0
        for c1 in range(3):
            for c2 in range(3):
                if bb.shell[c1, t1] != bb.shell[c2, t2]:
                    continue
                tr1 = list(COMBOS[ci1])
                tr2 = list(COMBOS[ci2])
                if swap:
                    tr1[c1], tr2[c2] = COMBOS[ci2][c2], COMBOS[ci1][c1]
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
                    * self.g(fc, ks[:, t1], c1, 1, tr1, l1, m)
                    * np.conj(self.g(fc, ks[:, t2], c2, sign, tr2, l2, m))
                    for m, wi in zip(mu_e, w)
                )
                tot += A / (4 * np.pi * k**2 * dk / fc.k_f**3)
        return tot

    @pytest.mark.parametrize("pt", [False, True])
    def test_matches_brute_force(self, small_mt, pt):
        fc = small_mt
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
            ci1, ci2 = rng.integers(0, len(COMBOS), 2)
            got, ref = [], []
            for l1, l2 in [(0, 0), (1, 2), (3, 1)]:
                got.append(bb_element(bb, t1, LN.index(l1) * 4 + ci1, t2, LN.index(l2) * 4 + ci2))
                args = (fc, bb, t1, ci1, l1, t2, ci2, l2)
                if pt:
                    ref.append(
                        self.brute(*args, swap=False, sign=-1) + self.brute(*args, swap=False, sign=1, shot=True)
                    )
                else:
                    ref.append(self.brute(*args, swap=True, sign=-1))
            # relative to the pair's largest element - some vanish by symmetry. 1/P_gal is not polynomial in mu, so
            # PT's shot noise is not exact at n_mu=12
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
    def bk_tot(self, tracers, legs, theta, mus):
        return np.ones_like(mus[1], dtype=np.complex128)  # mus[0] is the bare mu nodes


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
def test_bb_pt_positive_semidefinite(fixture, request):
    """BB alone is indefinite (odd l) - with PT it is a covariance, multi-tracer included"""
    fc = request.getfixturevalue(fixture)
    ev = np.linalg.eigvalsh(dense_bb(BBCovBk(fc, TERMS, LN, pt=False)))
    assert ev.min() < -0.1 * ev.max()
    ev = np.linalg.eigvalsh(dense_bb(BBCovBk(fc, TERMS, LN)))
    assert ev.min() > -1e-12 * ev.max()


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
    """cov_ng switches the precompute to WoodburyInvCov, and adding a positive semi-definite term lowers the SNR"""
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


class PairResponseB(BBCovBk):
    """B -> 2 Z1(k_w) P(k_w) R_xy: the part carrying P of the shared side, which the s channel factorises into"""

    def bk_tot(self, tracers, legs, theta, mus):
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
    The t and u channels only to quadrature error: swapping puts each triangle on the other's psi grid"""
    pt = PTTreeCovBk(small_mt, TERMS, LN, n_psi=4, channels="tu")  # cheap - only blocks() is used, on a finer grid
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


def test_pb_matches_brute_force(small_mt, small_mt_pk):
    """Independent of PBCov's algebra: in the frame of the P mode q, the triangle sits on +q (P^{a t}, t -> b) or
    on -q (P^{t b}(q), t -> a) - PBCov writes both on the triangle's side, with (-1)^l. The swap t -> b is the
    squeezed-limit one, Z1^b/Z1^t on the shared side"""
    fc, pk_fc = small_mt, small_mt_pk
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

    def Z(x, mu, k):
        return nbk.get_Z1(TERMS, pk_fc.cf_mat[x][x], zz, np.array([mu]), np.array([k]))[0]

    checked = 0
    for t in rng.permutation(np.flatnonzero(closed)):
        sides = [i for i in range(3) if bb.shell[i, t] < bb.n_shell]
        kb = np.flatnonzero(pb.shell[0] == bb.shell[sides[0], t])
        if checked == 4 or not len(kb):
            continue
        kb = kb[0]
        k = kk[kb]
        for rP, (l, (a, b)) in enumerate(rows):
            ci, l2 = rng.integers(0, len(COMBOS)), LN[rng.integers(0, len(LN))]
            ref = 0
            for i in range(3):
                if bb.shell[i, t] != pb.shell[0, kb]:
                    continue
                own = list(COMBOS[ci])
                ti = own[i]
                for m, wm in zip(mu_q, w_q):
                    g_plus = Z(b, m, k) / Z(ti, m, k) * geom.g(fc, ks[:, t], i, 1, own, l2, m)  # shared side on +q
                    g_minus = Z(a, -m, k) / Z(ti, -m, k) * geom.g(fc, ks[:, t], i, -1, own, l2, m)  # on -q
                    leg = (2 * l + 1) * eval_legendre(l, m)
                    ref += 0.5 * wm * leg * (P(a, ti, m, k) * np.conj(g_plus) + P(ti, b, m, k) * np.conj(g_minus))
            ref /= 4 * np.pi * k**2 * dk / fc.k_f**3
            got = pb_element(pb, bb, kb, rP, t, LN.index(l2) * len(COMBOS) + ci)
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
