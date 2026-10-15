"""Classes for computing the full covariance matrix for the power spectrum and bispectrum multipoles.
This is done by doing the full mu integral over the relevant expressions for the covariance matrix elements. Multi-tracer is supported."""

import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as mtransforms
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.linalg import lu_factor, lu_solve
from scipy.special import eval_legendre, sph_harm_y

import cosmo_wap.pk as pk
from cosmo_wap.forecast import triangle_bins
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import bk as numeric_mu_bk
from cosmo_wap.numeric_mu import pk as numeric_mu_pk
from cosmo_wap.numeric_mu.utils import split_kernels

__all__ = ["bk_placements", "FullCovPk", "FullCovBk", "BBCovBk", "WoodburyInvCov", "PBCov", "PTTreeCovBk"]


_SIGMA_UNSET = object()


def bk_placements(all_tracer):
    """Tracer on each leg (k1 >= k2 >= k3) for each bispectrum row. all_tracer=True (or 4) is FFF, FFB, FBB, BBB with
    the second tracer on the shortest sides, 8 every placement - on a scalene triangle FFB, FBF, BFF are different
    observables. On equal sides some coincide (k2=k3: FFB and FBF, with (-1)^m for m > 0) - the covariance is then
    singular in those directions and the pseudoinverse drops them, see Forecast.invert_matrix"""
    if not all_tracer:
        return [(0, 0, 0)]
    if all_tracer is True or all_tracer == 4:
        return [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
    if all_tracer == 8:
        return [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    raise ValueError(f"all_tracer is True (or 4) or 8 for the bispectrum - got {all_tracer!r}")


def _as_lm(l):
    """(l, m) from a multipole given as l or (l, m)"""
    return tuple(l) if isinstance(l, (tuple, list)) else (l, 0)


def _los_phi(mu_a, mu_b, cos_ab, fallback):
    """LOS azimuth in [0, pi] about k_a from the plane of k_a, k_b - on k_b's side, as the signal's phi (see
    FullCovBk.re_ylm). Flat triangles have no plane - and no m != 0 signal - so they take fallback's cos(phi)"""
    num = mu_b - mu_a * cos_ab
    sin = np.sqrt(np.clip((1 - mu_a**2) * (1 - cos_ab**2), 0, None))
    num, sin, cos_phi = np.broadcast_arrays(num, sin, fallback)
    cos_phi = np.divide(num, sin, out=cos_phi.astype(float), where=sin > 0)
    return np.arccos(np.clip(cos_phi, -1, 1))


def _flat(ks):
    """Flat triangles - one side the sum of the other two. With no plane there is no LOS azimuth, so their m != 0
    rows get zero weight in every covariance term (null rows for invert_matrix). Otherwise each term picks its own
    phi - FullCovBk's uniform, BBCovBk's 0 - and these rows, with no signal, pass for noise templates of the rest"""
    k1, k2, k3 = ks
    return np.isclose(k1, k2 + k3) | np.isclose(k2, k1 + k3) | np.isclose(k3, k1 + k2)


def shell_modes(kk, dk, bb):
    """x = dk delta/k on bb's delta nodes across each shell (n_delta, N_k), and S2 = sum w (1 + x)^2 = 1 + (dk/k)^2/12
    the shell's modes over the thin-shell count 4 pi k^2 dk - exact, as the quadratic is for Gauss-Legendre.
    The modes of a node go as K^2 = k^2 (1 + x)^2, the triangles on each of them as closure_profile/K"""
    x = dk * bb.delta[:, np.newaxis] / kk
    return x, np.sum(bb.w_delta[:, np.newaxis] * (1 + x) ** 2, axis=0)


# so could create a base Cov class - but there is not a huge amount of overlap - but perhaps for cross-PkBK
class FullCovPk:
    def __init__(self, fc, cf_mat, cov_terms, sigma=None, n_mu=64, fast=False, nonlin=False, kernels=True):
        """
        Does full (multi-tracer) multipole covariance for given terms in a single redshift bin.
        Takes in PkForecast object.
        Do numerical mu integrals over regular expressions to get everything we need!"""
        self.fc = fc
        self.terms = cov_terms
        self.sigma = sigma
        self.nonlin = nonlin
        self.kernels = kernels  # work directly with kernels or just P(k, mu)
        self.window = fc.bin_window()  # integrated terms as the bin sees them - see numeric_mu.pk.get_int_window

        self.cf_mat = cf_mat

        nodes, self.weights = utils.leggauss(n_mu)  # legendre gauss - get nodes and weights for given n
        nodes = np.real(nodes)
        # it cancels when both l of a pair have the same parity - a mixed pair kept the half-integral
        # and put C(1,2) at 61% of the largest entry. Same node count either way, so no cheaper.
        if fast:  # only go from 0,1 and use symmetry - cut mu integral in half - just need to know when it cancels!
            self.mu = (1) * (nodes + 1) / 2.0  # sample mu range [0,1]
        else:
            self.mu = nodes  # sample mu range [-1,1] - so this is the natural gauss legendre range!

        # make k,z broadcastable # z will always be a float tbh
        cosmo_funcs, kk, self.zz = fc.args
        self.kk_shape = len(kk)
        kk = kk[:, np.newaxis]
        self.args = (cosmo_funcs, kk, self.zz)

        # basically we dont have an amazing system of including nonlinear effects in the covariance
        # (bispectrum is different and not yet implemented under the same method)
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            initial_state = cf_mat[0][0].nonlin
            for row in cf_mat:
                for cf in row:
                    if cf:  # entries can be empty - see create_cache
                        cf.nonlin = True

        self.create_cache(*self.args)

        if nonlin:
            for row in cf_mat:
                for cf in row:
                    if cf:
                        cf.nonlin = initial_state

    def get_cov(self, ln, sigma=_SIGMA_UNSET):
        """Gets full covariance matrix. Omitted sigma keeps the constructor setting; None disables FoG."""
        if sigma is not _SIGMA_UNSET:
            self.sigma = sigma

        if self.fc.all_tracer:
            ll_cov = self.get_multi_tracer(self.terms, ln)  # full covariance matrix
        else:
            # simple single tracer case:
            ll_cov = np.zeros((len(ln), len(ln), self.kk_shape), dtype=np.complex128)

            for i in range(len(ln)):
                for j in range(i, len(ln)):
                    ll_cov[i, j] = self.get_single_tracer_ll(self.terms, ln[i], ln[j])
                    if i != j:  # only need to compute top half!
                        ll_cov[j, i] = np.conjugate(ll_cov[i, j])

        self.cov = ll_cov  # caching dont hurt
        return ll_cov

    def create_cache(self, *args, **kwargs):
        """Store all Pks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store the total P(k,mu) for each tracer combination
        | XX XY |
        | YX YY | where YX = np.conjugate(XY) in this case"""

        N = len(self.cf_mat)
        self.pk_cache = [[None for _ in range(N)] for _ in range(N)]

        for i in range(N):
            for j in range(N):
                if not self.kernels:  # each analytic term is already a full contribution to P(k, mu)
                    self.pk_cache[i][j] = sum(
                        getattr(pk, term).mu(self.mu, self.cf_mat[i][j], *args[1:], **kwargs) for term in self.terms
                    )
                else:  # kernels sum before squaring - so the cross terms, e.g. <N LP*>, are included
                    self.pk_cache[i][j] = numeric_mu_pk.get_mu_window(
                        self.mu,
                        list(self.terms),
                        list(self.terms),
                        self.cf_mat[i][j],
                        *args[1:],
                        self.window,
                        **kwargs,
                    )  # so can change to new mechanism -new int

        # the hermitian part, P^ij -> (P^ij + P^ji*)/2: P^XX real and P^YX = P^XY*, as get_tracer takes them. Only the
        # integrated terms move - get_mu's LOS sits on the first field, so IS != SI* - this averages the two LOS
        P = self.pk_cache
        self.pk_cache = [[(P[i][j] + np.conjugate(P[j][i])) / 2 for j in range(N)] for i in range(N)]

    def integrate_mu(self, i1, i2, j1, j2, terms, l1, l2, mu):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer t1=t2=t3=t4=0 (i.e. P_XX P_XX)
        For say: P_XY P_XX t1=t2=t4=0;t3=1 - P_t1t3 P_t2t4
        """
        coef = (2 * l1 + 1) * (2 * l2 + 1) * eval_legendre(l1, self.mu) * eval_legendre(l2, mu) * self.weights

        fog = 1 if self.sigma is None else np.exp(-((self.args[1] * self.mu) ** 2) * self.sigma**2)  # as numeric_mu.pk

        # add shot noise - is zero in XY case
        a = self.pk_cache[i1][i2] * fog + 1 / self.cf_mat[i1][i2].n_g(self.zz)
        b = self.pk_cache[j1][j2] * fog + 1 / self.cf_mat[j1][j2].n_g(self.zz)
        return np.sum(coef * a * np.conjugate(b), axis=(-1))  # sum over last axis - mu

    def get_tracer(self, a, b, c, d, terms, l1, l2):
        """Get C[P^ab_{l}, P^cd_{l2}](k)
        C[P^ab_{l1}, P^cd_{l2}](k) = ((2*l1 + 1)(2*l2 + 1) / N_k) ( Int (d(Omega_k) / 4*pi) * L_1(mu) *
                                        [L_2(mu)*P^ac(k,mu)*P^bd(k,mu)^* + L_2(-mu)*P^ad(k,mu)*P^bc(k,mu)^*]"""

        return (1 / 2) * (
            self.integrate_mu(a, c, b, d, terms, l1, l2, self.mu)
            + self.integrate_mu(a, d, b, c, terms, l1, l2, -self.mu)
        )

    def get_single_tracer_ll(self, terms, l1, l2):
        """Get full single-tracer covariance for multipole pair"""
        if len(self.cf_mat) > 1:  # then we have XY term
            return self.get_tracer(0, 1, 0, 1, terms, l1, l2)
        return self.get_tracer(0, 0, 0, 0, terms, l1, l2)

    def get_multi_tracer(self, terms, ln):
        """Now compute full matrix:

        Get full multi-tracer matrix for multipole pair:
        C(Pi, Pj) = │ C[P_li^XX, P_lj^XX]   C[P_li^XY, P_lj^XX]   C[P_li^YY, P_lj^XX] │
                    │ C[P_li^XX, P_lj^XY]   C[P_li^XY, P_lj^XY]   C[P_li^YY, P_lj^XY] │
                    │ C[P_li^XX, P_lj^YY]   C[P_li^XY, P_lj^YY]   C[P_li^YY, P_lj^YY] │

        So only l_odd x l_even thing are imaginary - the rest are purely real after mu integration
        """

        # find shape of covariance matrix: (len(data_vector),len(data_vector))
        length = 0
        for l in ln:
            if l & 1:  # if odd
                length += 1
            else:
                length += 3

        cov_mt = np.zeros((length, length, self.kk_shape), dtype=np.complex128)  # create empty complex array

        # lets build our covariance matix!
        # so first we loop over l and then over tracers
        # even multipoles have tracers XX,XY,YY but odd just have XY

        # keep track of row and column of each submatrix
        row = 0
        column = 0

        tt = [(0, 0), (0, 1), (1, 1)]  # XX,XY,YY
        for _, li in enumerate(ln):
            for _, lj in enumerate(ln):
                if li & 1:  # binary operator to specify odd
                    tracer = [tt[1]]
                else:
                    tracer = tt

                if lj & 1:
                    tracer2 = [tt[1]]
                else:
                    tracer2 = tt

                # now loop over tracers - k1,k2 keep track of where we are in this submatrix
                for k1, t1 in enumerate(tracer):
                    for k2, t2 in enumerate(tracer2):
                        cov_mt[row + k1, column + k2] = self.get_tracer(*t1, *t2, terms, li, lj)  # get matrix element

                # update what bit of covariance is being calculated - overarching (not on the level of the submatrices)
                column += len(tracer2)
            row += len(tracer)
            column = 0

        return cov_mt

    def plot_cov(
        self,
        ln,
        kn=0,
        real=True,
        log=True,
        vmin=None,
        vmax=None,
        cmap="RdBu",
        lnrwidth=None,
        figsize=(10, 7),
        rotation=0,
        **kwargs,
    ):
        """Lets plot the covariance"""
        if hasattr(self, "cov"):
            cov = self.cov
        else:
            cov = self.get_cov(ln)

        labels = []
        for l in ln:
            if l & 1:
                labels.append(rf"$P^{{\rm BF}}_{l}$")
            else:  # If even
                labels.extend([rf"$P^{{\rm BB}}_{{{l}}}$", rf"$P^{{\rm BF}}_{{{l}}}$", rf"$P^{{\rm FF}}_{{{l}}}$"])

        if log and not lnrwidth:  # for regular log plots set zero value to white
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_under("white")  # You can also use 'white', '#dddddd', etc.

        plt.figure(figsize=figsize)
        if real:
            if log:
                if not vmin:
                    vmin = np.abs(cov[..., kn].real).min()
                if not vmax:
                    vmax = np.abs(cov[..., kn].real).max()

                if lnrwidth:
                    plt.pcolormesh(
                        cov[..., kn].real,
                        cmap=cmap,
                        norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax),
                    )
                else:
                    plt.pcolormesh(np.abs(cov[..., kn].real), cmap=cmap, norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[..., kn].real, cmap=cmap)
        else:
            if log:
                if not vmin:
                    vmin = np.abs(cov[..., kn].imag).min()
                if not vmax:
                    vmax = np.abs(cov[..., kn].imag).max()

                if lnrwidth:
                    plt.pcolormesh(
                        cov[..., kn].imag,
                        cmap=cmap,
                        norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax),
                    )
                else:
                    plt.pcolormesh(np.abs(cov[..., kn].imag), cmap=cmap, norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[..., kn].imag, cmap=cmap)

        ha = "right" if rotation else "center"
        plt.xticks(
            np.arange(0.5, len(labels) + 0.5),
            labels=labels,
            rotation=rotation,
            ha=ha,
            rotation_mode="anchor",
        )
        if rotation:
            ax = plt.gca()
            offset = mtransforms.ScaledTranslation(15 / 72, 0, ax.figure.dpi_scale_trans)
            for lbl in ax.get_xticklabels():
                lbl.set_transform(lbl.get_transform() + offset)
        plt.yticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        cbar = plt.colorbar()
        cbar.set_label(r"$|C[P^{ab}_{\ell_i},P^{cd}_{\ell_j}](k)|$", **kwargs)
        return cbar


class FullCovBk:
    def __init__(self, fc, cf_mat, cov_terms, sigma=None, n_mu=64, n_phi=32, fast=False, nonlin=False, kernels=True):
        """
        Does full (multi-tracer) multipole covariance for given terms in a single redshift bin.
        Takes in BkForecast object.
        Do numerical mu integrals over regular expressions to get everything we need!
        Same as Pk but now for bispectrum!
        Covariance has shape [k1,k2,k3,mu,phi]
        On the triangles of fc.cov_args (the effective ones, or the centres with bin_avg=False - see
        forecast.triangle_bins), averaged to the bins in get_cov - the equal sides of the permutation terms and the
        flat m != 0 rows are the bin centres'"""
        self.fc = fc
        self.terms = cov_terms
        self.sigma = sigma
        self.nonlin = nonlin
        self.kernels = kernels  # work directly with kernels or just Bk
        self.window = fc.bin_window()  # integrated terms as the bin sees them - see numeric_mu.pk.get_int_window

        self.cf_mat = cf_mat

        nodes, weights_mu = utils.leggauss(n_mu)  # legendre gauss - get nodes and weights for given n
        # even-mu integrand only, as in FullCovPk
        if fast:  # only go from 0,1 and use symmetry - cut mu integral in half - just need to know when it cancels!
            mu = (1) * (nodes + 1) / 2.0  # sample mu range [0,1]
        else:
            mu = nodes  # sample mu range [-1,1] - so this is the natural gauss legendre range!

        # for phi
        nodes, weights_phi = utils.leggauss(n_phi)  # legendre gauss - get nodes and weights for given n
        phi = (2 * np.pi) * (nodes + 1) / 2.0  # sample mu range [0,2 *np.pi]
        self.weights = weights_mu[:, np.newaxis] * weights_phi  # 2D GL weights

        # make k1,k2,k3,z broadcastable
        _, k1, k2, k3, _, self.zz = fc.cov_args
        self.centre = fc.k_centre[:, fc.cov_tri.bins, np.newaxis, np.newaxis]  # each triangle's bin centre, as ks
        mu = mu[:, np.newaxis]
        k1, k2, k3 = utils.enable_broadcasting(
            k1, k2, k3, n=2
        )  # if arrays add newaxis at the end so is broadcastable with mu!

        # lets define some bispectrum stuff
        _, theta = utils.get_theta_k3(k1, k2, k3, None)

        mu2 = mu * np.cos(theta) + np.sqrt(1 - mu**2) * np.sin(theta) * np.cos(phi)
        mu3 = -(mu * k1 + mu2 * k2) / k3
        self.mus = mu, mu2, mu3
        self.phi = phi
        self._ylm_cache = {}  # re_ylm by (l, m, legs)

        self.ks = np.array([k1, k2, k3])
        self.N_tri = len(k1)  # number of triangles evaluated (fc.cov_args) - k1,k2,k3 are flattened to this shape

        # basically we dont have an amazing system of including nonlinear effects in the covariance
        # (bispectrum is different and not yet implemented under the same method)
        # so now whether they use the halofit pk it is defined by the cosmo_funcs attribute so we just turn it off and on again if we need to
        if nonlin:
            initial_state = cf_mat[0][0].nonlin
            for i in range(len(cf_mat)):
                for j in range(len(cf_mat[i])):
                    cf_mat[i][j].nonlin = True

        self.create_cache()

        if nonlin:
            for i in range(len(cf_mat)):
                for j in range(len(cf_mat[i])):
                    cf_mat[i][j].nonlin = initial_state

    def get_cov(self, ln):  # is the same - so could create a parent class.
        """Gets full covariance matrix"""

        if self.fc.all_tracer:
            ll_cov = self.get_multi_tracer(self.terms, ln)  # full covariance matrix
        else:
            # simple single tracer case:
            ll_cov = np.zeros((len(ln), len(ln), self.N_tri), dtype=np.complex128)

            for i in range(len(ln)):
                for j in range(i, len(ln)):
                    ll_cov[i, j] = self.get_single_tracer_ll(self.terms, ln[i], ln[j])
                    if i != j:  # only need to compute top half!
                        ll_cov[j, i] = np.conjugate(ll_cov[i, j])

        ll_cov = self.fc.cov_tri.mean(ll_cov)  # triangles -> bins
        self.cov = ll_cov  # caching dont hurt
        return ll_cov

    def create_cache(self, **kwargs):
        """Store all Bks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store the total P(k,mu) for each tracer combination

        | XX XY XZ |
        | YX YY YZ |
        | ZX ZY ZZ | where YX = np.conjugate(XY) in this case - all n^2 pairs, then the hermitian part
        However the standard case will be working with 2 tracers - like the power spectrum

        also store for k1,k2,k3 etc
        so shape probably 2,2,3,N
        We also now do it with numerical mu-int
        """

        N = len(self.cf_mat)
        self.pk_cache = [[[None for _ in range(N)] for _ in range(N)] for _ in range(3)]

        for i in range(N):
            for j in range(N):
                for ki in range(3):
                    if not self.kernels:  # then get from mathematica expressions - each already a full contribution
                        self.pk_cache[ki][i][j] = sum(
                            getattr(pk, term).mu(self.mus[ki], self.cf_mat[i][j], self.ks[ki], self.zz, **kwargs)
                            for term in self.terms
                        )
                    else:  # kernels sum before squaring - so the cross terms, e.g. <N LP*>, are included
                        self.pk_cache[ki][i][j] = numeric_mu_pk.get_mu_window(
                            self.mus[ki],
                            list(self.terms),
                            list(self.terms),
                            self.cf_mat[i][j],
                            self.ks[ki],
                            self.zz,
                            self.window,
                        )

        P = self.pk_cache  # the hermitian part - see FullCovPk.create_cache
        self.pk_cache = [
            [[(P[ki][i][j] + np.conjugate(P[ki][j][i])) / 2 for j in range(N)] for i in range(N)] for ki in range(3)
        ]

    def re_ylm(self, lm, legs=(0, 1)):
        """Re Y_lm of the LOS in the frame of the triangle whose k1, k2 are legs (a, b) of this one - z along k_a and
        k_b in the xz-plane at positive x, as (mu, phi) of the signal. Re Y_lm is the positive-m estimator: B(mu,phi)
        is even in phi (see numeric_mu.bk.project_multipole), so it averages B_lm and (-1)^m B_l-m, same mean with
        the noise of both. Only cos(phi) is needed, but the sign of x matters - swapping k2,k3 sends phi -> pi - phi"""
        l, m = _as_lm(lm)
        a, b = legs
        key = (l, m, a) if m == 0 else (l, m, a, b)
        if key not in self._ylm_cache:
            if m == 0:
                phi = 0
            elif legs == (0, 1):
                phi = self.phi
            else:
                ks = self.ks
                c = 3 - a - b
                cos_ab = (ks[c] ** 2 - ks[a] ** 2 - ks[b] ** 2) / (2 * ks[a] * ks[b])  # k_a.k_b, closed triangle
                phi = _los_phi(self.mus[a], self.mus[b], cos_ab, np.cos(self.phi))
            ylm = np.real(sph_harm_y(l, m, np.arccos(self.mus[a]), phi))
            self._ylm_cache[key] = ylm if m == 0 else np.where(_flat(self.centre), 0, ylm)  # see _flat
        return self._ylm_cache[key]

    def integrate_mu(self, i1, i2, j1, j2, k1, k2, terms, l1, l2, legs=(0, 1)):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer i1=i2=j1=j2=k1=k2=0 (i.e. P_XX P_XX P_XX)
        For say: P_XY P_XX P_XX -> i1=j1=j2=k1=k2=0;j1=1
        l1, l2 are l or (l, m); legs are the legs of this triangle the second's k1, k2 pair with - see get_tracer
        """
        coef = 4 * np.pi * self.re_ylm(l1) * self.re_ylm(l2, legs) * self.weights

        # FOG - one factor per k, each P's two fields (see numeric_mu.pk)
        if self.sigma is None:
            fog = (1, 1, 1)
        else:
            fog = tuple(np.exp(-((self.ks[i] * self.mus[i]) ** 2) * self.sigma**2) for i in range(3))

        # add shot noise - is zero in XY case
        a = self.pk_cache[0][i1][i2] * fog[0] + 1 / self.cf_mat[i1][i2].n_g(self.zz)
        b = self.pk_cache[1][j1][j2] * fog[1] + 1 / self.cf_mat[j1][j2].n_g(self.zz)
        c = self.pk_cache[2][k1][k2] * fog[2] + 1 / self.cf_mat[k1][k2].n_g(self.zz)
        return (2 * np.pi) / 2.0 * np.sum(coef * a * b * c, axis=(-2, -1))  # sum over last 2 axes - mu and phi

    def get_single_tracer_ll(self, terms, l1, l2):
        """Get single-tracer covariance for multipole pair"""
        if len(self.cf_mat) > 1:  # then we have XY term
            return self.get_tracer(0, 1, 0, 0, 1, 0, terms, l1, l2)
        return self.get_tracer(0, 0, 0, 0, 0, 0, terms, l1, l2)  # degenerate triangles as multi-tracer, not s123

    def get_tracer(self, a, b, c, d, e, f, terms, l1, l2):
        """Get C[B^abc_{l}, B^def_{l2}](k) - i.e. PPP term to bispectrum covariance
        C[B^abc_{l}, B^def_{l2}](k1,k2,k3) = ( Int (d(Omega_k) / 4*pi) * Y_l1m1(mu,phi) *Y_l2m2(mu,phi)
                                        [P^ad(k1,mu)*P^be(k2,mu2)*P^cf(k3,mu3)]
        only compute unique (perm, legs) combinations
        """
        l2, m2 = _as_lm(l2)

        perms = list(itertools.permutations([d, e, f]))
        perms_index = list(itertools.permutations(["d", "e", "f"]))

        # [(d, e, f), (d, f, e), (e, d, f), (e, f, d), (f, d, e), (f, e, d)]
        # deduplicate on (tracer_perm, legs) - q1 goes to k_legs[0] and q2 to k_legs[1]
        def key(i):
            legs = (perms_index[i].index("d"), perms_index[i].index("e"))
            if l2 == 0:
                return perms[i], None  # we can ignore d placement for monopole
            return perms[i], legs if m2 else legs[0]  # and e placement for m=0

        results = {}
        for i, perm in enumerate(perms):
            if key(i) not in results:
                legs = (perms_index[i].index("d"), perms_index[i].index("e"))
                results[key(i)] = self.integrate_mu(a, perm[0], b, perm[1], c, perm[2], terms, l1, (l2, m2), legs)

        # build ordered list matching the original 6 permutations
        cov_list = [results[key(i)] for i in range(6)]

        cov_tot = cov_list[0]
        # now for equilateral and isoceles triangles - we have addtional terms from dirac-deltas - rememeber k1\geq k2\geq k3
        k1, k2, k3 = self.centre[..., 0, 0]  # so shape kk - the bin's, the evaluated triangle's need not be equal
        cov_tot = np.where(k2 == k3, cov_tot + cov_list[1], cov_tot)  # k2=k3
        cov_tot = np.where(k1 == k2, cov_tot + cov_list[2], cov_tot)  # k1=k2
        # equilateral - sum the rest where k1=2=k3
        cov_tot = np.where((k1 == k2) & (k2 == k3), cov_tot + np.add.reduce(cov_list[3:]), cov_tot)

        return cov_tot

    def get_multi_tracer(self, terms, ln):
        """Now compute full matrix:

        Get full multi-tracer matrix for multipole pair:
        C(Bli, Blj) = │ C[B_li^xXX, B_lj^xXX]   C[B_li^XXY, B_lj^xXX]   C[B_li^XYY, B_lj^xXX]   C[B_li^YYY, B_lj^xXX] │
                      │ C[B_li^xXX, B_lj^XXY]   C[B_li^XXY, B_lj^XXY]   C[B_li^XYY, B_lj^XXY]   C[B_li^YYY, B_lj^XXY] │
                      │ C[B_li^xXX, B_lj^XYY]   C[B_li^XXY, B_lj^XYY]   C[B_li^XYY, B_lj^XYY]   C[B_li^YYY, B_lj^XYY] │
                      │ C[B_li^xXX, B_lj^YYY]   C[B_li^XXY, B_lj^YYY]   C[B_li^XYY, B_lj^YYY]   C[B_li^YYY, B_lj^YYY] │

        So only l_odd x l_even thing are imaginary - the rest are purely real after mu integration
        Shape [4xln,4xln] - or 8 with all_tracer=8, see bk_placements
        Exploit overall hermitian symmetry
        """
        tracers = self.fc.placements
        nl = len(ln)
        nt = len(tracers)
        cov_mt = np.zeros((nt * nl, nt * nl, self.N_tri), dtype=np.complex128)  # create empty complex array

        # lets build our covariance matix!
        # so first we loop over l and then over tracers
        for i, li in enumerate(ln):
            for j, lj in enumerate(ln):
                # now loop over tracers - k1,k2 keep track of where we are in this submatrix
                for k1, t1 in enumerate(tracers):
                    for k2, t2 in enumerate(tracers):
                        row = i * nt + k1  # keep track of which submatrix
                        col = j * nt + k2
                        if col < row:
                            continue  # only upper triangle of full matrix
                        val = self.get_tracer(*t1, *t2, terms, li, lj)  # get matrix element
                        cov_mt[row, col] = val
                        if row != col:
                            cov_mt[col, row] = np.conjugate(val)

        return cov_mt

    def plot_cov(
        self,
        ln,
        kn=0,
        real=True,
        log=True,
        vmin=None,
        vmax=None,
        cmap="RdBu",
        lnrwidth=None,
        figsize=(10, 7),
        rotation=0,
        **kwargs,
    ):
        """Lets plot the covariance"""
        if hasattr(self, "cov"):
            cov = self.cov
        else:
            cov = self.get_cov(ln)

        labels = []
        for l in ln:
            labels.extend(
                [
                    rf"$B^{{\rm BBB}}_{{{l}}}$",
                    rf"$B^{{\rm BBF}}_{{{l}}}$",
                    rf"$B^{{\rm BFF}}_{{{l}}}$",
                    rf"$B^{{\rm FFF}}_{{{l}}}$",
                ]
            )

        if log and not lnrwidth:  # for regular log plots set zero value to white
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_under("white")  # You can also use 'white', '#dddddd', etc.

        plt.figure(figsize=figsize)
        if real:
            if log:
                if not vmin:
                    vmin = np.abs(cov[..., kn].real).min()
                if not vmax:
                    vmax = np.abs(cov[..., kn].real).max()

                if lnrwidth:
                    plt.pcolormesh(
                        cov[..., kn].real,
                        cmap=cmap,
                        norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax),
                    )
                else:
                    plt.pcolormesh(np.abs(cov[..., kn].real), cmap=cmap, norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[..., kn].real, cmap=cmap)
        else:
            if log:
                if not vmin:
                    vmin = np.abs(cov[..., kn].imag).min()
                if not vmax:
                    vmax = np.abs(cov[..., kn].imag).max()

                if lnrwidth:
                    plt.pcolormesh(
                        cov[..., kn].imag,
                        cmap=cmap,
                        norm=SymLogNorm(linthresh=lnrwidth, linscale=1, vmin=-vmin, vmax=vmax),
                    )
                else:
                    plt.pcolormesh(np.abs(cov[..., kn].imag), cmap=cmap, norm=LogNorm(vmin, vmax=vmax))

            else:
                plt.pcolormesh(cov[..., kn].imag, cmap=cmap)

        ha = "right" if rotation else "center"
        plt.xticks(
            np.arange(0.5, len(labels) + 0.5),
            labels=labels,
            rotation=rotation,
            ha=ha,
            rotation_mode="anchor",
        )
        if rotation:
            ax = plt.gca()
            offset = mtransforms.ScaledTranslation(15 / 72, 0, ax.figure.dpi_scale_trans)
            for lbl in ax.get_xticklabels():
                lbl.set_transform(lbl.get_transform() + offset)
        plt.yticks(np.arange(0.5, len(labels) + 0.5), labels=labels)
        cbar = plt.colorbar()
        cbar.set_label(r"$|C[B^{abc}_{\ell_i},B^{def}_{\ell_j}](k_1,k_2,k_3)|$", **kwargs)
        return cbar


def _triangle_layout(fc):
    """tracer combinations (rows of the data vector), number of tracers, sides (3, N_tri) and the k-bin of each side"""
    cosmo_funcs, k1, k2, k3, _, _ = fc.args
    if fc.all_tracer:
        combos, n_t = list(fc.placements), 2  # as get_multi_tracer and get_data_vector
    elif cosmo_funcs.multi_tracer:
        raise NotImplementedError("non-Gaussian covariance only for all_tracer or a single tracer")
    else:
        combos, n_t = [(0, 0, 0)], 1

    ks = np.array([k1, k2, k3])
    shell = np.rint(ks / (fc.forecast.s_k * fc.k_f)).astype(int) - 1  # k_bin = (i+1) dk
    return combos, n_t, ks, shell


class BBCovBk:
    def __init__(self, fc, cov_terms, ln, sigma=None, n_mu=12, n_psi=8, pt=True, n_delta=3, n_avg=None):
        """
        BB term of the bispectrum covariance (+ its PT partner), as low-rank factors U Lambda U^dagger - see
        WoodburyInvCov. Takes in BkForecast object.

        The six-point function split into two bispectra, each with two fields of one triangle and one of the other,
        so the triangles share a side (q_c = -p_c'). 9 pairings, nonzero when the two sides are in the same k-shell,
        so it couples different triangles. For each shared pair, in the thin-shell limit:

        C[B^T_l1, B^T'_l2] += k_f^2/(8 pi s_k k^2) Int dmu g^T_l1(mu) conj(g^T'_l2(-mu))

        g_l(mu) = <sqrt(4 pi (2l+1)) L_l(mu_1) B^(N)>_psi: the triangle with LOS cosine mu on the shared side,
        averaged over its rotation psi about that side. In real space this is the usual B_T B_T' / N_k
        (2111.05887 eq 2.30, 2403.08634). Closure: the bin-averaged beta as V123 for each triangle's count, resolved
        across the shared shell by n_delta, and B averaged over the bin's triangles by n_avg (below).

        Shot noise as the catalogue-subtracted estimators: galaxies coinciding within one triangle are subtracted,
        across the two kept. So B^(N) = B + delta/n (P(k_a) + P(k_b)), the other triangle's galaxy on the shared
        side merged with each of the other two - no P(k_c)/n or 1/n^2 (1908.06234 eqs 23-24). Finite as B, P -> 0.

        Separable in mu - so columns are (shell, mu node, T, T'), with T the tracer the row's triangle has on the
        shared side and T' the other triangle's, which each swaps into its own B^(N). Lambda pairs (s, n, T, T')
        with (s, -n, T', T). B's swap is the squeezed limit, (Z1^T'/Z1^T) B with Z1 on the shared side - its ratios
        cancel in each pair (Z1(-mu)* = Z1(mu)), so the clustering part is own-tracer BB. The shot noise's is exact.

        BB alone is indefinite: the -k gives conj(g(-mu)) = (-1)^l2 g(mu), so odd l have negative variance.
        pt adds the collapsed limit of the PT term (P x trispectrum), T -> B B*/P - the same with the other
        triangle on +k. In real space without shot noise it equals BB (the 2BB of 2403.08634). Only the squeezed
        limit of PT - so approximate when the shared side is not the soft one.
        On the shared mode each triangle is delta_T(q) Y(-q), Y its other two fields: BB is the exchange
        <delta Y'><Y delta'> and PT the direct <delta delta'><Y Y'> - a covariance if (delta, Y) has one. With
        delta_X = Z1^X delta_m + eps_X (eps Poisson), <Y delta_X> = Z1^X R + s_X is BB's column (n, T, X): R B's
        clustering response and s_X its shot noise with X on the shared side. The least <Y Y'> is then
        R R'*/P_lin + sum_X n_X s_X s_X'*, so PT = P_tot^{T T'} (R R'*/P_lin + sum_X n_X s_X s_X'*) with the pairing's
        shot noise in P_tot - columns F^{T i} (R/sqrt(P_lin), sqrt(n_X) s_X) after BB's, F F^dagger = P_tot, each paired
        with itself. BB + PT is positive semi-definite by construction, and only the triangles' own tracers enter, so
        an all-X block is the single-tracer one. Without B's shot noise it is the own-tracer pairing with
        P_tot/P_gal = 1 + 1/(n P_gal) on the same tracer (2403.08634 keep that 1/n in C_PT but not in their PT = BB -
        the same when n P >> 1).
        PT's B is B^(N) too. Not derived - T's own shot noise (numeric_mu.bk.get_T_shot) has no collapsed limit - but
        against the tree-level PT with it (PTTreeCovBk) the own-tracer pairing was closer than B alone, which left BB's
        shot noise on odd l uncancelled, so the NG term raised their SNR. Until 2026-10-05 PT was that pairing,
        (s, n, T, T) with (s, n, T', T') - with BB's exact shot noise swap the joint covariance went indefinite for a
        sparse tracer (BGS bright split at z ~ 0.45, k_max 0.15 or all_m).

        The number of a bin's triangles on a mode of the shared shell follows the closure fraction at its |k|, which
        varies across the shell for (nearly) flattened bins - n_delta > 1 repeats the columns for nodes across the
        shell width, each side weighted by its closure fraction there (2111.05887 sec 2.3). Closed bins are unchanged.
        n_delta=1 takes the bin average, off by up to ~25% on flattened pairs. 2111.05887 stress this (sec 2.3,
        App. A): counting flattened and open bins as if every mode closed is off by >~ O(1), so they integrate over
        the overlap of the two closure regions on the shared shell - exactly, for the shared long-wavelength side.
        n_delta -> inf is that, for every side. Default 3: against 8 nodes the NG entries above 1% of the correlation
        are off by up to 24% with n_delta=1, 3% with 2 and < 0.5% with 3 (Euclid 1 < z < 1.2, k_max 0.1, s_k 4,
        single tracer and BF split; 15%, 2% and < 0.5% at s_k 2). But the rank, so the Woodbury setup ~ n_delta^3
        and its memory ~ n_delta^2: 4 is too much for multi-tracer at k_max ~ 0.15 - pass a smaller n_delta there
        through FullForecast(ng_kwargs).

        On each node g is the mean over the bin's triangles with the shared side there - n_avg x n_avg nodes over the
        other two sides where they close, weighted q_a q_b as the count (triangle_bins.bin_nodes), with Z1, P_tot and
        P_lin at the node - and PBCov's P (delta_K). The sides of a bin vary by dk, so with k3 ~ dk its triangles run
        from collinear to perpendicular: on the bin centre (n_avg=0, the default with FullForecast(bin_avg=False)) the
        BB diagonal is ~3.5x the bin average for (i, i-1, 1) bins - the collinear maximum of the squeezed response -
        and ~0.5x for (i, i, 1), within 7% only where every side and k2 + k3 - k1 are >= 4 dk (Euclid 1 < z < 1.2,
        k_max 0.1, s_k 4). Against 8 nodes the NG entries above 1% of the correlation are within ~1% with n_avg=4, 3%
        with 3 and 14% with 2 - so default 4 (n_avg=None, as triangle_bins.OPTIONS; 0 with bin_avg=False). The setup
        is ~ n_delta n_avg^2 times the bin centre's, the rank unchanged.
        """
        ln = [_as_lm(l) for l in fc.multipoles(ln)]  # rows as FullCovBk - Re Y_lm for m > 0, see FullCovBk.re_ylm
        self.zz = fc.args[-1]
        self.cf_mat_bk = fc.cf_mat_bk
        self.terms = split_kernels(cov_terms)[1]  # local - integrated terms are only in the Gaussian part (FullCovBk)
        self.sigma = sigma

        combos, n_t, ks, self.shell = _triangle_layout(fc)
        self.n_shell = self.shell.max() + 1
        self.n_t = n_t
        s_k = fc.forecast.s_k
        dk = s_k * fc.k_f
        if n_avg is None:
            _, _, n_avg = triangle_bins.OPTIONS[fc.bin_avg]

        # B is even in psi (reflection through the plane of the shared side and the LOS) - so [0, pi] as get_mu_phi_grid
        self.mu, w_mu = utils.leggauss(n_mu)
        self.psi = np.pi * (np.arange(n_psi) + 0.5) / n_psi
        delta, w_delta = utils.leggauss(n_delta)  # where the shared mode sits in its shell
        self.delta, self.w_delta = delta / 2, w_delta / 2  # nodes in [-1/2, 1/2] bin widths, weights sum to 1
        # B on each node of the shared side - or the bin centre for all. PBCov takes its P there too
        self.delta_K = self.delta if n_avg else np.zeros(1)
        n_K = len(self.delta_K)

        N_tri = ks.shape[1]
        n_c = len(combos)
        flat = _flat(ks)[:, np.newaxis, np.newaxis]
        b_bb = n_mu * n_t * n_t  # BB's columns, then PT's
        U_all = np.zeros((N_tri, len(ln) * n_c, 3, n_K, b_bb + pt * n_mu * n_t * (1 + n_t)), dtype=np.complex128)
        U = U_all[..., :b_bb].reshape(*U_all.shape[:4], n_mu, n_t, n_t)  # views
        U_pt = U_all[..., b_bb:].reshape(*U_all.shape[:4], n_mu, n_t, 1 + n_t) if pt else None
        n_g = [self.cf_mat_bk[t][t][t].survey[0].n_g(self.zz) for t in range(n_t)]

        # relabel so the shared side is the first leg: mu is then its LOS cosine and phi is psi
        for w, order in enumerate([(0, 1, 2), (1, 0, 2), (2, 0, 1)]):
            for d, delta_d in enumerate(self.delta_K):
                K = ks[w] + dk * delta_d
                nodes = triangle_bins.bin_nodes(K, ks[order[1]], ks[order[2]], dk, n_avg)
                g_B, g_S = self.g_avg(ln, combos, order, *nodes, flat)
                k = K[:, np.newaxis]
                Z = [
                    numeric_mu_bk.get_Z1(self.terms, self.cf_mat_bk[t][t][t], self.zz, self.mu, k) for t in range(n_t)
                ]  # on the shared side
                if pt:  # PT's factors on the shared side, see above
                    ev, vec = np.linalg.eigh(self.p_tot(k))
                    F = vec * np.sqrt(np.clip(ev, 0, None))[..., np.newaxis, :]  # (N_tri, n_mu, T, i), F F^dag = P_tot
                    P_lin = BaseInt(self.cf_mat_bk[0][0][0]).pk(k, self.zz)
                    if sigma is not None:
                        P_lin = P_lin * np.exp(-((k * self.mu) ** 2) * sigma**2)  # Z1 R is the damped field's

                for ci, combo in enumerate(combos):
                    own = tuple(combo[i] for i in order)
                    for li in range(len(ln)):
                        gB = g_B[own, li] / ks[w][:, np.newaxis]
                        gS = [g_S[(t2, *own[1:]), li] / ks[w][:, np.newaxis] for t2 in range(n_t)]
                        for t2 in range(n_t):
                            U[:, li * n_c + ci, w, d, :, combo[w], t2] = Z[t2] / Z[combo[w]] * gB + gS[t2]
                        if pt:  # F^{T i} (R/sqrt(P_lin), sqrt(n_X) s_X), R = B/Z1^T
                            y = np.broadcast_arrays(
                                gB / (Z[combo[w]] * np.sqrt(P_lin)), *(np.sqrt(n_g[x]) * gS[x] for x in range(n_t))
                            )
                            U_pt[:, li * n_c + ci, w, d] = (
                                F[:, :, combo[w], :, np.newaxis] * np.stack(y, axis=-1)[:, :, None]
                            )

        # Lambda within a shell - the same for each, column s*b + c
        idx = np.arange(n_mu * n_t * n_t).reshape(n_mu, n_t, n_t)
        w = fc.k_f**2 / (8 * np.pi * s_k) * w_mu[:, np.newaxis, np.newaxis]
        self.lam = np.zeros((idx.size, idx.size))
        self.lam[idx, idx[::-1].transpose(0, 2, 1)] = w  # BB: (n, T, T') -> (-n, T', T)
        if pt:  # PT: its own columns, each paired with itself
            lam = np.zeros((idx.size + n_mu * n_t * (1 + n_t),) * 2)
            lam[: idx.size, : idx.size] = self.lam
            lam[idx.size :, idx.size :] = np.diag(np.repeat(w[:, 0, 0], n_t * (1 + n_t)))
            self.lam = lam

        # the triangles on each node: per side, the closure fraction there over its K-weighted mean (so the triangle
        # count is kept), and a block of columns per node
        frac = np.empty((N_tri, 3, n_delta))
        for w_side in range(3):
            a, b = [i for i in range(3) if i != w_side]
            k = ks[w_side][:, np.newaxis]
            profile = triangle_bins.closure_profile(k + dk * self.delta, ks[a][:, np.newaxis], ks[b][:, np.newaxis], dk)
            frac[:, w_side] = profile / np.sum(
                self.w_delta * (1 + dk * self.delta / k) * profile, axis=-1, keepdims=True
            )
        if n_avg:
            U_all *= frac[:, np.newaxis, :, :, np.newaxis]
        else:  # the bin centre's columns on every node
            U_all = U_all * frac[:, np.newaxis, :, :, np.newaxis]
        self.U = U_all.reshape(N_tri, len(ln) * n_c, 3, -1)
        self.lam = np.kron(np.diag(self.w_delta), self.lam)

    def g_avg(self, ln, combos, order, nodes, weights, flat):
        """g_l(mu) of B, by its own tracers, and of its shot noise, by its tracers with the other triangle's on the
        shared side - each (N_tri, n_mu), averaged over the nodes: sides (3, N_tri, n_node) in the relabelled order"""
        owns = {tuple(combo[i] for i in order) for combo in combos}
        shots = {(t2, *own[1:]) for own in owns for t2 in range(self.n_t)}
        g_B, g_S = {}, {}
        for j in range(weights.shape[1]):
            legs = tuple(side[:, j, np.newaxis, np.newaxis] for side in nodes)
            theta = utils.get_theta(*legs)
            mus = numeric_mu_bk.los_cosines(self.mu[:, np.newaxis], self.psi, *legs, theta)
            mu_1 = mus[order.index(0)]  # the multipoles are defined by the original k1 - and k2 for m > 0
            if any(m for _, m in ln):
                k1, k2, k3 = (legs[order.index(i)] for i in range(3))
                phi_1 = _los_phi(mu_1, mus[order.index(1)], (k3**2 - k1**2 - k2**2) / (2 * k1 * k2), 1.0)
            leg_l = [
                (
                    np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu_1)
                    if m == 0
                    else 4 * np.pi * np.real(sph_harm_y(l, m, np.arccos(mu_1), phi_1)) * ~flat  # see _flat
                )
                * weights[:, j, np.newaxis, np.newaxis]
                / len(self.psi)
                for l, m in ln
            ]
            for own in owns:
                B = self.bk_clust(own, legs, theta, mus)
                for li, Y in enumerate(leg_l):
                    g_B[own, li] = g_B.get((own, li), 0) + np.sum(Y * B, axis=-1)
            for tracers in shots:
                S = self.bk_shot(tracers, legs, mus)
                for li, Y in enumerate(leg_l):
                    g_S[tracers, li] = g_S.get((tracers, li), 0) + np.sum(Y * S, axis=-1)
        return g_B, g_S

    def p_tot(self, k):
        """P^{T T'}(k, mu) + delta_TT'/n_T on the mu nodes, (k, mu, T, T') - of the local cov_terms with FOG as
        FullCovPk, from the Z1 B uses on the shared side"""
        cfs = [self.cf_mat_bk[t][t][t] for t in range(self.n_t)]
        shape = np.broadcast_shapes(np.shape(k), self.mu.shape)  # Z1 of 'N' alone has no k
        Z = [np.broadcast_to(numeric_mu_bk.get_Z1(self.terms, cf, self.zz, self.mu, k), shape) for cf in cfs]
        Z_m = [np.broadcast_to(numeric_mu_bk.get_Z1(self.terms, cf, self.zz, -self.mu, k), shape) for cf in cfs]
        P = np.einsum("akm,bkm->kmab", np.array(Z), np.array(Z_m)) * BaseInt(cfs[0]).pk(k, self.zz)[..., None, None]
        if self.sigma is not None:
            P = P * np.exp(-((k * self.mu) ** 2) * self.sigma**2)[..., np.newaxis, np.newaxis]
        return P + np.diag([1 / cf.survey[0].n_g(self.zz) for cf in cfs])

    def bk_clust(self, tracers, legs, theta, mus):
        """B for tracers at the three legs, on the (mu, psi) grid - without shot noise"""
        cf = self.cf_mat_bk[tracers[0]][tracers[1]][tracers[2]]
        B = numeric_mu_bk.get_mu_phi_sym(
            self.mu, self.psi, self.terms, self.terms, self.terms, cf, *legs, theta, self.zz
        )
        if self.sigma is not None:
            B = B * np.exp(-(1 / 2) * sum((k * mu) ** 2 for k, mu in zip(legs, mus)) * self.sigma**2)
        return B

    def bk_shot(self, tracers, legs, mus):
        """Shot noise of B^(N): the galaxy on the shared leg 0 - the other triangle's - merged with leg 1 or 2"""
        cf = self.cf_mat_bk[tracers[0]][tracers[1]][tracers[2]]

        # two fields on one galaxy: delta_ab/n_a P^ac(k_c) with the merged field at -k_c. FOG per P as FullCovBk
        pk = BaseInt(cf).pk
        S = 0
        for i, j, c in [(0, 1, 2), (0, 2, 1)]:
            if tracers[i] != tracers[j]:
                continue
            k, mu = legs[c], mus[c]
            P = (
                numeric_mu_bk.get_Z1(self.terms, cf, self.zz, -mu, k, ti=i)
                * numeric_mu_bk.get_Z1(self.terms, cf, self.zz, mu, k, ti=c)
                * pk(k, self.zz)
            )
            if self.sigma is not None:  # the merged galaxy and the third, each exp(-(k_c mu_c sigma)^2/2)
                P = P * np.exp(-((k * mu) ** 2) * self.sigma**2)
            S = S + P / cf.survey[i].n_g(self.zz)
        return S


class WoodburyInvCov:
    def __init__(self, blocks, lam, n_shell, chunk=32):
        """
        (D + W Xi W^dagger)^-1 = D^-1 - D^-1 W Xi M^-1 W^dagger D^-1,  M = I + G Xi,  G = W^dagger D^-1 W
        for one or more data vectors (bk, or pk and bk). blocks holds (D_inv, U, shell) for each: D_inv as from
        invert_matrix, block diagonal in its bins; U (N_bins, n_rows, n_sides, b) its factors per side, and shell
        (n_sides, N_bins) the k-shell of each side - from BBCovBk and PBCov. W is block diagonal over the data vectors
        and Xi only couples columns within a shell, the same in each: lam is Xi on one shell's columns, blocks in order.
        Columns of W zero everywhere are dropped (PBCov's padding - exact), and with one data vector Xi is reduced to its
        nonzero eigenvalues, negative ones included - the M to factorise shrinks as their number cubed.
        D_inv the pseudoinverse of invert_matrix gives the exact inverse of the covariance projected onto the subspace
        the Gaussian part keeps - the directions the Gaussian-only inverse drops too.
        Only holds arrays, so can be pickled with the Sampler's inv_covs.
        M is (n_shell b)^2 - ~0.5 GB for a multi-tracer P+B bin (MegaMapper-like, k_max 0.15) - so it is built in place:
        G in M's row and column order, Xi and the identity applied to it there. C-ordered M is M^T in Fortran order, so
        lu holds M^T's LU, factorised in M's memory with no copy, and contract solves with trans=1. chunk triangles at a
        time for G, as each makes a chunk x b x b block.
        """
        keep = np.concatenate([np.any(U != 0, axis=(0, 1, 2)) for _, U, _ in blocks])
        edges = np.cumsum([0] + [U.shape[-1] for _, U, _ in blocks])
        blocks = [
            (D_inv, U[..., keep[lo:hi]], shell) for (D_inv, U, shell), lo, hi in zip(blocks, edges[:-1], edges[1:])
        ]
        lam = lam[np.ix_(keep, keep)]
        if len(blocks) == 1:
            ev, Q = np.linalg.eigh(lam)
            nonzero = np.abs(ev) > 1e-12 * np.abs(ev).max()
            blocks = [(blocks[0][0], blocks[0][1] @ Q[:, nonzero], blocks[0][2])]
            lam = np.diag(ev[nonzero])

        self.blocks = [(np.moveaxis(D_inv, -1, 0), U, shell) for D_inv, U, shell in blocks]  # D_inv (N_bins, n, n)
        self.lam, self.n_shell = lam, n_shell
        self.offsets = np.cumsum([0] + [U.shape[-1] for _, U, _ in self.blocks])
        b = self.offsets[-1]

        G = np.zeros((n_shell, b, n_shell, b), dtype=np.complex128)  # (shell, column) rows and columns, as M
        for (D_inv, U_all, shell), lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            for start in range(0, len(U_all), chunk):
                sl = slice(start, start + chunk)
                U = U_all[sl]
                DU = np.einsum("tij,tjwc->tiwc", D_inv[sl], U)
                for w1 in range(len(shell)):
                    for w2 in range(len(shell)):
                        block = np.einsum("tic,tid->tcd", U[:, :, w1].conj(), DU[:, :, w2])
                        np.add.at(G[:, lo:hi, :, lo:hi], (shell[w1, sl], slice(None), shell[w2, sl]), block)

        M = G.reshape(n_shell * b, -1)  # a view
        cols = M.reshape(len(M), n_shell, b)
        for s in range(n_shell):  # G (1 x Xi), a shell's columns at a time
            cols[:, s] = cols[:, s] @ lam
        M[np.diag_indices_from(M)] += 1
        self.lu = lu_factor(M.T, overwrite_a=True)  # M^T = P L U - see contract

    def _to_shells(self, xs):
        """(N_bins, n_sides, b) per side of each block -> summed onto the global columns"""
        out = np.zeros((self.n_shell, self.offsets[-1]), dtype=np.complex128)
        for x, (_, _, shell), lo, hi in zip(xs, self.blocks, self.offsets[:-1], self.offsets[1:]):
            for w in range(len(shell)):
                np.add.at(out[:, lo:hi], shell[w], x[:, w])
        return out.ravel()

    def contract(self, d1, d2):
        """sum over bins of conj(d1)^T C^-1 d2 - see core.contract. d1 and d2 hold a data vector per block, or are
        the data vector when there is one"""
        if len(self.blocks) == 1 and not isinstance(d1, (tuple, list)):
            d1, d2 = [d1], [d2]
        tot, x1, x2 = 0, [], []
        for (D_inv, U, _), p, q in zip(self.blocks, d1, d2):
            p = np.conjugate(p)
            D_q = np.einsum("tij,jt->ti", D_inv, q)
            tot = tot + np.sum(p.T * D_q)
            x1.append(np.einsum("ti,tiwc->twc", np.einsum("it,tij->tj", p, D_inv), U))
            x2.append(np.einsum("tiwc,ti->twc", U.conj(), D_q))
        y = lu_solve(self.lu, self._to_shells(x2), trans=1).reshape(self.n_shell, -1)  # M y = x, lu being M^T's
        return tot - self._to_shells(x1) @ (y @ self.lam.T).ravel()


class PBCov:
    def __init__(self, pk_fc, bb, ln):
        """
        Power spectrum-bispectrum cross-covariance as V diag(lam) U^dagger, U the BB columns of BBCovBk - see
        WoodburyInvCov. Takes in PkForecast object, with the same tracers and k-bins as bb's BkForecast.

        The five-point function split into a power spectrum and a bispectrum: one field of the P estimator pairs with
        a triangle side in its shell and the other closes a bispectrum with the other two sides, so per shared side
        (2111.05887 eq 2.24 in real space):

        C[P^ab_l, B^T_l'] += (2l+1)/N_k Int dOmega/4pi L_l(mu) [P^{at}(mu) Bbar^{T[t->b]}_l'(mu)^* + (-1)^l P^{bt}(mu) Bbar^{T[t->a]}_l'(mu)^*]

        t the triangle's tracer on the shared side, which the other field of P replaces, and Bbar^{T[t->b]} BBCovBk's
        column (n, t, b) - B^(N), as P's field and the triangle's galaxies are in different estimators. The mode count
        is exact - the triangles on each mode of a shell sum to N_T. Neglects the connected (tetraspectrum) term.
        N_k is the shell's exact mode count, S2 = 1 + (dk/k)^2/12 times the thin-shell 4 pi k^2 dk (shell_modes) - BB's
        S2 cancels against its mode average, P x B's does not. With the thin-shell N_k (until 2026-10-15) C_PB was S2
        too large, 8% on the first shell, against the C_PP of PkForecast.get_cov_mat(shell=bb) - each mode's P, P x B,
        BB and PT are then not one covariance (the Gram matrix of delta and Y), and the multi-tracer joint covariance
        was indefinite (MegaMapper-like bright fractions, n_avg on).
        P is P(k,mu) of the local cov_terms kernels with FOG, plus undamped shot noise, as FullCovPk - at the |q| of
        each of bb's nodes across the shell, with its B (bb.delta_K: the bin centre without n_avg). Until 2026-10-13
        it stayed on the centre while B moved, up to ~20% off on the first pk shell.

        B's swap is BBCovBk's squeezed limit, (Z1^b/Z1^t) B with Z1 on the shared side - with the exact swap the
        multi-tracer joint covariance is not positive definite. The same for one tracer.
        """
        cosmo_funcs, kk, zz = pk_fc.args
        n_t = bb.n_t
        if bool(pk_fc.all_tracer) != (n_t == 2) or (not pk_fc.all_tracer and cosmo_funcs.multi_tracer):
            raise NotImplementedError("pk-bk cross-covariance needs pk and bk both all_tracer, or both one tracer")

        # rows as PkForecast.get_data_vector - odd multipoles only for XY
        pairs = (
            {True: [(0, 0), (0, 1), (1, 1)], False: [(0, 1)]} if pk_fc.all_tracer else {True: [(0, 0)], False: [(0, 0)]}
        )
        rows = [(l, pair) for l in ln for pair in pairs[l % 2 == 0]]

        terms = split_kernels(pk_fc.cov_terms)[1]
        dk = pk_fc.forecast.s_k * pk_fc.k_f
        k = kk[:, np.newaxis, np.newaxis]
        K = k + dk * bb.delta_K[:, np.newaxis]  # (N_k, n_K, 1) - |q| on each node bb's columns are on
        fog = 1 if bb.sigma is None else np.exp(-((K * bb.mu) ** 2) * bb.sigma**2)
        P = [
            [
                numeric_mu_pk.get_mu(bb.mu, terms, terms, pk_fc.cf_mat[a][t], K, zz) * fog
                + (a == t) / pk_fc.cf_mat[a][a].n_g(zz)
                for t in range(n_t)
            ]
            for a in range(n_t)
        ]  # P^{at}(K, mu), a at +q

        # bb's columns on each node - padded past BB's (n, T, T'), as PB does not touch PT's shot noise ones
        V_all = np.zeros((len(kk), len(rows), 1, len(bb.delta_K), bb.U.shape[-1] // len(bb.delta)), dtype=np.complex128)
        V = V_all[..., : len(bb.mu) * n_t * n_t].reshape(*V_all.shape[:4], len(bb.mu), n_t, n_t)  # a view
        for r, (l, (a, b)) in enumerate(rows):
            leg = (2 * l + 1) * eval_legendre(l, bb.mu) / k
            for t in range(n_t):
                V[:, r, 0, ..., t, b] += leg * P[a][t]
                V[:, r, 0, ..., t, a] += leg * (-1) ** l * P[b][t]

        self.shell = np.rint(kk / dk).astype(int)[np.newaxis] - 1  # (1, N_k)
        past = self.shell[0] >= bb.n_shell  # pk bins past the bispectrum's k_max share no side with a triangle
        V_all[past] = 0
        self.shell[:, past] = 0
        w_mu = utils.leggauss(len(bb.mu))[1]
        lam = np.repeat(pk_fc.k_f**2 / (8 * np.pi * pk_fc.forecast.s_k) * w_mu, n_t * n_t)  # as BBCovBk, per column
        lam = np.concatenate([lam, np.zeros(V_all.shape[-1] - lam.size)])

        # a block per delta node of bb: the P estimator weights every mode of the shell, so each node by its share of modes
        # - over the exact mode count, S2 = 1 + (dk/k)^2/12 times the thin-shell N_k of lam
        x, S2 = shell_modes(kk, dk, bb)
        modes = ((1 + x) / S2).T
        self.V = (V_all * modes[:, np.newaxis, np.newaxis, :, np.newaxis]).reshape(len(kk), len(rows), 1, -1)
        self.lam = np.kron(bb.w_delta, lam)


class PTTreeCovBk:
    def __init__(self, fc, cov_terms, ln, n_mu=12, n_psi=16, channels="stu3", shot=True, chunk=64):
        """
        PT term of the bispectrum covariance (P x trispectrum) with the tree-level trispectrum - see
        numeric_mu.bk.get_T_tree, channels 's', 't', 'u' of the exchange part and '3' for T3111. Takes in BkForecast
        object. Dense, so only for small k_max: cov is (N_tri*n_rows)^2, triangle-major. A check on the collapsed limit
        in BBCovBk(pt=True), which is the s channel with B in place of its P(k) part.

        Same shells and mode count as BBCovBk, but the P pairs the shared sides as q_c = p_c' = k, so per shared pair:

        C[B^T_l1, B^T'_l2] += k_f^2/(8 pi s_k k^2) Int dmu <(4pi)^2 Y_l1 Y_l2 P^ab(k) T(q_a, q_b, -p_x, -p_y)>_psi1,psi2

        averaged over the rotations of the two triangles about k - which the t and u channels do not separate.
        T is Newtonian: LP's Z2 has 1/q^2 terms, IR sensitive in the t and u channels. Shot noise in P^ab and, with
        shot, T's between the two triangles (numeric_mu.bk.get_T_shot) - as BBCovBk's estimators.
        """
        self.zz = fc.args[-1]
        self.terms = split_kernels(cov_terms)[1]  # local, as BBCovBk
        self.channels = channels
        self.shot = shot
        self.ln = ln
        self.combos, n_t, self.ks, self.shell = _triangle_layout(fc)
        self.views = [fc.cf_mat_bk[t][t][t] for t in range(n_t)]
        self.pref = fc.k_f**2 / (8 * np.pi * fc.forecast.s_k)

        self.mu, self.w_mu = utils.leggauss(n_mu)
        # T is even under reflecting both triangles through the plane of k and the LOS - so psi1 in [0, pi].
        # psi2 on nodes offset from psi1's, so the triangles never exactly coincide
        self.psi1 = np.pi * (np.arange(n_psi // 2) + 0.5) / (n_psi // 2)
        self.psi2 = 2 * np.pi * np.arange(n_psi) / n_psi

        N_tri, n_l, n_c = self.ks.shape[1], len(ln), len(self.combos)
        C = np.zeros((N_tri, n_l, n_c, N_tri, n_l, n_c), dtype=np.complex128)
        upper = np.triu(np.ones((N_tri, N_tri), dtype=bool))  # the rest by hermiticity
        for c1 in range(3):
            for c2 in range(3):
                t1, t2 = np.nonzero((self.shell[c1][:, np.newaxis] == self.shell[c2]) & upper)
                for start in range(0, len(t1), chunk):
                    sl = slice(start, start + chunk)
                    np.add.at(C, (t1[sl], slice(None), slice(None), t2[sl]), self.blocks(t1[sl], c1, t2[sl], c2))

        C = C.reshape(N_tri, n_l * n_c, N_tri, n_l * n_c)
        C = C + C.conj().transpose(2, 3, 0, 1)
        diag = np.arange(N_tri)
        C[diag, :, diag, :] /= 2  # counted twice above
        self.cov = C.reshape(N_tri * n_l * n_c, -1)

    def blocks(self, t1, c1, t2, c2):
        """C_PT between triangles t1 and t2 (arrays) from the pairing of side c1 of t1 with side c2 of t2,
        shape (len(t1), n_l, n_c, n_l, n_c)"""
        mu = self.mu[:, np.newaxis, np.newaxis, np.newaxis]
        e = np.concatenate([np.sqrt(1 - mu**2), 0 * mu, mu], axis=-1)  # direction of k, (n_mu, 1, 1, 3)
        e1 = np.concatenate([mu, 0 * mu, -np.sqrt(1 - mu**2)], axis=-1)
        e2 = np.array([0.0, 1.0, 0.0])
        rot1 = np.cos(self.psi1)[:, np.newaxis, np.newaxis] * e1 + np.sin(self.psi1)[:, np.newaxis, np.newaxis] * e2
        rot2 = np.cos(self.psi2)[:, np.newaxis] * e1 + np.sin(self.psi2)[:, np.newaxis] * e2  # (n_mu, 1, n_psi, 3)

        def sides(t, c, rot):
            """the three sides of triangles t - side c on k e, rotated by rot about it"""
            a, b = [i for i in range(3) if i != c]
            kc, ka, kb = (self.ks[i, t][:, None, None, None, None] for i in (c, a, b))
            cos_a = np.clip((kb**2 - ka**2 - kc**2) / (2 * ka * kc), -1, 1)
            v = [None] * 3
            v[c] = kc * e
            v[a] = ka * (cos_a * e + np.sqrt(1 - cos_a**2) * rot)
            v[b] = -(v[a] + v[c])
            return v, (a, b)

        v1, (a1, b1) = sides(t1, c1, rot1)
        v2, (x2, y2) = sides(t2, c2, rot2)
        vecs = np.stack(np.broadcast_arrays(v1[a1], v1[b1], -v2[x2], -v2[y2]))  # q_a, q_b, -p_x, -p_y
        k = self.ks[c1, t1][:, None, None, None]

        # sqrt((4pi)^2 (2l1+1)(2l2+1)) L_l1(mu_1) L_l2(mu_1) of each triangle's original k1 - over psi1, psi2
        mu1 = np.broadcast_to(v1[0][..., 0, 2] / self.ks[0, t1][:, None, None], (len(t1), len(self.mu), len(self.psi1)))
        mu2 = np.broadcast_to(
            v2[0][..., 0, :, 2] / self.ks[0, t2][:, None, None], (len(t1), len(self.mu), len(self.psi2))
        )
        L1 = np.array([np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu1) for l in self.ln])
        L2 = np.array([np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu2) for l in self.ln])
        w = self.pref * self.w_mu / (len(self.psi1) * len(self.psi2))

        pk = BaseInt(self.views[0]).pk
        mu_k = self.mu[:, None, None]
        out = np.zeros((len(t1), len(self.ln), len(self.combos), len(self.ln), len(self.combos)), dtype=np.complex128)
        T_cache = {}
        for ci1, combo1 in enumerate(self.combos):
            for ci2, combo2 in enumerate(self.combos):
                tracers = (combo1[a1], combo1[b1], combo2[x2], combo2[y2])
                if tracers not in T_cache:
                    T_cache[tracers] = numeric_mu_bk.get_T_tree(
                        [self.views[t] for t in tracers], self.zz, vecs, self.channels, self.shot
                    )

                al, be = self.views[combo1[c1]], self.views[combo2[c2]]
                P = (
                    numeric_mu_bk.get_Z1(self.terms, al, self.zz, mu_k, k)
                    * numeric_mu_bk.get_Z1(self.terms, be, self.zz, -mu_k, k)
                    * pk(k, self.zz)
                )
                if combo1[c1] == combo2[c2]:
                    P = P + 1 / al.survey[0].n_g(self.zz)
                out[:, :, ci1, :, ci2] = (
                    np.einsum("amni,bmnj,mnij,n->mab", L1, L2, P * T_cache[tracers], w) / k[:, 0, 0, 0, None, None] ** 2
                )
        return out
