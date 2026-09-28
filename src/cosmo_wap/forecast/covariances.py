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
from cosmo_wap.lib import utils
from cosmo_wap.lib.integrated import BaseInt
from cosmo_wap.numeric_mu import bk as numeric_mu_bk
from cosmo_wap.numeric_mu import pk as numeric_mu_pk

__all__ = ["FullCovPk", "FullCovBk", "BBCovBk", "WoodburyInvCov", "PBCov", "PTTreeCovBk"]


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

    def get_cov(self, ln, sigma=None):
        """Gets full covariance matrix"""
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
            for j in range(i, N):
                if self.cf_mat[i][j]:  # we can skip some calculation for the XY non all-tracer case
                    if not self.kernels:  # each analytic term is already a full contribution to P(k, mu)
                        self.pk_cache[i][j] = sum(
                            getattr(pk, term).mu(self.mu, self.cf_mat[i][j], *args[1:], **kwargs) for term in self.terms
                        )
                    else:  # kernels sum before squaring - so the cross terms, e.g. <N LP*>, are included
                        self.pk_cache[i][j] = numeric_mu_pk.get_mu_sym(
                            self.mu, list(self.terms), list(self.terms), self.cf_mat[i][j], *args[1:], **kwargs
                        )  # so can change to new mechanism -new int
                    if i != j:
                        self.pk_cache[j][i] = np.conjugate(self.pk_cache[i][j])  # this holds currently P_YX = P_XY*

    def integrate_mu(self, i1, i2, j1, j2, terms, l1, l2, mu):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer t1=t2=t3=t4=0 (i.e. P_XX P_XX)
        For say: P_XY P_XX t1=t2=t4=0;t3=1 - P_t1t3 P_t2t4
        """
        coef = (2 * l1 + 1) * (2 * l2 + 1) * eval_legendre(l1, self.mu) * eval_legendre(l2, mu) * self.weights

        # add shot noise - is zero in XY case
        a = self.pk_cache[i1][i2] + 1 / self.cf_mat[i1][i2].n_g(self.zz)
        b = self.pk_cache[j1][j2] + 1 / self.cf_mat[j1][j2].n_g(self.zz)
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
        Covariance has shape [k1,k2,k3,mu,phi]"""
        self.fc = fc
        self.terms = cov_terms
        self.sigma = sigma
        self.nonlin = nonlin
        self.kernels = kernels  # work directly with kernels or just Bk

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
        _, k1, k2, k3, _, self.zz = fc.args
        mu = mu[:, np.newaxis]
        k1, k2, k3 = utils.enable_broadcasting(
            k1, k2, k3, n=2
        )  # if arrays add newaxis at the end so is broadcastable with mu!

        # lets define some bispectrum stuff
        _, theta = utils.get_theta_k3(k1, k2, k3, None)

        mu2 = mu * np.cos(theta) + np.sqrt(1 - mu**2) * np.sin(theta) * np.cos(phi)
        mu3 = -(mu * k1 + mu2 * k2) / k3
        self.mus = mu, mu2, mu3

        self.ks = np.array([k1, k2, k3])
        self.N_tri = len(k1)  # is literally number of triangles - k1,k2,k3 are flattened to this shape

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

        self.cov = ll_cov  # caching dont hurt
        return ll_cov

    def create_cache(self, **kwargs):
        """Store all Bks as a function of mu! - this can then be reused for each l!
        This should be the expensive function - at least for integrated stuff
        so store the total P(k,mu) for each tracer combination

        | XX XY XZ |
        | YX YY YZ |
        | ZX ZY ZZ | where YX = np.conjugate(XY) in this case - so triangle numbers compute (n+1)*n/2 pairs!
        However the standard case will be working with 2 tracers - like the power spectrum

        also store for k1,k2,k3 etc
        so shape probably 2,2,3,N
        We also now do it with numerical mu-int
        """

        N = len(self.cf_mat)
        self.pk_cache = [[[None for _ in range(N)] for _ in range(N)] for _ in range(3)]

        for i in range(N):
            for j in range(i, N):
                for ki in range(3):
                    if not self.kernels:  # then get from mathematica expressions - each already a full contribution
                        self.pk_cache[ki][i][j] = sum(
                            getattr(pk, term).mu(self.mus[ki], self.cf_mat[i][j], self.ks[ki], self.zz, **kwargs)
                            for term in self.terms
                        )
                    else:  # kernels sum before squaring - so the cross terms, e.g. <N LP*>, are included
                        self.pk_cache[ki][i][j] = numeric_mu_pk.get_mu_sym(
                            self.mus[ki], list(self.terms), list(self.terms), self.cf_mat[i][j], self.ks[ki], self.zz
                        )
                    if i != j:
                        self.pk_cache[ki][j][i] = np.conjugate(
                            self.pk_cache[ki][i][j]
                        )  # this holds currently P_YX = P_XY*

    def integrate_mu(self, i1, i2, j1, j2, k1, k2, terms, l1, l2, mu, mu2=None):
        """Combine all powerspectrum contributions and integrate to get the full contribution
        Uses the stored P(k,mu) cache!
        Is called for each tracer combination
        For single tracer i1=i2=j1=j2=k1=k2=0 (i.e. P_XX P_XX P_XX)
        For say: P_XY P_XX P_XX -> i1=j1=j2=k1=k2=0;j1=1
        """
        if mu2 is None:
            mu2 = mu

        m = 0
        phi = 0  # can edit later for m\neq0
        coef = (
            4
            * np.pi
            * np.conjugate(sph_harm_y(l1, m, np.arccos(mu), phi))
            * sph_harm_y(l2, m, np.arccos(mu2), phi)
            * self.weights
        )

        # FOG - one factor per k
        if self.sigma is None:
            fog = (1, 1, 1)
        else:
            fog = tuple(np.exp(-(1 / 2) * ((self.ks[i] * self.mus[i]) ** 2) * self.sigma**2) for i in range(3))

        # add shot noise - is zero in XY case
        a = self.pk_cache[0][i1][i2] * fog[0] + 1 / self.cf_mat[i1][i2].n_g(self.zz)
        b = self.pk_cache[1][j1][j2] * fog[1] + 1 / self.cf_mat[j1][j2].n_g(self.zz)
        c = self.pk_cache[2][k1][k2] * fog[2] + 1 / self.cf_mat[k1][k2].n_g(self.zz)
        return (2 * np.pi) / 2.0 * np.sum(coef * a * b * c, axis=(-2, -1))  # sum over last 2 axes - mu and phi

    def get_single_tracer_ll(self, terms, l1, l2):
        """Get single-tracer covariance for multipole pair"""
        if len(self.cf_mat) > 1:  # then we have XY term
            return self.get_tracer(0, 1, 0, 0, 1, 0, terms, l1, l2)
        return self.fc.s123 * self.integrate_mu(0, 0, 0, 0, 0, 0, terms, l1, l2, self.mus[0])  # with s123

    def get_tracer(self, a, b, c, d, e, f, terms, l1, l2):
        """Get C[B^abc_{l}, B^def_{l2}](k) - i.e. PPP term to bispectrum covariance
        C[B^abc_{l}, B^def_{l2}](k1,k2,k3) = ( Int (d(Omega_k) / 4*pi) * Y_l1m1(mu,phi) *Y_l2m2(mu,phi)
                                        [P^ad(k1,mu)*P^be(k2,mu2)*P^cf(k3,mu3)]
        only compute unique (perm, mu_idx) combinations
        """

        perms = list(itertools.permutations([d, e, f]))
        perms_index = list(itertools.permutations(["d", "e", "f"]))
        # [(d, e, f), (d, f, e), (e, d, f), (e, f, d), (f, d, e), (f, e, d)]
        # deduplicate on (tracer_perm, mu_idx)
        results = {}
        for i, perm in enumerate(perms):
            mu_idx = perms_index[i].index("d")  # q1 goes to ki
            key = (perm,) if l2 == 0 else (perm, mu_idx)  # we can ignore d placement for monopole
            if key not in results:
                results[key] = self.integrate_mu(
                    a, perm[0], b, perm[1], c, perm[2], terms, l1, l2, self.mus[0], mu2=self.mus[mu_idx]
                )

        # build ordered list matching the original 6 permutations
        cov_list = [results[(perms[i],) if l2 == 0 else (perms[i], perms_index[i].index("d"))] for i in range(6)]

        cov_tot = cov_list[0]
        # now for equilateral and isoceles triangles - we have addtional terms from dirac-deltas - rememeber k1\geq k2\geq k3
        k1, k2, k3 = self.ks.squeeze()  # so shape kk
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
        Shape [4xln,4xln]
        Exploit overall hermitian symmetry
        """
        nl = len(ln)
        nt = 4
        cov_mt = np.zeros((nt * nl, nt * nl, self.N_tri), dtype=np.complex128)  # create empty complex array

        # lets build our covariance matix!
        # so first we loop over l and then over tracers
        tracers = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
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
        combos, n_t = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)], 2  # as get_multi_tracer and get_data_vector
    elif cosmo_funcs.multi_tracer:
        raise NotImplementedError("non-Gaussian covariance only for all_tracer or a single tracer")
    else:
        combos, n_t = [(0, 0, 0)], 1

    ks = np.array([k1, k2, k3])
    shell = np.rint(ks / (fc.forecast.s_k * fc.k_f)).astype(int) - 1  # k_bin = (i+1) dk
    return combos, n_t, ks, shell


def _closure_profile(K, k_a, k_b, dk, n=16):
    """Fraction of the shells of k_a and k_b that close a triangle with a side of length K, weighted q_a q_b as the
    triangle count - 1 unless the bin is (nearly) flattened. Averaged over K's shell (weighted by K) it is the bin's
    beta, see forecast.core._triangle_beta. q_a by Gauss-Legendre, q_b analytic"""
    x, w = utils.leggauss(n)
    q_a = k_a[..., np.newaxis] + dk * x / 2
    lo = np.maximum(k_b[..., np.newaxis] - dk / 2, np.abs(K[..., np.newaxis] - q_a))
    hi = np.minimum(k_b[..., np.newaxis] + dk / 2, K[..., np.newaxis] + q_a)
    q_b_int = np.where(hi > lo, (hi**2 - lo**2) / 2, 0)  # int q_b dq_b over the part of shell b that closes
    return np.sum(w / 2 * q_a * q_b_int, axis=-1) / (k_a * k_b * dk)


class BBCovBk:
    def __init__(self, fc, cov_terms, ln, sigma=None, n_mu=12, n_psi=8, pt=True, n_delta=1):
        """
        BB term of the bispectrum covariance (+ its PT partner), as low-rank factors U Lambda U^dagger - see
        WoodburyInvCov. Takes in BkForecast object.

        The six-point function split into two bispectra, each with two fields of one triangle and one of the other,
        so the triangles share a side (q_c = -p_c'). 9 pairings, nonzero when the two sides are in the same k-shell,
        so it couples different triangles. For each shared pair, in the thin-shell limit:

        C[B^T_l1, B^T'_l2] += k_f^2/(8 pi s_k k^2) Int dmu g^T_l1(mu) conj(g^T'_l2(-mu))

        g_l(mu) = <sqrt(4 pi (2l+1)) L_l(mu_1) B_tot>_psi: the triangle with LOS cosine mu on the shared side,
        averaged over its rotation psi about that side. B_tot includes shot noise. In real space this is the usual
        B_T B_T' / N_k (2111.05887 eq 2.30, 2403.08634). Closure uses the bin-averaged beta as V123, so is only
        approximate for (nearly) flattened triangles.

        Separable in mu - so columns are (shell, mu node, T, T'), with T the tracer the row's triangle has on the
        shared side and T' the other triangle's, which each swaps into its own B. Lambda pairs (s, n, T, T')
        with (s, -n, T', T).

        BB alone is indefinite: the -k gives conj(g(-mu)) = (-1)^l2 g(mu), so odd l have negative variance.
        pt adds the collapsed limit of the PT term (P x trispectrum), T -> B B*/P - the same with the other
        triangle on +k and each keeping its own tracers: Lambda pairs (s, n, T, T) with (s, n, T', T'). In real
        space it equals BB (the 2BB of 2403.08634). Only the squeezed limit of PT - so approximate when the
        shared side is not the soft one.
        With pt, BB also keeps its own tracers - (s, n, T, T) with (s, -n, T', T') - the squeezed limit, where
        swapping only swaps Z1 on the shared side. Then BB + PT is 1/2 sum_k (a(k) + a(-k))(a(k) + a(-k))^dagger,
        positive semi-definite - the exact swap with this PT is not. Single tracer is the same either way.
        The P pairing PT's shared sides carries shot noise, P_gal + 1/n, which B B*/P misses: on the same tracer PT
        gains Bbar Bbar*/(n P_gal). Its weight depends on k so it goes in U - a column per (n, T) after the others,
        the own-tracer one over sqrt(n P_gal) on the shared side, paired with itself. 2403.08634 keep it in C_PT
        but not in their PT = BB - the same when n P >> 1.

        The number of a bin's triangles on a mode of the shared shell follows the closure fraction at its |k|, which
        varies across the shell for (nearly) flattened bins - n_delta > 1 repeats the columns for nodes across the
        shell width, each side weighted by its closure fraction there (2111.05887 sec 2.3). Closed bins are unchanged.
        n_delta=1 takes the bin average, off by up to ~25% on flattened pairs (4 nodes converge) - but the rank, so
        the Woodbury setup ~ n_delta^3 and its memory ~ n_delta^2: 4 is too much for multi-tracer at k_max ~ 0.15.
        """
        self.zz = fc.args[-1]
        self.cf_mat_bk = fc.cf_mat_bk
        self.terms = cov_terms
        self.sigma = sigma

        combos, n_t, ks, self.shell = _triangle_layout(fc)
        self.n_shell = self.shell.max() + 1
        self.n_t = n_t
        s_k = fc.forecast.s_k

        # B is even in psi (reflection through the plane of the shared side and the LOS) - so [0, pi] as get_mu_phi_grid
        self.mu, w_mu = utils.leggauss(n_mu)
        self.psi = np.pi * (np.arange(n_psi) + 0.5) / n_psi

        N_tri = ks.shape[1]
        n_c = len(combos)
        U = np.zeros((N_tri, len(ln) * n_c, 3, n_mu, n_t, n_t), dtype=np.complex128)

        # relabel so the shared side is the first leg: mu is then its LOS cosine and phi is psi
        for w, order in enumerate([(0, 1, 2), (1, 0, 2), (2, 0, 1)]):
            legs = tuple(ks[i][:, np.newaxis, np.newaxis] for i in order)
            theta = utils.get_theta(*legs)
            mus = numeric_mu_bk.los_cosines(self.mu[:, np.newaxis], self.psi, *legs, theta)
            mu_1 = mus[order.index(0)]  # the multipoles are defined by the original k1
            leg_l = [np.sqrt(4 * np.pi * (2 * l + 1)) * eval_legendre(l, mu_1) / n_psi for l in ln]

            B_cache = {}
            for ci, combo in enumerate(combos):
                for t2 in [combo[w]] if pt else range(n_t):
                    tracers = tuple(
                        t2 if i == w else combo[i] for i in order
                    )  # other triangle's tracer on the shared side
                    if tracers not in B_cache:
                        B_cache[tracers] = self.bk_tot(tracers, legs, theta, mus)

                    for li in range(len(ln)):
                        U[:, li * n_c + ci, w, :, combo[w], t2] = (
                            np.sum(leg_l[li] * B_cache[tracers], axis=-1) / legs[0][..., 0]
                        )

        self.U = U.reshape(N_tri, len(ln) * n_c, 3, -1)

        # Lambda within a shell - the same for each, column s*b + c
        idx = np.arange(n_mu * n_t * n_t).reshape(n_mu, n_t, n_t)
        w = fc.k_f**2 / (8 * np.pi * s_k) * w_mu[:, np.newaxis, np.newaxis]
        self.lam = np.zeros((idx.size, idx.size))
        if pt:
            own = idx[:, range(n_t), range(n_t)]  # (n, T, T)
            self.lam[own[:, :, np.newaxis], own[::-1, np.newaxis, :]] = w  # BB
            self.lam[own[:, :, np.newaxis], own[:, np.newaxis, :]] += w  # PT
        else:
            self.lam[idx, idx[::-1].transpose(0, 2, 1)] = w  # (n, T, T') -> (-n, T', T)

        if pt:  # PT's shot noise
            U_shot = np.zeros((N_tri, len(ln) * n_c, 3, n_mu, n_t), dtype=np.complex128)
            for t in range(n_t):
                cf = self.cf_mat_bk[t][t][t]
                n_g = cf.survey[0].n_g(self.zz)
                for w_side in range(3):
                    P_gal = numeric_mu_pk.get_mu(
                        self.mu, list(cov_terms), list(cov_terms), cf, ks[w_side][:, np.newaxis], self.zz
                    ).real
                    U_shot[:, :, w_side, :, t] = U[:, :, w_side, :, t, t] / np.sqrt(n_g * P_gal)[:, np.newaxis, :]
            self.U = np.concatenate([self.U, U_shot.reshape(N_tri, len(ln) * n_c, 3, -1)], axis=-1)
            lam = np.zeros((idx.size + n_mu * n_t,) * 2)
            lam[: idx.size, : idx.size] = self.lam
            lam[idx.size :, idx.size :] = np.diag(np.repeat(w[:, 0, 0], n_t))
            self.lam = lam

        # where the shared mode sits in its shell: per side, the closure fraction there over its K-weighted mean
        # (so the triangle count is kept), and a block of columns per node
        delta, w_delta = utils.leggauss(n_delta)
        self.delta, self.w_delta = delta / 2, w_delta / 2  # nodes in [-1/2, 1/2] bin widths, weights sum to 1
        dk = s_k * fc.k_f
        frac = np.empty((N_tri, 3, n_delta))
        for w_side in range(3):
            a, b = [i for i in range(3) if i != w_side]
            k = ks[w_side][:, np.newaxis]
            profile = _closure_profile(k + dk * self.delta, ks[a][:, np.newaxis], ks[b][:, np.newaxis], dk)
            frac[:, w_side] = profile / np.sum(
                self.w_delta * (1 + dk * self.delta / k) * profile, axis=-1, keepdims=True
            )
        self.U = (self.U[:, :, :, np.newaxis] * frac[:, np.newaxis, :, :, np.newaxis]).reshape(*self.U.shape[:3], -1)
        self.lam = np.kron(np.diag(self.w_delta), self.lam)

    def bk_tot(self, tracers, legs, theta, mus):
        """B + shot noise for tracers at the three legs, on the (mu, psi) grid"""
        cf = self.cf_mat_bk[tracers[0]][tracers[1]][tracers[2]]
        B = numeric_mu_bk.get_mu_phi_sym(
            self.mu, self.psi, self.terms, self.terms, self.terms, cf, *legs, theta, self.zz
        )
        if self.sigma is not None:
            B = B * np.exp(-(1 / 2) * sum((k * mu) ** 2 for k, mu in zip(legs, mus)) * self.sigma**2)

        # two fields on one galaxy: delta_ab/n_a P^ac(k_c) with the merged field at -k_c. FOG per P as FullCovBk
        pk = BaseInt(cf).pk
        for i, j, c in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
            if tracers[i] != tracers[j]:
                continue
            k, mu = legs[c], mus[c]
            P = (
                numeric_mu_bk.get_Z1(self.terms, cf, self.zz, -mu, k, ti=i)
                * numeric_mu_bk.get_Z1(self.terms, cf, self.zz, mu, k, ti=c)
                * pk(k, self.zz)
            )
            if self.sigma is not None:
                P = P * np.exp(-(1 / 2) * (k * mu) ** 2 * self.sigma**2)
            B = B + P / cf.survey[i].n_g(self.zz)

        if tracers[0] == tracers[1] == tracers[2]:
            B = B + 1 / cf.survey[0].n_g(self.zz) ** 2
        return B


class WoodburyInvCov:
    def __init__(self, blocks, lam, n_shell, chunk=1024):
        """
        (D + W Xi W^dagger)^-1 = D^-1 - D^-1 W Xi M^-1 W^dagger D^-1,  M = I + G Xi,  G = W^dagger D^-1 W
        for one or more data vectors (bk, or pk and bk). blocks holds (D_inv, U, shell) for each: D_inv as from
        invert_matrix, block diagonal in its bins; U (N_bins, n_rows, n_sides, b) its factors per side, and shell
        (n_sides, N_bins) the k-shell of each side - from BBCovBk and PBCov. W is block diagonal over the data vectors
        and Xi only couples columns within a shell, the same in each: lam is Xi on one shell's columns, blocks in order.
        Only holds arrays, so can be pickled with the Sampler's inv_covs.
        """
        self.blocks = [(np.moveaxis(D_inv, -1, 0), U, shell) for D_inv, U, shell in blocks]  # D_inv (N_bins, n, n)
        self.lam, self.n_shell = lam, n_shell
        self.offsets = np.cumsum([0] + [U.shape[-1] for _, U, _ in self.blocks])
        b = self.offsets[-1]

        G = np.zeros((n_shell, n_shell, b, b), dtype=np.complex128)
        for (D_inv, U_all, shell), lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            for start in range(0, len(U_all), chunk):
                sl = slice(start, start + chunk)
                U = U_all[sl]
                DU = np.einsum("tij,tjwc->tiwc", D_inv[sl], U)
                for w1 in range(len(shell)):
                    for w2 in range(len(shell)):
                        block = np.einsum("tic,tid->tcd", U[:, :, w1].conj(), DU[:, :, w2])
                        np.add.at(G[:, :, lo:hi, lo:hi], (shell[w1, sl], shell[w2, sl]), block)

        G = G.transpose(0, 2, 1, 3).reshape(n_shell * b, -1)
        self.lu = lu_factor(np.eye(len(G)) + (G.reshape(len(G), n_shell, b) @ lam).reshape(len(G), -1))

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
        y = lu_solve(self.lu, self._to_shells(x2)).reshape(self.n_shell, -1)
        return tot - self._to_shells(x1) @ (y @ self.lam.T).ravel()


class PBCov:
    def __init__(self, pk_fc, bb, ln):
        """
        Power spectrum-bispectrum cross-covariance as V diag(lam) U^dagger, U from BBCovBk(pt=True) - see
        WoodburyInvCov. Takes in PkForecast object, with the same tracers and k-bins as bb's BkForecast.

        The five-point function split into a power spectrum and a bispectrum: one field of the P estimator pairs with
        a triangle side in its shell and the other closes a bispectrum with the other two sides, so per shared side
        (2111.05887 eq 2.24 in real space):

        C[P^ab_l, B^T_l'] += (2l+1)/N_k Int dOmega/4pi L_l(mu) [P^{at}(mu) Bbar^{T[t->b]}_l'(mu)^* + (-1)^l P^{bt}(mu) Bbar^{T[t->a]}_l'(mu)^*]

        t the triangle's tracer on the shared side, which the other field of P replaces, and Bbar as in BBCovBk.
        The mode count is exact - the triangles on each mode of a shell sum to N_T. Neglects the connected
        (tetraspectrum) term. P is P(k,mu) of the local cov_terms kernels plus shot noise, without FOG as FullCovPk.

        The swap is taken in the squeezed limit, Bbar^{T[t->b]} = (Z1^b/Z1^t) Bbar^T with Z1 on the shared side, as
        BBCovBk(pt=True) keeps own tracers - with the exact swap the multi-tracer joint covariance is not positive
        definite. The same for one tracer.
        """
        cosmo_funcs, kk, zz = pk_fc.args
        n_t = bb.n_t
        if pk_fc.all_tracer != (n_t == 2) or (not pk_fc.all_tracer and cosmo_funcs.multi_tracer):
            raise NotImplementedError("pk-bk cross-covariance needs pk and bk both all_tracer, or both one tracer")

        # rows as PkForecast.get_data_vector - odd multipoles only for XY
        pairs = (
            {True: [(0, 0), (0, 1), (1, 1)], False: [(0, 1)]} if pk_fc.all_tracer else {True: [(0, 0)], False: [(0, 0)]}
        )
        rows = [(l, pair) for l in ln for pair in pairs[l % 2 == 0]]

        terms = list(pk_fc.cov_terms)
        k = kk[:, np.newaxis]
        P = [
            [
                numeric_mu_pk.get_mu(bb.mu, terms, terms, pk_fc.cf_mat[a][t], k, zz)
                + (a == t) / pk_fc.cf_mat[a][a].n_g(zz)
                for t in range(n_t)
            ]
            for a in range(n_t)
        ]  # P^{at}(k, mu), a at +k

        Z = [numeric_mu_bk.get_Z1(terms, pk_fc.cf_mat[t][t], zz, bb.mu, k) for t in range(n_t)]  # on the shared side

        V = np.zeros(
            (len(kk), len(rows), 1, len(bb.mu), n_t, n_t), dtype=np.complex128
        )  # columns (n, T, T') as BBCovBk
        for r, (l, (a, b)) in enumerate(rows):
            leg = (2 * l + 1) * eval_legendre(l, bb.mu) / k
            for t in range(n_t):
                V[:, r, 0, :, t, t] = leg * (
                    P[a][t] * np.conj(Z[b] / Z[t]) + (-1) ** l * P[b][t] * np.conj(Z[a] / Z[t])
                )

        self.shell = np.rint(kk / (pk_fc.forecast.s_k * pk_fc.k_f)).astype(int)[np.newaxis] - 1  # (1, N_k)
        past = self.shell[0] >= bb.n_shell  # pk bins past the bispectrum's k_max share no side with a triangle
        V[past] = 0
        self.shell[:, past] = 0
        # padded to bb's columns - PB does not touch PT's shot noise ones
        pad = bb.U.shape[-1] // len(bb.delta) - V[0, 0, 0].size
        V = np.concatenate([V.reshape(len(kk), len(rows), 1, -1), np.zeros((len(kk), len(rows), 1, pad))], axis=-1)
        w_mu = utils.leggauss(len(bb.mu))[1]
        lam = np.repeat(pk_fc.k_f**2 / (8 * np.pi * pk_fc.forecast.s_k) * w_mu, n_t * n_t)  # as BBCovBk, per column
        lam = np.concatenate([lam, np.zeros(pad)])

        # a block per delta node of bb: the P estimator weights every mode of the shell, so each node by its share of modes
        modes = 1 + pk_fc.forecast.s_k * pk_fc.k_f * bb.delta / kk[:, np.newaxis]
        self.V = (V[:, :, :, np.newaxis] * modes[:, np.newaxis, np.newaxis, :, np.newaxis]).reshape(
            len(kk), len(rows), 1, -1
        )
        self.lam = np.kron(bb.w_delta, lam)


class PTTreeCovBk:
    def __init__(self, fc, cov_terms, ln, n_mu=12, n_psi=16, channels="stu", chunk=64):
        """
        PT term of the bispectrum covariance (P x trispectrum) with the tree-level exchange trispectrum - see
        numeric_mu.bk.get_T_exchange (no Z3 terms). Takes in BkForecast object. Dense, so only for small k_max: cov is
        (N_tri*n_rows)^2, triangle-major. A check on the collapsed limit in BBCovBk(pt=True), which is the s channel
        with B in place of its P(k) part.

        Same shells and mode count as BBCovBk, but the P pairs the shared sides as q_c = p_c' = k, so per shared pair:

        C[B^T_l1, B^T'_l2] += k_f^2/(8 pi s_k k^2) Int dmu <(4pi)^2 Y_l1 Y_l2 P^ab(k) T(q_a, q_b, -p_x, -p_y)>_psi1,psi2

        averaged over the rotations of the two triangles about k - which the t and u channels do not separate.
        T is Newtonian: LP's Z2 has 1/q^2 terms, IR sensitive in the t and u channels. Shot noise only in P^ab.
        """
        self.zz = fc.args[-1]
        self.terms = cov_terms
        self.channels = channels
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
                    T_cache[tracers] = numeric_mu_bk.get_T_exchange(
                        ["N"], [self.views[t] for t in tracers], self.zz, vecs, self.channels
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
