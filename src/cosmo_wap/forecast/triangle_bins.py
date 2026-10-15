"""The triangles a bispectrum bin's B is evaluated on - FullForecast(bin_avg).

A bin holds every triangle with sides in its three k-shells, and for thin bins these differ by a lot: when k3 ~ dk
they run from collinear to perpendicular, so B at the bin centre can be far from its mean over the bin. Each option is
a function of the bin centres (k1, k2, k3, each (N_tri,)) and the bin width dk, returning a TriangleSet: the triangles
to evaluate B on, each one's bin and weight. BkForecast evaluates on them, and TriangleSet.mean averages to the bins.

- 'centre' (bin_avg=False): the bin centre, as before 2026-10-12.
- 'effective' (bin_avg=True, the default): the sorted effective triangle of 1908.01774 (eq 3.8), once per labelling of
  the equal-shell sides. One B a bin - B_0 within 2-9% of the bin average, but the GR dipole of (i, i-1, 1) bins ~3x
  low and of others such as (8, 8, 1) 17-27% high (Euclid 1 < z < 1.2, s_k 4).
- 'exact': B on n^3 Gauss-Legendre nodes over each bin's closed triangles - the bin average itself, to the
  quadrature, at ~n^3 times the cost of the signal.

OPTIONS gives the triangles of the signal (and its derivatives) and of the Gaussian covariance for each, and how many
nodes a side BBCovBk averages its B over (its n_avg, on bin_nodes - 0 is the bin centre). Under 'exact' the
covariance keeps the effective triangle - its <PPP> is within 2%, and n^3 triangles a bin there would be n^3 times the
memory.

A new option is a function and a row of OPTIONS, e.g.
OPTIONS["exact6"] = (functools.partial(exact, n=6), effective, N_AVG).

The closure of a bin's triangles is here too: triangle_beta, the fraction that close (BkForecast.V123), and
closure_profile, how that varies across one side's shell (BBCovBk's n_delta).

The power spectrum follows the same option through BBCovBk's n_avg: P on N_DELTA nodes across each k-shell, averaged
over its modes (PkForecast.bin_mean), whenever B is averaged over the bin - on the centre with 'centre'.
"""

import itertools

import numpy as np

from cosmo_wap.lib import utils

N_AVG = 4  # nodes a side for the bin average of B - exact's n and BBCovBk's n_avg
N_DELTA = 3  # nodes across a k-shell for its modes - BBCovBk's n_delta and the P of PkForecast.bin_mean


class TriangleSet:
    """Triangles for a set of bins: sides (3, N_eval), each one's bin (N_eval,) - bin-major, each bin at least once -
    and weights, summing to 1 over each bin's"""

    def __init__(self, sides, bins, weights):
        self.sides, self.bins, self.weights = np.asarray(sides), np.asarray(bins), np.asarray(weights)
        self.start = np.flatnonzero(np.diff(self.bins, prepend=-1))  # each bin's first triangle

    def __len__(self):
        return len(self.bins)

    def mean(self, arr):
        """(..., N_eval) on these triangles -> (..., N_bins), each bin the weighted mean over its triangles"""
        if len(self.bins) == len(self.start) or np.ndim(arr) == 0 or np.shape(arr)[-1] == 1:
            return arr  # one triangle per bin, or the same on every triangle
        return np.add.reduceat(arr * self.weights, self.start, axis=-1)


def relabellings(k1, k2, k3):
    """For each bin the permutations of its sides that keep each in its shell - 2 on isosceles bins, 6 on equilateral.
    The estimator sums every labelling, while the multipoles are taken about k1 (and k2 for m > 0)"""
    perms = []
    for a, b, c in zip(k1, k2, k3):
        if np.isclose(a, b) and np.isclose(b, c):
            perms.append(list(itertools.permutations(range(3))))
        elif np.isclose(a, b):
            perms.append([(0, 1, 2), (1, 0, 2)])
        elif np.isclose(b, c):
            perms.append([(0, 1, 2), (0, 2, 1)])
        else:
            perms.append([(0, 1, 2)])
    return perms


def relabel(tri, perms):
    """tri with each bin's triangles under every one of its perms, its weight shared between them"""
    sides, bins, weights = [], [], []
    for t, (a, b) in enumerate(zip(tri.start, np.append(tri.start[1:], len(tri)))):
        for p in perms[t]:
            sides.append(tri.sides[list(p), a:b])
            bins.append(tri.bins[a:b])
            weights.append(tri.weights[a:b] / len(perms[t]))
    return TriangleSet(np.concatenate(sides, axis=1), np.concatenate(bins), np.concatenate(weights))


def _closing(K, q_a, k_b, dk):
    """(lo, hi): the part of the shell of k_b that closes a triangle with sides K and q_a - none where hi <= lo"""
    return np.maximum(k_b - dk / 2, np.abs(K - q_a)), np.minimum(k_b + dk / 2, K + q_a)


def bin_nodes(K, k_a, k_b, dk, n, norm=True):
    """The bin's triangles with a side of length K and the others in the shells of k_a, k_b: n x n nodes (q_a, q_b)
    and their weights, as closure_profile - weighted q_a q_b as the triangle count, q_a by Gauss-Legendre across its
    shell and q_b across the part of its shell that closes (so no step at closure). Normalised for each K, or with
    norm=False left as its count - 4/dk x the integral of q_a q_b over the part of the two shells that closes.
    n = 0 is the bin centre alone. Sides (3, N, n^2) and weights (N, n^2)"""
    if not n:
        return np.array([K, k_a, k_b])[..., np.newaxis], np.ones((len(K), 1))
    x, w = utils.leggauss(n)
    K, k_a, k_b = K[:, np.newaxis], k_a[:, np.newaxis], k_b[:, np.newaxis]
    q_a = k_a + dk * x / 2
    lo, hi = _closing(K, q_a, k_b, dk)
    width = np.clip(hi - lo, 0, None)
    lo = np.where(width > 0, lo, np.maximum(K, q_a))  # where nothing closes: any closed triangle, weight 0
    q_b = lo[..., np.newaxis] + width[..., np.newaxis] * (x + 1) / 2
    weight = w[:, np.newaxis] * w * width[..., np.newaxis] * q_a[..., np.newaxis] * q_b
    if norm:
        total = np.sum(weight, axis=(1, 2), keepdims=True)
        weight = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)  # 0 where K closes nothing
    sides = np.broadcast_arrays(K[..., np.newaxis], q_a[..., np.newaxis], q_b)
    return np.array(sides).reshape(3, len(K), -1), weight.reshape(len(K), -1)


def closure_profile(K, k_a, k_b, dk, n=16):
    """Fraction of the shells of k_a and k_b that close a triangle with a side of length K, weighted q_a q_b as the
    triangle count - 1 unless the bin is (nearly) flattened. Averaged over K's shell (weighted by K) it is the bin's
    beta, see triangle_beta. q_a by Gauss-Legendre, q_b analytic"""
    x, w = utils.leggauss(n)
    q_a = k_a[..., np.newaxis] + dk * x / 2
    lo, hi = _closing(K[..., np.newaxis], q_a, k_b[..., np.newaxis], dk)
    q_b_int = np.where(hi > lo, (hi**2 - lo**2) / 2, 0)  # int q_b dq_b over the part of shell b that closes
    return np.sum(w / 2 * q_a * q_b_int, axis=-1) / (k_a * k_b * dk)


def triangle_beta(k1, k2, k3, dk, n=16):
    """Closure fraction <q1 q2 q3 Theta(q2+q3-q1)>/(k1 k2 k3) of a bin - beta of Eq 24, 1610.06585.

    The 1/2 usually quoted on k1 = k2+k3 is its thin-bin limit; at s_k=4 it is nearer 0.6.
    u is the distance from closure in bin widths, and beta is exactly 1 for u >= 3/2 and 0
    for u <= -3/2, so only the boundary shell is quadratured - over q2,q3, the q1 cut being
    analytic.
    """
    u = (k2 + k3 - k1) / dk
    beta = np.where(u >= 1.5, 1.0, 0.0)
    edge = np.abs(u) < 1.5
    if not np.any(edge):
        return beta

    nodes, weights = utils.leggauss(n)  # read-only, so rescale into new arrays
    d, w = nodes / 2.0, weights / 2.0  # -> [-1/2, 1/2], weights then sum to 1
    d2, d3 = d[:, np.newaxis], d[np.newaxis, :]

    k1e, k2e, k3e = (arr[edge][:, np.newaxis, np.newaxis] for arr in (k1, k2, k3))
    hi = np.clip(u[edge][:, np.newaxis, np.newaxis] + d2 + d3, -0.5, 0.5)
    q1_int = k1e * (hi + 0.5) + dk * (hi**2 - 0.25) / 2  # -> 0 where the bin is excluded

    integrand = q1_int * (k2e + dk * d2) * (k3e + dk * d3)
    beta[edge] = np.einsum("ijk,j,k->i", integrand, w, w) / (k1e * k2e * k3e)[:, 0, 0]
    return beta


def centre(k1, k2, k3, dk):
    """The bin centres"""
    return TriangleSet(np.array([k1, k2, k3]), np.arange(len(k1)), np.ones(len(k1)))


def effective(k1, k2, k3, dk, n=16, chunk=256):
    """Each bin's effective triangle under every relabelling of its equal-shell sides.

    The sorted effective triangle of 1908.01774 (eq 3.8): the mean longest, middle and shortest side over the bin's
    closed triangles, weighted q1 q2 q3 as their count - closed, as a mean of closed triangles, and never flattened.
    Against the exact bin average of the tree-level redshift-space B_0 it is within 2-9%, where the bin centre is off by
    up to 70% on k3 ~ dk: the centre of (i, i-1, 1) is the collinear maximum of the squeezed response, of (i, i, 1) the
    perpendicular minimum. Its <P P P> is within 2% (the centre up to 15%).

    The estimator sums every labelling of sides in the same shell - with the multipoles of k1 (and k2 for m > 0), so on
    k1 = k2 the mean longest side has the Legendre half the time. So each bin is the average over those relabellings:
    2 on isosceles bins, 6 on equilateral ones. Shells do not overlap, so sorting only permutes within them.
    """
    x, w = utils.leggauss(n)
    x, w = x / 2, w / 2  # [-1/2, 1/2] bin widths, weights sum to 1
    ks = np.array([k1, k2, k3])
    eff = np.empty_like(ks)
    for start in range(0, ks.shape[1], chunk):
        k = ks[:, start : start + chunk, None, None, None]
        q = np.stack(np.broadcast_arrays(k[0] + dk * x[:, None, None], k[1] + dk * x[:, None], k[2] + dk * x))
        closed = (q[0] <= q[1] + q[2]) & (q[1] <= q[0] + q[2]) & (q[2] <= q[0] + q[1])
        weight = w[:, None, None] * w[:, None] * w * np.prod(q, axis=0) * closed  # (N, n, n, n)
        eff[:, start : start + chunk] = np.sum(np.sort(q, axis=0)[::-1] * weight, axis=(2, 3, 4)) / np.sum(
            weight, axis=(1, 2, 3)
        )
    return relabel(centre(*eff, dk), relabellings(k1, k2, k3))


def exact(k1, k2, k3, dk, n=N_AVG):
    """n^3 Gauss-Legendre nodes over each bin's closed triangles, weighted q1 q2 q3 as their count - the mean of B on
    them is its bin average, to the quadrature.

    Two sides span their shells and the third the part of its shell that closes with them (bin_nodes), so closure is no
    step in the integrand; nodes where nothing closes are dropped. The third is k3, or k1 where only k2 and k3 share a
    shell - so the nodes are symmetric in the sides that share one, and already hold the mean over the relabellings the
    estimator sums. The exception is an equilateral bin that does not always close, (1, 1, 1), which is relabelled.
    n = N_AVG as the bin average of BBCovBk (n_avg)."""
    ks = np.array([k1, k2, k3])
    N = ks.shape[1]
    # sides in the order (spanning, spanning, closing)
    order = np.where(np.isclose(k2, k3) & ~np.isclose(k1, k2), [[1], [2], [0]], [[0], [1], [2]])
    k_c, k_a, k_b = np.take_along_axis(ks, order, axis=0)
    x, w = utils.leggauss(n)
    q_c = k_c[:, np.newaxis] + dk * x / 2  # (N, n)
    nodes, weights = bin_nodes(q_c.ravel(), np.repeat(k_a, n), np.repeat(k_b, n), dk, n, norm=False)
    weights = (weights.reshape(N, n, n * n) * (w * q_c)[..., np.newaxis]).reshape(N, -1)
    sides = np.empty((3, N, n**3))
    sides[order, np.arange(N)] = nodes.reshape(3, N, -1)  # back to (k1, k2, k3)

    keep = weights > 0
    bins = np.nonzero(keep)[0]
    tri = TriangleSet(sides[:, keep], bins, weights[keep] / np.bincount(bins, weights[keep])[bins])

    open_eq = np.isclose(k1, k2) & np.isclose(k2, k3) & (k1 < 1.5 * dk)  # (1, 1, 1): closure cuts the k3 nodes
    if np.any(open_eq):
        all6 = list(itertools.permutations(range(3)))
        tri = relabel(tri, [all6 if e else [(0, 1, 2)] for e in open_eq])
    return tri


OPTIONS = {  # bin_avg: (signal and its derivatives, Gaussian covariance, BBCovBk's n_avg)
    "centre": (centre, centre, 0),
    "effective": (effective, effective, N_AVG),
    "exact": (exact, effective, N_AVG),
}
_ALIASES = {False: "centre", True: "effective"}


def option(bin_avg):
    """The name in OPTIONS of a FullForecast(bin_avg) - True is 'effective' and False 'centre'"""
    name = _ALIASES.get(bin_avg, bin_avg) if isinstance(bin_avg, (bool, np.bool_)) else bin_avg
    if name not in OPTIONS:
        raise ValueError(f"bin_avg must be True, False or one of {list(OPTIONS)} - not {bin_avg!r}")
    return name
