"""Optional numba kernels for the two hot blocks of the numeric-mu power spectrum.

Both blocks are scalar work on small arrays - a spline evaluation with a breakpoint search
per point, and a Filon quadrature over a handful of nodes - so numpy pays more in temporaries
and per-operation overhead than in arithmetic. Called in isolation on a sampler-sized grid
(12 k-values, 48 mu, 8 line-of-sight nodes) that is 3-4x: kernel_sum 235 -> 59 us on the II
grid and 40 -> 15 us on IS, filon_integrate 99 -> 37 us.

Inside a sampler it is smaller, and that is the number to quote: kernel_sum 1.70 -> 1.15 ms
per likelihood call, filon_integrate 0.75 -> 0.35 ms, 1.15x on get_multipoles and ~5% of a
whole fast step. The spline coefficient array is 640 kB per redshift bin and cold on every
call, so what sets the floor there is memory traffic rather than arithmetic - which is also
why repacking it buys nothing (see _kernel_sum_loop).

numba is optional (``pip install cosmowap[fast]``) and lib.jit decides whether it is used.
Each jitted block sits next to the numpy one it replaces and the choice is bound once, at
import, so the call sites carry no branch - the same arrangement cosmo_wap.bk uses for its
compiled kernels. tests/test_numeric_mu_accel.py runs both and checks they agree.

Threading is deliberately absent: the blocks are ~35 us, well under numba's parallel dispatch
cost, and measured slower with prange at any thread count. The compiled bk kernels are the
place to spend cores (see bk/c_compile.py).
"""

import numpy as np

from cosmo_wap.lib.jit import have_numba, njit

from .integration import filon_integrate as _filon_np

HAVE_NUMBA = have_numba()


def _kernel_sum_np(spline, powers, weights, qq, mu):
    """sum_m weights[m] * I_m(q*mu) * q**j * mu**i over the kernel terms powers[m] = (i, j).

    Only q**j and mu**i vary between terms and most are trivial, so cache the powers and
    skip the zero ones."""
    r1_arr = spline(qq * mu)  # all kernel terms in one spline call
    q_pows, mu_pows = {}, {}
    tot = 0
    for m, (i, j) in enumerate(powers):
        term = r1_arr[..., m] * weights[m]
        if j:
            if j not in q_pows:
                q_pows[j] = qq**j
            term = term * q_pows[j]
        if i:
            if i not in mu_pows:
                mu_pows[i] = mu**i
            term = term * mu_pows[i]
        tot = tot + term
    return tot


@njit(cache=True, fastmath=True, inline="always")
def _int_pow(v, n):
    """v**n for small integer n - ** on a negative exponent goes out to libm's pow."""
    if n == 0:
        return 1.0
    if n < 0:
        v = 1.0 / v
        n = -n
    out = 1.0
    for _ in range(n):
        out *= v
    return out


@njit(cache=True, fastmath=True)
def _kernel_sum_loop(x, c, q, m, i_pow, j_pow, wt, out):
    """One pass over the evaluation points: breakpoint search, Horner over the spline
    components, then the survey weight wt[n] and the q**j mu**i factors - no temporaries.

    c is scipy's (4, n_int, n_term) coefficient array as it comes: transposing it to put one
    interval contiguous measures the same and costs more to build than it saves. Applying wt
    here rather than folding it into c is what makes that array reusable across calls - see
    get_int_K1."""
    nx = x.shape[0]
    n_term = c.shape[2]
    for t in range(q.size):
        qv = q[t]
        mv = m[t]
        v = qv * mv

        lo = 0  # last breakpoint at or below v
        hi = nx - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if x[mid] <= v:
                lo = mid
            else:
                hi = mid

        dx = v - x[lo]
        tot = 0j
        for n in range(n_term):
            val = ((c[0, lo, n] * dx + c[1, lo, n]) * dx + c[2, lo, n]) * dx + c[3, lo, n]
            tot += val * wt[n] * (_int_pow(qv, j_pow[n]) * _int_pow(mv, i_pow[n]))
        out[t] = tot


def _kernel_sum_nb(spline, powers, weights, qq, mu):
    qb, mb = np.broadcast_arrays(qq, mu)
    q = np.ascontiguousarray(qb).ravel()
    m = np.ascontiguousarray(mb).ravel()
    out = np.empty(q.size, dtype=np.complex128)
    i_pow = np.array([i for i, _ in powers], dtype=np.int64)
    j_pow = np.array([j for _, j in powers], dtype=np.int64)
    _kernel_sum_loop(spline.x, spline.c, q, m, i_pow, j_pow, weights, out)
    return out.reshape(qb.shape)


@njit(cache=True, fastmath=True)
def _filon_loop(u, w, integrand, out):
    """Filon quadrature of f(u) exp(i*w*u) over the last axis - the weights and the
    exponentials are carried along the node loop rather than built as full arrays. Same
    weights and small-theta series as integration.filon_integrate; E_right is reused as the
    next interval's E_left, so each node's exponential is evaluated once."""
    n_k, n_mu, n_u = integrand.shape
    for a in range(n_k):
        for b in range(n_mu):
            ww = w[a, b]
            phase = ww * u[0]
            e_left = complex(np.cos(phase), np.sin(phase))
            tot = 0j
            for n in range(n_u - 1):
                du = u[n + 1] - u[n]
                phase = ww * u[n + 1]
                e_right = complex(np.cos(phase), np.sin(phase))

                theta = ww * du  # phase change over the interval
                if abs(theta) < 1e-3:  # series in theta - the closed form cancels catastrophically
                    du_e = du * e_left
                    w_right = du_e * (0.5 + 1j * theta / 3 - theta**2 / 8)
                    w_left = du_e * (0.5 + 1j * theta / 6 - theta**2 / 24)
                else:
                    inv_iw = 1.0 / (1j * ww)
                    v0 = (e_right - e_left) * inv_iw
                    v1 = (du * e_right - v0) * inv_iw
                    w_right = v1 / du
                    w_left = v0 - w_right

                tot += w_left * integrand[a, b, n] + w_right * integrand[a, b, n + 1]
                e_left = e_right
            out[a, b] = tot


def _filon_nb(u, kk, mu, integrand, d):
    u = np.ascontiguousarray(u)  # s1_sum passes both of these reversed, so as strided views
    integrand = np.ascontiguousarray(integrand)
    w = np.ascontiguousarray(np.broadcast_to(d * kk * mu, integrand.shape)[..., 0])  # one per (k, mu)
    out = np.empty(integrand.shape[:-1], dtype=np.complex128)
    _filon_loop(u, w, integrand, out)
    return out


# bound once - the call sites in pk.py carry no branch
kernel_sum = _kernel_sum_nb if HAVE_NUMBA else _kernel_sum_np
filon_integrate = _filon_nb if HAVE_NUMBA else _filon_np
