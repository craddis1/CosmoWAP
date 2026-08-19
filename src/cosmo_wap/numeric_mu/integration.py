import numpy as np
from scipy.special import factorial


def compute_robust_integral(d, p_arr, r_data, z_data, deg=5):
    """
    Computes the integral I(p) = d * ∫₀¹ exp(i*p*d*u) * Z(u) du robustly
    for complex Z(r), large d values, and wide ranges of p.

    Uses normalized domain u = r/d and automatically switches between
    Taylor series (small |p*d|) and analytic recurrence (large |p*d|) methods.

    Args:
        d: Characteristic length scale
        p_arr: Array of p values to evaluate (can include negative values)
        r_data: Radial coordinate data
        z_data: Complex function values Z(r), shape (N_r, N_terms) - the polynomial fit and
            the exponentials are shared by every column, so pass all kernel terms at once

    Returns:
        I_total: Complex array of integral values, shape (N_p, N_terms)
        coeffs_u: Complex polynomial coefficients of Z(u), shape (deg+1, N_terms)
    """
    # Fit Z(u) as a complex polynomial: Z(u) = Σ c_j * u^j, on the normalized radial
    # coordinate u = r/d, which avoids large powers like (3000)^5. NumPy handles complex
    # coefficients automatically, and fits every column of z_data in the one lstsq.
    coeffs_u = np.polyfit(r_data / d, z_data, deg=deg)
    n = len(coeffs_u) - 1

    # Define dimensionless parameter φ = p * d
    # This controls which integration method to use
    phi = p_arr * d

    I_total = np.zeros((p_arr.size, z_data.shape[1]), dtype=complex)

    # Choose method based on |φ| magnitude:
    # Small |φ|: Taylor series converges rapidly and avoids division by zero
    # Large |φ|: Recurrence relation is stable
    mask_taylor = np.abs(phi) < 0.5
    mask_recurrence = ~mask_taylor

    # TAYLOR SERIES METHOD (for |φ| < 0.5)
    # Expand exp(i*φ*u) = Σ (i*φ)^k / k! * u^k and integrate term-by-term:
    # for each coefficient c_j with power (n-j), ∫₀¹ c_j * u^(n-j+k) du = c_j / (n-j+k+1)
    if np.any(mask_taylor):
        k = np.arange(15)  # terms needed for machine precision
        pow_u = n - np.arange(n + 1)
        poly_int = (coeffs_u[:, None, :] / (pow_u[:, None] + k + 1)[:, :, None]).sum(axis=0)  # (n_taylor, N_terms)
        exp_term = (1j * phi[mask_taylor, None]) ** k / factorial(k)  # (N_p, n_taylor)
        I_total[mask_taylor] = (exp_term @ poly_int) * d

    # ANALYTIC RECURRENCE METHOD (for |φ| ≥ 0.5)
    # F_k = ∫₀¹ u^k * exp(i*φ*u) du via F_k = exp(i*φ)/(i*φ) - k*F_{k-1}/(i*φ)
    if np.any(mask_recurrence):
        phi_large = phi[mask_recurrence]

        ip = 1j * phi_large
        exp_ip = np.exp(ip)
        F = (exp_ip - 1.0) / ip  # base case: F_0 = ∫₀¹ exp(i*φ*u) du

        # Start accumulation with the constant term (coeffs_u[-1] is c_0), then apply the
        # recurrence for powers k = 1 to n
        res = F[:, None] * coeffs_u[-1]
        for k in range(1, n + 1):
            F = exp_ip / ip - (k / ip) * F
            res += F[:, None] * coeffs_u[-(k + 1)]  # c_k

        I_total[mask_recurrence] = res * d

    return I_total, coeffs_u


def filon_integrate(u, kk, mu, integrand, d):
    """
    Computes int f(u, k, mu) * e^{i * d * k * mu * u} du
    Result has shape (N_k, N_mu)
    """

    # Compute Frequency Omega (w)
    # w = d * k * mu
    # Shape becomes (N_k, N_mu, 1)
    # Let's calculate w directly from the phase term logic.
    w = d * kk * mu

    # Compute Intervals (du) along the last axis
    # du shape: (N_u - 1)
    du = np.diff(u, axis=-1)

    # Compute Exact Exponentials once over the full u grid - left/right points overlap
    # Shape: (N_k, N_mu, N_u)
    E = np.exp(1j * w * u)
    E_left = E[..., :-1]
    E_right = E[..., 1:]

    # Filon Weights (Standard formulas, just propagated in 3D)
    # the standard formulas suffer catastrophic cancellation as the phase change per
    # interval theta -> 0 (e.g. mu = 0) - overwrite those entries with the Taylor series
    theta = w * du
    small = np.abs(theta) < 1e-3

    with np.errstate(divide="ignore", invalid="ignore"):  # w = 0 entries are overwritten below
        inv_iw = 1.0 / (1j * w)

        v0 = (E_right - E_left) * inv_iw
        v1 = (du * E_right - v0) * inv_iw

        W_right = v1 / du
        W_left = v0 - W_right

    if np.any(small):
        # series in theta (leading order is the trapezoid weight du/2)
        th = theta[small]
        du_E = np.broadcast_to(du, small.shape)[small] * E_left[small]
        W_right[small] = du_E * (0.5 + 1j * th / 3 - th**2 / 8)
        W_left[small] = du_E * (0.5 + 1j * th / 6 - th**2 / 24)

    # Apply Weights to Integrand
    # integrand shape must be (N_k, N_mu, N_u)
    f_left = integrand[:, :, :-1]
    f_right = integrand[:, :, 1:]

    segments = (W_left * f_left) + (W_right * f_right)

    # Sum over the last axis (u)
    # Result shape: (N_k, N_mu)
    return np.sum(segments, axis=-1)
