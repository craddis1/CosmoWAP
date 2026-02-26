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
        z_data: Complex function values Z(r)

    Returns:
        I_total: Complex array of integral values
        coeffs_u: Complex polynomial coefficients of Z(u)
    """
    # Normalize radial coordinate to prevent numerical overflow
    # Map r → u where u ∈ [0, 1], avoiding large powers like (3000)^5
    u_data = r_data / d

    # Fit Z(u) as a complex polynomial: Z(u) = Σ c_j * u^j
    # NumPy handles complex coefficients automatically
    coeffs_u = np.polyfit(u_data, z_data, deg=deg)
    n = len(coeffs_u) - 1

    # Define dimensionless parameter φ = p * d
    # This controls which integration method to use
    phi = p_arr * d

    # Initialize result array
    I_total = np.zeros_like(p_arr, dtype=complex)

    # Choose method based on |φ| magnitude:
    # Small |φ|: Taylor series converges rapidly and avoids division by zero
    # Large |φ|: Recurrence relation is stable
    mask_taylor = np.abs(phi) < 0.5
    mask_recurrence = ~mask_taylor

    # TAYLOR SERIES METHOD (for |φ| < 0.5)
    # Expand exp(i*φ*u) = Σ (i*φ)^k / k! * u^k
    if np.any(mask_taylor):
        phi_small = phi[mask_taylor]
        res_taylor = np.zeros_like(phi_small, dtype=complex)

        n_taylor = 15  # Terms needed for machine precision

        for k in range(n_taylor):
            # Exponential expansion coefficient
            exp_term = (1j * phi_small) ** k / factorial(k)

            # Integrate polynomial term-by-term
            # For each coefficient c_j with power (n-j), compute:
            # ∫₀¹ c_j * u^(n-j+k) du = c_j / (n-j+k+1)
            poly_int_val = 0j  # Initialize as complex
            for j, c_val in enumerate(coeffs_u):
                pow_u = n - j
                poly_int_val += c_val / (pow_u + k + 1)

            res_taylor += exp_term * poly_int_val

        I_total[mask_taylor] = res_taylor * d

    # ANALYTIC RECURRENCE METHOD (for |φ| ≥ 0.5)
    # Use recurrence: F_k = exp(i*φ)/φ - k*F_{k-1}/φ
    if np.any(mask_recurrence):
        phi_large = phi[mask_recurrence]

        # Base case: F_0 = ∫₀¹ exp(i*φ*u) du
        ip = 1j * phi_large
        exp_ip = np.exp(ip)
        F_current = (exp_ip - 1.0) / ip

        # Start accumulation with constant term (coeffs_u[-1] is c_0)
        recurrence_res = coeffs_u[-1] * F_current

        # Apply recurrence for powers k = 1 to n
        for k in range(1, n + 1):
            # Compute F_k = ∫₀¹ u^k * exp(i*φ*u) du
            F_next = exp_ip / ip - (k / ip) * F_current

            # Add contribution from coefficient c_k
            c_k = coeffs_u[-(k + 1)]
            recurrence_res += c_k * F_next
            F_current = F_next

        I_total[mask_recurrence] = recurrence_res * d

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

    w[w == 0] = 1e-15

    # Compute Intervals (du) along the last axis
    # du shape: (N_u - 1)
    du = np.diff(u, axis=-1)

    # Left and Right u-points
    ul = u[:-1]
    ur = u[1:]

    # Compute Exact Exponentials
    # Shape: (N_k, N_mu, N_u - 1)
    inv_iw = 1.0 / (1j * w)
    E_left = np.exp(1j * w * ul)
    E_right = np.exp(1j * w * ur)

    # Filon Weights (Standard formulas, just propagated in 3D)
    v0 = (E_right - E_left) * inv_iw
    v1 = (du * E_right * inv_iw) - (v0 * inv_iw)

    W_right = v1 / du
    W_left = v0 - W_right

    # Apply Weights to Integrand
    # integrand shape must be (N_k, N_mu, N_u)
    f_left = integrand[:, :, :-1]
    f_right = integrand[:, :, 1:]

    segments = (W_left * f_left) + (W_right * f_right)

    # Sum over the last axis (u)
    # Result shape: (N_k, N_mu)
    return np.sum(segments, axis=-1)
