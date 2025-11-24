import numpy as np
from cosmo_wap.integrated import BaseInt
from cosmo_wap.lib import utils

from cosmo_wap.lib.kernels import K1
from scipy.interpolate import CubicSpline, RegularGridInterpolator
import scipy
from scipy.special import factorial

def compute_robust_integral(d, p_arr, r_data, z_data,deg=5):
    """
    Computes the integral I(p) = d * ∫₀¹ exp(i*p*d*u) * Z(u) du robustly
    for complex Z(r), large d values, and wide ranges of p.
    
    Uses normalized domain u = r/d and automatically switches between
    Taylor series (small |p*d|) and analytic recurrence (large |p*d|) methods.
    
    Args:
        d: Characteristic length scale
        p_arr: Array of p values to evaluate (can include negative values)
        r_data: Radial coordinate data
        z_data_complex: Complex function values Z(r)
    
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
            exp_term = (1j * phi_small)**k / factorial(k)
            
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

    # 2. Compute Frequency Omega (w)
    # w = d * k * mu
    # Shape becomes (N_k, N_mu, 1)
    # Let's calculate w directly from the phase term logic.
    w = d * kk * mu 
    
    # SAFETY: If mu includes 0, w will be 0. Div by zero error in Filon.
    # Quick fix: replace 0 with epsilon or mask.
    w[w == 0] = 1e-15 

    # 3. Compute Intervals (du) along the last axis
    # du shape: (N_u - 1)
    du = np.diff(u, axis=-1)
    
    # Left and Right u-points
    ul = u[:-1]
    ur = u[1:]
    
    # 4. Compute Exact Exponentials
    # Shape: (N_k, N_mu, N_u - 1)
    inv_iw = 1.0 / (1j * w)
    E_left  = np.exp(1j * w * ul)
    E_right = np.exp(1j * w * ur)
    
    # 5. Filon Weights (Standard formulas, just propagated in 3D)
    v0 = (E_right - E_left) * inv_iw
    v1 = (du * E_right * inv_iw) - (v0 * inv_iw)
    
    W_right = v1 / du
    W_left  = v0 - W_right
    
    # 6. Apply Weights to Integrand
    # integrand shape must be (N_k, N_mu, N_u)
    f_left  = integrand[:, :, :-1]
    f_right = integrand[:, :, 1:]
    
    segments = (W_left * f_left) + (W_right * f_right)
    
    # 7. Sum over the last axis (u)
    # Result shape: (N_k, N_mu)
    return np.sum(segments, axis=-1)

def get_int_K1(kernel,cosmo_funcs,zz,deg=5):
    """Get interpolated dictionary (with mu and k depedence for the r1 integral)"""
    p_arr = np.concatenate((-np.logspace(-6,3,2000)[::-1],np.logspace(-6,3,2000)))
    d = cosmo_funcs.comoving_dist(zz)
    r1 = np.linspace(1e-3,d,1000)

    arr_dict = {}
    for kern in kernel:
        func = getattr(K1,kern)

        tmp_dict = func(r1,cosmo_funcs,zz=zz,tracer=0)

        # loop over dict to merge them
        for i in tmp_dict.keys():
            if i in arr_dict:
                for j in tmp_dict[i].keys():
                        if j in arr_dict[i]:
                            arr_dict[i][j] = arr_dict[i][j] + tmp_dict[i][j] # Element-wise addition
                        else:
                            arr_dict[i][j] = tmp_dict[i][j]
            else:
                arr_dict[i] = tmp_dict[i].copy()

    # now get interpolated function in p for all kernels
    for i in arr_dict.keys():
        for j in arr_dict[i].keys():
            I_arr, _ = compute_robust_integral(d, p_arr, r1, arr_dict[i][j],deg=deg)
            arr_dict[i][j] = [CubicSpline(p_arr,I_arr.real),CubicSpline(p_arr,I_arr.imag)]

    return arr_dict

def get_int_K2(kernel,r2,cosmo_funcs,zz,mu,kk,n=128):
    """Get kernel 2 array"""
    d = cosmo_funcs.comoving_dist(zz)
    mu, kk = utils.enable_broadcasting(mu,kk,n=1)
    
    G = r2/d
    qq = kk/G
    k2_arr = np.zeros((r2*mu*kk).shape,dtype=np.complex128)

    for kern in kernel:
        func = getattr(K1,kern)
        k2_arr += func(r2,cosmo_funcs,zz,-mu,qq,tracer=1)

    return k2_arr

def I1_sum(arr_dict,r2_arr,mu,kk,cosmo_funcs,zz,n=128,I2=False):
    
    baseint = BaseInt(cosmo_funcs)
    d = cosmo_funcs.comoving_dist(zz)
    if I2: # II
        nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given
        r2 = (d)*(nodes+1)/2.0 # sample r range [0,d] 
        mu, kk = utils.enable_broadcasting(mu,kk,n=1)
        G = r2/d
        qq = kk/G
    else:
        #only IS
        qq=kk

    tot_arr = np.zeros((len(kk),len(mu)),dtype=np.complex128) # shape (mu,kk)
    for i in arr_dict.keys():
        for j in arr_dict[i].keys():
            coef = qq**j * mu**i # this is the mu from first order field
            r1_arr  = (arr_dict[i][j][0](qq*mu) + 1j*arr_dict[i][j][1](qq*mu)) # get complex array
            tmp_arr = np.exp(-1j *kk*mu*d)*coef*r2_arr*r1_arr
            if I2:
                tot_arr += ((d) / 2.0) *np.sum(G**(-3)* baseint.pk(qq,zz)*weights *tmp_arr,axis=-1)
            else:
                tot_arr += baseint.pk(qq,zz)*tmp_arr
    
    return tot_arr

def s1_sum(s_k1,r2_arr,mu,kk,cosmo_funcs,zz,n=128,I2=False,new=True,tmp=None):
    """Now for case where S if first field"""
    baseint = BaseInt(cosmo_funcs)
    if I2: # SI
        d = cosmo_funcs.comoving_dist(zz)
        if new:
            # lets sample log
            r2 = np.logspace(np.log10(0.01), np.log10(d), n)

            # u = d / r = 1 / G
            u = d / r2 # Note: u_grid now goes from 1.0 to (d/r_min)

            # Jacobian from Eq 31 transformation: J = d * u
            # (Recall: dr = -d/u^2 du, and G^-3 = u^3, so net is d*u)
            Jac = d * u

            r2_arr = get_int_K2(tmp,r2,cosmo_funcs,zz,mu,kk,n=n)
            mu, kk = utils.enable_broadcasting(mu,kk,n=1)
            qq = kk * u

            # Full Integrand
            s1_arr = get_K(s_k1,cosmo_funcs,zz,mu,qq,tracer=0)
            integrand = Jac * baseint.pk(qq,zz)* r2_arr * s1_arr

            # u_grid is now strictly increasing, so Filon works perfectly.
            tot_arr = np.exp(-1j *kk[...,0]*mu[...,0]*d)*filon_integrate(u[::-1], kk, mu, integrand[...,::-1], d)
        else:
            nodes, weights = np.polynomial.legendre.leggauss(n) #legendre gauss - get nodes and weights for given
            r2 = (d)*(nodes+1)/2.0 # sample r range [0,d]
            mu, kk = utils.enable_broadcasting(mu,kk,n=1)
            G = r2/d
            qq = kk/G

            s1_arr = get_K(s_k1,cosmo_funcs,zz,mu,qq,tracer=0)
            tot_arr = ((d) / 2.0) *np.sum(np.exp(-1j *kk*mu*d)*np.exp(1j *d*qq*mu) *G**(-3)*weights*baseint.pk(qq,zz)*s1_arr*r2_arr,axis=-1)
    else:
        #only SS
        s1_arr = get_K(s_k1,cosmo_funcs,zz,mu,kk,tracer=0)
        tot_arr = baseint.pk(kk,zz)*r2_arr*s1_arr

    return tot_arr

def split_kernels(kernels):
    """Seperate integrated and normal kernels"""
    int_kernels = set(['I','L','TD','ISW','L1']) # these are the integrated kernels - will need updating if add more
    if not isinstance(kernels, list):
        kernels = [kernels]
    return [s for s in kernels if s in int_kernels],[s for s in kernels if s not in int_kernels]

def get_K(kernels,cosmo_funcs,zz,mu,kk,tracer=0):
    tot_arr = np.zeros((kk*mu).shape,dtype=np.complex128)
    for kern in kernels:
        func = getattr(K1,kern)
        tot_arr += func(cosmo_funcs,zz,mu,kk,tracer=tracer)
    return tot_arr

def get_mu(mu,kernels1,kernels2,cosmo_funcs,kk,zz,n=16,deg=8,new=True):
    """Collect power spectrum contribution"""

    kk = kk[:,np.newaxis]

    d = cosmo_funcs.comoving_dist(zz)
    nodes, weights = np.polynomial.legendre.leggauss(n)#legendre gauss - get nodes and weights for given
    r2 = (d)*(nodes+1)/2.0 # sample r range [0,d] 

    # lets split kernels into integrated and not!
    int_k1,s_k1  = split_kernels(kernels1)
    int_k2,s_k2  = split_kernels(kernels2)

    # precompute ------------------------------------------------------------------------
    if s_k2:
        s2_arr = get_K(s_k2,cosmo_funcs,zz,-mu,kk,tracer=1)
    if int_k1:
        arr_dict = get_int_K1(int_k1,cosmo_funcs,zz,deg=deg) # is dict
    if int_k2:
        r2_arr = get_int_K2(int_k2,r2,cosmo_funcs,zz,mu,kk,n=n) # is array
    
    # sum stuff -----------------------------------------------------------------------------
    tot_arr = np.zeros((kk*mu).shape,dtype=np.complex128) # shape (mu,kk)
    if int_k1:
        if int_k2: #II
            tot_arr += I1_sum(arr_dict,r2_arr,mu,kk,cosmo_funcs,zz,n=n,I2=True)
        if s_k2: #IS
            tot_arr += I1_sum(arr_dict,s2_arr,mu,kk,cosmo_funcs,zz,I2=False)

    if s_k1:
        if int_k2: #SI
            tot_arr += s1_sum(s_k1,r2_arr,mu,kk,cosmo_funcs,zz,n=n,I2=True,new=new,tmp=int_k2)
        if s_k2: # SS
            tot_arr += s1_sum(s_k1,s2_arr,mu,kk,cosmo_funcs,zz,I2=False)

    return tot_arr

def get_multipole(kernel1,kernel2,l,cosmo_funcs,kk,zz,sigma=None,n=16,n_mu=512,deg=8,new=True):
    mu, weights = np.polynomial.legendre.leggauss(n_mu)#legendre gauss - get nodes and weights for given n

    arr = get_mu(mu,kernel1,kernel2,cosmo_funcs,kk,zz,n=n,deg=deg,new=new)

    # get legendre
    leg = scipy.special.eval_legendre(l,mu)
        
    if sigma is None: #no FOG
        dfog_val = 1
    else:
        dfog_val = np.exp(-(1/2)*((kk*mu)**2)*sigma**2) # exponential damping

    return ((2*l+1)/2)*np.sum(weights*leg*dfog_val*arr,axis=-1)