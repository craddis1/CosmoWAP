import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_beta_funcs(cf,ti = 0):
    """
    Function that relies on biases and functions defined in ClassWAP to return beta coefficients (beta expressions adapted from Eline de Weerd GitHub) from paper 1711.01812v4
        
    Calculate and return beta coefficients values for given redshift and tracer
    
    Parameters:
    -----------
    cf : ClassWAP instance
        The cosmology and function class instance
    tracer : object
        Survey tracer object
        
    Returns:
    --------
    list of beta values
    """
    tracer = cf.survey[ti]

    #Remove nested interpolators!
    # ok so lets change things so we only interpolate at the end and work normally with arrays in redshift till then

    #these derivs here are wrt to conformal time so we convert to derivs wrt to z
    #d/dt = da/dt d/da = a H dz/da d/dz =  -(1+z) H d/dz # everything here is conformal both t and H
    #d^2/d^2 t = (1+z)^2 H^2 d^2/d z^2 + H(1+z)(H+(1+z)H')d/dz

    zz = tracer.z_survey

    Q = tracer.Q(zz)
    b_e = tracer.be(zz)
    b_1 = tracer.b_1(zz)

    #derivatives wrt redshift
    dQ_dz  = np.gradient(Q,zz,edge_order=2)
    dbe_dz = np.gradient(b_e,zz,edge_order=2)
    db1_dz = np.gradient(b_1,zz,edge_order=2)

    #reduce intepolations
    H_c = cf.H_c(zz)
    f = cf.f(zz)
    Om = cf.Om_m(zz)
    xi = cf.comoving_dist(zz)

    ddH_c    = cf.H_c.derivative(nu=2) # second derivative wrt z
    #derivatives wrt conformal time
    dH_dt = -(1+zz)*H_c*cf.dH_c(zz)
    dH_dt2 = (1+zz)**2 *H_c**2 *ddH_c(zz)+H_c*(1+zz)*(H_c+(1+zz)*cf.dH_c(zz))*cf.dH_c(zz)# can also do numerically
    #dH22 = np.gradient(dH_dt(cf.z_cl),cf.conf_time(cf.z_cl))
                
    dQ_dt  = -(1+zz)*H_c*dQ_dz
    dbe_dt = -(1+zz)*H_c*dbe_dz
    db1_dt = -(1+zz)*H_c*db1_dz

    # generally set these partial derivatives to 0
    partdQ=0
    partdb1=0

    # for 1st order petrubation theory
    tracer.gr1 = CubicSpline(zz, H_c*f*(b_e-2*Q-2*(1-Q)/(xi*H_c)-dH_dt/H_c**2))
    tracer.gr2 = CubicSpline(zz, H_c**2 *(f*(3-b_e) + (3/2)*Om*(2+b_e-f-4*Q-2*(1-Q)/(xi*H_c)-dH_dt/H_c**2)))

    # for second order pertubration theory
    beta = np.empty(20,dtype=object)
    beta[6]  = CubicSpline(zz, H_c**2 * ((3 / 2) * Om * (2 - 2 * f + b_e - 4 * Q - ((2 * (1 - Q)) / (xi * H_c)) - (dH_dt/ H_c**2))))
    beta[7]  = CubicSpline(zz, H_c**2 * (f* (3 - b_e)))
    beta[8]  = CubicSpline(zz, H_c**2 * (3 * Om * f * (2 - f - 2 * Q) + f**2 * (4 + b_e - b_e**2 + 4 * b_e * Q - 6 * Q - 4 * Q**2 + 4 * partdQ + 4 * (dQ_dt / H_c) -(dbe_dt / H_c) - (2 / (xi**2 * H_c**2)) * (1 - Q + 2 * Q**2 - 2 * partdQ) - (2 / (xi * H_c)) * (3 - 2 * b_e + 2 * b_e * Q - Q - 4 * Q**2 + ((3 * dH_dt) / (H_c**2)) * (1 - Q) + 4 * partdQ + 2 * (dQ_dt / H_c)) - (dH_dt/ H_c**2) * (3 - 2 * b_e + 4 * Q + ((3 * dH_dt) / (H_c**2))) + (dH_dt2 / H_c**3))))
    beta[9]  = CubicSpline(zz, H_c**2 * ( -(9 / 2) * Om * f))
    beta[10] = CubicSpline(zz, H_c**2 * (3 * Om * f))
    beta[11] = CubicSpline(zz, H_c**2 * ( (3/2 ) * Om * (1 + 2 * f/ (3 * Om)) + 3 * Om * f- f**2 * (-1 + b_e - 2 * Q - ((2 * (1 + Q)) / (xi * H_c)) - (dH_dt/ H_c**2))))
    beta[12] = CubicSpline(zz, H_c**2 * ( -3 * Om * (1 + 2 * f/ (3 * Om)) - f* ( b_1 * (f- 3 + b_e) + (db1_dt / H_c) ) + (3 / 2) * Om * (b_1 * (2 + b_e - 4 * Q - 2 * ((1 - Q)/(xi * H_c)) - (dH_dt/ H_c**2) ) + db1_dt /H_c + 2 * (2 - (1 / (xi * H_c)) ) * partdb1 ) ))    
    beta[13] = CubicSpline(zz, H_c**2 * (( (9 / 4) * Om**2 + (3 / 2) * Om * f* (1 - (2 * f) + 2 * b_e - 6 * Q - ((4 * (1 - Q))/(xi * H_c)) - ((3 * dH_dt) / H_c**2) ) ) + ( f**2 * (3 - b_e) ) ))
    beta[14] = CubicSpline(zz, H_c * ( - (3 / 2) * Om * b_1))
    beta[15] = CubicSpline(zz, H_c * 2 * f**2)
    beta[16] = CubicSpline(zz, H_c * (f* (b_1 * (f+ b_e - 2 * Q - ((2 * (1 - Q)) / (xi * H_c)) - (dH_dt/ H_c**2)) + (db1_dt / H_c) + 2 * (1 - (1 / (xi * H_c))) * partdb1 )))
    beta[17] = CubicSpline(zz, H_c * (- (3 / 2) * Om * f))
    beta[18] = CubicSpline(zz, H_c * ( (3 / 2) * Om * f - f**2 * (3 - 2 * b_e + 4 * Q + ((4 * (1 - Q)) / (xi * H_c)) + (3 * dH_dt/ H_c**2)) ))
    beta[19] = CubicSpline(zz, H_c * (f* (b_e - 2 * Q - ((2 * (1 - Q)) / (xi * H_c)) - (dH_dt/ H_c**2))))

    return np.concatenate((np.array([tracer.gr1,tracer.gr2]),beta[6:]))