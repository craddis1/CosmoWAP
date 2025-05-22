import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_beta_funcs(cf,tracer = None):
    """
        Function that relies on biases and functions defined above to return beta coefficients (beta expressions adapted from Eline de Weerd GitHub) from paper 1711.01812v4
        
    Calculate and return beta coefficient values for given redshift and tracer
    
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
    if tracer is None:
        tracer = cf.survey

    #these derivs here are wrt to conformal time so we convert to derivs wrt to z
    #d/dt = da/dt d/da = a H dz/da d/dz =  -(1+z) H d/dz # everything here is conformal both t and H
    #d^2/d^2 t = (1+z)^2 H^2 d^2/d z^2 + H(1+z)(H+(1+z)H')d/dz

    dQ_dz  = CubicSpline(tracer.z_survey,np.gradient(tracer.Q_survey(tracer.z_survey),tracer.z_survey))
    dbe_dz = CubicSpline(tracer.z_survey,np.gradient(tracer.be_survey(tracer.z_survey),tracer.z_survey))
    db1_dz = CubicSpline(tracer.z_survey,np.gradient(tracer.b_1(tracer.z_survey),tracer.z_survey))

    #derivatives wrt conformal time
    dH_dt = lambda xx: -(1+xx)*cf.H_c(xx)*cf.dH_c(xx)
    dH_dt2 = lambda xx: (1+xx)**2 *cf.H_c(xx)**2 *cf.ddH_c(xx)+cf.H_c(xx)*(1+xx)*(cf.H_c(xx)+(1+xx)*cf.dH_c(xx))*cf.dH_c(xx)# can also do numerically
    #cf.dH22 = interp1d(cf.z_cl,np.gradient(dH_dt(cf.z_cl),cf.conf_time(cf.z_cl)))
    cf.dH_dt = dH_dt                   
    dQ_dt = lambda xx: -(1+xx)*cf.H_c(xx)*dQ_dz(xx)
    dbe_dt = lambda xx: -(1+xx)*cf.H_c(xx)*dbe_dz(xx)#0*xx-1.6*1e-4
    db1_dt = lambda xx: -(1+xx)*cf.H_c(xx)*db1_dz(xx)  

    cf.Q = tracer.Q_survey  # bit silly but whatever
    cf.b_e = tracer.be_survey

    # generally set these partial derivatives to 0
    partdQ=0
    partdb1=0

    # for 1st order petrubation theory
    tracer.gr1 = lambda xx: cf.H_c(xx)*cf.f_intp(xx)*(cf.b_e(xx)-2*cf.Q(xx)-2*(1-cf.Q(xx))/(cf.comoving_dist(xx)*cf.H_c(xx))-dH_dt(xx)/cf.H_c(xx)**2)
    tracer.gr2 = lambda xx: cf.H_c(xx)**2 *(cf.f_intp(xx)*(3-cf.b_e(xx))+ (3/2)*cf.Om(xx)*(2+cf.b_e(xx)-cf.f_intp(xx)-4*cf.Q(xx)-2*cf.Q(xx)-2*(1-cf.Q(xx))/(cf.comoving_dist(xx)*cf.H_c(xx))-dH_dt(xx)/cf.H_c(xx)**2))
    tracer.grd1 = cf.lnd_derivatives([tracer.gr1])[0]#get derivative for 1st order coef

    # for second order pertubration theory
    beta = np.empty(20,dtype=object)
    beta[6] = lambda xx: cf.H_c(xx)**2 * ((3 / 2) * cf.Om(xx) * (2 - 2 * cf.f_intp(xx)+ cf.b_e(xx) - 4 * cf.Q(xx) - ((2 * (1 - cf.Q(xx))) / (cf.comoving_dist(xx) * cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2)))
    beta[7] = lambda xx: cf.H_c(xx)**2 * (cf.f_intp(xx)* (3 - cf.b_e(xx)))
    beta[8] = lambda xx: cf.H_c(xx)**2 * (3 * cf.Om(xx) * cf.f_intp(xx) * (2 - cf.f_intp(xx) - 2 * cf.Q(xx)) + cf.f_intp(xx)**2 * (4 + cf.b_e(xx) - cf.b_e(xx)**2 + 4 * cf.b_e(xx) * cf.Q(xx) - 6 * cf.Q(xx) - 4 * cf.Q(xx)**2 + 4 * partdQ + 4 * (dQ_dt(xx) / cf.H_c(xx)) -(dbe_dt(xx) / cf.H_c(xx)) - (2 / (cf.comoving_dist(xx)**2 * cf.H_c(xx)**2)) * (1 - cf.Q(xx) + 2 * cf.Q(xx)**2 - 2 * partdQ) - (2 / (cf.comoving_dist(xx) * cf.H_c(xx))) * (3 - 2 * cf.b_e(xx) + 2 * cf.b_e(xx) * cf.Q(xx) - cf.Q(xx) - 4 * cf.Q(xx)**2 + ((3 * dH_dt(xx)) / (cf.H_c(xx)**2)) * (1 - cf.Q(xx)) + 4 * partdQ + 2 * (dQ_dt(xx) / cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2) * (3 - 2 * cf.b_e(xx) + 4 * cf.Q(xx) + ((3 * dH_dt(xx)) / (cf.H_c(xx)**2))) + (dH_dt2(xx) / cf.H_c(xx)**3)))
    beta[9] = lambda xx: cf.H_c(xx)**2 * ( -(9 / 2) * cf.Om(xx) * cf.f_intp(xx))
    beta[10] = lambda xx: cf.H_c(xx)**2 * (3 * cf.Om(xx) * cf.f_intp(xx))
    beta[11] = lambda xx: cf.H_c(xx)**2 * ( (3/2 ) * cf.Om(xx) * (1 + 2 * cf.f_intp(xx)/ (3 * cf.Om(xx))) + 3 * cf.Om(xx) * cf.f_intp(xx)- cf.f_intp(xx)**2 * (-1 + cf.b_e(xx) - 2 * cf.Q(xx) - ((2 * (1 + cf.Q(xx))) / (cf.comoving_dist(xx) * cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2)))
    beta[12] = lambda xx: cf.H_c(xx)**2 * ( -3 * cf.Om(xx) * (1 + 2 * cf.f_intp(xx)/ (3 * cf.Om(xx))) - cf.f_intp(xx)* ( tracer.b_1(xx) * (cf.f_intp(xx)- 3 + cf.b_e(xx)) + (db1_dt(xx) / cf.H_c(xx)) ) + (3 / 2) * cf.Om(xx) * (tracer.b_1(xx) * (2 + cf.b_e(xx) - 4 * cf.Q(xx) - 2 * ((1 - cf.Q(xx))/(cf.comoving_dist(xx) * cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2) ) + db1_dt(xx) /cf.H_c(xx) + 2 * (2 - (1 / (cf.comoving_dist(xx) * cf.H_c(xx))) ) * partdb1 ) )    
    beta[13] = lambda xx: cf.H_c(xx)**2 * (( (9 / 4) * cf.Om(xx)**2 + (3 / 2) * cf.Om(xx) * cf.f_intp(xx)* (1 - (2 * cf.f_intp(xx)) + 2 * cf.b_e(xx) - 6 * cf.Q(xx) - ((4 * (1 - cf.Q(xx)))/(cf.comoving_dist(xx) * cf.H_c(xx))) - ((3 * dH_dt(xx)) / cf.H_c(xx)**2) ) ) + ( cf.f_intp(xx)**2 * (3 - cf.b_e(xx)) ) )
    beta[14] = lambda xx: cf.H_c(xx) * ( - (3 / 2) * cf.Om(xx) * tracer.b_1(xx))
    beta[15] = lambda xx: cf.H_c(xx) * 2 * cf.f_intp(xx)**2
    beta[16] = lambda xx: cf.H_c(xx) * (cf.f_intp(xx)* (tracer.b_1(xx) * (cf.f_intp(xx)+ cf.b_e(xx) - 2 * cf.Q(xx) - ((2 * (1 - cf.Q(xx))) / (cf.comoving_dist(xx) * cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2)) + (db1_dt(xx) / cf.H_c(xx)) + 2 * (1 - (1 / (cf.comoving_dist(xx) * cf.H_c(xx)))) * partdb1 ))
    beta[17] = lambda xx: cf.H_c(xx) * (- (3 / 2) * cf.Om(xx) * cf.f_intp(xx))
    beta[18] = lambda xx: cf.H_c(xx) * ( (3 / 2) * cf.Om(xx) * cf.f_intp(xx) - cf.f_intp(xx)**2 * (3 - 2 * cf.b_e(xx) + 4 * cf.Q(xx) + ((4 * (1 - cf.Q(xx))) / (cf.comoving_dist(xx) * cf.H_c(xx))) + (3 * dH_dt(xx)/ cf.H_c(xx)**2)) )
    beta[19] = lambda xx: cf.H_c(xx) * (cf.f_intp(xx)* (cf.b_e(xx) - 2 * cf.Q(xx) - ((2 * (1 - cf.Q(xx))) / (cf.comoving_dist(xx) * cf.H_c(xx))) - (dH_dt(xx)/ cf.H_c(xx)**2)))

    #get betad - derivatives wrt to ln(d)  - for radial evolution terms
    betad = np.empty(20,dtype=object)
    betad[14:20] = np.array(cf.lnd_derivatives(beta[14:20]),dtype=object)#14-19

    tracer.beta = beta 
    tracer.betad = betad
    return np.concatenate((np.array([tracer.gr1,tracer.gr2,tracer.grd1]),beta[6:],betad[14:20]))