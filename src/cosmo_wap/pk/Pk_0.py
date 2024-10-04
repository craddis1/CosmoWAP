import numpy as np
import scipy

#1st order terms
class Pk0:    
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = D1**2*Pk*(5*b1*(f + 3*xb1) + f*(3*f + 5*xb1))/15
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = D1**2*Pk*(-2*f*sigma*(f*(sigma**2 + 3) + sigma**2*(b1 + xb1))*np.exp(-sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(b1*sigma**4*xb1 + 3*f**2 + f*sigma**2*(b1 + xb1))*Erf(np.sqrt(2)*sigma/2))/(2*sigma**5)
        
        return expr
    
    def l2(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*(7*b1 + 6*f + 7*xb1)/21
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = -5*D1**2*Pk*(2*sigma*(3*b1*sigma**4*xb1 + f**2*(2*sigma**4 + 12*sigma**2 + 45) + f*sigma**2*(b1 + xb1)*(2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b1*sigma**4*xb1*(sigma**2 - 3) + 3*f**2*(sigma**2 - 15) + f*sigma**2*(b1 + xb1)*(sigma**2 - 9))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(4*sigma**7)
        
        return expr
    
    
#1st order terms
class GR1:        
    def l1(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey1)
        
        expr = 1j*D1**2*Pk*(-5*b1*xgr1 + 3*f*(gr1 - xgr1) + 5*gr1*xb1)/(5*k1)
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = 3*1j*D1**2*Pk*(-2*f*sigma*(gr1 - xgr1)*(sigma**2 + 3) + 2*sigma**3*(b1*xgr1 - gr1*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(3*f*(gr1 - xgr1) + sigma**2*(-b1*xgr1 + gr1*xb1))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*k1*sigma**5)
            
        return expr
    
    def l3(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey1)
        
        expr = 2*1j*D1**2*Pk*f*(gr1 - xgr1)/(5*k1)
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = 3*1j*D1**2*Pk*(-2*f*sigma*(gr1 - xgr1)*(sigma**2 + 3) + 2*sigma**3*(b1*xgr1 - gr1*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(3*f*(gr1 - xgr1) + sigma**2*(-b1*xgr1 + gr1*xb1))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*k1*sigma**5)
        
        return expr
    
    
#2nd order terms
class GR2:        
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey1)
        
        expr = D1**2*Pk*(3*b1*xgr2 + f*(gr2 + xgr2) - gr1*xgr1 + 3*gr2*xb1)/(3*k1**2)
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = D1**2*Pk*(-2*sigma*(f*(gr2 + xgr2) + gr1*xgr1) + np.sqrt(2)*np.sqrt(np.pi)*(b1*sigma**2*xgr2 + f*(gr2 + xgr2) + gr1*xgr1 + gr2*sigma**2*xb1)*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*k1**2*sigma**3)
        
        return expr
    
    def l2(cosmo_functions,k1,zz=0,t=0,sigma=0,nonlin=False):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz,nonlinear=nonlin)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_functions.get_beta_funcs(zz,tracer = cosmo_functions.survey1)
        
        expr = 2*D1**2*Pk*(f*(gr2 + xgr2) + gr1*xgr1)/(3*k1**2)
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = -5*D1**2*Pk*(2*sigma*(3*b1*sigma**2*xgr2 + f*(gr2 + xgr2)*(2*sigma**2 + 9) + 2*gr1*sigma**2*xgr1 + 9*gr1*xgr1 + 3*gr2*sigma**2*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(b1*sigma**4*xgr2 - 3*b1*sigma**2*xgr2 + f*(gr2 + xgr2)*(sigma**2 - 9) + gr1*sigma**2*xgr1 - 9*gr1*xgr1 + gr2*sigma**2*xb1*(sigma**2 - 3))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(4*k1**2*sigma**5)
        
        return expr
   
    
    