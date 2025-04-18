import numpy as np
import scipy


#2nd order terms
class WAGR:
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey1)
        
        expr = -2*D1**2*(Pk + Pkd*k1)*(-5*b1*xgr1*(t - 1) + f*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) + 5*gr1*t*xb1)/(15*d*k1**2)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = D1**2*(2*f*k1*sigma*(2*Pk*(k1**2*sigma**2 + 6) - 3*Pkd*k1)*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) + 2*k1**3*sigma**3*(2*Pk - Pkd*k1)*(-b1*xgr1*(t - 1) + gr1*t*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(f*(2*Pk*(k1**2*sigma**2 - 6) + Pkd*k1*(-k1**2*sigma**2 + 3))*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) - k1**2*sigma**2*(2*Pk + Pkd*k1*(k1**2*sigma**2 - 1))*(-b1*xgr1*(t - 1) + gr1*t*xb1))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*d*k1**7*sigma**5)
        
        return expr
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey1)
        
        expr = -2*D1**2*(f*(10*Pk + Pkd*k1)*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) + 7*(2*Pk - Pkd*k1)*(-b1*xgr1*(t - 1) + gr1*t*xb1))/(21*d*k1**2)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = 5*D1**2*(2*k1*sigma*(f*(2*Pk*(2*k1**4*sigma**4 + 15*k1**2*sigma**2 + 90) - 3*Pkd*k1*(k1**2*sigma**2 + 15))*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) + k1**2*sigma**2*(2*Pk*(2*k1**2*sigma**2 + 9) + Pkd*k1*(k1**2*sigma**2 - 9))*(-b1*xgr1*(t - 1) + gr1*t*xb1)) + np.sqrt(2)*np.sqrt(np.pi)*(-f*(2*Pk*(k1**4*sigma**4 - 15*k1**2*sigma**2 + 90) - Pkd*k1*(k1**4*sigma**4 - 12*k1**2*sigma**2 + 45))*(gr1*(3*t - 2) - 3*t*xgr1 + xgr1) + k1**2*sigma**2*(2*Pk*(k1**2*sigma**2 - 9) + Pkd*k1*(k1**4*sigma**4 - 4*k1**2*sigma**2 + 9))*(-b1*xgr1*(t - 1) + gr1*t*xb1))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*d*k1**9*sigma**7)
        
        return expr
    
    
class RRGR:
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey1)
        
        fd,Dd,_,_,bd1,fdd,Ddd,_,_,bdd1 = cosmo_funcs.get_derivs(zz,tracer = cosmo_funcs.survey)
        fd,Dd,_,_,xbd1,fdd,Ddd,_,_,xbdd1 = cosmo_funcs.get_derivs(zz,tracer = cosmo_funcs.survey1)
        
        expr = -D1*(Pk + Pkd*k1)*(D1*(-5*b1*t*xgrd1 + 5*b1*xgrd1 - 5*bd1*t*xgr1 + 3*f*(grd1*t - t*xgrd1 + xgrd1) + 3*fd*gr1*(t - 1) - 3*fd*t*xgr1 + 5*gr1*xbd1*(t - 1) + 5*grd1*t*xb1) + Dd*(2*t - 1)*(-5*b1*xgr1 + 3*f*(gr1 - xgr1) + 5*gr1*xb1))/(15*d*k1**2)

        Erf = scipy.special.erf
        if sigma != None:
            expr = D1*(-2*k1*sigma*(D1*(f*(Pk*(k1**2*sigma**2 + 12) - Pkd*k1*(k1**2*sigma**2 + 3))*(grd1*t - t*xgrd1 + xgrd1) + fd*(Pk*(k1**2*sigma**2 + 12) - Pkd*k1*(k1**2*sigma**2 + 3))*(gr1*(t - 1) - t*xgr1) + k1**2*sigma**2*(2*Pk - Pkd*k1)*(-b1*t*xgrd1 + b1*xgrd1 - bd1*t*xgr1 + gr1*xbd1*(t - 1) + grd1*t*xb1)) + Dd*(2*t - 1)*(f*(gr1 - xgr1)*(Pk*(k1**2*sigma**2 + 12) - Pkd*k1*(k1**2*sigma**2 + 3)) + k1**2*sigma**2*(2*Pk - Pkd*k1)*(-b1*xgr1 + gr1*xb1)))*np.exp(-k1**2*sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(D1*(-3*f*(Pk*(k1**2*sigma**2 - 4) + Pkd*k1)*(grd1*t - t*xgrd1 + xgrd1) - 3*fd*(Pk*(k1**2*sigma**2 - 4) + Pkd*k1)*(gr1*(t - 1) - t*xgr1) - k1**2*sigma**2*(Pk*(k1**2*sigma**2 - 2) + Pkd*k1)*(-b1*t*xgrd1 + b1*xgrd1 - bd1*t*xgr1 + gr1*xbd1*(t - 1) + grd1*t*xb1)) - Dd*(2*t - 1)*(3*f*(gr1 - xgr1)*(Pk*(k1**2*sigma**2 - 4) + Pkd*k1) + k1**2*sigma**2*(Pk*(k1**2*sigma**2 - 2) + Pkd*k1)*(-b1*xgr1 + gr1*xb1)))*Erf(np.sqrt(2)*k1*sigma/2))/(2*d*k1**7*sigma**5)
        
        return expr
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        gr1,gr2,grd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey)
        xgr1,xgr2,xgrd1,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = cosmo_funcs.get_beta_funcs(zz,tracer = cosmo_funcs.survey1)
        
        fd,Dd,_,_,bd1,fdd,Ddd,_,_,bdd1 = cosmo_funcs.get_derivs(zz,tracer = cosmo_funcs.survey)
        fd,Dd,_,_,xbd1,fdd,Ddd,_,_,xbdd1 = cosmo_funcs.get_derivs(zz,tracer = cosmo_funcs.survey1)
        
        expr = 2*D1*(D1*(3*f*(Pk - 2*Pkd*k1)*(grd1*t - t*xgrd1 + xgrd1) + 3*fd*(Pk - 2*Pkd*k1)*(gr1*(t - 1) - t*xgr1) + 7*(2*Pk - Pkd*k1)*(-b1*t*xgrd1 + b1*xgrd1 - bd1*t*xgr1 + gr1*xbd1*(t - 1) + grd1*t*xb1)) + Dd*(2*t - 1)*(3*f*(Pk - 2*Pkd*k1)*(gr1 - xgr1) + 7*(2*Pk - Pkd*k1)*(-b1*xgr1 + gr1*xb1)))/(21*d*k1**2)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = 5*D1*(-2*k1*sigma*(D1*(f*(Pk*(2*k1**4*sigma**4 + 21*k1**2*sigma**2 + 180) - Pkd*k1*(2*k1**4*sigma**4 + 12*k1**2*sigma**2 + 45))*(grd1*t - t*xgrd1 + xgrd1) + fd*(Pk*(2*k1**4*sigma**4 + 21*k1**2*sigma**2 + 180) - Pkd*k1*(2*k1**4*sigma**4 + 12*k1**2*sigma**2 + 45))*(gr1*(t - 1) - t*xgr1) + k1**2*sigma**2*(Pk*(k1**2*sigma**2 + 18) - Pkd*k1*(2*k1**2*sigma**2 + 9))*(-b1*t*xgrd1 + b1*xgrd1 - bd1*t*xgr1 + gr1*xbd1*(t - 1) + grd1*t*xb1)) + Dd*(2*t - 1)*(f*(gr1 - xgr1)*(Pk*(2*k1**4*sigma**4 + 21*k1**2*sigma**2 + 180) - Pkd*k1*(2*k1**4*sigma**4 + 12*k1**2*sigma**2 + 45)) + k1**2*sigma**2*(Pk*(k1**2*sigma**2 + 18) - Pkd*k1*(2*k1**2*sigma**2 + 9))*(-b1*xgr1 + gr1*xb1))) + np.sqrt(2)*np.sqrt(np.pi)*(D1*(3*f*(Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + Pkd*k1*(k1**2*sigma**2 - 15))*(grd1*t - t*xgrd1 + xgrd1) + 3*fd*(Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + Pkd*k1*(k1**2*sigma**2 - 15))*(gr1*(t - 1) - t*xgr1) + k1**2*sigma**2*(Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 18) + Pkd*k1*(k1**2*sigma**2 - 9))*(-b1*t*xgrd1 + b1*xgrd1 - bd1*t*xgr1 + gr1*xbd1*(t - 1) + grd1*t*xb1)) + Dd*(2*t - 1)*(3*f*(gr1 - xgr1)*(Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + Pkd*k1*(k1**2*sigma**2 - 15)) + k1**2*sigma**2*(Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 18) + Pkd*k1*(k1**2*sigma**2 - 9))*(-b1*xgr1 + gr1*xb1)))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*d*k1**9*sigma**7)
        
        return expr
 