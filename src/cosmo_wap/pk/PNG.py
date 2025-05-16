import numpy as np
import scipy

class Loc:
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Loc')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        return D1**2*Pk*fNL*(b01*(f*mu**2 + xb1) + xb01*(b1 + f*mu**2))/Mk1
        
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Loc')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = D1**2*Pk*fNL*(-2*f*k1*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + k1**2*sigma**2*xb1) + xb01*(b1*k1**2*sigma**2 + f))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*Mk1*k1**3*sigma**3)
        
        return expr
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Loc')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = -5*D1**2*Pk*fNL*(2*k1*sigma*(b01*f*(2*k1**2*sigma**2 + 9) + 3*b01*k1**2*sigma**2*xb1 + 3*b1*k1**2*sigma**2*xb01 + f*xb01*(2*k1**2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(k1**2*sigma**2 - 9) + b01*k1**2*sigma**2*xb1*(k1**2*sigma**2 - 3) + b1*k1**2*sigma**2*xb01*(k1**2*sigma**2 - 3) + f*xb01*(k1**2*sigma**2 - 9))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*Mk1*k1**5*sigma**5)
        
        return expr
    
class Eq:
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Loc')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        return D1**2*Pk*fNL*k1**2*(b01*(f*mu**2 + xb1) + xb01*(b1 + f*mu**2))/Mk1
    
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Eq')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Eq')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*k1**2*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = D1**2*Pk*fNL*(-2*f*sigma*(b01 + xb01)*np.exp(-k1**2*sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + k1**2*sigma**2*xb1) + xb01*(b1*k1**2*sigma**2 + f))*Erf(np.sqrt(2)*k1*sigma/2)/k1)/(2*Mk1*sigma**3)
        
        return expr
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Eq')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Eq')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*k1**2*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = -5*D1**2*Pk*fNL*(2*k1*sigma*(b01*f*(2*k1**2*sigma**2 + 9) + 3*b01*k1**2*sigma**2*xb1 + 3*b1*k1**2*sigma**2*xb01 + f*xb01*(2*k1**2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(k1**2*sigma**2 - 9) + b01*k1**2*sigma**2*xb1*(k1**2*sigma**2 - 3) + b1*k1**2*sigma**2*xb01*(k1**2*sigma**2 - 3) + f*xb01*(k1**2*sigma**2 - 9))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*Mk1*k1**3*sigma**5)
        
        return expr
    
class Orth:
    def l(mu,cosmo_funcs,k1,zz=0,t=0,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Loc')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        return D1**2*Pk*fNL*k1*(b01*(f*mu**2 + xb1) + xb01*(b1 + f*mu**2))/Mk1
    
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Orth')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Orth')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*k1*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = D1**2*Pk*fNL*(-2*f*k1*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + k1**2*sigma**2*xb1) + xb01*(b1*k1**2*sigma**2 + f))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*Mk1*k1**2*sigma**3)
        
        return expr
    
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = None, shape='Orth')
        xb01,Mk1 = cosmo_funcs.get_PNGparams_pk(zz,k1,tracer = cosmo_funcs.survey1, shape='Orth')
        
        b1 = cosmo_funcs.survey.b_1(zz)
        xb1 = cosmo_funcs.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*k1*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma != None:
            expr = -5*D1**2*Pk*fNL*(2*k1*sigma*(b01*f*(2*k1**2*sigma**2 + 9) + 3*b01*k1**2*sigma**2*xb1 + 3*b1*k1**2*sigma**2*xb01 + f*xb01*(2*k1**2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(k1**2*sigma**2 - 9) + b01*k1**2*sigma**2*xb1*(k1**2*sigma**2 - 3) + b1*k1**2*sigma**2*xb01*(k1**2*sigma**2 - 3) + f*xb01*(k1**2*sigma**2 - 9))*Erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*Mk1*k1**4*sigma**5)
        
        return expr