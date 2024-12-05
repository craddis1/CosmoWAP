import numpy as np

class Loc:    
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Loc')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = D1**2*Pk*fNL*(-2*f*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + sigma**2*xb1) + xb01*(b1*sigma**2 + f))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*Mk1*sigma**3)
        
        return expr
    
    def l2(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Loc')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Loc')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = -5*D1**2*Pk*fNL*(2*sigma*(b01*f*(2*sigma**2 + 9) + 3*b01*sigma**2*xb1 + 3*b1*sigma**2*xb01 + f*xb01*(2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(sigma**2 - 9) + b01*sigma**2*xb1*(sigma**2 - 3) + b1*sigma**2*xb01*(sigma**2 - 3) + f*xb01*(sigma**2 - 9))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(4*Mk1*sigma**5)
        
        return expr
    
class Eq:    
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Eq')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Eq')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*k1**2*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = D1**2*Pk*fNL*k1**2*(-2*f*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + sigma**2*xb1) + xb01*(b1*sigma**2 + f))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*Mk1*sigma**3)
        
        return expr
    
    def l2(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Eq')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Eq')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*k1**2*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = -5*D1**2*Pk*fNL*k1**2*(2*sigma*(b01*f*(2*sigma**2 + 9) + 3*b01*sigma**2*xb1 + 3*b1*sigma**2*xb01 + f*xb01*(2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(sigma**2 - 9) + b01*sigma**2*xb1*(sigma**2 - 3) + b1*sigma**2*xb01*(sigma**2 - 3) + f*xb01*(sigma**2 - 9))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(4*Mk1*sigma**5)
        
        return expr
    
class Orth:    
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Orth')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Orth')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = D1**2*Pk*fNL*k1*(b01*(f + 3*xb1) + xb01*(3*b1 + f))/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = D1**2*Pk*fNL*k1*(-2*f*sigma*(b01 + xb01) + np.sqrt(2)*np.sqrt(np.pi)*(b01*(f + sigma**2*xb1) + xb01*(b1*sigma**2 + f))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(2*Mk1*sigma**3)
        
        return expr
    
    def l2(cosmo_functions,k1,zz=0,t=0,sigma=None,fNL=1):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = None, shape='Orth')
        xb01,Mk1 = cosmo_functions.get_PNGparams_pk(self,zz,k1,tracer = cosmo_functions.survey1, shape='Orth')
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = 2*D1**2*Pk*f*fNL*k1*(b01 + xb01)/(3*Mk1)
        
        Erf = scipy.special.erf
        if sigma is not None:
            expr = -5*D1**2*Pk*fNL*k1*(2*sigma*(b01*f*(2*sigma**2 + 9) + 3*b01*sigma**2*xb1 + 3*b1*sigma**2*xb01 + f*xb01*(2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b01*f*(sigma**2 - 9) + b01*sigma**2*xb1*(sigma**2 - 3) + b1*sigma**2*xb01*(sigma**2 - 3) + f*xb01*(sigma**2 - 9))*Erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(4*Mk1*sigma**5)
        
        return expr