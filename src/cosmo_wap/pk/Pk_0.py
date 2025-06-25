import numpy as np
from scipy.special import erf  # Error function needed from integral over FoG

#1st order terms
class NPP:
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        #so we could always change to work with a namespace object or locals
        #unpack all necessary terms
        Pk,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz)
        
        expr = D1**2*Pk*(5*b1*(f + 3*xb1) + f*(3*f + 5*xb1))/15
        
        if sigma != None:
            expr = D1**2*Pk*(-2*f*k1*sigma*(f*(k1**2*sigma**2 + 3) + k1**2*sigma**2*(b1 + xb1))*np.exp(-k1**2*sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(b1*k1**4*sigma**4*xb1 + 3*f**2 + f*k1**2*sigma**2*(b1 + xb1))*erf(np.sqrt(2)*k1*sigma/2))/(2*k1**5*sigma**5)

        return expr
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz)
        
        expr = 2*D1**2*Pk*f*(7*b1 + 6*f + 7*xb1)/21
        
        if sigma != None:
            expr = -5*D1**2*Pk*(2*k1*sigma*(3*b1*k1**4*sigma**4*xb1 + f**2*(2*k1**4*sigma**4 + 12*k1**2*sigma**2 + 45) + f*k1**2*sigma**2*(b1 + xb1)*(2*k1**2*sigma**2 + 9)) + np.sqrt(2)*np.sqrt(np.pi)*(b1*k1**4*sigma**4*xb1*(k1**2*sigma**2 - 3) + 3*f**2*(k1**2*sigma**2 - 15) + f*k1**2*sigma**2*(b1 + xb1)*(k1**2*sigma**2 - 9))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*k1**7*sigma**7)
        return expr
    
    @staticmethod
    def l4(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1 = cosmo_funcs.unpack_pk(k1,zz)
        
        expr = 8*D1**2*Pk*f**2/35
        
        if sigma != None:
            expr = -9*D1**2*Pk*(10*b1**2*k1**5*sigma**5*(k1**2*sigma**2 + 21) + 4*b1*f*k1**3*sigma**3*(8*k1**4*sigma**4 + 85*k1**2*sigma**2 + 525) + 2*f**2*k1*sigma*(8*k1**6*sigma**6 + 104*k1**4*sigma**4 + 775*k1**2*sigma**2 + 3675) - 3*np.sqrt(2)*np.sqrt(np.pi)*(b1**2*k1**4*sigma**4*(k1**4*sigma**4 - 10*k1**2*sigma**2 + 35) + 2*b1*f*k1**2*sigma**2*(k1**4*sigma**4 - 30*k1**2*sigma**2 + 175) + f**2*(3*k1**4*sigma**4 - 150*k1**2*sigma**2 + 1225))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(16*k1**9*sigma**9)
        return expr
    
    
#1st order terms
class GR1:
    @staticmethod
    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,gr1,_,xgr1,_ = cosmo_funcs.unpack_pk(k1,zz,GR=True)
        
        expr = 1j*D1**2*Pk*(-5*b1*xgr1 + 3*f*(gr1 - xgr1) + 5*gr1*xb1)/(5*k1)

        if sigma != None:
            expr = 3*1j*D1**2*Pk*(-2*f*k1*sigma*(gr1 - xgr1)*(k1**2*sigma**2 + 3) + 2*k1**3*sigma**3*(b1*xgr1 - gr1*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(3*f*(gr1 - xgr1) + k1**2*sigma**2*(-b1*xgr1 + gr1*xb1))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*k1**6*sigma**5)
            
        return expr
    
    @staticmethod
    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,gr1,_,xgr1,_ = cosmo_funcs.unpack_pk(k1,zz,GR=True)
        
        expr = 2*1j*D1**2*Pk*f*(gr1 - xgr1)/(5*k1)
        
        if sigma != None:
            expr = -7*1j*D1**2*Pk*(2*k1*sigma*(f*(gr1 - xgr1)*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75) + k1**2*sigma**2*(-b1*xgr1 + gr1*xb1)*(2*k1**2*sigma**2 + 15)) + 3*np.sqrt(2)*np.sqrt(np.pi)*(f*(gr1 - xgr1)*(3*k1**2*sigma**2 - 25) + k1**2*sigma**2*(-b1*xgr1 + gr1*xb1)*(k1**2*sigma**2 - 5))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*k1**8*sigma**7)
        
        return expr
    
    
#2nd order terms
class GR2:
    @staticmethod
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,gr1,gr2,xgr1,xgr2 = cosmo_funcs.unpack_pk(k1,zz,GR=True)
        
        expr = D1**2*Pk*(3*b1*xgr2 + f*(gr2 + xgr2) - gr1*xgr1 + 3*gr2*xb1)/(3*k1**2)
        
        if sigma != None:
            expr = D1**2*Pk*(-2*k1*sigma*(f*(gr2 + xgr2) + gr1*xgr1)*np.exp(-k1**2*sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(b1*k1**2*sigma**2*xgr2 + f*(gr2 + xgr2) + gr1*xgr1 + gr2*k1**2*sigma**2*xb1)*erf(np.sqrt(2)*k1*sigma/2))/(2*k1**5*sigma**3)
        
        return expr
    
    @staticmethod
    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,gr1,gr2,xgr1,xgr2 = cosmo_funcs.unpack_pk(k1,zz,GR=True)
        
        expr = 2*D1**2*Pk*(f*(gr2 + xgr2) + gr1*xgr1)/(3*k1**2)
        
        if sigma != None:
            expr = -5*D1**2*Pk*(2*k1*sigma*(3*b1*k1**2*sigma**2*xgr2 + f*(gr2 + xgr2)*(2*k1**2*sigma**2 + 9) + 2*gr1*k1**2*sigma**2*xgr1 + 9*gr1*xgr1 + 3*gr2*k1**2*sigma**2*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(b1*k1**4*sigma**4*xgr2 - 3*b1*k1**2*sigma**2*xgr2 + f*(gr2 + xgr2)*(k1**2*sigma**2 - 9) + gr1*k1**2*sigma**2*xgr1 - 9*gr1*xgr1 + gr2*k1**2*sigma**2*xb1*(k1**2*sigma**2 - 3))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*k1**7*sigma**5)
        
        return expr
   
    
    