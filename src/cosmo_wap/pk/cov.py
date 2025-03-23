import numpy as np

class COV:
    def __init__(self,cosmo_funcs,k1,zz=0,t=0,nonlin=False):
        
        #get generic cosmology parameters
        k1,Pk1tmp,Pkd1,Pkdd1,d,f,D1 = cosmo_funcs.get_params_pk(k1,zz)
        
        if nonlin: # use nonlinear power spectrum
            Pk1tmp = cosmo_funcs.Pk_NL(k1)
            
        self.params = k1,Pk1tmp,Pkd1,Pkdd1,d,f,D1
        
        self.nbar = cosmo_funcs.survey.n_g(zz)  # number density
        self.Nk = 1                             # number of modes -set to 1 is accounted for elsewhere 
        
        self.cosmo_funcs = cosmo_funcs
        self.z = zz

    def N00(self):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(315*b1**4 + 420*b1**3*f + 378*b1**2*f**2 + 180*b1*f**3 + 35*f**4) + 630*b1**2 + 420*b1*f + 126*f**2) + 630)/(315*Nk*nbar**2)

        return expr
    
    def N20(self):
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = 16*D1**2*Pk*f*(D1**2*Pk*nbar*(231*b1**3 + 297*b1**2*f + 165*b1*f**2 + 35*f**3) + 231*b1 + 99*f)/(693*Nk*nbar)

        return expr
    
    def N22(self):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (10*D1**2*Pk*nbar*(D1**2*Pk*nbar*(9009*b1**4 + 18876*b1**3*f + 23166*b1**2*f**2 + 13260*b1*f**3 + 2905*f**4) + 18018*b1**2 + 18876*b1*f + 7722*f**2) + 90090)/(9009*Nk*nbar**2)

        return expr
    
    def N40(self):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (10*D1**2*Pk*nbar*(D1**2*Pk*nbar*(9009*b1**4 + 18876*b1**3*f + 23166*b1**2*f**2 + 13260*b1*f**3 + 2905*f**4) + 18018*b1**2 + 18876*b1*f + 7722*f**2) + 90090)/(9009*Nk*nbar**2)

        return expr
    
    def N42(self):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = 32*D1**2*Pk*f*(3*D1**2*Pk*nbar*(143*b1**3 + 221*b1**2*f + 145*b1*f**2 + 35*f**3) + 429*b1 + 221*f)/(1001*Nk*nbar)

        return expr
    
    def N44(self):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (18*D1**2*Pk*nbar*(D1**2*Pk*nbar*(85085*b1**4 + 172380*b1**3*f + 196758*b1**2*f**2 + 111180*b1*f**3 + 24885*f**4) + 170170*b1**2 + 172380*b1*f + 65586*f**2) + 1531530)/(85085*Nk*nbar**2)

        return expr
    
    #also duplicate things for new naming scheme that allows for cross-multipole covariances...
    def N02(self): 
        return self.N20()
    
    def N04(self):
        return self.N40()
    
    def N24(self):
        return self.N42()
    
    # these aren't really used or tested yet...
    
    def N11(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(1155*b1**4 + 2772*b1**3*f + 2970*b1**2*f**2 + 1540*b1*f**3 + 315*f**4) + 2310*b1**2 + 2772*b1*f + 990*f**2) + 2310)/(385*Nk*nbar**2)

        return expr
    
    def N31(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = 16*D1**2*Pk*f*(3*D1**2*Pk*nbar*(429*b1**3 + 715*b1**2*f + 455*b1*f**2 + 105*f**3) + 1287*b1 + 715*f)/(2145*Nk*nbar)

        return expr
    
    def N33(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (14*D1**2*Pk*nbar*(D1**2*Pk*nbar*(6435*b1**4 + 13156*b1**3*f + 15210*b1**2*f**2 + 8820*b1*f**3 + 1995*f**4) + 12870*b1**2 + 13156*b1*f + 5070*f**2) + 90090)/(6435*Nk*nbar**2)

        return expr