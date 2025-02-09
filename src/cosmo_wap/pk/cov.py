import numpy as np

class COV:
    def __init__(self,cosmo_funcs,k1,zz=0,t=0):
        
        #get generic cosmology parameters
        self.params = cosmo_funcs.get_params_pk(k1,zz)
        
        self.nbar = cosmo_funcs.survey.n_g(zz)  # number density
        self.Nk = 1                             # number of modes -set to 1 is accounted for elsewhere 
        
        self.cosmo_funcs = cosmo_funcs
        self.z = zz

    def N00(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(315*b1**4 + 420*b1**3*f + 378*b1**2*f**2 + 180*b1*f**3 + 35*f**4) + 630*b1**2 + 420*b1*f + 126*f**2) + 630)/(315*Nk*nbar**2)

        return expr
    
    def N20(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = 16*D1**2*Pk*f*(D1**2*Pk*nbar*(231*b1**3 + 297*b1**2*f + 165*b1*f**2 + 35*f**3) + 231*b1 + 99*f)/(693*Nk*nbar)

        return expr
    
    def N22(self,params=None):  
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Nk = self.Nk 

        b1 = self.cosmo_funcs.survey.b_1(self.z)

        expr = (10*D1**2*Pk*nbar*(D1**2*Pk*nbar*(9009*b1**4 + 18876*b1**3*f + 23166*b1**2*f**2 + 13260*b1*f**3 + 2905*f**4) + 18018*b1**2 + 18876*b1*f + 7722*f**2) + 90090)/(9009*Nk*nbar**2)

        return expr