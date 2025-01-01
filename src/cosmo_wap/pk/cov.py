import numpy as np

class COV:
    def __init__(self,cosmo_funcs,k1,zz=0,t=0):
        
        #get generic cosmology parameters
        self.params = cosmo_funcs.get_params(k1,zz)
        
        self.nbar = cosmo_funcs.survey.n_g(zz)
        self.Nk = 1
        
        self.cosmo_funcs = cosmo_funcs

    def N00(self,params=None)    
        k1,Pk,Pkd,Pkdd,d,f,D1 = self.params

        nbar = self.nbar
        Ntri = self.Ntri

        b1 = self.cosmo_funcs.survey.b_1(zz)

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(315*b1**4 + 420*b1**3*f + 378*b1**2*f**2 + 180*b1*f**3 + 35*f**4) + 630*b1**2 + 420*b1*f + 126*f**2) + 630)/(315*Nk*nbar**2)

        return expr