import numpy as np
from scipy.special import erf, eval_legendre   # Error function needed from integral over FoG

import cosmo_wap.pk as pk
from cosmo_wap.lib import integrate

class COV_MU:
    @staticmethod
    def get_coef(l1,l2,mu):
        return (2*l1+1)*(2*l2+1)*eval_legendre(l1,mu)*eval_legendre(l2,mu) # So k_f**3/N_k will be included on the forecast end...
    
    @staticmethod
    def mu_arr(mu,term1,term2,*args,**kwargs):
        """get single covariance component for the powerspectrum"""
        
        P1 = getattr(pk,term1).mu(mu,*args,**kwargs) # first power spectra
        if term1==term2:
            P2=P1
        else:
            P2 = getattr(pk,term2).mu(mu,*args,**kwargs)
        
        return P1*P2
    
    @staticmethod
    def cov_l1l2(term1,term2,l1,l2,*args,n_mu=16,fast=False,**kwargs):
        """get single covariance component for the powerspectrum"""

        def mu_integrand(mu,*args,**kwargs):
            return COV_MU.mu_arr(mu,term1,term2,*args,**kwargs)*COV_MU.get_coef(l1,l2,mu)

        return integrate.int_mu(mu_integrand,n_mu,*args,fast=fast,**kwargs)


class COV:
    def __init__(self,cosmo_funcs,k1,zz=0,t=0,sigma=None,nonlin=False):
        
        Pk1tmp,f,D1,b1,_ = cosmo_funcs.unpack_pk(k1,zz)
        
        if nonlin: # use nonlinear power spectrum
            # Halofit P(k) - has different redshift dependence so translate for given redshift
            Pk1tmp = cosmo_funcs.Pk_NL(k1,zz)
        
        self.nbar = cosmo_funcs.n_g(zz)  # number density
        self.Nk = 1                      # number of modes - set to 1 is accounted for elsewhere 
        
        self.params = k1,Pk1tmp,f,D1,b1,sigma,self.nbar,self.Nk # complete set of necessary params

    def N00(self):
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(315*b1**4 + 420*b1**3*f + 378*b1**2*f**2 + 180*b1*f**3 + 35*f**4) + 630*b1**2 + 420*b1*f + 126*f**2) + 630)/(315*Nk*nbar**2)
        
        if sigma != None:
            expr = (-2*D1**4*Pk**2*f*k1*(32*b1**3*k1**6*sigma**6 + 24*b1**2*f*k1**4*sigma**4*(2*k1**2*sigma**2 + 3) + 8*b1*f**2*k1**2*sigma**2*(4*k1**4*sigma**4 + 10*k1**2*sigma**2 + 15) + f**3*(8*k1**6*sigma**6 + 28*k1**4*sigma**4 + 70*k1**2*sigma**2 + 105))*np.exp(-k1**2*sigma**2) + np.sqrt(np.pi)*D1**4*Pk**2*(16*b1**4*k1**8*sigma**8 + 32*b1**3*f*k1**6*sigma**6 + 72*b1**2*f**2*k1**4*sigma**4 + 120*b1*f**3*k1**2*sigma**2 + 105*f**4)*erf(k1*np.sqrt(sigma**2))/np.sqrt(sigma**2) - 64*D1**2*Pk*f*k1**5*sigma**4*(2*b1*k1**2*sigma**2 + f*(k1**2*sigma**2 + 3))*np.exp(-k1**2*sigma**2/2)/nbar + 32*np.sqrt(2)*np.sqrt(np.pi)*D1**2*Pk*k1**4*sigma**4*(b1**2*k1**4*sigma**4 + 2*b1*f*k1**2*sigma**2 + 3*f**2)*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)/(nbar*np.sqrt(sigma**2)) + 32*k1**9*sigma**8/nbar**2)/(16*Nk*k1**9*sigma**8)

        return expr
    
    def N20(self):
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params
        
        expr = 16*D1**2*Pk*f*(D1**2*Pk*nbar*(231*b1**3 + 297*b1**2*f + 165*b1*f**2 + 35*f**3) + 231*b1 + 99*f)/(693*Nk*nbar)
        
        if sigma != None:
            expr = -5*D1**2*Pk*(np.sqrt(np.pi)*D1**2*Pk*nbar*(16*b1**4*k1**8*sigma**8*(2*k1**2*sigma**2 - 3) + 32*b1**3*f*k1**6*sigma**6*(2*k1**2*sigma**2 - 9) + 72*b1**2*f**2*k1**4*sigma**4*(2*k1**2*sigma**2 - 15) + 120*b1*f**3*k1**2*sigma**2*(2*k1**2*sigma**2 - 21) + 105*f**4*(2*k1**2*sigma**2 - 27))*erf(k1*np.sqrt(sigma**2))*np.exp(k1**2*sigma**2) + 64*np.sqrt(2)*np.sqrt(np.pi)*k1**4*sigma**4*(b1**2*k1**4*sigma**4*(k1**2*sigma**2 - 3) + 2*b1*f*k1**2*sigma**2*(k1**2*sigma**2 - 9) + 3*f**2*(k1**2*sigma**2 - 15))*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)*np.exp(k1**2*sigma**2) + 2*k1*(D1**2*Pk*nbar*(48*b1**4*k1**8*sigma**8 + 32*b1**3*f*k1**6*sigma**6*(4*k1**2*sigma**2 + 9) + 24*b1**2*f**2*k1**4*sigma**4*(8*k1**4*sigma**4 + 24*k1**2*sigma**2 + 45) + 8*b1*f**3*k1**2*sigma**2*(16*k1**6*sigma**6 + 64*k1**4*sigma**4 + 180*k1**2*sigma**2 + 315) + f**4*(32*k1**8*sigma**8 + 160*k1**6*sigma**6 + 616*k1**4*sigma**4 + 1680*k1**2*sigma**2 + 2835)) + 64*k1**4*sigma**4*(3*b1**2*k1**4*sigma**4 + 2*b1*f*k1**2*sigma**2*(2*k1**2*sigma**2 + 9) + f**2*(2*k1**4*sigma**4 + 12*k1**2*sigma**2 + 45))*np.exp(k1**2*sigma**2/2))*np.sqrt(sigma**2))*np.exp(-k1**2*sigma**2)/(64*Nk*k1**11*nbar*sigma**10*np.sqrt(sigma**2))
            
        return expr
    
    def N22(self):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (10*D1**2*Pk*nbar*(D1**2*Pk*nbar*(9009*b1**4 + 18876*b1**3*f + 23166*b1**2*f**2 + 13260*b1*f**3 + 2905*f**4) + 18018*b1**2 + 18876*b1*f + 7722*f**2) + 90090)/(9009*Nk*nbar**2)
        
        if sigma != None:
            expr = (-25*D1**4*Pk**2*(48*b1**4*k1**8*sigma**8*(2*k1**2*sigma**2 + 9) + 32*b1**3*f*k1**6*sigma**6*(16*k1**4*sigma**4 + 54*k1**2*sigma**2 + 135) + 24*b1**2*f**2*k1**4*sigma**4*(32*k1**6*sigma**6 + 144*k1**4*sigma**4 + 450*k1**2*sigma**2 + 945) + 8*b1*f**3*k1**2*sigma**2*(64*k1**8*sigma**8 + 352*k1**6*sigma**6 + 1488*k1**4*sigma**4 + 4410*k1**2*sigma**2 + 8505) + f**4*(128*k1**10*sigma**10 + 832*k1**8*sigma**8 + 4384*k1**6*sigma**6 + 17808*k1**4*sigma**4 + 51030*k1**2*sigma**2 + 93555))*np.exp(-k1**2*sigma**2)/(128*k1**12*sigma**12) + 25*np.sqrt(np.pi)*D1**4*Pk**2*(16*b1**4*k1**8*sigma**8*(4*k1**4*sigma**4 - 12*k1**2*sigma**2 + 27) + 32*b1**3*f*k1**6*sigma**6*(4*k1**4*sigma**4 - 36*k1**2*sigma**2 + 135) + 72*b1**2*f**2*k1**4*sigma**4*(4*k1**4*sigma**4 - 60*k1**2*sigma**2 + 315) + 120*b1*f**3*k1**2*sigma**2*(4*k1**4*sigma**4 - 84*k1**2*sigma**2 + 567) + 105*f**4*(4*k1**4*sigma**4 - 108*k1**2*sigma**2 + 891))*erf(k1*np.sqrt(sigma**2))/(256*k1**13*sigma**12*np.sqrt(sigma**2)) - 25*D1**2*Pk*(3*b1**2*k1**4*sigma**4*(k1**2*sigma**2 + 9) + 2*b1*f*k1**2*sigma**2*(4*k1**4*sigma**4 + 27*k1**2*sigma**2 + 135) + f**2*(4*k1**6*sigma**6 + 36*k1**4*sigma**4 + 225*k1**2*sigma**2 + 945))*np.exp(-k1**2*sigma**2/2)/(k1**8*nbar*sigma**8) + 25*np.sqrt(2)*np.sqrt(np.pi)*D1**2*Pk*(b1**2*k1**4*sigma**4*(k1**4*sigma**4 - 6*k1**2*sigma**2 + 27) + 2*b1*f*k1**2*sigma**2*(k1**4*sigma**4 - 18*k1**2*sigma**2 + 135) + 3*f**2*(k1**4*sigma**4 - 30*k1**2*sigma**2 + 315))*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)/(2*k1**9*nbar*sigma**8*np.sqrt(sigma**2)) + 10/nbar**2)/Nk

        return expr
    
    def N40(self):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (10*D1**2*Pk*nbar*(D1**2*Pk*nbar*(9009*b1**4 + 18876*b1**3*f + 23166*b1**2*f**2 + 13260*b1*f**3 + 2905*f**4) + 18018*b1**2 + 18876*b1*f + 7722*f**2) + 90090)/(9009*Nk*nbar**2)
           
        if sigma != None:
            expr = -9*D1**2*Pk*(-3*np.sqrt(np.pi)*D1**2*Pk*nbar*(16*b1**4*k1**8*sigma**8*(4*k1**4*sigma**4 - 20*k1**2*sigma**2 + 35) + 32*b1**3*f*k1**6*sigma**6*(4*k1**4*sigma**4 - 60*k1**2*sigma**2 + 175) + 24*b1**2*f**2*k1**4*sigma**4*(12*k1**4*sigma**4 - 300*k1**2*sigma**2 + 1225) + 120*b1*f**3*k1**2*sigma**2*(4*k1**4*sigma**4 - 140*k1**2*sigma**2 + 735) + 105*f**4*(4*k1**4*sigma**4 - 180*k1**2*sigma**2 + 1155))*erf(k1*np.sqrt(sigma**2))*np.exp(k1**2*sigma**2) - 384*np.sqrt(2)*np.sqrt(np.pi)*k1**4*sigma**4*(b1**2*k1**4*sigma**4*(k1**4*sigma**4 - 10*k1**2*sigma**2 + 35) + 2*b1*f*k1**2*sigma**2*(k1**4*sigma**4 - 30*k1**2*sigma**2 + 175) + f**2*(3*k1**4*sigma**4 - 150*k1**2*sigma**2 + 1225))*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)*np.exp(k1**2*sigma**2) + 2*k1*(D1**2*Pk*nbar*(80*b1**4*k1**8*sigma**8*(2*k1**2*sigma**2 + 21) + 32*b1**3*f*k1**6*sigma**6*(32*k1**4*sigma**4 + 170*k1**2*sigma**2 + 525) + 24*b1**2*f**2*k1**4*sigma**4*(64*k1**6*sigma**6 + 416*k1**4*sigma**4 + 1550*k1**2*sigma**2 + 3675) + 8*b1*f**3*k1**2*sigma**2*(128*k1**8*sigma**8 + 960*k1**6*sigma**6 + 4800*k1**4*sigma**4 + 15750*k1**2*sigma**2 + 33075) + f**4*(256*k1**10*sigma**10 + 2176*k1**8*sigma**8 + 13440*k1**6*sigma**6 + 60480*k1**4*sigma**4 + 185850*k1**2*sigma**2 + 363825)) + 128*k1**4*sigma**4*(5*b1**2*k1**4*sigma**4*(k1**2*sigma**2 + 21) + 2*b1*f*k1**2*sigma**2*(8*k1**4*sigma**4 + 85*k1**2*sigma**2 + 525) + f**2*(8*k1**6*sigma**6 + 104*k1**4*sigma**4 + 775*k1**2*sigma**2 + 3675))*np.exp(k1**2*sigma**2/2))*np.sqrt(sigma**2))*np.exp(-k1**2*sigma**2)/(512*Nk*k1**13*nbar*sigma**12*np.sqrt(sigma**2))

        return expr
    
    def N42(self):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = 32*D1**2*Pk*f*(3*D1**2*Pk*nbar*(143*b1**3 + 221*b1**2*f + 145*b1*f**2 + 35*f**3) + 429*b1 + 221*f)/(1001*Nk*nbar)
        
        if sigma != None:
            expr = -45*D1**2*Pk*(3*np.sqrt(np.pi)*D1**2*Pk*nbar*(16*b1**4*k1**8*sigma**8*(8*k1**6*sigma**6 - 52*k1**4*sigma**4 + 250*k1**2*sigma**2 - 525) + 32*b1**3*f*k1**6*sigma**6*(8*k1**6*sigma**6 - 156*k1**4*sigma**4 + 1250*k1**2*sigma**2 - 3675) + 24*b1**2*f**2*k1**4*sigma**4*(24*k1**6*sigma**6 - 780*k1**4*sigma**4 + 8750*k1**2*sigma**2 - 33075) + 120*b1*f**3*k1**2*sigma**2*(8*k1**6*sigma**6 - 364*k1**4*sigma**4 + 5250*k1**2*sigma**2 - 24255) + 105*f**4*(8*k1**6*sigma**6 - 468*k1**4*sigma**4 + 8250*k1**2*sigma**2 - 45045))*erf(k1*np.sqrt(sigma**2))*np.exp(k1**2*sigma**2) + 768*np.sqrt(2)*np.sqrt(np.pi)*k1**4*sigma**4*(b1**2*k1**4*sigma**4*(k1**6*sigma**6 - 13*k1**4*sigma**4 + 125*k1**2*sigma**2 - 525) + 2*b1*f*k1**2*sigma**2*(k1**6*sigma**6 - 39*k1**4*sigma**4 + 625*k1**2*sigma**2 - 3675) + f**2*(3*k1**6*sigma**6 - 195*k1**4*sigma**4 + 4375*k1**2*sigma**2 - 33075))*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)*np.exp(k1**2*sigma**2) + 2*k1*(D1**2*Pk*nbar*(16*b1**4*k1**8*sigma**8*(76*k1**4*sigma**4 + 300*k1**2*sigma**2 + 1575) + 32*b1**3*f*k1**6*sigma**6*(128*k1**6*sigma**6 + 908*k1**4*sigma**4 + 3600*k1**2*sigma**2 + 11025) + 24*b1**2*f**2*k1**4*sigma**4*(256*k1**8*sigma**8 + 2048*k1**6*sigma**6 + 11300*k1**4*sigma**4 + 39900*k1**2*sigma**2 + 99225) + 8*b1*f**3*k1**2*sigma**2*(512*k1**10*sigma**10 + 4608*k1**8*sigma**8 + 30720*k1**6*sigma**6 + 149940*k1**4*sigma**4 + 491400*k1**2*sigma**2 + 1091475) + f**4*(1024*k1**12*sigma**12 + 10240*k1**10*sigma**10 + 79872*k1**8*sigma**8 + 483840*k1**6*sigma**6 + 2198700*k1**4*sigma**4 + 6860700*k1**2*sigma**2 + 14189175)) + 256*k1**4*sigma**4*(b1**2*k1**4*sigma**4*(19*k1**4*sigma**4 + 150*k1**2*sigma**2 + 1575) + 2*b1*f*k1**2*sigma**2*(16*k1**6*sigma**6 + 227*k1**4*sigma**4 + 1800*k1**2*sigma**2 + 11025) + f**2*(16*k1**8*sigma**8 + 256*k1**6*sigma**6 + 2825*k1**4*sigma**4 + 19950*k1**2*sigma**2 + 99225))*np.exp(k1**2*sigma**2/2))*np.sqrt(sigma**2))*np.exp(-k1**2*sigma**2)/(2048*Nk*k1**15*nbar*sigma**14*np.sqrt(sigma**2))
            
        return expr
    
    def N44(self):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (18*D1**2*Pk*nbar*(D1**2*Pk*nbar*(85085*b1**4 + 172380*b1**3*f + 196758*b1**2*f**2 + 111180*b1*f**3 + 24885*f**4) + 170170*b1**2 + 172380*b1*f + 65586*f**2) + 1531530)/(85085*Nk*nbar**2)
        
        if sigma != None:
            expr = (243*np.sqrt(np.pi)*D1**4*Pk**2*nbar**2*(16*b1**4*k1**8*sigma**8*(48*k1**8*sigma**8 - 480*k1**6*sigma**6 + 4440*k1**4*sigma**4 - 21000*k1**2*sigma**2 + 42875) + 96*b1**3*f*k1**6*sigma**6*(16*k1**8*sigma**8 - 480*k1**6*sigma**6 + 7400*k1**4*sigma**4 - 49000*k1**2*sigma**2 + 128625) + 72*b1**2*f**2*k1**4*sigma**4*(48*k1**8*sigma**8 - 2400*k1**6*sigma**6 + 51800*k1**4*sigma**4 - 441000*k1**2*sigma**2 + 1414875) + 360*b1*f**3*k1**2*sigma**2*(16*k1**8*sigma**8 - 1120*k1**6*sigma**6 + 31080*k1**4*sigma**4 - 323400*k1**2*sigma**2 + 1226225) + 315*f**4*(16*k1**8*sigma**8 - 1440*k1**6*sigma**6 + 48840*k1**4*sigma**4 - 600600*k1**2*sigma**2 + 2627625))*erf(k1*np.sqrt(sigma**2))*np.exp(k1**2*sigma**2) + 124416*np.sqrt(2)*np.sqrt(np.pi)*D1**2*Pk*k1**4*nbar*sigma**4*(b1**2*k1**4*sigma**4*(3*k1**8*sigma**8 - 60*k1**6*sigma**6 + 1110*k1**4*sigma**4 - 10500*k1**2*sigma**2 + 42875) + 6*b1*f*k1**2*sigma**2*(k1**8*sigma**8 - 60*k1**6*sigma**6 + 1850*k1**4*sigma**4 - 24500*k1**2*sigma**2 + 128625) + f**2*(9*k1**8*sigma**8 - 900*k1**6*sigma**6 + 38850*k1**4*sigma**4 - 661500*k1**2*sigma**2 + 4244625))*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)*np.exp(k1**2*sigma**2) + 18*k1*(-9*D1**4*Pk**2*nbar**2*(80*b1**4*k1**8*sigma**8*(88*k1**6*sigma**6 + 1124*k1**4*sigma**4 + 4550*k1**2*sigma**2 + 25725) + 32*b1**3*f*k1**6*sigma**6*(1024*k1**8*sigma**8 + 10680*k1**6*sigma**6 + 81300*k1**4*sigma**4 + 330750*k1**2*sigma**2 + 1157625) + 24*b1**2*f**2*k1**4*sigma**4*(2048*k1**10*sigma**10 + 23552*k1**8*sigma**8 + 201000*k1**6*sigma**6 + 1215900*k1**4*sigma**4 + 4520250*k1**2*sigma**2 + 12733875) + 8*b1*f**3*k1**2*sigma**2*(4096*k1**12*sigma**12 + 51200*k1**10*sigma**10 + 496640*k1**8*sigma**8 + 3616200*k1**6*sigma**6 + 19233900*k1**4*sigma**4 + 66701250*k1**2*sigma**2 + 165540375) + f**4*(8192*k1**14*sigma**14 + 110592*k1**12*sigma**12 + 1198080*k1**10*sigma**10 + 10214400*k1**8*sigma**8 + 67246200*k1**6*sigma**6 + 329937300*k1**4*sigma**4 + 1087836750*k1**2*sigma**2 + 2483105625)) - 4608*D1**2*Pk*k1**4*nbar*sigma**4*(5*b1**2*k1**4*sigma**4*(11*k1**6*sigma**6 + 281*k1**4*sigma**4 + 2275*k1**2*sigma**2 + 25725) + 2*b1*f*k1**2*sigma**2*(64*k1**8*sigma**8 + 1335*k1**6*sigma**6 + 20325*k1**4*sigma**4 + 165375*k1**2*sigma**2 + 1157625) + f**2*(64*k1**10*sigma**10 + 1472*k1**8*sigma**8 + 25125*k1**6*sigma**6 + 303975*k1**4*sigma**4 + 2260125*k1**2*sigma**2 + 12733875))*np.exp(k1**2*sigma**2/2) + 16384*k1**16*sigma**16*np.exp(k1**2*sigma**2))*np.sqrt(sigma**2))*np.exp(-k1**2*sigma**2)/(16384*Nk*k1**17*nbar**2*sigma**16*np.sqrt(sigma**2))

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
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (2*D1**2*Pk*nbar*(D1**2*Pk*nbar*(1155*b1**4 + 2772*b1**3*f + 2970*b1**2*f**2 + 1540*b1*f**3 + 315*f**4) + 2310*b1**2 + 2772*b1*f + 990*f**2) + 2310)/(385*Nk*nbar**2)
        
        if sigma != None:
            expr = (-18*D1**4*Pk**2*k1*nbar**2*sigma**2*(16*b1**4*k1**8*sigma**8 + 32*b1**3*f*k1**6*sigma**6*(2*k1**2*sigma**2 + 3) + 24*b1**2*f**2*k1**4*sigma**4*(4*k1**4*sigma**4 + 10*k1**2*sigma**2 + 15) + 8*b1*f**3*k1**2*sigma**2*(8*k1**6*sigma**6 + 28*k1**4*sigma**4 + 70*k1**2*sigma**2 + 105) + f**4*(16*k1**8*sigma**8 + 72*k1**6*sigma**6 + 252*k1**4*sigma**4 + 630*k1**2*sigma**2 + 945))*np.exp(-k1**2*sigma**2) + 9*np.sqrt(np.pi)*D1**4*Pk**2*nbar**2*(16*b1**4*k1**8*sigma**8 + 96*b1**3*f*k1**6*sigma**6 + 360*b1**2*f**2*k1**4*sigma**4 + 840*b1*f**3*k1**2*sigma**2 + 945*f**4)*np.sqrt(sigma**2)*erf(k1*np.sqrt(sigma**2)) - 1152*D1**2*Pk*k1**5*nbar*sigma**6*(b1**2*k1**4*sigma**4 + 2*b1*f*k1**2*sigma**2*(k1**2*sigma**2 + 3) + f**2*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))*np.exp(-k1**2*sigma**2/2) + 576*np.sqrt(2)*np.sqrt(np.pi)*D1**2*Pk*k1**4*nbar*sigma**4*(b1**2*k1**4*sigma**4 + 6*b1*f*k1**2*sigma**2 + 15*f**2)*np.sqrt(sigma**2)*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2) + 192*k1**11*sigma**12)/(32*Nk*k1**11*nbar**2*sigma**12)

        return expr
    
    def N31(self,params=None):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = 16*D1**2*Pk*f*(3*D1**2*Pk*nbar*(429*b1**3 + 715*b1**2*f + 455*b1*f**2 + 105*f**3) + 1287*b1 + 715*f)/(2145*Nk*nbar)
        
        if sigma != None:
            expr = -21*D1**2*Pk*(2*D1**2*Pk*k1*nbar*sigma**2*(16*b1**4*k1**8*sigma**8*(4*k1**2*sigma**2 + 15) + 32*b1**3*f*k1**6*sigma**6*(8*k1**4*sigma**4 + 32*k1**2*sigma**2 + 75) + 24*b1**2*f**2*k1**4*sigma**4*(16*k1**6*sigma**6 + 80*k1**4*sigma**4 + 260*k1**2*sigma**2 + 525) + 8*b1*f**3*k1**2*sigma**2*(32*k1**8*sigma**8 + 192*k1**6*sigma**6 + 840*k1**4*sigma**4 + 2520*k1**2*sigma**2 + 4725) + f**4*(64*k1**10*sigma**10 + 448*k1**8*sigma**8 + 2448*k1**6*sigma**6 + 10080*k1**4*sigma**4 + 28980*k1**2*sigma**2 + 51975)) + 3*np.sqrt(np.pi)*D1**2*Pk*nbar*(16*b1**4*k1**8*sigma**8*(2*k1**2*sigma**2 - 5) + 32*b1**3*f*k1**6*sigma**6*(6*k1**2*sigma**2 - 25) + 120*b1**2*f**2*k1**4*sigma**4*(6*k1**2*sigma**2 - 35) + 840*b1*f**3*k1**2*sigma**2*(2*k1**2*sigma**2 - 15) + 315*f**4*(6*k1**2*sigma**2 - 55))*np.sqrt(sigma**2)*erf(k1*np.sqrt(sigma**2))*np.exp(k1**2*sigma**2) + 256*k1**5*sigma**6*(b1**2*k1**4*sigma**4*(2*k1**2*sigma**2 + 15) + 2*b1*f*k1**2*sigma**2*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75) + f**2*(2*k1**6*sigma**6 + 20*k1**4*sigma**4 + 130*k1**2*sigma**2 + 525))*np.exp(k1**2*sigma**2/2) + 384*np.sqrt(2)*np.sqrt(np.pi)*k1**4*(b1**2*k1**4*sigma**4*(k1**2*sigma**2 - 5) + 2*b1*f*k1**2*sigma**2*(3*k1**2*sigma**2 - 25) + 5*f**2*(3*k1**2*sigma**2 - 35))*(sigma**2)**(5/2)*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)*np.exp(k1**2*sigma**2))*np.exp(-k1**2*sigma**2)/(128*Nk*k1**13*nbar*sigma**14)

        return expr
    
    def N33(self,params=None):  
        k1, Pk,f,D1,b1,sigma,nbar,Nk = self.params

        expr = (14*D1**2*Pk*nbar*(D1**2*Pk*nbar*(6435*b1**4 + 13156*b1**3*f + 15210*b1**2*f**2 + 8820*b1*f**3 + 1995*f**4) + 12870*b1**2 + 13156*b1*f + 5070*f**2) + 90090)/(6435*Nk*nbar**2)
        
        if sigma != None:
            expr = (-49*D1**4*Pk**2*(16*b1**4*k1**8*sigma**8*(16*k1**4*sigma**4 + 70*k1**2*sigma**2 + 375) + 32*b1**3*f*k1**6*sigma**6*(32*k1**6*sigma**6 + 208*k1**4*sigma**4 + 850*k1**2*sigma**2 + 2625) + 24*b1**2*f**2*k1**4*sigma**4*(64*k1**8*sigma**8 + 480*k1**6*sigma**6 + 2640*k1**4*sigma**4 + 9450*k1**2*sigma**2 + 23625) + 8*b1*f**3*k1**2*sigma**2*(128*k1**10*sigma**10 + 1088*k1**8*sigma**8 + 7200*k1**6*sigma**6 + 35280*k1**4*sigma**4 + 116550*k1**2*sigma**2 + 259875) + f**4*(256*k1**12*sigma**12 + 2432*k1**10*sigma**10 + 18752*k1**8*sigma**8 + 113760*k1**6*sigma**6 + 519120*k1**4*sigma**4 + 1628550*k1**2*sigma**2 + 3378375))*np.exp(-k1**2*sigma**2)/(256*k1**14*sigma**14) + 147*np.sqrt(np.pi)*D1**4*Pk**2*(16*b1**4*k1**8*sigma**8*(12*k1**4*sigma**4 - 60*k1**2*sigma**2 + 125) + 32*b1**3*f*k1**6*sigma**6*(36*k1**4*sigma**4 - 300*k1**2*sigma**2 + 875) + 360*b1**2*f**2*k1**4*sigma**4*(12*k1**4*sigma**4 - 140*k1**2*sigma**2 + 525) + 2520*b1*f**3*k1**2*sigma**2*(4*k1**4*sigma**4 - 60*k1**2*sigma**2 + 275) + 315*f**4*(36*k1**4*sigma**4 - 660*k1**2*sigma**2 + 3575))*np.sqrt(sigma**2)*erf(k1*np.sqrt(sigma**2))/(512*k1**15*sigma**16) - 49*D1**2*Pk*(b1**2*k1**4*sigma**4*(4*k1**4*sigma**4 + 35*k1**2*sigma**2 + 375) + 2*b1*f*k1**2*sigma**2*(4*k1**6*sigma**6 + 52*k1**4*sigma**4 + 425*k1**2*sigma**2 + 2625) + f**2*(4*k1**8*sigma**8 + 60*k1**6*sigma**6 + 660*k1**4*sigma**4 + 4725*k1**2*sigma**2 + 23625))*np.exp(-k1**2*sigma**2/2)/(k1**10*nbar*sigma**10) + 147*np.sqrt(2)*np.sqrt(np.pi)*D1**2*Pk*(b1**2*k1**4*sigma**4*(3*k1**4*sigma**4 - 30*k1**2*sigma**2 + 125) + 2*b1*f*k1**2*sigma**2*(9*k1**4*sigma**4 - 150*k1**2*sigma**2 + 875) + 15*f**2*(3*k1**4*sigma**4 - 70*k1**2*sigma**2 + 525))*(sigma**2)**(5/2)*erf(np.sqrt(2)*k1*np.sqrt(sigma**2)/2)/(2*k1**11*nbar*sigma**16) + 14/nbar**2)/Nk

        return expr