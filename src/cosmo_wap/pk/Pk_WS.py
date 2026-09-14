import numpy as np
from scipy.special import erf  # Error function needed from integral over FoG

from cosmo_wap.lib.angular_integrate import legendre
from cosmo_wap.lib.utils import add_empty_methods_pk

# Need to fix: RR2.l0 is single tracer only currently

#1st order terms
@add_empty_methods_pk('l0','l2','l4')
class WA1:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        Pk,f,D1,b1,xb1,Pkd,_,d = cosmo_funcs.unpack_pk(k1,zz,WS=True) #unpack all necessary terms
        return 2*1j*D1**2*f*mu*(b1*(t - 1)*(Pkd*k1 + mu**2*(2*Pk - Pkd*k1)) + f*mu**2*(2*t - 1)*(Pk*(4*mu**2 - 2) - Pkd*k1*(mu**2 - 1)) + t*xb1*(Pkd*k1 + mu**2*(2*Pk - Pkd*k1)))/(d*k1)

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(WA1.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

    ################################### Regular Multipoles #############################################################

    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,_,d = cosmo_funcs.unpack_pk(k1,zz,WS=True)

        if sigma is not None:
            expr = 3*1j*D1**2*f*(-2*f*k1*sigma*(2*t - 1)*(2*Pk*(k1**4*sigma**4 + 7*k1**2*sigma**2 + 30) - Pkd*k1*(2*k1**2*sigma**2 + 15)) - 2*k1**3*sigma**3*(2*Pk*(k1**2*sigma**2 + 3) - 3*Pkd*k1)*(b1*(t - 1) + t*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(3*f*(2*t - 1)*(Pk*(-2*k1**2*sigma**2 + 20) + Pkd*k1*(k1**2*sigma**2 - 5)) + k1**2*sigma**2*(6*Pk + Pkd*k1*(k1**2*sigma**2 - 3))*(b1*(t - 1) + t*xb1))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(d*k1**8*sigma**7)
        else:
            expr = 4*1j*D1**2*f*(3*Pk + Pkd*k1)*(7*b1*(t - 1) + f*(6*t - 3) + 7*t*xb1)/(35*d*k1)

        return expr

    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,_,d = cosmo_funcs.unpack_pk(k1,zz,WS=True)

        if sigma is not None:
            expr = -7*1j*D1**2*f*(2*f*k1*sigma*(2*t - 1)*(Pk*(4*k1**6*sigma**6 + 48*k1**4*sigma**4 + 370*k1**2*sigma**2 + 2100) - Pkd*k1*(4*k1**4*sigma**4 + 55*k1**2*sigma**2 + 525)) + 2*k1**3*sigma**3*(2*Pk*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75) - Pkd*k1*(k1**2*sigma**2 + 75))*(b1*(t - 1) + t*xb1) - 3*np.sqrt(2)*np.sqrt(np.pi)*(f*(2*t - 1)*(2*Pk*(3*k1**4*sigma**4 - 55*k1**2*sigma**2 + 350) + Pkd*k1*(-3*k1**4*sigma**4 + 40*k1**2*sigma**2 - 175)) - k1**2*sigma**2*(Pk*(6*k1**2*sigma**2 - 50) + Pkd*k1*(k1**4*sigma**4 - 8*k1**2*sigma**2 + 25))*(b1*(t - 1) + t*xb1))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*d*k1**10*sigma**9)
        else:
            expr = 4*1j*D1**2*f*(9*b1*(2*Pk - Pkd*k1)*(t - 1) + f*(22*Pk - Pkd*k1)*(2*t - 1) + 9*t*xb1*(2*Pk - Pkd*k1))/(45*d*k1)

        return expr

#1st order terms
@add_empty_methods_pk('l0','l2','l4')
class RR1:
    @staticmethod
    def mu(mu,cosmo_funcs,k1,zz=0,t=0):
        Pk,f,D1,b1,xb1,Pkd,_,d,fd,Dd,bd1,xbd1,_,_,_,_ = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True) #unpack all necessary terms
        return 1j*D1*mu*(D1*(Pkd*bd1*k1*t*xb1 - f*fd*mu**2*(2*t - 1)*(4*Pk*(mu**2 - 1) - Pkd*k1*mu**2) - f*(2*Pk*(mu**2 - 1) - Pkd*k1*mu**2)*(bd1*t + xbd1*(t - 1)) + fd*t*xb1*(-2*Pk*(mu**2 - 1) + Pkd*k1*mu**2)) - Dd*f*(2*t - 1)*(-4*Pk*f*mu**2 + 2*Pk*xb1*(mu**2 - 1) - Pkd*k1*mu**2*xb1 + f*mu**4*(4*Pk - Pkd*k1)) - b1*(D1*(t - 1)*(2*Pk*fd*(mu**2 - 1) - Pkd*fd*k1*mu**2 - Pkd*k1*xbd1) + Dd*(2*t - 1)*(2*Pk*f*(mu**2 - 1) - Pkd*f*k1*mu**2 - Pkd*k1*xb1)))/(d*k1)

    @staticmethod
    def l(l,cosmo_funcs, k1, zz=0, t=0, sigma=None,n_mu=16):
        """Returns lth multipole with numeric mu integration over P(k,mu) power spectra"""
        return legendre(RR1.mu,l,cosmo_funcs, k1, zz, t=t, sigma=sigma,n_mu=n_mu)

    ################################### Regular Multipoles #############################################################

    def l1(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,_,d,fd,Dd,bd1,xbd1,_,_,_,_ = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = 3*1j*D1*(-2*k1*sigma*(D1*(f*(-fd*(2*t - 1)*(Pk*(8*k1**2*sigma**2 + 60) - Pkd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15)) + k1**2*sigma**2*(-6*Pk + Pkd*k1*(k1**2*sigma**2 + 3))*(bd1*t + xbd1*(t - 1))) + k1**2*sigma**2*(-b1*(t - 1)*(6*Pk*fd - Pkd*fd*k1*(k1**2*sigma**2 + 3) - Pkd*k1**3*sigma**2*xbd1) + t*xb1*(-6*Pk*fd + Pkd*bd1*k1**3*sigma**2 + Pkd*fd*k1*(k1**2*sigma**2 + 3)))) - Dd*(2*t - 1)*(-Pkd*b1*k1**5*sigma**4*xb1 + f**2*(Pk*(8*k1**2*sigma**2 + 60) - Pkd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15)) - f*k1**2*sigma**2*(-6*Pk + Pkd*k1*(k1**2*sigma**2 + 3))*(b1 + xb1))) + np.sqrt(2)*np.sqrt(np.pi)*(D1*(f*(3*fd*(2*t - 1)*(4*Pk*(k1**2*sigma**2 - 5) + 5*Pkd*k1) + k1**2*sigma**2*(2*Pk*(k1**2*sigma**2 - 3) + 3*Pkd*k1)*(bd1*t + xbd1*(t - 1))) + k1**2*sigma**2*(b1*(t - 1)*(2*Pk*fd*(k1**2*sigma**2 - 3) + 3*Pkd*fd*k1 + Pkd*k1**3*sigma**2*xbd1) + t*xb1*(2*Pk*fd*(k1**2*sigma**2 - 3) + Pkd*bd1*k1**3*sigma**2 + 3*Pkd*fd*k1))) + Dd*(2*t - 1)*(Pkd*b1*k1**5*sigma**4*xb1 + 3*f**2*(4*Pk*(k1**2*sigma**2 - 5) + 5*Pkd*k1) + f*k1**2*sigma**2*(b1 + xb1)*(2*Pk*(k1**2*sigma**2 - 3) + 3*Pkd*k1)))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*d*k1**8*sigma**7)
        else:
            expr = 1j*D1*(D1*(3*f*fd*(8*Pk + 5*Pkd*k1)*(2*t - 1) + 7*f*(4*Pk + 3*Pkd*k1)*(bd1*t + xbd1*(t - 1)) + 7*t*xb1*(4*Pk*fd + 5*Pkd*bd1*k1 + 3*Pkd*fd*k1)) + Dd*f*(2*t - 1)*(3*f*(8*Pk + 5*Pkd*k1) + 7*xb1*(4*Pk + 3*Pkd*k1)) + 7*b1*(D1*(t - 1)*(4*Pk*fd + 3*Pkd*fd*k1 + 5*Pkd*k1*xbd1) + Dd*(2*t - 1)*(4*Pk*f + 3*Pkd*f*k1 + 5*Pkd*k1*xb1)))/(35*d*k1)

        return expr

    def l3(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,_,d,fd,Dd,bd1,xbd1,_,_,_,_ = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = -7*1j*D1*(2*k1*sigma*(D1*(f*(-fd*(2*t - 1)*(4*Pk*(4*k1**4*sigma**4 + 55*k1**2*sigma**2 + 525) - Pkd*k1*(2*k1**6*sigma**6 + 20*k1**4*sigma**4 + 130*k1**2*sigma**2 + 525)) + k1**2*sigma**2*(-2*Pk*(k1**2*sigma**2 + 75) + Pkd*k1*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75))*(bd1*t + xbd1*(t - 1))) + k1**2*sigma**2*(-b1*(t - 1)*(2*Pk*fd*(k1**2*sigma**2 + 75) - Pkd*fd*k1*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75) - Pkd*k1**3*sigma**2*xbd1*(2*k1**2*sigma**2 + 15)) + t*xb1*(-2*Pk*fd*(k1**2*sigma**2 + 75) + Pkd*bd1*k1**3*sigma**2*(2*k1**2*sigma**2 + 15) + Pkd*fd*k1*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75)))) - Dd*(2*t - 1)*(-Pkd*b1*k1**5*sigma**4*xb1*(2*k1**2*sigma**2 + 15) + f**2*(4*Pk*(4*k1**4*sigma**4 + 55*k1**2*sigma**2 + 525) - Pkd*k1*(2*k1**6*sigma**6 + 20*k1**4*sigma**4 + 130*k1**2*sigma**2 + 525)) - f*k1**2*sigma**2*(b1 + xb1)*(-2*Pk*(k1**2*sigma**2 + 75) + Pkd*k1*(2*k1**4*sigma**4 + 16*k1**2*sigma**2 + 75)))) + 3*np.sqrt(2)*np.sqrt(np.pi)*(D1*(f*(fd*(2*t - 1)*(4*Pk*(3*k1**4*sigma**4 - 40*k1**2*sigma**2 + 175) + 5*Pkd*k1*(3*k1**2*sigma**2 - 35)) + k1**2*sigma**2*(2*Pk*(k1**4*sigma**4 - 8*k1**2*sigma**2 + 25) + Pkd*k1*(3*k1**2*sigma**2 - 25))*(bd1*t + xbd1*(t - 1))) + k1**2*sigma**2*(b1*(t - 1)*(2*Pk*fd*(k1**4*sigma**4 - 8*k1**2*sigma**2 + 25) + Pkd*fd*k1*(3*k1**2*sigma**2 - 25) + Pkd*k1**3*sigma**2*xbd1*(k1**2*sigma**2 - 5)) + t*xb1*(2*Pk*fd*(k1**4*sigma**4 - 8*k1**2*sigma**2 + 25) + Pkd*bd1*k1**3*sigma**2*(k1**2*sigma**2 - 5) + Pkd*fd*k1*(3*k1**2*sigma**2 - 25)))) + Dd*(2*t - 1)*(Pkd*b1*k1**5*sigma**4*xb1*(k1**2*sigma**2 - 5) + f**2*(4*Pk*(3*k1**4*sigma**4 - 40*k1**2*sigma**2 + 175) + 5*Pkd*k1*(3*k1**2*sigma**2 - 35)) + f*k1**2*sigma**2*(b1 + xb1)*(2*Pk*(k1**4*sigma**4 - 8*k1**2*sigma**2 + 25) + Pkd*k1*(3*k1**2*sigma**2 - 25))))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*d*k1**10*sigma**9)
        else:
            expr = 2*1j*D1*(D1*(-2*f*fd*(2*Pk - 5*Pkd*k1)*(2*t - 1) - 9*f*(2*Pk - Pkd*k1)*(bd1*t + xbd1*(t - 1)) + 9*fd*t*xb1*(-2*Pk + Pkd*k1)) - Dd*f*(2*t - 1)*(4*Pk*f + 18*Pk*xb1 - 10*Pkd*f*k1 - 9*Pkd*k1*xb1) - 9*b1*(2*Pk - Pkd*k1)*(D1*fd*(t - 1) + Dd*f*(2*t - 1)))/(45*d*k1)

        return expr

#########################################################################################################

#2nd order terms
@add_empty_methods_pk('l1','l3','l4')
class WA2:
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d = cosmo_funcs.unpack_pk(k1,zz,WS=True)

        if sigma is not None:
            expr = -D1**2*f*(-2*f*k1*sigma*(6*Pk*(k1**4*sigma**4*(10*t**2 - 10*t + 3) + 2*k1**2*sigma**2*(54*t**2 - 54*t + 17) + 720*t**2 - 720*t + 240) + k1*(-6*Pkd*(k1**4*sigma**4*(2*t**2 - 2*t + 1) + k1**2*sigma**2*(30*t**2 - 30*t + 11) + 270*t**2 - 270*t + 90) + Pkdd*k1*(k1**2*sigma**2*(6*t**2 - 6*t + 5) + 180*t**2 - 180*t + 60))) + 2*k1**3*sigma**3*(-2*Pk*(7*k1**2*sigma**2 + 48) + k1*(6*Pkd*(k1**2*sigma**2 + 10) + Pkdd*k1*(k1**2*sigma**2 - 12)))*(b1*(t - 1)**2 + t**2*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(f*(2*Pk*(k1**4*sigma**4*(18*t**2 - 18*t + 7) - 6*k1**2*sigma**2*(66*t**2 - 66*t + 23) + 2160*t**2 - 2160*t + 720) + k1*(-8*Pkd*k1**4*sigma**4*(3*t**2 - 3*t + 1) + 6*Pkd*k1**2*sigma**2*(60*t**2 - 60*t + 19) - 540*Pkd*(3*t**2 - 3*t + 1) + Pkdd*k1**5*sigma**4*(6*t**2 - 6*t + 1) - 3*Pkdd*k1**3*sigma**2*(18*t**2 - 18*t + 5) + 60*Pkdd*k1*(3*t**2 - 3*t + 1))) + k1**2*sigma**2*(Pk*(-18*k1**2*sigma**2 + 96) + k1*(2*Pkd*(7*k1**2*sigma**2 - 30) + Pkdd*k1*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12)))*(b1*(t - 1)**2 + t**2*xb1))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(2*d**2*k1**9*sigma**7)
        else:
            expr = -2*D1**2*f*(7*b1*(3*Pk + k1*(5*Pkd + Pkdd*k1))*(t - 1)**2 + f*(Pk*(18*t**2 - 18*t - 1) + k1*(Pkd*(30*t**2 - 30*t - 11) + Pkdd*k1*(6*t**2 - 6*t - 5))) + 7*t**2*xb1*(3*Pk + k1*(5*Pkd + Pkdd*k1)))/(105*d**2*k1**2)

        return expr

    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d = cosmo_funcs.unpack_pk(k1,zz,WS=True)

        if sigma is not None:
            expr =5*D1**2*f*(2*f*k1*sigma*(6*Pk*(2*k1**6*sigma**6*(10*t**2 - 10*t + 3) + 3*k1**4*sigma**4*(98*t**2 - 98*t + 31) + 30*k1**2*sigma**2*(78*t**2 - 78*t + 25) + 15120*t**2 - 15120*t + 5040) + k1*(-6*Pkd*(2*k1**6*sigma**6*(2*t**2 - 2*t + 1) + 4*k1**4*sigma**4*(21*t**2 - 21*t + 8) + 15*k1**2*sigma**2*(48*t**2 - 48*t + 17) + 5670*t**2 - 5670*t + 1890) + Pkdd*k1*(k1**4*sigma**4*(30*t**2 - 30*t + 13) + 135*k1**2*sigma**2*(2*t**2 - 2*t + 1) + 3780*t**2 - 3780*t + 1260))) + 2*k1**3*sigma**3*(2*Pk*(14*k1**4*sigma**4 + 111*k1**2*sigma**2 + 720) + k1*(-6*Pkd*(2*k1**4*sigma**4 + 19*k1**2*sigma**2 + 150) + Pkdd*k1*(k1**4*sigma**4 + 3*k1**2*sigma**2 + 180)))*(b1*(t - 1)**2 + t**2*xb1) + np.sqrt(2)*np.sqrt(np.pi)*(f*(2*Pk*(k1**6*sigma**6*(18*t**2 - 18*t + 7) - 3*k1**4*sigma**4*(186*t**2 - 186*t + 67) + 90*k1**2*sigma**2*(90*t**2 - 90*t + 31) - 45360*t**2 + 45360*t - 15120) + k1*(-2*Pkd*(4*k1**6*sigma**6*(3*t**2 - 3*t + 1) - 3*k1**4*sigma**4*(96*t**2 - 96*t + 31) + 45*k1**2*sigma**2*(78*t**2 - 78*t + 25) - 17010*t**2 + 17010*t - 5670) + Pkdd*k1*(k1**6*sigma**6*(6*t**2 - 6*t + 1) - 12*k1**4*sigma**4*(9*t**2 - 9*t + 2) + 15*k1**2*sigma**2*(66*t**2 - 66*t + 19) - 3780*t**2 + 3780*t - 1260))) + k1**2*sigma**2*(-6*Pk*(3*k1**4*sigma**4 - 43*k1**2*sigma**2 + 240) + k1*(2*Pkd*(7*k1**4*sigma**4 - 93*k1**2*sigma**2 + 450) + Pkdd*k1*(k1**6*sigma**6 - 8*k1**4*sigma**4 + 57*k1**2*sigma**2 - 180)))*(b1*(t - 1)**2 + t**2*xb1))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*d**2*k1**11*sigma**9)
        else:
            expr = 2*D1**2*f*(-11*b1*(6*Pk - k1*(2*Pkd + Pkdd*k1))*(t - 1)**2 + f*(-2*Pk*(54*t**2 - 54*t + 13) + k1*(Pkd*(-12*t**2 + 12*t + 8) + 3*Pkdd*k1*(2*t**2 - 2*t + 1))) + 11*t**2*xb1*(-6*Pk + k1*(2*Pkd + Pkdd*k1)))/(21*d**2*k1**2)

        return expr

@add_empty_methods_pk('l1','l3','l4')
class WARR:
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d,fd,Dd,bd1,xbd1,_,_,_,_ = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = D1*(2*k1*sigma*(D1*(-f*(fd*(1 - 2*t)**2*(2*Pk*(k1**4*sigma**4 + 18*k1**2*sigma**2 + 180) + k1*(-Pkd*(2*k1**4*sigma**4 + 15*k1**2*sigma**2 + 135) + Pkdd*k1*(2*k1**2*sigma**2 + 15))) + k1**2*sigma**2*t*(bd1 + xbd1)*(t - 1)*(2*Pk*(k1**2*sigma**2 + 12) - Pkd*k1*(k1**2*sigma**2 + 15) + 3*Pkdd*k1**2)) + fd*k1**2*sigma**2*(-2*Pk*(k1**2*sigma**2 + 12) + k1*(Pkd*(k1**2*sigma**2 + 15) - 3*Pkdd*k1))*(b1*(t - 1)**2 + t**2*xb1)) - Dd*f*(2*t - 1)*(f*(2*t - 1)*(2*Pk*(k1**4*sigma**4 + 18*k1**2*sigma**2 + 180) + k1*(-Pkd*(2*k1**4*sigma**4 + 15*k1**2*sigma**2 + 135) + Pkdd*k1*(2*k1**2*sigma**2 + 15))) + k1**2*sigma**2*(b1*(t - 1) + t*xb1)*(2*Pk*(k1**2*sigma**2 + 12) - Pkd*k1*(k1**2*sigma**2 + 15) + 3*Pkdd*k1**2)))*np.exp(-k1**2*sigma**2/2) - np.sqrt(2)*np.sqrt(np.pi)*(D1*(f*(-3*fd*(1 - 2*t)**2*(2*Pk*(k1**4*sigma**4 - 14*k1**2*sigma**2 + 60) - k1*(Pkd*(k1**4*sigma**4 - 10*k1**2*sigma**2 + 45) + Pkdd*k1*(k1**2*sigma**2 - 5))) + k1**2*sigma**2*t*(bd1 + xbd1)*(t - 1)*(6*Pk*(k1**2*sigma**2 - 4) + k1*(Pkd*(k1**4*sigma**4 - 4*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 - 3)))) + fd*k1**2*sigma**2*(6*Pk*(k1**2*sigma**2 - 4) + k1*(Pkd*(k1**4*sigma**4 - 4*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 - 3)))*(b1*(t - 1)**2 + t**2*xb1)) - Dd*f*(2*t - 1)*(3*f*(2*t - 1)*(2*Pk*(k1**4*sigma**4 - 14*k1**2*sigma**2 + 60) - k1*(Pkd*(k1**4*sigma**4 - 10*k1**2*sigma**2 + 45) + Pkdd*k1*(k1**2*sigma**2 - 5))) - k1**2*sigma**2*(6*Pk*(k1**2*sigma**2 - 4) + k1*(Pkd*(k1**4*sigma**4 - 4*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 - 3)))*(b1*(t - 1) + t*xb1)))*erf(np.sqrt(2)*k1*sigma/2))/(d**2*k1**9*sigma**7)
        else:
            expr = -4*D1*(3*Pk + k1*(5*Pkd + Pkdd*k1))*(D1*(3*f*fd*(1 - 2*t)**2 + 7*f*t*(bd1 + xbd1)*(t - 1) + 7*fd*t**2*xb1) + Dd*f*(2*t - 1)*(f*(6*t - 3) + 7*t*xb1) + 7*b1*(t - 1)*(D1*fd*(t - 1) + Dd*f*(2*t - 1)))/(105*d**2*k1**2)

        return expr

    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d,fd,Dd,bd1,xbd1,_,_,_,_ = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = 5*D1*(2*k1*sigma*(-D1*(f*(fd*(1 - 2*t)**2*(2*Pk*(2*k1**6*sigma**6 + 51*k1**4*sigma**4 + 450*k1**2*sigma**2 + 3780) + k1*(-Pkd*(4*k1**6*sigma**6 + 51*k1**4*sigma**4 + 360*k1**2*sigma**2 + 2835) + Pkdd*k1*(4*k1**4*sigma**4 + 45*k1**2*sigma**2 + 315))) + k1**2*sigma**2*t*(bd1 + xbd1)*(t - 1)*(Pk*(4*k1**4*sigma**4 + 42*k1**2*sigma**2 + 360) + k1*(-Pkd*(5*k1**4*sigma**4 + 24*k1**2*sigma**2 + 225) + 3*Pkdd*k1*(k1**2*sigma**2 + 15)))) + fd*k1**2*sigma**2*(Pk*(4*k1**4*sigma**4 + 42*k1**2*sigma**2 + 360) + k1*(-Pkd*(5*k1**4*sigma**4 + 24*k1**2*sigma**2 + 225) + 3*Pkdd*k1*(k1**2*sigma**2 + 15)))*(b1*(t - 1)**2 + t**2*xb1)) - Dd*f*(2*t - 1)*(f*(2*t - 1)*(2*Pk*(2*k1**6*sigma**6 + 51*k1**4*sigma**4 + 450*k1**2*sigma**2 + 3780) + k1*(-Pkd*(4*k1**6*sigma**6 + 51*k1**4*sigma**4 + 360*k1**2*sigma**2 + 2835) + Pkdd*k1*(4*k1**4*sigma**4 + 45*k1**2*sigma**2 + 315))) + k1**2*sigma**2*(Pk*(4*k1**4*sigma**4 + 42*k1**2*sigma**2 + 360) + k1*(-Pkd*(5*k1**4*sigma**4 + 24*k1**2*sigma**2 + 225) + 3*Pkdd*k1*(k1**2*sigma**2 + 15)))*(b1*(t - 1) + t*xb1)))*np.exp(-k1**2*sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(D1*(f*(-3*fd*(1 - 2*t)**2*(2*Pk*(k1**6*sigma**6 - 23*k1**4*sigma**4 + 270*k1**2*sigma**2 - 1260) - k1*(Pkd*(k1**6*sigma**6 - 19*k1**4*sigma**4 + 195*k1**2*sigma**2 - 945) + Pkdd*k1*(k1**4*sigma**4 - 20*k1**2*sigma**2 + 105))) + k1**2*sigma**2*t*(bd1 + xbd1)*(t - 1)*(6*Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + k1*(Pkd*(k1**6*sigma**6 - 7*k1**4*sigma**4 + 51*k1**2*sigma**2 - 225) + Pkdd*k1*(k1**4*sigma**4 - 12*k1**2*sigma**2 + 45)))) + fd*k1**2*sigma**2*(6*Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + k1*(Pkd*(k1**6*sigma**6 - 7*k1**4*sigma**4 + 51*k1**2*sigma**2 - 225) + Pkdd*k1*(k1**4*sigma**4 - 12*k1**2*sigma**2 + 45)))*(b1*(t - 1)**2 + t**2*xb1)) - Dd*f*(2*t - 1)*(3*f*(2*t - 1)*(2*Pk*(k1**6*sigma**6 - 23*k1**4*sigma**4 + 270*k1**2*sigma**2 - 1260) - k1*(Pkd*(k1**6*sigma**6 - 19*k1**4*sigma**4 + 195*k1**2*sigma**2 - 945) + Pkdd*k1*(k1**4*sigma**4 - 20*k1**2*sigma**2 + 105))) - k1**2*sigma**2*(6*Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + k1*(Pkd*(k1**6*sigma**6 - 7*k1**4*sigma**4 + 51*k1**2*sigma**2 - 225) + Pkdd*k1*(k1**4*sigma**4 - 12*k1**2*sigma**2 + 45)))*(b1*(t - 1) + t*xb1)))*erf(np.sqrt(2)*k1*sigma/2))/(2*d**2*k1**11*sigma**9)
        else:
            expr = -4*D1*(D1*(f*(fd*(1 - 2*t)**2*(6*Pk + k1*(6*Pkd + Pkdd*k1)) - t*(6*Pk - k1*(2*Pkd + Pkdd*k1))*(bd1 + xbd1)*(t - 1)) + fd*t**2*xb1*(-6*Pk + k1*(2*Pkd + Pkdd*k1))) + Dd*f*(2*t - 1)*(f*(6*Pk + k1*(6*Pkd + Pkdd*k1))*(2*t - 1) + t*xb1*(-6*Pk + k1*(2*Pkd + Pkdd*k1))) - b1*(6*Pk - k1*(2*Pkd + Pkdd*k1))*(t - 1)*(D1*fd*(t - 1) + Dd*f*(2*t - 1)))/(21*d**2*k1**2)

        return expr

@add_empty_methods_pk('l1','l3','l4')
class RR2:
    def l0(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d,fd,Dd,bd1,xbd1,fdd,Ddd,bdd1,xbdd1 = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = (-2*k1*sigma*(D1**2*(b1*k1**2*sigma**2*(2*t**2 - 2*t + 1)*(2*Pk*fd*(k1**2*sigma**2 + 24) + fd*k1*(-2*Pkd*(k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 + 6)) + fdd*(2*Pk*(k1**2*sigma**2 - 12) + 15*Pkd*k1 - Pkdd*k1**2*(k1**2*sigma**2 + 3)) + k1**3*sigma**2*(-Pkd + Pkdd*k1)*(2*bd1 - bdd1)) + f*(2*t**2 - 2*t + 1)*(fd*(Pk*(8*k1**4*sigma**4 + 84*k1**2*sigma**2 + 720) + k1*(-2*Pkd*(k1**4*sigma**4 + 21*k1**2*sigma**2 + 135) + Pkdd*k1*(k1**4*sigma**4 + 7*k1**2*sigma**2 + 30))) - fdd*(12*Pk*(k1**2*sigma**2 + 30) + k1*(-9*Pkd*(2*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))) + k1**2*sigma**2*(2*Pk*bd1*(k1**2*sigma**2 + 24) + 2*Pk*bdd1*(k1**2*sigma**2 - 12) + 15*Pkd*bdd1*k1 - Pkdd*bdd1*k1**2*(k1**2*sigma**2 + 3) + bd1*k1*(-2*Pkd*(k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 + 6)))) - 2*t*(t - 1)*(bd1**2*k1**5*sigma**4*(-Pkd + Pkdd*k1) + 2*bd1*fd*k1**2*sigma**2*(Pk*(-2*k1**2*sigma**2 + 24) + k1*(-15*Pkd + Pkdd*k1**3*sigma**2 + 3*Pkdd*k1)) + fd**2*(12*Pk*(k1**2*sigma**2 + 30) + k1*(-9*Pkd*(2*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))))) + D1*(Dd*(2*b1*k1**2*sigma**2*(fd*(1 - 2*t)**2*(2*Pk*(k1**2*sigma**2 - 12) + 15*Pkd*k1 - Pkdd*k1**2*(k1**2*sigma**2 + 3)) + k1**3*sigma**2*(-Pkd + Pkdd*k1)*(b1*(2*t**2 - 2*t + 1) - bd1*(1 - 2*t)**2)) + f**2*(Pk*(8*k1**4*sigma**4 + 84*k1**2*sigma**2 + 720) + k1*(-2*Pkd*(k1**4*sigma**4 + 21*k1**2*sigma**2 + 135) + Pkdd*k1*(k1**4*sigma**4 + 7*k1**2*sigma**2 + 30)))*(2*t**2 - 2*t + 1) - 2*f*(fd*(1 - 2*t)**2*(12*Pk*(k1**2*sigma**2 + 30) + k1*(-9*Pkd*(2*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))) + k1**2*sigma**2*(-b1*(2*Pk*(k1**2*sigma**2 + 24) + k1*(-2*Pkd*(k1**2*sigma**2 + 15) + Pkdd*k1*(k1**2*sigma**2 + 6)))*(2*t**2 - 2*t + 1) + bd1*(1 - 2*t)**2*(Pk*(-2*k1**2*sigma**2 + 24) + k1*(-15*Pkd + Pkdd*k1*(k1**2*sigma**2 + 3)))))) - Ddd*(2*t**2 - 2*t + 1)*(b1**2*k1**5*sigma**4*(-Pkd + Pkdd*k1) + 2*b1*f*k1**2*sigma**2*(Pk*(-2*k1**2*sigma**2 + 24) + k1*(-15*Pkd + Pkdd*k1**3*sigma**2 + 3*Pkdd*k1)) + f**2*(12*Pk*(k1**2*sigma**2 + 30) + k1*(-9*Pkd*(2*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))))) - 2*Dd**2*t*(t - 1)*(b1**2*k1**5*sigma**4*(-Pkd + Pkdd*k1) + 2*b1*f*k1**2*sigma**2*(Pk*(-2*k1**2*sigma**2 + 24) + k1*(-15*Pkd + Pkdd*k1**3*sigma**2 + 3*Pkdd*k1)) + f**2*(12*Pk*(k1**2*sigma**2 + 30) + k1*(-9*Pkd*(2*k1**2*sigma**2 + 15) + Pkdd*k1*(k1**4*sigma**4 + 5*k1**2*sigma**2 + 15))))) + np.sqrt(2)*np.sqrt(np.pi)*(D1**2*(-b1*k1**2*sigma**2*(2*t**2 - 2*t + 1)*(fd*(-2*Pk*(k1**4*sigma**4 - 7*k1**2*sigma**2 + 24) + k1*(Pkd*(-8*k1**2*sigma**2 + 30) + Pkdd*k1*(k1**2*sigma**2 - 6))) + fdd*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1)) + k1**3*sigma**2*(2*Pkd*bd1 + Pkd*bdd1*(k1**2*sigma**2 - 1) + Pkdd*bd1*k1*(k1**2*sigma**2 - 2) + Pkdd*bdd1*k1)) + f*(2*t**2 - 2*t + 1)*(3*fd*(4*Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + k1*(16*Pkd*k1**2*sigma**2 - 90*Pkd - Pkdd*k1**3*sigma**2 + 10*Pkdd*k1)) - 3*fdd*(4*Pk*(k1**4*sigma**4 - 9*k1**2*sigma**2 + 30) + k1*(9*Pkd*(k1**2*sigma**2 - 5) + 5*Pkdd*k1)) + k1**2*sigma**2*(2*Pk*bd1*(k1**4*sigma**4 - 7*k1**2*sigma**2 + 24) - 2*Pk*bdd1*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + bd1*k1*(Pkd*(8*k1**2*sigma**2 - 30) + Pkdd*k1*(-k1**2*sigma**2 + 6)) - bdd1*k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1))) - 2*t*(t - 1)*(bd1**2*k1**5*sigma**4*(Pkd*(k1**2*sigma**2 - 1) + Pkdd*k1) + 2*bd1*fd*k1**2*sigma**2*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1)) + 3*fd**2*(4*Pk*(k1**4*sigma**4 - 9*k1**2*sigma**2 + 30) + k1*(9*Pkd*(k1**2*sigma**2 - 5) + 5*Pkdd*k1)))) + D1*(Dd*(-b1*k1**2*sigma**2*(2*fd*(1 - 2*t)**2*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1)) + k1**3*sigma**2*(b1*(2*Pkd + Pkdd*k1*(k1**2*sigma**2 - 2))*(2*t**2 - 2*t + 1) + 2*bd1*(1 - 2*t)**2*(Pkd*(k1**2*sigma**2 - 1) + Pkdd*k1))) + 3*f**2*(4*Pk*(k1**4*sigma**4 - 13*k1**2*sigma**2 + 60) + k1*(2*Pkd*(8*k1**2*sigma**2 - 45) + Pkdd*k1*(-k1**2*sigma**2 + 10)))*(2*t**2 - 2*t + 1) - 2*f*(3*fd*(1 - 2*t)**2*(4*Pk*(k1**4*sigma**4 - 9*k1**2*sigma**2 + 30) + k1*(9*Pkd*(k1**2*sigma**2 - 5) + 5*Pkdd*k1)) + k1**2*sigma**2*(b1*(-2*Pk*(k1**4*sigma**4 - 7*k1**2*sigma**2 + 24) + k1*(Pkd*(-8*k1**2*sigma**2 + 30) + Pkdd*k1*(k1**2*sigma**2 - 6)))*(2*t**2 - 2*t + 1) + bd1*(1 - 2*t)**2*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1))))) - Ddd*(2*t**2 - 2*t + 1)*(b1**2*k1**5*sigma**4*(Pkd*(k1**2*sigma**2 - 1) + Pkdd*k1) + 2*b1*f*k1**2*sigma**2*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1)) + 3*f**2*(4*Pk*(k1**4*sigma**4 - 9*k1**2*sigma**2 + 30) + k1*(9*Pkd*(k1**2*sigma**2 - 5) + 5*Pkdd*k1)))) - 2*Dd**2*t*(t - 1)*(b1**2*k1**5*sigma**4*(Pkd*(k1**2*sigma**2 - 1) + Pkdd*k1) + 2*b1*f*k1**2*sigma**2*(2*Pk*(k1**4*sigma**4 - 5*k1**2*sigma**2 + 12) + k1*(5*Pkd*(k1**2*sigma**2 - 3) + 3*Pkdd*k1)) + 3*f**2*(4*Pk*(k1**4*sigma**4 - 9*k1**2*sigma**2 + 30) + k1*(9*Pkd*(k1**2*sigma**2 - 5) + 5*Pkdd*k1))))*erf(np.sqrt(2)*k1*sigma/2)*np.exp(k1**2*sigma**2/2))*np.exp(-k1**2*sigma**2/2)/(4*d**2*k1**9*sigma**7)
        else:
            x0 = k1**2
            x1 = D1*Dd
            x2 = Pk*x1
            x3 = 112*b1
            x4 = f*x3
            x5 = b1*fd
            x6 = x2*x5
            x7 = 48*Pk
            x8 = f*fd
            x9 = x1*x8
            x10 = D1*Ddd
            x11 = b1*f
            x12 = f**2
            x13 = D1*x12
            x14 = Dd*x13
            x15 = Ddd*x13
            x16 = D1**2
            x17 = Pk*x16
            x18 = x17*x5
            x19 = fdd*x17
            x20 = b1*x19
            x21 = 48*x17
            x22 = 224*x2
            x23 = x11*x22
            x24 = fd*t
            x25 = f*x24
            x26 = 192*x2
            x27 = b1*k1
            x28 = Pkd*x27
            x29 = f*x28
            x30 = 140*x1
            x31 = fd*x28
            x32 = Pkd*k1
            x33 = 108*x32
            x34 = Pk*x4
            x35 = x10*x34
            x36 = Pk*x14
            x37 = b1**2
            x38 = x1*x37
            x39 = 70*x32
            x40 = 66*x32
            x41 = x15*x7
            x42 = x10*x37
            x43 = 54*x32
            x44 = x17*x24
            x45 = 56*x20
            x46 = f*fdd
            x47 = x21*x46
            x48 = 70*x16
            x49 = x16*x8
            x50 = x16*x46
            x51 = Dd**2
            x52 = t*x51
            x53 = 280*x29
            x54 = x1*x53
            x55 = x1*x24
            x56 = x1*x25
            x57 = 432*x32
            x58 = t*x10
            x59 = fd**2
            x60 = x21*x59
            x61 = x12*x7
            x62 = t**2
            x63 = 140*x32
            x64 = t*x63
            x65 = 132*x32
            x66 = t*x14
            x67 = Pkdd*x0
            x68 = b1*x67
            x69 = f*x68
            x70 = x1*x69
            x71 = x5*x67
            x72 = x1*x71
            x73 = 30*x67
            x74 = t*x33
            x75 = 140*x28
            x76 = x16*x24
            x77 = fdd*x16
            x78 = x75*x77
            x79 = x16*x25
            x80 = 96*x62
            x81 = 35*x67
            x82 = 9*x67
            x83 = 15*x67
            x84 = x16*x59
            x85 = t*x84
            x86 = 7*x16
            x87 = x68*x77
            x88 = x51*x62
            x89 = x37*x52
            x90 = x12*x52
            x91 = x62*x9
            x92 = 28*x70
            x93 = 120*x67
            x94 = x10*x62
            x95 = 84*x69
            x96 = x62*x63
            x97 = x14*x62
            x98 = 70*x67
            x99 = t*x98
            x100 = 18*x67
            x101 = x33*x62
            x102 = t*x73
            x103 = x16*x62
            x104 = x49*x62
            x105 = 42*x87
            x106 = x62*x84
            x107 = x37*x88
            x108 = x12*x88
            x109 = x62*x98
            x110 = x62*x73
            x111 = Pkdd*k1
            x112 = 2*Pkd + x111
            x113 = t - 1
            x114 = 2*t
            x115 = -x114 + 2*x62 + 1
            x116 = 4*Pk
            x117 = 5*x112*x27
            x118 = 10*Pkd
            x119 = k1*(3*x111 + x118)
            x120 = f*x116 + f*x119 + x117
            expr = -1/210*(7*D1*bd1*(D1*(-f*x115*(8*Pk + k1*(x111 + x118)) + 4*x113*x24*(x116 + x119) + x115*x117) + 2*Dd*x120*(x114 - 1)**2) + 56*Pk*x10*x11 + 24*Pk*x15 - b1*x22*x24 + bd1**2*k1*t*x112*x113*x48 + bdd1*x115*x120*x86 + 24*f*x19 + 96*f*x44 + fdd*x28*x48 - t*x105 + t*x23 - t*x35 + 96*t*x36 - t*x41 - t*x45 - t*x47 + t*x54 - t*x60 - t*x78 + t*x92 + 560*x1*x31*x62 + 140*x10*x29 + 42*x10*x69 - x100*x104 + x100*x66 + x100*x79 - x100*x97 + x101*x15 + x101*x50 - x102*x15 - x102*x50 - 140*x103*x31 - 14*x103*x71 - x104*x65 + x105*x62 + x106*x33 + x106*x73 + x107*x63 + x107*x98 + x108*x33 + x108*x73 + x109*x38 + x109*x42 + x110*x15 + x110*x50 - x14*x40 - x14*x7 - x14*x82 + x15*x43 - x15*x74 + x15*x83 - x17*x8*x80 - 112*x18*x62 - 56*x18 - x2*x4 + 28*x20 - x21*x8 - x23*x62 - x25*x26 + x26*x62*x8 - 560*x28*x55 - x29*x30 + x3*x44 + x30*x31 - x31*x48 - x33*x85 + x33*x9 - x33*x90 - x34*x52 + x34*x88 + x35*x62 - x36*x80 + x38*x39 - x38*x64 + x38*x81 + x38*x96 - x38*x99 + x39*x42 - x40*x49 + x41*x62 - x42*x64 + x42*x81 + x42*x96 - x42*x99 + x43*x50 + x45*x62 + x47*x62 - x49*x82 - x50*x74 + x50*x83 - x52*x53 - x52*x61 - x52*x95 - x53*x58 + x53*x88 + x53*x94 - x54*x62 - 168*x55*x68 - x56*x57 - x56*x93 + x57*x91 - x58*x95 + 224*x6*x62 + 56*x6 + x60*x62 + x61*x88 + 168*x62*x72 + x62*x78 - x62*x92 - x63*x89 + x65*x66 + x65*x79 - x65*x97 + 14*x68*x76 + x7*x9 - 14*x70 - x71*x86 + 42*x72 - x73*x85 + x73*x9 - x73*x90 + x75*x76 + 21*x87 + x88*x95 - x89*x98 + x91*x93 + x94*x95)/(d**2*x0)

        return expr

    def l2(cosmo_funcs,k1,zz=0,t=0,sigma=None):
        Pk,f,D1,b1,xb1,Pkd,Pkdd,d,fd,Dd,bd1,xbd1,fdd,Ddd,bdd1,xbdd1 = cosmo_funcs.unpack_pk(k1,zz,WS=True,RR=True)

        if sigma is not None:
            expr = (-10*sigma*(D1**2*(b1*sigma**2*(t - 1)**2*(fd*(2*Pk*(5*sigma**4 + 33*sigma**2 + 360) + k1*(-2*Pkd*(2*sigma**4 + 24*sigma**2 + 225) + Pkdd*k1*(2*sigma**4 + 15*sigma**2 + 90))) - fdd*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))) + k1*sigma**2*(-Pkd*(2*xbd1*(2*sigma**2 + 9) + xbdd1*(sigma**2 - 9)) + Pkdd*k1*(xbd1*(sigma**2 + 18) - xbdd1*(2*sigma**2 + 9)))) + f*(fd*(4*Pk*(4*sigma**6 + 63*sigma**4 + 495*sigma**2 + 3780) + k1*(-2*Pkd*(2*sigma**6 + 48*sigma**4 + 450*sigma**2 + 2835) + Pkdd*k1*(2*sigma**6 + 20*sigma**4 + 135*sigma**2 + 630)))*(2*t**2 - 2*t + 1) - fdd*(60*Pk*(sigma**4 + 9*sigma**2 + 126) + k1*(-9*Pkd*(4*sigma**4 + 45*sigma**2 + 315) + Pkdd*k1*(2*sigma**6 + 16*sigma**4 + 90*sigma**2 + 315)))*(2*t**2 - 2*t + 1) + sigma**2*(bd1*t**2*(2*Pk*(5*sigma**4 + 33*sigma**2 + 360) + k1*(-2*Pkd*(2*sigma**4 + 24*sigma**2 + 225) + Pkdd*k1*(2*sigma**4 + 15*sigma**2 + 90))) - bdd1*t**2*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))) + (t - 1)**2*(2*Pk*(xbd1*(5*sigma**4 + 33*sigma**2 + 360) - xbdd1*(sigma**4 + 3*sigma**2 + 180)) + k1*(-2*Pkd*xbd1*(2*sigma**4 + 24*sigma**2 + 225) + 15*Pkd*xbdd1*(sigma**2 + 15) + Pkdd*k1*xbd1*(2*sigma**4 + 15*sigma**2 + 90) - Pkdd*k1*xbdd1*(2*sigma**4 + 12*sigma**2 + 45))))) - t*(2*fd**2*(t - 1)*(60*Pk*(sigma**4 + 9*sigma**2 + 126) + k1*(-9*Pkd*(4*sigma**4 + 45*sigma**2 + 315) + Pkdd*k1*(2*sigma**6 + 16*sigma**4 + 90*sigma**2 + 315))) + fd*sigma**2*(-2*Pk*(t*xb1*(5*sigma**4 + 33*sigma**2 + 360) - 2*t*xbd1*(sigma**4 + 3*sigma**2 + 180) + 2*xbd1*(sigma**4 + 3*sigma**2 + 180)) + 2*bd1*(t - 1)*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))) + k1*(Pkd*(t*xb1*(4*sigma**4 + 48*sigma**2 + 450) - 30*t*xbd1*(sigma**2 + 15) + 30*xbd1*(sigma**2 + 15)) - Pkdd*k1*(t*xb1*(2*sigma**4 + 15*sigma**2 + 90) - 2*t*xbd1*(2*sigma**4 + 12*sigma**2 + 45) + 2*xbd1*(2*sigma**4 + 12*sigma**2 + 45)))) + fdd*sigma**2*t*xb1*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))) + k1*sigma**4*(bd1*(2*Pkd*(t*xb1*(2*sigma**2 + 9) + t*xbd1*(sigma**2 - 9) - xbd1*(sigma**2 - 9)) - Pkdd*k1*(t*xb1*(sigma**2 + 18) - 2*t*xbd1*(2*sigma**2 + 9) + 2*xbd1*(2*sigma**2 + 9))) + bdd1*t*xb1*(Pkd*(sigma**2 - 9) + Pkdd*k1*(2*sigma**2 + 9))))) + D1*(Dd*(f**2*(4*Pk*(4*sigma**6 + 63*sigma**4 + 495*sigma**2 + 3780) + k1*(-2*Pkd*(2*sigma**6 + 48*sigma**4 + 450*sigma**2 + 2835) + Pkdd*k1*(2*sigma**6 + 20*sigma**4 + 135*sigma**2 + 630)))*(2*t**2 - 2*t + 1) - 2*f*fd*(1 - 2*t)**2*(60*Pk*(sigma**4 + 9*sigma**2 + 126) + k1*(-9*Pkd*(4*sigma**4 + 45*sigma**2 + 315) + Pkdd*k1*(2*sigma**6 + 16*sigma**4 + 90*sigma**2 + 315))) + f*sigma**2*(20*Pk*sigma**4*t**2*xb1 - 8*Pk*sigma**4*t**2*xbd1 - 20*Pk*sigma**4*t*xb1 + 12*Pk*sigma**4*t*xbd1 + 10*Pk*sigma**4*xb1 - 4*Pk*sigma**4*xbd1 + 132*Pk*sigma**2*t**2*xb1 - 24*Pk*sigma**2*t**2*xbd1 - 132*Pk*sigma**2*t*xb1 + 36*Pk*sigma**2*t*xbd1 + 66*Pk*sigma**2*xb1 - 12*Pk*sigma**2*xbd1 + 1440*Pk*t**2*xb1 - 1440*Pk*t**2*xbd1 - 1440*Pk*t*xb1 + 2160*Pk*t*xbd1 + 720*Pk*xb1 - 720*Pk*xbd1 - 8*Pkd*k1*sigma**4*t**2*xb1 + 8*Pkd*k1*sigma**4*t*xb1 - 4*Pkd*k1*sigma**4*xb1 - 96*Pkd*k1*sigma**2*t**2*xb1 + 60*Pkd*k1*sigma**2*t**2*xbd1 + 96*Pkd*k1*sigma**2*t*xb1 - 90*Pkd*k1*sigma**2*t*xbd1 - 48*Pkd*k1*sigma**2*xb1 + 30*Pkd*k1*sigma**2*xbd1 - 900*Pkd*k1*t**2*xb1 + 900*Pkd*k1*t**2*xbd1 + 900*Pkd*k1*t*xb1 - 1350*Pkd*k1*t*xbd1 - 450*Pkd*k1*xb1 + 450*Pkd*k1*xbd1 + 4*Pkdd*k1**2*sigma**4*t**2*xb1 - 8*Pkdd*k1**2*sigma**4*t**2*xbd1 - 4*Pkdd*k1**2*sigma**4*t*xb1 + 12*Pkdd*k1**2*sigma**4*t*xbd1 + 2*Pkdd*k1**2*sigma**4*xb1 - 4*Pkdd*k1**2*sigma**4*xbd1 + 30*Pkdd*k1**2*sigma**2*t**2*xb1 - 48*Pkdd*k1**2*sigma**2*t**2*xbd1 - 30*Pkdd*k1**2*sigma**2*t*xb1 + 72*Pkdd*k1**2*sigma**2*t*xbd1 + 15*Pkdd*k1**2*sigma**2*xb1 - 24*Pkdd*k1**2*sigma**2*xbd1 + 180*Pkdd*k1**2*t**2*xb1 - 180*Pkdd*k1**2*t**2*xbd1 - 180*Pkdd*k1**2*t*xb1 + 270*Pkdd*k1**2*t*xbd1 + 90*Pkdd*k1**2*xb1 - 90*Pkdd*k1**2*xbd1 + b1*(2*Pk*(5*sigma**4 + 33*sigma**2 + 360) + k1*(-2*Pkd*(2*sigma**4 + 24*sigma**2 + 225) + Pkdd*k1*(2*sigma**4 + 15*sigma**2 + 90)))*(2*t**2 - 2*t + 1) - 2*bd1*t*(2*t - 1)*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45)))) - sigma**2*(b1*(2*fd*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45)))*(2*t**2 - 3*t + 1) + k1*sigma**2*(2*Pkd*(xb1*(2*sigma**2 + 9)*(2*t**2 - 2*t + 1) + xbd1*(sigma**2 - 9)*(2*t**2 - 3*t + 1)) + Pkdd*k1*(-xb1*(sigma**2 + 18)*(2*t**2 - 2*t + 1) + 2*xbd1*(2*sigma**2 + 9)*(2*t**2 - 3*t + 1)))) + 2*t*xb1*(2*t - 1)*(bd1*k1*sigma**2*(Pkd*(sigma**2 - 9) + Pkdd*k1*(2*sigma**2 + 9)) + fd*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45)))))) - Ddd*(2*t**2 - 2*t + 1)*(b1*k1*sigma**4*xb1*(Pkd*(sigma**2 - 9) + Pkdd*k1*(2*sigma**2 + 9)) + f**2*(60*Pk*(sigma**4 + 9*sigma**2 + 126) + k1*(-9*Pkd*(4*sigma**4 + 45*sigma**2 + 315) + Pkdd*k1*(2*sigma**6 + 16*sigma**4 + 90*sigma**2 + 315))) + f*sigma**2*(b1 + xb1)*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))))) - 2*Dd**2*t*(t - 1)*(b1*k1*sigma**4*xb1*(Pkd*(sigma**2 - 9) + Pkdd*k1*(2*sigma**2 + 9)) + f**2*(60*Pk*(sigma**4 + 9*sigma**2 + 126) + k1*(-9*Pkd*(4*sigma**4 + 45*sigma**2 + 315) + Pkdd*k1*(2*sigma**6 + 16*sigma**4 + 90*sigma**2 + 315))) + f*sigma**2*(b1 + xb1)*(2*Pk*(sigma**4 + 3*sigma**2 + 180) + k1*(-15*Pkd*(sigma**2 + 15) + Pkdd*k1*(2*sigma**4 + 12*sigma**2 + 45))))) + 5*np.sqrt(2)*np.sqrt(np.pi)*(D1**2*(-b1*sigma**2*(t - 1)**2*(fd*(2*Pk*(sigma**6 - 10*sigma**4 + 87*sigma**2 - 360) + k1*(2*Pkd*(4*sigma**4 - 51*sigma**2 + 225) - Pkdd*k1*(sigma**4 - 15*sigma**2 + 90))) + fdd*(-2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(-5*Pkd*(sigma**4 - 12*sigma**2 + 45) - 3*Pkdd*k1*(sigma**2 - 15))) - k1*sigma**2*(2*Pkd*xbd1*(sigma**2 - 9) + Pkd*xbdd1*(sigma**4 - 4*sigma**2 + 9) + Pkdd*k1*xbd1*(sigma**4 - 5*sigma**2 + 18) + Pkdd*k1*xbdd1*(sigma**2 - 9))) + f*(-3*fd*(4*Pk*(sigma**6 - 22*sigma**4 + 255*sigma**2 - 1260) + k1*(2*Pkd*(8*sigma**4 - 165*sigma**2 + 945) - Pkdd*k1*(sigma**4 - 25*sigma**2 + 210)))*(2*t**2 - 2*t + 1) + 3*fdd*(4*Pk*(sigma**6 - 18*sigma**4 + 165*sigma**2 - 630) + k1*(9*Pkd*(sigma**4 - 20*sigma**2 + 105) + 5*Pkdd*k1*(sigma**2 - 21)))*(2*t**2 - 2*t + 1) + sigma**2*(bd1*t**2*(-2*Pk*(sigma**6 - 10*sigma**4 + 87*sigma**2 - 360) + k1*(-2*Pkd*(4*sigma**4 - 51*sigma**2 + 225) + Pkdd*k1*(sigma**4 - 15*sigma**2 + 90))) + bdd1*t**2*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15))) - (t - 1)**2*(2*Pk*(xbd1*(sigma**6 - 10*sigma**4 + 87*sigma**2 - 360) - xbdd1*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180)) - k1*(-2*Pkd*xbd1*(4*sigma**4 - 51*sigma**2 + 225) + 5*Pkd*xbdd1*(sigma**4 - 12*sigma**2 + 45) + Pkdd*k1*xbd1*(sigma**4 - 15*sigma**2 + 90) + 3*Pkdd*k1*xbdd1*(sigma**2 - 15))))) + t*(6*fd**2*(t - 1)*(4*Pk*(sigma**6 - 18*sigma**4 + 165*sigma**2 - 630) + k1*(9*Pkd*(sigma**4 - 20*sigma**2 + 105) + 5*Pkdd*k1*(sigma**2 - 21))) + fd*sigma**2*(-2*Pk*(t*xb1*(sigma**6 - 10*sigma**4 + 87*sigma**2 - 360) - 2*t*xbd1*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + 2*xbd1*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180)) + 2*bd1*(t - 1)*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15))) + k1*(-2*Pkd*(t*xb1*(4*sigma**4 - 51*sigma**2 + 225) - 5*t*xbd1*(sigma**4 - 12*sigma**2 + 45) + 5*xbd1*(sigma**4 - 12*sigma**2 + 45)) + Pkdd*k1*(t*xb1*(sigma**4 - 15*sigma**2 + 90) + 6*t*xbd1*(sigma**2 - 15) - 6*xbd1*(sigma**2 - 15)))) + fdd*sigma**2*t*xb1*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15))) + k1*sigma**4*(bd1*(2*Pkd*(t*xb1*(sigma**2 - 9) + t*xbd1*(sigma**4 - 4*sigma**2 + 9) - xbd1*(sigma**4 - 4*sigma**2 + 9)) + Pkdd*k1*(t*xb1*(sigma**4 - 5*sigma**2 + 18) + 2*t*xbd1*(sigma**2 - 9) - 2*xbd1*(sigma**2 - 9))) + bdd1*t*xb1*(Pkd*(sigma**4 - 4*sigma**2 + 9) + Pkdd*k1*(sigma**2 - 9))))) + D1*(Dd*(-3*f**2*(4*Pk*(sigma**6 - 22*sigma**4 + 255*sigma**2 - 1260) + k1*(2*Pkd*(8*sigma**4 - 165*sigma**2 + 945) - Pkdd*k1*(sigma**4 - 25*sigma**2 + 210)))*(2*t**2 - 2*t + 1) + f*(6*fd*(1 - 2*t)**2*(4*Pk*(sigma**6 - 18*sigma**4 + 165*sigma**2 - 630) + k1*(9*Pkd*(sigma**4 - 20*sigma**2 + 105) + 5*Pkdd*k1*(sigma**2 - 21))) + sigma**2*(-4*Pk*sigma**6*t**2*xb1 + 8*Pk*sigma**6*t**2*xbd1 + 4*Pk*sigma**6*t*xb1 - 12*Pk*sigma**6*t*xbd1 - 2*Pk*sigma**6*xb1 + 4*Pk*sigma**6*xbd1 + 40*Pk*sigma**4*t**2*xb1 - 64*Pk*sigma**4*t**2*xbd1 - 40*Pk*sigma**4*t*xb1 + 96*Pk*sigma**4*t*xbd1 + 20*Pk*sigma**4*xb1 - 32*Pk*sigma**4*xbd1 - 348*Pk*sigma**2*t**2*xb1 + 456*Pk*sigma**2*t**2*xbd1 + 348*Pk*sigma**2*t*xb1 - 684*Pk*sigma**2*t*xbd1 - 174*Pk*sigma**2*xb1 + 228*Pk*sigma**2*xbd1 + 1440*Pk*t**2*xb1 - 1440*Pk*t**2*xbd1 - 1440*Pk*t*xb1 + 2160*Pk*t*xbd1 + 720*Pk*xb1 - 720*Pk*xbd1 - 16*Pkd*k1*sigma**4*t**2*xb1 + 20*Pkd*k1*sigma**4*t**2*xbd1 + 16*Pkd*k1*sigma**4*t*xb1 - 30*Pkd*k1*sigma**4*t*xbd1 - 8*Pkd*k1*sigma**4*xb1 + 10*Pkd*k1*sigma**4*xbd1 + 204*Pkd*k1*sigma**2*t**2*xb1 - 240*Pkd*k1*sigma**2*t**2*xbd1 - 204*Pkd*k1*sigma**2*t*xb1 + 360*Pkd*k1*sigma**2*t*xbd1 + 102*Pkd*k1*sigma**2*xb1 - 120*Pkd*k1*sigma**2*xbd1 - 900*Pkd*k1*t**2*xb1 + 900*Pkd*k1*t**2*xbd1 + 900*Pkd*k1*t*xb1 - 1350*Pkd*k1*t*xbd1 - 450*Pkd*k1*xb1 + 450*Pkd*k1*xbd1 + 2*Pkdd*k1**2*sigma**4*t**2*xb1 - 2*Pkdd*k1**2*sigma**4*t*xb1 + Pkdd*k1**2*sigma**4*xb1 - 30*Pkdd*k1**2*sigma**2*t**2*xb1 + 12*Pkdd*k1**2*sigma**2*t**2*xbd1 + 30*Pkdd*k1**2*sigma**2*t*xb1 - 18*Pkdd*k1**2*sigma**2*t*xbd1 - 15*Pkdd*k1**2*sigma**2*xb1 + 6*Pkdd*k1**2*sigma**2*xbd1 + 180*Pkdd*k1**2*t**2*xb1 - 180*Pkdd*k1**2*t**2*xbd1 - 180*Pkdd*k1**2*t*xb1 + 270*Pkdd*k1**2*t*xbd1 + 90*Pkdd*k1**2*xb1 - 90*Pkdd*k1**2*xbd1 - b1*(2*Pk*(sigma**6 - 10*sigma**4 + 87*sigma**2 - 360) + k1*(2*Pkd*(4*sigma**4 - 51*sigma**2 + 225) - Pkdd*k1*(sigma**4 - 15*sigma**2 + 90)))*(2*t**2 - 2*t + 1) + 2*bd1*t*(2*t - 1)*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15))))) + sigma**2*(b1*(2*fd*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15)))*(2*t**2 - 3*t + 1) + k1*sigma**2*(2*Pkd*(xb1*(sigma**2 - 9)*(2*t**2 - 2*t + 1) + xbd1*(sigma**4 - 4*sigma**2 + 9)*(2*t**2 - 3*t + 1)) + Pkdd*k1*(xb1*(sigma**4 - 5*sigma**2 + 18)*(2*t**2 - 2*t + 1) + 2*xbd1*(sigma**2 - 9)*(2*t**2 - 3*t + 1)))) + 2*t*xb1*(2*t - 1)*(bd1*k1*sigma**2*(Pkd*(sigma**4 - 4*sigma**2 + 9) + Pkdd*k1*(sigma**2 - 9)) + fd*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15)))))) + Ddd*(2*t**2 - 2*t + 1)*(b1*k1*sigma**4*xb1*(Pkd*(sigma**4 - 4*sigma**2 + 9) + Pkdd*k1*(sigma**2 - 9)) + 3*f**2*(4*Pk*(sigma**6 - 18*sigma**4 + 165*sigma**2 - 630) + k1*(9*Pkd*(sigma**4 - 20*sigma**2 + 105) + 5*Pkdd*k1*(sigma**2 - 21))) + f*sigma**2*(b1 + xb1)*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15))))) + 2*Dd**2*t*(t - 1)*(b1*k1*sigma**4*xb1*(Pkd*(sigma**4 - 4*sigma**2 + 9) + Pkdd*k1*(sigma**2 - 9)) + 3*f**2*(4*Pk*(sigma**6 - 18*sigma**4 + 165*sigma**2 - 630) + k1*(9*Pkd*(sigma**4 - 20*sigma**2 + 105) + 5*Pkdd*k1*(sigma**2 - 21))) + f*sigma**2*(b1 + xb1)*(2*Pk*(sigma**6 - 8*sigma**4 + 57*sigma**2 - 180) + k1*(5*Pkd*(sigma**4 - 12*sigma**2 + 45) + 3*Pkdd*k1*(sigma**2 - 15)))))*erf(np.sqrt(2)*sigma/2)*np.exp(sigma**2/2))*np.exp(-sigma**2/2)/(8*d**2*k1**2*sigma**9)
        else:
            x0 = k1**2
            x1 = Dd**2
            x2 = t - 1
            x3 = 12*Pk
            x4 = 22*Pk
            x5 = 9*Pkd
            x6 = Pkdd*k1
            x7 = 5*x6
            x8 = k1*(x5 + x7)
            x9 = 5*Pkd
            x10 = k1*(6*x6 + x9)
            x11 = f*x3 - f*x8 - x10*xb1 + x4*xb1
            x12 = k1*xb1
            x13 = Pkd - x6
            x14 = 7*x13
            x15 = -f*x10 + f*x4 + x12*x14
            x16 = 2*t
            x17 = D1**2
            x18 = 2*Pk
            x19 = fdd*x4
            x20 = 4*Pkd
            x21 = k1*(x20 - x7)
            x22 = fdd*x10
            x23 = k1*x14
            x24 = t**2
            x25 = 2*x24
            x26 = x25 + 1
            x27 = -x16 + x26
            x28 = f*x27
            x29 = -3*t + x26
            x30 = fd*(-x10 + x4)
            x31 = x27*(6*Pk + k1*(3*Pkd + 2*x6))
            x32 = x16 - 1
            x33 = bd1*k1
            x34 = x16*xb1
            x35 = Pk*xbd1
            x36 = t*xb1
            x37 = 4*Pk
            x38 = t*x35
            x39 = k1*xbd1
            x40 = Pkd*x39
            x41 = Pkd*k1
            x42 = 8*x41
            x43 = t*x40
            x44 = x24*xb1
            x45 = Pkdd*x0
            x46 = 5*x45
            x47 = 12*xbd1
            x48 = 10*x45
            x49 = x45*xbd1
            x50 = t*x49
            x51 = k1*x9
            x52 = 6*x45
            x53 = x3 - x8
            x54 = 2*fd
            x55 = 22*xbd1
            x56 = 5*xbd1
            x57 = x4*xbdd1
            x58 = t*xbdd1
            x59 = x20*x39
            x60 = x51*xbdd1
            x61 = bdd1*x24
            x62 = x45*x56
            x63 = x52*xbdd1
            expr = (1/21)*(D1*(Dd*(2*f**2*x31 - f*(bd1*x16*x32*(-x4 + x51 + x52) + x12*x20 + x18*xb1 - 88*x24*x35 + 20*x24*x40 + 24*x24*x49 - x32**2*x53*x54 - 44*x35 - x36*x37 - x36*x42 + x36*x48 + x37*x44 + 132*x38 + 10*x40 + x42*x44 - 30*x43 - x44*x48 + x45*x47 - x46*xb1 - 36*x50) + x32*x34*(-fd*x10 + fd*x4 + x14*x33)) + Ddd*x11*x28) - b1*(-D1*(Dd*(-14*k1*x13*(x27*xb1 - x29*xbd1) - x28*(x18 + x21) + 2*x29*x30) + Ddd*x15*x27) - x1*x15*x16*x2 + x17*x2**2*(fd*x18 + fd*x21 - x19 + x22 + x23*(2*xbd1 - xbdd1))) + 2*f*t*x1*x11*x2 - x17*(f*(Pk*bd1*x25 + 44*Pk*x58 - bd1*x24*x46 - fdd*x27*(12*Pk - k1*x5 - x46) + x18*xbd1 + x20*x24*x33 - x24*x57 + x24*x59 + x24*x60 - x24*x62 + x24*x63 + x25*x35 - x31*x54 - 4*x38 - x4*x61 - 10*x41*x58 - 8*x43 - 12*x45*x58 + 10*x50 + x51*x61 + x52*x61 - x57 + x59 + x60 - x62 + x63) - t*(2*bd1*(x2*x30 - x23*(t*(xb1 - xbd1) + xbd1)) + 2*fd**2*x2*x53 - fd*(-k1*(-2*Pkd*(t*x56 + x34 - x56) + Pkdd*k1*(-t*x47 + 5*x36 + x47)) + x18*(t*(-x55 + xb1) + x55)) + x36*(bdd1*x23 + x19 - x22))))/(d**2*x0)

        return expr

