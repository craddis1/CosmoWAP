import numpy as np
   
#Netwonian, plane parallel constant redshift limit
class Bk0:
    def l0(cosmo_functions,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,nonlin=False,growth2=False):
        
        #get generic cosmology parameters
        k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2 = cosmo_functions.get_params(k1,k2,k3,theta,zz,nonlin=nonlin,growth2=growth2)

        cos = lambda x: np.cos(x)
        sin = lambda x: np.sin(x)
        
        perm12 = D1**4*Pk1*Pk2*(420*C*b1**2*f*k1**3*k2 - 420*C*b1**2*f*k1**2*k2**2*cos(3*theta) + 420*C*b1**2*f*k1*k2**3 - 42*C*b1*f**2*k1**3*k2*cos(4*theta) + 378*C*b1*f**2*k1**3*k2 - 504*C*b1*f**2*k1**2*k2**2*cos(3*theta) - 42*C*b1*f**2*k1*k2**3*cos(4*theta) + 378*C*b1*f**2*k1*k2**3 - 36*C*f**3*k1**3*k2*cos(4*theta) + 72*C*f**3*k1**3*k2 - 126*C*f**3*k1**2*k2**2*cos(3*theta) - 18*C*f**3*k1**2*k2**2*cos(5*theta) - 36*C*f**3*k1*k2**3*cos(4*theta) + 72*C*f**3*k1*k2**3 + 630*K*b1**3*k1*k2*k3**2 + 420*K*b1**2*f*k1*k2*k3**2 - 21*K*b1*f**2*k1*k2*k3**2*cos(4*theta) + 63*K*b1*f**2*k1*k2*k3**2 + 840*b1**3*f*k1*k2*k3**2 + 1890*b1**3*k1*k2*k3**2 + 1260*b1**2*b2*k1*k2*k3**2 + 1176*b1**2*f**2*k1*k2*k3**2 + 840*b1**2*f*k1**3*k2 + 420*b1**2*f*k1**2*k2**2*cos(3*theta) + 840*b1**2*f*k1*k2**3 + 1260*b1**2*f*k1*k2*k3**2 - 1260*b1**2*g2*k1*k2*k3**2 + 840*b1*b2*f*k1*k2*k3**2 + 36*b1*f**3*k1**2*k3**2*cos(3*theta) + 648*b1*f**3*k1*k2*k3**2 + 36*b1*f**3*k2**2*k3**2*cos(3*theta) + 42*b1*f**2*k1**4*cos(3*theta) + 42*b1*f**2*k1**3*k2*cos(4*theta) + 966*b1*f**2*k1**3*k2 + 588*b1*f**2*k1**2*k2**2*cos(3*theta) + 42*b1*f**2*k1**2*k3**2*cos(3*theta) + 42*b1*f**2*k1*k2**3*cos(4*theta) + 966*b1*f**2*k1*k2**3 + 21*b1*f**2*k1*k2*k3**2*cos(4*theta) + 273*b1*f**2*k1*k2*k3**2 + 42*b1*f**2*k2**4*cos(3*theta) + 42*b1*f**2*k2**2*k3**2*cos(3*theta) - 840*b1*f*g2*k1*k2*k3**2 + 168*b2*f**2*k1*k2*k3**2 + 20*f**4*k1**2*k3**2*cos(3*theta) + 8*f**4*k1*k2*k3**2*cos(4*theta) + 144*f**4*k1*k2*k3**2 + 20*f**4*k2**2*k3**2*cos(3*theta) + 36*f**3*k1**4*cos(3*theta) + 54*f**3*k1**3*k2*cos(4*theta) + 306*f**3*k1**3*k2 + 270*f**3*k1**2*k2**2*cos(3*theta) + 18*f**3*k1**2*k2**2*cos(5*theta) + 54*f**3*k1*k2**3*cos(4*theta) + 306*f**3*k1*k2**3 + 36*f**3*k2**4*cos(3*theta) + 42*f**2*g2*k1*k2*k3**2*cos(4*theta) - 126*f**2*g2*k1*k2*k3**2 - 2*k1*k2*(315*b1**3*k3**2*(K - 1) - 42*b1**2*(4*f**2*k3**2 - 5*f*(k1**2*(C - 2) + k2**2*(C - 2) + k3**2*(K - 1)) + 15*g2*k3**2) - 3*b1*f*(72*f**2*k3**2 - 7*f*(k1**2*(8*C - 24) + k2**2*(8*C - 24) + k3**2*(K - 5)) + 140*g2*k3**2) - 2*f**2*(32*f**2*k3**2 + f*(90 - 9*C)*(k1**2 + k2**2) + k3**2*(21*b2 + 21*g2)))*cos(2*theta) + (420*b1**3*k3**2*(f + 3)*(k1**2 + k2**2) + 84*b1**2*f*(5*k1**4 + k1**2*(k2**2*(5*C + 25) + k3**2*(9*f + 10)) + 5*k2**4 + k2**2*k3**2*(9*f + 10)) + 42*b1*f**2*(11*k1**4 + k1**2*(k2**2*(12*C + 58) + k3**2*(12*f + 5)) + 11*k2**4 + k2**2*k3**2*(12*f + 5)) + 24*f**3*(5*f*k2**2*k3**2 + 6*k1**4 + k1**2*(5*f*k3**2 + k2**2*(6*C + 33)) + 6*k2**4))*cos(theta))/(1260*k1*k2*k3**2)
        perm13 = -D1**4*Pk1*Pk3*(840*C*b1**2*f*k1**3*k3**2 + 420*C*b1**2*f*k1*k2**2*k3**2 - 840*C*b1**2*f*k1*k3**4 + 336*C*b1*f**2*k1**5 + 168*C*b1*f**2*k1**4*k2*cos(3*theta) + 42*C*b1*f**2*k1**3*k2**2*cos(4*theta) + 1722*C*b1*f**2*k1**3*k2**2 + 252*C*b1*f**2*k1**2*k2**3*cos(3*theta) + 168*C*b1*f**2*k1**2*k2*k3**2*cos(3*theta) + 252*C*b1*f**2*k1*k2**4 + 42*C*b1*f**2*k1*k2**2*k3**2*cos(4*theta) - 294*C*b1*f**2*k1*k2**2*k3**2 - 336*C*b1*f**2*k1*k3**4 + 216*C*f**3*k1**5 + 216*C*f**3*k1**4*k2*cos(3*theta) + 108*C*f**3*k1**3*k2**2*cos(4*theta) + 1008*C*f**3*k1**3*k2**2 - 216*C*f**3*k1**3*k3**2 + 342*C*f**3*k1**2*k2**3*cos(3*theta) + 18*C*f**3*k1**2*k2**3*cos(5*theta) - 72*C*f**3*k1**2*k2*k3**2*cos(3*theta) + 36*C*f**3*k1*k2**4*cos(4*theta) + 144*C*f**3*k1*k2**4 - 216*C*f**3*k1*k2**2*k3**2 + 1260*K*b1**3*k1**3*k3**2 + 630*K*b1**3*k1*k2**2*k3**2 - 1260*K*b1**3*k1*k3**4 + 420*K*b1**2*f*k1**5 + 1470*K*b1**2*f*k1**3*k2**2 + 210*K*b1**2*f*k1**2*k2**3*cos(3*theta) + 210*K*b1**2*f*k1*k2**4 - 210*K*b1**2*f*k1*k2**2*k3**2 - 420*K*b1**2*f*k1*k3**4 + 252*K*b1*f**2*k1**5 + 798*K*b1*f**2*k1**3*k2**2 - 252*K*b1*f**2*k1**3*k3**2 + 210*K*b1*f**2*k1**2*k2**3*cos(3*theta) + 21*K*b1*f**2*k1*k2**4*cos(4*theta) + 105*K*b1*f**2*k1*k2**4 - 168*K*b1*f**2*k1*k2**2*k3**2 - 420*b1**3*f*k1*k2**2*k3**2 - 630*b1**3*k1*k2**2*k3**2 - 1260*b1**2*b2*k1*k3**4 - 504*b1**2*f**2*k1**3*k2**2 - 252*b1**2*f**2*k1*k2**4 + 336*b1**2*f**2*k1*k2**2*k3**2 - 630*b1**2*f*k1**3*k2**2 - 420*b1**2*f*k1**3*k3**2 - 210*b1**2*f*k1**2*k2**3*cos(3*theta) - 210*b1**2*f*k1*k2**4 - 210*b1**2*f*k1*k2**2*k3**2 + 420*b1**2*f*k1*k3**4 - 2520*b1**2*g2*k1**3*k3**2 - 1260*b1**2*g2*k1*k2**2*k3**2 + 2520*b1**2*g2*k1*k3**4 - 420*b1*b2*f*k1**3*k3**2 - 420*b1*b2*f*k1*k2**2*k3**2 - 420*b1*b2*f*k1*k3**4 - 216*b1*f**3*k1**3*k2**2 + 216*b1*f**3*k1*k2**4 + 324*b1*f**3*k1*k2**2*k3**2 + 36*b1*f**3*k2**3*k3**2*cos(3*theta) - 168*b1*f**2*k1**5 - 126*b1*f**2*k1**4*k2*cos(3*theta) - 42*b1*f**2*k1**3*k2**2*cos(4*theta) - 1596*b1*f**2*k1**3*k2**2 - 420*b1*f**2*k1**2*k2**3*cos(3*theta) - 84*b1*f**2*k1**2*k2*k3**2*cos(3*theta) - 21*b1*f**2*k1*k2**4*cos(4*theta) - 357*b1*f**2*k1*k2**4 - 42*b1*f**2*k1*k2**2*k3**2*cos(4*theta) + 546*b1*f**2*k1*k2**2*k3**2 + 168*b1*f**2*k1*k3**4 + 42*b1*f**2*k2**3*k3**2*cos(3*theta) + 42*b1*f**2*k2*k3**4*cos(3*theta) - 840*b1*f*g2*k1**5 - 2940*b1*f*g2*k1**3*k2**2 - 420*b1*f*g2*k1**2*k2**3*cos(3*theta) - 420*b1*f*g2*k1*k2**4 + 420*b1*f*g2*k1*k2**2*k3**2 + 840*b1*f*g2*k1*k3**4 - 252*b2*f**2*k1**3*k3**2 - 168*b2*f**2*k1*k2**2*k3**2 + 80*f**4*k1**3*k2**2 + 60*f**4*k1**2*k2**3*cos(3*theta) + 12*f**4*k1*k2**4*cos(4*theta) + 216*f**4*k1*k2**4 + 20*f**4*k2**5*cos(3*theta) - 108*f**3*k1**5 - 144*f**3*k1**4*k2*cos(3*theta) - 90*f**3*k1**3*k2**2*cos(4*theta) - 738*f**3*k1**3*k2**2 + 108*f**3*k1**3*k3**2 - 306*f**3*k1**2*k2**3*cos(3*theta) - 18*f**3*k1**2*k2**3*cos(5*theta) + 72*f**3*k1**2*k2*k3**2*cos(3*theta) - 36*f**3*k1*k2**4*cos(4*theta) - 144*f**3*k1*k2**4 + 18*f**3*k1*k2**2*k3**2*cos(4*theta) + 270*f**3*k1*k2**2*k3**2 + 36*f**3*k2**3*k3**2*cos(3*theta) - 504*f**2*g2*k1**5 - 1596*f**2*g2*k1**3*k2**2 + 504*f**2*g2*k1**3*k3**2 - 420*f**2*g2*k1**2*k2**3*cos(3*theta) - 42*f**2*g2*k1*k2**4*cos(4*theta) - 210*f**2*g2*k1*k2**4 + 336*f**2*g2*k1*k2**2*k3**2 + 6*k1*(105*b1**3*k2**2*k3**2*(K - 1) + 7*b1**2*k2**2*(f**2*(-6*k1**2 + 4*k3**2) + 5*f*(k1**2*(5*K - 3) + k2**2*(K - 1) + k3**2*(2*C + K - 1)) - 30*g2*k3**2) + b1*f*(12*f**2*k2**2*(-2*k1**2 + 2*k2**2 + 3*k3**2) + 7*f*(k1**4*(4*C - 2) + k1**2*k2**2*(30*C + 17*K - 33) + k2**4*(6*C + 3*K - 9) + k2**2*k3**2*(6*C - 2*K + 6) + k3**4*(2 - 4*C)) - 70*g2*k2**2*(5*k1**2 + k2**2 + k3**2)) + 2*f**2*(f**2*(5*k1**2*k2**2 + 16*k2**4) + 3*f*(k1**4*(4*C - 2) + k1**2*(k2**2*(29*C - 22) + k3**2*(2 - 4*C)) + k2**4*(5*C - 5) + k2**2*k3**2*(7 - 4*C)) - 7*k2**2*(b2*k3**2 + g2*(17*k1**2 + 3*k2**2 - 2*k3**2))))*cos(2*theta) + 6*k2*(70*b1**3*k3**2*(-k1**2*(-6*K + f + 3) + k3**2*(f + 3)) - 7*b1**2*(6*f**2*(k1**4 + 3*k1**2*k2**2 - k3**2*(2*k2**2 + k3**2)) - 5*f*(k1**4*(8*K - 2) + k1**2*(k2**2*(7*K - 5) + k3**2*(8*C - 6)) + 2*k3**2*(k2**2 + 2*k3**2)) + 120*g2*k1**2*k3**2) + b1*f*(-6*f**2*(5*k1**4 - 5*k1**2*k3**2 - 5*k2**4 - 9*k2**2*k3**2) + 7*f*(k1**4*(44*C + 24*K - 33) + k1**2*(k2**2*(42*C + 19*K - 50) - k3**2*(4*C + 12*K - 8)) + 11*k2**2*k3**2 + 5*k3**4) - 70*k1**2*(2*b2*k3**2 + 8*g2*k1**2 + 7*g2*k2**2)) + 2*f**2*(10*f**2*(3*k1**2*k2**2 + k2**4) + 3*f*(k1**4*(34*C - 21) + k1**2*(k2**2*(30*C - 26) + k3**2*(13 - 18*C)) + 4*k2**2*k3**2) - 7*k1**2*(6*b2*k3**2 + g2*(24*k1**2 + 19*k2**2 - 12*k3**2))))*cos(theta))/(1260*k1*k3**4)
        perm23 = -D1**4*Pk2*Pk3*(420*C*b1**2*f*k1**2*k2*k3**2 + 840*C*b1**2*f*k2**3*k3**2 - 840*C*b1**2*f*k2*k3**4 + 252*C*b1*f**2*k1**4*k2 + 252*C*b1*f**2*k1**3*k2**2*cos(3*theta) + 42*C*b1*f**2*k1**2*k2**3*cos(4*theta) + 1722*C*b1*f**2*k1**2*k2**3 + 42*C*b1*f**2*k1**2*k2*k3**2*cos(4*theta) - 294*C*b1*f**2*k1**2*k2*k3**2 + 168*C*b1*f**2*k1*k2**4*cos(3*theta) + 168*C*b1*f**2*k1*k2**2*k3**2*cos(3*theta) + 336*C*b1*f**2*k2**5 - 336*C*b1*f**2*k2*k3**4 + 36*C*f**3*k1**4*k2*cos(4*theta) + 144*C*f**3*k1**4*k2 + 342*C*f**3*k1**3*k2**2*cos(3*theta) + 18*C*f**3*k1**3*k2**2*cos(5*theta) + 108*C*f**3*k1**2*k2**3*cos(4*theta) + 1008*C*f**3*k1**2*k2**3 - 216*C*f**3*k1**2*k2*k3**2 + 216*C*f**3*k1*k2**4*cos(3*theta) - 72*C*f**3*k1*k2**2*k3**2*cos(3*theta) + 216*C*f**3*k2**5 - 216*C*f**3*k2**3*k3**2 + 630*K*b1**3*k1**2*k2*k3**2 + 1260*K*b1**3*k2**3*k3**2 - 1260*K*b1**3*k2*k3**4 + 210*K*b1**2*f*k1**4*k2 + 210*K*b1**2*f*k1**3*k2**2*cos(3*theta) + 1470*K*b1**2*f*k1**2*k2**3 - 210*K*b1**2*f*k1**2*k2*k3**2 + 420*K*b1**2*f*k2**5 - 420*K*b1**2*f*k2*k3**4 + 21*K*b1*f**2*k1**4*k2*cos(4*theta) + 105*K*b1*f**2*k1**4*k2 + 210*K*b1*f**2*k1**3*k2**2*cos(3*theta) + 798*K*b1*f**2*k1**2*k2**3 - 168*K*b1*f**2*k1**2*k2*k3**2 + 252*K*b1*f**2*k2**5 - 252*K*b1*f**2*k2**3*k3**2 - 420*b1**3*f*k1**2*k2*k3**2 - 630*b1**3*k1**2*k2*k3**2 - 1260*b1**2*b2*k2*k3**4 - 252*b1**2*f**2*k1**4*k2 - 504*b1**2*f**2*k1**2*k2**3 + 336*b1**2*f**2*k1**2*k2*k3**2 - 210*b1**2*f*k1**4*k2 - 210*b1**2*f*k1**3*k2**2*cos(3*theta) - 630*b1**2*f*k1**2*k2**3 - 210*b1**2*f*k1**2*k2*k3**2 - 420*b1**2*f*k2**3*k3**2 + 420*b1**2*f*k2*k3**4 - 1260*b1**2*g2*k1**2*k2*k3**2 - 2520*b1**2*g2*k2**3*k3**2 + 2520*b1**2*g2*k2*k3**4 - 420*b1*b2*f*k1**2*k2*k3**2 - 420*b1*b2*f*k2**3*k3**2 - 420*b1*b2*f*k2*k3**4 + 216*b1*f**3*k1**4*k2 + 36*b1*f**3*k1**3*k3**2*cos(3*theta) - 216*b1*f**3*k1**2*k2**3 + 324*b1*f**3*k1**2*k2*k3**2 - 21*b1*f**2*k1**4*k2*cos(4*theta) - 357*b1*f**2*k1**4*k2 - 420*b1*f**2*k1**3*k2**2*cos(3*theta) + 42*b1*f**2*k1**3*k3**2*cos(3*theta) - 42*b1*f**2*k1**2*k2**3*cos(4*theta) - 1596*b1*f**2*k1**2*k2**3 - 42*b1*f**2*k1**2*k2*k3**2*cos(4*theta) + 546*b1*f**2*k1**2*k2*k3**2 - 126*b1*f**2*k1*k2**4*cos(3*theta) - 84*b1*f**2*k1*k2**2*k3**2*cos(3*theta) + 42*b1*f**2*k1*k3**4*cos(3*theta) - 168*b1*f**2*k2**5 + 168*b1*f**2*k2*k3**4 - 420*b1*f*g2*k1**4*k2 - 420*b1*f*g2*k1**3*k2**2*cos(3*theta) - 2940*b1*f*g2*k1**2*k2**3 + 420*b1*f*g2*k1**2*k2*k3**2 - 840*b1*f*g2*k2**5 + 840*b1*f*g2*k2*k3**4 - 168*b2*f**2*k1**2*k2*k3**2 - 252*b2*f**2*k2**3*k3**2 + 20*f**4*k1**5*cos(3*theta) + 12*f**4*k1**4*k2*cos(4*theta) + 216*f**4*k1**4*k2 + 60*f**4*k1**3*k2**2*cos(3*theta) + 80*f**4*k1**2*k2**3 - 36*f**3*k1**4*k2*cos(4*theta) - 144*f**3*k1**4*k2 - 306*f**3*k1**3*k2**2*cos(3*theta) - 18*f**3*k1**3*k2**2*cos(5*theta) + 36*f**3*k1**3*k3**2*cos(3*theta) - 90*f**3*k1**2*k2**3*cos(4*theta) - 738*f**3*k1**2*k2**3 + 18*f**3*k1**2*k2*k3**2*cos(4*theta) + 270*f**3*k1**2*k2*k3**2 - 144*f**3*k1*k2**4*cos(3*theta) + 72*f**3*k1*k2**2*k3**2*cos(3*theta) - 108*f**3*k2**5 + 108*f**3*k2**3*k3**2 - 42*f**2*g2*k1**4*k2*cos(4*theta) - 210*f**2*g2*k1**4*k2 - 420*f**2*g2*k1**3*k2**2*cos(3*theta) - 1596*f**2*g2*k1**2*k2**3 + 336*f**2*g2*k1**2*k2*k3**2 - 504*f**2*g2*k2**5 + 504*f**2*g2*k2**3*k3**2 + 6*k1*(70*b1**3*k3**2*(-k2**2*(-6*K + f + 3) + k3**2*(f + 3)) - 7*b1**2*(6*f**2*(k1**2*(3*k2**2 - 2*k3**2) + k2**4 - k3**4) - 5*f*(k1**2*(k2**2*(7*K - 5) + 2*k3**2) + k2**4*(8*K - 2) + k2**2*k3**2*(8*C - 6) + 4*k3**4) + 120*g2*k2**2*k3**2) + b1*f*(6*f**2*(5*k1**4 + 9*k1**2*k3**2 - 5*k2**4 + 5*k2**2*k3**2) + 7*f*(k1**2*(k2**2*(42*C + 19*K - 50) + 11*k3**2) + k2**4*(44*C + 24*K - 33) - k2**2*k3**2*(4*C + 12*K - 8) + 5*k3**4) - 70*k2**2*(2*b2*k3**2 + 7*g2*k1**2 + 8*g2*k2**2)) + 2*f**2*(10*f**2*(k1**4 + 3*k1**2*k2**2) + 3*f*(k1**2*(k2**2*(30*C - 26) + 4*k3**2) + k2**4*(34*C - 21) + k2**2*k3**2*(13 - 18*C)) - 7*k2**2*(6*b2*k3**2 + g2*(19*k1**2 + 24*k2**2 - 12*k3**2))))*cos(theta) + 6*k2*(105*b1**3*k1**2*k3**2*(K - 1) + 7*b1**2*k1**2*(f**2*(-6*k2**2 + 4*k3**2) + 5*f*(k1**2*(K - 1) + k2**2*(5*K - 3) + k3**2*(2*C + K - 1)) - 30*g2*k3**2) + b1*f*(12*f**2*k1**2*(2*k1**2 - 2*k2**2 + 3*k3**2) + 7*f*(k1**4*(6*C + 3*K - 9) + k1**2*(k2**2*(30*C + 17*K - 33) + k3**2*(6*C - 2*K + 6)) + (4*C - 2)*(k2**4 - k3**4)) - 70*g2*k1**2*(k1**2 + 5*k2**2 + k3**2)) + 2*f**2*(f**2*(16*k1**4 + 5*k1**2*k2**2) + 3*f*(k1**4*(5*C - 5) + k1**2*(k2**2*(29*C - 22) + k3**2*(7 - 4*C)) + k2**2*(4*C - 2)*(k2**2 - k3**2)) - 7*k1**2*(b2*k3**2 + g2*(3*k1**2 + 17*k2**2 - 2*k3**2))))*cos(2*theta))/(1260*k2*k3**4)
        return np.sqrt((4*np.pi))*(perm12+perm13+perm23)
    
    def l2(cosmo_functions,k1,k2,k3=None,theta=None,zz=0,r=0,s=0,nonlin=False,growth2=False):
        
        #get generic cosmology parameters
        k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2 = cosmo_functions.get_params(k1,k2,k3,theta,zz,nonlin=nonlin,growth2=growth2)
        
        cos = lambda x: np.cos(x)
        sin = lambda x: np.sin(x)
        
        perm12 = D1**4*Pk1*Pk2*f*(3696*C*b1**2*k1**3*k2 - 3696*C*b1**2*k1**2*k2**2*cos(3*theta) - 1386*C*b1**2*k1*k2**3*cos(4*theta) - 462*C*b1**2*k1*k2**3 - 726*C*b1*f*k1**3*k2*cos(4*theta) + 4158*C*b1*f*k1**3*k2 - 4554*C*b1*f*k1**2*k2**2*cos(3*theta) - 594*C*b1*f*k1**2*k2**2*cos(5*theta) - 1914*C*b1*f*k1*k2**3*cos(4*theta) + 594*C*b1*f*k1*k2**3 - 594*C*f**2*k1**3*k2*cos(4*theta) + 858*C*f**2*k1**3*k2 - 1254*C*f**2*k1**2*k2**2*cos(3*theta) - 462*C*f**2*k1**2*k2**2*cos(5*theta) - 528*C*f**2*k1*k2**3*cos(4*theta) - 66*C*f**2*k1*k2**3*cos(6*theta) + 528*C*f**2*k1*k2**3 - 693*K*b1**2*k1*k2*k3**2*cos(4*theta) + 1617*K*b1**2*k1*k2*k3**2 - 363*K*b1*f*k1*k2*k3**2*cos(4*theta) + 495*K*b1*f*k1*k2*k3**2 + 4620*b1**3*k1*k2*k3**2 + 594*b1**2*f*k1**2*k3**2*cos(3*theta) + 10824*b1**2*f*k1*k2*k3**2 + 1188*b1**2*f*k2**2*k3**2*cos(3*theta) + 7392*b1**2*k1**3*k2 + 5082*b1**2*k1**2*k2**2*cos(3*theta) + 1386*b1**2*k1**2*k3**2*cos(3*theta) + 1386*b1**2*k1*k2**3*cos(4*theta) + 6006*b1**2*k1*k2**3 + 693*b1**2*k1*k2*k3**2*cos(4*theta) + 7623*b1**2*k1*k2*k3**2 + 1386*b1**2*k2**4*cos(3*theta) + 1386*b1**2*k2**2*k3**2*cos(3*theta) + 4620*b1*b2*k1*k2*k3**2 + 924*b1*f**2*k1**2*k3**2*cos(3*theta) + 396*b1*f**2*k1*k2*k3**2*cos(4*theta) + 7920*b1*f**2*k1*k2*k3**2 + 1584*b1*f**2*k2**2*k3**2*cos(3*theta) + 726*b1*f*k1**4*cos(3*theta) + 1320*b1*f*k1**3*k2*cos(4*theta) + 11352*b1*f*k1**3*k2 + 9570*b1*f*k1**2*k2**2*cos(3*theta) + 594*b1*f*k1**2*k2**2*cos(5*theta) + 726*b1*f*k1**2*k3**2*cos(3*theta) + 2508*b1*f*k1*k2**3*cos(4*theta) + 10164*b1*f*k1*k2**3 + 363*b1*f*k1*k2*k3**2*cos(4*theta) + 2937*b1*f*k1*k2*k3**2 + 1914*b1*f*k2**4*cos(3*theta) + 726*b1*f*k2**2*k3**2*cos(3*theta) + 1386*b1*g2*k1*k2*k3**2*cos(4*theta) - 3234*b1*g2*k1*k2*k3**2 + 1716*b2*f*k1*k2*k3**2 + 410*f**3*k1**2*k3**2*cos(3*theta) + 272*f**3*k1*k2*k3**2*cos(4*theta) + 2016*f**3*k1*k2*k3**2 + 530*f**3*k2**2*k3**2*cos(3*theta) + 30*f**3*k2**2*k3**2*cos(5*theta) + 594*f**2*k1**4*cos(3*theta) + 1056*f**2*k1**3*k2*cos(4*theta) + 4224*f**2*k1**3*k2 + 4422*f**2*k1**2*k2**2*cos(3*theta) + 528*f**2*k1**2*k2**2*cos(5*theta) + 1254*f**2*k1*k2**3*cos(4*theta) + 66*f**2*k1*k2**3*cos(6*theta) + 4026*f**2*k1*k2**3 + 726*f**2*k2**4*cos(3*theta) + 66*f**2*k2**4*cos(5*theta) + 726*f*g2*k1*k2*k3**2*cos(4*theta) - 990*f*g2*k1*k2*k3**2 + 2*k1*k2*(1386*b1**3*k3**2 - 66*b1**2*(k1**2*(28*C - 56) - k2**2*(14*C + 56) + k3**2*(7*K - 62*f - 49)) + 66*b1*(57*f**2*k3**2 + f*(k1**2*(96 - 26*C) + k2**2*(10*C + 96) - k3**2*(K - 23)) + k3**2*(21*b2 + 14*g2)) + f*(1096*f**2*k3**2 + f*k1**2*(2640 - 132*C) + f*k2**2*(33*C + 2607) + k3**2*(726*b2 + 132*g2)))*cos(2*theta) + (3696*b1**3*k3**2*(k1**2 + k2**2) + 66*b1**2*(56*k1**4 + k1**2*(k2**2*(56*C + 259) + k3**2*(135*f + 91)) + 7*k2**2*(5*k2**2 + k3**2*(18*f + 13))) + 66*b1*f*(85*k1**4 + k1**2*(k2**2*(78*C + 422) + k3**2*(106*f + 37)) + 67*k2**4 + k2**2*k3**2*(96*f + 37)) + 6*f**2*(280*f*k2**2*k3**2 + 341*k1**4 + k1**2*(305*f*k3**2 + k2**2*(286*C + 1815)) + 308*k2**4))*cos(theta))/(5544*k1*k2*k3**2)
        perm13 = -D1**4*Pk1*Pk3*f*(1848*C*b1**2*k1**3*k3**2 + 5544*C*b1**2*k1**2*k2*k3**2*cos(3*theta) + 1386*C*b1**2*k1*k2**2*k3**2*cos(4*theta) + 2310*C*b1**2*k1*k2**2*k3**2 - 1848*C*b1**2*k1*k3**4 + 3432*C*b1*f*k1**5 + 5280*C*b1*f*k1**4*k2*cos(3*theta) + 3102*C*b1*f*k1**3*k2**2*cos(4*theta) + 14322*C*b1*f*k1**3*k2**2 + 8514*C*b1*f*k1**2*k2**3*cos(3*theta) + 594*C*b1*f*k1**2*k2**3*cos(5*theta) + 528*C*b1*f*k1**2*k2*k3**2*cos(3*theta) + 1188*C*b1*f*k1*k2**4*cos(4*theta) + 1980*C*b1*f*k1*k2**4 + 726*C*b1*f*k1*k2**2*k3**2*cos(4*theta) + 858*C*b1*f*k1*k2**2*k3**2 - 3432*C*b1*f*k1*k3**4 + 2904*C*f**2*k1**5 + 4224*C*f**2*k1**4*k2*cos(3*theta) + 2706*C*f**2*k1**3*k2**2*cos(4*theta) + 13134*C*f**2*k1**3*k2**2 - 2904*C*f**2*k1**3*k3**2 + 6006*C*f**2*k1**2*k2**3*cos(3*theta) + 726*C*f**2*k1**2*k2**3*cos(5*theta) - 1848*C*f**2*k1**2*k2*k3**2*cos(3*theta) + 792*C*f**2*k1*k2**4*cos(4*theta) + 66*C*f**2*k1*k2**4*cos(6*theta) + 1848*C*f**2*k1*k2**4 - 264*C*f**2*k1*k2**2*k3**2*cos(4*theta) - 2376*C*f**2*k1*k2**2*k3**2 + 3696*K*b1**2*k1**5 + 10164*K*b1**2*k1**3*k2**2 + 4620*K*b1**2*k1**2*k2**3*cos(3*theta) + 693*K*b1**2*k1*k2**4*cos(4*theta) + 1155*K*b1**2*k1*k2**4 + 924*K*b1**2*k1*k2**2*k3**2 - 3696*K*b1**2*k1*k3**4 + 3168*K*b1*f*k1**5 + 9636*K*b1*f*k1**3*k2**2 - 3168*K*b1*f*k1**3*k3**2 + 3036*K*b1*f*k1**2*k2**3*cos(3*theta) + 363*K*b1*f*k1*k2**4*cos(4*theta) + 1221*K*b1*f*k1*k2**4 - 1716*K*b1*f*k1*k2**2*k3**2 - 924*b1**3*k1*k2**2*k3**2 - 5148*b1**2*f*k1**3*k2**2 - 1782*b1**2*f*k1**2*k2**3*cos(3*theta) - 792*b1**2*f*k1*k2**4 + 3432*b1**2*f*k1*k2**2*k3**2 + 1188*b1**2*f*k2**3*k3**2*cos(3*theta) - 5544*b1**2*k1**3*k2**2 - 924*b1**2*k1**3*k3**2 - 3234*b1**2*k1**2*k2**3*cos(3*theta) - 4158*b1**2*k1**2*k2*k3**2*cos(3*theta) - 693*b1**2*k1*k2**4*cos(4*theta) - 1155*b1**2*k1*k2**4 - 1386*b1**2*k1*k2**2*k3**2*cos(4*theta) - 462*b1**2*k1*k2**2*k3**2 + 924*b1**2*k1*k3**4 + 1386*b1**2*k2**3*k3**2*cos(3*theta) + 1386*b1**2*k2*k3**4*cos(3*theta) - 3696*b1*b2*k1**3*k3**2 - 924*b1*b2*k1*k2**2*k3**2 - 3696*b1*b2*k1*k3**4 - 2904*b1*f**2*k1**3*k2**2 + 264*b1*f**2*k1*k2**4*cos(4*theta) + 2376*b1*f**2*k1*k2**4 + 4356*b1*f**2*k1*k2**2*k3**2 + 660*b1*f**2*k2**5*cos(3*theta) + 924*b1*f**2*k2**3*k3**2*cos(3*theta) - 1716*b1*f*k1**5 - 3366*b1*f*k1**4*k2*cos(3*theta) - 2508*b1*f*k1**3*k2**2*cos(4*theta) - 15708*b1*f*k1**3*k2**2 - 9636*b1*f*k1**2*k2**3*cos(3*theta) - 594*b1*f*k1**2*k2**3*cos(5*theta) - 264*b1*f*k1**2*k2*k3**2*cos(3*theta) - 1551*b1*f*k1*k2**4*cos(4*theta) - 3201*b1*f*k1*k2**4 - 132*b1*f*k1*k2**2*k3**2*cos(4*theta) + 4092*b1*f*k1*k2**2*k3**2 + 1716*b1*f*k1*k3**4 + 1914*b1*f*k2**3*k3**2*cos(3*theta) + 726*b1*f*k2*k3**4*cos(3*theta) - 7392*b1*g2*k1**5 - 20328*b1*g2*k1**3*k2**2 - 9240*b1*g2*k1**2*k2**3*cos(3*theta) - 1386*b1*g2*k1*k2**4*cos(4*theta) - 2310*b1*g2*k1*k2**4 - 1848*b1*g2*k1*k2**2*k3**2 + 7392*b1*g2*k1*k3**4 - 3168*b2*f*k1**3*k3**2 - 1716*b2*f*k1*k2**2*k3**2 + 1220*f**3*k1**3*k2**2 + 1230*f**3*k1**2*k2**3*cos(3*theta) + 408*f**3*k1*k2**4*cos(4*theta) + 3024*f**3*k1*k2**4 + 530*f**3*k2**5*cos(3*theta) + 30*f**3*k2**5*cos(5*theta) - 1452*f**2*k1**5 - 2706*f**2*k1**4*k2*cos(3*theta) - 2112*f**2*k1**3*k2**2*cos(4*theta) - 9768*f**2*k1**3*k2**2 + 1452*f**2*k1**3*k3**2 - 5280*f**2*k1**2*k2**3*cos(3*theta) - 660*f**2*k1**2*k2**3*cos(5*theta) + 1518*f**2*k1**2*k2*k3**2*cos(3*theta) - 792*f**2*k1*k2**4*cos(4*theta) - 66*f**2*k1*k2**4*cos(6*theta) - 1848*f**2*k1*k2**4 + 594*f**2*k1*k2**2*k3**2*cos(4*theta) + 3366*f**2*k1*k2**2*k3**2 + 726*f**2*k2**3*k3**2*cos(3*theta) + 66*f**2*k2**3*k3**2*cos(5*theta) - 6336*f*g2*k1**5 - 19272*f*g2*k1**3*k2**2 + 6336*f*g2*k1**3*k3**2 - 6072*f*g2*k1**2*k2**3*cos(3*theta) - 726*f*g2*k1*k2**4*cos(4*theta) - 2442*f*g2*k1*k2**4 + 3432*f*g2*k1*k2**2*k3**2 - 6*k1*(462*b1**3*k2**2*k3**2 + 22*b1**2*(k1**2*(k2**2*(-91*K + 33*f + 42) + k3**2*(21 - 42*C)) + k2**4*(-14*K + 18*f + 14) + k2**2*k3**2*(-28*C + 7*K - 22*f + 14) + k3**4*(42*C - 21)) + 22*b1*(f**2*k2**2*(18*k1**2 - 20*k2**2 - 27*k3**2) + f*(k1**4*(11 - 22*C) + k1**2*k2**2*(-156*C - 71*K + 150) - k2**4*(24*C + 12*K - 36) + k2**2*k3**2*(12*C + 11*K - 42) + k3**4*(22*C - 11)) + 7*k2**2*(3*b2*k3**2 + g2*(26*k1**2 + 4*k2**2 - 2*k3**2))) + f*(-2*f**2*(85*k1**2*k2**2 + 274*k2**4) - 11*f*(k1**4*(36*C - 18) + 6*k1**2*(k2**2*(40*C - 30) + k3**2*(3 - 6*C)) + k2**4*(39*C - 39) + k2**2*k3**2*(60 - 40*C)) + 22*k2**2*(11*b2*k3**2 + 2*g2*(71*k1**2 + 12*k2**2 - 11*k3**2))))*cos(2*theta) + 6*k2*(616*b1**3*k3**2*(-k1**2 + k3**2) - 11*b1**2*(k1**4*(-224*K + 48*f + 56) + k1**2*(k2**2*(-154*K + 117*f + 119) + k3**2*(105 - 140*C)) - k3**2*(k2**2*(78*f + 35) + k3**2*(48*f + 91))) - 11*b1*(f**2*(40*k1**4 - 40*k1**2*k3**2 - 30*k2**4 - 66*k2**2*k3**2) - f*(k1**4*(304*C + 192*K - 237) + k1**2*(k2**2*(246*C + 146*K - 325) - k3**2*(8*C + 96*K - 52)) + 67*k2**2*k3**2 + 37*k3**4) + 28*k1**2*(4*b2*k3**2 + 16*g2*k1**2 + 11*g2*k2**2)) + f*(5*f**2*(183*k1**2*k2**2 + 56*k2**4) + 11*f*(k1**4*(256*C - 159) + k1**2*(k2**2*(218*C - 190) + k3**2*(97 - 132*C)) + 28*k2**2*k3**2) - 44*k1**2*(24*b2*k3**2 + g2*(96*k1**2 + 73*k2**2 - 48*k3**2))))*cos(theta))/(5544*k1*k3**4)
        perm23 = -D1**4*Pk2*Pk3*f*(3696*C*b1**2*k1**2*k2*k3**2 + 7392*C*b1**2*k2**3*k3**2 - 7392*C*b1**2*k2*k3**4 + 3168*C*b1*f*k1**4*k2 + 3168*C*b1*f*k1**3*k2**2*cos(3*theta) + 726*C*b1*f*k1**2*k2**3*cos(4*theta) + 21450*C*b1*f*k1**2*k2**3 + 726*C*b1*f*k1**2*k2*k3**2*cos(4*theta) - 3894*C*b1*f*k1**2*k2*k3**2 + 2904*C*b1*f*k1*k2**4*cos(3*theta) + 2904*C*b1*f*k1*k2**2*k3**2*cos(3*theta) + 3432*C*b1*f*k2**5 - 3432*C*b1*f*k2*k3**4 + 594*C*f**2*k1**4*k2*cos(4*theta) + 2046*C*f**2*k1**4*k2 + 5478*C*f**2*k1**3*k2**2*cos(3*theta) + 462*C*f**2*k1**3*k2**2*cos(5*theta) + 2640*C*f**2*k1**2*k2**3*cos(4*theta) + 66*C*f**2*k1**2*k2**3*cos(6*theta) + 13464*C*f**2*k1**2*k2**3 - 2904*C*f**2*k1**2*k2*k3**2 + 4752*C*f**2*k1*k2**4*cos(3*theta) + 264*C*f**2*k1*k2**4*cos(5*theta) - 1848*C*f**2*k1*k2**2*k3**2*cos(3*theta) + 264*C*f**2*k2**5*cos(4*theta) + 2376*C*f**2*k2**5 - 264*C*f**2*k2**3*k3**2*cos(4*theta) - 2376*C*f**2*k2**3*k3**2 + 1848*K*b1**2*k1**4*k2 + 1848*K*b1**2*k1**3*k2**2*cos(3*theta) + 693*K*b1**2*k1**2*k2**3*cos(4*theta) + 12243*K*b1**2*k1**2*k2**3 + 693*K*b1**2*k1**2*k2*k3**2*cos(4*theta) - 2541*K*b1**2*k1**2*k2*k3**2 + 2772*K*b1**2*k1*k2**4*cos(3*theta) + 2772*K*b1**2*k1*k2**2*k3**2*cos(3*theta) + 924*K*b1**2*k2**5 - 924*K*b1**2*k2*k3**4 + 363*K*b1*f*k1**4*k2*cos(4*theta) + 1221*K*b1*f*k1**4*k2 + 3333*K*b1*f*k1**3*k2**2*cos(3*theta) + 297*K*b1*f*k1**3*k2**2*cos(5*theta) + 1782*K*b1*f*k1**2*k2**3*cos(4*theta) + 7854*K*b1*f*k1**2*k2**3 - 1716*K*b1*f*k1**2*k2*k3**2 + 3564*K*b1*f*k1*k2**4*cos(3*theta) - 1188*K*b1*f*k1*k2**2*k3**2*cos(3*theta) + 792*K*b1*f*k2**5 - 792*K*b1*f*k2**3*k3**2 - 3696*b1**3*k1**2*k2*k3**2 - 3168*b1**2*f*k1**4*k2 - 5148*b1**2*f*k1**2*k2**3 + 3432*b1**2*f*k1**2*k2*k3**2 - 594*b1**2*f*k1*k2**4*cos(3*theta) + 594*b1**2*f*k1*k3**4*cos(3*theta) - 1848*b1**2*k1**4*k2 - 1848*b1**2*k1**3*k2**2*cos(3*theta) - 693*b1**2*k1**2*k2**3*cos(4*theta) - 4851*b1**2*k1**2*k2**3 - 693*b1**2*k1**2*k2*k3**2*cos(4*theta) - 1155*b1**2*k1**2*k2*k3**2 - 1386*b1**2*k1*k2**4*cos(3*theta) + 1386*b1**2*k1*k3**4*cos(3*theta) - 3696*b1**2*k2**3*k3**2 + 3696*b1**2*k2*k3**4 - 3696*b1*b2*k1**2*k2*k3**2 - 924*b1*b2*k2**3*k3**2 - 924*b1*b2*k2*k3**4 + 2904*b1*f**2*k1**4*k2 + 924*b1*f**2*k1**3*k3**2*cos(3*theta) - 264*b1*f**2*k1**2*k2**3*cos(4*theta) - 2376*b1*f**2*k1**2*k2**3 + 396*b1*f**2*k1**2*k2*k3**2*cos(4*theta) + 3564*b1*f**2*k1**2*k2*k3**2 - 660*b1*f**2*k1*k2**4*cos(3*theta) + 660*b1*f**2*k1*k2**2*k3**2*cos(3*theta) - 363*b1*f*k1**4*k2*cos(4*theta) - 4389*b1*f*k1**4*k2 - 5775*b1*f*k1**3*k2**2*cos(3*theta) - 297*b1*f*k1**3*k2**2*cos(5*theta) + 726*b1*f*k1**3*k3**2*cos(3*theta) - 1914*b1*f*k1**2*k2**3*cos(4*theta) - 18678*b1*f*k1**2*k2**3 - 132*b1*f*k1**2*k2*k3**2*cos(4*theta) + 6468*b1*f*k1**2*k2*k3**2 - 3366*b1*f*k1*k2**4*cos(3*theta) - 264*b1*f*k1*k2**2*k3**2*cos(3*theta) + 726*b1*f*k1*k3**4*cos(3*theta) - 1716*b1*f*k2**5 + 1716*b1*f*k2*k3**4 - 3696*b1*g2*k1**4*k2 - 3696*b1*g2*k1**3*k2**2*cos(3*theta) - 1386*b1*g2*k1**2*k2**3*cos(4*theta) - 24486*b1*g2*k1**2*k2**3 - 1386*b1*g2*k1**2*k2*k3**2*cos(4*theta) + 5082*b1*g2*k1**2*k2*k3**2 - 5544*b1*g2*k1*k2**4*cos(3*theta) - 5544*b1*g2*k1*k2**2*k3**2*cos(3*theta) - 1848*b1*g2*k2**5 + 1848*b1*g2*k2*k3**4 - 1716*b2*f*k1**2*k2*k3**2 - 1188*b2*f*k1*k2**2*k3**2*cos(3*theta) - 792*b2*f*k2**3*k3**2 + 410*f**3*k1**5*cos(3*theta) + 408*f**3*k1**4*k2*cos(4*theta) + 3024*f**3*k1**4*k2 + 1590*f**3*k1**3*k2**2*cos(3*theta) + 90*f**3*k1**3*k2**2*cos(5*theta) + 180*f**3*k1**2*k2**3*cos(4*theta) + 920*f**3*k1**2*k2**3 - 594*f**2*k1**4*k2*cos(4*theta) - 2046*f**2*k1**4*k2 - 4884*f**2*k1**3*k2**2*cos(3*theta) - 462*f**2*k1**3*k2**2*cos(5*theta) + 594*f**2*k1**3*k3**2*cos(3*theta) - 2178*f**2*k1**2*k2**3*cos(4*theta) - 66*f**2*k1**2*k2**3*cos(6*theta) - 9834*f**2*k1**2*k2**3 + 462*f**2*k1**2*k2*k3**2*cos(4*theta) + 3630*f**2*k1**2*k2*k3**2 - 3102*f**2*k1*k2**4*cos(3*theta) - 198*f**2*k1*k2**4*cos(5*theta) + 1650*f**2*k1*k2**2*k3**2*cos(3*theta) + 66*f**2*k1*k2**2*k3**2*cos(5*theta) - 132*f**2*k2**5*cos(4*theta) - 1188*f**2*k2**5 + 132*f**2*k2**3*k3**2*cos(4*theta) + 1188*f**2*k2**3*k3**2 - 726*f*g2*k1**4*k2*cos(4*theta) - 2442*f*g2*k1**4*k2 - 6666*f*g2*k1**3*k2**2*cos(3*theta) - 594*f*g2*k1**3*k2**2*cos(5*theta) - 3564*f*g2*k1**2*k2**3*cos(4*theta) - 15708*f*g2*k1**2*k2**3 + 3432*f*g2*k1**2*k2*k3**2 - 7128*f*g2*k1*k2**4*cos(3*theta) + 2376*f*g2*k1*k2**2*k3**2*cos(3*theta) - 1584*f*g2*k2**5 + 1584*f*g2*k2**3*k3**2 + 6*k1*(616*b1**3*k3**2*(-k2**2 + k3**2) - 11*b1**2*(4*k1**2*(k2**2*(-49*K + 36*f + 35) - k3**2*(24*f + 14)) + k2**4*(-182*K + 39*f + 35) + k2**2*k3**2*(-224*C + 42*K + 168) - k3**4*(39*f + 91)) + 11*b1*(f**2*(40*k1**4 + 66*k1**2*k3**2 + 30*k2**2*(-k2**2 + k3**2)) + f*(k1**2*(k2**2*(336*C + 137*K - 388) + 85*k3**2) + k2**4*(340*C + 138*K - 237) - k2**2*k3**2*(44*C + 78*K - 52) + 37*k3**4) - 28*k2**2*(4*b2*k3**2 + g2*(14*k1**2 + 13*k2**2 - 3*k3**2))) + f*(5*f**2*(61*k1**4 + 168*k1**2*k2**2) + 11*f*(k1**2*(k2**2*(230*C - 199) + 31*k3**2) + 2*k2**2*(k2**2*(122*C - 75) + k3**2*(47 - 66*C))) - 22*k2**2*(39*b2*k3**2 + g2*(137*k1**2 + 138*k2**2 - 78*k3**2))))*cos(theta) + 6*k2*(22*b1**2*(21*K*(k2**4 - k3**4) + k1**4*(14*K - 14) + k1**2*(k2**2*(70*K - 33*f - 42) + k3**2*(28*C + 14*K + 22*f - 14))) + 22*b1*(-21*b2*k3**2*(k2**2 + k3**2) + 2*f**2*k1**2*(9*k1**2 - 10*k2**2 + 15*k3**2) + f*(k1**4*(24*C + 12*K - 36) + k1**2*(k2**2*(120*C + 71*K - 132) + k3**2*(24*C - 11*K + 24)) + (k2**2 - k3**2)*(k2**2*(22*C + 18*K - 11) + k3**2*(22*C - 11))) - 7*g2*(4*k1**4 + 4*k1**2*(5*k2**2 + k3**2) + 6*k2**4 - 6*k3**4)) + f*(-22*b2*k3**2*(11*k1**2 + 18*k2**2) + 2*f**2*(274*k1**4 + 95*k1**2*k2**2) + 11*f*(k1**4*(40*C - 40) + k1**2*(k2**2*(235*C - 177) + k3**2*(58 - 36*C)) + k2**2*(40*C - 20)*(k2**2 - k3**2)) - 44*g2*(12*k1**4 + k1**2*(71*k2**2 - 11*k3**2) + 18*k2**4 - 18*k2**2*k3**2)))*cos(2*theta))/(5544*k2*k3**4)
        return np.sqrt((4*np.pi)/5)*(perm12+perm13+perm23)