class COV:    
    def l0(cosmo_functions,k1,zz=0,t=0,sigma=0):
        k1,Pk,Pkd,Pkdd,d,f,D1 = cosmo_functions.get_params_pk(k1,zz)
        
        b1 = cosmo_functions.survey.b_1(zz)
        xb1 = cosmo_functions.survey1.b_1(zz)
        
        expr = D1**2*Pk*(5*b1*(f + 3*xb1) + f*(3*f + 5*xb1))/15
        
        Erf = scipy.special.erf
        if sigma != 0:
            expr = D1**2*Pk*(-2*f*sigma*(f*(sigma**2 + 3) + sigma**2*(b1 + xb1))*np.exp(-sigma**2/2) + np.sqrt(2)*np.sqrt(np.pi)*(b1*sigma**4*xb1 + 3*f**2 + f*sigma**2*(b1 + xb1))*Erf(np.sqrt(2)*sigma/2))/(2*sigma**5)
        
        return expr