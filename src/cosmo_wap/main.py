import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import odeint

from cosmo_wap.peak_background_bias import *
from cosmo_wap.survey_params import *

#from useful_funcs import *            

class CosmoWAP:
    """
           Main functions - takes in cosmology from CLASS and bias models and then can called to generate cosmology (f,P(k),P'(k),D(z) etc) and all other biases including relativstic parts
    
    """
    def __init__(self,cosmo,survey_params,compute_bias=True):
        """
           Inputs CLASS and bias dict to return all bias and cosmological parameters defined within the class object

        """
        self.cosmo = cosmo
        baLCDM = cosmo.get_background()
        
        D_cl = baLCDM['gr.fac. D']
        z_cl = baLCDM['z']
        self.D_intp = interp1d(z_cl,D_cl,kind='cubic')
        
        f_cl = baLCDM['gr.fac. f']
        D_cl = baLCDM['gr.fac. D']
        z_cl = baLCDM['z']
        H_cl = baLCDM['H [1/Mpc]']
        xi_cl = baLCDM['comov. dist.']
        t_cl = baLCDM['conf. time [Mpc]']
        
        self.z_cl = z_cl#save for later
        self.Om_0 = cosmo.get_current_derived_parameters(['Omega_m'])['Omega_m']
        self.h = cosmo.get_current_derived_parameters(['h'])['h']
        
        #define functions and add in h
        self.H_c = interp1d(z_cl,H_cl*(1/(1+z_cl))/self.h,kind='cubic') # now in h/Mpc!#is conformal
        self.dH_c = interp1d(z_cl,np.gradient(H_cl*(1/(1+z_cl))/self.h,z_cl),kind='cubic') # derivative wrt z
        self.ddH_c = interp1d(z_cl,np.gradient(self.dH_c(z_cl),z_cl),kind='cubic') # second derivative wrt z
        self.comoving_dist = interp1d(z_cl,xi_cl*self.h,kind='cubic') # just use class background as quick
        self.d_to_z = interp1d(xi_cl*self.h,z_cl,kind='cubic') # useful to map other way
        self.f_intp = interp1d(z_cl,f_cl,kind='cubic')#get f #omega_mz = Omega_m *(1+zt)**3 /(Omega_m *(1+zt)**3 + Omega_l)
        self.D_intp = interp1d(z_cl,D_cl,kind='cubic')
        self.dD_dz = interp1d(z_cl,np.gradient(D_cl,z_cl),kind='cubic')
        self.conf_time = interp1d(z_cl,self.h*t_cl,kind='cubic')#convert between the two
        #misc
        self.c = 2.99792e+5 #km/s
        self.H0 = 100/self.c #[h/Mpc]
        self.Om = lambda xx: self.Om_0 * (self.H0**2 / self.H_c(xx)**2) * (1+xx)
        
        #for critical density
        GG = 4.300917e-3 #[pc SolarMass (km/s)^2]
        self.G = GG/(1e+6*self.c**2)#gravitational constant
        self.rho_crit = lambda xx: 3*self.H_c(xx)**2/(8*np.pi*self.G)  #in units of h^3 Mo/ Mpc^3 where Mo is solar mass
        self.rho_m = lambda xx: self.rho_crit(xx)*self.Om(xx)          #in units of h^3 Mo/ Mpc^3
        
        #######################################################################################################
       
        #setup surveys and compute all bias params including for multi tracer case...        
        self = self.setup_survey(survey_params, compute_bias)

        #get ranges of cross survey
        z_min = max([self.survey.z_range[0],self.survey1.z_range[0]])
        z_max = min([self.survey.z_range[1],self.survey1.z_range[1]])
        if z_min > z_max:
            raise ValueError("incompatible survey redshifts.")
        self.z_survey = np.linspace(z_min,z_max,int(1e+5))
        self.f_sky = min([self.survey.f_sky,self.survey1.f_sky])

        ####################################################################################################
        #get power spectras
        def get_class_powerspectrum(kk,zz=0): #h are needed to convert to 1/Mpc for k then convert pk back to (Mpc/h)^3
            return np.array([self.cosmo.pk_lin(ki, zz) for ki in kk*self.h])*self.h**3
    
        def get_Pk_NL(k,z=0): # for halofit non-linear power spectrum
            pk_nl = interp1d(k,np.array([self.cosmo.pk(ki, z) for ki in k*self.h])*self.h**3)
            return pk_nl

        def get_pkinfo_z(k,z):
            """get Pk and its k derivatives"""
            Plin = get_class_powerspectrum(k,0)#just always get present day power spectrum
            Pk = interp1d(k,Plin)#get linear power spectrum
            Pk_d = interp1d(k,np.gradient(Plin,k))
            Pk_dd = interp1d(k,np.gradient(Pk_d(k),k))
            return Pk,Pk_d,Pk_dd
        
        K_MAX = 100
        k = np.logspace(-5, np.log10(K_MAX), num=1000)
        self.Pk,self.Pk_d,self.Pk_dd = get_pkinfo_z(k,0)
        self.Pk_L = get_class_powerspectrum
        self.Pk_NL = get_Pk_NL(k,0) #if you want HALOFIT P(k)
        
    #############################################################################################################
    #to do move functions defined in init function to here...
    
    
    
    ################################################################################################################
            
    #getter functions and their requisites    
    ###########################################################
    def _process_survey(self, survey_params, compute_bias):
        """
        Get bias funcs for a given survey - compute biases from HMF and HOD relations if flagged
        """
        class_bias = SurveyFunctions(survey_params, compute_bias)
            
        if compute_bias:
            print("Computing bias params...")
            PBs  = PBBias(self,survey_params)
            PBs.add_bias_attr(class_bias)

        return class_bias

    def setup_survey(self, survey_params, compute_bias):
        """
        Get bias functions for given surveys allowing for two surveys in list or not list format
        """
        
        # If survey_params is a list 
        if type(survey_params)==list:
            self.survey = self._process_survey(survey_params[0], compute_bias)
            #check for additional surveys
            if len(survey_params) > 1:
                #Process second survey
                self.survey1 = self._process_survey(survey_params[1], compute_bias)
            else:
                self.survey1 = self.survey
        else:
            # Process survey
            self.survey = self._process_survey(survey_params, compute_bias)
            self.survey1 = self.survey
            
        return self
                
    #######################################################################################################
    
    def Pk_phi(self,k, k0=0.05, units=True):
        """Power spectrum of the Bardeen potential Phi in the matter-dominated era - k in units of h/Mpc.
        """
        k_pivot = k0/self.h
        resp = (9.0/25.0) * self.cosmo.get_current_derived_parameters(['A_s'])['A_s'] * (k/k_pivot)**(self.cosmo.get_current_derived_parameters(['n_s'])['n_s'] - 1.0)

        if units:
            resp *= 2*np.pi**2.0/k**3.0    #[Mpc/h]^3

        return resp

    def M(self,k, z):
        """The scaling factor between the primordial scalar power spectrum and the late-time matter power spectrum
        """
        return np.sqrt(self.D_intp(z)**2 *self.Pk(k) / self.Pk_phi(k))

    def solve_second_order_KC(self):
        c = 2.99792*10**5 #km/s
        def F_func(u,zz): # so variables are F and H and D
            f,fd = u # unpack u vector
            D_zz = self.D_intp(zz)
            return [fd,(-self.H_c(zz)*self.dH_c(zz)*(1+zz)**2 *fd + ((3*(self.H0)**2 * self.Om_0 *(1+zz))/(2))*(f+D_zz**2))/(self.H_c(zz)**2 *(1+zz)**2)]

        odeint_zz = np.linspace(20,0.05,int(1e+5))# so z=20 should be pretty much matter dominated

        #set initial params for F
        F0 = [(3/7)*self.D_intp(odeint_zz[0])**2,(3/7)*2*self.D_intp(odeint_zz[0])*self.dD_dz(odeint_zz[0])]
        sol1 = odeint(F_func,F0,odeint_zz)
        K = (sol1[:,0]/self.D_intp(odeint_zz)**2)
        C = sol1[:,1]/(2*self.D_intp(odeint_zz)*self.dD_dz(odeint_zz))
        self.K_intp = interp1d(odeint_zz,K,kind='cubic')
        self.C_intp = interp1d(odeint_zz,C,kind='cubic')

    def lnd_derivatives(self,functions_to_differentiate):
        """
            calculates derivatives of a list of functions wrt log comoving dist numerically
        """

        # Store first derivatives in a list
        function_derivatives = []

        for func in functions_to_differentiate:
            # Calculate numerical derivatives of the function with respect to ln(d)
            derivative_func = interp1d(self.z_survey, np.gradient(func(self.z_survey), np.log(self.comoving_dist(self.z_survey))))
            function_derivatives.append(derivative_func)

        return function_derivatives

    def get_params(self,k1,k2,k3=None,theta=None,zz=0,tracer = None,nonlin=False,growth2=False):
        """
            return arrays of redshift and k dependent parameters
        """
        if theta is None:
            if k3 is None:
                raise  ValueError('Define either theta or k3')
            else:
                theta = np.arccos((k3**2 - k1**2 - k2**2)/(2*k1*k2))
                
        if tracer is None:
            tracer = self.survey
        
        k3 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(theta))#get k3 from triangle condition
        k3 = np.where(k3==0,1e-4,k3)
        
        Pk1 = self.Pk(k1)
        Pk2 = self.Pk(k2)
        Pk3 = self.Pk(k3)

        Pkd1 = self.Pk_d(k1)
        Pkd2 = self.Pk_d(k2)
        Pkd3 = self.Pk_d(k3)

        Pkdd1= self.Pk_dd(k1)
        Pkdd2 = self.Pk_dd(k2)
        Pkdd3 = self.Pk_dd(k3)
        
        if nonlin:
            Pk1 = Pk_NL(k1)
            Pk2 = Pk_NL(k2)
            Pk3 = Pk_NL(k3)
        
        #redshift dependendent terms
        d = self.comoving_dist(zz)
        
        K = 3/7 # from einstein-de-sitter
        C = 3/7
        if growth2:
            self.solve_second_order_KC()#get K and C
            K = self.K_intp(zz)
            C = self.C_intp(zz)

        f = self.f_intp(zz)
        D1 = self.D_intp(zz)
        
        #survey stuff
        b1 = tracer.b_1(zz)
        b2 = tracer.b_2(zz)
        g2 = tracer.g_2(zz)
        return k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2
    
    def get_params_pk(self,k1,zz,nonlinear=False):
        """
           return arrays of redshift and k dependent parameters but just for single knitting vector
        """
        
        Pk1 = self.Pk(k1)
        Pkd1 = self.Pk_d(k1)
        Pkdd1= self.Pk_dd(k1)
        
        if nonlinear:
            Pk1 = Pk_NL(k1)
            
        #redshift dependendent terms
        d = self.comoving_dist(zz)


        f = self.f_intp(zz)
        D1 = self.D_intp(zz)
        
        return k1,Pk1,Pkd1,Pkdd1,d,f,D1

    def get_derivs(self,zz,tracer = None):
        """
        Derivatives of redshift dependent parameters for radial wide separation terms
        """
        if tracer is None:
            tracer = self.survey
            
        #get derivs of these redshift dependent functions
        func_deriv_list = self.lnd_derivatives([tracer.b_1,tracer.b_2,tracer.g_2,self.f_intp,self.D_intp])
        tracer.b1_d,tracer.b2_d,tracer.g2_d,self.f_d,self.D_d = func_deriv_list
        func_second_deriv_list = self.lnd_derivatives(func_deriv_list)
        tracer.b1_dd,tracer.b2_dd,tracer.g2_dd,self.f_dd,self.D_dd = func_second_deriv_list
        
        #1st deriv
        fd = self.f_d(zz)
        Dd = self.D_d(zz)
        gd2 = tracer.g2_d(zz)
        bd2 = tracer.b2_d(zz)
        bd1 = tracer.b1_d(zz)
        #2nd deriv
        fdd = self.f_dd(zz)
        Ddd = self.D_dd(zz)
        gdd2 = tracer.g2_dd(zz)
        bdd2 = tracer.b2_dd(zz)
        bdd1 = tracer.b1_dd(zz)
        return fd,Dd,gd2,bd2,bd1,fdd,Ddd,gdd2,bdd2,bdd1
    
    def get_PNGparams(self,fNL,zz,k1,k2,k3,tracer = None):
        """
        returns terms needed for local type fnl including scale dependent bias
        """
        if tracer is None:
            tracer = self.survey
            
        delta_c = 1.686
        bL10 = lambda xx: tracer.b_1(xx) - 1
        bL20 = lambda xx: tracer.b_2(xx) - (8/21)*bL10(xx)

        bL01 = lambda xx: 2*delta_c*fNL*bL10(xx)
        bL11 = lambda xx: 2*fNL*(delta_c*bL20(xx)-bL10(xx))
        bL02 = lambda xx: 4*fNL**2 * delta_c*(delta_c*bL20(xx)-2*bL10(xx))

        bE01 = bL01
        bE11 = lambda xx: bL01(xx) + bL11(xx)
        bE02 = bL02

        alphak1 = self.M(k1, 0)
        alphak2 = self.M(k2, 0)
        alphak3 = self.M(k3, 0)
        return fNL,1*bE01(zz),1*bE11(zz),1*bE02(zz),alphak1,alphak2,alphak3
    
    def get_beta_funcs(self,zz,tracer = None):
        """
            Function that relies on biases and functions defined above to return beta coefficients (beta expressions adapted from Eline de Weerd GitHub) from paper 1711.01812v4
        """
        if tracer is None:
            tracer = self.survey
            
        #these derivs here are wrt to conformal time so we convert to derivs wrt to z
        #d/dt = da/dt d/da = a H dz/da d/dz =  -(1+z) H d/dz # everything here is conformal both t and H
        #d^2/d^2 t = (1+z)^2 H^2 d^2/d z^2 + H(1+z)(H+(1+z)H')d/dz
        
        dQ_dz = interp1d(self.z_survey,np.gradient(tracer.Q_survey(self.z_survey),self.z_survey))
        dbe_dz = interp1d(self.z_survey,np.gradient(tracer.be_survey(self.z_survey),self.z_survey))
        db1_dz = interp1d(self.z_survey,np.gradient(tracer.b_1(self.z_survey),self.z_survey))
        
        dH_dt = lambda xx: -(1+xx)*self.H_c(xx)*self.dH_c(xx)
        dH_dt2 = lambda xx: (1+xx)**2 *self.H_c(xx)**2 *self.ddH_c(xx)+self.H_c(xx)*(1+xx)*(self.H_c(xx)+(1+xx)*self.dH_c(xx))*self.dH_c(xx)#should just check with doing it numerically as is easier
        #self.dH22 = interp1d(self.z_cl,np.gradient(dH_dt(self.z_cl),self.conf_time(self.z_cl)))
        self.dH_dt = dH_dt                   
        dQ_dt = lambda xx: -(1+xx)*self.H_c(xx)*dQ_dz(xx)
        dbe_dt = lambda xx: -(1+xx)*self.H_c(xx)*dbe_dz(xx)#0*xx-1.6*1e-4
        db1_dt = lambda xx: -(1+xx)*self.H_c(xx)*db1_dz(xx)  
        
        self.Q = tracer.Q_survey  # bit silly but whatever
        self.b_e = tracer.be_survey

        partdQ=0
        partdb1=0
        
        #from 1st order petrubation theory
        tracer.gr1 = lambda xx: self.H_c(xx)*self.f_intp(xx)*(self.b_e(xx)-2*self.Q(xx)-2*(1-self.Q(xx))/(self.comoving_dist(xx)*self.H_c(xx))-dH_dt(xx)/self.H_c(xx)**2)
        tracer.gr2 = lambda xx: self.H_c(xx)**2 *(self.f_intp(xx)*(3-self.b_e(xx))+ (3/2)*self.Om(xx)*(2+self.b_e(xx)-self.f_intp(xx)-4*self.Q(xx)-2*self.Q(xx)-2*(1-self.Q(xx))/(self.comoving_dist(xx)*self.H_c(xx))-dH_dt(xx)/self.H_c(xx)**2))
        tracer.grd1 = self.lnd_derivatives([tracer.gr1])[0]#get derivative for 1st order coef
        
        beta = np.empty(20,dtype=object)
        beta[6] = lambda xx: self.H_c(xx)**2 * ((3 / 2) * self.Om(xx) * (2 - 2 * self.f_intp(xx)+ self.b_e(xx) - 4 * self.Q(xx) - ((2 * (1 - self.Q(xx))) / (self.comoving_dist(xx) * self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2)))
        beta[7] = lambda xx: self.H_c(xx)**2 * (self.f_intp(xx)* (3 - self.b_e(xx)))
        beta[8] = lambda xx: self.H_c(xx)**2 * (3 * self.Om(xx) * self.f_intp(xx) * (2 - self.f_intp(xx) - 2 * self.Q(xx)) + self.f_intp(xx)**2 * (4 + self.b_e(xx) - self.b_e(xx)**2 + 4 * self.b_e(xx) * self.Q(xx) - 6 * self.Q(xx) - 4 * self.Q(xx)**2 + 4 * partdQ + 4 * (dQ_dt(xx) / self.H_c(xx)) -(dbe_dt(xx) / self.H_c(xx)) - (2 / (self.comoving_dist(xx)**2 * self.H_c(xx)**2)) * (1 - self.Q(xx) + 2 * self.Q(xx)**2 - 2 * partdQ) - (2 / (self.comoving_dist(xx) * self.H_c(xx))) * (3 - 2 * self.b_e(xx) + 2 * self.b_e(xx) * self.Q(xx) - self.Q(xx) - 4 * self.Q(xx)**2 + ((3 * dH_dt(xx)) / (self.H_c(xx)**2)) * (1 - self.Q(xx)) + 4 * partdQ + 2 * (dQ_dt(xx) / self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2) * (3 - 2 * self.b_e(xx) + 4 * self.Q(xx) + ((3 * dH_dt(xx)) / (self.H_c(xx)**2))) + (dH_dt2(xx) / self.H_c(xx)**3)))
        beta[9] = lambda xx: self.H_c(xx)**2 * ( -(9 / 2) * self.Om(xx) * self.f_intp(xx))
        beta[10] = lambda xx: self.H_c(xx)**2 * (3 * self.Om(xx) * self.f_intp(xx))
        beta[11] = lambda xx: self.H_c(xx)**2 * ( (3/2 ) * self.Om(xx) * (1 + 2 * self.f_intp(xx)/ (3 * self.Om(xx))) + 3 * self.Om(xx) * self.f_intp(xx)- self.f_intp(xx)**2 * (-1 + self.b_e(xx) - 2 * self.Q(xx) - ((2 * (1 + self.Q(xx))) / (self.comoving_dist(xx) * self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2)))
        beta[12] = lambda xx: self.H_c(xx)**2 * ( -3 * self.Om(xx) * (1 + 2 * self.f_intp(xx)/ (3 * self.Om(xx))) - self.f_intp(xx)* ( tracer.b_1(xx) * (self.f_intp(xx)- 3 + self.b_e(xx)) + (db1_dt(xx) / self.H_c(xx)) ) + (3 / 2) * self.Om(xx) * (tracer.b_1(xx) * (2 + self.b_e(xx) - 4 * self.Q(xx) - 2 * ((1 - self.Q(xx))/(self.comoving_dist(xx) * self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2) ) + db1_dt(xx) /self.H_c(xx) + 2 * (2 - (1 / (self.comoving_dist(xx) * self.H_c(xx))) ) * partdb1 ) )    
        beta[13] = lambda xx: self.H_c(xx)**2 * (( (9 / 4) * self.Om(xx)**2 + (3 / 2) * self.Om(xx) * self.f_intp(xx)* (1 - (2 * self.f_intp(xx)) + 2 * self.b_e(xx) - 6 * self.Q(xx) - ((4 * (1 - self.Q(xx)))/(self.comoving_dist(xx) * self.H_c(xx))) - ((3 * dH_dt(xx)) / self.H_c(xx)**2) ) ) + ( self.f_intp(xx)**2 * (3 - self.b_e(xx)) ) )
        beta[14] = lambda xx: self.H_c(xx) * ( - (3 / 2) * self.Om(xx) * tracer.b_1(xx))
        beta[15] = lambda xx: self.H_c(xx) * 2 * self.f_intp(xx)**2
        beta[16] = lambda xx: self.H_c(xx) * (self.f_intp(xx)* (tracer.b_1(xx) * (self.f_intp(xx)+ self.b_e(xx) - 2 * self.Q(xx) - ((2 * (1 - self.Q(xx))) / (self.comoving_dist(xx) * self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2)) + (db1_dt(xx) / self.H_c(xx)) + 2 * (1 - (1 / (self.comoving_dist(xx) * self.H_c(xx)))) * partdb1 ))
        beta[17] = lambda xx: self.H_c(xx) * (- (3 / 2) * self.Om(xx) * self.f_intp(xx))
        beta[18] = lambda xx: self.H_c(xx) * ( (3 / 2) * self.Om(xx) * self.f_intp(xx) - self.f_intp(xx)**2 * (3 - 2 * self.b_e(xx) + 4 * self.Q(xx) + ((4 * (1 - self.Q(xx))) / (self.comoving_dist(xx) * self.H_c(xx))) + (3 * dH_dt(xx)/ self.H_c(xx)**2)) )
        beta[19] = lambda xx: self.H_c(xx) * (self.f_intp(xx)* (self.b_e(xx) - 2 * self.Q(xx) - ((2 * (1 - self.Q(xx))) / (self.comoving_dist(xx) * self.H_c(xx))) - (dH_dt(xx)/ self.H_c(xx)**2)))

        
        #get betad - derivatives wrt to ln(d)  
        betad = np.empty(20,dtype=object)
        betad[14:20] = np.array(self.lnd_derivatives(beta[14:20]),dtype=object)#14-19
        
        tracer.beta = beta 
        tracer.betad = betad
        all_betas = np.concatenate((np.array([tracer.gr1,tracer.gr2,tracer.grd1]),beta[6:],betad[14:20]))
        return [all_betas[i](zz) for i in range(len(all_betas))]
    
