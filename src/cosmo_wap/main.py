import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

from cosmo_wap.peak_background_bias import *
from cosmo_wap.survey_params import *
from cosmo_wap.lib.utils import *
from cosmo_wap.lib import betas

class ClassWAP:
    """
           Main class - takes in cosmology from CLASS and bias models and then can called to generate cosmology (f,P(k),P'(k),D(z) etc) and all other biases including relativstic parts
    
    """
    def __init__(self,cosmo,survey_params,compute_bias=False,HMF='Tinker2010',nonlin=False,growth2=False):
        """
           Inputs CLASS and bias dict to return all bias and cosmological parameters defined within the class object

        """
        self.nonlin  = nonlin  #use nonlin halofit powerspectra
        self.growth2 = growth2 #second order growth corrections to F2 and G2 kernels
        
        # get background parameters
        self.cosmo = cosmo
        baLCDM = cosmo.get_background()
        
        z_cl  = baLCDM['z'][::-1]        # CubicSpline only allows increasing funcs
        f_cl  = baLCDM['gr.fac. f'][::-1]
        D_cl  = baLCDM['gr.fac. D'][::-1]
        z_cl  = baLCDM['z'][::-1]
        H_cl  = baLCDM['H [1/Mpc]'][::-1]
        xi_cl = baLCDM['comov. dist.'][::-1]
        t_cl  = baLCDM['conf. time [Mpc]'][::-1]
        
        self.z_cl = z_cl#save for later
        self.Om_0 = cosmo.get_current_derived_parameters(['Omega_m'])['Omega_m']
        self.h = cosmo.get_current_derived_parameters(['h'])['h']
        self.A_s = cosmo.get_current_derived_parameters(['A_s'])['A_s']
        self.n_s = cosmo.get_current_derived_parameters(['n_s'])['n_s']
        
        #define functions and add in h
        self.H_c           = CubicSpline(z_cl,H_cl*(1/(1+z_cl))/self.h) # now in h/Mpc!#is conformal
        self.dH_c          = CubicSpline(z_cl,np.gradient(H_cl*(1/(1+z_cl))/self.h,z_cl)) # derivative wrt z
        self.ddH_c         = CubicSpline(z_cl,np.gradient(self.dH_c(z_cl),z_cl)) # second derivative wrt z
        self.comoving_dist = CubicSpline(z_cl,xi_cl*self.h) # just use class background as quick
        self.d_to_z        = CubicSpline(xi_cl*self.h,z_cl) # useful to map other way
        self.f_intp        = CubicSpline(z_cl,f_cl)#get f #omega_mz = Omega_m *(1+zt)**3 /(Omega_m *(1+zt)**3 + Omega_l)
        self.D_intp        = CubicSpline(z_cl,D_cl)
        self.dD_dz         = CubicSpline(z_cl,np.gradient(D_cl,z_cl))
        self.conf_time     = CubicSpline(z_cl,self.h*t_cl)#convert between the two
        #misc
        self.c = 2.99792e+5 #km/s
        self.H0 = 100/self.c #[h/Mpc]
        self.Om = lambda xx: self.Om_0 * (self.H0**2 / self.H_c(xx)**2) * (1+xx)
        
        #for critical density
        GG = 4.300917e-3 #[pc SolarMass (km/s)^2]
        self.G = GG/(1e+6*self.c**2)#gravitational constant
        self.rho_crit = lambda xx: 3*self.H_c(xx)**2/(8*np.pi*self.G)  #in units of h^3 Mo/ Mpc^3 where Mo is solar mass
        self.rho_m = lambda xx: self.rho_crit(xx)*self.Om(xx)          #in units of h^3 Mo/ Mpc^3
        
        ##################################################################################
        #get powerspectra
        K_MAX = 10
        k = np.logspace(-5, np.log10(K_MAX), num=1000)
        self.Pk,self.Pk_d,self.Pk_dd = self.get_pkinfo_z(k,0)
        self.Pk_NL = self.get_Pk_NL(k,0) #if you want HALOFIT P(k)

        ################################################################################## survey stuff
        #useful later i'm sure
        self.survey_params = survey_params
        self.compute_bias = compute_bias
        self.HMF = HMF
        
        #setup surveys and compute all bias params including for multi tracer case...        
        updated_self = self.update_survey(survey_params) # can't reassign self in method
        self.__dict__.update(updated_self.__dict__) 

        #get ranges of cross survey
        self.z_min = max([self.survey.z_range[0],self.survey1.z_range[0]])
        self.z_max = min([self.survey.z_range[1],self.survey1.z_range[1]])
        if self.z_min >= self.z_max:
            raise ValueError("incompatible survey redshifts.")
        self.z_survey = np.linspace(self.z_min,self.z_max,int(1e+3))
        self.f_sky = min([self.survey.f_sky,self.survey1.f_sky])
          
    ####################################################################################
    #get power spectras - linear and non-linear Halofit
    def get_class_powerspectrum(self,kk,zz=0): #h are needed to convert to 1/Mpc for k then convert pk back to (Mpc/h)^3
        return np.array([self.cosmo.pk_lin(ki, zz) for ki in kk*self.h])*self.h**3

    def get_Pk_NL(self,kk,zz=0): # for halofit non-linear power spectrum
        pk_nl = CubicSpline(kk,np.array([self.cosmo.pk(ki, zz) for ki in kk*self.h])*self.h**3)
        return pk_nl

    def get_pkinfo_z(self,k,z):
        """get Pk and its k derivatives"""
        Plin = self.get_class_powerspectrum(k,0)#just always get present day power spectrum
        Pk = CubicSpline(k,Plin)#get linear power spectrum
        Pk_d = Pk.derivative(nu=1)  
        Pk_dd = Pk.derivative(nu=2)       
        return Pk,Pk_d,Pk_dd
               
    ###########################################################
        
    #read in survey_params class and define self.survey and self.survey1 
    def _process_survey(self, survey_params, compute_bias, HMF,verbose=True):
        """
        Get bias funcs for a given survey - compute biases from HMF and HOD relations if flagged
        """
        class_bias = SetSurveyFunctions(survey_params, compute_bias)
        class_bias.z_survey = np.linspace(class_bias.z_range[0],class_bias.z_range[1],int(1e+3))
            
        if compute_bias:
            if verbose:
                print("Computing bias functions...")
            PBs  = PBBias(self,survey_params, HMF)
            PBs.add_bias_attr(class_bias)

        return class_bias

    def update_survey(self, survey_params,verbose=True):
        """
        Get bias functions for given surveys allowing for two surveys in list or not list format
        - including betas
        Deepcopy of everything except cosmo as it is cythonized classy object
        """
        new_self = create_copy(self)
        new_self.multi_tracer = False # is it a multi-tracer case?
        
        # If survey_params is a list
        if type(survey_params)==list:
            new_self.survey = new_self._process_survey(survey_params[0], new_self.compute_bias, new_self.HMF,verbose=verbose)
            new_self.survey.betas = None # these are not computed until called - resets if updating biases
        
            #check for additional surveys
            if len(survey_params) > 1:
                self.multi_tracer = True
                #Process second survey 
                new_self.survey1 = new_self._process_survey(survey_params[1], new_self.compute_bias, new_self.HMF,verbose=verbose)
                new_self.survey1.betas = None
                 
            else:
                new_self.survey1 = new_self.survey
        else:
            # Process survey
            new_self.survey = new_self._process_survey(survey_params, new_self.compute_bias, new_self.HMF,verbose=verbose)
            new_self.survey.betas = None
            new_self.survey1 = new_self.survey
        
        new_self = new_self.compute_derivs() # set up derivatives for cosmology dependent functions
        return new_self
                
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
    
    ############################################################################################################

    def solve_second_order_KC(self):
        """
        Get second order growth factors - redshift dependent corrections to F2 and G2 kernels (very minimal)
        """
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
        self.K_intp = CubicSpline(odeint_zz,K)
        self.C_intp = CubicSpline(odeint_zz,C)

    def lnd_derivatives(self,functions_to_differentiate,tracer=None):
        """
            calculates derivatives of a list of functions wrt log comoving dist numerically
        """
        if tracer is None:
            tracer = self.survey

        # Store first derivatives in a list
        function_derivatives = []

        for func in functions_to_differentiate:
            # Calculate numerical derivatives of the function with respect to ln(d)
            derivative_func =  CubicSpline(tracer.z_survey, np.gradient(func(tracer.z_survey), np.log(self.comoving_dist(tracer.z_survey))))
            function_derivatives.append(derivative_func)

        return function_derivatives
    
    #######################################################################################
    #getter functrions
    def get_params(self,k1,k2,k3=None,theta=None,zz=0,tracer = None,nonlin=False,growth2=False):
        """
            return arrays of redshift and k dependent parameters for bispectrum
        """
        if theta is None:
            if k3 is None:
                raise  ValueError('Define either theta or k3')
            else:
                theta = get_theta(k1,k2,k3) #from utils
        else:
            if k3 is None:
                k3 = get_k3(theta,k1,k2)
                
        if tracer is None:
            tracer = self.survey
        
        Pk1 = self.Pk(k1)
        Pk2 = self.Pk(k2)
        Pk3 = self.Pk(k3)

        Pkd1 = self.Pk_d(k1)
        Pkd2 = self.Pk_d(k2)
        Pkd3 = self.Pk_d(k3)

        Pkdd1= self.Pk_dd(k1)
        Pkdd2 = self.Pk_dd(k2)
        Pkdd3 = self.Pk_dd(k3)
        
        if self.nonlin:
            Pk1 = self.Pk_NL(k1)
            Pk2 = self.Pk_NL(k2)
            Pk3 = self.Pk_NL(k3)
        
        #redshift dependendent terms
        d = self.comoving_dist(zz)
        
        K = 3/7 # from einstein-de-sitter
        C = 3/7
        if self.growth2:
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
    
    def get_params_pk(self,k1,zz):
        """
           return arrays of redshift and k dependent parameters for power spectrum
        """
        
        Pk1 = self.Pk(k1)
        Pkd1 = self.Pk_d(k1)
        Pkdd1= self.Pk_dd(k1)
        
        if self.nonlin:
            Pk1 = self.Pk_NL(k1)
            
        #redshift dependendent terms
        d = self.comoving_dist(zz)

        f = self.f_intp(zz)
        D1 = self.D_intp(zz)
        
        return k1,Pk1,Pkd1,Pkdd1,d,f,D1
    
    def get_PNGparams(self,zz,k1,k2,k3,tracer = None, shape='Loc'):
        """
        returns terms needed to compute PNG contribution including scale-dependent bias for bispectrum
        """
        if tracer is None:
            tracer = self.survey
        
        if shape == 'Loc':
            bE01 = tracer.loc.b_01(zz)
            bE11 = tracer.loc.b_11(zz)
            
        elif shape == 'Eq':
            try:
                bE01 = tracer.eq.b_01(zz)
                bE11 = tracer.eq.b_11(zz)
                
            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")
                
        elif shape == 'Orth':
            try:
                bE01 = tracer.orth.b_01(zz)
                bE11 = tracer.orth.b_11(zz)
                
            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")
        else:
            raise ValueError("Select PNG shape: Loc,Eq,Orth")

        Mk1 = self.M(k1, zz)
        Mk2 = self.M(k2, zz)
        Mk3 = self.M(k3, zz)
        
        return bE01,bE11,Mk1,Mk2,Mk3
    
    def get_PNGparams_pk(self,zz,k1,tracer = None, shape='Loc'):
        """
        returns terms needed to compute PNG contribution including scale-dependent bias for power spectra
        """
        if tracer is None:
            tracer = self.survey
        
        if shape == 'Loc':
            bE01 = tracer.loc.b_01(zz) # only need b_phi
            #bE11 = tracer.loc.b_11(zz)
            
        elif shape == 'Eq':
            try:
                bE01 = tracer.eq.b_01(zz)
                #bE11 = tracer.eq.b_11(zz)
                
            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")
                
        elif shape == 'Orth':
            try:
                bE01 = tracer.orth.b_01(zz)
                #bE11 = tracer.orth.b_11(zz)
                
            except AttributeError:
                raise ValueError("For non-local PNG use compute_bias=True when instiating cosmo_funcs")
        else:
            raise ValueError("Select PNG shape: Loc,Eq,Orth")

        Mk1 = self.M(k1, zz)
        
        return bE01,Mk1
    
    def compute_derivs(self,tracer = None):
        """
        Compute derivatives wrt comoving distance of redshift dependent parameters for radial evolution terms
        Splits functions into survey dependent and cosmology dependent functions.

        if tracer is not None, computes survey dependent derivatives for the given tracer.
        if tracer is None, computes cosmology dependent derivatives. This is only called on initialisation of ClassWAP
        """
        
        if tracer is not None:
            
            tracer.deriv.b1_d,tracer.deriv.b2_d,tracer.deriv.g2_d = self.lnd_derivatives([tracer.b_1,tracer.b_2,tracer.g_2],tracer=tracer)
            tracer.deriv.b1_dd,tracer.deriv.b2_dd,tracer.deriv.g2_dd = self.lnd_derivatives([tracer.b1_d,tracer.b2_d,tracer.g2_d],tracer=tracer)
            
            return tracer
        else:
            # Just compute for the cosmology dependent functions and reset the survey derivatives
            if hasattr(self, 'f_d'):
                # If already computed, just return
                self.survey.deriv = None
                self.survey1.deriv = None
                return self
            else:
                #get derivs of cosmology dependent functions
                self.f_d,self.D_d = self.lnd_derivatives([self.f_intp,self.D_intp])
                self.f_dd,self.D_dd = self.lnd_derivatives([self.f_d,self.D_d])

                self.survey.deriv = None
                self.survey1.deriv = None
                return self

    
    def get_derivs(self,zz,tracer = None):
        """
        Get derivatives wrt comoving distance of redshift dependent parameters for radial evolution terms
        """
        if tracer is None:
            tracer = self.survey

        if not hasattr(tracer, 'deriv'):
            tracer = self.compute_derivs(tracer=tracer)
            if not self.multi_tracer: # no need to recompute for second survey
                self.survey1.deriv = tracer.deriv
        
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
    
    def get_beta_funcs(self,zz,tracer = None):
        """Get betas for given redshifts for given tracer if they are not already computed.
        If not computed then compute them using betas.interpolate_beta_funcs"""

        if tracer is None:
            tracer = self.survey

        if hasattr(tracer, 'betas') and tracer.betas is not None:
            return [tracer.betas[i](zz) for i in range(len(tracer.betas))]
        else:
            tracer.betas = betas.interpolate_beta_funcs(self,tracer = tracer)
            if not self.multi_tracer: # no need to recompute for second survey
                self.survey1.betas = tracer.betas

            return [tracer.betas[i](zz) for i in range(len(tracer.betas))]
    
