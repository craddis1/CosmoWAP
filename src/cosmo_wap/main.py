import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

from cosmo_wap.peak_background_bias import *
from cosmo_wap.survey_params import *
from cosmo_wap.lib import utils
from cosmo_wap.lib import betas

class ClassWAP:
    r"""
        Willkommen, Bienvenue, Welcome...
           ______                         _       _____    ____ 
          / ____/___  _________ ___  ____| |     / /   |  / __ \
         / /   / __ \/ ___/ __ `__ \/ __ \ | /| / / /| | / /_/ /
        / /___/ /_/ (__  ) / / / / / /_/ / |/ |/ / ___ |/ ____/ 
        \____/\____/____/_/ /_/ /_/\____/|__/|__/_/  |_/_/      
                                                            
        Main class - takes in cosmology from CLASS and survey parameters and then can called to generate cosmology (f,P(k),P'(k),D(z) etc) and all other biases including relativstic parts
    """
    def __init__(self,cosmo,survey_params=None,compute_bias=False,HMF='Tinker2010',emulator=False,verbose=True,params=None,fast=False,nonlin=False):
        """
        Inputs CLASS and bias dict to return all bias and cosmological parameters defined within the class object
        """
        self.nonlin  = nonlin  #use nonlin halofit powerspectra
        self.growth2 = False #second order growth corrections to F2 and G2 kernels
        self.n = 128 # default n for integrated terms - used currently in forecast stuff 
        self.term_list = ['NPP','RR1','RR2','WA1','WA2','WAGR','WS','WAGR','RRGR','WSGR','Full','GR1','GR2','GRX','Loc','Eq','Orth','IntInt','IntNPP'] # list of terms currently implemented. Does not include composites - see pk/combined.py etc
        
        # so we can use emulators for Pk to speed up sampling cosmological parameter space
        if emulator:
            self.emulator = True
            K_MAX_h = 10 # in Units of [1/Mpc] - is limit of emulator
            if emulator is True:
                self.emu = utils.Emulator() # iniate nested emulator class using CosmoPower
            else:
                self.emu = emulator # use pre-loaded estimators

        else:
            self.emu = None
            self.emulator = False
            K_MAX_h = cosmo.pars['P_k_max_1/Mpc'] # in Units of [1/Mpc]

        #############  interpolate cosmological functions  - this is quick as we call class to calculate a limited set of parameters
        self.compute_background(cosmo,params)

        ##################################################################################
        #get powerspectra
        self.K_MAX = K_MAX_h/self.h               # in Units of [h/Mpc]
        k = np.logspace(-5, np.log10(self.K_MAX), num=400)
        self.Pk,self.Pk_d,self.Pk_dd = self.get_pk(k)
        
        ################################################################################## survey stuff
        self.survey_params = survey_params
        self.compute_bias = compute_bias
        self.HMF = HMF
        
        # setup surveys and compute all bias params including for multi tracer case...    
        if survey_params:    
            self.update_survey(survey_params,verbose=verbose)

        if nonlin or not fast:
            # get 2D interpolated halofit powerspectrum function (k,z) - need maximum redshift here
            z_range = np.linspace(0,5,50) # for integrated effects need all values below maximum redshift
            self.Pk_NL = self.get_Pk_NL(k,z_range)

    def compute_background(self,cosmo,params):
        """Use class to compute background without much overhead."""
        zz = np.linspace(0,10,100) # for now we have a redshift range up z=10 - so sample every 0.1 z
        self.D = CubicSpline(zz,cosmo.scale_independent_growth_factor(zz))
        self.f = CubicSpline(zz,cosmo.scale_independent_growth_factor_f(zz))
        self.cosmo = cosmo
        self.load_cosmology(params) # load cosmological paramerters into object
        self.H_c           = CubicSpline(zz,cosmo.Hubble(zz)*(1/(1+zz))/self.h)  # now in h/Mpc! - is conformal
        self.dH_c          = self.H_c.derivative(nu=1)                           # first derivative wrt z
        xi_zz              = self.h*cosmo.comoving_distance(zz)                  # Mpc/h
        self.comoving_dist = CubicSpline(zz,xi_zz)                                                          
        self.d_to_z        = CubicSpline(xi_zz,zz)                               # useful to map other way
        # let see if faster use simpler stuff for Om_m # less than 0.2% at z=5 and 0.5% at z=10
        #self.Om_m          = CubicSpline(zz,cosmo.Om_m(zz))
        self.Om_m          = CubicSpline(zz,(2/3)*(1 + (1+zz)*self.dH_c(zz)/self.H_c(zz)))
        #misc
        self.c = 2.99792e+5 #km/s

    def load_cosmology(self,params):
        """Unify the way we call cosmological parameters so they are defined within cosmo_funcs"""

        if params is None:
            self.A_s = self.cosmo.get_current_derived_parameters(['A_s'])['A_s'] #from cosmo (classy)
            self.h = self.cosmo.h()
            self.Omega_m = self.cosmo.Omega_m()
            self.Omega_b = self.cosmo.Omega_b()
            self.Omega_cdm = self.Omega_m - self.Omega_b
            self.n_s = self.cosmo.n_s()
            if not self.emulator:
                self.sigma8 = self.cosmo.sigma8() # not computed in classy if it doesn't compute P(k)
        else:
            self.__dict__.update(params) # get from params dict (quicker)  

        return self
          
    ####################################################################################
    #get power spectras - linear and non-linear Halofit
    def get_class_powerspectrum(self,kk,zz=0): #h are needed to convert to 1/Mpc for k then convert pk back to (Mpc/h)^3
        return np.array([self.cosmo.pk_lin(ki, zz) for ki in kk*self.h])*self.h**3
    
    def get_pk(self,k):
        """get Pk and its k derivatives"""
        if self.emulator:
            params_lin = {'omega_b': [self.Omega_b*self.h**2], # in terms of Omega_b*h**2
                    'omega_cdm': [self.Omega_m*self.h**2 - self.Omega_b*self.h**2],
                    'h': [self.h],
                    'n_s': [self.n_s],
                    'ln10^{10}A_s': [np.log(10**10 *self.A_s)],
                    'z': [0],   # just linear pk at z=0
                    }
            
            # originally maps to log10(Pk)
            self.plin = self.emu.Pk.predictions_np(params_lin)[0] # can use for non-lin part as well
            Plin = 10.**(self.plin)*self.h**3 # is array in k (k defined by emu_k)
            k = self.emu.k/self.h # set k_modes to output of emulator
        else:
            Plin = self.get_class_powerspectrum(k,0)#just always get present day power spectrum

        Pk = CubicSpline(k,Plin) # get linear power spectrum
        Pk_d = Pk.derivative(nu=1)  
        Pk_dd = Pk.derivative(nu=2)
        return Pk,Pk_d,Pk_dd

    def get_Pk_NL(self,kk,zz): # for halofit non-linear power spectrum
        """
        Get 2D (k,z) interpolated nonlinear power spectrum - has non-trivial time dependence
        only want non-linear correction on small scales - use linear P(k) for large scales
        """

        if self.emulator:
            # all input arrays must have the same shape
            n = len(zz)
            batch_params_lin = {'omega_b': [self.Omega_b*self.h**2]*n,
                    'omega_cdm': [self.Omega_m*self.h**2 - self.Omega_b*self.h**2]*n,
                    'h': [self.h]*n,
                    'n_s': [self.n_s]*n,
                    'ln10^{10}A_s': [np.log(10**10 *self.A_s)]*n,
                    'z': zz,   # just linear pk at z=0
                    }
            # hmcode parameters - can play around here
            batch_params_hmcode = {'c_min': [3]*n,
                       'eta_0': [0.6]*n}
            
            #combine parameters
            batch_params_nlboost = {**batch_params_lin, **batch_params_hmcode}
            total_log_power = self.plin + self.emu.Pk_NL.predictions_np(batch_params_nlboost)
            pks = (10.**(total_log_power)*self.h**3).T /(self.D(zz)**2) # make (k,z) shape
            kk = self.emu.k/self.h # set k_modes to output of emulator

        else:
            # so most efficiently get 2D grid using cosmo.get_pk - class 
            kk_base = kk[:,np.newaxis,np.newaxis] *self.h
            kk_arr = np.broadcast_to(kk_base, (kk.size,zz.size,1)) # we do this rearranging for cosmo.get_pk
            pk_nonlin = self.h**3 *self.cosmo.get_pk(kk_arr,zz,kk.size,zz.size,1)[...,0]/(self.D(zz)**2)[np.newaxis,:]

            # use linear on large scales...
            pk_lin = np.broadcast_to(self.Pk(kk)[:,np.newaxis], (pk_nonlin.shape))
            pks = np.where(pk_nonlin>pk_lin,pk_nonlin,pk_lin)

        interp = scipy.interpolate.RegularGridInterpolator((kk, zz), pks, bounds_error=False)

        def f(x, y):
            """
            A wrapper for the RegularGridInterpolator that allows calling with individual coordinates.
            """
            return interp((x, y))

        return f
    
    def pk(self,k):
        """After K_MAX we just have K^{-3} power law - just linear power spectra"""
        return np.where(k > self.K_MAX,self.Pk(self.K_MAX)*(k/self.K_MAX)**(-3),self.Pk(k))
    
    ###########################################################
        
    #read in survey_params class and define self.survey and self.survey1 
    def _process_survey(self, survey_params, compute_bias, HMF,verbose=True):
        """
        Get bias funcs for a given survey - compute biases from HMF and HOD relations if flagged
        """
        class_bias = SetSurveyFunctions(survey_params, compute_bias)
        class_bias.z_survey = np.linspace(class_bias.z_range[0],class_bias.z_range[1],100)
            
        if compute_bias:
            if verbose:
                print("Computing bias functions...")
            PBs  = PBBias(self,survey_params, HMF)
            PBs.add_bias_attr(class_bias)

        return class_bias

    def update_survey(self,survey_params,verbose=True):
        """
        Get bias functions for given surveys allowing for two surveys in list or not list format
        - including betas
        Deepcopy of everything except cosmo as it is cythonized classy object
        """
        self.multi_tracer = False # is it multi-tracer
        
        # If survey_params is a list
        if type(survey_params)==list:
            self.survey = self._process_survey(survey_params[0], self.compute_bias, self.HMF,verbose=verbose)
            self.survey.betas = None # these are not computed until called - resets if updating biases
            self.survey.t1 = True # is tracer 1 is useful flag for multi-tracer forecasts
        
            #check for additional surveys
            if len(survey_params) > 1:
                self.multi_tracer = True
                #Process second survey 
                self.survey1 = self._process_survey(survey_params[1], self.compute_bias, self.HMF,verbose=verbose)
                self.survey1.betas = None
                self.survey1.t1 = False # is tracer2
            else:
                self.survey1 = self.survey
        else:
            # Process survey
            self.survey = self._process_survey(survey_params, self.compute_bias, self.HMF,verbose=verbose)
            self.survey.betas = None
            self.survey.t1 = True
            self.survey1 = self.survey
        
        self.compute_derivs() # set up derivatives for cosmology dependent functions
        self.update_shared_survey() # update z_range,f_sky,n_g etc
        return self
    
    def update_shared_survey(self):
        """define quanties defined by the cross of the two tracers- redshift range,f_sky etc"""
        self.z_min = max([self.survey.z_range[0],self.survey1.z_range[0]])
        self.z_max = min([self.survey.z_range[1],self.survey1.z_range[1]])
        if self.z_min >= self.z_max:
            raise ValueError("Incompatible survey redshifts.")
        self.z_survey = np.linspace(self.z_min,self.z_max,int(1e+2))
        self.f_sky = min([self.survey.f_sky,self.survey1.f_sky])
        if self.multi_tracer:
            self.n_g = lambda xx: 0*xx + 1e+10 # for multi-tracer set shot noise to zero...
        else:
            self.n_g = self.survey.n_g
        return self
                
    #######################################################################################################
    
    def Pk_phi(self,k, k0=0.05):
        """Power spectrum of the Bardeen potential Phi in the matter-dominated era - k in units of h/Mpc.
        """
        k_pivot = k0/self.h # get pivot scale in [h/Mpc]
        resp = (9.0/25.0) * self.A_s * (k/k_pivot)**(self.n_s - 1.0)
      
        resp *= 2*np.pi**2.0/k**3.0    #[Mpc/h]^3

        return resp

    def M(self,k, z):
        """The scaling factor between the primordial scalar power spectrum and the late-time matter power spectrum in linear theory
        """
        return np.sqrt(self.D(z)**2 *self.Pk(k) / self.Pk_phi(k))
    
    ############################################################################################################

    def solve_second_order_KC(self):
        """
        Get second order growth factors - redshift dependent corrections to F2 and G2 kernels (very minimal)
        """
        dD_dz = self.D.derivative(nu=1) # first derivative wrt to z

        def F_func(u,zz): # so variables are F and H and D
            f,fd = u # unpack u vector
            D_zz = self.D(zz)
            return [fd,(-self.H_c(zz)*self.dH_c(zz)*(1+zz)**2 *fd + ((3*(self.H_c(0))**2 * self.Om_m(0) *(1+zz))/(2))*(f+D_zz**2))/(self.H_c(zz)**2 *(1+zz)**2)]

        odeint_zz = np.linspace(10,0.05,int(1e+5))# so z=10 is peak matter domination...

        #set initial params for F
        F0 = [(3/7)*self.D(odeint_zz[0])**2,(3/7)*2*self.D(odeint_zz[0])*dD_dz(odeint_zz[0])]
        sol1 = odeint(F_func,F0,odeint_zz)
        K = (sol1[:,0]/self.D(odeint_zz)**2)
        C = sol1[:,1]/(2*self.D(odeint_zz)*dD_dz(odeint_zz))
        self.K_intp = CubicSpline(odeint_zz[::-1],K[::-1]) # strictly increasing
        self.C_intp = CubicSpline(odeint_zz[::-1],C[::-1])

    def lnd_derivatives(self,functions_to_differentiate,tracer=None):
        """
        Calculates derivatives of a list of functions wrt log comoving dist numerically
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
            return arrays of redshift and k dependent parameters for bispectrum - nonlin and growth2 are legacy and are slowly being removed
        """
        if theta is None:
            if k3 is None:
                raise  ValueError('Define either theta or k3')
            else:
                theta = utils.get_theta(k1,k2,k3) #from utils
        else:
            if k3 is None:
                k3 = utils.get_k3(theta,k1,k2)
                
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
            Pk1 = self.Pk_NL(k1,zz)
            Pk2 = self.Pk_NL(k2,zz)
            Pk3 = self.Pk_NL(k3,zz)
        
        #redshift dependendent terms
        d = self.comoving_dist(zz)
        
        K = 3/7 # from einstein-de-sitter
        C = 3/7
        if self.growth2:
            self.solve_second_order_KC()#get K and C
            K = self.K_intp(zz)
            C = self.C_intp(zz)

        f = self.f(zz)
        D1 = self.D(zz)
        
        #survey stuff
        b1 = tracer.b_1(zz)
        b2 = tracer.b_2(zz)
        g2 = tracer.g_2(zz)
        return k1,k2,k3,theta,Pk1,Pk2,Pk3,Pkd1,Pkd2,Pkd3,Pkdd1,Pkdd2,Pkdd3,d,K,C,f,D1,b1,b2,g2
    
    def unpack_pk(self,k1,zz,GR=False,fNL_type=None,WS=False,RR=False):
        """Helper function to unpack all necessary terms with flag for each different type of term
        Should reduce the number of duplicated lines and make maintanence easier
        
        Is multi-tracer compliant
        
        Returns: list of parameters in order:
        - Base: [Pk, f, D1, b1, xb1] 
        - +GR: [gr1, gr2, xgr1, xgr2]
        - +fNL: [bE01, Mk1, xbE01]
        - +WS/RR: [Pkd1, Pkdd1, d]
        - +RR: [fd, Dd, bd1, xbd1, fdd, Ddd, bdd1, xbdd1]
        - +RR+GR: [grd1, xgrd1]
        Total: [Pk, f, D1, b1, xb1, gr1, gr2, xgr1, xgr2, bE01, Mk1, xbE01, Pkd1, Pkdd1, d, fd, Dd, bd1, xbd1, fdd, Ddd, bdd1, xbdd1, grd1, xgrd1]"""

        #basic params
        if self.nonlin:
            Pk1 = self.Pk_NL(k1,zz)
        else:
            Pk1 = self.Pk(k1)

        f = self.f(zz)
        D1 = self.D(zz)

        b1 = self.survey.b_1(zz)
        xb1 = self.survey1.b_1(zz)

        params = [Pk1,f,D1,b1,xb1]

        if GR:
            gr1,gr2   = self.get_beta_funcs(zz,tracer = self.survey)[:2]
            xgr1,xgr2 = self.get_beta_funcs(zz,tracer = self.survey1)[:2]
            params.extend([gr1,gr2,xgr1,xgr2])
        
        if fNL_type is not None:
            bE01,Mk1 =  self.get_PNGparams_pk(zz,k1,tracer=self.survey, shape=fNL_type)
            xbE01,Mk1 =  self.get_PNGparams_pk(zz,k1,tracer=self.survey1, shape=fNL_type)
            params.extend([bE01,Mk1,xbE01])

        if WS or RR:
            Pkd1  = self.Pk_d(k1)
            Pkdd1 = self.Pk_dd(k1)
            d = self.comoving_dist(zz)
            params.extend([Pkd1,Pkdd1,d])

            if RR:
                if not hasattr(self.survey, 'deriv') or not self.survey.deriv:
                    self.survey = self.compute_derivs(tracer=self.survey)
                    if self.multi_tracer: # no need to recompute for second survey
                        self.survey1 = self.compute_derivs(tracer=self.survey1)
                    else:
                        self.survey1.deriv = self.survey.deriv

                fd = self.f_d(zz)
                Dd = self.D_d(zz)
                bd1 = self.survey.deriv['b1_d'](zz)
                xbd1 = self.survey1.deriv['b1_d'](zz)
                fdd = self.f_dd(zz)
                Ddd = self.D_dd(zz)
                bdd1 = self.survey.deriv['b1_dd'](zz)
                xbdd1 = self.survey1.deriv['b1_dd'](zz)
                params.extend([fd,Dd,bd1,xbd1,fdd,Ddd,bdd1,xbdd1])
        
                if GR:
                    #beta derivs
                    grd1 = self.get_beta_derivs(zz,tracer=self.survey)[0]
                    xgrd1 = self.get_beta_derivs(zz,tracer=self.survey1)[0]
                    params.extend([grd1,xgrd1])
            
        return params
    
    def get_PNG_bias(self,zz,tracer,shape):
        """Get b_01 and b_11 arrays depending on tracer redshift and shape"""
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
    
        return bE01,bE11
    
    def get_PNGparams(self,zz,k1,k2,k3,tracer = None, shape='Loc'):
        """
        returns terms needed to compuself.survey1 = self.compute_derivs(tracer=self.survey1)te PNG contribution including scale-dependent bias for bispectrum
        """
        if tracer is None:
            tracer = self.survey
        
        bE01,bE11 = self.get_PNG_bias(zz,tracer,shape)

        Mk1 = self.M(k1, zz)
        Mk2 = self.M(k2, zz)
        Mk3 = self.M(k3, zz)
        
        return bE01,bE11,Mk1,Mk2,Mk3

    def get_PNGparams_pk(self,zz,k1,tracer=None,shape='Loc'):
        """
        returns terms needed to compute PNG contribution including scale-dependent bias for power spectra
        """
        if tracer is None:
            tracer = self.survey
        
        bE01,_ = self.get_PNG_bias(zz,tracer,shape) # only need b_phi

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
            
            tracer.deriv['b1_d'],tracer.deriv['b2_d'],tracer.deriv['g2_d'] = self.lnd_derivatives([tracer.b_1,tracer.b_2,tracer.g_2],tracer=tracer)
            tracer.deriv['b1_dd'],tracer.deriv['b2_dd'],tracer.deriv['g2_dd'] = self.lnd_derivatives([tracer.deriv['b1_d'],tracer.deriv['b2_d'],tracer.deriv['g2_d']],tracer=tracer)
            
            return tracer
        else:
            # Just compute for the cosmology dependent functions and reset the survey derivatives
            if hasattr(self, 'f_d'):
                # If already computed, just return
                self.survey.deriv = {}
                self.survey1.deriv = {}
                return self
            else:
                #get derivs of cosmology dependent functions
                self.f_d,self.D_d = self.lnd_derivatives([self.f,self.D])
                self.f_dd,self.D_dd = self.lnd_derivatives([self.f_d,self.D_d])

                self.survey.deriv = {}
                self.survey1.deriv = {}
                return self
    
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
        
    def get_beta_derivs(self,zz,tracer=None):
        """Get betas derivatives wrt comoving distance for given redshifts for given tracer if they are not already computed.
        If not computed then compute"""

        if tracer is None:
            tracer = self.survey

        if hasattr(tracer.deriv,'beta'):
            return [tracer.deriv['beta'][i](zz) for i in range(len(tracer.deriv['beta']))]
        else:
            #get betad - derivatives wrt to ln(d)  - for radial evolution terms
            betad = np.array(self.lnd_derivatives(tracer.betas[-6:]),dtype=object) #beta14-19
            grd1 = self.lnd_derivatives([tracer.betas[0]])
            tracer.deriv['beta'] = np.concatenate((grd1,betad))

            if not self.multi_tracer: # no need to recompute for second survey
                self.survey1.deriv = tracer.deriv
            
            return [tracer.deriv['beta'][i](zz) for i in range(len(tracer.deriv['beta']))]
    
