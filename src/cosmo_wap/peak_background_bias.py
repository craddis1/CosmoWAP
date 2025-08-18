import numpy as np
import scipy
import scipy.integrate
import scipy.special as sp
from scipy.interpolate import CubicSpline
#from scipy.interpolate import RegularGridInterpolator

#from hmf import MassFunction

class PBBias:
    def __init__(self,cosmo_funcs,survey_params,HMF='Tinker2010'):
        """
           Gets non-gaussian biases from Peak Background split approach - assumes HMF (Tinker 2010) and HOD (yankelevich and porciani 2018)
        """
        
        self.cosmo_funcs = cosmo_funcs #for later
        self.survey_params = survey_params
        delta_c = 1.686 #from spherical collapse
        self.delta_c = delta_c

        #for critical density
        GG = 4.300917e-3 #[pc M_sun^-1 (km/s)^2]
        G = GG/(1e+6 * cosmo_funcs.c**2)#gravitational constant # [Mpc M_sun^-1]
        self.rho_crit = lambda xx: 3*(cosmo_funcs.H_c(xx)*(1+xx))**2/(8*np.pi*G)  #in units of h^2 Mo/ Mpc^3 where Mo is solar mass
        self.rho_m = lambda xx: self.rho_crit(xx)*cosmo_funcs.Om_m(xx)            #in units of h^2 Mo/ Mpc^3
        
        #so several parameters are 2 dimensional - dependent on R and z
        #therefore we interpolate over redshift here for convenience later where we call things as a fuction of z
        """
        #define R range 
        self.R = np.logspace(-2.5,2.5,300,dtype=np.float32) #upper limit - causes errors for large R 
        
        # M is the mass enclosed within the radius which also redshift dependent
        self.M_func = lambda xx: ((4*np.pi*self.rho_m(xx)*self.R**3)/3)
        """

        # so instead lets use the hmf library
        #mf = MassFunction(hmf_model=HMF,z=0)
        #self.M = mf.m
        self.R = np.logspace(-1.5,1.5,100,dtype=np.float32) # [Mpc/h]
        self.M = lambda xx: ((4*np.pi*self.rho_m(xx)*self.R**3)/3) # [M_sun/h] - array in R as func of z
        
        #precompute sigma^2 - only compute once
        sigmaR0 = self.sigma_R_n(self.R,0)
        sigmaR1 = self.sigma_R_n(self.R,-1)
        sigmaR2 = self.sigma_R_n(self.R,-2)
        
        #for sigma 
        sig_R = {}
        
        sig_R['0'] = lambda xx: sigmaR0 * cosmo_funcs.D(xx)**2 # (z,R)
        sig_R['1'] = lambda xx: sigmaR1 * cosmo_funcs.D(xx)**2
        sig_R['2'] = lambda xx: sigmaR2 * cosmo_funcs.D(xx)**2
        
        self.sig_R = sig_R
        
        #peak height (z,R)
        self.nu_func = lambda xx: delta_c/np.sqrt(sig_R['0'](xx))# interpolate along z   
        
        #init classes 
        if HMF == 'ST':
            self.multiplicity = self.multiplicity_ST
            self.lagbias = self.LagBias_ST(self)
        elif HMF == 'Tinker10':
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias_Tinker10(self) #default is tinker2010 mass function
        else:
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias(self) #default is tinker2010 mass function
            
        self.eulbias = self.EulBias(self)
        
        # so get fittted values of NO and M0
        z_arr = np.linspace(survey_params.z_range[0],survey_params.z_range[1],10)
        self.M0_func = self.fit_M0(z_arr)
        self.NO_func = self.fit_NO(z_arr)

        ###############################################################################################
        def get_galaxy_bias(b_h,A=1,alpha=0):
            """
            set z_range and fit cubic spline for a given bias function
            """
            z_samps = np.linspace(survey_params.z_range[0],survey_params.z_range[1],int(40))
            
            bias_arr = np.ones_like(z_samps)
            for i,zz in enumerate(z_samps):
            
                bias_arr[i] = self.general_galaxy_bias(b_h,zz,self.M0_func(zz),self.NO_func(zz),A,alpha)/survey_params.n_g(zz)
            
            return CubicSpline(z_samps,bias_arr)
        
        def get_number_density():
            """
            set z_range and fit cubic spline for a given bias function
            """
            z_samps = np.linspace(survey_params.z_range[0],survey_params.z_range[1],int(40))
            
            n_g = np.ones_like(z_samps)
            for i,zz in enumerate(z_samps):
            
                n_g[i] = self.number_density(zz,self.M0_func(zz),self.NO_func(zz))
            
            return CubicSpline(z_samps,n_g)
        
        self.get_galaxy_bias = get_galaxy_bias
        self.n_g = get_number_density()
        
        #################################################################################################
        # so save all required params to object  
        self.b_1 = get_galaxy_bias(self.eulbias.b1)
        self.b_2 = get_galaxy_bias(self.eulbias.b2)
        self.g_2 = lambda xx: -(2/7)*(self.b_1(xx)-1)#tidal bias - e.g. baldauf
        
        #get PNG biases for each type
        self.loc = self.Loc(self)
        self.eq = self.Eq(self)
        self.orth = self.Orth(self)
        
    
    #########################################################################################
    def number_density(self,zz,M0,NO): 
        """
        return number density for given z 
        """

        return scipy.integrate.simpson(self.HOD(zz,M0,NO)*self.n_h(zz), (self.M(zz)),axis=-1)

    # so implement Eq.
    def general_galaxy_bias(self,b_h,zz,M0,NO,A=1,alpha=0):
        """
        return bias for given z
        """

        # Integrate over M for each value of z
        integral_values = scipy.integrate.simpson(b_h(zz, A, alpha)*self.n_h(zz)*self.HOD(zz,M0,NO), (self.M(zz)),axis=-1)

        return integral_values
    ############################################################################################

    #fit NO and M0 to given n_g and b_1
    def fit_M0(self,z_arr): #M0 is in units of M_sun/h
        """
        fit M0 from linear bias (b_1 is independent of NO)
        """
        def objective(M0,zz,NO=1):
            """
            diff between linear bias from PBS and survey specifications
            """
            return self.general_galaxy_bias(self.eulbias.b1,zz,M0,NO)/self.number_density(zz,M0,NO) - self.survey_params.b_1(zz)

        M0_arr = np.array([scipy.optimize.newton(objective, x0=1e+12, args=(z,),rtol=1e-5) for i,z in enumerate(z_arr)])

        return CubicSpline(z_arr,M0_arr) # now returns M0 as function of redshift

    #now can find NO from n_g
    def fit_NO(self,z_arr):
        """
        fit NO from number density
        """
        def objective(NO,zz,M0):
            """
            diff between number density from PBS and survey specifications
            """  
            return self.number_density(zz,M0,NO) - self.survey_params.n_g(zz)

        # Use the secant method by not providing a derivative
        NO_arr = np.array([scipy.optimize.newton(objective, x0=2.0, args=(z,self.M0_func(z)),rtol=1e-5) for i,z in enumerate(z_arr)])

        return CubicSpline(z_arr,NO_arr) # now returns M0 as function of redshift
    
        
    #############################################################################################################
    def sigma_R_n(self, R, n, K_MIN = 5e-5,steps=int(1e+3)):
        """ Works -well in agreement with other codes...
        compute sigma^2 for a given radius and n i.e does interal over k

        Uses differential equation approach for the yinit
        """
        #k = np.logspace(np.log10(K_MIN), np.log10(self.cosmo_funcs.K_MAX), num=steps)

        def deriv_sigma(x,y,R,n): # adapted from Pylians
            kR=x*R
            W=3.0*(np.sin(kR)-kR*np.cos(kR))/kR**3
            return [x**(2+n)*self.cosmo_funcs.pk(x)*W**2]

        #this function computes sigma(R)
        def sigma_sq(R,n):
            k_limits=[K_MIN,self.cosmo_funcs.K_MAX]; yinit=[0.0]

            I = scipy.integrate.solve_ivp(deriv_sigma,k_limits,yinit,args=(R,n),method='RK45')

            return I.y[0][-1] #get last value

        sigma_squared = np.zeros((len(R)))
        for i,_ in enumerate(R):
            sigma_squared[i] = sigma_sq(R[i],n)/(2.0*np.pi**2)

        return sigma_squared
    
    ##################################################################### halo mass function stuff 
    #so lets just get lagrangian bias parameters from the derivates of the multiplicity functions
    class LagBias:
        """
        Get numeric local-in-matter bias expansion
        """
        def __init__(self,parent_class):
            self.pc = parent_class
            
        def b1(self, zz):
            return self.get_bn(zz,1)
        
        def b2(self, zz):
            return self.get_bn(zz, 2)
               
        def get_bn(self,zz,N):
            
            coef = (-self.pc.nu_func(zz))**N /(self.pc.delta_c**N * self.pc.multiplicity(zz))
            # computes derivatives of the multiplicity function
            deriv = self.pc.multiplicity(zz) # so only 1D - only for 1 z value
            for _ in range(N):
                deriv = np.gradient(deriv,self.pc.nu_func(zz),axis=-1)
 
            return coef*deriv
    
    
    #######################################################################
    def halo_bias_params(self,zz):#tinker 2010
        #define initial free paremeters from Tinker 2010 table 4, Delta=200
        #alpha0 = 0.368
        beta0 = 0.589
        gamma0 = 0.864
        eta0 = -0.243
        phi0 = -0.729

        #free_args0 = alpha,beta,gamma,eta,psi
        beta = beta0*(1+zz)**(0.2)
        gamma = gamma0*(1+zz)**(-0.01)
        eta = eta0*(1+zz)**(0.27)
        phi = phi0*(1+zz)**(-0.08)
        
        alpha = 1 / (2 ** (eta - phi - 0.5)* beta ** (-2 * phi)* gamma ** (-0.5 - eta)* (2**phi * beta ** (2 * phi) * sp.gamma(eta + 0.5)+ gamma**phi * sp.gamma(0.5 + eta - phi)))

        return alpha,beta,gamma,eta,phi
    
    #get langrangian and then eulerian biases
    #see Appendix C arXiv:1911.03964v3
    class LagBias_Tinker10:
        """
        define lagrangian biases in terms of z,M and the halo bias params - these are all arrays in (z,R)
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz):
            _,beta,gamma,eta,psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2*psi)/(delta_c*((beta*nu)**(2*psi)+1)) + (gamma*nu**2-2*eta-1)/delta_c

        def b2(self,zz):
            _,beta,gamma,eta,psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2*psi*(2*gamma*nu**2-4*eta+2*psi-1))/(delta_c**2 *((beta*nu)**(2*psi)+1)) + (gamma**2 *nu**4-4*gamma*eta*nu**2-3*gamma*nu**2+4*eta**2+2*eta)/delta_c**2
    
    #######################################  Which HMF: f(nu)
    def multiplicity_Tinker10(self,zz): # from Tinker 2008,2010
        alpha,beta,gamma,eta,psi = self.halo_bias_params(zz)
        nu = self.nu_func(zz)

        return nu*alpha*(1+(beta*nu)**(-2*psi))*nu**(2*eta)*np.exp(-gamma*nu**2 /2)

    def multiplicity_ST(self,zz): # ShethTormen
        A = 0.3221
        p = 0.3
        a = 0.707 # 1 #

        nu = self.nu_func(zz)

        part1 = A*(1+(a*nu**2)**(-p))
        part2 = nu*(2*a/(np.pi))**(1/2)*(np.exp(- a*nu**2 /2))
        return part1*part2
    
    ############################################################################
    
    class LagBias_ST: #Sheth-Tormen mass functions biases
        """
        define lagrangian biases in terms of z,M and the halo bias params - these are all arrays in (z,M)
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz):
            A = 0.322
            p = 0.3
            q = 0.707
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            
            return (q*nu**2-1)/delta_c + 2*p/(delta_c*(1+(q*nu**2)**p))

        def b2(self,zz):
            A = 0.322
            p = 0.3
            q = 0.707
            nu = (self.pc.nu_func(zz))
            delta_c = self.pc.delta_c
            
            part1 = q*nu**2 *(q*nu**2-3)/delta_c**2
            part2 = (1+2*p+2*(q*nu**2-1))*(2*p)/(delta_c**2 *(1+q*nu**(2*p)))
            return part1+part2
    
    #############################################################################################################

    class EulBias:
        """
        define Eulerian biases in terms of lagrangian biases
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz,A=1,alpha=0):
            return 1 + self.pc.lagbias.b1(zz)

        def b2(self,zz,A=1,alpha=0):
            return self.pc.lagbias.b2(zz) + (8/21)*self.pc.lagbias.b1(zz) 

        def b_01(self,zz,A=1,alpha=0):
            delta_c = self.pc.delta_c
            return A*(2*delta_c*self.pc.lagbias.b1(zz)+4*(self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)),np.log(self.pc.sig_R['0'](zz)))-1))*(self.pc.sig_R[str(alpha)](zz)/self.pc.sig_R['0'](zz))

        def b_11(self,zz,A=1,alpha=0):
            delta_c = self.pc.delta_c
            return A*(delta_c*(self.b2(zz)+(13/21)*(self.b1(zz)-1))+self.b1(zz)*(2*self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)),np.log(self.pc.sig_R['0'](zz)))-3)+1)*(self.pc.sig_R[str(alpha)](zz)/self.pc.sig_R['0'](zz))

    # get halo biases these will be arrays with repsect to M - for integration
    def dy_ov_dx(self,dy,dx):
        """
        Compute the derivative using spline derivatives.
        """
        # Fit cubic splines and caclulate their derivatives for each R coord
        dum1 = np.gradient(dy,self.R,axis=-1)
        dum2 = np.gradient(dx,self.R,axis=-1)

        return dum1 / dum2 # Compute the ratio of the derivatives
    
    #############################################################################################
    def n_h(self,zz): #dn/dM
        """
        define halo mass function - number density of halos per unit mass- Tinker2010
        Return array in (R,z)
        """
        #derivate of sigma wrt to M
        dSdM = np.gradient(np.sqrt(self.sig_R['0'](zz)),self.M(zz),axis=-1) # is 2D array R,z
        
        return (self.rho_m(zz)/self.M(zz))*self.multiplicity(zz)*np.abs(dSdM)/np.sqrt(self.sig_R['0'](zz))
    
    
    ################################################################################################################
    
    #define n_h mean number of halos and N(M,z) mean number of galaxies per halo
    def HOD(self, zz, M0, NHO):
        #define HoD from Yankelevich and porciani 2018: arXiv:1807.07076
        M = self.M(zz)

        theta_HoD = 1 + scipy.special.erf(2*np.log10(M/M0))
        N_c = np.exp(-10*(np.log10(M/M0))**2)+0.05*theta_HoD #central galaxies 
        N_s = 0.003*(M/M0)*theta_HoD

        return NHO*(N_c+N_s) # 2D (r,z)
    
    ###############################################################################################################

    #for the 3 different types of PNG
    class Loc:
        def __init__(self,parent):
            self.A = 1
            self.alpha = 0
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)

    class Eq:
        def __init__(self,parent):
            self.A = 3
            self.alpha = 2
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)

    class Orth:
        def __init__(self,parent):
            self.A = -3
            self.alpha = 1
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)
            
    def add_bias_attr(self,other_class):
        """
        Collect computed biases
        """
        other_class.n_g = self.n_g
        other_class.b_1 = self.b_1
        other_class.b_2 = self.b_2 
        other_class.g_2 = self.g_2
        
        #get PNG biases for each type
        other_class.loc = self.loc 
        other_class.eq = self.eq
        other_class.orth = self.orth 
        
        #also other useful info about HOD parameter fitting
        other_class.M0_func = self.M0_func
        other_class.NO_func = self.NO_func
        
#old
class PBBias1:
    def __init__(self,cosmo_funcs,survey_params,HMF='Tinker2010'):
        """
           Gets non-gaussian biases from Peak Background split approach - assumes HMF (Tinker 2010) and HOD (yankelevich and porciani 2018)
        """
        
        self.cosmo_funcs = cosmo_funcs #for later
        self.survey_params = survey_params
        delta_c = 1.686 #from spherical collapse
        self.delta_c = delta_c
        
        #so several parameters are 2 dimensional - dependent on R and z
        #therefore we interpolate over redshift here for convenience later where we call things as a fuction of z
        """
        #define R range 
        self.R = np.logspace(-2.5,2.5,300,dtype=np.float32) #upper limit - causes errors for large R 
        
        # M is the mass enclosed within the radius which also redshift dependent
        self.M_func = lambda xx: ((4*np.pi*cosmo_funcs.rho_m(xx)*self.R**3)/3)
        """

        # so instead lets use the hmf library
        #mf = MassFunction(hmf_model=HMF,z=0)
        #self.M = mf.m
        
        self.M = np.logspace(9,16,100,dtype=np.float64)#so M is in units of solar Mass
        
        self.R = ((3*self.M)/(4*np.pi*self.rho_m(0)))**(1/3) #[Mpc/h]
        
        #precompute sigma^2
        sigmaR0 = self.sigma_R_n(self.R,0,cosmo_funcs)
        sigmaR1 = self.sigma_R_n(self.R,-1,cosmo_funcs)
        sigmaR2 = self.sigma_R_n(self.R,-2,cosmo_funcs)
        
        #for sigma 
        sig_R = {}
        
        sig_R['0'] = lambda xx: sigmaR0 * cosmo_funcs.D(xx)**2 # (z,R)
        sig_R['1'] = lambda xx: sigmaR1 * cosmo_funcs.D(xx)**2
        sig_R['2'] = lambda xx: sigmaR2 * cosmo_funcs.D(xx)**2
        
        self.sig_R = sig_R
        
        #peak height (z,R)
        self.nu_func = lambda xx: delta_c/np.sqrt(sig_R['0'](xx))# interpolate along z
        
        #init classes 
        
        if HMF == 'ST':
            self.multiplicity = self.multiplicity_ST
            self.lagbias = self.LagBias_ST(self)
        elif HMF == 'Tinker10':
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias_Tinker10(self) #default is tinker2010 mass function
        else:
            self.multiplicity = self.multiplicity_Tinker10
            self.lagbias = self.LagBias(self) #default is tinker2010 mass function
            
        self.eulbias = self.EulBias(self)
        

        # so get fittted values of NO and M0
        z_arr = np.linspace(survey_params.z_range[0],survey_params.z_range[1],10)
        self.M0_func = self.fit_M0(z_arr)
        self.NO_func = self.fit_NO(z_arr)

        ###############################################################################################
        def get_galaxy_bias(b_h,A=1,alpha=0):
            """
            set z_range and fit cubic spline for a given bias function
            """
            z_samps = np.linspace(survey_params.z_range[0],survey_params.z_range[1],int(40))
            
            bias_arr = np.ones_like(z_samps)
            for i,zz in enumerate(z_samps):
            
                bias_arr[i] = self.general_galaxy_bias(b_h,zz,self.M0_func(zz),self.NO_func(zz),A,alpha)/survey_params.n_g(zz)
            
            return CubicSpline(z_samps,bias_arr)
        
        def get_number_density():
            """
            set z_range and fit cubic spline for a given bias function
            """
            z_samps = np.linspace(survey_params.z_range[0],survey_params.z_range[1],int(40))
            
            n_g = np.ones_like(z_samps)
            for i,zz in enumerate(z_samps):
            
                n_g[i] = self.number_density(zz,self.M0_func(zz),self.NO_func(zz))
            
            return CubicSpline(z_samps,n_g)
        
        self.get_galaxy_bias = get_galaxy_bias
        self.n_g = get_number_density()
        
        #################################################################################################
        # so save all required params to object  
        self.b_1 = get_galaxy_bias(self.eulbias.b1)
        self.b_2 = get_galaxy_bias(self.eulbias.b2)
        self.g_2 = lambda xx: -(2/7)*(self.b_1(xx)-1)#tidal bias - e.g. baldauf
        
        #get PNG biases for each type
        self.loc = self.Loc(self)
        self.eq = self.Eq(self)
        self.orth = self.Orth(self)
    
    #########################################################################################
    def number_density(self,zz,M0,NO): 
        """
        return number density for given z 
        """

        return np.trapz(self.HOD(zz,M0,NO)*self.n_h(zz), (self.M))

    # so implement Eq.
    def general_galaxy_bias(self,b_h,zz,M0,NO,A=1,alpha=0):
        """
        return bias for given z
        """

        # Integrate over M for each value of z
        integral_values = np.trapz(b_h(zz, A, alpha)*self.n_h(zz)*self.HOD(zz,M0,NO), (self.M))

        return integral_values
    ############################################################################################

    #fit NO and M0 to given n_g and b_1
    def fit_M0(self,z_arr):
        """
        fit M0 from linear bias (b_1 is independent of NO)
        """
        def objective(M0,zz,NO=1):
            """
            diff between linear bias from PBS and survey specifications
            """
            return self.general_galaxy_bias(self.eulbias.b1,zz,M0,NO)/self.number_density(zz,M0,NO) - self.survey_params.b_1(zz)

        M0_arr = np.array([scipy.optimize.newton(objective, x0=1e+12, args=(z,),rtol=1e-5) for i,z in enumerate(z_arr)])

        return CubicSpline(z_arr,M0_arr) # now returns M0 as function of redshift

    #now can find NO from n_g
    def fit_NO(self,z_arr):
        """
        fit NO from number density
        """
        def objective(NO,zz,M0):
            """
            diff between number density from PBS and survey specifications
            """  
            return self.number_density(zz,M0,NO) - self.survey_params.n_g(zz)

        # Use the secant method by not providing a derivative
        NO_arr = np.array([scipy.optimize.newton(objective, x0=2.0, args=(z,self.M0_func(z)),rtol=1e-5) for i,z in enumerate(z_arr)])

        return CubicSpline(z_arr,NO_arr) # now returns M0 as function of redshift
  
        
    #############################################################################################################
    
    def sigma_R_n(self, R, n,cosmo_funcs, K_MIN = 5e-5,K_MAX=5e+0,steps=int(1e+3)):
        """ Works -well in agreement with other codes...
        compute sigma^2 for a given radius and n i.e does interal over k

        Uses differential equation approach for the yinit
        """
        k = np.logspace(np.log10(K_MIN), np.log10(K_MAX), num=steps)
        pk_L = cosmo_funcs.get_class_powerspectrum(k)

        def deriv_sigma(x,y,k,Pk,R,n): # from Pylians
            Pkp=np.interp((x),(k),(Pk))
            kR=x*R
            W=3.0*(np.sin(kR)-kR*np.cos(kR))/kR**3 
            return [x**(2+n)*Pkp*W**2]

        #this function computes sigma(R)
        def sigma_sq(k,Pk,R,n):
            k_limits=[k[0],k[-1]]; yinit=[0.0]

            I = scipy.integrate.solve_ivp(deriv_sigma,k_limits,yinit,args=(k,Pk,R,n),method='RK45')

            return I.y[0][-1]#get last value

        sigma_squared = np.zeros((len(R)))
        for i,_ in enumerate(R):
            sigma_squared[i] = sigma_sq(k,pk_L,R[i],n)/(2.0*np.pi**2)

        return sigma_squared
    
    ##################################################################### halo mass function stuff 
    #so lets just get lagrangian bias parameters from the derivates of the multiplicity functions
    class LagBias:
        def __init__(self,parent_class):
            self.pc = parent_class
            
        def b1(self, zz):
            return self.get_bn(zz, 1)
        
        def b2(self, zz):
            return self.get_bn(zz, 2)
               
        def get_bn(self,zz,N):
            """
            Get numeric local-in-matter bias expansion
            """
            coef = (-self.pc.nu_func(zz))**N /(self.pc.delta_c**N * self.pc.multiplicity(zz))
            deriv = CubicSpline(self.pc.nu_func(zz), self.pc.multiplicity(zz)).derivative(nu=N)(self.pc.nu_func(zz))
            return coef*deriv
    
    
    #######################################################################
    def halo_bias_params(self,zz):#tinker 2010
        #define initial free paremeters from Tinker 2010 table 4, Delta=200
        #alpha0 = 0.368
        beta0 = 0.589
        gamma0 = 0.864
        eta0 = -0.243
        phi0 = -0.729

        #free_args0 = alpha,beta,gamma,eta,psi
        beta = beta0*(1+zz)**(0.2)
        gamma = gamma0*(1+zz)**(-0.01)
        eta = eta0*(1+zz)**(0.27)
        phi = phi0*(1+zz)**(-0.08)
        
        alpha = 1 / (2 ** (eta - phi - 0.5)* beta ** (-2 * phi)* gamma ** (-0.5 - eta)* (2**phi * beta ** (2 * phi) * sp.gamma(eta + 0.5)+ gamma**phi * sp.gamma(0.5 + eta - phi)))

        return alpha,beta,gamma,eta,phi
    
    #get langrangian and then eulerian biases
    #see Appendix C arXiv:1911.03964v3
    class LagBias_Tinker10:
        """
        define lagrangian biases in terms of z,M and the halo bias params - these are all arrays in (z,M)
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz):
            _,beta,gamma,eta,psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2*psi)/(delta_c*((beta*nu)**(2*psi)+1)) + (gamma*nu**2-2*eta-1)/delta_c

        def b2(self,zz):
            _,beta,gamma,eta,psi = self.pc.halo_bias_params(zz)
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            return (2*psi*(2*gamma*nu**2-4*eta+2*psi-1))/(delta_c**2 *((beta*nu)**(2*psi)+1)) + (gamma**2 *nu**4-4*gamma*eta*nu**2-3*gamma*nu**2+4*eta**2+2*eta)/delta_c**2
    
    #######################################  Which HMF: f(nu)
    def multiplicity_Tinker10(self,zz): # from Tinker 2008,2010
        alpha,beta,gamma,eta,psi = self.halo_bias_params(zz)
        nu = self.nu_func(zz)

        return nu*alpha*(1+(beta*nu)**(-2*psi))*nu**(2*eta)*np.exp(-gamma*nu**2 /2)

    def multiplicity_ST(self,zz): # ShethTormen
        A = 0.3221
        p = 0.3
        a = 0.707 # 1 #

        nu = self.nu_func(zz)

        part1 = A*(1+(a*nu**2)**(-p))
        part2 = nu*(2*a/(np.pi))**(1/2)*(np.exp(- a*nu**2 /2))
        return part1*part2
    
    ############################################################################
    
    class LagBias_ST: #Sheth-Tormen mass functions biases
        """
        define lagrangian biases in terms of z,M and the halo bias params - these are all arrays in (z,M)
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz):
            A = 0.322
            p = 0.3
            q = 0.707
            nu = self.pc.nu_func(zz)
            delta_c = self.pc.delta_c
            
            return (q*nu**2-1)/delta_c + 2*p/(delta_c*(1+(q*nu**2)**p))

        def b2(self,zz):
            A = 0.322
            p = 0.3
            q = 0.707
            nu = (self.pc.nu_func(zz))
            delta_c = self.pc.delta_c
            
            part1 = q*nu**2 *(q*nu**2-3)/delta_c**2
            part2 = (1+2*p+2*(q*nu**2-1))*(2*p)/(delta_c**2 *(1+q*nu**(2*p)))
            return part1+part2
    
    #############################################################################################################

    class EulBias:
        """
        define Eulerian biases in terms of lagrangian biases
        """
        def __init__(self,parent_class):
            self.pc = parent_class
        
        def b1(self,zz,A=1,alpha=0):
            return 1 + self.pc.lagbias.b1(zz)

        def b2(self,zz,A=1,alpha=0):
            return self.pc.lagbias.b2(zz) + (8/21)*self.pc.lagbias.b1(zz) 

        def b_01(self,zz,A=1,alpha=0):
            delta_c = self.pc.delta_c
            return A*(2*delta_c*self.pc.lagbias.b1(zz)+4*(self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)),np.log(self.pc.sig_R['0'](zz)))-1))*(self.pc.sig_R[str(alpha)](zz)/self.pc.sig_R['0'](zz))

        def b_11(self,zz,A=1,alpha=0):
            delta_c = self.pc.delta_c
            return A*(delta_c*(self.b2(zz)+(13/21)*(self.b1(zz)-1))+self.b1(zz)*(2*self.pc.dy_ov_dx(np.log(self.pc.sig_R[str(alpha)](zz)),np.log(self.pc.sig_R['0'](zz)))-3)+1)*(self.pc.sig_R[str(alpha)](zz)/self.pc.sig_R['0'](zz))

    # get halo biases these will be arrays with repsect to M - for integration
    def dy_ov_dx(self,dy,dx):
        """
        Compute the derivative using spline derivatives.
        """
        # Fit cubic splines and caclulate their derivatives for each R coord
        dum1 = CubicSpline(self.R, dy).derivative()(self.R)
        dum2 = CubicSpline(self.R, dx).derivative()(self.R)

        return dum1 / dum2 # Compute the ratio of the derivatives
    
    #############################################################################################
    def n_h(self,zz): #dn/dm
        """
        define halo mass function - number density of halos per unit mass- Tinker2010 
        """
        #derivate of sigma wrt to M
        dSdM = CubicSpline(self.M,np.sqrt(self.sig_R['0'](zz))).derivative()(self.M)
        
        #self.nu_func(zz)*
        return (self.rho_m(0)/self.M)*self.multiplicity(zz)*np.abs(dSdM)/np.sqrt(self.sig_R['0'](zz))
    
    
    ################################################################################################################
    
    #define n_h mean number of halos and N(M,z) mean number of galaxies per halo
    def HOD(self, zz, M0, NHO):
        #define HoD from Yankalevich and poricani 2018
        M = self.M

        theta_HoD = 1 + scipy.special.erf(2*np.log10(M/M0))
        N_c = np.exp(-10*(np.log10(M/M0))**2)+0.05*theta_HoD #central galaxies 
        N_s = 0.003*(M/M0)*theta_HoD

        return NHO*(N_c+N_s)
    
    ###############################################################################################################

    #for the 3 different types of PNG
    class Loc:
        def __init__(self,parent):
            self.A = 1
            self.alpha = 0
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)

    class Eq:
        def __init__(self,parent):
            self.A = 3
            self.alpha = 2
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)

    class Orth:
        def __init__(self,parent):
            self.A = -3
            self.alpha = 1
            self.b_01 = parent.get_galaxy_bias(parent.eulbias.b_01,A=self.A,alpha=self.alpha)
            self.b_11 = parent.get_galaxy_bias(parent.eulbias.b_11,A=self.A,alpha=self.alpha)
            
    def add_bias_attr(self,other_class):
        """
        Collect computed biases
        """
        other_class.n_g = self.n_g
        other_class.b_1 = self.b_1
        other_class.b_2 = self.b_2 
        other_class.g_2 = self.g_2
        
        #get PNG biases for each type
        other_class.loc = self.loc 
        other_class.eq = self.eq
        other_class.orth = self.orth 
        
        #also other useful info about HOD parameter fitting
        other_class.M0_func = self.M0_func
        other_class.NO_func = self.NO_func
                
