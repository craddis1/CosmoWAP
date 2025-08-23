import numpy as np
import scipy.integrate as integrate
import cosmo_wap as cw
from cosmo_wap.lib import utils

class HaLuminosityFunction:
    """ Parent class of H-alpha luminosity function e.g. Euclid, Roman
    Works with schechter type luminosity functions where:
    Φ(z, y) = φ∗(z) g(y) where y ≡ L/L∗ 

    φ∗(z), g(y) are defined in the child classes for a specific luminosity function

    Here these surveys can detect a minimum flux (F_c) 
    
    See: arXiv:2107.13401 for an overview"""
    
    def luminosity_function(self, L, zz):
        """
        Schechter luminosity function values for  given luminosity L and redshift zz

        Φ(z, y) = φ∗(z) g(y) where y ≡ L/L∗ 
        
        Parameters:
        -----------
        L : array or float
            Luminosity of source
        zz: array or float
            Redshift

        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]   
        """

        # make sure mm and zz are 2D arrays for broadcasting
        if isinstance(zz, (np.ndarray)) and isinstance(L, (np.ndarray)) and zz.size != L.size:
            zz = zz[:,np.newaxis]
            L = L[np.newaxis,:]
        
        # get y L/L*
        y = self.get_y(L,zz)

        return self.get_phi_star(zz) * self.g(y)
    
    def L_c(self,F_c,zz):
        """
        Convert flux [erg cm^−2 s^−1] to luminosity [erg s^−1] at redshift z
        """
        convert_cm_to_mpc = 3.0857e+24 # Mpc in cm
        return F_c * (1 + zz)**2 * 4 * np.pi * self.cosmo.comoving_distance(zz)**2 * convert_cm_to_mpc**2

    def number_density(self, F_c, zz):
        """ 
        Calculate the number density of H-alpha emitters for a given flux cut F_c and redshift zz

        n_g(z,F_c) = φ∗(z) G(F_c,z) where G(y) = ∫_0^y g(y') dy'
        
        Parameters:
        -----------
        F_c : float
            Flux cut [erg cm^−2 s^−1]

        zz : float or array
            Redshift
            
        Returns:
        --------
        Number density : float or array
            Total number density [h^3 Mpc^-3]
        """

        return self.get_phi_star(zz)*self.get_G(F_c,zz)
    
    def get_G(self,F_c,zz):
        """
        G(y) = ∫_0^y g(y') dy'
        """
        # so this is 2D array 1st dimension is redshift, 2nd is luminosity
        L = np.zeros((len(zz),1000))
        for i in range(len(zz)):
            L[i] = np.logspace(np.log10(self.L_c(F_c,zz[i])), 47, 1000) # integrate over luminosity with a given cut
        
        zz = zz[:,np.newaxis] # make sure zz is 2D for broadcasting

        y = self.get_y(L,zz)

        return integrate.simpson(self.g(y), y ,axis=-1)
    
    def get_Q(self,F_c,zz):
        """
        Q(z, Mc) = 
        """
        L_c = self.L_c(F_c,zz)
        y_c = self.get_y(L_c,zz)

        return y_c * self.g(y_c) / self.get_G(F_c,zz)
    
    def get_be(self,F_c,zz):

        # change in number density
        d_ln_ng_dln = np.gradient(np.log(self.number_density(F_c,zz)),np.log(1+zz))

        terms = 2*(1 + (1+zz)/(self.cosmo.Hubble(zz)*self.cosmo.comoving_distance(zz)))*self.get_Q(F_c,zz)
        return  - d_ln_ng_dln - terms
    
    def b_1(self,x,zz):
        a = 0.844
        b = 0.116
        c = 42.623
        d = 1.186
        e = 1.765

        return a+b*(1+zz)**e *(1+np.exp((x-c)*d))
    
    def get_b_1(self,F_c,zz):
        """Semi-anlaytic model with free parameters from table 2 in 1909.12069
        (∫_x^inf \phi(x) b_1(x) dx)/(∫_x^inf \phi(x) dx)
        Returns linear bias as an array in redshift above a given flux cut
        """
        # so this is 2D array 1st dimension is redshift, 2nd is luminosity
        x = np.zeros((len(zz),100))

        integrand = np.zeros((len(zz),100))
        ng_integrand = np.zeros((len(zz),100))
        for i in range(len(zz)): # loop over z - for each z we have different cut in luminosity
            x_arr = np.linspace(np.log10(self.L_c(F_c,zz[i])), 47, 100) # integrate over luminosity with a given cut
            x[i] = x_arr
            lf = self.luminosity_function(10**x_arr, zz[i])
            b1 = self.b_1(x_arr,zz[i])
            integrand[i] = lf*b1
            ng_integrand[i] = lf

        return integrate.simpson(integrand, x ,axis=-1)/integrate.simpson(ng_integrand, x ,axis=-1) # integrate over luminosities above flux cut

class Model1LuminosityFunction(HaLuminosityFunction):
    def __init__(self, cosmo=None):
        """
        H-alpha Luminosity Function calculator
        Luminsotiy in Units h^3 Mpc^-3
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7,2,1000)
    
    def g(self,y):
        alpha = - 1.35
        return y**alpha * np.exp(-y)
    
    def get_phi_star(self,zz):
    
        def phi_star_phi_star0(zz):
            """
            L* as a function of redshift
            """
            eta = 1
            z_b = 1.3
            return np.where(zz < z_b,(1+zz)**eta, (1+z_b)**(2*eta) * (1 + zz)**(-eta))
    
        # all redshift dependent
        phi_star0 = 10**(-2.8)/self.cosmo.h()**3 # phi* at z=0 in h^3 Mpc^-3
        phi_star = phi_star_phi_star0(zz)*phi_star0
        return phi_star

    def get_y(self,L,zz):
        """ Calculate y = L/L* for given luminosity and redshift """
        L0_star = 10 ** (41.5)  # L* at z=0 in erg/s
        delta = 2

        L_star = L0_star * (1 + zz)**delta  # L*
        return L/L_star
    
class Model3LuminosityFunction(HaLuminosityFunction):
    def __init__(self, cosmo=None):
        """
        H-alpha Luminosity Function calculator
        Luminsotiy in Units h^3 Mpc^-3
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.7,2,1000)
    
    def g(self,y):
        alpha = - 1.587
        nu = 2.288
        return y**alpha /(1 + (np.e-1)* y**nu)
    
    def get_phi_star(self,zz):
    
        return 10**(-2.92)/self.cosmo.h()**3 # phi* at z=0 in h^3 Mpc^-3

    def get_y(self,L,zz):
        """ Calculate y = L/L* for given luminosity and redshift """
        L_star_inf = 10**(42.956)
        L_star_half = 10**(41.733)
        beta = 1.615
        log_L_star = np.log10(L_star_inf) + (1.5/(1+zz))**beta * np.log10(L_star_half/L_star_inf)

        return L/(10**log_L_star)
    
########################################################################## apparent magnitude limited surveys

class KCorrectionLuminosityFunction:
    """ 
    Parent class for K-corrected luminosity functions e.g. BGS, Megamapper
    If a survey measures galaxy fluxes in fixed wavelength bands, this leads to a K-correction
    for the redshifting effect on the bands. In that case, it is standard to work in terms of
    dimensionless magnitudes.

    Here these surveys can detect objects above a minimum apparent magnitude (m_c) which is likes to the threshold absolute magnitude:

    M_c(z) = m_c − 5 log[ dL(z)/10 pc] − K(z)

    Works with schechter type luminosity functions where:
    Φ(z, y) = φ∗(z) g(y) where y ≡ M - M*(z)

    φ∗(z), g(y) are defined in the child classes for a specific luminosity function
    
    See: arXiv:2107.13401 for an overview
    """
    def M_UV(self, mm, zz):
        """
        Convert apparent to absolute UV magnitude (Equation 2.6)
        """

        if zz is None:
            zz = self.z_values
            
        if isinstance(mm, (np.ndarray)):
            mm = mm[:,np.newaxis]

        # Luminosity distance in pc
        D_L = 1e6 * self.cosmo.luminosity_distance(zz) #*self.cosmo.get_current_derived_parameters(['h'])['h']
        
        # Distance modulus
        distance_modulus = 5 * np.log10(D_L / 10.0)
        
        # K-correction
        k_correction = self.K(zz)

        # Equation 2.6
        M_UV = mm - distance_modulus - k_correction
        
        return M_UV
    
    def number_density(self, m_cut, zz=None):
        """
        Calculate the number density for k corrected survey for given apparent magnitude cut
        
        Parameters:
        -----------
        m_cut : float
            Apparent magnitude cut
        zz : float or array
            redshift
            
        Returns:
        --------
        Number density : float or array
            Total number density [h^3 Mpc^-3]
        """

        if zz is None:
            zz = self.z_values
        
        mm = np.linspace(15, m_cut, 1000) # apparent magntiude values to integrate over
        luminosity_arr = self.luminosity_function(mm ,zz)

        return integrate.simpson(luminosity_arr, self.M_UV(mm,zz),axis=0)
    
    def get_Q(self,m_c,zz=None):
        """
        Q(z, Mc) = (5/(2 * ln(10))) * (Φ(z, Mc) / ng(z, Mc))
        """
        return (5/(2*np.log(10)))*self.luminosity_function(m_c,zz)/self.number_density(m_c,zz)
    
    def get_Q2(self,m_c,zz=None):
        """
        Q(z, Mc) = (5/2) * (∂ log10 ng(z, Mc) / ∂Mc)

        Not used but in agreement with above definition of Q
        """

        h_m = 0.01
        # change in number density
        deriv = np.log10(self.number_density(m_c+h_m,zz))-np.log10(self.number_density(m_c-h_m,zz))

        # so the change in M_c
        h_M = self.M_UV(m_c+h_m,zz)-self.M_UV(m_c,zz)

        d_log10_ng_dMc = (deriv)/(2*h_M)
        return (5/2)*d_log10_ng_dMc
    
    def get_be(self,m_c,zz=None):

        if zz is None:
            zz = self.z_values

        # change in number density
        d_ln_ng_dln = np.gradient(np.log(self.number_density(m_c,zz)),np.log(1+zz))

        terms = 2*(1 + (1+zz)/(self.cosmo.Hubble(zz)*self.cosmo.comoving_distance(zz)) + 2 *np.log(10)/(5) *np.gradient(self.K(zz),np.log(1+zz)))*self.get_Q(m_c,zz)

        return  - d_ln_ng_dln - terms
    

class LBGLuminosityFunction(KCorrectionLuminosityFunction):
    def __init__(self, cosmo=None):
        """
        Lyman Break Galaxy Luminosity Function calculator (MegaMapper)
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        
        # Here we have luminosity function parameters fitted for some given redshifts
        # Redshifts and parameters from Table 3 [1904.13378] 
        self.z_values = np.array([2.0, 3.0, 3.8, 4.9, 5.9])
        self.M_star = np.array([-20.60, -20.86, -20.63, -20.96, -20.91])
        self.phi_star = np.array([9.70e-3, 5.04e-3, 9.25e-3, 3.22e-3, 1.64e-3])  # h^-3 Mpc^3
        self.alpha = np.array([-1.60, -1.78, -1.57, -1.60, -1.87])
    
    def luminosity_function(self, mm, zz=None):
        """
        Schechter luminosity function values for given apparent magnitude and fitted params
        
        Parameters:
        -----------
        m_obs : float
            Observed apparent magnitude

        zz : float or array
            Redshift
            
        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]   
        """
        # Convert to absolute magnitude at this redshift    
        M_UV = self.M_UV(mm,self.z_values)

        if isinstance(mm, (np.ndarray)):
            mm = mm[:,np.newaxis]
        
        # Calculate Schechter function (Equation 2.5)
        L_Lstar = 10**(-0.4 * (M_UV - self.M_star))
            
        phi_values= ((np.log(10) / 2.5) * self.phi_star * 
                        10**(-0.4 * (1 + self.alpha) * (M_UV - self.M_star)) * 
                        np.exp(-L_Lstar))
            
        return phi_values
    
    def K(self, zz):
        """
        K-correction for LBGs
        """
        return -2.5 * np.log10(1 + zz)
    
    def b_1(self,mm,zz=None):
        """From 1904.13378 - magnitude and redshift dependent biass"""
        if isinstance(mm, (np.ndarray)):
            mm = mm[:,np.newaxis]
            
        A = -0.98*(mm-25) + 0.11 # from Eq.(2.7) 1904.13378v2
        B = 0.12*(mm-25) + 0.17
        return A*(1+zz) + B *(1+zz)**2
    
    def get_b_1(self,m_c,zz):
        """(∫_x^inf \phi(x) b_1(x) dx)/(∫_x^inf \phi(x) dx)
        Returns linear bias as an array in redshift above a given flux cut
        """

        mm = np.linspace(15, m_c, 1000) # apparent magntiude values to integrate over
        luminosity_arr = self.luminosity_function(mm ,zz) # so array m,zz
        bias_arr = self.b_1(mm,zz)

        # integrate over apparent magnitudes for a given cut
        return integrate.simpson(luminosity_arr*bias_arr, self.M_UV(mm,zz),axis=0)/integrate.simpson(luminosity_arr, self.M_UV(mm,zz),axis=0)
    
class BGSLuminosityFunction(KCorrectionLuminosityFunction):
    def __init__(self, cosmo=None):
        """
        BGS Luminosity function class
        """
        self.cosmo = cosmo if cosmo is not None else cw.lib.utils.get_cosmo()
        self.z_values = np.linspace(0.01,0.7,1000)
    
    def luminosity_function(self, mm, zz):
        """
        Schechter luminosity function values for given apparent magnitude and fitted params
        
        Parameters:
        -----------
        m_obs : float or array
            Observed apparent magnitude

        zz : float or array
            Redshift
            
        Returns:
        --------
        Luminosity Function : float or array
            Luminosity [h^3 Mpc^-3]   
        """
        # Convert to absolute magnitude at this redshift
        M_UV = self.M_UV(mm,zz)

        # make sure mm and zz are 2D arrays for broadcasting
        if isinstance(zz, (np.ndarray)) and isinstance(mm, (np.ndarray)):
            mm = mm[:,np.newaxis]
        
        alpha = - 1.23
        M_star =  5*np.log10(self.cosmo.h()) - 20.64 - 0.6 * zz
        y = M_UV - M_star
        
        g = (np.log(10) / 2.5)*10**(-0.4 * (1 + alpha)*y)* np.exp(-10**(-0.4 *y))

        phi_star = 10**(-2.022+0.92*zz)
        phi = phi_star*g
        return phi
    
    def K(self, zz):
        """
        K-correction
        """
        return 0.87*zz