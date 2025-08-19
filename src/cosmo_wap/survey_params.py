import numpy as np
from cosmo_wap.lib import utils
from cosmo_wap.lib.luminosity_funcs import *
from scipy.interpolate import CubicSpline

# Define the path to the data file relative to the current script location
import os
module_dir = os.path.dirname(os.path.abspath(__file__))
SKAO1Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO1Data.txt'))
SKAO2Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO2Data.txt'))

class SurveyParams():
    def get(self,cosmo,survey):
        """
        Initialize and get survey parameters for some set surveys
        """
        return getattr(self,survey)(cosmo)
    
    #ok want to inherit this function to update variables - could use dataclasses
    class SurveyBase:
        def update(self, **kwargs):
            """update survey class parameters-
            Input dictionary with parameters you want to change."""
            new_self = utils.create_copy(self)
        
            for key, value in kwargs.items():
                if hasattr(new_self, key):
                    setattr(new_self, key, value)
            return new_self
        
        def compute_luminosity(self,LF,cut,zz):
            """Get biases from given luminosity function and magnitude/luminosity cut
            
            LF: Luminosity function class
            cut: Magnitude/flux cut
            zz: redshift
            Returns:
            Obeject with Q,be and n_g defined"""

            self.Q   = CubicSpline(zz,LF.get_Q(cut,zz))
            self.be  = CubicSpline(zz,LF.get_be(cut,zz))
            self.n_g = CubicSpline(zz,LF.number_density(cut,zz))
            if hasattr(LF,'get_b_1'): # then also get linear bias from fits in Table. 2 1909.12069
                self.b_1 = CubicSpline(zz,LF.get_b_1(cut,zz))
            return self
        
        def BF_split(self,split):
            """
            Split is magnitude or flux cut at which we seperate our samples
            Call from survey class with a luminosity function.
            """
            if not hasattr(self,'LF'):
                raise ValueError("No luminosity function - use survey with defined luminosity function!")
            
            self.bright = utils.create_copy(self) # bright
            self.faint = utils.create_copy(self) # faint
            """
            We already have total sample biases.
            n_F = n_T - n_B
            bF = n_T b_T − n_B b_B /nF
            Q_F = n_T/(n_T - n_B) Q_T - n_B/(n_T - n_B)*Q_B
            be_F = d ln(n_T - n_B)/ d ln (1 + z)
            """
            zz= self.zz
            # define total survey biases
            Q_T = self.Q(zz)
            n_T = self.n_g(zz)

            Q_B = self.LF.get_Q(split,zz)
            be_B = self.LF.get_be(split,zz)
            n_B = self.LF.number_density(split,zz)

            self.bright.Q   = CubicSpline(zz,Q_B)
            self.bright.be  = CubicSpline(zz,be_B)
            self.bright.n_g = CubicSpline(zz,n_B)

            # so for faint
            self.faint.Q    = CubicSpline(zz,n_T /(n_T - n_B)*Q_T - n_B /(n_T - n_B)*Q_B)
            self.faint.be   = CubicSpline(zz,np.gradient(np.log(n_T-n_B),np.log(1+zz)))
            self.faint.n_g  = CubicSpline(zz,n_T - n_B)

            # then for linear bias if we can use semi-analytical fit from 1909.12069
            if hasattr(self.LF,'get_b_1'):
                b_T = self.b_1(zz)
                b_B = self.LF.get_b_1(split,zz)
                self.bright.b_1   = CubicSpline(zz,b_B)
                self.faint.b_1  = CubicSpline(zz,(n_T*b_T-n_B*b_B)/(n_T - n_B))

            return [self.bright,self.faint] # two tracers defined
        
    class Euclid(SurveyBase):
        def __init__(self,cosmo,fitting=False, model3=True,F_c=None):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 0.9 + 0.4*xx
            self.f_sky   = 15000/41253
            self.z_range = [0.9,1.8] #get zmin and zmax

            if fitting:
                self.be = lambda xx: -7.29 + 0.470*xx + 1.17*xx**2 - 0.290*xx**3 #euclid_data[:,2]
                self.Q  = lambda xx: 0.583 + 2.02*xx - 0.568*xx**2 + 0.0411*xx**3
                self.n_g = lambda zz: 0.0193*zz**(-0.0282) *np.exp(-2.81*zz)
            else:
                self.zz = np.linspace(self.z_range[0],self.z_range[1],100)
                #from lumnosity function
                if model3:
                    if F_c is None: # set defualt values - this one agrees with fitting functions above
                        F_c = 2e-16               
                    self.LF = Model3LuminosityFunction(cosmo)
                else:
                    if F_c is None:
                        F_c = 3e-16
                    self.LF = Model1LuminosityFunction(cosmo) 
                self = self.compute_luminosity(self.LF,F_c,self.zz)

    class Roman(SurveyBase):
        def __init__(self,cosmo, model3=False,F_c=None):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 0.9 + 0.4*xx
            self.f_sky   = 2000/41253
            self.z_range = [0.5,2.0] #get zmin and zmax
            self.zz = np.linspace(self.z_range[0],self.z_range[1],100)

            #from lumnosity function
            if model3:
                if F_c is None: # set defualt values
                    F_c = 1e-16              
                self.LF = Model3LuminosityFunction(cosmo)
            else:
                if F_c is None:
                    F_c = 1e-16
                self.LF = Model1LuminosityFunction(cosmo) 
            self = self.compute_luminosity(self.LF,F_c,self.zz)
        
    class BGS(SurveyBase):
        def __init__(self,cosmo,m_c=20,fitting=False):
            self.cosmo = cosmo 
            self.b_1      = lambda xx: 1.34/cosmo.scale_independent_growth_factor(xx)
            self.z_range  = [0.05,0.6]
            self.f_sky    = 15000/41253

            if fitting:
                self.be   = lambda xx:  -2.25 - 4.02*xx + 0.318*xx**2 - 14.6*xx**3
                self.Q    = lambda xx: 0.282 + 2.36*xx + 2.27*xx**2 + 11.1*xx**3
                self.n_g  = lambda zz: 0.023*zz**(-0.471)*np.exp(-5.17*zz)-0.002 #fitting from Maartens
            else:
                #from lumnosity function
                self.zz = np.linspace(self.z_range[0],self.z_range[1],100)
                self.LF = BGSLuminosityFunction(cosmo)
                self = self.compute_luminosity(self.LF,m_c,self.zz)

    class MegaMapper(SurveyBase):
        def __init__(self,cosmo,m_c=24.5):
            self.cosmo = cosmo 
            self.A        = -0.98*(m_c-25) + 0.11 # from Eq.(2.7) 1904.13378v2
            self.B        = 0.12*(m_c-25) + 0.17
            self.b_1      = lambda xx: self.A*(1+xx) + self.B *(1+xx)**2 # so linear bias for given apparent magnitude limit
            self.z_range  = [2.1,5] #get zmin and zmax
            self.f_sky    = 20000/41253

            #from lumnosity function
            self.LF = LBGLuminosityFunction(cosmo)
            self.zz = self.LF.z_values
            self = self.compute_luminosity(self.LF,m_c,self.zz)
            
    class SKAO1(SurveyBase):
        def __init__(self,cosmo):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 0.616*np.exp(1.017*xx)
            self.z_range = [SKAO1Data[:,0][0],SKAO1Data[:,0][-1]]
            self.be      = CubicSpline(SKAO1Data[:,0], SKAO1Data[:,4])
            self.Q       = CubicSpline(SKAO1Data[:,0],  SKAO1Data[:,3])
            self.n_g     = CubicSpline(SKAO1Data[:,0], SKAO1Data[:,2]) #fitting from Maartens
            self.f_sky   = 5000/41253
    
    class SKAO2(SurveyBase):
        def __init__(self,cosmo):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 0.554*np.exp(0.783*xx)
            self.z_range =  [SKAO2Data[:,0][0],SKAO2Data[:,0][-1]]
            self.be      = CubicSpline(SKAO2Data[:,0], SKAO2Data[:,4])
            self.Q       = CubicSpline(SKAO2Data[:,0],  SKAO2Data[:,3])
            self.n_g     = CubicSpline(SKAO2Data[:,0], SKAO2Data[:,2]) #fitting from Maartens
            self.f_sky   = 30000/41253
     
    class DM_part(SurveyBase):
        def __init__(self,cosmo):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 1 + 0*xx   # for dark matter particles 
            self.b_2     = lambda xx:  0*xx
            self.g_2     = lambda xx:  0*xx
            self.z_range = [0.01,5]
            self.be      = lambda xx:  0*xx 
            self.Q       = lambda xx: 0*xx
            self.n_g     = lambda xx: 1e+5 + 0*xx
            self.f_sky   = 1
    
    class InitNew(SurveyBase):#for adding new surveys... # so could add some sort of dict unpacking method?
        def __init__(self,cosmo):
            self.cosmo = cosmo 
            self.b_1     = lambda xx: 1 + 0*xx   # for dark matter particles 
            self.z_range = [0.01,5]
            self.be      = lambda xx:  0*xx 
            self.Q       = lambda xx: 0*xx
            self.n_g     = lambda xx: 1e+5 + 0*xx
            self.f_sky   = 1
            
################################################################################
        
class SetSurveyFunctions:
    """ read in survey specific params and calculate higehr order ones if unprovided - if empty use default values 
    This is called once for survey and also again for survey1 if multi-tracer"""
    def __init__(self,survey_params,compute_bias=False):
        # set evolution and magnification bias
        if hasattr(survey_params, 'be'):
            self.be = survey_params.be
        else:
            self.be  =  lambda xx: 0*xx

        if hasattr(survey_params, 'Q'):
            self.Q = survey_params.Q
        else:
            self.Q =  lambda xx: 0*xx #=2/5

        if not compute_bias:
            # define other survey specific params not set by HOD and HMF
            if hasattr(survey_params, 'n_g'):
                self.n_g = survey_params.n_g
            else:
                self.n_g  =  lambda xx: 1e+5 + 0*xx

            #define default bias stuff
            if hasattr(survey_params, 'b_1'):
                self.b_1 = survey_params.b_1
            else:
                self.b_1  =  lambda xx: np.sqrt(1+xx)

            #define second order biases from b_1 if undefined in survey class

            if hasattr(survey_params, 'g_2'):
                self.g_2 = survey_params.g_2
            else:
                self.g_2 = lambda xx: -(2/7)*(self.b_1(xx)-1)#lambda xx: 0.524-0.547*self.b_1(xx)+0.046*self.b_1(xx)**2# # 

            if hasattr(survey_params, 'b_2'):
                self.b_2 = survey_params.b_2
            else:
                # 0.3 - 0.79*b1 + 0.2 * b2**2 + 0.12/b1
                # -0.741 - 0.125 z + 0.123 z**2 + 0.00673 z**3
                self.b_2 = lambda xx: 0.412 - 2.143*self.b_1(xx) +0.929*self.b_1(xx)**2 + 0.008*self.b_1(xx)**3 + 4/3 * self.g_2(xx)

            
            #for the 3 different types of PNG
            class Loc:
                def __init__(self,parent):
                    #also compute scale dependent biases for local type PNG
                    delta_c = 1.686
                    bL10 = lambda xx: parent.b_1(xx) - 1
                    bL20 = lambda xx: parent.b_2(xx) - (8/21)*bL10(xx)

                    bL01 = lambda xx: 2*delta_c*bL10(xx)
                    bL11 = lambda xx: 2*(delta_c*bL20(xx)-bL10(xx))
                    #bL02 = lambda xx: 4* * delta_c*(delta_c*bL20(xx)-2*bL10(xx))

                    self.b_01 = bL01
                    self.b_11 = lambda xx: bL01(xx) + bL11(xx)
                    #self.b_02 = bL02
             
            self.loc = Loc(self)
            
        ###########################################################
        #for non tracer things  
        if hasattr(survey_params, 'f_sky'):
            self.f_sky = survey_params.f_sky
        else:
            self.f_sky  =  1   

        if hasattr(survey_params, 'z_range'):
            self.z_range = survey_params.z_range
        else:
            self.z_range  = np.linspace(0,5,int(1e+5))
