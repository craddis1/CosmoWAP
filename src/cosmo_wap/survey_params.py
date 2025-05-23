import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from cosmo_wap.lib import utils

# Define the path to the data file relative to the current script location
import os
module_dir = os.path.dirname(os.path.abspath(__file__))
SKAO1Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO1Data.txt'))
SKAO2Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO2Data.txt'))

class SurveyParams():
    def __init__(self,cosmo=None):
        """
            Initialize and get survey parameters for some set surveys
        """
        self.Euclid = self.Euclid()
        self.SKAO1 = self.SKAO1()
        self.SKAO2 = self.SKAO2()
        self.DM_part = self.DM_part()
        self.CV_limit = self.GenSurvey()
        
        if cosmo is not None:
            self.BGS = self.BGS(cosmo)
    
    #ok want to inherit this function to update variables - could use dataclasses
    class SurveyBase:
        def update(self, **kwargs):
            """update survey class parameters"""
            new_self = utils.create_copy(self)
        
            for key, value in kwargs.items():
                if hasattr(new_self, key):
                    setattr(new_self, key, value)
            return new_self
        
        def modify_func(self, func_name, modifier):
            """Apply a modifier function to an existing function"""
            new_self = utils.create_copy(self)
            
            old_func = getattr(new_self, func_name)
            setattr(new_self, func_name, lambda xx, f=old_func: modifier(f(xx)))
            return new_self
        
    class Euclid(SurveyBase):
        def __init__(self):
            self.b_1       = lambda xx: 0.9 + 0.4*xx
            self.z_range   = [0.9,1.8] #get zmin and zmax
            self.be_survey = lambda xx: -7.29 + 0.470*xx + 1.17*xx**2 - 0.290*xx**3 #euclid_data[:,2]
            self.Q_survey  = lambda xx: 0.583 + 2.02*xx - 0.568*xx**2 + 0.0411*xx**3
            self.n_g       = lambda zz: 0.0193*zz**(-0.0282) *np.exp(-2.81*zz)
            self.f_sky     = 15000/41253
        
    class BGS(SurveyBase):
        def __init__(self,cosmo):
            #just need D(z)
            baLCDM = cosmo.get_background()
            D_cl = baLCDM['gr.fac. D']
            z_cl = baLCDM['z']
            D_intp = interp1d(z_cl,D_cl,kind='cubic')

            self.b_1       = lambda xx: 1.34/D_intp(xx)
            self.z_range   = [0.05,0.6]
            self.be_survey = lambda xx:  -2.25 - 4.02*xx + 0.318*xx**2 - 14.6*xx**3
            self.Q_survey  = lambda xx: 0.282 + 2.36*xx + 2.27*xx**2 + 11.1*xx**3
            self.n_g       = lambda zz: 0.023*zz**(-0.471)*np.exp(-5.17*zz)-0.002 #fitting from Maartens
            self.f_sky     = 15000/41253

    class SKAO1(SurveyBase):
        def __init__(self):
            self.b_1       = lambda xx: 0.616*np.exp(1.017*xx)
            self.z_range  = [SKAO1Data[:,0][0],SKAO1Data[:,0][-1]]
            self.be_survey = interp1d(SKAO1Data[:,0], SKAO1Data[:,4])
            self.Q_survey  = interp1d(SKAO1Data[:,0],  SKAO1Data[:,3])
            self.n_g       = interp1d(SKAO1Data[:,0], SKAO1Data[:,2]) #fitting from Maartens
            self.f_sky     = 5000/41253
    
    class SKAO2(SurveyBase):
        def __init__(self):
            self.b_1       = lambda xx: 0.554*np.exp(0.783*xx)
            self.z_range  =  [SKAO2Data[:,0][0],SKAO2Data[:,0][-1]]
            self.be_survey = interp1d(SKAO2Data[:,0], SKAO2Data[:,4])
            self.Q_survey  = interp1d(SKAO2Data[:,0],  SKAO2Data[:,3])
            self.n_g       = interp1d(SKAO2Data[:,0], SKAO2Data[:,2]) #fitting from Maartens
            self.f_sky     = 30000/41253
     
    class DM_part(SurveyBase):
        def __init__(self):
            self.b_1       = lambda xx: 1 + 0*xx   # for dark matter particles 
            self.b_2       = lambda xx:  0*xx
            self.g_2       = lambda xx:  0*xx
            self.z_range   = [0.01,5]
            self.be_survey = lambda xx:  0*xx 
            self.Q_survey  = lambda xx: 0*xx
            self.n_g       = lambda xx: 1e+5 + 0*xx
            self.f_sky     = 1
            
    class GenSurvey(SurveyBase): # create cosmic limited general spectroscopic survey
        def __init__(self):
            self.b_1       = lambda xx: 0.9 + 0.4*xx   # 1.45 + 0.68*(zz-1) # np.sqrt(1+zz) 
            self.z_range   = [0.01,3]
            self.be_survey = lambda xx: 1 
            self.Q_survey  = lambda xx: 2/5
            self.n_g       = lambda xx: 1e+5 + 0*xx
            self.f_sky     = 1
    
    class InitNew(SurveyBase):#for adding new surveys...
        def __init__(self):
            self.b_1       = lambda xx: 1 + 0*xx   # for dark matter particles 
            self.z_range   = [0.01,5]
            self.be_survey = lambda xx:  0*xx 
            self.Q_survey  = lambda xx: 0*xx
            self.n_g       = lambda xx: 1e+5 + 0*xx
            self.f_sky     = 1
            
    ################################################################################
        

class SetSurveyFunctions:
    """ read in survey specific params and calculate higehr order ones if unprovided - if empty use default values """
    def __init__(self,survey_params,compute_bias=False):
        # set evolution and magnification bias
        if hasattr(survey_params, 'be_survey'):
            self.be_survey = survey_params.be_survey
        else:
            self.be_survey  =  lambda xx: 0*xx

        if hasattr(survey_params, 'Q_survey'):
            self.Q_survey = survey_params.Q_survey
        else:
            self.Q_survey =  lambda xx: 0*xx #=2/5

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
