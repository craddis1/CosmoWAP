import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Define the path to the data file relative to the current script location
import os
module_dir = os.path.dirname(os.path.abspath(__file__))
SKAO1Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO1Data.txt'))
SKAO2Data = np.loadtxt(os.path.join(module_dir, 'data_library/SKAO2Data.txt'))

class SurveyParams():
    def __init__(self,cosmo):
        """
            gets specifications for some saved surveys (namely linear bias and number density as well as evolution and
            mangification biases
        """

        Euclid = {'b_1':  lambda xx: 0.9 + 0.4*xx,
                    'f_sky': 15000/41253,
                    'z_survey': [0.9,1.8], #get zmin and zmax
                    'be_survey':lambda xx: -7.29 + 0.470*xx + 1.17*xx**2 - 0.290*xx**3,#euclid_data[:,2]
                    'Q_survey': lambda xx: 0.583 + 2.02*xx - 0.568*xx**2 + 0.0411*xx**3,
                    'n_g': lambda zz: 0.0193*zz**(-0.0282) *np.exp(-2.81*zz)}

        #just need D(z)
        baLCDM = cosmo.get_background()
        D_cl = baLCDM['gr.fac. D']
        z_cl = baLCDM['z']
        D_intp = interp1d(z_cl,D_cl,kind='cubic')
        
        BGS =  {'b_1':  lambda xx: 1.34/D_intp(xx),
                'f_sky': 15000/41253,
                'z_survey': [0.05,0.6],
                'be_survey':lambda xx:  -2.25 - 4.02*xx + 0.318*xx**2 - 14.6*xx**3,
                'Q_survey': lambda xx: 0.282 + 2.36*xx + 2.27*xx**2 + 11.1*xx**3,
                'n_g':  lambda zz: 0.023*zz**(-0.471)*np.exp(-5.17*zz)-0.002} #fitting from Maartens


        SKAO1= {'b_1': lambda xx: 0.616*np.exp(1.017*xx),
                'f_sky': 5000/41253,
                'z_survey': [SKAO1Data[:,0][0],SKAO1Data[:,0][-1]],
                'be_survey': interp1d(SKAO1Data[:,0], SKAO1Data[:,4]),
                'Q_survey': interp1d(SKAO1Data[:,0],  SKAO1Data[:,3]),
                'n_g': interp1d(SKAO1Data[:,0], SKAO1Data[:,2])} #fitting from Maartens

        SKAO2= {'b_1': lambda xx: 0.554*np.exp(0.783*xx),
                'f_sky': 30000/41253,
                'z_survey':  [SKAO2Data[:,0][0],SKAO2Data[:,0][-1]],
                'be_survey': interp1d(SKAO2Data[:,0], SKAO2Data[:,4]),
                'Q_survey': interp1d(SKAO2Data[:,0],  SKAO2Data[:,3]),
                'n_g': interp1d(SKAO2Data[:,0], SKAO2Data[:,2])} #fitting from Maartens
        
        DM = {'b_1': lambda xx: 1 + 0*xx ,   # for dark matter particles 
              'b_2': lambda xx:  0*xx ,
              'g_2': lambda xx:  0*xx ,
              'f_sky': 1,
              'z_survey': [0,3],
              'be_survey':lambda xx:  0*xx ,
              'Q_survey': lambda xx: 0*xx,
              'n_g':  lambda zz: 1e+5 + 0*xx}
        
        self.survey_dict= {'Euclid':Euclid,
                             'BGS':BGS,
                             'SKAO1':SKAO1,
                             'SKAO2':SKAO2,
                             'DM':DM}
        
        ################################################################################
        
    def get_surveybias(self,survey):
        
        return self.survey_dict[survey]
    
class SurveyFunctions:
    """ read in survey specific params from dict - if empty use default values """
    def __init__(self,survey_params,all_bias=True):
        # set evolution and magnification bias
        if 'be_survey' in survey_params.keys():
            self.be_survey = survey_params['be_survey']
        else:
            self.be_survey  =  lambda xx: 0*xx

        if 'Q_survey' in survey_params.keys():
            self.Q_survey = survey_params['Q_survey']
        else:
            self.Q_survey =  lambda xx: 0*xx #=2/5

        if all_bias:
            # define other survey specific params not set by HOD and HMF
            if 'n_g' in survey_params.keys():
                self.n_g = survey_params['n_g']
            else:
                self.n_g  =  lambda xx: 1e+5 + 0*xx

            #define default bias stuff
            if 'b_1' in survey_params:
                self.b_1 = survey_params['b_1']
            else:
                self.b_1  =  lambda xx: np.sqrt(1+xx)

            if 'be_survey' in survey_params.keys():
                self.be_survey = survey_params['be_survey']
            else:
                self.be_survey  =  lambda xx: 0*xx

            if 'Q_survey' in survey_params.keys():
                self.Q_survey = survey_params['Q_survey']
            else:
                self.Q_survey =  lambda xx: 0*xx #=2/5

            #define second order biases from b_1 # or can define them in dict as above 

            if 'g_2' in survey_params.keys():
                self.g_2 = survey_params['g_2']
            else:
                self.g_2 = lambda xx: -(4/7)*(self.b_1(xx)-1) # 0.524-0.547*self.b_1(xx)+0.046*self.b_1(xx)**2#

            if 'b_2' in survey_params.keys():
                self.b_2 = survey_params['b_2']
            else:    
                self.b_2 = lambda xx: 0.412 - 2.143*self.b_1(xx) +0.929*self.b_1(xx)**2 + 0.008*self.b_1(xx)**3 + 4/3 * self.g_2(xx) 
            
        ###########################################################
        #for non tracer things  
        if 'f_sky' in survey_params.keys():
            self.f_sky = survey_params['f_sky']
        else:
            self.f_sky  =  1   

        if 'z_survey' in survey_params.keys():
            self.z_survey = np.linspace(survey_params['z_survey'][0],survey_params['z_survey'][1],int(1e+5))  
        else:
            self.z_survey  = np.linspace(0,3,int(1e+5))
