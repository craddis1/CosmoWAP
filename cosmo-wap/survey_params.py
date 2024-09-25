import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import odeint

#from useful_funcs import *

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

        """ model1
            z_euclid = euclid_data[:,0]
            self.be_survey = interp1d(z_euclid, euclid_data[:,1])
            self.Q_survey = interp1d(z_euclid,  euclid_data[:,2])
        """
        import os
        module_dir = os.path.dirname(os.path.abspath(__file__))
        SKAO1Data = np.loadtxt(os.path.join(module_dir, 'SKAO1Data.txt'))
        SKAO2Data = np.loadtxt(os.path.join(module_dir, 'SKAO2Data.txt'))

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
