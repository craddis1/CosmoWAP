import numpy as np
from classy import Class
import matplotlib.pyplot as plt
import copy


"""
#scoccimarro and noorikuhani cosmology
Omega_m = 0.32
sigma8 = 0.828
n_s = 0.968
"""

def get_cosmo(h = 0.6766,Omega_m = 0.30964144,Omega_b = 0.02242,A_s = 2.105e-9,n_s = 0.9665,k_max=10,sigma8=None):
    #h = 0.6766,Omega_b = 0.02242,Omega_cdm = 0.11933,A_s = 2.105e-9,n_s = 0.9665,k_max=10,sigma8=None
    """ calls class for some set of parameters and returns the cosmology - base cosmology is planck 2018"""
    Omega_b *= 1/h**2
    #Omega_cdm *= 1/h**2
    #Omega_m = Omega_cdm+Omega_b

    #Create a params dictionary
    params = {'output':'mPk,mTk',
                 'non linear':'halofit',
                 'Omega_b':Omega_b,
                 'Omega_cdm': Omega_m-Omega_b,#Omega_cdm
                 'h':h,
                 'n_s':n_s,
                 'A_s':A_s,#'n_s':n_s,#
                 'P_k_max_1/Mpc':k_max,
                 'z_max_pk':10. #Default value is 10
    }
    if sigma8 !=None:# if sigma8 define with sigma8 not A_s
        params = {'output':'mPk,mTk',
                 'non linear':'halofit',
                 'Omega_b':Omega_b,
                 'Omega_cdm':Omega_m-Omega_b,#Omega_cdm,#
                 'h':h,
                 'n_s':n_s,
                 'sigma8':sigma8,#'A_s':A_s,'sigma8':0.828,
                 'P_k_max_1/Mpc':k_max,
                 'z_max_pk':10. #Default value is 10
                 }
        

    #Initialize the cosmology and compute everything
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    return cosmo
###################################################

# useful for defining the triangle (just cosine rule)
def get_theta(k1,k2,k3):
    """
    get theta for given triangle - being careful with rounding
    """
    cos_theta = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    cos_theta = np.where(np.isclose(np.abs(cos_theta), 1), np.sign(cos_theta), cos_theta)
    return np.arccos(cos_theta)

def get_k3(theta,k1,k2):
    """
    get k3 for given triangle
    """
    k3 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(theta))
    return np.where(k3==0,1e-4,k3)

def get_theta_k3(k1,k2,k3,theta):
    if theta is None:
        if k3 is None:
            raise  ValueError('Define either theta or k3')
        else:
            theta = get_theta(k1,k2,k3) #from utils
    else:
        if k3 is None:
            k3 = get_k3(theta,k1,k2)
    return k3, theta

def enable_broadcasting(*args,n=2):
    """Make last n axes size 1 if arrays, to allow numpy broadcasting"""
    result = []
    
    for var in args:
        if isinstance(var, np.ndarray):
            # Create a tuple of n trailing None dimensions
            new_axes = (None,) * n
            result.append(var[(...,) + new_axes])
        else:
            result.append(var)
           
    return tuple(result)

#################################################################### Misc
def create_copy(self):
    """
    Create a deep copy of the object, preserving the cosmo reference
    (cosmo is not deep copied as it's a cythonized classy object)
    """
    # Create empty object of same type
    new_self = self.__class__.__new__(self.__class__)
    
    # Copy everything except cosmo with deep copy
    new_self.__dict__ = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != 'cosmo'}
    
    # Add back cosmo reference (shallow copy) if it exists
    if hasattr(self, 'cosmo'):
        new_self.cosmo = self.cosmo
    
    return new_self

###############################################################################
#for plotting
def flat_bool(arr,slice_=None):#make flat and impose condtion k1>k2>k3
    if slice_ == None:
        return np.abs(arr).flatten()[tri_bool.flatten()]
    else:
        return np.abs(arr[slice_].flatten()[tri_bool[slice_].flatten()])
    
#plots over all triangles     
def plot_all(ks,ymin=0,ymax=4,ax=None):
    k1,k2,k3 = np.meshgrid(ks,ks,ks,indexing='ij')

    #get theta from triagle condition - this create warnings from non-closed triangles
    # Handle cases where floating-point errors might cause cos_theta to go slightly beyond [-1, 1]
    cos_theta = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    cos_theta = np.where(np.isclose(np.abs(cos_theta), 1), np.sign(cos_theta), cos_theta) #need to get rid of rounding errors

    #if we want to suprress warnings from invalid values we can reset cos(theta) these terms are ignored anyway
    cos_theta = np.where(np.logical_or(cos_theta < -1, cos_theta > 1), 0, cos_theta)
    theta = np.arccos(cos_theta)

    #create bool for closed traingles with k1>k2>k3...
    tri_bool = np.full_like(k1,False).astype(np.bool_)
    fake_k = np.zeros_like(k1)# also create fake k- to plot over as in we define k1 then allow space to fill for all triangles there
    mesh_index = np.zeros_like(k1)
    tri_num = 0
    
    for i in range(k1.shape[0]+1):
        if i == 0:
            continue
        for j in range(i+1):#enforce k1>k2
            if j == 0:
                continue
            for k in range(i-j,j+1):# enforce k2>k3 and triangle condition |k1-k2|<k3
                if k==0:
                    continue

                #for indexing
                ii = i-1
                jj = j-1
                kk = k-1
                
                #print(ii,jj,kk)
                tri_bool[ii,jj,kk] = True
                mesh_index[ii,jj,kk] = tri_num
                tri_num += 1
                fake_k[ii,ii,ii] = 1
                
    
    if ax==None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    def flat_bool(arr):#make flat and impose condtion k1>k2>k3
        return np.abs(arr.flatten()[tri_bool.flatten()])

    for i in range(len(flat_bool(fake_k))):
        if flat_bool(fake_k)[i]>0:
            ax.vlines(i+1,ymin,ymax,linestyles='--',color='grey',alpha=0.6)

    def thin_xticks(arr,split=7,frac=3):#there are too many xticks at the start
        return np.concatenate((arr[:split][::frac],arr[split:]))
    
    tri_index = np.arange(len(flat_bool(k1)))
    index_ticks = tri_index[flat_bool(fake_k)>0]+1#+1 as get where k1 steps not the equalateral before...
    ticks = ks#_bin
    _ = plt.xticks(thin_xticks(index_ticks)[1:], [round(i, 3) for i in thin_xticks(ticks)[1:]])
    
    #ax.set_yscale('log')
    #ax.set_ylim(ks[0],ks[-1])
    ax.set_ylim(ymin,ymax)
    
    ax.set_xlim(0,tri_index[-1])
    ax.set_xlabel('$k_1$ [h/Mpc]')
    
    #np.array([flat_bool(k1/bin_width),flat_bool(k2/bin_width),flat_bool(k3/bin_width)]).astype(np.int_).T
    
    #plot where k2 steps..
    for i in range(mesh_index.shape[0]):
        for j in range(mesh_index.shape[0]):
            ax.vlines(mesh_index[i,j,j]+1,ymin,ymax,linestyles='--',color='grey',alpha=0.2)
    
    if False:
        #so lets find and shade squeezed limit
        is_squeeze = np.zeros_like(mesh_index)
        for i in range(mesh_index.shape[0]):
            for j in range(i+1):
                for k in range(i-j-1,j+1):
                    if k < 0:
                        continue
                    if i+1 > 3*(k+1) and j+1 > 3*(k+1):
                        is_squeeze[i,j,k] = 1e+8

        ax.fill_between(flat_bool(mesh_index), ymin,ymax, where=flat_bool(is_squeeze), color='gray', alpha=0.2)

    return flat_bool(k1), flat_bool(k2), flat_bool(k3),flat_bool(theta),mesh_index,tri_bool

