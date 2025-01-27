import numpy as np
from classy import Class

def get_cosmology(h = 0.6766,Omega_b = 0.02242,Omega_cdm = 0.11933,A_s = 2.105e-9,n_s = 0.9665,k_max=10):
    """ calls class for some set of parameters and returns the cosmology"""
    Omega_b *= h**2
    Omega_cdm *= h**2
    Omega_m = Omega_cdm+Omega_b

    params = {'output':'mPk,mTk',
                 'non linear':'halofit',
                 'Omega_b':Omega_b,
                 'Omega_cdm':Omega_cdm,#Omega_m-Omega_b,#
                 'h':h,
                 'n_s':n_s,
                 'A_s':A_s,#'n_s':n_s,'sigma8':0.828,#
                 'P_k_max_1/Mpc':k_max,
                 'z_max_pk':10. #Default value is 10
    }

    #Initialize the cosmology and compute everything
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo

# these two could be moved out of ClassWAP
def get_theta(self,k1,k2,k3):
    """
    get theta for given triangle - being careful with rounding
    """
    cos_theta = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
    cos_theta = np.where(np.isclose(np.abs(cos_theta), 1), np.sign(cos_theta), cos_theta)
    return np.arccos(cos_theta)

def get_k3(self,theta,k1,k2):
    return np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(theta))

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

###############################################  old stuff ######################################################


def get_avg_dist(obs_pos):
    """create d to integrate over - lets say it's a 1000 MPc/h (128x128x128) grid situated at (x,y,z)"""
    Nside_theory= 128
    conf_space = np.linspace(0,1000,Nside_theory)
    x_unorm , y_unorm , z_unorm = np.meshgrid(conf_space-obs_pos[0], conf_space-obs_pos[1], conf_space-obs_pos[2],indexing='ij') 
    conf_norm = np.sqrt(x_unorm**2 + y_unorm**2 + z_unorm**2) # make a unit vector - normalise
    ds = np.where(conf_norm==0,1,conf_norm)
    return 1/np.mean(1/ds**3)


def index_tuple(tup,index_cmd):
    """index the array in a tuple"""
    indexed_list = []
    for i,item in enumerate(tup):
        if np.array(item).size>1:
            indexed_list.append(item[index_cmd])
        else:
            indexed_list.append(item)
    return indexed_list  


def tuple_bool(tup,bool_arr):
    """boolean the array in a tuple - when the array are n-dimensional with last two dimensions to 1 """
    indexed_list = []
    for i,item in enumerate(tup):
        if np.array(item).size>1:

            indexed_list.append(item.flatten()[bool_arr][...,None,None])
        else:
            indexed_list.append(item)
    return indexed_list
