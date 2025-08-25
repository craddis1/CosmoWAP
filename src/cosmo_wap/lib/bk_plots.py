"""Plotting function for the bispectrum which i find useful - perhaps a good starting point."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm

import cosmo_wap.bk as bk

# use colourblind firednly colours
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

############################################
#for plotting
def flat_bool(arr,slice_=None):#make flat and impose condtion k1>k2>k3
    if slice_ == None:
        return np.abs(arr).flatten()[tri_bool.flatten()]
    else:
        return np.abs(arr[slice_].flatten()[tri_bool[slice_].flatten()])
    
#plots over all triangles     
def plot_all(ks,ymin=1e-3,ymax=1,shade_squeeze=False,ax=None,log=True):
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
        _, ax = plt.subplots(figsize=(12, 6))
        
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
    

    ax.set_ylim(ymin,ymax)
    ax.set_xlim(0,tri_index[-1])
    ax.set_xlabel('$k_1$ [h/Mpc]')
    if log:
        ax.set_yscale('log')

    #plot where k2 steps..
    for i in range(mesh_index.shape[0]):
        for j in range(mesh_index.shape[0]):
            ax.vlines(mesh_index[i,j,j]+1,ymin,ymax,linestyles='--',color='grey',alpha=0.2)
    
    if shade_squeeze:
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

################## triangle plots ################################

def create_mask():
    # Create a mask for the triangular region
    size = int(1e+3)
    x = np.linspace(0.0, 1, size,dtype=np.float32)
    y = np.linspace(0.5, 1,  size,dtype=np.float32)
    xx,yy = np.meshgrid(x,y)
    return np.where(np.logical_and((yy > 1 - xx), (yy>xx)),np.nan,1)

def triangle_plot(term,l,cosmo_funcs, zz=1, k1=0.05,r=0,s=0,size=500):
    """Get array of bk over k2,k3 in shape (size,size)"""
    
    x = np.linspace(0.01, 1, size) # fraction ok k1
    y = np.linspace(0.495, 1, size)

    xx,yy = np.meshgrid(x,y,indexing='ij')#create meshgrid where xx= k3/k1, yy= k2/k1
    
    #array version:
    k3 = xx*k1
    k2 = yy*k1
        
    which_k2 = np.logical_and((yy > 1 - xx), (yy>xx))#restrict to closed triangles
    bk_tmp = bk.bk_func(term,l,cosmo_funcs,k1,k2,k3,zz=zz)
    return np.where(which_k2,bk_tmp,1)

def plot_triangle(term,l,cosmo_funcs, zz=1, k1=0.05,r=0,s=0,norm=False,vmax=None,vmin=None,log=True,size=500): #plot triangle with mask!
    """Traingle plot over shape for some contribution for a given scale"""
    
    bk = triangle_plot(term,l,cosmo_funcs, zz=zz, k1=k1,r=r,s=s,size=size) # get array for triangle
    if norm:
        norm_bk = triangle_plot('NPP',l,cosmo_funcs, zz=zz, k1=k1,r=r,s=s,size=size) # get normalization
    else:
        norm_bk = 1
    
    plt.figure(figsize=(12,4))
    # Create the colormap plot
    if not vmax:
        vmax = np.max(np.abs(bk/norm_bk))
        if log:
            vmin = 1e-4*np.max(np.abs(bk/norm_bk)[np.abs(bk/norm_bk)>0])
        else: 
            vmin = np.min(bk/norm_bk)
            
    if not log:
        im = plt.imshow(((bk/norm_bk).T), extent=[0, 1, 0.5, 1], interpolation='bilinear', origin='lower', cmap='RdBu',vmin=-vmin,vmax=vmax)#
    else:
        im = plt.imshow(np.abs((bk/norm_bk).T), extent=[0, 1, 0.5, 1], interpolation='bilinear', origin='lower', cmap='Spectral',norm=mpl.colors.LogNorm(vmin, vmax=vmax))#,vmin=0,vmax=vmax)#

    cbar = plt.colorbar(im)
    if norm:
        cbar.set_label(fr"$B^{{\rm {term}}}_{{ \ell={l} }}(k_1={k1})/B^{{\rm NPP}}_{{ \ell=0 }}(k_1={k1})$")
    else:
        cbar.set_label(fr"$B^{{\rm {term}}}_{{ \ell={l} }}(k_1={k1})$")
    
    # Add labels and title
    plt.xlabel(r'$k_3/k_1$')
    plt.ylabel(r'$k_2/k_1$')

    #so masking 
    mask = create_mask()
    im = plt.imshow(mask, extent=[0, 1, 0.5, 1], interpolation='bilinear', origin='lower', cmap='binary')
    
    plt.grid(ls='--',lw=0.75,color='k',alpha=0.1)
    
    x_bound = [0,0.495,1]
    y_bound = [1,0.50,1]#
    #plt.text(0.06,0.55,'$r=s=1/3$')
    plt.plot(x_bound,y_bound,'k',linewidth=3)
    plt.show()

def plot_triangle_multi(term_list,l,cosmo_funcs, zz=1, k1=0.05,r=0,s=0,norm=False,vmax=None,vmin=None,size=500,log=True): #plot triangle with mask!   
    """Similiar to plot traingle but create Nx1 subplot"""
    N = len(term_list)
    bks = []
    for term in term_list:
        bks.append(triangle_plot(term,l,cosmo_funcs, zz=zz, k1=k1,r=r,s=s,size=size)) # get array for triangle

    if norm:
        norm_bk = triangle_plot('NPP',l,cosmo_funcs, zz=zz, k1=k1,r=r,s=s,size=size) # get normalization
    else:
        norm_bk = 1
        
    mask = create_mask() # get mask
    
    if not vmax: # set limits
        vmax = np.max(np.abs(np.array(bks)/norm_bk))
    if not vmin:
        if log:
            vmin = 1e-4*np.max(np.abs(np.array(bks)/norm_bk)[np.abs(np.array(bks)/norm_bk)>0])
        else: 
            vmin = np.min(np.array(bks)/norm_bk)
        
    fig = plt.figure(figsize=(14, 6))

    # Define gridspec with 1 rows and 3 columns
    gs = GridSpec(1, N)

    # Create subplots with custom aspect ratios
    axs = [fig.add_subplot(gs[0, i]) for i in range(N)]  # Subplots in the first row

    #fig, axs = plt.subplots(1, 3,figsize=(14,5),sharey=True)
    fig.subplots_adjust(wspace=0)
    # Create the colormap plot
    for i in range(N):
        if log:
            im = axs[i].imshow(np.abs((bks[i]/norm_bk).T),aspect=1.5, extent=[0, 1, 0.5, 1], interpolation='bilinear', origin='lower', cmap='Spectral',norm=mpl.colors.LogNorm(vmin, vmax=vmax))
            axs[i].imshow(mask, extent=[0, 1, 0.5, 1], aspect=1.5, interpolation='bilinear', origin='lower', cmap='binary')
        else:
            im = axs[i].imshow((bks[i]/norm_bk).T,aspect=1.5, extent=[0, 1, 0.5, 1], interpolation='bilinear', origin='lower', cmap='RdBu',vmin=vmin,vmax=vmax)
            axs[i].imshow(mask, extent=[0, 1, 0.5, 1], aspect=1.5, interpolation='bilinear', origin='lower', cmap='binary')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.3, 0.01, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$ |B_{(1,0)}(k_1=0.05)|/B^{\rm N}_{\ell=0}$")
    x_bound = [0,0.4965,1]
    y_bound = [1,0.50,1]#

    for i in range(N):       
        axs[i].plot(x_bound,y_bound,'k',linewidth=3)
        #axs[i].text(0.08,0.55,r'$\boldsymbol{d}=\boldsymbol{x}_%d$'%(i+1),fontsize=20)
        axs[i].set_xlabel('$k_3/k_1$')
        axs[i].set_xticks(np.arange(0.1, 0.91, 0.2))
        
        if i!=0:
            axs[i].yaxis.set_major_locator(plt.NullLocator())
    
    if False:
        def triple_triangle_coords(base_x=0.4,base_y=0.5):
            triangle_coords = {
                0: np.array([[base_x+0.05, base_y +0.1-0.3*np.sin(np.pi/3)], [base_x+0.2, base_y+0.1], [base_x+0.35, base_y +0.1-0.3*np.sin(np.pi/3)]]),  # Equilateral triangle
                1: np.array([[base_x, base_y-0.05], [base_x+0.2, base_y+0.05], [base_x+0.4, base_y-0.05]]),  # Folded triangle
                2: np.array([[base_x, base_y-0.05], [base_x+0.4, base_y+0.05], [base_x+0.4, base_y-0.05]])   # Squeezed triangle
            }
            return triangle_coords

        #triangle_labels = {0: 'Equilateral', 1: 'Folded', 2: 'Squeezed'}

        pos_x = [0.8,0.4,0.4]
        pos_y = [0.8,0.4,0.4]
        # Plot the triangles on the additional axis
        for i in range(1):

            triangle = Polygon(triple_triangle_coords(pos_x[i],pos_y[i])[i], closed=True, fill=None, edgecolor='black')
            axs[0].add_patch(triangle)
            #axs[i].text(base_x-0.1, base_y+0.15, triangle_labels[i])
    
    for i,term in enumerate(term_list):
        axs[i].text(0.08,0.55,f'{term}',fontsize=20)
    
    return fig, axs


################################## plots over r,s LOS choice in position space ###################################

def triple_triangle_coords(base_x=0.4,base_y=0.5):
    """Repersent traingles in diagram"""
    triangle_coords = {
        0: np.array([[base_x+0.05, base_y +0.1-0.3*np.sin(np.pi/3)], [base_x+0.2, base_y+0.1], [base_x+0.35, base_y +0.1-0.3*np.sin(np.pi/3)]]),  # Equilateral triangle
        1: np.array([[base_x, base_y-0.05], [base_x+0.2, base_y+0.05], [base_x+0.4, base_y-0.05]]),  # Folded triangle
        2: np.array([[base_x, base_y-0.05], [base_x+0.4, base_y+0.05], [base_x+0.4, base_y-0.05]])   # Squeezed triangle
    }
    return triangle_coords

def create_mask_rs():
    size = int(1e+3)
    r = np.linspace(0.0, 1, size,dtype=np.float32)
    s = np.linspace(0.0, 1,  size,dtype=np.float32)
    rr,ss = np.meshgrid(r,s)
    return np.where((rr + ss <= 1),np.nan,1)

#lets plot as a function of r and s
def rs_plot(term,l,cosmo_funcs, zz, size= 500):
    """Get array in r,s for 3 different shapes"""
    
    r = np.linspace(0, 1, size)
    s = np.linspace(0, 1, size)
    rr,ss = np.meshgrid(r,s,indexing='ij')
    bk_tmp = np.zeros((3,size,size),dtype=np.complex64)
    
    for type_tri in tqdm(range(3)):
        #set triangle
        if type_tri ==0: #equilateral
            k1 = 0.01
            k2 = k3 = k1
        elif type_tri==1: # folded
            k1 = 0.02
            k3 = k2 = k1/2
        else: #squeezed
            k1 = k2 = 0.05
            k3 = 0.005

        bk_tmp[type_tri] = bk.bk_func(term,l,cosmo_funcs,k1,k2,k3,zz=zz,r=rr,s=ss)
        which_rs = (rr + ss <= 1)#restrict to r+s \leq 1
        bk_tmp[type_tri] = np.where(which_rs,bk_tmp[type_tri],0)
        
    return bk_tmp

def plot_rs_multi(term,l,cosmo_funcs, zz, size= 500,vmax=None,vmin=None,log=True):
    """Plot Eq,folded and equalteral triangles over r,s"""
    bks = rs_plot(term,l,cosmo_funcs, zz, size= size)
    fig = plt.figure(figsize=(12,5))
    #mask = create_mask_rs() # get mask - not used currently

    # Define gridspec with 1 rows and 3 columns
    gs = GridSpec(1, 3, width_ratios=[3, 3, 3])

    # Create subplots with custom aspect ratios
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]  # Subplots in the first row

    fig.subplots_adjust(wspace=0)
    # Create the colormap plot
    for i in range(3):
        if log:
            im = axs[i].imshow(np.abs((bks[i]).T),aspect=1, extent=[0, 1, 0, 1], interpolation='bilinear', origin='lower', cmap='Spectral',norm=mpl.colors.LogNorm(vmin, vmax=vmax))#,vmin=0,vmax=vmax)#
        else:
            im = axs[i].imshow(((bks[i]).T),aspect=1, extent=[0, 1, 0, 1], interpolation='bilinear', origin='lower', cmap='RdBu',vmin=-vmax,vmax=vmax)#,vmin=0,vmax=vmax)#
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)#,ticks=[0.01, 0.1, 1]
    #cbar.set_label(r"$ |B^{\rm wa}_{\ell=1}(k_1=0.05)|/B^{\rm pp}_{\ell=0}$")#"$ |B_{\ell=1}(k_1=0.05)|/B_0 $"
    cbar.set_label(r"$ |B^{\rm WA/GR}_{(0,0)}|/B^{\rm N}_{(0,0)}$")#"$ |B_{\ell=1}(k_1=0.05)|/B_0 $"
    
    x_bound = [0,1]
    y_bound = [1,0]

    #triangle_labels = {0: 'Equilateral', 1: 'Folded', 2: 'Squeezed'}
    
    base_x,base_y = 0.55,0.85
    # Plot the triangles on the additional axis
    for i in range(3):
        triangle = Polygon(triple_triangle_coords(base_x,base_y)[i], closed=True, fill=None, edgecolor='black')
        axs[i].add_patch(triangle)
        #axs[i].text(base_x-0.1, base_y+0.15, triangle_labels[i])
        
    axs[0].set_ylabel('$s$')
    for i in range(3):
        
        axs[i].plot(x_bound,y_bound,'k',linewidth=3)
        #axs[i].text(0.08,0.55,r'$\boldsymbol{d}=\boldsymbol{x}_%d$'%(i+1),fontsize=20)
        axs[i].set_xlabel('$r$')
        axs[i].set_xticks(np.arange(0.2, 0.8, 0.2))
        axs[i].plot(1/3,1/3,'x',markersize=10,color='black')
        #axs[i].set_box_aspect(aspect=0.8)
        #axs[i].set_xlim(0,1)
        #axs[i].set_ylim(0.5,1)
        #axs[i].grid(ls='--',lw=0.75,color='k',alpha=0.1)
        
        if i!=0:
            axs[i].yaxis.set_major_locator(plt.NullLocator())
            
    axs[0].set_yticks(np.arange(0, 1.01, 0.2))
    
    return fig, axs


###### now just some paper plots##################

def plot_3x2(surveys,first=True):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
    
    cosmo = cw.utils.get_cosmo() 
    survey_params = cw.SurveyParams()
    cosmo_funcs = cw.ClassWAP(cosmo) # no survey

    # Create subplots with custom aspect ratios
    #axs = [fig.add_subplot(gs[0, i], aspect='10') for i in range(2)]  # Subplots in the first row

    #horizonatally
    for i in range(2):
        axs[0,i].get_xaxis().set_tick_params(which='both', size=0, labelsize=0)
        axs[1,i].get_xaxis().set_tick_params(which='both', size=0, labelsize=0)

    #vertically
    for i in range(3):
        axs[i,0].set_xticks(np.arange(0.1, 0.59, 0.2))
        axs[i,1].set_xticks(np.arange(0.9, 1.81, 0.3))

        #set triangle
        if i ==0: #
            k1 = 0.01
            k2 = k3 = k1
            theta = np.arccos((k3**2 - k1**2 - k2**2)/(2*k1*k2))
        elif i==1: # folded
            k1 = 0.02
            k2 = k3 = 0.01
            theta = np.arccos((k3**2 - k1**2 - k2**2)/(2*k1*k2))
        else: #squeezed
            k1 = k2 = 0.05
            k3 = 0.005
            theta = np.arccos((k3**2 - k1**2 - k2**2)/(2*k1*k2))

        for j in range(2):

            cosmo_funcs = cosmo_funcs.update_survey(getattr(survey_params,surveys[j])(cosmo))
            axs[0,j].text(0.25+0.9*j,1.1,surveys[j],fontsize=20)
            
            zz = cosmo_funcs.z_survey
            norm = bk.NPP.l0(cosmo_funcs,k1,k2,k3,theta,zz)
            
            #plot different contributions
            if first:#for first order stuff
                r=0;s=0
                axs[i,j].set_ylim(1.0001e-3,.9)
                axs[i,j].plot(zz,np.abs(bk.GR1.l1(cosmo_funcs,k1,k2,k3,theta,zz)/norm),'-',linewidth=3,label='GR',color=colors[1])
                axs[i,j].plot(zz,np.abs(bk.WA1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',linewidth=2.25,label='WA',color=colors[0])
                axs[i,j].plot(zz,np.abs(bk.RR1.l1(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',linewidth=1.5,label='RR',color=colors[2])
                axs[1,0].set_ylabel('$|B_{(1,0)}|/B^N_{(0,0)}$')
                axs[0,1].legend(ncol=3,frameon=False,title_fontsize=12,handlelength=1,columnspacing=1,handleheight=0.5, borderpad=0)

            else:#for second order stuff
                r=s=1/3
                axs[i,j].set_ylim(5e-4,.9)
                if i ==0 and j ==0:
                    axs[i,j].plot(zz,np.abs(bk.GR2.l0(cosmo_funcs,k1,k2,k3,theta,zz)/norm),'-',label='GR',color=colors[1])

                    axs[i,j].plot(zz,np.abs(bk.WA2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',label='WA',color=colors[0])
                    axs[i,j].plot(zz,np.abs(bk.RR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',label='RR',color=colors[2])
                    axs[i,j].plot(zz,np.abs(bk.WARR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'--',label='WA/RR',color=colors[3])

                    axs[i,j].plot(zz,np.abs(bk.WAGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-.',label='WA/GR',color=colors[4])
                    axs[i,j].plot(zz,np.abs(bk.RRGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-.',label='RR/GR',color=colors[5])
                else:
                    axs[i,j].plot(zz,np.abs(bk.GR2.l0(cosmo_funcs,k1,k2,k3,theta,zz)/norm),'-',color=colors[2])

                    axs[i,j].plot(zz,np.abs(bk.WA2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',color=colors[0])
                    axs[i,j].plot(zz,np.abs(bk.RR2.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-',color=colors[1])
                    axs[i,j].plot(zz,np.abs(bk.WARR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'--',color=colors[3])

                    axs[i,j].plot(zz,np.abs(bk.WAGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-.',color=colors[4])
                    axs[i,j].plot(zz,np.abs(bk.RRGR.l0(cosmo_funcs,k1,k2,k3,theta,zz,r,s)/norm),'-.',color=colors[5])

                axs[1,0].set_ylabel('$|\Delta B_{(0,0)}|/B^N_{(0,0)}$')
                fig.legend(loc=[0.05,0.95],ncol=6,frameon=False,title_fontsize=12,handlelength=1, columnspacing=1, handleheight=0.5, borderpad=0)
            
            #we plot scale where it breaks down
            chi = 2*np.pi/k3
            axs[i,0].vlines(cosmo_funcs.d_to_z(chi),0,1e+10,linestyles='--',color='black',alpha=1)
            
            axs[i,j].set_yscale('log')
            
            axs[i, j].grid(ls='--', lw=0.75, color='k', alpha=0.2)
            
            axs[2,j].set_xlabel('$z$')
                        
            axs[i,1].get_yaxis().set_tick_params(which='both', size=0, labelsize=0)
    
    triangle_labels = {0: 'Equilateral', 1: 'Folded', 2: 'Squeezed'}
    
    base_x,base_y = 0.4,0.5
    # Plot the triangles on the additional axis
    for i in range(3):
        
        # Create an additional axis for the triangles
        ax_triangles = fig.add_axes([0.8, 0.6-0.25*i, 0.3, 0.3], frameon=False)
        ax_triangles.set_xticks([])
        ax_triangles.set_yticks([])
    
        triangle = Polygon(triple_triangle_coords(0.4,0.5)[i], closed=True, fill=None, edgecolor='black')
        ax_triangles.add_patch(triangle)
        ax_triangles.text(base_x, base_y+0.15, triangle_labels[i])

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    
