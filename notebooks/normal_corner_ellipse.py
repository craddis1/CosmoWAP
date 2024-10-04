from itertools import combinations
from scipy.stats import multivariate_normal
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

def normal_corner(covm,mean,varlabels,fixedvarindex=None,fixedvarvalue=None,
           covm2=[None],mean2=None,scale_factor=3,diagnostic=False,
                  color='red',color2='blue', figsize=(12,6), **fig_kw):
    
    """Taken for normal contour - https://github.com/bvgoncharov/normal_corner/blob/master/normal_corner/normal_corner.py
    and edited the initial subplot creation"""
    
    """ 
    indexing now works like | plot_id |- for 3x3 matrix - 1 D variance not included - we index from the origin (sorry)
    
    2   | 5 |
    1   | 3 | 4 |
    0   | 0 | 1 | 2 |
       
          3   2   1
    
    So we create
    """

    # Plotting 2D contour distributions
    N = len(mean)
    
    fig = plt.figure(figsize=figsize)    
    
    # Define gridspec with N-1 rows and N-1 columns
    N_grid = N-1
    gs = GridSpec(N_grid, N_grid)

    # get corner structure
    axs = []
    for i in range(N_grid)[::-1]:
        for j in range(i+1):
            ax = fig.add_subplot(gs[i, j])
            axs.append(ax)
    
    axs1 = []
    #for pairs    
    for pair in combinations(range(N),2):
        #need to map pair to grid layour like the diagram
        y_ind = pair[0]*N_grid - pair[0]*(pair[0]-1)/2
        plot_id = int(y_ind + (N_grid-pair[1])) # index row and then column - see diagram above

        ax = axs[plot_id]
        
        twocov = covm[np.ix_(pair,pair)]
        twomu = mean[np.ix_(pair)]
        
        ax = plot_ellipse(ax, twocov, twomu, color=color)
        
        #for second cov mat
        if  None not in covm2:
            twocov2 = covm2[np.ix_(pair,pair)]
            twomu2 = mean2[np.ix_(pair)]
            ax = plot_ellipse(ax, twocov2, twomu2, color=color2)
            
            if twocov2[0, 0]>twocov[0, 0]:#define limits by largest ellipse
                twocov[0, 0] = twocov2[0, 0] 
            if twocov2[1, 1]>twocov[1, 1]:
                twocov[1, 1] = twocov2[1, 1] 
                 
        #now sort out axis
        ax.set_ylim(twomu[0] - scale_factor * np.sqrt(twocov[0, 0]), twomu[0] + scale_factor * np.sqrt(twocov[0, 0]))
        ax.set_xlim(twomu[1] - scale_factor * np.sqrt(twocov[1, 1]), twomu[1] + scale_factor * np.sqrt(twocov[1, 1]))
        ax = configure_axis(ax,N_grid,pair,varlabels)
            
        axs1.append(ax)

    plt.subplots_adjust(wspace=0, hspace=0)#

    return fig,axs1

### Helper functions ###


def plot_ellipse(ax, sigma, mean, color='red'):
    
    def ellipse_pars2(sigma):
        a2 = (sigma[0,0]+sigma[1,1])/2 + np.sqrt((sigma[0,0]-sigma[1,1])**2/4+sigma[0,1]**2)
        b2 = (sigma[0,0]+sigma[1,1])/2 - np.sqrt((sigma[0,0]-sigma[1,1])**2/4+sigma[0,1]**2)
        theta2 = np.arctan(-2*sigma[0,1]/(sigma[0,0]-sigma[1,1]))#switched minus - so should be good
        return np.sqrt(a2),np.sqrt(b2),(theta2)/2
    
    if sigma[0,0] > sigma[1,1]:
        a1,b1,t1 = ellipse_pars2(sigma)
    else:
        b1,a1,t1 = ellipse_pars2(sigma)
    
    ell1 = Ellipse(xy=mean, width=2*1.52*b1, height=2*1.52*a1, angle=t1*180/np.pi, linestyle = '-', edgecolor=color, fc='None', lw=2)
    ax.add_patch(ell1)
    ell2 = Ellipse(xy=mean, width=2*2.48*b1, height=2*2.48*a1, angle=t1*180/np.pi, linestyle = '--', edgecolor=color, fc='None', lw=2)
    ax.add_patch(ell2)

    return ax


def configure_axis(ax,N_grid,pair,varlabels):
    '''
    Remove axis for subplots that are not adjacent to bottom and left corner plot edges
    Set axis labels for remaining axis
    '''
    
    if pair[0]==0:
        ax.set_xlabel(varlabels[pair[1]])
    else:
        ax.xaxis.set_major_locator(plt.NullLocator())
        
    
    if pair[1]==N_grid:
        ax.set_ylabel(varlabels[pair[0]])
    else:
        ax.yaxis.set_major_locator(plt.NullLocator())
    

    return ax
