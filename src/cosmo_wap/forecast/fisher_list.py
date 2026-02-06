"""
Batch processing of Fisher matrices across survey cuts and splits.
"""
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from .base_posterior import BasePosterior


class FisherList(BasePosterior):
    """
    Class to store and handle lists of Fisher Matrices - particulalry for different cuts and splits
    """
    def __init__(self, fish_list, forecast, param_list,cuts,splits):
        """
        Args:
            fish_list (np.ndarray): List of FisherMat objects
            param_list (list): List of parameter names
        """
        super().__init__(forecast, param_list)

        self.fish_list = fish_list
        self.cuts = cuts
        self.splits = splits

    def plot(self,param=None,cmap='viridis',gamma=0.5,save=True,vmax=5,**kwargs):
        """Plot multi-tracer SNRs: kwargs affect imshow"""

        if param is None:
            param = self.param_list[0]

        err_arr =  np.zeros((len(self.cuts),len(self.splits)))

        for i,_ in enumerate(self.cuts):
            for j,_ in enumerate(self.splits):
                if self.fish_list[i][j] != None:
                    err_arr[i,j] = self.fish_list[i][j].get_error(param) # marginalsed error

        # mask the entries that are NaN so they appear white
        mask = (err_arr == 0) | np.isnan(err_arr)

        cmap = copy.copy(plt.cm.get_cmap(cmap))
        cmap.set_bad(color='white') # set nans to zero


        # so to make rows be smoother we interpolate and have higher resolution sampling
        new_res = len(self.splits)*100
        smooth_data = np.full((len(self.cuts), new_res), np.nan)
        samps = np.linspace(min(self.splits), max(self.splits), new_res) #higher res

        for i in range(len(self.cuts)):
            # Extract the raw row and its mask
            row_data = err_arr[i]
            row_mask = mask[i]

            # ignore splits less than cut
            valid_x = self.splits[~row_mask]
            valid_y = row_data[~row_mask]

            if len(valid_x) > 0:
                # Interpolate only using valid points.
                # 'left=np.nan' ensures that any sample to the left of the first valid
                # data point becomes NaN (White), creating the sharp cut-off.
                smooth_data[i] = np.interp(samps, valid_x, valid_y, left=np.nan, right=np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))
        norm = mcolors.PowerNorm(gamma=gamma,vmax=vmax)

        extent = [min(self.splits), max(self.splits), min(self.cuts), max(self.cuts)]

        im = ax.imshow(
            smooth_data,
            extent=extent,
            origin='upper',            # for fluxes
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
            aspect='auto',             # Forces the image to stretch to fill the axes
            **kwargs
        )

        ax.set_xlabel(r'Splitting flux, $F_s$')
        ax.set_ylabel(r'Flux cut, $F_c$')
        #ax.set_xticks(self.splits[::2])
        half_step = (max(self.cuts) - min(self.cuts)) / (2 * len(self.cuts))
        tick_positions = np.linspace(min(self.cuts) + half_step, max(self.cuts) - half_step, len(self.cuts))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(self.cuts[::-1])
        #plt.xticks(rotation=45)

        # Add gridlines behind the plot
        #ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

        # Colorbar (horizontal at the bottom)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, aspect=40)
        cbar.set_label('Error')

        # save to plots folders
        if save:
            plt.savefig('plots/cuts_splits.png', dpi=300, bbox_inches='tight', transparent=False)

        plt.tight_layout()
        plt.show()
