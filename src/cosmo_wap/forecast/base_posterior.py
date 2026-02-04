"""
Base class for posterior analysis methods (Fisher matrices & MCMC samples).
Shared functionality for storing parameters and plotting with ChainConsumer.
"""
import numpy as np
import warnings
from matplotlib import pyplot as plt
from abc import ABC

from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig, ChainConfig


class BasePosterior(ABC):
    """Base class for different meethod of analysing the posterior distributions
    Either for Fishers or MCMC samples.
    Shared functionality of storing parameters and plotting with Chainconsumer."""
    def __init__(self,forecast, param_list, name=None):
        self.forecast = forecast
        self.cosmo_funcs = forecast.cosmo_funcs
        # so if one "param" is a list itself - then lets just call our parameter in some frankenstein way
        self.param_list = forecast._rename_composite_params(param_list)
        self.name = name or "_".join(self.param_list) # sample name is amalgamation of parameters
        self.handle_latex() # use latex label if latex is available
        self.fiducial = self._get_fiducial()

    def handle_latex(self):
        try:
            # A lightweight test to see if LaTeX is available
            fig = plt.figure(figsize=(0.1, 0.1))
            plt.text(0, 0, 'test', usetex=True)
            plt.close(fig)
            self.USE_LATEX = True
        except RuntimeError:
            warnings.warn("LaTeX not found. Will someone think of the plots!!")
            self.USE_LATEX = False

        if self.USE_LATEX:
            self.latex = {"fNL": r"$f_{\rm NL}$",
                      "n_s": "$n_s$",
                      "A_s": "$A_s$",
                      "h"  : "$h$",
                      "Omega_m": r"$\Omega_m$",
                      "Omega_cdm": r"$\Omega_{cdm}$",
                      "Omega_b": r"$\Omega_{b}$",
                      "X_b_1": r"$\alpha^X_{b_1}$",
                      "X_be": r"$\alpha^X_{be}$",
                      "X_Q": r"$\alpha^X_{Q}$",
                      "Y_b_1": r"$\alpha^Y_{b_1}$",
                      "Y_be": r"$\alpha^Y_{be}$",
                      "Y_Q": r"$\alpha^Y_{Q}$",
                      "A_b_1": r"$\alpha_{b_1}$",
                      "A_be": r"$\alpha_{be}$",
                      "A_Q": r"$\alpha_{Q}$",
                      "A_loc_b_01": r"$\alpha^{Loc}_{b_{01}}$",
                      "X_loc_b_01": r"$\alpha^{X,Loc}_{b_{01}}$",
                      "Y_loc_b_01": r"$\alpha^{Y,Loc}_{b_{01}}$"}  # define dictionary of latex strings for plotting for all of our parameters

            self.columns = [self.latex.get(param, param) for param in self.param_list] # have latex version of param_list
        else:
            # don't use latex
            self.latex = {}
            self.columns = self.param_list

    def _get_fiducial(self):
        """
        Get a dictionary of fiducial values for all free parameters.

        For redshift-dependent parameters, the value at the mean redshift of the survey is used.
        Parameters not explicitly defined default to 0, except for term amplitudes which default to 1.
        """
        fid_dict = {}
        for param in self.param_list: # Default to 0
            fid_dict[param] = 0

        mid_z = (self.cosmo_funcs.z_min + self.cosmo_funcs.z_max) / 2

        # Fiducial values for bias parameters
        for param in ['b_1', 'b_2', 'g_2', 'be', 'Q']:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs.survey, param)(mid_z)

        # Fiducial values for standard cosmological parameters
        for param in ['Omega_m', 'Omega_cdm', 'Omega_b', 'A_s', 'n_s', 'h']:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs, param)

        # Amplitudes of each contribution default to 1
        for param in self.cosmo_funcs.term_list:
            if param in self.param_list:
                fid_dict[param] = 1

        # Amplitude of bias parameters (Nuisance parameters)
        for param in ['X_b_1','X_be','X_Q','Y_b_1','Y_be','Y_Q','A_b_1','A_be','A_Q']:
            if param in self.param_list:
                fid_dict[param] = 1

        return fid_dict

    def planck_cov(self):
        """Returns Planck-BAO parameter covariance:
        Uses: parameter covariance from base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO"""

        full_cov = np.array([
            [1.8259383e-08, -4.9452862e-08, 1.1363663e-08, 1.8051082e-07, 3.3472141e-07, 1.3359350e-07],
            [-4.9452862e-08, 8.3384991e-07, -5.0175067e-08, -1.9209722e-06, -2.1282794e-06, -1.9412337e-06],
            [1.1363663e-08, -5.0175067e-08, 8.5677597e-08, 2.2499557e-07, 4.5210639e-07, 2.0928688e-07],
            [1.8051082e-07, -1.9209722e-06, 2.2499557e-07, 5.0196921e-05, 9.1926469e-05, 6.8814113e-06],
            [3.3472141e-07, -2.1282794e-06, 4.5210639e-07, 9.1926469e-05, 1.9787144e-04, 8.1080876e-06],
            [1.3359350e-07, -1.9412337e-06, 2.0928688e-07, 6.8814113e-06, 8.1080876e-06, 1.4266360e-05]])

        # ok so lets convert units: ['omega_b','omega_cdm','theta','tau','logA','n_s']
        full_cov[:2] *= 1/self.cosmo_funcs.h**2
        full_cov[:,:2] *= 1/self.cosmo_funcs.h**2
        # lnAs to A_s - conversion factor is actually A_s from error propogation
        full_cov[4] = full_cov[4]*self.cosmo_funcs.A_s
        full_cov[:,4] = full_cov[:,4]*self.cosmo_funcs.A_s

        # if we wanted to switch to using Omega_m (instead of Omega_cdm/Omega_b then we can combine errors)

        #find what parameters in this prior we are sampling over!
        params = ['Omega_b','Omega_cdm','theta','tau','A_s','n_s']
        columns = []
        for param in self.param_list:
            for j,prior_param in enumerate(params):
                if param==prior_param:
                    columns.append(j) # get columns/rows in cov_mat

        return full_cov[columns][:,columns] # NxN matrix

    def _name_chain(self,c,name):
        """define chainconsumer object and name of chain if none"""
        # Create ChainConsumer object
        if c == None:
            c = ChainConsumer()

        if name is None:
            # Generate unique name based on existing chains
            existing_names = c.get_names()

            # Find the next available number
            chain_number = 1
            while f"chain_{chain_number}" in existing_names:
                chain_number += 1

            name = f"chain_{chain_number}"

        return c,name

    def get_fisher_centre(self,param_list,bias_values=None):
        """Get centre of fisher - fiducial + bias"""
        if bias_values:
            if isinstance(bias_values, list):
                bias_values = bias_values[-1] # use last entry which is sum of all terms if bias list is a list
        else:
            bias_values = {} # then yeet is empty

        # Use fiducial parameter values and apply bias if provided
        mean_values = np.zeros(len(param_list))
        for i, param in enumerate(param_list):
            if param in bias_values:
                offset = bias_values[param]
            else:
                offset = 0

            if param in self.fiducial:
                fid = self.fiducial[param]
            else:
                fid = 0

            mean_values[i] = fid + offset

        return mean_values

    def add_chain_cov(self,c=None,bias_values=None,name=None,cov=None,param_list=None,**kwargs):
        """
        Get chain from a covariance matrix - defualt is planck-covaraince-
        But later uses inverse fisher matrices

        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            bias_values (dict, optional): Best fit bias on parameter mean - calculate using get_bias etc.
                Keys should match parameter names, e.g., {'b_1': 1.0, 'sigma_8': 0.1}.
                If not provided then default is 0.
            name - name of chain
            cov - covariance matrix, if none then defaults to planck parameter covariance
            param_list - parameters to use, can take submatrix in full covariance

        Returns:
            ChainConsumer: ChainConsumer object with a brand new chain!
        """
        if cov is None:
            cov = self.planck_cov() # so if no covaraince provided then defaults is planck parameter covariance
            param_list = [param for param in self.param_list if param in ['Omega_b','Omega_cdm','theta','tau','A_s','n_s']]
        else:
            if not param_list:
                param_list = self.param_list

        mean_values = self.get_fisher_centre(param_list,bias_values) # get fiducial + bias (if we compute bias)

        c,name = self._name_chain(c,name)

        # Create chain from covariance
        ch = Chain.from_covariance(
            mean_values,
            cov,
            columns=param_list,
            name=name,
            **kwargs
        )
        c.add_chain(ch)

        return c

    def corner_plot(self, c=None, extents=None, figsize=None, truth=True, width=3, fid2=None, fontsize =16,tick_fontsize=12, **plot_kwargs):
        """
        Plot parameter contours using ChainConsumer.

        Args:
        c (ChainConsumer, optional): ChainConsumer object containing chains to plot.
            If None, creates a new ChainConsumer with just this Fisher matrix.
        extents (dict, optional): Plot extents (as tuples) for specific parameters.
            e.g., {'Omega_m': (0.2, 0.4), 'sigma_8': (0.6, 1.0)}.
        figsize (tuple, optional): Figure size as (width, height). If None, uses
            default size plus 3 inches in each dimension.
        truth (bool, optional): If True, adds fiducial parameter values as truth points
            on the plot. Default is True.
        **plot_kwargs: Additional keyword arguments passed to ChainConsumer's plot method.
        """
        if c is None:
            c = self.add_chain()

        # Add fiducial values as truth lines
        if truth and self.fiducial:
            c.add_truth(Truth(location=self.fiducial, color="#500724"))
        if fid2:
            c.add_truth(Truth(location=fid2, color="#16A085"))

        plot_config = PlotConfig(usetex=True,label_font_size=fontsize,tick_font_size=tick_fontsize)
        if extents:
            plot_config.extents = extents
        else:
            # Auto-generate extents to fit all chains and truth lines
            extents = {}
            param_list = c.get_chain(name=c.get_names()[0]).data_columns # cannot remember why we get names from here and not self.param_list - but maybe more flexible
            for i,param in enumerate(param_list):
                mins, maxs = [], []
                largest_error = 0

                for chain_name in c.get_names():
                    samps = c.get_chain(name=chain_name).samples[param]
                    mean, error = samps.mean(), samps.std()
                    mins.append(mean - width * error)
                    maxs.append(mean + width * error)
                    largest_error = max(largest_error, error)

                if truth and param in self.fiducial:
                    mins.append(self.fiducial[param] - largest_error * 0.1)
                    maxs.append(self.fiducial[param] + largest_error * 0.1)
                if fid2 and param in fid2:
                    mins.append(fid2[param] - largest_error * 0.1)
                    maxs.append(fid2[param] + largest_error * 0.1)

                if mins and maxs:
                    extents[self.param_list[i]] = (min(mins), max(maxs))
            plot_config.extents = extents

        plot_config.labels = self.latex # use latex labels
        c.set_plot_config(plot_config)

        fig = c.plotter.plot(**plot_kwargs)

        if figsize:
            fig.set_size_inches(figsize)
        else:
            current_size = fig.get_size_inches()
            fig.set_size_inches(current_size + 3)

        return fig, c

    def _setup_1Dplot(self,param,figsize=(8,5),fontsize=22):
        _, ax = plt.subplots(figsize=figsize)
        # --- Customize the plot ---
        ax.set_xlabel(self.latex.get(param, param), fontsize=fontsize)
        ax.set_ylabel('')
        ax.yaxis.set_ticks([]) # Hide y-axis ticks and labels

        # Set x-axis limits and rotate ticks
        #ax.set_xlim(min(x_values), max(x_values))
        ax.tick_params(axis='x', labelsize=fontsize-8, rotation=45)

        # Remove the box border (spines) for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add vertical line at 0
        ax.axvline(self.fiducial[param], color='black', linestyle='--', linewidth=1.5)

        #ax.set_ylim(bottom=0)
        return ax
