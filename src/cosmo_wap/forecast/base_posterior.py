"""
Base class for posterior analysis methods (Fisher matrices & MCMC samples).
Shared functionality for storing parameters and plotting with ChainConsumer.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any

import numpy as np
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cosmo_wap.lib import utils

if TYPE_CHECKING:
    from cosmo_wap.forecast import FullForecast


# Planck 2018 parameter covariances in the basis we sample - from scripts/planck_prior_cov.py.
# Omega_m = Omega_b + Omega_cdm (massless neutrinos, as in utils.get_cosmo).
PLANCK_PARAMS = ["Omega_m", "Omega_b", "Omega_cdm", "h", "ln_A_s", "n_s", "sigma8"]

# base_plikHM_TTTEEE_lowl_lowE_lensing (CMB only)
PLANCK_COV = np.array(
    [
        [5.41043834e-05, 4.26838158e-06, 4.98360018e-05, -3.92332446e-05, -2.75773484e-05, -2.08520105e-05, 1.27693501e-05],
        [4.26838158e-06, 3.65816356e-07, 3.90256523e-06, -3.01800121e-06, -1.99880671e-06, -1.66575468e-06, 1.03697143e-06],
        [4.98360018e-05, 3.90256523e-06, 4.59334366e-05, -3.62152434e-05, -2.55785417e-05, -1.91862558e-05, 1.17323786e-05],
        [-3.92332446e-05, -3.01800121e-06, -3.62152434e-05, 2.89474114e-05, 2.09274328e-05, 1.50487202e-05, -8.63360291e-06],
        [-2.75773484e-05, -1.99880671e-06, -2.55785417e-05, 2.09274328e-05, 2.02830717e-04, 1.36688622e-05, 7.14110885e-05],
        [-2.08520105e-05, -1.66575468e-06, -1.91862558e-05, 1.50487202e-05, 1.36688622e-05, 1.76660704e-05, -7.62132456e-07],
        [1.27693501e-05, 1.03697143e-06, 1.17323786e-05, -8.63360291e-06, 7.14110885e-05, -7.62132456e-07, 3.66545922e-05],
    ]
)

# base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO (CMB + BAO)
PLANCK_COV_BAO = np.array(
    [
        [3.06683951e-05, 2.40900356e-06, 2.82593915e-05, -2.26854792e-05, -1.47148233e-05, -1.18618736e-05, 8.15476279e-06],
        [2.40900356e-06, 2.19097956e-07, 2.18990561e-06, -1.70518240e-06, -9.74939990e-07, -9.59886461e-07, 6.62837535e-07],
        [2.82593915e-05, 2.18990561e-06, 2.60694859e-05, -2.09802968e-05, -1.37398833e-05, -1.09019872e-05, 7.49192526e-06],
        [-2.26854792e-05, -1.70518240e-06, -2.09802968e-05, 1.72605966e-05, 1.17753801e-05, 8.64892500e-06, -5.44208020e-06],
        [-1.47148233e-05, -9.74939990e-07, -1.37398833e-05, 1.17753801e-05, 1.97919164e-04, 8.11004327e-06, 7.44502926e-05],
        [-1.18618736e-05, -9.59886461e-07, -1.09019872e-05, 8.64892500e-06, 8.11004327e-06, 1.42698009e-05, 7.46987215e-07],
        [8.15476279e-06, 6.62837535e-07, 7.49192526e-06, -5.44208020e-06, 7.44502926e-05, 7.46987215e-07, 3.58154965e-05],
    ]
)


class BasePosterior(ABC):
    """Base class for different meethod of analysing the posterior distributions
    Either for Fishers or MCMC samples.
    Shared functionality of storing parameters and plotting with Chainconsumer."""

    def __init__(self, forecast: FullForecast, param_list: list[str], name: str | None = None) -> None:
        self.forecast = forecast
        self.cosmo_funcs = forecast.cosmo_funcs
        # so if one "param" is a list itself - then lets just call our parameter in some frankenstein way
        self.param_list = forecast._rename_composite_params(param_list)
        self.name = name or "_".join(self.param_list)  # sample name is amalgamation of parameters
        self.handle_latex()  # use latex label if latex is available
        self.fiducial = self._get_fiducial()

    @staticmethod
    def _split_tracer(param: str) -> tuple[str, tuple[str, ...]]:
        """Split a per-bin/LF param into (base, tracers) - fisher convention:
        'Xb_1' -> ('b_1', ('X',)); 'b_1' -> ('b_1', ('X', 'Y'))."""
        if param[:1] in ("X", "Y"):
            return param[1:], (param[0],)
        return param, ("X", "Y")

    def _get_tracer(self, label: str) -> Any:
        """The survey tracer carrying the 'X'/'Y' label (X = tracer 0, Y = tracer 1)."""
        return next(s for s in self.cosmo_funcs.survey if ["X", "Y"][s.t] == label)

    def handle_latex(self) -> None:
        try:
            # A lightweight test to see if LaTeX is available
            fig = plt.figure(figsize=(0.1, 0.1))
            plt.text(0, 0, "test", usetex=True)
            plt.close(fig)
            self.USE_LATEX = True
        except RuntimeError:
            warnings.warn("LaTeX not found. Will someone think of the plots!!")
            self.USE_LATEX = False

        if self.USE_LATEX:
            self.latex = {
                "fNL": r"$f_{\rm NL}$",
                "fNL_eq": r"$f^{\rm Eq}_{\rm NL}$",
                "fNL_loc": r"$f^{\rm Loc}_{\rm NL}$",
                "fNL_orth": r"$f^{\rm Orth}_{\rm NL}$",
                "n_s": "$n_s$",
                "A_s": "$A_s$",
                "ln_A_s": r"$\ln(10^{10}A_s)$",
                "h": "$h$",
                "w0": "$w_0$",
                "wa": "$w_a$",
                "Omega_m": r"$\Omega_m$",
                "Omega_cdm": r"$\Omega_{cdm}$",
                "Omega_b": r"$\Omega_{b}$",
                "sigma8": r"$\sigma_8$",
                "S8": r"$S_8$",
                "gamma": r"$\gamma$",
                "X_b_1": r"$\alpha^X_{b_1}$",
                "X_be": r"$\alpha^X_{be}$",
                "X_Q": r"$\alpha^X_{Q}$",
                "Y_b_1": r"$\alpha^Y_{b_1}$",
                "Y_be": r"$\alpha^Y_{be}$",
                "Y_Q": r"$\alpha^Y_{Q}$",
                "A_b_1": r"$\alpha_{b_1}$",
                "A_be": r"$\alpha_{be}$",
                "A_Q": r"$\alpha_{Q}$",
                "b_phi": r"$b_{\phi}$",
                "Xb_phi": r"$b^X_{\phi}$",
                "Yb_phi": r"$b^Y_{\phi}$",
                "b_phi_e": r"$b_{\phi e}$",
                "Xb_phi_e": r"$b^X_{\phi e}$",
                "Yb_phi_e": r"$b^Y_{\phi e}$",
                "A_b_phi_e": r"$\alpha_{b_{\phi e}}$",
                "X_b_phi_e": r"$\alpha^X_{b_{\phi e}}$",
                "Y_b_phi_e": r"$\alpha^Y_{b_{\phi e}}$",
            }  # define dictionary of latex strings for plotting for all of our parameters

            # PNG bias amplitudes - {X,Y,A}_{loc,eq,orth}_{b_01,b_11}, e.g. A_loc_b_11 -> \alpha^{Loc}_{b_{11}}
            for param in self.forecast.png_amp_bias:
                tracer, shape, _, order = param.split("_")
                sup = shape.capitalize() if tracer == "A" else f"{tracer},{shape.capitalize()}"
                self.latex[param] = rf"$\alpha^{{{sup}}}_{{b_{{{order}}}}}$"

            self.columns = [
                self.latex.get(param, param) for param in self.param_list
            ]  # have latex version of param_list
        else:
            # don't use latex
            self.latex = {}
            self.columns = self.param_list

    def _get_fiducial(self) -> dict[str, float]:
        """
        Get a dictionary of fiducial values for all free parameters.

        For redshift-dependent parameters, the value at the mean redshift of the survey is used.
        Parameters not explicitly defined default to 0, except for term amplitudes which default to 1.
        """
        fid_dict = {}
        for param in self.param_list:  # Default to 0
            fid_dict[param] = 0

        mid_z = (self.cosmo_funcs.z_min + self.cosmo_funcs.z_max) / 2

        # Fiducial values for bias parameters
        for param in self.forecast.biases:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs.survey, param)(mid_z)

        # Linked biases - b_phi and b_phi_e are both normalised on their tracer's b_phi
        for param in self.param_list:
            base, tracers = self._split_tracer(param)
            if base in self.forecast.linked_bias:
                fid_dict[param] = utils.linked_bias_fid(self._get_tracer(tracers[0]), base)(mid_z)

        # Fiducial values for standard cosmological parameters
        for param in utils.COSMO_PARAMS:
            if param in self.param_list:
                fid_dict[param] = getattr(self.cosmo_funcs, param)

        # Derived cosmological parameters
        if "S8" in self.param_list:
            fid_dict["S8"] = self.cosmo_funcs.sigma8 * np.sqrt(self.cosmo_funcs.Omega_m / 0.3)
        if "gamma" in self.param_list:
            fid_dict["gamma"] = np.log(self.cosmo_funcs.f(mid_z)) / np.log(self.cosmo_funcs.Om_m(mid_z))

        # Amplitudes of each contribution default to 1
        for param in self.cosmo_funcs.term_list:
            if param in self.param_list:
                fid_dict[param] = 1

        # Amplitude of bias parameters (Nuisance parameters)
        # all of these are multiplicative on the survey bias, so they sit at 1 - matching the
        # sampler's prior ref and the f*(1+h) the Fisher derivative takes
        for param in self.forecast.amp_bias + self.forecast.linked_amp_bias + self.forecast.png_amp_bias:
            if param in self.param_list:
                fid_dict[param] = 1

        return fid_dict

    def planck_cov(self, bao: bool = False) -> tuple[np.ndarray, list[str]]:
        """Returns Planck parameter covariance (CMB only, or Planck-BAO if bao) and the params it covers:
        Uses: parameter covariance from base_plikHM_TTTEEE_lowl_lowE_lensing(_post_BAO) - see scripts/planck_prior_cov.py"""

        full_cov = (PLANCK_COV_BAO if bao else PLANCK_COV).copy()

        # native ln_A_s = ln(10^10 A_s). If sampling ln_A_s, keep native units;
        # if sampling A_s, propagate to A_s units (sigma_A_s = A_s * sigma_logA).
        amp = "ln_A_s" if "ln_A_s" in self.param_list else "A_s"
        i_amp = PLANCK_PARAMS.index("ln_A_s")
        if amp == "A_s":
            full_cov[i_amp] *= self.cosmo_funcs.A_s
            full_cov[:, i_amp] *= self.cosmo_funcs.A_s
        planck_params = [amp if p == "ln_A_s" else p for p in PLANCK_PARAMS]

        # find what parameters in this prior we are sampling over!
        params = [p for p in self.param_list if p in planck_params]
        columns = [planck_params.index(p) for p in params]  # get columns/rows in cov_mat

        return full_cov[np.ix_(columns, columns)], params  # NxN matrix

    def _name_chain(self, c: ChainConsumer | None, name: str | None) -> tuple[ChainConsumer, str]:
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

        return c, name

    def get_fisher_centre(
        self, param_list: list[str], bias_values: dict[str, float] | list[dict[str, float]] | None = None
    ) -> np.ndarray:
        """Get centre of fisher - fiducial + bias"""
        if bias_values:
            if isinstance(bias_values, list):
                bias_values = bias_values[-1]  # use last entry which is sum of all terms if bias list is a list
        else:
            bias_values = {}  # then yeet is empty

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

    def add_chain_cov(
        self,
        c: ChainConsumer | None = None,
        bias_values: dict[str, float] | list[dict[str, float]] | None = None,
        name: str | None = None,
        cov: np.ndarray | None = None,
        param_list: list[str] | None = None,
        **kwargs,
    ) -> ChainConsumer:
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
            cov, param_list = self.planck_cov()  # so if no covaraince provided then defaults is planck parameter covariance
        else:
            if not param_list:
                param_list = self.param_list

        mean_values = self.get_fisher_centre(param_list, bias_values)  # get fiducial + bias (if we compute bias)

        c, name = self._name_chain(c, name)

        # default bar_shade (1D fill) to match shade (2D fill) so a single flag controls both
        if "shade" in kwargs:
            kwargs.setdefault("bar_shade", kwargs["shade"])

        # Create chain from covariance
        ch = Chain.from_covariance(mean_values, cov, columns=param_list, name=name, **kwargs)
        c.add_chain(ch)

        return c

    def corner_plot(
        self,
        c: ChainConsumer | None = None,
        extents: dict | None = None,
        figsize: tuple[float, float] | None = None,
        truth: bool = True,
        width: float = 3,
        fid2: dict[str, float] | None = None,
        fontsize: int = 16,
        tick_fontsize: int = 12,
        compact_legend: bool = False,
        serif: bool = True,
        **plot_kwargs,
    ) -> tuple[Figure, ChainConsumer]:
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

        # Add fiducial values as truth lines — zorder=15 puts them on top of chain contours (default zorder=10)
        if truth and self.fiducial:
            c.add_truth(Truth(location=self.fiducial, color="#500724", zorder=15))
        if fid2:
            c.add_truth(Truth(location=fid2, color="#16A085", zorder=15))

        plot_config = PlotConfig(
            usetex=True,
            serif=serif,  # ChainConsumer defaults serif off, so set it explicitly to match a serif/LaTeX style
            label_font_size=fontsize,
            tick_font_size=tick_fontsize,
            show_legend=not compact_legend,
        )
        if extents:
            plot_config.extents = extents
        else:
            # Auto-generate extents to fit all chains and truth lines
            extents = {}
            # union of all chain parameters (preserving order) - allows chains with different param sets
            param_list = []
            for chain_name in c.get_names():
                for col in c.get_chain(name=chain_name).data_columns:
                    if col not in param_list:
                        param_list.append(col)
            for param in param_list:
                mins, maxs = [], []
                largest_error = 0

                for chain_name in c.get_names():
                    chain_data = c.get_chain(name=chain_name)
                    if param not in chain_data.data_columns:
                        continue  # skip chains that don't have this parameter
                    samps = chain_data.samples[param]
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
                    extents[param] = (min(mins), max(maxs))
            plot_config.extents = extents

        plot_config.labels = self.latex  # use latex labels
        c.set_plot_config(plot_config)

        fig = c.plotter.plot(**plot_kwargs)

        if figsize:
            fig.set_size_inches(figsize)
        else:
            current_size = fig.get_size_inches()
            fig.set_size_inches(current_size + 3)

        return fig, c

    def _setup_1Dplot(self, param: str, figsize: tuple[float, float] = (8, 5), fontsize: int = 22) -> Axes:
        """Set up a 1D plot for a single parameter PDF w/wo bias"""
        _, ax = plt.subplots(figsize=figsize)
        # --- Customize the plot ---
        ax.set_xlabel(self.latex.get(param, param), fontsize=fontsize)
        ax.set_ylabel("")
        ax.yaxis.set_ticks([])  # Hide y-axis ticks and labels

        # Set x-axis limits and rotate ticks
        # ax.set_xlim(min(x_values), max(x_values))
        ax.tick_params(axis="x", labelsize=fontsize - 8, rotation=45)

        # Remove the box border (spines) for a cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add vertical line at 0
        ax.axvline(self.fiducial[param], color="black", linestyle="--", linewidth=1.5)

        # ax.set_ylim(bottom=0)
        return ax
