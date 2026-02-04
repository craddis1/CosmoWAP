"""
Holds the class that store fisher matrices -  then contains a bunch of routines for plotting and analysing (they are computed in forecast.py)
"""
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy import stats

from .base_posterior import BasePosterior


class FisherMat(BasePosterior):
    """
    Class to store and handle Fisher matrix results with built-in plotting capabilities.
    """
    def __init__(self, fisher_matrix, forecast, param_list, term=None, config=None, name=None):
        """
        Initialize Fisher result object.

        Args:
            fisher_matrix (np.ndarray): The Fisher information matrix
            param_list (list): List of parameter names
            term (str or list): The term(s) used in the forecast
            config (dict): Configuration used for the forecast (pkln, bkln, etc.)
            name (str): Optional name for this result
        """
        super().__init__(forecast, param_list, name=name)

        self.fisher_matrix = fisher_matrix
        self.term = term
        self.config = config or {}

        # if not computed then is None and if it is and a list then add all previous entries to get sum bias
        if isinstance(config['bias'],list) and len(config['bias'])>1:
            self.bias = config['bias']
            keys = self.bias[0].keys()
            tot = {}
            for key in keys:
                tot[key] = 0
                for i,b_dict in enumerate(self.bias):
                    tot[key] += b_dict[key]
            self.bias.append(tot)

        self.bias = config['bias']

        # Compute derived quantities
        self.covariance = np.linalg.inv(fisher_matrix)
        self.errors = np.sqrt(np.diag(self.covariance))
        # add check for singular values:
        if np.isnan(self.errors).any():
            nan_indices = np.where(np.isnan(self.errors))[0]
            nan_params = [param_list[i] for i in nan_indices]
            warnings.warn(f"Singular matrix in {nan_params}")
            #raise ValueError(f"Singular matrix in {nan_params}")

        self.correlation = self._compute_correlation()

    def _compute_correlation(self):
        """Compute correlation matrix from covariance."""
        correlation = np.zeros_like(self.covariance)
        for i in range(len(self.param_list)):
            for j in range(len(self.param_list)):
                correlation[i,j] = self.covariance[i,j] / (self.errors[i] * self.errors[j])
        return correlation

    def get_error(self, param):
        """Get 1-sigma error for a specific parameter."""
        if param in self.param_list:
            idx = self.param_list.index(param)
            return self.errors[idx]
        else:
            raise ValueError(f"Parameter {param} not found in {self.param_list}")

    def get_correlation(self, param1, param2):
        """Get correlation coefficient between two parameters."""
        if param1 in self.param_list and param2 in self.param_list:
            i = self.param_list.index(param1)
            j = self.param_list.index(param2)
            return self.correlation[i,j]
        else:
            raise ValueError(f"One or both parameters not found in {self.param_list}")

    def summary(self):
        """Print summary of Fisher matrix results."""
        print(f"\n=== Fisher Matrix Results: {self.name} ===")
        print(f"Parameters: {self.param_list}")
        print(f"Term(s): {self.term}")
        print("\n1-sigma errors:")
        for i, param in enumerate(self.param_list):
            print(f"  σ({param}) = {self.errors[i]:.4f}")

        print("\nCorrelation matrix:")
        print("     ", end="")
        for param in self.param_list:
            print(f"{param:>8}", end="")
        print()
        for i, param1 in enumerate(self.param_list):
            print(f"{param1:>4} ", end="")
            for j, _ in enumerate(self.param_list):
                print(f"{self.correlation[i,j]:>8.3f}", end="")
            print()

    def add_chain(self,c=None,bias_values=None,name=None,cov=None,**kwargs):
        """
        Add the covariance (inverse Fisher) matrix as a chain to ChainConsumer object.
        Is wrapper for add_chain_cov in base class

        Args:
            c (ChainConsumer, optional): Existing ChainConsumer object to add chain to.
                If None, creates a new ChainConsumer object.
            bias_values (dict, optional): Best fit bias on parameter mean - calculate using get_bias etc.
                Keys should match parameter names, e.g., {'b_1': 1.0, 'sigma_8': 0.1}.
                If not provided then default is 0.

        Returns:
            ChainConsumer: ChainConsumer object with this Fisher matrix added as a chain.
        """
        # reset defaults for FisherMat object
        if cov is None:
            cov = self.covariance
        if bias_values is None:
            bias_values = self.bias
        return self.add_chain_cov(c=c,bias_values=bias_values,name=name,cov=cov,**kwargs)

    def compute_biases(self,bias_term,verbose=True):
        """Wrapper function of best_fit_bias in FullForecast:
        Compute biases for all parameters in fisher matrix
        Uses same config used to compute fisher."""

        terms = self.config['terms']
        pkln = self.config['pkln']
        bkln = self.config['bkln']
        t = self.config['t']
        s = self.config['s']
        r = self.config['r']
        sigma = self.config['sigma']
        nonlin = self.config['nonlin']

        bias_dict,_ = self.forecast.best_fit_bias(self.param_list, bias_term, terms,
                                                pkln,bkln,t=t,r=r,s=s,verbose=verbose,sigma=sigma,nonlin=nonlin)
        return bias_dict

    def plot_errors(self, relative=False, figsize=(8, 6)):
        """
        Plot parameter errors as a bar chart.
        parameters
        Args:
            relative (bool): Plot relative errors (σ/|mean|) if True
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        if relative:
            # Compute relative errors (avoid division by zero)
            mean_values = np.zeros(len(self.param_list))  # Could be modified to use actual means
            rel_errors = []
            labels = []
            for i, param in enumerate(self.param_list):
                if abs(mean_values[i]) > 1e-10:
                    rel_errors.append(self.errors[i] / abs(mean_values[i]))
                    labels.append(param)
                else:
                    rel_errors.append(self.errors[i])
                    labels.append(param)

            ax.bar(labels, rel_errors)
            ax.set_ylabel('Relative Error (σ/|μ|)')
            ax.set_title(f'Relative Parameter Errors: {self.name}')
        else:
            ax.bar(self.param_list, self.errors)
            ax.set_ylabel('1-σ Error')
            ax.set_title(f'Parameter Errors: {self.name}')

        ax.set_xlabel('Parameters')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig, ax

    def plot_1D(self, param, ci=0.68, ax=None, shade=True, color='royalblue',normalise_height=False, shade_alpha=0.2, figsize=(8,5), **kwargs):
        """
        Plots a 1D Gaussian (Normal) PDF given a mean and standard deviation.

        Args:
            param (str): parameter to plot
            ci (float, optional): The confidence interval to shade (e.g., 0.68 for 1-sigma). Defaults to 0.68.
            ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure is created.
            label (str, optional): The label for the x-axis.
            shade (bool, optional): If True, shades the confidence interval.
            color (str, optional): The color for the plot line and shade.
            **kwargs: Additional keyword arguments passed to ax.plot().
        """
        # --- 1. Set up the plot if an axis isn't provided ---
        if not ax:
            ax = self._setup_1Dplot(param,figsize=figsize,fontsize=22)

        mean = self.get_fisher_centre([param],self.bias)[0] # get fiducial + bias (if we compute bias)
        std_dev = self.get_error(param)

        # --- 2. Define the Normal distribution using the provided mean and std_dev ---
        norm_dist = stats.norm(loc=mean, scale=std_dev)

        # --- 3. Determine the x-range for plotting ---
        # A range of ±4 standard deviations from the mean covers >99.9% of the distribution
        x_eval = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 500)
        pdf_values = norm_dist.pdf(x_eval)

        if normalise_height: # normalise height of peak to 1
            peak  = np.max(pdf_values)
            boost = 1/peak
        else:
            boost = 1

        # --- 4. Plot the main PDF curve ---
        ax.plot(x_eval, boost*pdf_values, color=color, **kwargs)

        # --- 5. Shade the confidence interval ---
        if shade:
            # Calculate the interval boundaries using the Percent Point Function (inverse of CDF)
            # This is more robust than assuming ci=0.68 is always 1-sigma
            lower_bound, upper_bound = norm_dist.interval(ci)

            # Create x values just for the shaded region
            x_fill = np.linspace(lower_bound, upper_bound, 100)
            y_fill = norm_dist.pdf(x_fill)
            ax.fill_between(x_fill, boost*y_fill, color=color, alpha=shade_alpha)

        return ax

    def save(self, filename):
        """Save Fisher result to file."""
        np.savez(filename,
                fisher_matrix=self.fisher_matrix,
                param_list=self.param_list,
                term=self.term,
                config=self.config,
                name=self.name)

    @classmethod
    def load(cls, filename):
        """Load Fisher result from file."""
        data = np.load(filename, allow_pickle=True)
        return cls(
            data['fisher_matrix'],
            data['param_list'].tolist(),
            data['term'].item(),
            data['config'].item(),
            data['name'].item()
        )
