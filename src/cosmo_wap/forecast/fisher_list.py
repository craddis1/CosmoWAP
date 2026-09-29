"""
Batch processing of Fisher matrices across a grid of settings - e.g. survey cuts and splits.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from .base_posterior import BasePosterior


class FisherList(BasePosterior):
    """
    Class to store and handle lists of Fisher Matrices - particulalry for different cuts and splits
    Created by FullForecast.get_fish_list - fish has one axis per grid key, None where a point was skipped.
    """

    def __init__(self, fish_list, forecast, param_list, grid):
        """
        Args:
            fish_list (np.ndarray): object array of FisherMat objects (or None), shaped like the grid
            param_list (list): List of parameter names
            grid (dict): {axis name: values}
        """
        super().__init__(forecast, param_list)

        self.fish = fish_list
        self.grid = {ax: list(values) for ax, values in grid.items()}

    def __getitem__(self, idx):
        return self.fish[idx]

    def _index(self, ax, value):
        """Index of value along a grid axis - numbers compared with isclose."""
        for i, v in enumerate(self.grid[ax]):
            if v is value or v == value:
                return i
            if _is_numeric([v, value]) and np.isclose(v, value, rtol=1e-8, atol=0):
                return i
        raise ValueError(f"{value} not in grid axis '{ax}': {self.grid[ax]}")

    def at(self, **point):
        """FisherMat at a grid point by value, e.g. fl.at(cut=2e-16, split=3e-16).
        Leave axes out to get the array along them."""
        return self.fish[tuple(self._index(ax, point[ax]) if ax in point else slice(None) for ax in self.grid)]

    def get_error(self, param):
        """Marginalised error on param at every grid point - NaN where skipped."""
        err = np.full(self.fish.shape, np.nan)
        for idx, fish in np.ndenumerate(self.fish):
            if fish is not None:
                err[idx] = fish.get_error(param)
        return err

    def best(self, param):
        """Grid point with the smallest error on param."""
        idx = np.unravel_index(np.nanargmin(self.get_error(param)), self.fish.shape)
        return {ax: values[i] for (ax, values), i in zip(self.grid.items(), idx)}

    def map(self, func):
        """New FisherList with func applied to every FisherMat - e.g. fl.map(lambda f: f.add_planck_prior())"""
        new = np.full(self.fish.shape, None, dtype=object)
        for idx, fish in np.ndenumerate(self.fish):
            if fish is not None:
                new[idx] = func(fish)
        first = next(f for f in new.flat if f is not None)
        return FisherList(new, self.forecast, list(first.param_specs.values()), self.grid)

    def plot(
        self,
        param=None,
        fixed=None,
        labels=None,
        smooth=True,
        cmap="viridis",
        gamma=0.5,
        vmax=None,
        save=None,
        ax=None,
        **kwargs,
    ):
        """Plot errors over the grid: 1D as a line, 2D as a heatmap (rows the first axis, columns the second).
        fixed: {axis: value} to take a 1D or 2D slice of a bigger grid
        labels: {axis: label} for the axis labels
        smooth: interpolate along the columns - needs numeric column values
        save: path to save the figure to
        kwargs affect imshow (2D) or plot (1D)"""

        if param is None:
            param = self.param_list[0]

        fixed = fixed or {}
        labels = labels or {}
        err_arr = self.get_error(param)[
            tuple(self._index(a, fixed[a]) if a in fixed else slice(None) for a in self.grid)
        ]
        axes = [a for a in self.grid if a not in fixed]
        if len(axes) not in (1, 2):
            raise ValueError(f"Can plot 1 or 2 grid axes, not {axes} - fix the rest with fixed={{axis: value}}")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5) if len(axes) == 1 else (10, 6))
        else:
            fig = ax.figure
        err_label = r"$\sigma$(" + self.latex.get(param, param) + ")"

        if len(axes) == 1:
            x, ticks = _positions(self.grid[axes[0]])
            ax.plot(x, err_arr, **kwargs)
            if ticks is not None:
                ax.set_xticks(x, ticks)
            ax.set_xlabel(labels.get(axes[0], axes[0]))
            ax.set_ylabel(err_label)
        else:
            rows, cols = (self.grid[a] for a in axes)

            # mask the entries that are NaN so they appear white
            mask = np.isnan(err_arr)

            cmap = plt.get_cmap(cmap).copy()
            cmap.set_bad(color="white")  # set nans to zero

            if smooth and _is_numeric(cols):
                # so to make rows be smoother we interpolate and have higher resolution sampling
                cols = np.asarray(cols)
                order = np.argsort(cols)
                new_res = len(cols) * 100
                smooth_data = np.full((len(rows), new_res), np.nan)
                samps = np.linspace(min(cols), max(cols), new_res)  # higher res

                for i in range(len(rows)):
                    # Extract the raw row and its mask
                    row_data = err_arr[i][order]
                    row_mask = mask[i][order]

                    # ignore splits less than cut
                    valid_x = cols[order][~row_mask]
                    valid_y = row_data[~row_mask]

                    if len(valid_x) > 0:
                        # Interpolate only using valid points.
                        # 'left=np.nan' ensures that any sample to the left of the first valid
                        # data point becomes NaN (White), creating the sharp cut-off.
                        smooth_data[i] = np.interp(samps, valid_x, valid_y, left=np.nan, right=np.nan)
                x_extent = [min(cols), max(cols)]
            else:
                smooth_data = err_arr
                x_extent = [0, len(cols)]
                ax.set_xticks(np.arange(len(cols)) + 0.5, _positions(cols, force_labels=True)[1])

            vmax = np.nanmax(smooth_data) if vmax is None else vmax
            norm = mcolors.PowerNorm(gamma=gamma, vmin=np.nanmin(smooth_data), vmax=vmax)

            extent = [*x_extent, 0, len(rows)]

            im = ax.imshow(
                smooth_data,
                extent=extent,
                origin="upper",  # for fluxes
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                aspect="auto",  # Forces the image to stretch to fill the axes
                **kwargs,
            )

            ax.set_xlabel(labels.get(axes[1], axes[1]))
            ax.set_ylabel(labels.get(axes[0], axes[0]))
            # rows are categorical - one tick at each row centre, the first row at the top
            ax.set_yticks(np.arange(len(rows)) + 0.5, _positions(rows, force_labels=True)[1][::-1])

            # Add gridlines behind the plot
            # ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

            # Colorbar (horizontal at the bottom)
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2, aspect=40)
            cbar.set_label(err_label)

        fig.tight_layout()

        if save:
            filepath = Path(save)

            # Create plots folder if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches="tight", transparent=False)

        return fig, ax


def _is_numeric(values):
    try:
        return np.issubdtype(np.asarray(values).dtype, np.number)
    except (TypeError, ValueError):
        return False


def _positions(values, force_labels=False):
    """x positions for a grid axis and tick labels - numbers are used as they are, anything else
    (terms, kmax functions, ...) goes at 0, 1, 2 ... with str labels."""
    if _is_numeric(values) and not force_labels:
        return np.asarray(values), None
    ticks = [f"{v:.3g}" if _is_numeric([v]) else getattr(v, "__name__", str(v)) for v in values]
    return np.arange(len(values)), ticks
