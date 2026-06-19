from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from cosmo_wap.lib import utils

logger = logging.getLogger(__name__)


class LFBiasPrior:
    """Forward-model luminosity-function fit-parameter uncertainties onto a prior on the
    selection functions evolution bias b_e(z) and magnification bias Q(z).

    Monte-Carlo push-forward (cf. HorizonGRound; Wang, Beutler & Bacon 2020,
    arXiv:2007.01802): the luminosity-function fit parameters carry (here diagonal) errors,
    so we draw them from their error distribution, recompute the selection functions for
    each draw and summarise the resulting joint distribution. The per-redshift covariance
    is then used as a prior on b_e/Q in the Fisher forecast and the MCMC sampler.

    The forward model is described by a set of named ``components`` and an ``evaluate``
    callback that, given a luminosity function with sampled parameters, returns each
    component on the dense ``z_grid``. For a single tracer the components are ('be', 'Q');
    for a bright/faint split they are the per-tracer ('Xbe', 'XQ', 'Ybe', 'YQ') with
    X = bright (tracer 0) and Y = faint (tracer 1) - their correlation through the shared
    luminosity function is captured automatically. Use :meth:`from_survey` to build these
    from a survey; the constructor is the lower-level entry point.

    Parameters:
    -----------
    LF : luminosity function instance
        Must expose ``fit_params`` (list of sampleable attribute names), the fiducial
        values as attributes and (unless ``errors``/``cov`` are given) ``fit_errors``.
    evaluate : callable
        ``evaluate(LF) -> array`` of shape ``(len(components), len(z_grid))`` giving each
        component for a luminosity function whose ``fit_params`` have been set.
    components : list of str
        Names of the forward-modelled selection functions.
    z_grid : array
        Dense redshift grid on which the components are evaluated before interpolation
        (kept dense so the ``np.gradient`` inside ``get_be`` stays accurate).
    errors : dict, optional
        Map ``{param_name: error}`` at ``sigma_level`` sigma. Defaults to ``LF.fit_errors``.
    cov : array, optional
        Full (1-sigma) parameter covariance ordered as ``LF.fit_params``; overrides
        ``errors`` and enables correlated draws.
    sigma_level : float
        Confidence level of ``errors`` (default 2, i.e. ``errors`` are 2-sigma).
    n_samples : int
        Number of Monte-Carlo draws.
    seed : int, optional
        Seed for the random generator (reproducibility).
    """

    def __init__(
        self,
        LF: object,
        evaluate: Callable[[object], np.ndarray],
        components: list[str],
        z_grid: ArrayLike,
        *,
        errors: dict | None = None,
        cov: ArrayLike | None = None,
        sigma_level: float = 2.0,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> None:
        if not hasattr(LF, "fit_params"):
            raise ValueError("Luminosity function has no 'fit_params' - cannot forward-model fit errors.")

        self.LF = LF
        self.evaluate = evaluate
        self.components = list(components)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.params = list(LF.fit_params)
        self.n_samples = n_samples
        self.sigma_level = sigma_level

        # fiducial parameter vector (mean of the draws)
        self.mean_params = np.array([getattr(LF, p) for p in self.params], dtype=float)

        # parameter covariance: full matrix if given, else diagonal from the errors
        if cov is not None:
            self.param_cov = np.asarray(cov, dtype=float)
        else:
            if errors is None:
                errors = getattr(LF, "fit_errors", None)
            if not errors:
                raise ValueError("No fit errors provided and LF.fit_errors is empty - nothing to forward-model.")
            std = np.array([errors.get(p, 0.0) for p in self.params], dtype=float) / sigma_level
            self.param_cov = np.diag(std**2)

        rng = np.random.default_rng(seed)
        self._draws = rng.multivariate_normal(self.mean_params, self.param_cov, size=n_samples)
        self._samples = self._push_forward()  # (n_valid, n_components, len(z_grid))

    def _push_forward(self) -> np.ndarray:
        """Recompute every component on ``z_grid`` for each parameter draw.

        Returns a (n_valid, n_components, len(z_grid)) array; draws giving non-finite
        components (e.g. a pathological slope making the luminosity integral diverge) are
        dropped so the covariance stays well-defined.
        """
        samples = np.empty((self.n_samples, len(self.components), self.z_grid.size))
        for i, draw in enumerate(self._draws):
            lf = utils.copy(self.LF)  # preserves the shared cosmo reference
            for name, value in zip(self.params, draw):
                setattr(lf, name, value)
            samples[i] = self.evaluate(lf)

        valid = np.all(np.isfinite(samples), axis=(1, 2))
        n_drop = np.count_nonzero(~valid)
        if n_drop:
            logger.warning("LFBiasPrior: dropped %d/%d non-finite samples.", n_drop, self.n_samples)
        return samples[valid]

    def _interp(self, zz: ArrayLike, names: list[str] | None) -> tuple[np.ndarray, list[str]]:
        """Spline the requested components from ``z_grid`` to ``zz``.

        Returns ``(values, names)`` where values has shape ``(n_valid, len(names), len(zz))``.
        """
        zz = np.atleast_1d(zz)
        names = list(self.components if names is None else names)
        cols = [self.components.index(n) for n in names]
        vals = CubicSpline(self.z_grid, self._samples, axis=2)(zz)  # (n_valid, n_components, len(zz))
        return vals[:, cols, :], names

    def sample(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        """Component sample array at ``zz`` with shape ``(n_valid_samples, len(names), len(zz))``."""
        return self._interp(zz, names)[0]

    def mean(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        """Sample mean of the components at ``zz`` with shape ``(len(names), len(zz))``."""
        return self._interp(zz, names)[0].mean(axis=0)

    def std(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        """Sample standard deviation of the components at ``zz``, shape ``(len(names), len(zz))``."""
        return self._interp(zz, names)[0].std(axis=0)

    def covariance(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        """Joint covariance of the (requested) components at each redshift.

        Returns an array of shape ``(len(zz), m, m)`` (m = number of components) where entry
        ``k`` is the covariance over the parameter draws at ``zz[k]`` - this captures the
        correlations induced by the shared luminosity-function parameters (e.g. b_e-Q and,
        for a split, bright-faint).
        """
        vals, names = self._interp(zz, names)  # (n_valid, m, len(zz))
        m, nz = len(names), vals.shape[2]
        cov = np.empty((nz, m, m))
        for k in range(nz):
            cov[k] = np.cov(vals[:, :, k].T)
        return cov

    @classmethod
    def from_survey(cls, survey: object, **kwargs) -> "LFBiasPrior":
        """Build the prior from a survey carrying a luminosity function.

        Reads ``survey.LF`` and ``survey.cut`` (set in ``survey_params`` via
        ``compute_luminosity``); fit errors default to ``survey.LF.fit_errors``. A
        bright/faint split is detected from ``survey.split`` and forward-modelled with the
        per-tracer components ('Xbe', 'XQ', 'Ybe', 'YQ'), reproducing the survey's faint
        derivation (see survey_params._get_faint) per draw.
        """
        if not hasattr(survey, "LF"):
            raise ValueError("Survey has no luminosity function (survey.LF) - cannot forward-model fit errors.")

        LF = survey.LF
        cut = survey.cut
        # dense grid matching the survey's own bias grid where possible
        z_grid = np.asarray(getattr(survey, "zz", np.linspace(np.min(LF.z_values), np.max(LF.z_values), 100)))

        split = getattr(survey, "split", None)
        if split is None:
            components = ["be", "Q"]

            def evaluate(lf):
                return np.array([lf.get_be(cut, z_grid), lf.get_Q(cut, z_grid)])
        else:
            # bright/faint split: X = bright (cut=split), Y = faint (total minus bright).
            # Mirrors survey_params._get_faint but only for the b_e/Q we need.
            components = ["Xbe", "XQ", "Ybe", "YQ"]

            def evaluate(lf):
                n_T = lf.number_density(cut, z_grid)
                Q_T = lf.get_Q(cut, z_grid)
                be_B = lf.get_be(split, z_grid)
                Q_B = lf.get_Q(split, z_grid)
                n_B = lf.number_density(split, z_grid)
                n_F = n_T - n_B
                Q_F = utils.get_faint_bias(z_grid, n_T, n_B, Q_T, Q_B)(z_grid)
                be_F = lf.get_be(None, z_grid, n_g=n_F, Q=Q_F)
                return np.array([be_B, Q_B, be_F, Q_F])

        return cls(LF, evaluate, components, z_grid, **kwargs)


class ConstantBiasPrior:
    """Constant (z-independent) Gaussian prior on the per-bin b_e/Q.

    For surveys whose b_e/Q are not from a forward-modellable luminosity function (e.g. the
    DESI BGS Smith HOD), assume a constant 1-sigma error instead of pushing fit parameters
    forward. Defaults sigma(b_e)=1, sigma(Q)=0.5 are round, conservative values comparable
    to the Euclid Halpha push-forward (:class:`LFBiasPrior`, mean over 0.9<z<1.8:
    sigma(b_e)~0.85, sigma(Q)~0.3).

    Duck-types the ``components``/``covariance(zz, names)`` interface the forecast and sampler
    use, so an instance can be passed straight through as ``lf_prior``. For a bright/faint
    split pass ``components=['Xbe','XQ','Ybe','YQ']``; the 'X'/'Y' tag maps to the same base error.
    """

    def __init__(self, be: float = 1.0, Q: float = 0.5, components: list[str] = ("be", "Q")) -> None:
        self.errors = {"be": be, "Q": Q}
        self.components = list(components)

    def covariance(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        names = list(self.components if names is None else names)
        var = [self.errors[n[1:] if n[:1] in ("X", "Y") else n] ** 2 for n in names]
        return np.broadcast_to(np.diag(var), (np.atleast_1d(zz).size, len(names), len(names))).copy()
