from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from cosmo_wap.lib import utils

logger = logging.getLogger(__name__)


class LumFuncBiasPrior:
    """Forward-model luminosity-function fit-parameter uncertainties onto a prior on
    evolution bias b_e(z) and magnification bias Q(z).

    Monte-Carlo push-forward (cf. HorizonGRound; Wang, Beutler & Bacon 2020,
    arXiv:2007.01802): the luminosity-function fit parameters carry (here diagonal)
    errors, so we draw them from their error distribution, recompute b_e(z), Q(z) for
    each draw via the existing ``LF.get_be`` / ``LF.get_Q`` and summarise the resulting
    joint (b_e, Q) distribution. The per-redshift covariance is then used as a prior on
    b_e/Q in the Fisher forecast and the MCMC sampler.

    The fit parameters are the ``fit_params`` attributes of the luminosity function (see
    e.g. ``Model3LuminosityFunction``); samples are drawn from a Gaussian centred on the
    fiducial values with covariance built from the diagonal errors (or a supplied full
    covariance for correlated parameters).

    Parameters:
    -----------
    LF : luminosity function instance
        Must expose ``fit_params`` (list of sampleable attribute names), the fiducial
        values as attributes, and ``get_be(cut, zz)`` / ``get_Q(cut, zz)``.
    cut : float
        Flux/magnitude cut passed to ``get_be``/``get_Q``.
    errors : dict, optional
        Map ``{param_name: error}`` at ``sigma_level`` sigma. Defaults to ``LF.fit_errors``.
    cov : array, optional
        Full (1-sigma) parameter covariance, ordered as ``LF.fit_params``. Overrides
        ``errors`` and enables correlated draws.
    sigma_level : float
        Confidence level of ``errors`` (default 2, i.e. ``errors`` are 2-sigma).
    n_samples : int
        Number of Monte-Carlo draws.
    z_grid : array, optional
        Dense redshift grid on which b_e/Q are evaluated before interpolation (defaults
        to a 100-point grid spanning ``LF.z_values``). Kept dense so the ``np.gradient``
        in ``get_be`` stays accurate; outputs are splined to the requested redshifts.
    seed : int, optional
        Seed for the random generator (reproducibility).
    """

    def __init__(
        self,
        LF: object,
        cut: float,
        *,
        errors: dict | None = None,
        cov: ArrayLike | None = None,
        sigma_level: float = 2.0,
        n_samples: int = 1000,
        z_grid: ArrayLike | None = None,
        seed: int | None = None,
    ) -> None:
        if not hasattr(LF, "fit_params"):
            raise ValueError("Luminosity function has no 'fit_params' - cannot forward-model fit errors.")

        self.LF = LF
        self.cut = cut
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

        # dense grid for evaluating the (gradient-based) biases before interpolation
        if z_grid is None:
            z_grid = np.linspace(np.min(LF.z_values), np.max(LF.z_values), 100)
        self.z_grid = np.asarray(z_grid, dtype=float)

        rng = np.random.default_rng(seed)
        self._draws = rng.multivariate_normal(self.mean_params, self.param_cov, size=n_samples)

        # evaluate b_e(z), Q(z) on the dense grid for every parameter draw (done once)
        self._be_grid, self._Q_grid = self._push_forward()

    def _push_forward(self) -> tuple[np.ndarray, np.ndarray]:
        """Recompute b_e(z), Q(z) on ``z_grid`` for each parameter draw.

        Returns two (n_valid, len(z_grid)) arrays; draws giving non-finite biases (e.g.
        a pathological slope making the luminosity integral diverge) are dropped.
        """
        be_samples = np.empty((self.n_samples, self.z_grid.size))
        Q_samples = np.empty((self.n_samples, self.z_grid.size))

        for i, draw in enumerate(self._draws):
            lf = utils.copy(self.LF)  # preserves the shared cosmo reference
            for name, value in zip(self.params, draw):
                setattr(lf, name, value)
            Q = lf.get_Q(self.cut, self.z_grid)
            be = lf.get_be(self.cut, self.z_grid, Q=Q)
            be_samples[i] = be
            Q_samples[i] = Q

        valid = np.all(np.isfinite(be_samples), axis=1) & np.all(np.isfinite(Q_samples), axis=1)
        n_drop = np.count_nonzero(~valid)
        if n_drop:
            logger.warning(
                "LumFuncBiasPrior: dropped %d/%d non-finite b_e/Q samples.", n_drop, self.n_samples
            )
        return be_samples[valid], Q_samples[valid]

    def sample(self, zz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the b_e and Q sample arrays at redshift(s) ``zz``.

        Each output has shape ``(n_valid_samples, len(zz))`` (splined from the dense grid).
        """
        zz = np.atleast_1d(zz)
        be = CubicSpline(self.z_grid, self._be_grid, axis=1)(zz)
        Q = CubicSpline(self.z_grid, self._Q_grid, axis=1)(zz)
        return be, Q

    def mean(self, zz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Sample mean of b_e and Q at ``zz`` (each shape ``(len(zz),)``)."""
        be, Q = self.sample(zz)
        return be.mean(axis=0), Q.mean(axis=0)

    def std(self, zz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Sample standard deviation of b_e and Q at ``zz`` (each shape ``(len(zz),)``)."""
        be, Q = self.sample(zz)
        return be.std(axis=0), Q.std(axis=0)

    def covariance(self, zz: ArrayLike) -> np.ndarray:
        """Joint (b_e, Q) covariance at each redshift.

        Returns an array of shape ``(len(zz), 2, 2)`` where entry ``k`` is the 2x2
        covariance of ``(b_e(z_k), Q(z_k))`` over the parameter draws - this captures the
        b_e-Q correlation induced by the shared luminosity-function parameters.
        """
        be, Q = self.sample(zz)
        cov = np.empty((be.shape[1], 2, 2))
        for k in range(be.shape[1]):
            cov[k] = np.cov(np.vstack([be[:, k], Q[:, k]]))
        return cov

    @classmethod
    def from_survey(cls, survey: object, **kwargs) -> "LumFuncBiasPrior":
        """Build the prior straight from a survey carrying a luminosity function.

        Reads ``survey.LF`` and ``survey.cut`` (set in ``survey_params`` via
        ``compute_luminosity``); the fit errors default to ``survey.LF.fit_errors``.
        """
        if not hasattr(survey, "LF"):
            raise ValueError("Survey has no luminosity function (survey.LF) - cannot forward-model fit errors.")
        return cls(survey.LF, survey.cut, **kwargs)
