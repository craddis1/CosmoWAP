from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from cosmo_wap.lib import utils
from cosmo_wap.lib.luminosity_funcs import fraction_cut

logger = logging.getLogger(__name__)


def count_slope(cosmo: object, n_g: ArrayLike, zz: np.ndarray) -> np.ndarray:
    """n_evol = d ln N/d ln(1+z), N = r^2 n_g/H the counts per z per solid angle - the observable
    SetSurveyFunctions.be derives b_e from. Units drop out of the log derivative."""
    N = cosmo.comoving_distance(zz) ** 2 / cosmo.Hubble(zz) * n_g
    return np.gradient(np.log(N), np.log(1 + zz), edge_order=2 if len(zz) > 2 else 1)


class LFBiasPrior:
    """Forward-model luminosity-function fit-parameter uncertainties onto a prior on the
    selection functions - the count slope n_evol(z) (or evolution bias b_e(z)) and magnification bias Q(z).

    Monte-Carlo push-forward (cf. HorizonGRound; Wang, Beutler & Bacon 2020,
    arXiv:2007.01802): the luminosity-function fit parameters carry (here diagonal) errors,
    so we draw them from their error distribution, recompute the selection functions for
    each draw and summarise the resulting joint distribution. The per-redshift covariance
    is then used as a prior on n_evol (or b_e) and Q in the Fisher forecast and the MCMC sampler.

    The forward model is described by a set of named ``components`` and an ``evaluate``
    callback that, given a luminosity function with sampled parameters, returns each
    component on the dense ``z_grid``. For a single tracer the components are ('n_evol', 'be', 'Q') -
    the forecast takes the ones in its per_bin_params, so either selection basis works (see
    Forecast.set_selection_basis); for a bright/faint split they are the per-tracer
    ('Xn_evol', 'Xbe', 'XQ', 'Yn_evol', 'Ybe', 'YQ') - a fixed-fraction split shares one 'n_evol' - with
    X = bright (tracer 0) and Y = faint (tracer 1) - their correlation through the shared
    luminosity function is captured automatically. Use :meth:`from_survey` to build these
    from a survey; the constructor is the lower-level entry point.

    The width and the n_evol-Q correlation this gives are set by the fitted functional form as much as by the
    data (Model 1 and Model 3 H-alpha differ by ~2x in sigma(n_evol)) - see SelectionPrior for a prior on the
    selection functions as a catalogue measures them.

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

        # fiducial parameter vector (mean of the draws) - for each free parameter
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
            self.param_cov = np.diag(std**2)  # diagonal parameter covariance

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
        per-tracer components ('Xn_evol', 'Xbe', 'XQ', 'Yn_evol', 'Ybe', 'YQ'), reproducing the survey's faint
        derivation (see survey_params._get_faint) per draw - and a fixed-fraction split from ``survey.fraction``,
        re-solving its moving bright cut per draw (survey_params._get_fraction).

        Accepts either a survey or the ``[bright, faint]`` list returned by
        ``survey.BF_split`` - both tracers carry the shared LF/cut/split, so the split is
        forward-modelled from the parent survey in either case.
        """
        if isinstance(survey, (list, tuple)):
            survey = next((s for s in survey if hasattr(s, "LF")), survey[0])
        if not hasattr(survey, "LF"):
            raise ValueError("Survey has no luminosity function (survey.LF) - cannot forward-model fit errors.")

        LF = survey.LF
        cut = survey.cut
        # dense grid matching the survey's own bias grid where possible
        z_grid = np.asarray(getattr(survey, "zz", np.linspace(np.min(LF.z_values), np.max(LF.z_values), 100)))

        split, fraction = getattr(survey, "split", None), getattr(survey, "fraction", None)
        if fraction is not None:
            # fixed-fraction split: the bright cut moves with z to keep n_B = fraction n_T, so is solved again per
            # draw. Mirrors survey_params._get_fraction, but with one cut per redshift as an array - which the
            # flux-limited LFs, the only ones with fit_params, broadcast. n_evol is common to bright, faint and total
            components = ["n_evol", "Xbe", "XQ", "Ybe", "YQ"]

            def evaluate(lf):
                n_T, Q_T = lf.get_nQ(cut, z_grid)
                dlnL = fraction_cut(lambda d: lf.number_density(lf.shift_cut(cut, d), z_grid), fraction * n_T)
                D = np.gradient(dlnL, np.log(1 + z_grid), edge_order=2 if len(z_grid) > 2 else 1)
                n_B, n_F = fraction * n_T, (1 - fraction) * n_T
                Q_B = lf.get_Q(lf.shift_cut(cut, dlnL), z_grid)
                Q_F = utils.get_faint_bias(z_grid, n_T, n_B, Q_T, Q_B)(z_grid)
                be_B = lf.get_be(None, z_grid, n_g=n_B, Q=Q_B) - D * Q_B
                be_F = lf.get_be(None, z_grid, n_g=n_F, Q=Q_F) + fraction / (1 - fraction) * D * Q_B
                return np.array([count_slope(lf.cosmo, n_T, z_grid), be_B, Q_B, be_F, Q_F])
        elif split is None:
            components = ["n_evol", "be", "Q"]

            def evaluate(lf):
                n_g, Q = lf.get_nQ(cut, z_grid)
                return np.array([count_slope(lf.cosmo, n_g, z_grid), lf.get_be(cut, z_grid, n_g=n_g, Q=Q), Q])
        else:
            # bright/faint split: X = bright (cut=split), Y = faint (total minus bright).
            # Mirrors survey_params._get_faint but only for the n_evol/b_e/Q we need.
            components = ["Xn_evol", "Xbe", "XQ", "Yn_evol", "Ybe", "YQ"]

            def evaluate(lf):
                n_T, Q_T = lf.get_nQ(cut, z_grid)
                n_B, Q_B = lf.get_nQ(split, z_grid)
                be_B = lf.get_be(split, z_grid, n_g=n_B, Q=Q_B)
                n_F = n_T - n_B
                Q_F = utils.get_faint_bias(z_grid, n_T, n_B, Q_T, Q_B)(z_grid)
                be_F = lf.get_be(None, z_grid, n_g=n_F, Q=Q_F)
                nev_B, nev_F = count_slope(lf.cosmo, n_B, z_grid), count_slope(lf.cosmo, n_F, z_grid)
                return np.array([nev_B, be_B, Q_B, nev_F, be_F, Q_F])

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


class SelectionPrior:
    """Gaussian prior on the per-bin (n_evol, Q) as a catalogue measures them, rather than as a fitted
    luminosity function constrains them: n_evol = d ln N/d ln(1+z) from the binned counts, Q from the counts
    about the flux cut (or source injection) - so the two errors are independent, and of any functional form.
    Absolute sigma_n on n_evol, which crosses zero at the N(z) peak, and fractional sigma_Q on each measured Q.

    For a bright/faint split what is measured is the total's Q_T and the bright's Q_B - the faint's follows,
    Q_F = (Q_T - r Q_B)/(1 - r) with r = n_B/n_T (2406.19908 eq. 3.13). The prior on ('XQ', 'YQ') is that pair
    propagated, which correlates them and inflates sigma(Q_F) by ~1/(1 - r). n_evol takes sigma_n whether
    common ('n_evol') or per tracer ('Xn_evol'/'Yn_evol', fixed-cut split only).

    Duck-types the ``components``/``covariance(zz, names)`` interface of LFBiasPrior - pass it as ``lf_prior``.
    The fiducial Q and n_g are read off the tracers of cosmo_funcs when it is built.
    """

    def __init__(self, cosmo_funcs: object, sigma_n: float = 0.5, sigma_Q: float = 0.05) -> None:
        self.sigma_n, self.sigma_Q = sigma_n, sigma_Q
        tracers = {["X", "Y"][s.t]: s for s in set(cosmo_funcs.survey)}
        # held now - the forecast and sampler shift these attributes on the tracers in place
        self.Q = {k: s.Q for k, s in tracers.items()}
        self.n_g = {k: s.n_g for k, s in tracers.items()}

        self.split = cosmo_funcs.multi_tracer
        if self.split:
            survey_params = cosmo_funcs.survey_params
            survey_params = survey_params if isinstance(survey_params, (list, tuple)) else [survey_params]
            if all(getattr(s, "split", None) is None and getattr(s, "fraction", None) is None for s in survey_params):
                raise ValueError("SelectionPrior propagates (Q_T, Q_B) - the two tracers must be a BF_split.")
            self.components = ["n_evol", "Xn_evol", "Yn_evol", "XQ", "YQ"]
        else:
            self.components = ["n_evol", "Q"]

    def Q_covariance(self, zz: ArrayLike) -> np.ndarray:
        """Covariance of the per-tracer Q at each redshift - (len(zz), 1, 1), or (len(zz), 2, 2) over (Q_B, Q_F)."""
        zz = np.atleast_1d(zz)
        if not self.split:
            return (self.sigma_Q * self.Q["X"](zz))[:, None, None] ** 2

        Q_B, Q_F = self.Q["X"](zz), self.Q["Y"](zz)
        n_B, n_F = self.n_g["X"](zz), self.n_g["Y"](zz)
        r = n_B / (n_B + n_F)
        var_T, var_B = (self.sigma_Q * (r * Q_B + (1 - r) * Q_F)) ** 2, (self.sigma_Q * Q_B) ** 2

        cov = np.empty((zz.size, 2, 2))
        cov[:, 0, 0] = var_B
        cov[:, 0, 1] = cov[:, 1, 0] = -r / (1 - r) * var_B
        cov[:, 1, 1] = (var_T + r**2 * var_B) / (1 - r) ** 2
        return cov

    def covariance(self, zz: ArrayLike, names: list[str] | None = None) -> np.ndarray:
        zz = np.atleast_1d(zz)
        names = list(self.components if names is None else names)
        cov = np.zeros((zz.size, len(names), len(names)))

        Q_cov = self.Q_covariance(zz)
        q_slot = {"Q": 0} if not self.split else {"XQ": 0, "YQ": 1}
        for i, a in enumerate(names):
            if a.endswith("n_evol"):
                cov[:, i, i] = self.sigma_n**2
            for j, b in enumerate(names):
                if a in q_slot and b in q_slot:
                    cov[:, i, j] = Q_cov[:, q_slot[a], q_slot[b]]
        return cov
