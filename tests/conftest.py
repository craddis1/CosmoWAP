"""Shared fixtures for the CosmoWAP test suite.

Session-scoped so CLASS is initialised once across all test modules.
"""
import pytest
import numpy as np

from cosmo_wap.lib import utils
import cosmo_wap as cw
from cosmo_wap.forecast import FullForecast


@pytest.fixture(scope="session")
def cosmo():
    """CLASS cosmology object (Planck-like)."""
    return utils.get_cosmo(h=0.67, Omega_m=0.31, k_max=1.0, z_max=4.0)


@pytest.fixture(scope="session")
def survey_params(cosmo):
    """Euclid survey parameter set."""
    return cw.SurveyParams.Euclid(cosmo)


@pytest.fixture(scope="session")
def cosmo_funcs(cosmo, survey_params):
    """ClassWAP wrapper around cosmology + survey."""
    return cw.ClassWAP(cosmo, survey_params, verbose=False)


@pytest.fixture(scope="session")
def forecast(cosmo_funcs):
    """Small FullForecast (2 bins, kmax=0.1) for fast tests."""
    return FullForecast(cosmo_funcs, kmax_func=0.1, s_k=2, N_bins=2)


@pytest.fixture(scope="session")
def pk_bin(forecast):
    """PkForecast object for the first redshift bin."""
    return forecast.get_pk_bin(0)


@pytest.fixture(scope="session")
def bk_bin(forecast):
    """BkForecast object for the first redshift bin."""
    return forecast.get_bk_bin(0)
