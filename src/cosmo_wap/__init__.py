from importlib.metadata import version

from . import forecast, lib
from .HOD.peak_background_bias import PBBias
from .main import ClassWAP
from .survey_params import SetSurveyFunctions, SurveyParams

__version__ = version("cosmowap")
__all__ = ["ClassWAP", "PBBias", "SurveyParams", "SetSurveyFunctions", "forecast", "lib"]
