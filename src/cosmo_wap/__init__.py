from importlib.metadata import version

from . import forecast, integrated, lib
from .main import ClassWAP
from .peak_background_bias import PBBias
from .survey_params import SetSurveyFunctions, SurveyParams

__version__ = version("cosmowap")
__all__ = ["ClassWAP", "PBBias", "SurveyParams", "SetSurveyFunctions", "forecast", "lib", "integrated"]
