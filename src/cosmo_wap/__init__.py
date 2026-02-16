from importlib.metadata import version

from .main import ClassWAP
from .peak_background_bias import PBBias
from .survey_params import SurveyParams, SetSurveyFunctions
from . import forecast, lib, integrated

__version__ = version("cosmowap")
__all__ = ['ClassWAP', 'PBBias', 'SurveyParams', 'SetSurveyFunctions', 'forecast', 'lib', 'integrated']