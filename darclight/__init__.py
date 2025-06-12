"""darclight.__init__.py"""
import logging
from .utils import enable_logging
from .io import DataCollection
from .reduction import Reducer, inv_median

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.propagate = False

__all__ = [
    "DataCollection",
    "Reducer",
    "inv_median",
    "enable_logging",
]
