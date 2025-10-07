"""Module to provide utility tools for the reduction and analysis process."""
import logging
import os
import numpy as np
from scipy.ndimage import rotate
from astropy.io import fits
from datetime import datetime

logger = logging.getLogger(__name__)

_logging_enabled = False

def enable_logging(file:bool=True, console:bool=False, filename:str='log'):
    """later"""
    global _logging_enabled

    if _logging_enabled:
        logger.info("Logging is already enabled.")
        return None
    
    local_logger = logging.getLogger('darclight')
    local_logger.setLevel(logging.DEBUG)

    if local_logger.hasHandlers():
        local_logger.info("Logging is already enabled.")
        return None

    if file:
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logfile = os.path.join(log_dir, f"{filename}_{timestamp}.log")

        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        local_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        local_logger.addHandler(console_handler)

    logger.info("Logging has been enabled.")
    _logging_enabled = True


class ImageData():
    def __init__(self, data:np.ndarray)->None:
        if data.ndim != 2:
            raise ValueError("Provided data has invalid number of dimensions."+
                             f"Got data with {data.ndim} and expected 2!")
        self.data = data

        # clean the data by removing nans and infs
        # TODO: include values on the edge
        # TODO: higher order interpolation to reduce the effect of close bright sources
        pxx, pxy = np.where(np.isnan(data))
        logger.info("Found and interpolated %s nan values", len(pxx))
        for x, y in zip(pxx, pxy):
            self.data[x, y] = (data[x-1, y] +  data[x+1, y] + data[x, y-1] + data[x, y+1]) / 4
        pxx, pxy = np.where(np.isinf(data))
        logger.info("Found and interpolated %s inf values", len(pxx))
        for x, y in zip(pxx, pxy):
            self.data[x, y] = (data[x-1, y] +  data[x+1, y] + data[x, y-1] + data[x, y+1]) / 4

    @classmethod
    def from_fits(cls, filename:str):
        """Creates the class from a fits file.

        :param filename: name of the fits file to use
        :type filename: str
        :return: Instance of the ImageData
        :rtype: ImageData
        """
        data = np.asarray(fits.getdata(filename))
        return cls(data)

    def rotate(self, angle:float):
        """Rotates the data by a given angle

        :param angle: angle to rotate the image in degree
        :type angle: float
        :return: returns itself
        :rtype: ImageData
        """
        self.data = rotate(self.data, angle)
        return self
