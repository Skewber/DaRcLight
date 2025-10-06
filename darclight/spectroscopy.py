"""Module to provide tools for spectroscopy"""
import logging
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, rotate
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def spec_norm(data:np.ndarray)->np.ndarray:
    """Normalization function used for spectroscopic flat frames.

    :param data: flat frame that shoud be normalized
    :type data: np.ndarray
    :return: normalized flat frame
    :rtype: np.ndarray
    """
    logger.info("The spectroscopy norm was used")
    gauss = np.copy(data)
    gaussian_filter(gauss, sigma=5, output=gauss)
    scaled = np.copy(data)
    scaled = data - gauss
    scaled = scaled - np.min(scaled)
    scaled = scaled / np.max(scaled)
    return scaled

def gaussian(x:float|np.ndarray, mean:float=0., stddev:float=1., amp:float=1., offset:float=0.)->float|np.ndarray:
    """Gaussian function"""
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

def multigauss(x:float|np.ndarray, *args)->float|np.ndarray:
    """sum of multiple gauss functions, the arguments must be provided in
    the following order: mean1, sigma1, amplitude1, mean2, sigma2, amplitude2, ... (offset)
    If the number of arguments is of the form 3n+1 the last argument is interpreted as an offset.
    If the number of arguments is of the form 3n no offset is beeing added
    """
    ngauss = len(args) // 3
    y = np.full_like(x, args[-1], dtype=np.float64) if len(args)%3 == 1 else np.zeros_like(x, dtype=np.float64)
    for i in range(ngauss):
        mean, stddev, amp = args[3*i:3*i+3]
        y += amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    return y


class SpectrumImage():
    """Class to work with and manipulate spectroscopic images.
    """
    def __init__(self, image_data:np.ndarray)->None:
        if image_data.ndim != 2:
            raise ValueError("Provided data has invalid number of dimensions."+
                             f"Got data with {image_data.ndim} and expected 2!")
        self.image_data = image_data

    @classmethod
    def from_fits(cls, filename):
        data = np.asarray(fits.getdata(filename))
        return cls(data)

    @staticmethod
    def find_slit(image_data:np.ndarray)->tuple[int,int]:
        # TODO: add code for slit detection
        return 750, 1150

    def _find_rotation(self)->float:
        # TODO: add code for rotation detection
        return 0.

    def rotate(self, angle:float|None=None):
        """Applies a rotation to the image data.
        If no angle is provided the best angle is estimated base on the slope of the spectrum.
        If the slope is less than 2 pixel across the whole image no rotation is applied
        to reduce interpolation errors.

        :param angle: desired rotation angle, if the angle should be determined automatically use None,
            defaults to None
        :type angle: float | None, optional
        :return: a new SpectrumImage object with the rotated data
        :rtype: SpectrumImage
        """
        if angle is None:
            angle = self._find_rotation()

        rotated_data = rotate(self.image_data, angle)
        logger.info("Rotated image by %.2f degree", angle)
        return SpectrumImage(rotated_data)

    @staticmethod
    def extract_full_slit(data:np.ndarray)->np.ndarray:
        """Extracts the spectrum by taking the sigma clipped mean across the whole slit length

        :param data: image data containing the spectrum
        :type data: np.ndarray
        :return: 1D array representing the extracted spectrum
        :rtype: np.ndarray
        """
        start, end = SpectrumImage.find_slit(data)
        spec, *_ = sigma_clipped_stats(data[start:end, :], axis=0)
        return np.asarray(spec)

    @staticmethod
    def extract_point_source(data:np.ndarray,
                             src_frac:float=0.5, sky_frac:float=0.05
                             )->tuple[np.ndarray, np.ndarray]:
        """Extracts the spectrum of a single point source alongside the background by fitting a gaussian profile

        :param data: image data containing the spectrum
        :type data: np.ndarray
        :param src_frac: fraction above which the median is calculated for extracting the source
            higher fractions lead to the usage of a more narrow region taken into account, defaults to 0.5
        :type src_frac: float, optional
        :param sky_frac: fraction below which the background is used,
            higher values lead to the usage of values closer to the target,
            which might influence the resulting spectrum, defaults to 0.05
        :type sky_frac: float, optional
        :return: the extracted spectrum from the point source as well as the background in the image
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        start, end = SpectrumImage.find_slit(data)
        img = data[start:end, :]
        # extract the profile to determine the position of the source
        profile = np.median(img, axis=1)
        # fit gaussian to the point source
        x = np.arange(len(profile))
        p0 = [np.argmax(profile), 1., np.max(profile)-np.min(profile), np.min(profile)]
        bounds = [[0., 0., 0., 0.],
                  [x[-1], x[-1]/2, np.inf, np.max(profile)]]
        popt, *_ = curve_fit(gaussian, x, profile, p0=p0, bounds=bounds)

        # mask all values below the half amplitude
        mask_src = popt[3]+popt[2]*src_frac <= gaussian(x, *popt)     # offset + amplitude*0.5
        # background is everything below 5% of the amplitude + offset
        mask_bkg = gaussian(x, *popt) <= popt[3]+popt[2]*sky_frac       # offset + amplitude*0.05

        spec_target = np.median(img[mask_src, :], axis=0)
        spec_bkg = np.median(img[mask_bkg, :], axis=0)

        return spec_target, spec_bkg

    def extract(self, method='full_slit', **kwargs)->np.ndarray|tuple[np.ndarray,np.ndarray]:
        """Extracts the spectra from the image data provided

        :param method: method to use for the extraction valid methods are 'full_slit' and 'point_source',
            defaults to 'full_slit'
        :type method: str, optional
        :raises ValueError: if the provided method is none of the valid ones
        :return: if method='full_slit' a single 1D array with the spectrum,
            if method='point_source' a tuple with te source spectrum and the background spectrum
        :rtype: np.ndarray|tuple[np.ndarray,np.ndarray]
        """
        match method:
            case 'full_slit':
                spectra = self.extract_full_slit(self.image_data)
            case 'point_source':
                spectra = self.extract_point_source(self.image_data, **kwargs)
            case _:
                raise ValueError("Provided invalid method for extraction. You provided '{method}'")

        return spectra
