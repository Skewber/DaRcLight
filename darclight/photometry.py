"""Module to provide tools for photometry"""
from glob import glob
import logging
from pathlib import Path
from warnings import warn
from typing import Tuple
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle
from astropy import units as u
from astropy.table import Table
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
import astroalign as aa
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)

def reference_from_file(fname:str, unit:Tuple[u.Unit, u.Unit])->Tuple:
    """reads the refenerce stars from a given file.

    :param fname: name of the file containing the reference data
    :type fname: str
    :param unit: unit for RA and Dec
    :type unit: Tuple[astropy.unit.Unit]
    :return: reference number, RA, Dec and one list for every band defined in the file
    :rtype: Tuple
    """
    star_id, ra, dec, *mags = np.genfromtxt(fname, dtype=str).T
    star_id = [int(sid) for sid in star_id]
    ra = Angle(ra, unit=unit[0]).degree
    dec = Angle(dec, unit=unit[1]).degree

    mags = [np.array(m, dtype=float) for m in mags]
    return star_id, ra, dec, *mags

def estimate_fwhm(data:np.ndarray)->float:
    """Function for approximating the FWHM value based on sources found in the provided data array.
    ! ! ! THIS FUNCTION IS NOT IMPLEMENTED YET ! ! !

    :param data: image data array to use for the estimation
    :type data: np.ndarray
    :return: value of the FWHM
    :rtype: float
    """
    return 4.5

class Photometer():
    """Class to perform basic photometry
    """
    def __init__(self, fwhm:float, output:str='output', target:str='none', fov:float=19.):
        self.fwhm = fwhm
        self.fov = fov
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)
        self.target = target
        self.ref_stars = {'ra':None,
                          'dec':None,
                          'xpixel':None,
                          'ypixel':None,
                          'mag':None}

    def __call__(self, data:np.ndarray)->Table:
        return Table()

    def find_ref_stars(self, data:np.ndarray, ref_ra:np.ndarray, ref_dec:np.ndarray,
                       gaia_data_file:str|None=None)->Tuple[list[float], list[float]]:
        """finds the pixel positions of reference stars in a given image.

        :param data: the image data where the reference stars should be found
        :type data: np.ndarray
        :param ref_ra: array of R.A. values in degree for the reference stars
        :type ref_ra: np.ndarray
        :param ref_dec: array of the declination values in degree for the reference stars
        :type ref_dec: np.ndarray
        :param max_mag: the limiting magnitude, used for a gaia query to find the reference stars, defaults to 16.
        :type max_mag: float, optional
        :param gaia_data_file: file that contains gaia data of the region. Needs to have the following three entries:
                'phot_g_mean_mag', 'ra' and 'dec', defaults to None
        :type gaia_data_file: str | None, optional
        :return: returns a Tuple of pixel positions in both axis
        :rtype: Tuple[list[float], list[float]]
        """
        # get the star position from the science data
        _, median, std = sigma_clipped_stats(data)
        fwhm = self.fwhm if self.fwhm is not None else estimate_fwhm(data)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=5.*std, brightest=50)
        sources = daofind(data - median)
        science_pixels = np.array((sources['xcentroid'], sources['ycentroid']))

        # get the star positions from gaia data
        mean_ra = np.mean(np.unwrap(ref_ra, period=360))
        mean_dec = np.mean(np.unwrap(ref_dec, period=360))

        cos = np.cos(np.deg2rad(mean_dec))
        dist = np.sqrt((mean_ra-ref_ra)**2 * cos**2 + (mean_dec-ref_dec)**2)
        # *2: twice the biggest distance from the center
        # *1.2: 10% more than the biggest distance since the ref stars are probably not at the very edge
        fov = np.max(dist) * 2.4
        logger.debug("FOV: %s arcmin", fov*60)
        gaia_data = self._load_gaia_data(gaia_data_file, mean_ra, mean_dec, fov/2)
        pixel_scale = fov / data.shape
        logger.debug("Approximate pixel scale: %s arcsec", pixel_scale*3600)
        gaia_pxx = data.shape[1]//2 + ((np.array(gaia_data['ra']) - mean_ra)*cos) / pixel_scale[1]
        gaia_pxy = data.shape[0]//2 + (np.array(gaia_data['dec']) - mean_dec) / pixel_scale[0]
        gaia_pixels = np.array((gaia_pxx, gaia_pxy))

        # find the necessary transformation and apply to reference pixel
        trans, *_ = aa.find_transform(gaia_pixels.T, science_pixels.T)
        ref_pxx = data.shape[1]//2 + ((ref_ra - mean_ra)*cos) / pixel_scale[1]
        ref_pxy = data.shape[0]//2 + (ref_dec - mean_dec) / pixel_scale[0]
        ref_pixels = trans(np.array((ref_pxx, ref_pxy)).T)
        self.ref_stars['xpixel'], self.ref_stars['ypixel'] = ref_pixels.T
        return ref_pixels

    def _load_gaia_data(self, file:str|None=None, ra:float=0, dec:float=0, radius:float=9.5)->Table:
        if file is not None:
            results = Table.read(file, format='fits')
            logger.info("Gaia data loaded from file '%s'", file)

        # try to find a file from an earlier run
        files = glob(f"{self.output}/*{self.target}*gaia*.fits")
        if len(files) == 1:
            results = Table.read(files[0], format='fits')
            logger.info("Gaia data loaded from file '%s'", files[0])
        elif len(files) > 1:
            message = "More than one file with gaia data was found matching the " \
            f"following expresion '{self.output}/*{self.target}*gaia*.fits'." \
            f"The first ({files[0]}) one will be used."
            warn(message)
            logger.warning(message)
            results = Table.read(files[0])
        # no file found --> db request
        else:
            logger.info("No file with Gaia data was provided or found. Starting DB request.")
            # caluclate the center of the reference stars
            logger.debug("Gaia query around: %s, %s with a radius of: %s", ra, dec, radius)

            query = f"""SELECT TOP 50 source_id, ra, dec, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
            )=1
            ORDER BY phot_g_mean_mag ASC"""

            job = Gaia.launch_job(query)
            results = job.get_results()

            # save the data for potential later use
            filename = f"{self.target}_gaia.fits"
            results.write(f"{self.output}/{filename}", format='fits', overwrite=True)

        return results

    def mark_stars(self, data:np.ndarray, pixels:np.ndarray):
        """Marks the reference stars in a given image with a circle and a number

        :param data: image data where the reference stars should be marked
        :type data: np.ndarray
        :param pixels: iterable that has the pixel positions in the form [[x1, y1], [x2, y2], ...]
        :type pixels: np.ndarray
        """
        height, width = data.shape
        dpi = 100
        _, ax = plt.subplots(figsize=(height/dpi, width/dpi), dpi=dpi)
        ax.imshow(data, cmap='grey', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))

        radius = self.fwhm * 3
        for i, (x, y) in enumerate(pixels, start=1):
            circ = Circle((x, y),radius, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(circ)
            ax.text(x+radius, y+radius, str(i), color='red', fontsize=25)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{self.output}/ref_stars_{self.target}.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
