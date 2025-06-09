"""Module to provide tools for photometry"""
from glob import glob
import logging
from pathlib import Path
from warnings import warn
from typing import Tuple
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
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

class Photometer():
    """Class to perform basic photometry
    """
    def __init__(self, fwhm:float, output:str='output', target:str='none', fov:float=19.):
        self.fwhm = fwhm
        self.fov = fov*u.arcmin
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
                       max_mag:float=16., gaia_data_file:str|None=None)->Tuple[list[float], list[float]]:
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
        # get star positions from data
        _, median, std = sigma_clipped_stats(data)
        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=5.*std, brightest=50)
        sources = daofind(data - median)
        science_pixels = np.array((sources['xcentroid'], sources['ycentroid']))

        # create artificial image
        mean_ra = np.mean(np.unwrap(ref_ra, period=360))
        mean_dec = np.mean(np.unwrap(ref_dec, period=360))
        center = SkyCoord(ra=mean_ra, dec=mean_dec, unit=(u.deg, u.deg))
        # load the 50 brightest stars from gaia
        gaia_data = self._load_gaia_data(gaia_data_file, center, max_mag)
        bright_idx = np.argsort(gaia_data['phot_g_mean_mag'])[:50]
        gaia_data = gaia_data[bright_idx]

        # setup for the wcs
        pixel_scale = self.fov.value / data.shape[0]
        fwhm = self.fwhm / 3600     # convert to degrees

        # create wcs to find artificial star positions
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [data.shape[1]/2, data.shape[0]/2]
        wcs.wcs.cdelt = [pixel_scale, pixel_scale]
        wcs.wcs.crval = [center.ra.deg, center.dec.deg]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # find gaia pixel positions
        gaia_coords = SkyCoord(ra=gaia_data['ra'], dec=gaia_data['dec'], unit=(u.deg, u.deg))
        gaia_pixels = np.array(wcs.world_to_pixel(gaia_coords))

        # find the transform and apply to reference pixels
        transform, *_ = aa.find_transform(gaia_pixels.T, science_pixels.T)
        refs = SkyCoord(ra=ref_ra, dec=ref_dec, unit=(u.deg, u.deg))
        ref_pixels = np.array(wcs.world_to_pixel(refs))
        ref_trans = transform(ref_pixels.T)
        return ref_trans

    def mark_ref_stars(self, data:np.ndarray, pixels:np.ndarray):
        """Marks the reference stars in a given image with a circle and a number

        :param data: image data where the reference stars should be marked
        :type data: np.ndarray
        :param pixels: iterable that has the pixel positions in the form [[x1, y1], [x2, y2], ...]
        :type pixels: np.ndarray
        """
        height, width = data.shape
        dpi = 100
        fig, ax = plt.subplots(figsize=(height/dpi, width/dpi), dpi=dpi)
        ax.imshow(data, cmap='grey', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))

        radius = self.fwhm * 3
        for i, (x, y) in enumerate(pixels, start=1):
            circ = Circle((x, y),radius, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(circ)
            ax.text(x+radius, y+radius, str(i), color='red', fontsize=25)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{self.output}/ref_stars.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _load_gaia_data(self, file:str|None=None, center:SkyCoord|None=None, max_mag:float=16)->Table:
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
            radius = np.sqrt(2*(self.fov/2)**2)

            query = f"""SELECT source_id, ra, dec, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {center.ra.degree}, {center.dec.degree}, {radius.to(u.deg).value})
            )=1
            AND phot_g_mean_mag < {max_mag}"""

            job = Gaia.launch_job(query)
            results = job.get_results()

            # save the data for potential later use
            filename = f"{self.target}_gaia.fits"
            results.write(f"{self.output}/{filename}", format='fits', overwrite=True)
            with fits.open(filename, 'update') as hdul:
                hdul[0].header['OBJECT'] = self.target

        # FIXME: account for the case that less than 50 stars are found
        # only select the brightes 50 stars
        bright_idx = np.argsort(results['phot_g_mean_mag'])[:50]
        results = results[bright_idx]

        return results
