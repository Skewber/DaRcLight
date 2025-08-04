"""Module to provide tools for spectroscopy"""
import logging
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from astroquery.nist import Nist
from astropy import units as u
from functools import lru_cache

logger = logging.getLogger(__name__)

def spec_norm(data:np.ndarray)->np.ndarray:
    """no"""
    logger.info("The spectroscopy norm was used")
    gauss = np.copy(data)
    gaussian_filter(gauss, sigma=5, output=gauss)
    scaled = np.copy(data)
    scaled = data - gauss
    scaled = scaled - np.min(scaled)
    scaled = scaled / np.max(scaled)
    return scaled

def gaussian(x, mean, stddev, amp, offset=0.):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

def fit_gauss(x, y):
    size = len(x)
    p0 = [x[size//2], size/2, np.max(y)-np.min(y), np.min(y)]
    bounds = [[x[0], 0., 0., np.min(y)],
              [x[-1], size, (np.max(y)-np.min(y))*1.2, np.max(y)]]
    popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds)
    return popt

class Calibrator():
    def __init__(self, elements=['Ne I', 'Ar I'], min_wavelength=4000., max_wavelength=8050):
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        nist = Nist.query(min_wavelength*u.AA, max_wavelength*u.AA,
                          linename=elements, wavelength_type='vac+air')
        
        def validate(value):
            """Removes all entries that have a letter in the relativ intensity"""
            try:
                return float(value)
            except Exception:
                return 0

        # validate the query results
        nist['Rel.'] = np.array([validate(val) for val in nist['Rel.']])
        nist = nist[np.where(nist['Rel.']!=0) and ~np.isnan(nist['Rel.'])]

        self.lines = {}
        self.intensities = {}
        for e in elements:
            self.lines[e] = np.array(nist[np.where(nist['Spectrum'] == e)]['Ritz'])
            self.intensities[e] = np.array(nist[np.where(nist['Spectrum'] == e)]['Rel.'])

    @staticmethod
    def find_brightest(spectrum, n=2):
        px = [2377.4726685051896, 2448.4087012508257]
        return px
    
    @staticmethod
    def _position_score(found_lines, lines):
        found_waves = sorted(found_lines)
        # outside: FWHM = 1. * (max-min)
        sigma_out = (found_waves[-1] - found_waves[0]) / (1 * np.sqrt(2 * np.log(2)))
        score = np.zeros(len(lines))
        for i, line in enumerate(lines):
            if line <= np.min(found_waves):
                score[i] = gaussian(line, mean=found_waves[0], stddev=sigma_out, amp=1., offset=0.)
            elif line >= np.max(found_waves):
                score[i] = gaussian(line, mean=found_waves[-1], stddev=sigma_out, amp=1., offset=0.)
            else:   # somwhere in between
                insert = np.digitize(line, found_waves)
                mean = (found_waves[insert-1] + found_waves[insert]) / 2
                # betwween two found lines: FWHM = 0.5 * diff
                sigma = (found_waves[insert] - found_waves[insert-1]) / (2 * np.sqrt(2 * np.log(2)))
                score[i] = gaussian(line, mean=mean, stddev=sigma, amp=1., offset=0.)
        score /= np.max(score)
        return score
    
    @staticmethod
    def _isolation_score(lines, min_spacing=0.1):
        wavelengths = np.sort(lines)  # Sort for neighbor distance computation
        score = np.zeros(len(wavelengths))
        
        # Compute distances to nearest neighbor
        for i in range(len(wavelengths)):
            if i == 0:
                dist = abs(wavelengths[i+1] - wavelengths[i])
            elif i == len(wavelengths) - 1:
                dist = abs(wavelengths[i] - wavelengths[i-1])
            else:
                dist = min(abs(wavelengths[i+1] - wavelengths[i]),
                        abs(wavelengths[i] - wavelengths[i-1]))
            score[i] = dist

        # Normalize scores: larger spacing = higher score
        # Prevent all scores from being zero
        score = np.clip(score, min_spacing, None)
        score /= np.max(score)  # Normalize to [0, 1]

        return score

    @staticmethod
    def find_next_line(found_lines, lines, intensities, mask):
        # calculate scores
        pos_score = Calibrator._position_score(found_lines, lines[mask])
        int_score = intensities[mask] / np.max(intensities)
        iso_score = Calibrator._isolation_score(lines)[mask]
        score = 3*iso_score + 3*pos_score + 2*int_score

        # update the mask
        best_subidx = np.argmax(score)
        best_fullidx = np.flatnonzero(mask)[best_subidx]
        mask[best_fullidx] = False
        return lines[best_fullidx], mask

    def search_lines(self, spectrum, init_pixel, init_waves, lines, intensities, mask, min_lines=15):
        pixels = np.arange(len(spectrum))
        # initial model for first guess
        coeffs = polyfit(init_waves, init_pixel, deg=1)
        wavelength_range = self.max_wavelength - self.min_wavelength
        # keep track of coverage to in order to know when to stop
        coverage = np.abs(np.max(init_waves) - np.min(init_waves)) / wavelength_range
        linecount = 0
        found_pixel = init_pixel.copy()
        found_waves = init_waves.copy()

        while coverage <= 0.9 or linecount < min_lines:
            next_line, mask = self.find_next_line(found_waves, lines, intensities, mask)
            expected_px = int(polyval(next_line, coeffs))
            # TODO: slice range determined dynamically
            s = slice(expected_px-10, expected_px+10)
            popt = fit_gauss(pixels[s], spectrum[s])
            found_pixel.append(popt[0])
            found_waves.append(next_line)

            # update the model
            # TODO: implement criteria for increasing the degree
            coeffs = polyfit(found_waves, found_pixel, deg=1)

            coverage = np.abs(np.max(found_waves) - np.min(found_waves)) / 4050
            linecount += 1

        # exclude the inital values
        found_pixel = found_pixel[2:]
        found_waves = found_waves[2:]
        return found_pixel, found_waves

    def find_calibration(self, spectrum, init_pixel, init_waves):
        found_lines = {}
        masks = [np.ones(len(self.lines[e]), dtype=bool) for e in self.lines]
        for (e, lines), intens, mask in zip(self.lines.items(), self.intensities.values(), masks):
            found_lines[e] = self.search_lines(spectrum, init_pixel, init_waves, lines, intens, mask, min_lines=20)
        return found_lines

    def __call__(self, pixels):
        self.coeffs = 0
        return polyval(pixels, self.coeffs)
