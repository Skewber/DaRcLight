"""Module to manage general information like path to data"""
import os
from glob import iglob
from fnmatch import fnmatch
from collections.abc import Iterable
from collections import defaultdict
import logging
from functools import cached_property
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

logger = logging.getLogger(__name__)

COMMON_KEYWORDS = {'BIAS':['bias', 'zero'],
                   'DARK':['dark'],
                   'FLAT':['flat'],
                   'LIGHT':['light', 'science', 'object']}


class DataCollection():
    """Class that organizes all files in a given directory.
    """
    def __init__(self, path:str|None=None, reduced_path:str|None=None,
                 ignore:list|None=None, filelist:list[str]|None=None):
        # ensure that both paths are an Path object
        if path is None:
            self.path = Path('.')
        else:
            self.path = Path(path)

        if reduced_path is None:
            self.reduced_path = Path('./reduced')
        else:
            self.reduced_path = Path(reduced_path)
        # create the path if it does not exist
        self.reduced_path.mkdir(exist_ok=True)

        self.ignore = [] if ignore is None else ignore
        self.raw_files = Table()
        self.reduced_files = Table()
        self.scan(raw=True, reduced=True)

        logger.debug("DataCollection created")

    @staticmethod
    def _get_imagetype(keyword:str)->str:
        keyword = keyword.lower()

        for typ, common_kwd in COMMON_KEYWORDS.items():
            if any([kwd in keyword for kwd in common_kwd]):
                return typ
        raise RuntimeError('The keyword "%s" could not be matched to any known type. ' \
        'Check the word or add a value to the "COMMON_KEYWORDS" variable.')

    def _scan_raw(self):
        # recreate the table
        self.raw_files = Table(names=('ID', 'FILENAME',
                                    'TYPE', 'DATE-OBS',
                                    'NIGHT', 'JD',
                                    'EXPOSURE', 'FILTER',
                                    'OBJECT'),
                            dtype=('i4', 'U',
                                    'U', 'U',
                                    'U', 'f8',
                                    'f4', 'U',
                                    'U'))
        # check files
        for file in iglob(str(self.path)+'/**/*', recursive=True):
            if (os.path.isdir(file) or
                any((fnmatch(file, pat) for pat in self.ignore)) or
                fnmatch(file, f"*{self.reduced_path}*")):
                # skip if directory or contains a pattern from the ignore list or is an reduced file
                continue

            hdr = fits.getheader(file)
            self.raw_files.add_row((0,
                                    file,
                                    self._get_imagetype(hdr.get('IMAGETYP', 'None')),
                                    hdr.get('DATE-OBS', '0000-00-00T00:00:00'),
                                    '0000-00-00',
                                    hdr.get('JD', 0),
                                    hdr.get('EXPOSURE', -1),
                                    hdr.get('FILTER', 'None'),
                                    hdr.get('OBJECT', 'None')))

    def _scan_reduced(self):
        self.reduced_files = Table(names=('ID', 'FILENAME',
                                        'TYPE', 'DATE',
                                        'NIGHT', 'EXPOSURE',
                                        'FILTER', 'OBJECT',
                                        'COMBINED'),
                                dtype=('i4', 'U',
                                        'U', 'U',
                                        'U', 'f4',
                                        'U', 'U',
                                        'bool'))
        for file in iglob(str(self.reduced_path)+'/**/*', recursive=True):
            if (os.path.isdir(file) or
                any((fnmatch(file, pat) for pat in self.ignore))):
                continue

            hdr = fits.getheader(file)
            self.reduced_files.add_row((0,
                                        file,
                                        self._get_imagetype(hdr['IMAGETYP']),
                                        hdr.get('DATE', '1999-01-01T00:00:00.000'),
                                        '0000-00-00',
                                        hdr.get('EXPOSURE', -1),
                                        hdr.get('FILTER', 'None'),
                                        hdr.get('OBJECT', 'None'),
                                        hdr.get('COMBINED', False)))


    def scan(self, raw:bool=True, reduced:bool=True):
        """scans the directories from scractch

        :param raw: whether or not the raw directory should be scaned, defaults to True
        :type raw: bool, optional
        :param reduced: whether or not the reduced directory should be scanned, defaults to True
        :type reduced: bool, optional
        """
        if raw:
            self._scan_raw()
            # add value for the night --> avoid wrap over during midnight
            self.raw_files['NIGHT'] = np.array((
                Time(self.raw_files['DATE-OBS'])-12*u.hour
                ).to_value('iso', subfmt='date'))
            # sort by time and add unique ID to each file
            self.raw_files.sort('DATE-OBS')
            self.raw_files['ID'] = np.arange(len(self.raw_files))
            # group the files by necessary values
            # self.raw_files = self.raw_files.group_by(['TYPE', 'NIGHT', 'FILTER', 'EXPOSURE', 'TARGET'])
        if reduced:
            self._scan_reduced()
            # only update the data if at least one file is found
            if len(self.reduced_files) > 0:
                self.reduced_files['NIGHT'] = np.array((
                    Time(self.reduced_files['DATE'])-12*u.hour
                ).to_value('iso', subfmt='date'))
                self.reduced_files.sort('DATE')
                self.reduced_files['ID'] = np.arange(len(self.reduced_files))

    def add_file(self, fname:str|Path, reduced:bool=False, row:tuple|None=None):
        """adds a row to an existing table

        :param fname: filename to add
        :type fname: str | Path
        :param reduced: if it should be added to the raw or reduced table, defaults to False
        :type reduced: bool, optional
        :param row: row informations, depends on raw or reduced what it should contain.
                    If None the file will be read and the values are derived automatically, defaults to None
        :type row: tuple | None, optional
        """
        table = self.reduced_files if reduced else self.raw_files
        fname = str(fname)

        if row is None:
            hdr = fits.getheader(fname)
            date = hdr.get('DATE', '2000-01-01T00:00:00.000')
            if reduced:
                row = (len(table)+1,
                        fname,
                        self._get_imagetype(hdr['IMAGETYP']),
                        date,
                        (Time(date)-12*u.hour).to_value('iso', subfmt='date'),
                        hdr.get('EXPOSURE', -1),
                        hdr.get('FILTER', 'None'),
                        hdr.get('OBJECT', 'None'),
                        hdr.get('COMBINED', False))
            else:
                row = (len(table)+1,
                        fname,
                        self._get_imagetype(hdr.get('IMAGETYP', 'None')),
                        hdr.get('DATE-OBS', '2000-01-01T00:00:00'),
                        (Time(date)-12*u.hour).to_value('iso', subfmt='date'),
                        hdr.get('JD', 0),
                        hdr.get('EXPOSURE', -1),
                        hdr.get('FILTER', 'None'),
                        hdr.get('OBJECT', 'None'))

        table.add_row(row)

    @staticmethod
    def hdu_from_file(file:str)->Tuple[np.ndarray, fits.header.Header]:
        """gives access to the data and header of a given file

        :param file: name of the file
        :type file: str
        :return: the data and header from that file
        :rtype: Tuple[np.ndarray,astropy.io.fits.header.Header]
        """
        with fits.open(file) as hdul: # type: ignore
            data = hdul[0].data
            header = hdul[0].header
            return data, header

    @staticmethod
    def save_file(filename:str|Path, data:np.ndarray, header:fits.header.Header|None=None)->None:
        """Saves the given data and header with the given filename in the reduced data directory.

        :param filename: Desired name for the file
        :type filename: str
        :param data: data that should be stored in the file
        :type data: np.ndarray
        :param header: header for the file, defaults to None
        :type header: astropy.io.fits.header.Header, optional
        """
        hdu = fits.PrimaryHDU(data, header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=True)
        logger.debug("Saved the file '%s'", filename)

    @property
    def used_filters(self)->list[str]:
        """List of used filters where a flat frame is available.

        :return: list of filters
        :rtype: list[str]
        """
        flats = self.raw_files[self.raw_files['TYPE']=='FLAT']
        return list({str(f) for f in flats['FILTER']})

    @property
    def dark_exposures(self)->list[int]:
        """List of exposure times where a dark frame is directly available.

        :return: list of exposure times
        :rtype: list[int]
        """
        darks = self.raw_files[self.raw_files['TYPE']=='DARK']
        return list({int(e) for e in darks['EXPOSURE']})

    @property
    def flat_exposures(self)->dict[str|None,set]:
        """exposure times for each filter

        :return: dictionary of the form {filter:exposure, ...}
        :rtype: dict[str|None,set]
        """
        result = defaultdict(set)

        flats = self.raw_files[self.raw_files['TYPE']=='FLAT']
        for filt, exp in flats['FILTER', 'EXPOSURE']:
            result[str(filt)].add(int(exp))
        return dict(result)

    @property
    def targets(self)->list[str]:
        """List of targets captured.

        :return: list of targets
        :rtype: list[str]
        """
        lights = self.raw_files[self.raw_files['TYPE']=='LIGHT']
        return [str(obj) for obj in set(lights['OBJECT'])]

    @cached_property
    def light_meta(self)->dict[str,set[Tuple[str, int]]]:
        """Metadata for the light frames

        :return: dictionary of the form {target:[(filter,exposure),...],...},
                for every target there is a set of tuples that each contain the filter and
                the corresponding exposure time.
        :rtype: dict[str,set[Tuple[str, int]]]
        """
        result = defaultdict(set)

        lights = self.raw_files[self.raw_files['TYPE']=='LIGHT']
        for target, filt, exp in lights['OBJECT', 'FILTER', 'EXPOSURE']:
            if exp is not None and filt is not None:
                result[target].add((filt, int(exp)))
        logger.debug("Created meta data for lights:\n\t%s", result)
        return dict(result)

    def get_files(self, reduced:bool=False, **filters)->np.ndarray:
        """returns the filenames of the files specified

        :param reduced: whether the desired files should be reduced or not, defaults to False
        :type reduced: bool, optional
        :return: filenames satisfy the filters
        :rtype: np.ndarray
        """
        table = self.reduced_files if reduced else self.raw_files

        mask = np.ones(len(table), dtype=bool)
        for filt, val in filters.items():
            mask &= table[filt.upper()] == val

        files = np.array(table['FILENAME'][mask])
        files = files[0] if len(files)==1 else files
        return files

    @property
    def bias_files(self):
        return self.get_files(type='BIAS')
    
    @cached_property
    def dark_files(self):
        return {expo:self.get_files(type='DARK', exposure=expo) for expo in self.dark_exposures}

    @cached_property
    def flat_files(self):
        return {filt:self.get_files(type='FLAT', filter=filt) for filt in self.used_filters}

    def get_master(self, imagetype:str, specifier:int|str|None=None, header=True)->np.ndarray|tuple|None:
        """looks for a specific stacked master frame

        :param imagetype: type of the image to check, use 'bias', 'dark', 'flat' or 'light'
        :type imagetype: str
        :param specifier: exposure time to look for, if imagetype='dark' or
          filter to look for if imagetype='flat' or target if imagetype='light,
          defaults to None
        :type specifier: int | str | None, optional
        :return: data of the required file, None if it does not exist
        :rtype: np.ndarray | None
        """
        combined = np.array(self.reduced_files['COMBINED'])
        match imagetype.lower():
            case 'bias':
                mask = np.array(self.reduced_files['TYPE']=='BIAS') & combined
            case 'dark':
                if specifier is None:
                    # if no specifier is given any stacked dark will do
                    mask = (np.array(self.reduced_files['TYPE']=='DARK') &
                            combined)
                # ensure that the specifier has the correct type
                elif isinstance(specifier, (int, float)):
                    mask = (np.array(self.reduced_files['TYPE']=='DARK') &
                            np.array(self.reduced_files['EXPOSURE']==specifier) &
                            combined)
                else:
                    raise ValueError("The specifier has to be an int or float, "+
                                     f"you provided {type(specifier)}")
            case 'flat':
                if specifier is None:
                    # if no specifier is given any flat will do
                    mask = (np.array(self.reduced_files['TYPE']=='FLAT') &
                            combined)
                # ensure that the specifier has the correct type
                elif isinstance(specifier, str):
                    mask = (np.array(self.reduced_files['TYPE']=='FLAT') &
                            np.array(self.reduced_files['FILTER']==specifier) &
                            combined)
                else:
                    raise ValueError(f"The specifier has to be a string, you provided {type(specifier)}.")
            case 'light':
                if specifier is None:
                    # if no specifier is given any flat will do
                    mask = (np.array(self.reduced_files['TYPE']=='FLAT') &
                            combined)
                # ensure that the specifier has the correct type
                elif isinstance(specifier, int):
                    mask = (np.array(self.reduced_files['TYPE']=='FLAT') &
                            np.array(self.reduced_files['FILTER']==specifier) &
                            combined)
                else:
                    raise ValueError(f"The specifier has to be a string, you provided {type(specifier)}.")
            case _:
                raise ValueError(f"You provided an invalid imagetype '{imagetype}', use 'bias', 'dark' or 'flat'")

        if np.sum(mask) == 0:
            return None
        elif np.sum(mask) == 1:
            idx = np.nonzero(mask)
            data, hdr = self.hdu_from_file(str(self.reduced_files['FILENAME'][idx][0]))
            if header:
                return data, header
            return data
        else:
            # too many matches
            files = [str(f) for f in self.reduced_files['FILENAME'][mask]]
            raise RuntimeError(f"Found {len(files)} frames matching ({files})." +
                               "Include a specifier or ensure only one master frame for the given specifier exists.")

    @staticmethod
    def file_data(filelist:Iterable[str], data:bool=True, header:bool=False,
                  fname:bool=False, return_kwds:list[str]|None=None, **filter_kwds)->Generator:
        """generator to get the data and/or header of the files in the provided list.

        :param filelist: list of files to iterate over
        :type filelist: list
        :param data: whether or not the data of the file should be returned, defaults to True
        :type data: bool, optional
        :param header: whether or not the header should be returned, defaults to False
        :type header: bool, optional
        :param fname: whether or not the filename should be returned, defaults to False
        :type fname: bool, optional
        :param return_kwds: additional keywords that should be returned alongside the data/header,
                            defaults to None
        :type return_kwds: list[str] | None
        :param filter_kwds: additional keywords the returned data should be filtered for from the header
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename, return_kwd 1,...)
        :rtype: Tuple
        """
        # TODO: include the possibility to only yield keywords?
        if not data and not header:
            raise ValueError("At least one of 'data' and 'header' must be True." \
                             f"You provided: data={data} and header={header}.")

        return_kwd = [] if return_kwds is None else return_kwds
        kwds = []
        for file in filelist:
            out = []
            if filter_kwds or return_kwd:
                hdr = fits.getheader(file)
                if any(hdr.get(kwd) != val for kwd, val in filter_kwds.items()):
                    # skip if the keywords do not match
                    continue
                for kwd in return_kwd:
                    kwds.append(hdr.get(kwd))

            if data:
                out.append(fits.getdata(file) if data else None)
            if header:
                out.append(fits.getheader(file) if header else None)
            if fname:
                out.append(file if fname else None)
            out = out + kwds    # add the desired keywords to the end
            yield tuple(out) if len(out)>1 else out[0]

    def bias(self, data:bool=True, header:bool=False, fname:bool=False, **filter_kwds)->Generator:
        """Generator to get the data and/or header of the files in the raw bias frames.

        :param data: whether or not the data of the file should be returned, defaults to True
        :type data: bool, optional
        :param header: whether or not the header should be returned, defaults to False
        :type header: bool, optional
        :param fname: whether or not the filename should be returned, defaults to False
        :type fname: bool, optional
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename)
        :rtype: Tuple
        """
        bias = self.raw_files[self.raw_files['TYPE']=='BIAS']
        bias_files = np.array(bias['FILENAME'])
        return self.file_data(bias_files, data, header, fname, **filter_kwds)

    def darks(self, exposure:int, data:bool=True, header:bool=False, fname:bool=False, **filter_kwds)->Generator:
        """Generator to get the data and/or header of the files of the raw dark frames
          for a specific exposure.

        :param exposure: the exposure time of the dark frame
        :type exposure: int
        :param data: whether or not the data of the file should be returned, defaults to True
        :type data: bool, optional
        :param header: whether or not the header should be returned, defaults to False
        :type header: bool, optional
        :param fname: whether or not the filename should be returned, defaults to False
        :type fname: bool, optional
        :raises ValueError: This error is raised if there is no dark frame with the given exposure
                            registered. Try 'update_raw()' if you think there should be one
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename)
        :rtype: Tuple
        """
        if exposure not in self.dark_exposures:
            raise ValueError(f"There is no dark frame for this exposure: {exposure}")
        darks = self.raw_files[self.raw_files['TYPE']=='DARK']
        dark_files = np.array(darks['FILENAME'][darks['EXPOSURE']==exposure])
        return self.file_data(dark_files, data, header, fname, **filter_kwds)

    def flats(self, used_filter:str|None, data:bool=True, header:bool=False, fname:bool=False,
              return_kwds:list[str]|None=None, **filter_kwds)->Generator:
        """Generator to get the data and/or header of the files of the raw flat frames
          for a specific filter.

        :param used_filter: the exposure time of the flat frame
        :type used_filter: str | None
        :param data: whether or not the data of the file should be returned, defaults to True
        :type data: bool, optional
        :param header: whether or not the header should be returned, defaults to False
        :type header: bool, optional
        :param fname: whether or not the filename should be returned, defaults to False
        :type fname: bool, optional
        :raises ValueError: This error is raised if there is no flat frame with the given filter
                            registered. Try 'update_raw()' if you think there should be one
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename)
        :rtype: Tuple
        """
        if used_filter not in self.used_filters:
            raise ValueError(f"There is no flat frame for this filter: {used_filter}")
        flat_files = self.get_files(type='FLAT', filter=used_filter)
        return self.file_data(flat_files, data, header, fname, return_kwds, **filter_kwds)

    def lights(self, target:str, data:bool=True, header:bool=False, fname:bool=False,
               return_kwds:list[str]|None=None, reduced:bool=False, **filter_kwds)->Generator:
        """Generator to get the data and/or header of the files of the raw light frames
          for a specific target.

        :param target: the target of the light frame
        :type target: str
        :param data: whether or not the data of the file should be returned, defaults to True
        :type data: bool, optional
        :param header: whether or not the header should be returned, defaults to False
        :type header: bool, optional
        :param fname: whether or not the filename should be returned, defaults to False
        :type fname: bool, optional
        :raises ValueError: This error is raised if there is no light frame with the given target
                            registered. Try 'update_raw()' if you think there should be one
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename)
        :rtype: Tuple
        """
        table = self.reduced_files if reduced else self.raw_files
        if target not in self.targets:
            raise ValueError(f"There is no light frame for the given target: {target}")
        light_files = self.get_files(reduced=reduced, type='LIGHT', object=target)
        return self.file_data(light_files, data, header, fname, return_kwds, **filter_kwds)
