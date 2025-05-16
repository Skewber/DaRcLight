"""Module to manage general information like path to data"""
import os
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
import astropy
from astropy.io import fits

class FileList():
    """Class for handling files that are grouped together in one list
    """
    def __init__(self, files:list[str]|str=None):
        if files is None:
            self.files = []
        elif isinstance(files, str):
            self.files = [files]
        elif isinstance(files, list):
            self.files = files
        elif isinstance(files, Path):
            self.files = [files]
        else:
            raise ValueError('Invalid argument provided for FileList. ' \
                            'Use a string or list of strings.')

    def append(self, file:str)->None:
        """Adds the given file to the filelist

        :param file: path to the file that should be added
        :type file: str
        """
        self.files.append(file)

    def __repr__(self):
        return str(self.files)

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

    def data(self, return_fname:bool=False)->Iterator[np.ndarray|Tuple[np.ndarray,str]]:
        """Iterator that yields one data array at the time for each file in the list

        :param return_fname: whether or not the filename should be returned along with the data, defaults to False
        :type return_fname: bool, optional
        :yield: Data (and filename) of the current file
        :rtype: Iterator[np.ndarray|Tuple[np.nparray,str]]
        """
        for f in self.files:
            yield (fits.getdata(f), f) if return_fname else fits.getdata(f)

    def headers(self, return_fname:bool=False)->Iterator[astropy.io.fits.header.Header|
                                                         Tuple[astropy.io.fits.header.Header,str]]:
        """Iterator that yields one header array at the time for each file in the list

        :param return_fname: whether or not the filename should be returned along with the header, defaults to False
        :type return_fname: bool, optional
        :yield: Header (and filename) of the current file
        :rtype: Iterator[np.ndarray|Tuple[np.nparray,str]]
        """
        for f in self.files:
            yield (fits.getheader(f), f) if return_fname else fits.getheader(f)

class Observation():
    """Class that organizes the individual imagetypes and data paths
    """
    def __init__(self, raw_path:str=None, reduced_path:str=None) -> None:
        self.raw_path = Path(raw_path)
        self.reduced_path = Path(reduced_path)
        self.reduced_path.mkdir(exist_ok=True, mode=777)
        self.bias, self.darks, self.flats, self.lights = self.files_from_header(raw_path)
        mbias, mdarks, mflats, mlights = self.files_from_header(reduced_path)
        self.masters = {'bias':mbias[0] if len(mbias)>0 else None,
                        'dark':mdarks if len(mdarks)>0 else None,
                        'flat':mflats if len(mflats)>0 else None,
                        'light':mlights if len(mlights)>0 else None}

    @staticmethod
    def files_from_header(path:Path
    )->Tuple[FileList,dict[int,FileList],dict[str,FileList],dict[str,FileList]]:
        """detects the individual images and sorts them according to their header

        :param path: the path where the files are located
        :type path: Path
        :return: a FileList object for every found type.
                Dark frames are sorted in a dictionary with their exposure as key,
                Flats are sorted into a dictionary according to the used filter and 
                the light frames according to the object.
        :rtype: Tuple[FileList,dict[int,FileList],dict[str,FileList],dict[str,FileList]]
        """
        path = Path(path)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # TODO: remove non .fits files
        common_kwds = {'bias':['bias', 'zero'],
                       'dark':['dark'],
                       'flat':['flat'],
                       'light':['light', 'science', 'object']}
        bias = FileList()
        darks = {}     # key = exposure
        flats = {}     # key = filter
        lights = {}    # key = object
        for file in files:
            header_kwd = fits.getval(path/file, 'imagetyp').lower()
            if any([kwd in header_kwd for kwd in common_kwds['bias']]):
                bias.append(path/file)

            elif any([kwd in header_kwd for kwd in common_kwds['dark']]):
                exp = int(fits.getval(path/file, 'exposure'))
                if exp in darks:
                    darks[exp].append(path/file)
                else:
                    darks[exp] = FileList(path/file)

            elif any([kwd in header_kwd for kwd in common_kwds['flat']]):
                used_filter = fits.getval(path/file, 'filter')
                if used_filter in flats:
                    flats[used_filter].append(path/file)
                else:
                    flats[used_filter] = FileList(path/file)

            elif any([kwd in header_kwd for kwd in common_kwds['light']]):
                target = fits.getval(path/file, 'object')
                if target in lights:
                    lights[target].append(path/file)
                else:
                    lights[target] = FileList(path/file)

        return bias, darks, flats, lights

    @staticmethod
    def safe_file(filename:str, data:np.ndarray, header:astropy.io.fits.header.Header=None)->None:
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

    @property
    def used_filters(self)->list[str]:
        """Return filters where a flat frame is available.

        :return: list of filters
        :rtype: list[str]
        """
        return list(self.flats.keys())

    @property
    def dark_exposures(self)->list[int]:
        """Exposure values where a darkframe is directly available.

        :return: list of exposure values
        :rtype: list[int]
        """
        return list(self.darks.keys())

    @staticmethod
    def hdu_from_file(file:str)->Tuple[np.ndarray,astropy.io.fits.header.Header]:
        """gives access to the data and header of a given file

        :param file: name of the file
        :type file: str
        :return: the data and header from that file
        :rtype: Tuple[np.ndarray,astropy.io.fits.header.Header]
        """
        with fits.open(file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            return data, header

    def master_exists(self, frame:str, used_filter:str=None) -> bool:
        """checks if a masterframe is registered or if it exists in the reduced data directory.

        :param frame: type of frame for which the check sould be performed.
                    Available options are 'bias', 'dark', 'flat' and 'light'
        :type frame: str
        :param used_filter: for which filter the check should be performed,
                            has to be the same string as in the header, defaults to None
        :type used_filter: str, optional
        :return: True if a master exists, False otherwise
        :rtype: bool
        """
        if self.masters[frame] is None:
            bias, darks, flats, lights = self.files_from_header(self.reduced_path)
            frames = {'bias':bias if len(bias)>0 else None,
                      'dark':darks if len(darks)>0 else None,
                      'flat':flats if len(flats)>0 else None,
                      'light':lights if len(lights)>0 else None}

            # for flats we need to check every filter individualy
            if frames[frame] is not None and frame != 'flat':
                return True
            elif frames[frame] is not None and frame == 'flat':
                return used_filter in frames[frame]
            # masters is None and no frames found
            return False
        else:
            if frame == 'flat':
                return used_filter in self.masters[frame]
            return True
