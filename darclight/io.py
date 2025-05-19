"""Module to manage general information like path to data"""
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from astropy.io import fits

class DataCollection():
    """Class that organizes all files in a given directory.
    """
    def __init__(self, raw_path:str|None=None, reduced_path:str|None=None):
        # ensure that both paths are an Path object
        if raw_path is None:
            self.raw_path = Path('.')
        else:
            self.raw_path = Path(raw_path)

        if reduced_path is None:
            self.reduced_path = Path('./reduced')
        else:
            self.reduced_path = Path(reduced_path)
        # create the path if it does not exist
        self.reduced_path.mkdir(exist_ok=True)

        self.update_raw()
        self.update_reduced()

    @staticmethod
    def sort_files(path:str|Path)->Tuple[list,dict,dict,dict]:
        """Reads all files from a directory and sorts them in seperate lists based on their header.

        :param path: Path to the data
        :type path: str | Path
        :return: A list of filenames for bias, dark, flat and light frames,
                the dark frames are a dict with the exposure as key to the filelists,
                the flat frames are a dict with the used filter as key to the filelists,
                the light frames are a dict with the target as key to the filelists.
        :rtype: Tuple[list,dict,dict,dict]
        """
        if isinstance(path, str):
            path = Path(path)
        files = [f for f in path.iterdir() if f.is_file()]

        common_kwds = {'bias':['bias', 'zero'],
                       'dark':['dark'],
                       'flat':['flat'],
                       'light':['light', 'science', 'object']}

        # use defaultdicts to reduce the if/elif statements to check if an enty exists
        bias_files = []
        dark_files = defaultdict(list)
        flat_files = defaultdict(list)
        light_files = defaultdict(list)

        for file in files:
            header_kwd = fits.getval(file, 'imagetyp').lower()

            if any([kwd in header_kwd for kwd in common_kwds['bias']]):
                bias_files.append(file.name)

            elif any(kwd in header_kwd for kwd in common_kwds['dark']):
                exp = int(fits.getval(file, 'exposure'))
                dark_files[exp].append(file.name)

            elif any(kwd in header_kwd for kwd in common_kwds['flat']):
                try:
                    # this value migth not exist always
                    used_filter = fits.getval(file, 'filter')
                except KeyError:
                    used_filter = None
                flat_files[used_filter].append(file.name)

            elif any(kwd in header_kwd for kwd in common_kwds['light']):
                target = fits.getval(file, 'object')
                light_files[target].append(file.name)

        # convert to normal dicts since we don't need a default value outside this function
        return bias_files, dict(dark_files), dict(flat_files), dict(light_files)

    def update_raw(self):
        """Rereads the raw file directory and recreates the list for each imagetyp.
        """
        files = self.sort_files(self.raw_path)
        self.bias_files = files[0]
        self.dark_files = files[1]
        self.flat_files = files[2]
        self.light_files = files[3]

    def update_reduced(self):
        """Rereads the reduced file directory and recreates the list for each imagetyp.
        """
        files = self.sort_files(self.reduced_path)
        self.master_bias_file = files[0][0] if len(files[0])>0 else None
        self.master_dark_files = files[1] if len(files[0])>0 else None
        self.master_flat_files = files[2] if len(files[0])>0 else None
        self.master_light_files = files[3] if len(files[0])>0 else None

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
    def safe_file(filename:str|Path, data:np.ndarray, header:fits.header.Header|None=None)->None:
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
        """List of used filters where a flat frame is available.

        :return: list of filters
        :rtype: list[str]
        """
        return list(self.flat_files.keys())

    @property
    def dark_exposures(self)->list[int]:
        """List of exposure times where a dark frame is directly available.

        :return: list of exposure times
        :rtype: list[int]
        """
        return list(self.dark_files.keys())

    @property
    def targets(self)->list[str]:
        """List of targets captured.

        :return: list of targets
        :rtype: list[str]
        """
        return list(self.light_files.keys())

    @staticmethod
    def generator(filelist:list, data:bool=True, header:bool=False, fname:bool=False)->Generator:
        """generator to get the data and/or header of the files in the provided list.

        :param filelist: list of files to iterate over
        :type filelist: list
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
        if not data and not header:
            raise ValueError("At least one of 'data' and 'header' must be True." \
                             f"You provided: data={data} and header={header}.")

        for file in filelist:
            out = []
            if data:
                out.append(fits.getdata(file) if data else None)
            if header:
                out.append(fits.getheader(file) if header else None)
            if fname:
                out.append(file.name if fname else None)
            yield tuple(out)

    def bias(self, data:bool=True, header:bool=False, fname:bool=False)->Generator:
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
        bias_files = [self.raw_path/f for f in self.bias_files]
        return self.generator(bias_files, data, header, fname)
