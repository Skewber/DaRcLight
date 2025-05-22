"""Module to manage general information like path to data"""
from fnmatch import fnmatch
from functools import cached_property
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from astropy.io import fits

class DataCollection():
    """Class that organizes all files in a given directory.
    """
    def __init__(self, raw_path:str|None=None, reduced_path:str|None=None, ignore:list|None=None):
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

        self.ignore = [] if ignore is None else ignore
        self.update_raw()
        self.update_reduced()

    @staticmethod
    def sort_files(path:str|Path, ignore:list|None=None)->Tuple[list,dict,dict,dict]:
        """Reads all files from a directory and sorts them in seperate lists based on their header.

        :param path: Path to the data
        :type path: str | Path
        :return: A list of filenames for bias, dark, flat and light frames,
                the dark frames are a dict with the exposure as key to the filelists,
                the flat frames are a dict with the used filter as key to the filelists,
                the light frames are a dict with the target as key to the filelists.
        :rtype: Tuple[list,dict,dict,dict]
        """
        ignore = [] if ignore is None else ignore
        if isinstance(path, str):
            path = Path(path)
        files = [f for f in path.iterdir()
                 if f.is_file() and not any(fnmatch(f.name, pat) for pat in ignore)]

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
        bias, darks, flats, lights = self.sort_files(self.raw_path, self.ignore)
        self.bias_files = bias
        self.dark_files = darks
        self.flat_files = flats
        self.light_files = lights

    def update_reduced(self):
        """Rereads the reduced file directory and recreates the list for each imagetyp.
        """
        bias, darks, flats, lights = self.sort_files(self.reduced_path, self.ignore)
        self.master_bias_file = bias[0] if len(bias)>0 else None
        def validate(frames:dict, imgtype:str):
            validated = {}
            for key, file in frames.items():
                if len(file) == 0:
                    validated[key] = None
                elif len(file) == 1:
                    validated[key] = file[0]
                else:
                    raise ValueError(f"More than one master {imgtype} is detected for {key}.")
            return validated
        self.master_dark_files = {exp:None for exp in self.dark_exposures} | validate(darks, 'dark')
        self.master_flat_files = {filt:None for filt in self.used_filters} | validate(flats, 'flat')
        # skip validation for lights since they are usually stacked seperately
        # => normally more than one
        self.master_light_files = {tar:None for tar in self.targets} | lights

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

    @cached_property
    def flat_exposures(self)->dict[str|None,set]:
        """exposure times for each filter

        :return: dictionary of the form {filter:exposure, ...}
        :rtype: dict[str|None,set]
        """
        result = defaultdict(set)

        for filt, files in self.flat_files.items():
            for fname in files:
                exp = fits.getval(self.raw_path/fname, "EXPOSURE")
                result[filt].add(int(exp))
        return dict(result)

    @property
    def targets(self)->list[str]:
        """List of targets captured.

        :return: list of targets
        :rtype: list[str]
        """
        return list(self.light_files.keys())

    @cached_property
    def light_meta(self)->dict[str,set[Tuple[str, int]]]:
        """Metadata for the light frames

        :return: dictionary of the form {target:[(filter,exposure),...],...},
                for every target there is a set of tuples that each contain the filter and
                the corresponding exposure time.
        :rtype: dict[str,set[Tuple[str, int]]]
        """
        result = defaultdict(set)

        for target, files in self.light_files.items():
            for fname in files:
                header = fits.getheader(self.raw_path/fname)
                filt = header.get("FILTER")
                exp = header.get("EXPOSURE")
                if exp is not None:
                    result[target].add((filt, int(exp)))
        return dict(result)

    @staticmethod
    def generator(filelist:list, data:bool=True, header:bool=False,
                  fname:bool=False, return_kwds:list[str]|None=None, **keywords)->Generator:
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
        :param keywords: additional keywords that should be extracted from the header
        :raises ValueError: if both (data and header) are set to False.
                            If you want only the filenames address the attribute directly.
        :yield: tuple of the desired outputs in the order (data, header, filename, return_kwd 1,...)
        :rtype: Tuple
        """
        if not data and not header:
            raise ValueError("At least one of 'data' and 'header' must be True." \
                             f"You provided: data={data} and header={header}.")

        return_kwd = [] if return_kwds is None else return_kwds
        kwds = []
        for file in filelist:
            out = []
            if keywords or return_kwd:
                hdr = fits.getheader(file)
                if any(hdr.get(kwd) != val for kwd, val in keywords.items()):
                    # skip if the keywords do not match
                    continue
                for kwd in return_kwd:
                    kwds.append(hdr.get(kwd))

            if data:
                out.append(fits.getdata(file) if data else None)
            if header:
                out.append(fits.getheader(file) if header else None)
            if fname:
                out.append(file.name if fname else None)
            out = out + kwds    # add the desired keywords to the end
            yield tuple(out) if len(out)>1 else out[0]

    def bias(self, data:bool=True, header:bool=False, fname:bool=False, **keywords)->Generator:
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
        return self.generator(bias_files, data, header, fname, **keywords)

    def darks(self, exposure:int, data:bool=True, header:bool=False, fname:bool=False, **keywords)->Generator:
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
        dark_files = [self.raw_path/f for f in self.dark_files[exposure]]
        return self.generator(dark_files, data, header, fname, **keywords)

    def flats(self, used_filter:str|None, data:bool=True, header:bool=False, fname:bool=False,
              return_kwds:list[str]|None=None, **keywords)->Generator:
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
        flat_files = [self.raw_path/f for f in self.flat_files[used_filter]]
        return self.generator(flat_files, data, header, fname, return_kwds, **keywords)

    def lights(self, target:str, data:bool=True, header:bool=False, fname:bool=False,
               return_kwds:list[str]|None=None, **keywords)->Generator:
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
        if target not in self.targets:
            raise ValueError(f"There is no light frame for the given target: {target}")
        light_files = [self.raw_path/l for l in self.light_files[target]]
        return self.generator(light_files, data, header, fname, return_kwds, **keywords)
