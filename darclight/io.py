"""Module to manage general information like path to data"""
import numpy as np
from astropy.io import fits
from typing import Generator
import os
from pathlib import Path

class FileList():
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

    def append(self, file:str):
        self.files.append(file)

    def __repr__(self):
        return str(self.files)
    
    def __iter__(self):
        return iter(self.files)

    def data(self, return_fname:bool=False) -> Generator[np.ndarray, None, None]:
        for f in self.files:
            yield (fits.getdata(f), f) if return_fname else fits.getdata(f)
    
    def headers(self, return_fname:bool=False) -> Generator[np.ndarray, None, None]:
        for f in self.files:
            yield (fits.getheader(f), f) if return_fname else fits.getheader(f)

class Observation():
    def __init__(self, raw_path:str=None, reduced_path:str=None) -> None:
        self.raw_path = Path(raw_path)
        self.reduced_path = Path(reduced_path)
        self.reduced_path.mkdir(exist_ok=True, mode=777)
        self._files_from_header(raw_path)
        self.masters = {'bias':None,
                        'dark':None,
                        'flat':None,
                        'light':None}
    
    def _files_from_header(self, path:Path):
        path = Path(path)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # TODO: remove non .fits files
        common_kwds = {'bias':['bias', 'zero'],
                       'dark':['dark'],
                       'flat':['flat'],
                       'light':['light', 'science', 'object']}
        self.bias = FileList()
        self.darks = {}     # key = exposure
        self.flats = {}     # key = filter
        self.lights = {}    # key = object
        for file in files:
            header_kwd = fits.getval(path/file, 'imagetyp').lower()
            if any([kwd in header_kwd for kwd in common_kwds['bias']]):
                self.bias.append(path/file)

            elif any([kwd in header_kwd for kwd in common_kwds['dark']]):
                exp = int(fits.getval(path/file, 'exposure'))
                if exp in self.darks:
                    self.darks[exp].append(path/file)
                else:
                    self.darks[exp] = FileList(path/file)

            elif any([kwd in header_kwd for kwd in common_kwds['flat']]):
                filter = fits.getval(path/file, 'filter')
                if filter in self.flats:
                    self.flats[filter].append(path/file)
                else:
                    self.flats[filter] = FileList(path/file)
        
            elif any([kwd in header_kwd for kwd in common_kwds['light']]):
                target = fits.getval(path/file, 'object')
                if target in self.lights:
                    self.lights[target].append(path/file)
                else:
                    self.lights[target] = FileList(path/file)

    @staticmethod
    def safe_file(filename, data, header=None):
        hdu = fits.PrimaryHDU(data, header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=True)

    @property
    def used_filters(self):
        return list(self.flats.keys())
    
    @property
    def dark_exposures(self):
        return list(self.darks.keys())
    
    def master_exists(self, frame:str, filter:str=None) -> bool:
        if self.masters[frame] is None:
            files = os.listdir(self.reduced_path)
            # TODO: add code for master frame detection

        else:
            if frame == 'flat':
                if filter not in self.used_filters:
                    raise ValueError(f'Invalid filter provided. ' \
                                     'Usable filters are: {self.used_filters}. ' \
                                     'You provided: {filter}')
                if self.masters['flat'][filter] is None:
                    return False
                else:
                    return True
                
            elif frame in ('bias', 'dark', 'light'):
                if self.masters['dark'] is None:
                    return False
                else:
                    return True
            
            else:
                raise ValueError('Invalid imagetype provided. ' \
                                 'valid options are "bias", "dark", "flat" and "light". ' \
                                 f'You provided: {frame}')