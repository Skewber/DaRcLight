"""Module to manage general information like path to data"""
import numpy as np
from astropy.io import fits
from typing import Generator
import os

class FileList():
    def __init__(self, files:list[str]|str=None):
        if files is None:
            self.files = []
        elif isinstance(files, str):
            self.files = [files]
        elif isinstance(files, list):
            self.files = files
        else:
            raise ValueError('Invalid argument provided for FileList. ' \
                            'Use a string or list of strings.')

    def append(self, file:str):
        self.files.append(file)

    def __repr__(self):
        return str(self.files)

    def data(self, return_fname:bool=False) -> Generator[np.ndarray, None, None]:
        for f in self.files:
            yield (fits.getdata(f), f) if return_fname else fits.getdata(f)
    
    def headers(self, return_fname:bool=False) -> Generator[np.ndarray, None, None]:
        for f in self.files:
            yield (fits.getheader(f), f) if return_fname else fits.getheader(f)

class Observation():
    def __init__(self, raw_path:str=None, reduced_path:str=None) -> None:
        self.raw_path = raw_path
        self.readuced_path = reduced_path
        self._files_from_header(raw_path)
    
    def _files_from_header(self, path:str):
        files = os.listdir(path)
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
            header_kwd = fits.getval(f'./{self.raw_path}/{file}', 'imagetyp').lower()
            if any([kwd in header_kwd for kwd in common_kwds['bias']]):
                self.bias.append(file)

            elif any([kwd in header_kwd for kwd in common_kwds['dark']]):
                exp = int(fits.getval(f'./{self.raw_path}/{file}', 'exposure'))
                if exp in self.darks:
                    self.darks[exp].append(file)
                else:
                    self.darks[exp] = FileList(file)

            elif any([kwd in header_kwd for kwd in common_kwds['flat']]):
                filter = fits.getval(f'./{self.raw_path}/{file}', 'filter')
                if filter in self.flats:
                    self.flats[filter].append(file)
                else:
                    self.flats[filter] = FileList(file)
        
            elif any([kwd in header_kwd for kwd in common_kwds['light']]):
                target = fits.getval(f'./{self.raw_path}/{file}', 'object')
                if target in self.lights:
                    self.lights[target].append(file)
                else:
                    self.lights[target] = FileList(file)

    @property
    def used_filters(self):
        return list(self.flats.keys())