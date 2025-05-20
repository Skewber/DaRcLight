"""This module provides tools for data reduction"""
from tkinter import W
import numpy as np
from astropy.stats import sigma_clip
from darclight.io import DataCollection

class Reducer():
    """Class for general reduction and correction of the provided data
    """
    def __init__(self, data:DataCollection):
        self.data = data
        self.master_bias = None
        self.master_darks:dict[int,np.ndarray|None] = {exp:None for exp in self.data.dark_exposures}
        self.master_flats:dict[str,np.ndarray|None] = {filt:None for filt in self.data.used_filters}
        self.master_lights:dict[str,np.ndarray|None] = {tar:None for tar in self.data.targets}

    @staticmethod
    def combine(data:list[np.ndarray], method:str='mean', sigmaclip:bool=True, sigma:int=5)->np.ndarray:
        """combines the given data to one 2D-array

        :param data: list of the image arrays
        :type data: list[np.ndarray]
        :param method: which method to use for the combination,
                        valid are 'mean', 'median' and 'add',
                        defaults to 'mean'
        :type method: str, optional
        :param sigmaclip: whether or not the data should be clipped or not, defaults to True
        :type sigmaclip: bool, optional
        :param sigma: standard deviation used for sigma clipping, defaults to 5
        :type sigma: int, optional
        :return: returns the combined data array
        :rtype: np.ndarray
        """
        if len(data) == 1:
            print("WARNING: only one file in the list!")
            return data[0]

        stack = np.stack(data)
        print(stack.shape)
        if sigmaclip:
            stack = sigma_clip(stack, sigma=sigma, axis=0)
        match method:
            case 'mean':
                return np.array(np.mean(stack, axis=0)) # type: ignore

            case 'median':
                return np.array(np.median(stack, axis=0)) # type: ignore

            case 'add':
                return np.array(np.sum(stack, axis=0)) # type: ignore

            case _:
                raise ValueError("You provided an invalid argument for the combination method.")

    @staticmethod
    def generate_filename(frame:str, obj:str|None=None,
                          filt:str|None=None, exposure:str|None=None)->str:
        """generates a suitable filename for a masterframe based on the given values.
        The name follows the following pattern: 'master_<frame>_<obj>_<exposure>s.fits'

        :param frame: name of the frame, for example 'bias', can be anything
        :type frame: str
        :param obj: name of the object in the frame, defaults to None
        :type obj: str, optional
        :param filt: the used filter in the frame, defaults to None
        :type filt: str, optional
        :param exposure: duration of the exposure, defaults to None
        :type exposure: str, optional
        :return: the name of the file in the format 'master_<frame>_<obj>_<exposure>s.fits'
        :rtype: str
        """
        # generate the right filename depending on what arguments are given
        file_name: str = f"master_{frame}"
        if obj is not None:
            file_name = f"{file_name}_{obj}"
        if filt is not None:
            file_name = f"{file_name}_filter_{filt}"
        if exposure is not None:
            file_name = f"{file_name}_{exposure}s"
        file_name = f"{file_name}.fits".replace(" ", "_")
        return file_name

    def create_master_bias(self, force_new:bool=False, method:str='mean', **kwargs)->np.ndarray:
        """Creates a master bias by combining the registered files if it
        does not exist yet and saves it.

        :param force_new: whether or not a new master file should be forced, defaults to False
        :type force_new: bool, optional
        :param method: method used for combination. Check Reducer.combine for more detail,
                        defaults to 'mean'
        :type method: str, optional
        :param kwargs: additional keyword arguments passed to 'combine'
        :return: the combined data
        :rtype: np.ndarray
        """
        # check if master bias exists
        if self.master_bias is not None and not force_new:
        # there is no master created yet and don't force a new one
            return self.master_bias


        if force_new or self.data.master_bias_file is None:
            # collect data
            biases = [b for b in self.data.bias()]
            _, header = self.data.hdu_from_file(self.data.raw_path/self.data.bias_files[0])
            # update header
            header['COMBINED'] = True
            header['NCOMBINE'] = len(biases)
            # stack the frames and save
            master = self.combine(biases, method=method, **kwargs)
            file_name = self.generate_filename('bias')
            self.data.safe_file(self.data.reduced_path/file_name, master, header)
            # update the masters
            self.data.master_bias_file = file_name
            self.master_bias = master
        else:
            data, _ = self.data.hdu_from_file(self.data.reduced_path/self.data.master_bias_file)
            self.master_bias = data

        return self.master_bias

    def create_master_dark(self, exposure:int=-1, force_new:bool=False,
                           method:str='mean', **kwargs)->np.ndarray|None:
        # create a master frame for every exposure
        if exposure == -1:
            for exp in self.data.dark_exposures:
                self.create_master_dark(exp, force_new, method, **kwargs)
            return None

        # check if master dark exists
        if self.master_darks[exposure] is not None and not force_new:
            return self.master_darks[exposure]

        if force_new or self.data.master_dark_files[exposure] is None:
            # collect data
            darks = [d for d in self.data.darks(exposure)]
            _, header = self.data.hdu_from_file(self.data.raw_path/self.data.dark_files[exposure][0])
            # update the header
            header['COMBINED'] = True
            header['NCOMBINE'] = len(darks)
            # stack the frames and save
            master = self.combine(darks, method=method, **kwargs)
            file_name = self.generate_filename('dark', exposure=str(exposure))
            self.data.safe_file(self.data.reduced_path/file_name, master, header)
            # update the masters
            self.data.master_dark_files[exposure] = file_name # type: ignore
            self.master_darks[exposure] = master
        else:
            path = self.data.reduced_path / self.data.master_dark_files[exposure] # type: ignore
            data, _ = self.data.hdu_from_file(path)
            self.master_darks[exposure] = data

        return self.master_darks[exposure]
