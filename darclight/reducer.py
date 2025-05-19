"""This module provides tools for data reduction"""
import numpy as np
from astropy.stats import sigma_clip
from darclight.io import DataCollection

class Reducer():
    """Class for general reduction and correction of the provided data
    """
    def __init__(self, data:DataCollection):
        self.data = data
        self.master_bias = None
        self.master_darks = None
        self.master_flats = None
        self.master_lights = None

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
        stack = np.stack(data)
        if sigmaclip:
            stack = sigma_clip(stack, sigma=sigma, axis=0)
        match method:
            case 'mean':
                return np.mean(stack, axis=0) # type: ignore

            case 'median':
                return np.median(stack, axis=0) # type: ignore

            case 'add':
                return np.sum(stack, axis=0) # type: ignore

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

    def create_master_bias(self, force_new:bool=False, method:str='mean')->np.ndarray:
        """Creates a master bias by combining the registered files if it
        does not exist yet and saves it.

        :param force_new: whether or not a new master file should be forced, defaults to False
        :type force_new: bool, optional
        :param method: method used for combination. Check Reducer.combine for more detail, defaults to 'mean'
        :type method: str, optional
        :return: the combined data
        :rtype: np.ndarray
        """
        # check if master bias exists
        if self.master_bias is not None and not force_new:
            return self.master_bias

        if force_new or self.data.master_bias_file is None:
            # collect data
            biases = [d for d in self.data.bias()]
            _, header = self.data.hdu_from_file(self.data.raw_path/self.data.bias_files[0])
            # update header
            header['combined'] = True
            header['ncombine'] = len(biases)
            # stack the frames
            master = self.combine(biases, method=method)
            # save master
            file_name = self.generate_filename('bias')
            self.data.safe_file(self.data.reduced_path/file_name, master, header)
            # update the masters
            self.data.master_bias_file = file_name
            self.master_bias = master
        else:
            self.master_bias = self.data.hdu_from_file(self.data.reduced_path/self.data.master_bias_file)[0]

        return self.master_bias
