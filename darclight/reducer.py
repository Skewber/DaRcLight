"""This module provides tools for data reduction"""
import numpy as np
from astropy.stats import sigma_clip
from darclight.io import Observation

class Reducer():
    """Class for general reduction and correction of the provided data
    """
    def __init__(self, data:Observation):
        self.data = data
        self.method = 'sigmaclip'

    @staticmethod
    def combine(data:list[np.ndarray], method:str='sigmaclip', sigma:int=5)->np.ndarray:
        """combines the given data to one 2D-array

        :param data: list of the image arrays
        :type data: list[np.ndarray]
        :param method: which method to use for the combination,
                        valid are 'mean', 'median' 'add' and 'sigmaclip',
                        defaults to 'sigmaclip'
        :type method: str, optional
        :param sigma: standard deviation used for sigma clipping, defaults to 5
        :type sigma: int, optional
        :return: returns the combined data array
        :rtype: np.ndarray
        """
        stack = np.stack(data)
        match method:
            case 'sigmaclip':
                clipped = sigma_clip(stack, sigma=sigma, axis=0)
                return np.mean(clipped, axis=0)

            case 'mean':
                return np.mean(stack, axis=0)

            case 'median':
                return np.median(stack, axis=0)

            case 'add':
                return np.sum(stack, axis=0)

    @staticmethod
    def generate_filename(frame:str, obj:str=None,
                          filt:str=None, exposure:str=None):
        """generates a suitable filename for a masterframe based on the given values.
        The name follows the following pattern: 'master_<frame>_<obj>_<exposure>s.fits'

        :param frame: name of the frame, for example 'bias', can be anything
        :type frame: str
        :param obj: _description_, defaults to None
        :type obj: str, optional
        :param filt: _description_, defaults to None
        :type filt: str, optional
        :param exposure: _description_, defaults to None
        :type exposure: str, optional
        :return: _description_
        :rtype: _type_
        """
        # generate the right filename depending on what arguments are given
        file_name: str = f"master_{frame}"
        if obj is not None:
            file_name = f"{file_name}_{obj}"
        if filt is not None:
            file_name = f"{file_name}_filter_{filt}"
        if exposure is not None:
            file_name = f"{file_name}_{exposure}s"
        file_name = f"{file_name}.fits"
        return file_name

    def create_master_bias(self, force_new:bool=False, method:str=None)->np.ndarray:
        """Creates a master bias by combining the registered files if it
        does not exist yet and saves it.

        :param force_new: whether or not a new master file should be forced, defaults to False
        :type force_new: bool, optional
        :param method: method used for combination. Check Reducer.combine for more detail, defaults to None
        :type method: str, optional
        :return: the combined data
        :rtype: np.ndarray
        """
        # check if master bias exists
        if not self.data.master_exists('bias') or force_new:
            # collect data
            biases = [d for d in self.data.bias.data()]
            header = next(self.data.bias.headers())
            # update header
            header['combined'] = True
            header['ncombine'] = len(biases)
            # stack the frames
            method = self.method if method is None else method
            master = self.combine(biases, method=method)
            # save master
            file_name = self.generate_filename('bias')
            print(file_name)
            Observation.safe_file(self.data.reduced_path/file_name, master, header)
            # update the object
            self.data.masters['bias'] = self.data.reduced_path/file_name

        return self.data.hdu_from_file(self.data.masters['bias'])[0]
