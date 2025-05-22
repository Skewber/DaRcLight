"""This module provides tools for data reduction"""
from typing import Callable
import numpy as np
from astropy.stats import sigma_clip
from darclight.io import DataCollection

def inv_median(data:np.ndarray)->np.ndarray:
    """Function to normalize data by division of the median

    :param data: data that should be normalized
    :type data: np.ndarray
    :return: normalized data
    :rtype: np.ndarray
    """
    return data / np.median(data)

class Reducer():
    """Class for general reduction and correction of the provided data
    """
    def __init__(self, data:DataCollection):
        self.data = data
        self.master_bias = None
        self.master_darks:dict[int,np.ndarray|None] = {exp:None for exp in self.data.dark_exposures}
        self.master_flats:dict[str|None,np.ndarray|None] = {filt:None for filt in self.data.used_filters}
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
        """Creates master darks by combining the registered files if they
        do not exist yet and saves them.

        :param exposure: the exposure time for which the master dark should be created
                        a time of -1 means that for every exposure registered the function is
                        called recursevly, defaults to -1
        :type exposure: int, optional
        :param force_new: whether or not a new master file should be forced, defaults to False
        :type force_new: bool, optional
        :param method: method used for combination. Check Reducer.combine for more detail,
                        defaults to 'mean'
        :type method: str, optional
        :param kwargs: additional keyword arguments passed to 'combine'
        :return: the combined data or None if the exposure time is set to -1
        :rtype: np.ndarray | None
        """
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
            # correction
            if self.master_bias is None:
                self.create_master_bias(method=method, **kwargs)
            darks = [d-self.master_bias for d in darks]

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

    def create_master_flats(self, used_filter:str|None='all', norm:Callable|None=None,
                            force_new:bool=False, method='mean', **kwargs)->np.ndarray|None:
        """Creates master flats by combining the registered files if they
        do not exist yet and saves them.

        :param used_filter: the filter for which the master flat should be created
                        a value of 'all' means that for every filter registered, the function is
                        called recursevly, defaults to 'all'
        :type used_filter: str, optional
        :param norm: function used to normalize each frame,
                    should take a 2d array as input and return the normalized array,
                    defaults to None
        :type norm: Callable | None, optional
        :param force_new: whether or not a new master file should be forced, defaults to False
        :type force_new: bool, optional
        :param method: method used for combination. Check Reducer.combine for more detail,
                        defaults to 'mean'
        :type method: str, optional
        :param kwargs: additional keyword arguments passed to 'combine'
        :return: the combined data or None if the used_filter is set to 'all'
        :rtype: np.ndarray | None
        """
        # execute the function for every filter
        if used_filter == 'all':
            for filt in self.data.used_filters:
                self.create_master_flats(filt, norm, force_new, method, **kwargs)
            return None

        # check if a master flat is loaded
        if self.master_flats[used_filter] is not None and not force_new:
            return self.master_flats[used_filter]

        # check if a flat exists
        if not force_new and self.data.master_flat_files[used_filter] is not None: # type: ignore
            path = self.data.reduced_path / self.data.master_flat_files[used_filter] # type: ignore
            data, _ = self.data.hdu_from_file(path)
            self.master_flats[used_filter] = data
        else:
            # collect data
            flats = [f for f in self.data.flats(used_filter)]
            _, header = self.data.hdu_from_file(self.data.raw_path/self.data.flat_files[used_filter][0])
            # correction
            mbias = self.create_master_bias(method=method, **kwargs) if self.master_bias is None else self.master_bias
            # find best dark and scale
            target = int(header.get('EXPOSURE'))    # type: ignore
            exposures = self.data.dark_exposures
            idx = np.searchsorted(sorted(exposures), target, side='left')
            best_time = exposures[idx] if idx<len(exposures) else exposures[idx-1]
            mdark = self.master_darks[best_time]
            mdark = mdark if mdark is not None else self.create_master_dark(best_time)
            mdark = self.master_darks[best_time] * target / best_time   # type:ignore
            flats = [f-mbias-mdark for f in flats]
            # update header
            header['COMBINED'] = True
            header['NCOMBINE'] = len(flats)
            # stack the frames and save
            master = self.combine(flats, method=method, **kwargs)
            if norm is not None:
                master = norm(master)
            file_name = self.generate_filename('flat', filt=used_filter)
            self.data.safe_file(self.data.reduced_path/file_name, master, header)
            # update the masters
            self.data.master_flat_files[used_filter] = file_name # type: ignore
            self.master_flats[used_filter] = master

        return self.master_flats[used_filter]

    def reduce_lights(self, target:str='all', force_new:bool=False,
                      method='mean', **kwargs)->None:
        """creates reduced light frames and saves them individually for later stacking/analysis

        :param target: the object for which the frames should be calibrated,
                    a value of 'all' means that the functioin is called recursevly for ever registered target,
                    defaults to 'all'
        :type target: str, optional
        :param force_new: wherther or not the frames should be overwritten if they exist,
                        defaults to False
        :type force_new: bool, optional
        :param method: method used for stacking, only used if there are certain calibration
                    frames (bias, dark, flat) missing,
                    defaults to 'mean'
        :type method: str, optional
        :raises RuntimeError: raised if force_new=False and a file with the same name exists 
                            in the reduced data directory
        :rtype: None
        """
        if target == 'all':
            for tar in self.data.light_meta:
                self.reduce_lights(tar, force_new, method, **kwargs)
            return None

        self.data.update_reduced()

        if not force_new:
        # check if reduced lights exist
            for file in self.data.light_files[target]:
                if self.data.master_light_files[target] is None:
                    break
                if file in self.data.master_light_files[target]:
                    raise RuntimeError(f"The file '{file}' is already registered in the reducd data direcory."\
                                    "Use force_new=True if you want to verwrite")

        # if this point is reached not file conflicts with the reduced data
        # collect data
        meta = self.data.light_meta[target]
        print(meta)
        for filt, expo in meta:
            lights, hdrs, fnames = zip(*self.data.lights(target, header=True, fname=True, filter=filt, exposure=expo))
            lights = list(lights)
            _, header = self.data.hdu_from_file(self.data.raw_path/fnames[0])
            # correction
            mbias = self.create_master_bias(method=method, **kwargs) if self.master_bias is None else self.master_bias
            # find best dark and scale
            target_time = int(header.get('EXPOSURE'))    # type: ignore
            exposures = self.data.dark_exposures
            idx = np.searchsorted(sorted(exposures), target_time, side='left')
            best_time = exposures[idx] if idx<len(exposures) else exposures[idx-1]
            mdark = self.master_darks[best_time]
            mdark = mdark if mdark is not None else self.create_master_dark(best_time)
            mdark = self.master_darks[best_time] * target_time / best_time   # type:ignore

            used_filter = str(header.get('FILTER'))
            if self.master_flats[used_filter] is None:
                mflat = self.create_master_flats(method=method, **kwargs)
            else:
                mflat = self.master_flats[used_filter]

            lights = [(l-mbias-mdark)/mflat for l in lights]
            for data, hdr, fname in zip(lights, hdrs, fnames):
                self.data.safe_file(self.data.reduced_path/fname, data, hdr)
        self.data.update_reduced()
