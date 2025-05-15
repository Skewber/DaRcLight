import numpy as np
from astropy.stats import sigma_clip
from .io import Observation
from shutil import copy

class Reducer():
    def __init__(self, data:Observation):
        self.data = data

    @staticmethod
    def combine(data:list[np.ndarray], method='sigmaclip', sigma:int=5):
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
    def stack_images(to_combine:list[np.ndarray],
                     frame:str, header=None,
                     obj:str=None,
                     filt:str=None, exposure:str=None,
                     method='sigmaclip', sigma=5,
                     dir:str=''):
        master = Reducer.combine(to_combine, method, sigma)
        if frame == 'flat':
            master = Reducer.normalize(master)
        
        # TODO: add history to header
        # generate the right filename depending on what arguments are given
        file_name: str = f"master_{frame}"
        if obj is not None:
            file_name = f"{file_name}_{obj}"
        if filt is not None:
            file_name = f"{file_name}_filter_{filt}"
        if exposure is not None:
            file_name = f"{file_name}_{exposure}s"
        file_name = f"{file_name}.fits"

        Observation.safe_file(dir/file_name, master, header)
    
    @staticmethod
    def normalize(data:np.ndarray):
        inv_median = lambda x: 1 / np.median(x)
        return data * inv_median(data)

    def create_master_bias(self, keep_files:bool=False, force_new:bool=False):
        # check if master bias exists
        master_exist = False
        if not master_exist or force_new:
            biases = [d for d in self.data.bias.data()]
            header = next(self.data.bias.headers())
            header['combined'] = True
            header['ncombine'] = len(biases)
            master = self.stack_images(biases, 'bias', header, dir=self.data.reduced_path)
            self.data.masters['bias'] = master

        return self.data.masters['bias']
