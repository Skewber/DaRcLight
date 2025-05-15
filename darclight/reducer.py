import numpy as np
from astropy.stats import sigma_clip
from .io import Observation

class Reducer():
    def __init__(self, data:Observation):
        self.data = data
        self.method = 'sigmaclip'

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
    def generate_filename(frame:str, obj:str=None,
                          filt:str=None, exposure:str=None):
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
    
    @staticmethod
    def normalize(data:np.ndarray):
        return data / np.median(data)

    def create_master_bias(self, force_new:bool=False, method:str=None):
        # check if master bias exists
        master_exist = False
        if not master_exist or force_new:
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
            self.data.masters['bias'] = master
            
        return self.data.masters['bias']
