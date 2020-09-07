#from .main_logger import BaseLogger
import numpy as np
import os



#class NumpyLogger(BaseLogger):
class NumpyLogger():
    r''' Numpy Logger

    args:
        save_path: *.npz file name.

    '''
    def __init__(self, save_path):
        self.save_path = save_path
        self.state_numbers = {}

    def end(self):
        base_dir = os.path.split(self.save_path)[0]
        if(not os.path.exists(base_dir)):
            os.makedirs(base_dir)
        np.savez(self.save_path, **self.state_numbers)
    