#from .main_logger import BaseLogger
import numpy as np




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
        np.savez(self.save_path, **self.state_numbers)
    