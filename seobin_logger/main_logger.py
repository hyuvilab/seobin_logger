'''
TODO:   
        * add matplotlib
        * add very simple one

        * Need to handle order of loggers
        * Test time load for all loggers!
'''
import atexit
from .excel_logger import ExcelLogger
from .numpy_logger import NumpyLogger
offline_logger_list = [ExcelLogger, NumpyLogger]




class BaseLogger(object):
    def __init__(self):
        self.main_logger = None

    def start(self):
        self.train_iter = self.main_logger.train_iter
        self.state_dict = self.main_logger.state_dict

    def step(self):
        if(not type(self) in offline_logger_list): raise NotImplementedError

    def end(self):
        pass




class SaveLogger(BaseLogger):
    def __init__(self):
        super(SaveLogger, self).__init__()
        self.state_numbers = {}
        self.offline_loggers = []

    def __le__(self, x):
        x.state_numbers = self.state_numbers
        self.offline_loggers.append(x)

    def start(self):
        super(SaveLogger, self).start()
        for state in self.state_dict.keys():
            self.state_numbers[state] = []
        self.state_numbers['validation'] = []

    def step(self, log_dict):
        for key in log_dict.keys():
            self.state_numbers[key].append([self.main_logger.global_iter, log_dict[key]])

    def validation(self, val):
        self.state_numbers['validation'].append([self.main_logger.global_iter, val])

    def end(self):
        for logger in self.offline_loggers:
            if(hasattr(logger, 'end')):
                logger.end()




class MainLogger(BaseLogger):
    r''' Main Logger to initiate logging

    Args: 
        state_list: List of states. They should all be fed to step afterwards.
        train_iter: Number of whole training steps. Default is None.

    Functions:
        start: Starts the logger
        step: Main function for logging step. 
        end: Ends the logger safely. 
    '''
    def __init__(self, state_list, train_iter=None):
        self.global_iter = 0
        self.train_iter = train_iter
        self.loggers = []
        self.save_logger = None
        self.valid_val = None
        self.valid_val_b = None
        self.__compile_state_list(state_list)
        self.__started = False
        self.__ended = False
        atexit.register(self.end)

    def __le__(self, x):
        assert not self.__started, 'Need to feed loggers before start'
        assert not self.__ended, 'Logger has ended'
        x.main_logger = self
        if(type(x) in offline_logger_list):
            if(self.save_logger is None):
                self.save_logger = SaveLogger()
                self.save_logger.main_logger = self
                self.loggers.append(self.save_logger)
            self.save_logger <= x
        else:
            self.loggers.append(x)

    def __gt__(self, x):
        assert self.save_logger is not None, 'Need to call start_save_logger before start!'
        assert self.__started, 'Need to feed offline loggers after start'
        assert not self.__ended, 'Logger has ended'
        assert type(x) in offline_logger_list, 'Need to feed offline loggers only'
        self.save_logger <= x

    def __getattr__(self, name):
        for logger in self.loggers:
            if(hasattr(logger, name)):
                # Not handling multiple logger with same function here!
                return getattr(logger, name)
        raise AttributeError("'MainLogger' object has no attributre '{}'".format(name))

    def __compile_state_list(self, state_list):
        self.state_dict = {}
        self.state_tqdm = {}
        for state in state_list:
            assert state != 'validation'
            self.state_dict[state] = RunningAverageMeter(0.97)

    def __update_state_dict(self, log_dict):
        for key in log_dict.keys():
            self.state_dict[key].update(log_dict[key])


    def start(self):
        if(self.__started): return
        [logger.start() for logger in self.loggers]
        self.__started = True

    def start_save_logger(self):
        assert not self.__started, 'Need to call start_save_logger before start'
        if(self.save_logger is not None): return
        self.save_logger = SaveLogger()
        self.save_logger.main_logger = self
        self.loggers.append(self.save_logger)

    def get_state(self, key, avg=False):
        assert key in self.state_dict.keys(), 'key {} is not in state_dict'.format(key)
        output_state = self.state_dict[key].avg if(avg) else self.state_dict[key].val
        return output_state

    def reset_state(self, key):
        self.state_dict[key].reset()

    def step(self, log_dict, validation_closure=None, validation_freq=1, 
            best_validation_closure=None, validation_mode='max'):

        assert self.__started and not self.__ended, 'Step must be called between .start() and .end()'
        assert set(log_dict.keys()) == set(self.state_dict.keys()), 'Need to feed all states.'
        assert validation_mode in ['min', 'max'], 'validation_mode should be min or max'

        self.global_iter += 1
        if(self.train_iter and self.global_iter > self.train_iter):
            self.end()
            return
        self.__update_state_dict(log_dict)
        [logger.step(log_dict) for logger in self.loggers]

        if(validation_closure and self.global_iter % validation_freq == 0):
            self.valid_val = validation_closure()
            if(self.valid_val_b):
                if(validation_mode=='min' and self.valid_val < self.valid_val_b):
                    best_valid = True
                elif(validation_mode=='max' and self.valid_val > self.valid_val_b):
                    best_valid = True
                else:
                    best_valid = False
            else:
                best_valid = True
            if(best_valid):
                self.valid_val_b = self.valid_val
                if(best_validation_closure): best_validation_closure()
            for logger in self.loggers:
                if(hasattr(logger, 'validation')):
                    logger.validation(self.valid_val)

    def end(self):
        if(self.__ended): return
        for logger in self.loggers:
            if(hasattr(logger, 'end')):
                logger.end()
        self.__ended = True






class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    """Code from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

