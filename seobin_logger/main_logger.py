'''
TODO:   
    * add matplotlib
    * add very simple one

    * Take the mean of some step interval to ease saving load!
    * Test time load for all loggers!
'''
import atexit
from numbers import Number
from collections.abc import Iterable
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

    def step(self, log_dict):
        for key in log_dict.keys():
            self.state_numbers[key].append([self.main_logger.global_iter, log_dict[key]])

    def end(self):
        for logger in self.offline_loggers:
            if(hasattr(logger, 'end')):
                logger.end()




class MainLogger(BaseLogger):
    r""" Main Logger to initiate logging

    Args: 
        state_list: List of states. They should all be fed to step afterwards.
        train_iter: Number of whole training steps. Default is None.

    Functions:
        start: Starts the logger
        step: Main function for logging step. 
        end: Ends the logger safely. 
    """
    def __init__(self, state_list, train_iter=None, step_size=1):
        self.global_iter = 0
        self.train_iter = train_iter
        self.step_size = step_size
        self.loggers = []
        self.save_logger = None
        self.__compile_state_list(state_list)
        self.__started = False
        self.__ended = False

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
        for state in state_list:
            self.state_dict[state] = ExactAverageMeter()

    def __read_value(self, value):
        # Need more strict typing here!
        if(isinstance(value, Iterable) and len(value) == 2):
            if(self.global_iter % value[1] == 0):
                if(value[0] is None):
                    return None, value[1]
                elif(isinstance(value[0], Number)):
                    return value[0], value[1]
                elif(callable(value[0])):
                    return value[0](), value[1]
            else:
                return None, value[1]
        else:
            if(value is None):
                return None, self.step_size
            elif(isinstance(value, Number)):
                return value, self.step_size
            elif(callable(value)):
                return value(), self.step_size
        raise ValueError('Wrong value type fed to log dictionary: {}'.format(value))


    def start(self):
        if(self.__started): return
        [logger.start() for logger in self.loggers]
        self.__started = True
        atexit.register(self.end)

    def start_save_logger(self):
        assert not self.__started, 'Need to call start before start_save_logger'
        if(self.save_logger is not None): return
        self.save_logger = SaveLogger()
        self.save_logger.main_logger = self
        self.loggers.append(self.save_logger)

    def step(self, log_dict):
        assert self.__started and not self.__ended, 'Step must be called between .start() and .end()'
        assert set(log_dict.keys()) == set(self.state_dict.keys()), 'Need to feed all states.'  # Need to consider this implementation
        self.global_iter += 1
        if(self.train_iter and self.global_iter > self.train_iter):
            self.end()
            return

        loggers_log_dir = {}
        for key in log_dict.keys():
            value, step_size = self.__read_value(log_dict[key])
            if(value is None):  continue
            self.state_dict[key].update(value)
            if(self.global_iter % step_size == 0):
                loggers_log_dir[key] = self.state_dict[key].avg
                self.state_dict[key].reset()
        [logger.step(loggers_log_dir) for logger in self.loggers]        

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
        if(val is None): return
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



class ExactAverageMeter(object):
    """Computes and stores the exact average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.n = 0

    def update(self, val):
        if(val is None): return
        self.n += 1
        self.avg = (self.avg * (self.n-1) + val) / self.n
        self.val = val

