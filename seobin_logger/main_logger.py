'''
TODO:   
        * add matplotlib
        * add numpy
        * add very simple one

        * Need to handle order of loggers
        * Test time load for all loggers!
        * Think for a better implementation than __del__ in MainLogger!
'''
import signal



class BaseLogger(object):
    def __init__(self):
        self.main_logger = None

    def start(self):
        self.train_iter = self.main_logger.train_iter
        self.state_dict = self.main_logger.state_dict

    def step(self):
        raise NotImplementedError





class MainLogger(BaseLogger):
    r'''
    TrainLogger

    [*] state_list: (state_name)

        > start
        > step
        > end

    '''
    def __init__(self, state_list, train_iter=None, loggers=[]):
        self.global_iter = 0
        self.train_iter = train_iter
        self.loggers = loggers
        self.validation_value = None
        self.__compile_state_list(state_list)
        self.__started = False
        self.__ended = False

    def __le__(self, x):
        x.main_logger = self
        self.loggers.append(x)

    def __del__(self):
        self.end()

    def __getattr__(self, name):
        for logger in self.loggers:
            if(hasattr(logger, name)):
                # Not handling multiple logger with same function here!
                return logger.name  
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

    def get_state(self, key, avg=False):
        assert key in self.state_dict.keys(), 'key {} is not in state_dict'.format(key)
        output_state = self.state_dict[key].avg if(avg) else self.state_dict[key].val
        return output_state

    def reset_state(self, key):
        self.state_dict[key].reset()

    def step(self, log_dict, validation_closure=None, validation_freq=1):
        assert set(log_dict.keys()) == set(self.state_dict.keys()), 'Need to feed all states.'
        assert self.__started and not self.__ended, 'Step must be called between .start() and .end()'
        self.global_iter += 1
        self.__update_state_dict(log_dict)
        [logger.step(log_dict) for logger in self.loggers]

        if(validation_closure and self.global_iter % validation_freq == 0):
            self.validation_value = validation_closure()
            for logger in self.loggers:
                if(hasattr(logger, 'validation')):
                    logger.validation(self.validation_value)

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

