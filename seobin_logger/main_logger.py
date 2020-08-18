'''
TODO:   * add tensorboard
            >> handle refreshing!!!
            >> set color??

        * add excel
        * add matplotlib?

        * tqdm exit problem
'''
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

    '''
    def __init__(self, state_list, train_iter, loggers=[]):
        self.global_iter = 0
        self.train_iter = train_iter
        self.loggers = loggers
        self.__compile_state_list(state_list)
        self.__started = False

    def __le__(self, x):
        x.main_logger = self
        self.loggers.append(x)

    def __compile_state_list(self, state_list):
        self.state_dict = {}
        self.state_tqdm = {}
        for state in state_list:
            assert state != 'validation'
            self.state_dict[state] = RunningAverageMeter(0.97)

    def __update_state_dict(self, log_dict):
        for key in log_dict.keys():
            self.state_dict[key].update(log_dict[key])
        

    def set_info(self, info_str):
        for logger in self.loggers:
            if(hasattr(logger, 'set_info')):  
                logger.set_info(info_str)

    def start(self):
        self.__started = True
        [logger.start() for logger in self.loggers]

    def get_state(self, key, avg=True):
        output_state = self.state_dict[key].avg if(avg) else self.state_dict[key].val
        return output_state

    def reset_state(self, key):
        self.state_dict[key].reset()

    def step(self, log_dict, validation_hook=None):
        assert set(log_dict.keys()) == set(self.state_dict.keys()), 'Need to feed all states.'
        assert self.__started, 'Need to start logger before step'
        self.global_iter += 1
        self.__update_state_dict(log_dict)

        [logger.step(log_dict) for logger in self.loggers]





class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    """from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py"""

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

