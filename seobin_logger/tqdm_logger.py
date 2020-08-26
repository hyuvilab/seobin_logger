from .main_logger import BaseLogger
from tqdm import tqdm
# https://tqdm.github.io/docs/tqdm/
# https://medium.com/@philipplies/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5
'''
TODO:   * Need to fix wierd output on Interrupt signal >> fixed
        * Need to fix wierd output after the end of the program
'''


class TQDMLogger(BaseLogger):
    r'''
    TQDMLogger

    '''
    def __init__(self):
        super(TQDMLogger, self).__init__()

    def __verbose_state(self, state, avg=False):
        verbose_string = ['{:.6f}', '{:.3f}', '{:.2e}']
        val = self.state_dict[state].avg if(avg) else self.state_dict[state].val
        if(val > 10):
            return verbose_string[1].format(val)
        elif(val < 0.001):
            return verbose_string[2].format(val)
        else:
            return verbose_string[0].format(val)


    def start(self):
        super(TQDMLogger, self).start()
        self.global_iter_tqdm = tqdm(total=self.train_iter, position=1)
        self.state_tqdm = {}
        for i, state in enumerate(self.state_dict.keys()):
            self.state_tqdm[state] = tqdm(total=0, bar_format='{desc}', position=i+2)
        self.validation_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+2)
        self.void_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+3)
        self.info_verbose_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+4)

    def tqdm_set_info(self, info_str):
        self.info_verbose_tqdm.set_description_str('\t\t[* it{}] '.format(self.main_logger.global_iter) + info_str)

    def step(self, log_dict):
        self.global_iter_tqdm.update()
        for key in log_dict.keys():
            desc_string = '\t >> {}: (cur: {}, avg: {})'.format(
                key, self.__verbose_state(key), self.__verbose_state(key, avg=True))
            self.state_tqdm[key].set_description_str(desc_string)

    def validation(self, val):
        self.validation_tqdm.set_description_str('\t >> validation: {:.3f}'.format(val))

    def end(self):
        self.global_iter_tqdm.close()
        [this_tqdm.close() for this_tqdm in self.state_tqdm.values()]
        self.validation_tqdm.close()
        self.void_tqdm.close()
        self.info_verbose_tqdm.close()

