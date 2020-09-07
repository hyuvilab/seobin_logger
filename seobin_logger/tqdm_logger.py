from .main_logger import BaseLogger
from tqdm import tqdm
# https://tqdm.github.io/docs/tqdm/
# https://medium.com/@philipplies/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5
'''
TODO:   * Need to fix wierd output on Interrupt signal >> fixed
        * Need to fix wierd output after the end of the program
'''


class TQDMLogger(BaseLogger):
    r''' TQDM Logger

    '''
    def __init__(self):
        super(TQDMLogger, self).__init__()

    def __verbose_state(self, val):
        verbose_string = ['{:.6f}', '{:.3f}', '{:.2e}']
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
        self.void_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+2)
        self.info_verbose_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+3)

    def tqdm_set_info(self, info_str):
        self.info_verbose_tqdm.set_description_str('\t\t[* it{}] '.format(self.main_logger.global_iter) + info_str)

    def step(self, log_dict):
        self.global_iter_tqdm.update()
        for key in log_dict.keys():
            desc_string = '\t >> {}: {}'.format(key, self.__verbose_state(log_dict[key]))
            self.state_tqdm[key].set_description_str(desc_string)

    def end(self):
        self.global_iter_tqdm.close()
        [this_tqdm.close() for this_tqdm in self.state_tqdm.values()]
        self.void_tqdm.close()
        self.info_verbose_tqdm.close()

