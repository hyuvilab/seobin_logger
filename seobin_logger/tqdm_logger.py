from .main_logger import BaseLogger
from tqdm import tqdm
# https://tqdm.github.io/docs/tqdm/
# https://medium.com/@philipplies/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5



class TQDMLogger(BaseLogger):
    r'''
    TQDMLogger

    '''
    def __init__(self):
        super(TQDMLogger, self).__init__()
        self.verbose_string = ['{:.6f}', '{:.3f}', '{:.2e}']

    def start(self):
        super(TQDMLogger, self).start()

        self.global_iter_tqdm = tqdm(total=self.train_iter, position=1)
        self.state_tqdm = {}
        for i, state in enumerate(self.state_dict.keys()):
            self.state_tqdm[state] = tqdm(total=0, bar_format='{desc}', position=i+2)
        self.validation_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+2)
        self.void_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+3)
        self.info_verbose_tqdm = tqdm(total=0, bar_format='{desc}', position=len(self.state_dict)+4)

    def set_info(self, info_str):
        self.info_verbose_tqdm.set_description_str('\t\t[* {}] '.format(self.main_logger.global_iter) + info_str)

    def step(self, log_dict):
        self.global_iter_tqdm.update()
        for key in log_dict.keys():
            desc_string = ('\t >> {}: ' + self.verbose_string[0]).format(key, log_dict[key])
            self.state_tqdm[key].set_description_str(desc_string)

