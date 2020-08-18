from .main_logger import BaseLogger
from openpyxl import Workbook
# https://myjamong.tistory.com/51
'''
TODO:   * Check time load

'''


class ExcelLogger(BaseLogger):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def start(self):
        super(ExcelLogger, self).start()
        self.workbook = Workbook()
        for i, key in enumerate(self.state_dict.keys()):
            self.workbook.active.cell(1, i+1, key)

    def step(self, log_dict):
        for i, key in enumerate(log_dict.keys()):
            self.workbook.active.cell(self.main_logger.global_iter+1, i+1, log_dict[key])
        self.workbook.save(self.save_dir)

    def end(self):
        self.workbook.close()