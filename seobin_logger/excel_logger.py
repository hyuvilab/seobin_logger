#from .main_logger import BaseLogger
from openpyxl import Workbook
import os
# https://myjamong.tistory.com/51
'''
TODO:   * Check time load

'''


#class ExcelLogger(BaseLogger):
class ExcelLogger():
    r''' Excel Logger
    (Depricated. It is not optimized.)

    args:   
        save_path: *.xlsx file name.

    '''
    def __init__(self, save_path):
        self.save_path = save_path
        self.workbook = Workbook()
        self.state_numbers = {}

    def __make_row(self, i):
        states = list(self.state_numbers.keys())
        states.remove('validation')
        row = [self.state_numbers[state][i][1] for state in states]
        if(i < len(self.state_numbers['validation'])):
            row += [self.state_numbers['validation'][i][1]]
        return row

    def end(self):
        base_dir = os.path.split(self.save_path)[0]
        if(not os.path.exists(base_dir)):
            os.makedirs(base_dir)

        self.workbook.active.append(list(self.state_numbers.keys()))
        max_length = max([len(self.state_numbers[state]) for state in self.state_numbers.keys()])
        for i in range(max_length):
            row = self.__make_row(i)
            self.workbook.active.append(row)
        '''
        # This is a row-wise code.
        for state in self.state_numbers.keys():
            row = [state] + [self.state_numbers[state][i][1] for i in range(len(self.state_numbers[state]))]
            self.workbook.active.append(row)
        '''
        self.workbook.save(self.save_path)
        self.workbook.close()
