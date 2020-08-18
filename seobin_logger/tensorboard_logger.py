from .main_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
# https://pytorch.org/docs/stable/tensorboard.html


class TensorboardLogger(BaseLogger):
    r'''
    TensorboardLogger

        [*] tensorboard --logdir tensorboard_test/ --host 166.104.140.111
    '''
    def __init__(self, log_dir=None):
        super(TensorboardLogger, self).__init__()
        self.log_dir = log_dir

    def start(self):
        super(TensorboardLogger, self).start()
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def step(self, log_dict):
        for key in log_dict.keys():
            self.writer.add_scalar(key, log_dict[key], self.main_logger.global_iter)

