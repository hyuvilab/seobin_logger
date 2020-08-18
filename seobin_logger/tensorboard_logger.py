from .main_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
# https://pytorch.org/docs/stable/tensorboard.html
'''
TODO:   * Make tensorboard webserver refreshing real-time >> let's not implement this.
        * Run tensorboard server on .start() given on flag >> done.

        * Change torch tensorboard to tensorboardX
        * Handle other stuff than scalars.. (histogram etc..)
'''

class TensorboardLogger(BaseLogger):
    r'''
    TensorboardLogger

        [*] tensorboard --logdir tensorboard_test/ --host 166.104.140.111
    '''
    def __init__(self, log_dir, log_base='tensorboard_logs', run_server=False, host='localhost'):
        super(TensorboardLogger, self).__init__()
        self.log_dir = log_dir
        self.log_base = log_base
        self.run_server = run_server
        self.host = host

    def start(self):
        super(TensorboardLogger, self).start()
        self.writer = SummaryWriter(log_dir='{}/{}'.format(self.log_base, self.log_dir), flush_secs=1.)
        if(self.run_server):
            f = open(os.path.join(self.log_base, 'tensorboard_server_log.out'), 'w')
            self.tensorboard_proc = subprocess.Popen(
                ['tensorboard', '--logdir', self.log_base, '--host', self.host],
                stderr=f
            )

    def step(self, log_dict):
        for key in log_dict.keys():
            self.writer.add_scalar(key, log_dict[key], self.main_logger.global_iter)

    def end(self):
        if(self.run_server and hasattr(self, 'tensorboard_proc')):
            self.tensorboard_proc.kill()




