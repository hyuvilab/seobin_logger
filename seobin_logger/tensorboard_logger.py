from .main_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import subprocess
# https://tensorboardx.readthedocs.io/en/latest/tensorboard.html

# https://pytorch.org/docs/stable/tensorboard.html
# https://www.tensorflow.org/tensorboard/image_summaries
# https://www.youtube.com/watch?v=91J7iQLq-6U
'''
TODO: 
    * Change torch tensorboard to tensorboardX
    * Handle other stuff than scalars.. (histogram etc..)
    * Make overlapping graphs (https://stackoverflow.com/questions/37146614/tensorboard-plot-training-and-validation-losses-on-the-same-graph/62203250#62203250)
'''

class TensorboardLogger(BaseLogger):
    r'''TensorboardLogger

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
        logdir = os.path.join(self.log_base, self.log_dir)
        if(os.path.exists(logdir)): shutil.rmtree(logdir)
        self.writer = SummaryWriter(log_dir=logdir, flush_secs=1.)
        if(self.run_server):
            f = open(os.path.join(self.log_base, 'tensorboard_server_log.out'), 'w')
            self.tensorboard_proc = subprocess.Popen(
                ['tensorboard', '--logdir', self.log_base, '--host', self.host], stderr=f
            )

    def tensorboard_plot_image(self, name, image):
        # Handle preprocessing image here?
        self.writer.add_image(name, image, self.main_logger.global_iter)

    def step(self, log_dict):
        for key in log_dict.keys():
            self.writer.add_scalar(key, log_dict[key], self.main_logger.global_iter)

    def validation(self, val):
        self.writer.add_scalar('validation', val, self.main_logger.global_iter)

    def end(self):
        self.writer.close()
        if(self.run_server and hasattr(self, 'tensorboard_proc')):
            self.tensorboard_proc.kill()




