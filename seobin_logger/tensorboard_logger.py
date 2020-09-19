from .main_logger import BaseLogger
from tensorboardX import SummaryWriter
import os
import shutil
import subprocess
# https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
# https://pytorch.org/docs/stable/tensorboard.html
# https://www.tensorflow.org/tensorboard/image_summaries
# https://www.youtube.com/watch?v=91J7iQLq-6U
'''
TODO: 
    * Handle other stuff than scalars.. (histogram etc..)
'''

class TensorboardLogger(BaseLogger):
    r"""TensorboardLogger

    Args:
        log_dir: Directory for tensorboard log output. It will be inside log_base directory
        log_base: Base directory for all output from TensorboardLogger.
        run_server: If True, run tensorboard server while from start() to end(). 
        host: Host name fed into --host flag in tensorboard server. 
    """
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

    def tensorboard_add_image(self, tag, image, walltime=None, dataformats='CHW'):
        self.writer.add_image(tag, image, global_step=self.main_logger.global_iter,
            walltime=walltime, dataformats=dataformats)

    def tensorboard_add_histogram(self, tag, value, bins='tensorflow', walltime=None, max_bins=None):
        self.writer.add_histogram(tag, value, global_step=self.main_logger.global_iter,
            walltime=walltime, bins=bins, max_bins=max_bins)

    def step(self, log_dict):
        for key in log_dict.keys():
            self.writer.add_scalar(key, log_dict[key], self.main_logger.global_iter)

    def end(self):
        self.writer.close()
        if(self.run_server and hasattr(self, 'tensorboard_proc')):
            self.tensorboard_proc.kill()