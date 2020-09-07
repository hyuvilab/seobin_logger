import seobin_logger
import argparse
import random
import math
from time import sleep


def validation(input_number):
    return input_number * 10

def best_validation(logger):
    logger.tqdm_set_info('Got best validation value: {:.3f}'.format(logger.valid_val_b))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-iteration', type=int, default=500)
    parser.add_argument('--time-interval', type=float, default=0.5)
    parser.add_argument('--tensorboard-log', default='test')
    parser.add_argument('--tensorboard-host', default='166.104.140.111')
    args = parser.parse_args()
    print('[*] Logger validation test start\n')

    train_logger = seobin_logger.MainLogger([
        'random', 'noisy_linear', 'noisy_sinusoid', 'validation'
        ], train_iter=args.train_iteration)
    train_logger <= seobin_logger.TensorboardLogger(
        args.tensorboard_log, run_server=True,
        host=args.tensorboard_host
    ) 
    train_logger <= seobin_logger.TQDMLogger() 
    train_logger <= seobin_logger.NumpyLogger('test.npz') 
    train_logger.start()

    for i in range(args.train_iteration):
        train_logger.step({
            'random': random.random(),
            'noisy_linear': i + random.random()*0.05,
            'noisy_sinusoid': math.sin(i/20.) + random.random()*0.05,
            'validation': (lambda: validation(random.random()), 10)
            })
        sleep(args.time_interval)

    train_logger > seobin_logger.ExcelLogger('save.xlsx')

