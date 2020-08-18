import seobin_logger
from time import sleep
import argparse
import random
import math



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-iteration', type=int, default=500)
    parser.add_argument('--time-interval', type=float, default=0.5)
    args = parser.parse_args()
    print('[*] Logger test start\n')

    train_logger = seobin_logger.MainLogger([
        'random', 'noisy_linear', 'noisy_sinusoid'
        ], args.train_iteration)
    train_logger <= seobin_logger.TQDMLogger()
    train_logger <= seobin_logger.TensorboardLogger(log_dir='tensorboard_test')
    train_logger.start()

    for i in range(args.train_iteration):
        train_logger.step({
            'random': random.random(),
            'noisy_linear': i + random.random()*0.05,
            'noisy_sinusoid': math.sin(i/20.) + random.random()*0.05,
        })
        if(i % 5 == 0):
            train_logger.set_info('some break at {}'.format(i))
        sleep(args.time_interval)




