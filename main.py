from logger import *
from time import sleep
import argparse
import random
import math



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-iteration', type=int, default=500)
    args = parser.parse_args()
    print('[*] Logger test start\n')


    train_logger = MainLogger([
        'random1', 'random2', 'linear', 'quadratic', 'sinusoid'
        ], args.train_iteration)
    train_logger <= TQDMLogger()
    train_logger <= TensorboardLogger(log_dir='tensorboard_test')
    train_logger.start()

    for i in range(args.train_iteration):
        train_logger.step({
            'random1': random.random(),
            'random2': 1. + random.random()*0.01,
            'linear': 1. * i,
            'quadratic': 0.5 * (i**2),
            'sinusoid': math.sin(i/20.)
        })
        if(i % 5 == 0):
            train_logger.set_info('[*] some break at {}'.format(i))
        sleep(1)




