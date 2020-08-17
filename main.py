from logger import *
from time import sleep
import random
import math



if __name__=='__main__':
    print('[*] Logger test start\n')

    train_iteration = 10

    train_logger = MainLogger([
        'test1', 'test2'
        ], train_iteration)
    train_logger <= TQDMLogger()
    #train_logger <= TensorboardLogger(log_dir='tensorboard_test')
    train_logger.start()

    for i in range(train_iteration):
        train_logger.step({
            'test1': random.random(),
            'test2': 1. + random.random()*0.01,
        })
        sleep(1)

        if(i % 5 == 0):
            train_logger.set_info('[*] some break at {}'.format(i))



