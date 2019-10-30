import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.PERIOD = 'train'

        self.DATE = time.strftime("%Y%m%d%_H%M%S_ubuntu", time.localtime())
        self.WORK_DIR = os.path.join('/data/workspace/speed-limit/workout/', self.DATE)

        self.DATA_NAME = 'speedlimit'
        self.DATA_PATH = '/data/workspace/speed-limit/'
        self.DATA_AUG = True

        self.GPU = True
        self.GPU_ID = '2'

        self.LOSS = None  ##   todo
        self.OPTIM = None ##   todo
        self.LR = None    ##   todo

        self.TRAIN_SET = self.DATA_PATH + self.PERIOD
        self.TRAIN_LR = 0.0001
        self.TRAIN_BZ = 128
        self.TRAIN_EPOCHS = 100
        self.TRAIN_CKPT = ''
        self.TRAIN_LOG_DIR = os.path.join(self.WORK_DIR, 'tf_logs')

        self.TEST_SET = self.DATA_PATH + self.PERIOD
        self.TEST_BZ = 1
        self.TEST_CKPT = ''
        self.OUTPUT = self.WORK_DIR + 'epoch21.pkl'

        self.__check()
        self.__add_path('/home/wyf/codes/traffic-signs-classification/')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')
        if not os.path.isdir(self.WORK_DIR):
            os.makedirs((self.WORK_DIR))
        if self.PERIOD == 'train':
            self.LOG_FILE = self.WORK_DIR + self.DATE + '.txt'
            if not os.path.isdir(self.TRAIN_LOG_DIR):
                os.makedirs((self.TRAIN_LOG_DIR))


    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

cfg = Configuration()