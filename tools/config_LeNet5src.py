import os
import sys
import time
import torch


class Configuration():
    def __init__(self):
        self.DATE = time.strftime("%Y%m%d%_H%M%S_ubuntu", time.localtime())
        self.MODEL = 'LeNet5src'
        self.WORK_DIR = os.path.join('/data/workspace/speed-limit/workout/', self.MODEL, self.DATE)

        self.DATA_NAME = 'speedlimit'
        self.DATA_PATH = '/data/workspace/speed-limit/'
        self.DATA_AUG = True

        self.GPU = True
        self.GPU_ID = '1,2,3,4'

        self.LOSS = None  ##   todo
        self.OPTIM = None ##   todo
        self.LR = None    ##   todo

        self.TRAIN_LR = 0.0001
        self.TRAIN_BZ = 480
        self.TRAIN_EPOCHS = 100
        self.TRAIN_CKPT = ''
        self.TRAIN_LOG_DIR = os.path.join(self.WORK_DIR, 'tf_logs')

        self.TEST_BZ = 1

        # 18 CLASS + LeNet5src
        self.SOFTMAX = False
        self.DROPOUT = False
        self.TRAIN_SET = self.DATA_PATH + 'trainval'
        self.TEST_SET = self.DATA_PATH + 'test'
        self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/99_itr7800.pth'
        self.RESULT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/result_99.txt'


        self.__check()
        self.__add_path('/home/wyf/codes/traffic-signs-classification/')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')


    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

cfg = Configuration()