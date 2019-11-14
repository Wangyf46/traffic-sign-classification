import os
import sys
import time
import torch
import shutil


class Configuration():
    def __init__(self):
        self.DATA_NAME = 'speedlimit'
        self.DATA_PATH = '/data/workspace/speed-limit/'
        self.DATA_AUG = True


        ## TODO-11.6: 18 + nonsoftmax + nondropout + LeNet5src
        # self.MODEL = 'LeNet5src'
        # self.SOFTMAX = False
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/result_99.txt'


        ## TODO-11.6: 18 + nonsoftmax + nondropout + VggNet
        # self.MODEL = 'VggNet'
        # self.SOFTMAX = False
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/VggNet/20191105223654_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/VggNet/20191105223654_ubuntu/result_99_nonsoftmax_18_nondropout.txt'


        ## TODO-11.6: 18 + nonsoftmax + dropout + VggNet
        # self.MODEL = 'VggNet'
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/VggNet/20191105223718_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/VggNet/20191105223718_ubuntu/result_99_nonsoftmax_18_dropout111.txt'


        ## TODO-11.5: 18 + nonsoftmax + nondropout + LeNet5(best)
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = False
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105144413_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105144413_ubuntu/result_99_nonsoftmax_18_nondropout.txt'


        ## TODO-11.4: 18 + softmax + dropout + LeNet5
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104164254_ubuntu/99_itr11700.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104164254_ubuntu/result_99_softmax_18.txt'
        ## old
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104174910_ubuntu/99_itr11700.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104174910_ubuntu/result_99_nonsoftmax_18.txt'


        ## TODO-11.5: 18 + nonsoftmax + dropout + LeNet5 (?) (retrain again 11.5)
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105215148_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105215148_ubuntu/result_99_nonsoftmax_dropout_18.txt'


        ## TODO-11.5: 15 + nonsoftmax + dropout + LeNet5  (?) (retrain again11.5)
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105215643_ubuntu/99_itr5900.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105215643_ubuntu/result_99_nonsoftmax_dropout_15.txt'


        ## TODO-11.5: 15 + nonsoftmax + nondropout + LeNet5 (best)
        self.MODEL = 'LeNet5'
        self.SOFTMAX = False
        self.DROPOUT = False
        self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105115627_ubuntu/88_itr5251.pth'
        self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105115627_ubuntu/1result_88_nonsoftmax_15_nondropout.txt'

        # self.TEST_SET = '/data/workspace/speed-limit-crop/test'
        # self.RESULT = '/data/workspace/speed-limit-crop/result_15.txt'




        ## TODO-11.5: 15 + softmax + nondropout + LeNet5
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105145236_ubuntu/8_itr531.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105145236_ubuntu/result_8_softmax_15_nondropout.txt'


        ## TODO-11.4: 15 + softmax + dropout + LeNet5
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104135855_ubuntu//99_itr10900.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104135855_ubuntu/result_99_softmax_15.txt'
        ## old
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104175418_ubuntu/99_itr8800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104175418_ubuntu/result_99_nonsoftmax_15.txt'


        ## TODO-10.31: 18 + softmax + dropout + LeNet5(src)
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = '/data/workspace/speed-limit-src/' + 'val'
        # self.TEST_CKPT = '/data/workspace/speed-limit-src/workout/LeNet5/20191031183744_ubuntu/99_itr19300.pth'
        # self.RESULT = '/data/workspace/speed-limit-src/workout/LeNet5/20191031183744_ubuntu/result_99_sotfmax_18.txt'


        ## TODO-11.1: 16 + softmax + dropout + LeNet5(src)----input18
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = '/data/workspace/speed-limit-src/' + 'val'
        # self.TEST_CKPT = '/data/workspace/speed-limit-src/workout/LeNet5/20191101110646_ubuntu/100_itr9797.pth'
        # self.RESULT = '/data/workspace/speed-limit-src/workout/LeNet5/20191101110646_ubuntu/result_100_softmax_15.txt'


        self.DATE = time.strftime("%Y%m%d%_H%M%S_ubuntu", time.localtime())
        self.WORK_DIR = os.path.join('/data/workspace/speed-limit/workout/', self.MODEL, self.DATE)

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

        self.__check()
        self.__add_path('/home/wyf/codes/traffic-signs-classification/')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not available')

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

cfg = Configuration()