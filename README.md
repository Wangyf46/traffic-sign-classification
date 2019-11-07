# traffic-sign-classification

        # 18 CLASS + nonsoftmax + nondropout + LeNet5src 
        self.MODEL = 'LeNet5src'
        self.SOFTMAX = False
        self.DROPOUT = False
        self.TRAIN_SET = self.DATA_PATH + 'trainval'
        self.TEST_SET = self.DATA_PATH + 'test'
        self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/99_itr7800.pth'
        self.RESULT = '/data/workspace/speed-limit/workout/LeNet5src/20191106115402_ubuntu/result_99.txt'


        # 18 CLASS + nonsoftmax + nondropout + VggNet 
        # self.MODEL = 'VggNet'
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/VggNet/20191105223654_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/VggNet/20191105223654_ubuntu/result0_99_nonsoftmax_18_nondropout.txt'


        # 18 CLASS + nonsoftmax + dropout + VggNet
        # self.MODEL = 'VggNet'  
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/VggNet/20191105223718_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/VggNet/20191105223718_ubuntu/result_99_nonsoftmax_18_dropout.txt'


        # 18 CLASS + nonsoftmax + nondropout + LeNet5(best)
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = False
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105144413_ubuntu/99_itr7800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105144413_ubuntu/result_99_nonsoftmax_18_nondropout.txt'


        # 18 CLASS + softmax + dropout + LeNet5
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104164254_ubuntu/99_itr11700.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104164254_ubuntu/result_99_softmax_18.txt'
        ## old
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104174910_ubuntu/99_itr11700.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104174910_ubuntu/result_99_nonsoftmax_181.txt'


        # ## 18 CLASS + nonsoftmax + dropout + LeNet5 (?) (retrain again11.5)
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'trainval'
        # self.TEST_SET = self.DATA_PATH + 'test'
        # # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105215148_ubuntu/99_itr7800.pth'
        # # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105215148_ubuntu/result_99_nonsoftmax_dropout_18.txt'

        # 15 CLASS + nonsoftmax + dropout + LeNet5  (?) (retrain again11.5)
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = False
        # self.DROPOUT = True
        # self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105215643_ubuntu/99_itr5900.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105215643_ubuntu/result_99_nonsoftmax_dropout_15.txt'


        # 15 CLASS + nonsoftmax + nondropout + LeNet5 (best)
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = False
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105115627_ubuntu/88_itr5251.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105115627_ubuntu/result_88_nonsoftmax_15_nondropout.txt'


        # 15 CLASS + softmax + nondropout + LeNet5
        # self.MODEL = 'LeNet5'  
        # self.SOFTMAX = True
        # self.DROPOUT = False
        # self.TRAIN_SET = self.DATA_PATH + 'Non-Negative/trainval'
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191105145236_ubuntu/8_itr531.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191105145236_ubuntu/result_8_softmax_15_nondropout.txt'


        # 15 CLASS + softmax + dropout + LeNet5
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = self.DATA_PATH + 'Non-Negative/test'
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104135855_ubuntu//99_itr10900.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104135855_ubuntu/result_99_softmax_15.txt'
        ## old
        # self.TEST_CKPT = '/data/workspace/speed-limit/workout/LeNet5/20191104175418_ubuntu/99_itr8800.pth'
        # self.RESULT = '/data/workspace/speed-limit/workout/LeNet5/20191104175418_ubuntu/result_99_nonsoftmax_15.txt'


        # 18 class + softmax + dropout + LeNet5(src)
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = '/data/workspace/speed-limit-src/' + 'val'
        # self.TEST_CKPT = '/data/workspace/speed-limit-src/workout/LeNet5/20191031183744_ubuntu/99_itr19300.pth'
        # self.RESULT = '/data/workspace/speed-limit-src/workout/LeNet5/20191031183744_ubuntu/result_99_sotfmax_18.txt'


        # 16 class + softmax + dropout + LeNet5(src)----input18
        # self.MODEL = 'LeNet5'
        # self.SOFTMAX = True
        # self.DROPOUT = True
        # self.TEST_SET = '/data/workspace/speed-limit-src/' + 'val'
        # self.TEST_CKPT = '/data/workspace/speed-limit-src/workout/LeNet5/20191101110646_ubuntu/100_itr9797.pth'
        # self.RESULT = '/data/workspace/speed-limit-src/workout/LeNet5/20191101110646_ubuntu/result_100_softmax_15.txt'
