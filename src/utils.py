import cv2
import torch
import torchvision
import numpy as np
from skimage import exposure
from termcolor import cprint


CLASS = ('p38_5', 'p38_10', 'p38_15', 'p38_20', 'p38_30', 'p38_35',
         'p38_40', 'p38_50', 'p38_60', 'p38_70', 'p38_80', 'p38_90',
         'p38_100', 'p38_110', 'p38_120', 'unknown', 'unknown2', 'unknown3')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n=1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count


## todo
def adjust_lr(optimizer, epoch, lr):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def log_print(text, log_file, color = None, on_color = None, attrs = None):
    print(text, file=log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)


def preprocess(img_path, category_id=None, aug=False):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (32, 32))

    # if aug:
    #     img = extra_aug(img)

    img = (img / 255.).astype(np.float32)
    img = exposure.equalize_adapthist(img)

    # Convert to one-hot encoding
    # if category_id is not None:
    #     label = np.eye(len(CLASS))[category_id]

    img = img.reshape(img.shape + (1,)).transpose((2,0,1)).astype(np.float32)

    return img, category_id


def extra_aug(img):
    noise = np.random.randint(5, size=(164, 278, 4), dtype='uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 255:
                img[i][j] += noise[i][j]

    return img