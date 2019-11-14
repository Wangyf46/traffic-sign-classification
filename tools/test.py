import sys
sys.path.append('./')

import os
import time
import pdb
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tools.config import cfg
from src.utils import *
from src.speedlimit import speedlimit
from src.LeNet5 import LeNet5



def test(args, cfg, net):
    result_file = open(cfg.RESULT, 'w')

    listDataset = speedlimit(args, cfg)

    test_loader = DataLoader(listDataset,
                             batch_size=cfg.TEST_BZ,
                             pin_memory=True)

    net.eval()
    TP = np.zeros(18)
    T = np.zeros(18)
    for i_batch, blob in enumerate(test_loader):
        if args.vis:
            img = blob['image'].squeeze(0).numpy().transpose((1,2,0))
            cv2.imshow('src', img)
            cv2.waitKey(0)
        Iin = blob['image'].cuda()
        pred = net(Iin)
        if cfg.SOFTMAX is False:
            pred = torch.nn.functional.softmax(pred)
        ## gpu->cpu
        pred_cpu = pred.detach().cpu().numpy()
        score = np.max(pred_cpu)
        pred_cpu = np.where(pred_cpu<score, np.full_like(pred_cpu, 0), pred_cpu)
        pred_lable = np.nonzero(pred_cpu[0])

        if blob['label'].numpy() == pred_lable:
            TP[pred_lable] += 1
        T[blob['label'].numpy()] += 1

        print_str = '{0}; label: {1}; pred: {2}\t'.format(blob['fname'][0], blob['label'].numpy()[0], pred_lable[0][0])
        log_print(print_str, result_file, color='yellow', attrs=['bold'])

    RECALL = TP / T
    avg_acc = np.sum(TP) / np.sum(T)
    log_print(RECALL, result_file, color='yellow', attrs=['bold'])
    log_print(avg_acc, result_file, color='yellow', attrs=['bold'])

    with open('/data/workspace/speed-limit/speedlimit.label', 'r') as fcat:
        cats = fcat.readlines()
        cats = list(map(lambda x: x.strip(), cats))

    log_print('***************dist: ****************************************', result_file)
    log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('idx', 'category', 'TP', 'T', 'RECALL'), result_file)
    log_print('-------------------------------------------------------------', result_file)
    for idx in range(18):
        log_print('{:<8}{:<15}{:<15}{:<15}{:<5}'.format(str([idx]), cats[idx], TP[idx], T[idx], RECALL[idx]), result_file)
    log_print('-------------------------------------------------------------', result_file)
    log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('', 'total', np.sum(TP), np.sum(T), avg_acc), result_file)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('period', type=str, choices=['trainval', 'test'],
                        help='run mode')
    parser.add_argument('classes', type=int,
                        help='samples classes')
    parser.add_argument('--vis', type=str, default=False,
                        help='vis sample')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if cfg.MODEL == 'LeNet5':
        from src.LeNet5 import LeNet5
        model = LeNet5(1, args.classes, cfg.SOFTMAX, cfg.DROPOUT)
    elif cfg.MODEL == 'VggNet':
        from src.vgg import VggNet
        model = VggNet(1, args.classes, cfg.SOFTMAX, cfg.DROPOUT)
    elif cfg.MODEL == 'LeNet5src':
        from src.LeNet5_src import LeNet5_src
        model = LeNet5_src(1, args.classes, cfg.SOFTMAX, cfg.DROPOUT)
    else:
        pass

    model = nn.DataParallel(model)  ## dist train
    model = model.cuda()

    if cfg.TEST_CKPT:
        print(torch.load(cfg.TEST_CKPT))
        model.load_state_dict(torch.load(cfg.TEST_CKPT))
        print('Model loaded from {}'.format(cfg.TEST_CKPT))
        print(model)
    test(args, cfg, model)

