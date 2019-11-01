import sys
sys.path.append('./')

import os
import time
import pdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import cfg
from src.utils import *
from src.speedlimit import speedlimit
from src.model import LeNet5

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--gpus', default=2, type=int,
                        help='GPU number used for testing')
    parser.add_argument('--proc_per_gpu', default=1, type=int,
                        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--eval', type=str, nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        help='eval types')
    parser.add_argument('--show',  action='store_true',
                        help='show results')
    parser.add_argument('--savepath',default=None, help='save result')
    args = parser.parse_args()
    return args


def test(cfg, net):
    result_file = open(cfg.RESULT, 'w')

    listDataset = speedlimit(cfg)

    test_loader = DataLoader(listDataset,
                             batch_size=cfg.TEST_BZ,
                              pin_memory=True)

    net.eval()
    right_idx = 0
    itr = 0
    c_0, c_1, c_2, c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16,c_17 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9, acc_10, acc_11, acc_12, acc_13, acc_14,acc_15, acc_16, acc_17 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for i_batch, blob in enumerate(test_loader):
        Iin = blob['image'].cuda()
        pred = net(Iin)
        score = torch.max(pred)
        # pdb.set_trace()
        pred = torch.where(pred<score, torch.full_like(pred, 0), pred).cpu()
        pred_lable = torch.nonzero(pred[0])
        print_str = 'image_id: {0}; label: {1}; pred: {2}\t'.format(blob['fname'], blob['label'], pred_lable)
        log_print(print_str, result_file, color='yellow', attrs=['bold'])
        if blob['label'] == 0:
            c_0 += 1
            if blob['label'] == pred_lable:
                acc_0 += 1

        if blob['label'] == 1:
            c_1 += 1
            if blob['label'] == pred_lable:
                acc_1 += 1

        if blob['label'] == 2:
            c_2 += 1
            if blob['label'] == pred_lable:
                acc_2 += 1

        if blob['label'] == 3:
            c_3 += 1
            if blob['label'] == pred_lable:
                acc_3 += 1

        if blob['label'] == 4:
            c_4 += 1
            if blob['label'] == pred_lable:
                acc_4 += 1

        if blob['label'] == 5:
            c_5 += 1
            if blob['label'] == pred_lable:
                acc_5 += 1

        if blob['label'] == 6:
            c_6 += 1
            if blob['label'] == pred_lable:
                acc_6 += 1

        if blob['label'] == 7:
            c_7 += 1
            if blob['label'] == pred_lable:
                acc_7 += 1

        if blob['label'] == 8:
            c_8 += 1
            if blob['label'] == pred_lable:
                acc_8 += 1

        if blob['label'] == 9:
            c_9 += 1
            if blob['label'] == pred_lable:
                acc_9 += 1

        if blob['label'] == 10:
            c_10 += 1
            if blob['label'] == pred_lable:
                acc_10 += 1

        if blob['label'] == 11:
            c_11 += 1
            if blob['label'] == pred_lable:
                acc_11 += 1
        if blob['label'] == 12:
            c_12 += 1
            if blob['label'] == pred_lable:
                acc_12 += 1

        if blob['label'] == 13:
            c_13 += 1
            if blob['label'] == pred_lable:
                acc_13 += 1

        if blob['label'] == 14:
            c_14 += 1
            if blob['label'] == pred_lable:
                acc_14 += 1

        if blob['label'] == 15:
            c_15 += 1
            if blob['label'] == pred_lable:
                acc_15 += 1

        # if blob['label'] == 16:
        #     c_16 += 1
        #     if blob['label'] == pred_lable:
        #         acc_16 += 1
        #
        # if blob['label'] == 17:
        #     c_17 += 1
        #     if blob['label'] == pred_lable:
        #         acc_17 += 1

        if pred_lable == blob['label']:
            right_idx += 1
        itr += 1
    result0 = acc_0 * 1.0 / c_0
    result1 = acc_1 * 1.0 / c_1
    result2 = acc_2 * 1.0 / c_2
    result3 = acc_3 * 1.0 / c_3
    result4 = acc_4 * 1.0 / c_4
    result5 = acc_5 * 1.0 / c_5
    result6 = acc_6 * 1.0 / c_6
    result7 = acc_7 * 1.0 / c_7
    result8 = acc_8 * 1.0 / c_8
    result9 = acc_9 * 1.0 / c_9
    result10 = acc_10 * 1.0 / c_10
    result11 = acc_11 * 1.0 / c_11
    result12 = acc_12 * 1.0 / c_12
    result13 = acc_13 * 1.0 / c_13
    result14 = acc_14 * 1.0 / c_14
    result15 = acc_15 * 1.0 / c_15
    # result16 = acc_16 * 1.0 / c_16
    # result17 = acc_17 * 1.0 / c_17
    result = right_idx * 1.0 / itr
    log_print(result0, result_file, color='yellow', attrs=['bold'])
    log_print(result1, result_file, color='yellow', attrs=['bold'])
    log_print(result2, result_file, color='yellow', attrs=['bold'])
    log_print(result3, result_file, color='yellow', attrs=['bold'])
    log_print(result4, result_file, color='yellow', attrs=['bold'])
    log_print(result5, result_file, color='yellow', attrs=['bold'])
    log_print(result6, result_file, color='yellow', attrs=['bold'])
    log_print(result7, result_file, color='yellow', attrs=['bold'])
    log_print(result8, result_file, color='yellow', attrs=['bold'])
    log_print(result9, result_file, color='yellow', attrs=['bold'])
    log_print(result10, result_file, color='yellow', attrs=['bold'])
    log_print(result11, result_file, color='yellow', attrs=['bold'])
    log_print(result12, result_file, color='yellow', attrs=['bold'])
    log_print(result13, result_file, color='yellow', attrs=['bold'])
    log_print(result14, result_file, color='yellow', attrs=['bold'])
    log_print(result15, result_file, color='yellow', attrs=['bold'])
    # log_print(result16, result_file, color='yellow', attrs=['bold'])
    # log_print(result17, result_file, color='yellow', attrs=['bold'])
    log_print(result, result_file, color='yellow', attrs=['bold'])


if __name__ == '__main__':
    os.environ['CUDA_VISILBE_DEVICES'] = cfg.GPU_ID

    model = LeNet5(1, 18)
    model = nn.DataParallel(model)  ## dist train
    model = model.cuda()

    if cfg.TEST_CKPT:
        model.load_state_dict(torch.load(cfg.TEST_CKPT))
        print('Model loaded from {}'.format(cfg.TEST_CKPT))
    test(cfg, model)

