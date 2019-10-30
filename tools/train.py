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
from src.speedlimit import speedlimit
# from lib.loss.loss import *
from src.utils import *
# from lib.localdimming.common import *


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


def train_net(cfg, net=None):
    tblogger = SummaryWriter(cfg.TRAIN_LOG_DIR)
    log_file = open(cfg.LOG_FILE, 'w')

    listDataset = speedlimit(cfg)                      ## TODO

    train_loader = DataLoader(listDataset,
                              batch_size=cfg.TRAIN_BZ,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    '''
    ## TODO
    optimizer = optim.Adam(net.parameters(),
                           lr=cfg.TRAIN_LR,
                           betas=(0.9, 0.999),
                           eps=1e-08)

    ## TODO
    criterion = nn.MSELoss(size_average=True)
    '''

    itr = 0
    max_itr = cfg.TRAIN_EPOCHS * len(train_loader)
    print(itr, max_itr, len(train_loader))
    # net.train()
    for epoch in range(cfg.TRAIN_EPOCHS):
        print('Starting epoch {}/{}.'.format(epoch + 1, cfg.TRAIN_EPOCHS))
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        pdb.set_trace()
        for  i_batch, img, category_id, image_id in enumerate(train_loader):
            pdb.set_trace()
            data_time.update(time.time() - end)                 # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, epoch, cfg.TRAIN_LR)  ## TODO

            ## todo
            Iouts = net(Iins)
            loss = criterion(Iins, Iouts)
            losses.update(loss.item(), cfg.TRAIN_BZ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            print_str = 'Epoch: [{0}/{1}]\t'.format(epoch, cfg.TRAIN_EPOCHS)
            print_str += 'Batch: [{0}]/{1}\t'.format(i_batch + 1, listDataset.__len__() // cfg.TRAIN_BZ)
            print_str += 'LR: {0}\t'.format(now_lr)
            print_str += 'Data time {data_time.cur:.3f}({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_str += 'Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            print_str += 'Loss {loss.cur:.4f}({loss.avg:.4f})\t'.format(loss=losses)
            log_print(print_str, log_file, color="green", attrs=["bold"])

            tblogger.add_scalar('loss', losses.avg, itr)
            tblogger.add_scalar('lr', now_lr, itr)

            end = time.time()
            itr += 1
        save_path = os.path.join(cfg.TRAIN_CKPT_DIR, '%s_itr%d.pth' % (epoch, itr))
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)


if __name__ == '__main__':

    seed = time.time()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    train_net(cfg)

    model = EDSR(1, 1)  ##TODO

    if cfg.TRAIN_CKPT:
        net.load_state_dict(torch.load(cfg.TRAIN_CKPT))
        print('Model loaded from {}'.format(cfg.TRAIN_CKPT))
    if cfg.GPU:
        os.environ['CUDA_VISILBE_DEVICES'] = cfg.GPU_ID
        model = nn.DataParallel(model)    ## dist train
        model = model.cuda()
    try:
        train_net(cfg, net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), cfg.EXP + 'INTERRUPTED.pth')   ##TODO
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

