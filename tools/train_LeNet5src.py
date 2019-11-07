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

from config_LeNet5src import cfg
from src.utils import *
from src.speedlimit import speedlimit
from src.LeNet5_src import LeNet5_src



def train_net(args, cfg, net):
    tblogger = SummaryWriter(cfg.TRAIN_LOG_DIR)
    log_file = open(cfg.LOG_FILE, 'w')

    record_srt = 'Softmax: {:<8}, Train_set: {:<8}, Classes: {:<8}  VggNet-nondropout'.format(cfg.SOFTMAX,cfg.TRAIN_SET, args.classes)
    log_print(record_srt, log_file)

    listDataset = speedlimit(args, cfg)                      ## TODO

    train_loader = DataLoader(listDataset,
                              batch_size=cfg.TRAIN_BZ,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    ## TODO
    optimizer = optim.Adam(net.parameters(),
                           lr=cfg.TRAIN_LR,
                           betas=(0.9, 0.999),
                           eps=1e-08)

    ## TODO
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()


    itr = 0
    max_itr = cfg.TRAIN_EPOCHS * len(train_loader)
    print(itr, max_itr, len(train_loader))
    net.train()
    for epoch in range(cfg.TRAIN_EPOCHS):
        print('Starting epoch {}/{}.'.format(epoch + 1, cfg.TRAIN_EPOCHS))
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i_batch, blob in enumerate(train_loader):
            data_time.update(time.time() - end)                 # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, epoch, cfg.TRAIN_LR)  ## TODO
            Iin = blob['image'].cuda()
            pred = net(Iin)
            loss = criterion(pred, blob['label'].type(torch.LongTensor).cuda())
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
        save_path = os.path.join(cfg.WORK_DIR, '%s_itr%d.pth' % (epoch, itr))
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)



def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('period', type=str, choices=['trainval', 'test'],
                        help='run mode')
    parser.add_argument('classes', type=int,
                        help='samples classes')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed = time.time()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    if args.period == 'trainval':
        cfg.LOG_FILE = cfg.WORK_DIR + '/' + cfg.DATE + '.txt'
        if not os.path.isdir(cfg.WORK_DIR):
            os.makedirs((cfg.WORK_DIR))
        if not os.path.isdir(cfg.TRAIN_LOG_DIR):
            os.makedirs((cfg.TRAIN_LOG_DIR))

    model = LeNet5_src(1, args.classes, cfg.SOFTMAX, cfg.DROPOUT)

    if cfg.TRAIN_CKPT:
        model.load_state_dict(torch.load(cfg.TRAIN_CKPT))
        print('Model loaded from {}'.format(cfg.TRAIN_CKPT))
    if cfg.GPU:
        os.environ['CUDA_VISILBE_DEVICES'] = cfg.GPU_ID
        model = nn.DataParallel(model)    ## dist train
        model = model.cuda()
    try:
        train_net(args, cfg, model)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), cfg.WORK_DIR + '/' + 'INTERRUPTED.pth')   ##TODO
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
