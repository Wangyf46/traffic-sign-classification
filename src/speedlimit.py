import pdb
import torch
import cv2
import json
import warnings

from torch.utils.data import Dataset

from src.utils import *

class speedlimit(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.PERIOD == 'train':
            self.img_file = cfg.TRAIN_SET + '.txt'
            self.ann_file = cfg.TRAIN_SET + '.json'
        else:
            self.img_file = cfg.TEST_SET + '.txt'
            self.ann_file = cfg.TEST_SET + '.json'
        with open(self.img_file, 'r') as fsets:
            self.name_list = list(map(lambda x: x.strip(), fsets.readlines()))

        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            self.images = data['images']
            self.annotations = data['annotations']

    def __len__(self):
        return len(self.images)

    ## TODO
    def __getitem__(self, idx):
        # print(idx)    ## TODO

        fname = self.images[idx]['file_name']
        category_id = self.annotations[idx]['category_id']
        image_id = self.annotations[idx]['image_id']
        img, label = preprocess(fname, category_id)
        blob = {}
        blob['image'] = img
        blob['fname'] = fname
        if category_id == 15 or category_id == 16 or category_id == 17:
            category_id =15
        blob['label'] = category_id
        blob['image_id'] = image_id
        return blob



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config import cfg

    listDataset = speedlimit(cfg)
    # loader = DataLoader(listDataset,
    #                           batch_size=1,
    #                           shuffle=True,
    #                           num_workers=4,
    #                           pin_memory=True)

    for i_batch, blob in enumerate(listDataset):
        pdb.set_trace()