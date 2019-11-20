#!/usr/bin/python

import os
import pdb
import cv2
import random
import math
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser('split Images dir and Annotation dir!')
    parser.add_argument('srcpath', help='all files list.')
    parser.add_argument('dstpath', help='save path .')
    parser.add_argument('txtfile', help='txt file')
    parser.add_argument('cpath', help='category path')
    parser.add_argument('jsonfile', help='json file')
    parser.add_argument('dist', help='sample dist file')
    return parser.parse_args()

def log_print(text, log_file, color = None, on_color = None, attrs = None):
    print(text, file=log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)



def rename():
    dir_path = '/data/workspace/speed-limit/Images/unknown3/'
    fs = os.listdir(dir_path)
    for fname in fs:
        new_name = 're_' + fname
        os.rename(dir_path + fname, dir_path + new_name)
    print('Done')


def splitsample(args):
    img_files = []
    fs = os.listdir(args.srcpath)
    idx = 1
    for f in fs:
        tmp_path = os.path.join(args.srcpath, f)
        print(tmp_path, idx)
        idx = idx + 1
        if not os.path.isdir(tmp_path):
            if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.jpeg' or os.path.splitext(f)[1] == '.png':
                img_files.append(f)
    print("has read all files")
    img_files_set = set(img_files)
    allfiles = list(img_files_set)    # set to list
    test_file = random.sample(allfiles, math.ceil(len(allfiles) * 0.3))  # 随机抽选 0.3
    test_file_set = set(test_file)
    trainval_file_set = img_files_set - test_file_set
    # val_file = random.sample(trainval_file_set, math.ceil(len(allfiles) * 0.2))
    # val_file_set = set(val_file)
    # train_file_set = trainval_file_set - val_file_set
    with open(args.dstpath + 'trainval.txt', 'a') as fw1:
        for name in trainval_file_set:
            fw1.write(args.srcpath + name + '\n')
    # with open(args.dstpath + 'val.txt', 'w') as fw2:
    #     for name in val_file_set:
    #         fw2.write(args.srcpath + name + '\n')
    with open(args.dstpath + 'test.txt', 'a') as fw3:
        for name in test_file_set:
            fw3.write(args.srcpath + name + '\n')
    print(args.srcpath)

def get_txt(args):
    with open(args.txtfile,'a') as fw: ## w
        fs = os.listdir(args.srcpath)
        idx = 1
        for fname in fs:
            img_path = os.path.join(args.srcpath, fname)
            fw.write(img_path + '\n')
            # print(img_path, idx)
            idx = idx + 1
    # print("has read all files")


class get_json(object):
    def __init__(self, args):
        self.c_path = args.cpath
        self.m_categories = None
        self.m_img_set = args.txtfile
        self.m_img_index = None
        self.save_file = args.jsonfile

        self.m_json_dict = {"images": [],
                            "annotations": []}

        self._load_img_index()
        self._load_categories()

    def _load_img_index(self):
        with open(self.m_img_set, 'r') as fsets:
            self.m_img_index = list(map(lambda x: x.strip(), fsets.readlines()))

    def _load_categories(self):
        with open(self.c_path, 'r') as fcat:
            cats = fcat.readlines()
            cats = list(map(lambda x: x.strip(), cats))
            self.m_categories = dict(zip(cats, range(len(cats))))

    def convert(self):
        image_id = 1
        for idx in tqdm(range(len(self.m_img_index)), ncols=100, desc="josn"):
            # pdb.set_trace()
            img_path = self.m_img_index[idx]
            img = cv2.imread(img_path)
            if not os.path.exists(img_path):
                print(img_path)
                raise ValueError("Non existed img path: %s" % img_path)

            label = {'image': None, 'annotation': []}
            label['image'] = dict(
                file_name=img_path,
                height=img.shape[0],
                width=img.shape[1],
            )
            self.m_json_dict['images'].append(label['image'])

            category = os.path.split(os.path.split(img_path)[0])[1]
            category_id = self.m_categories[category]
            label['annotation'] = {'category_id': category_id, 'image_id': image_id}
            self.m_json_dict['annotations'].append(label['annotation'])

            image_id += 1

        with open(self.save_file, 'w') as fjson:
            json.dump(self.m_json_dict, fjson)

def generate_data_dist(args):
    with open(args.txtfile, 'r') as fsets:
        name_list = list(map(lambda x: x.strip(), fsets.readlines()))
    with open(args.jsonfile, 'r') as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
    with open(args.cpath, 'r') as fcat:
        cats = fcat.readlines()
        cats = list(map(lambda x: x.strip(), cats))

    label = np.zeros(18)
    ratio = np.zeros(18)
    for idx in range(len(name_list)):
        fname = images[idx]['file_name']
        image_id = annotations[idx]['image_id']
        category_id = annotations[idx]['category_id']
        label[category_id] += 1

    result_file = open(args.dist, 'w')
    log_print('***************dist: ********************', result_file)
    log_print('{:<8}{:<15}{:<15}{:<8}'.format('idx', 'category', 'count', 'ratio'), result_file)
    log_print('----------------------------------------', result_file)
    ratio = label / np.sum(label)
    for idx in range(18):
        log_print('{:<8}{:<15}{:<15}{:<8}'.format(str([idx]), cats[idx],  int(label[idx]), round(ratio[idx], 4)), result_file)
    log_print('----------------------------------------', result_file)
    log_print('{:<8}{:<15}{:<15}{:<8}'.format('', 'total', int(np.sum(label)), 1.0), result_file)

if __name__ == '__main__':
    args = parse_args()
    # rename()

    # splitsample(args)

    # get_txt(args)

    speedlimit = get_json(args)
    speedlimit.convert()

    generate_data_dist(args)




