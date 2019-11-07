#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
proc_num=$(echo $CUDA_VISIBLE_DEVICES | awk '{len=split($0,arr,","); print len}')

period='trainval'
classes=18

script_path=$(cd "$(dirname "$0")"; pwd)

python $script_path/../tools/train_LeNet5src.py $period $classes