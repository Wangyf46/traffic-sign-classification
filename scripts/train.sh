#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
proc_num=$(echo $CUDA_VISIBLE_DEVICES | awk '{len=split($0,arr,","); print len}')

period='trainval'
classes=18
config=/home/wyf/codes/traffic-sign-classification/tools/config.py

script_path=$(cd "$(dirname "$0")"; pwd)

python $script_path/../tools/train.py $period $classes $config