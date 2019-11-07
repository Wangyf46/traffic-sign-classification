#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

proc_num=$(echo $CUDA_VISIBLE_DEVICES | awk '{len=split($0,arr,","); print len}')

period='test'
classes=18
#vis=True

script_path=$(cd "$(dirname "$0")"; pwd)

python $script_path/../tools/test.py $period $classes $vis
