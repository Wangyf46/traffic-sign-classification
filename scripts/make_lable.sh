#!/usr/bin/env bash


srcpath='/data/workspace/speed-limit/Images/val/unknown3/'
cpath='/data/workspace/speed-limit/speedlimit.label'
txtfile='/data/workspace/speed-limit/val.txt'
jsonfile='/data/workspace/speed-limit/val.json'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../tools/make_label.py $srcpath $txtfile $cpath $jsonfile