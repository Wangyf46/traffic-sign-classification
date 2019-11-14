#!/usr/bin/env bash

## TODO:speed-limit
#srcpath='/data/workspace/speed-limit/Images/p38_120/'
#dstpath='/data/workspace/speed-limit/Non-Negative/'
#cpath='/data/workspace/speed-limit/speedlimit.label'
#txtfile='/data/workspace/speed-limit-src/val.txt'
#jsonfile='/data/workspace/speed-limit-src/val.json'
#dist='/data/workspace/speed-limit-src/dist_val.txt'


## TODO:speed-limit-crop
srcpath='/data/workspace/speed-limit-crop/p38_5/'
dstpath=None
txtfile='/data/workspace/speed-limit-crop/test.txt'
cpath='/data/workspace/speed-limit/speedlimit.label'
jsonfile='/data/workspace/speed-limit-crop/test.json'
dist='/data/workspace/speed-limit-crop/dist_test.txt'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../tools/make_label.py $srcpath $dstpath $txtfile $cpath $jsonfile $dist