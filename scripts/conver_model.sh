#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
proc_num=$(echo $CUDA_VISIBLE_DEVICES | awk '{len=split($0,arr,","); print len}')

## TODO-11.5: 15 + nonsoftmax + nondropout + LeNet5 (best)
#model_name='./models/LeNet5_15'
#classes=15
#PyModel='/data/workspace/speed-limit/workout/LeNet5/20191105115627_ubuntu/88_itr5251.pth'

### TODO-11.5: 18 + nonsoftmax + nondropout + LeNet5(best)
#model_name='./models/LeNet5_18'
#classes=18
#PyModel='/data/workspace/speed-limit/workout/LeNet5/20191105144413_ubuntu/99_itr7800.pth'


## TODO-11.18(mixed-1): 18 + nonsoftmax + nondropout + LeNet5(best)
model_name='/data/workspace/speed-limit-mixed-1/workout-aug/LeNet5/20191118163803_ubuntu/LeNet5_18'
classes=18
PyModel='/data/workspace/speed-limit-mixed-1/workout-aug/LeNet5/20191118163803_ubuntu/99_itr7500.pth'

script_path=$(cd "$(dirname "$0")"; pwd)
python $script_path/../tools/convert_model.py  $model_name --classes $classes $PyModel
mnnconvert -f ONNX --modelFile /data/workspace/speed-limit-mixed-1/workout-aug/LeNet5/20191118163803_ubuntu/LeNet5_18.onnx --MNNModel /data/workspace/speed-limit-mixed-1/workout-aug/LeNet5/20191118163803_ubuntu/LeNet5_18.mnn --bizCode MNN




## TODO: ONNX to Caffe2
# convert-onnx-to-caffe2 $script_path/../models/LeNet5.onnx --output pred_net.pb --init-net-output init_net.pb

