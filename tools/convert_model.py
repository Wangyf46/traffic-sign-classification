import sys
sys.path.append('./')

import pdb
import argparse
import numpy as np

import torch.onnx
import onnx
import onnx_caffe2.backend

from src.LeNet5 import LeNet5


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch to ONNX to Caffe2')
    parser.add_argument('--classes', type=int, help='samples classes')
    parser.add_argument('model_name', type=str, help='model name')
    parser.add_argument('PyModel', type=str, help='pytorch model path')
    parser.add_argument('--softmax', default=False, help='the softmax type of model')
    parser.add_argument('--dropout', default=False, help='the dropout type of model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input = torch.rand(1, 1, 32, 32)
    model = LeNet5(1, args.classes, args.softmax, args.dropout)

    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(args.PyModel))
    # output = model(input)

    model.load_state_dict({k.replace('module.', ''): v for k,v in torch.load(args.PyModel).items()})
    output = model(input)

    # input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(25)]   ## TODO
    # output_names = ["output1"]

    ## '.onnx' and '.onnx.pd' and '.proto'
    model_name = args.model_name
    torch.onnx.export(model, input, model_name + '.onnx', verbose=True)
                      # input_names=input_names, output_names=output_names)

    ## TODO: test onnx file
    onnx_model = onnx.load(model_name + '.onnx')
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)

    # ## TODO: onnx to caffe2
    init_net, pred_net = onnx_caffe2.backend.Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph, device='CPU')
    with open(model_name + '_init.pd', 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(model_name + '_pred.pd', 'wb') as f:
        f.write(pred_net.SerializeToString())

    # rep = onnx_caffe2.backend.prepare(onnx_model, device='')
    # result = rep.run(np.random.randn(1,1,32,32).astype(np.float32))
    # print(result)






