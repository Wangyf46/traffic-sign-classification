import torch
import torch.nn as nn
import pdb



## TODO: dropout-11.5

class LeNet5_src(nn.Module):
    def __init__(self, in_channels, out_channels, softmax=False, dropout=False):
        super(LeNet5_src, self).__init__()
        self.block1 = conv(in_channels, 6, 2)
        self.block2 = conv(6, 16, 2)

        self.fc1 = nn.Sequential(nn.Linear(400, 120),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                 nn.ReLU())
        if softmax == True:
            self.fc3 = nn.Sequential(nn.Linear(84, out_channels),
                                     nn.Softmax(dim=1))
        else:
            self.fc3 = nn.Linear(84, out_channels)

        ## Vavier initialized method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, input):
        block1_out = self.block1(input)
        block2_out = self.block2(block1_out)

        flattened = block2_out.view(block2_out.size(0), -1)

        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)

        return fc3_out


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize):
        super(conv, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.AvgPool2d(kernel_size=ksize, stride=ksize, padding=0))

    def forward(self, input):
        out = self.layer(input)
        return out


if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32).cuda()
    # input = torch.ones(1, 1, 32, 32).cuda()
    speedlimit = LeNet5_src(1, 18).cuda()
    speedlimit.eval()
    output = speedlimit(input)
    print(output)
