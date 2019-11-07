import torch
import torch.nn as nn
import pdb



## TODO: dropout with train or eval

class VggNet(nn.Module):
    def __init__(self, in_channels, out_channels, softmax=False, drop=False):
        super(VggNet, self).__init__()
        self.flag = drop

        self.layer1 = conv(in_channels, 32)

        self.layer2 = conv(32, 64)

        self.layer3 = conv(64, 128)

        self.fc1 = linear(4*4*128, 128)

        self.fc2 = linear(128, 128)

        self.fc3 = nn.Linear(128, out_channels)

        if self.flag:
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.5)
            self.dropout3 = nn.Dropout(p=0.5)
            self.dropout4 = nn.Dropout(p=0.5)
            self.dropout5 = nn.Dropout(p=0.5)

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


    def forward(self, x):
        x1 = self.layer1(x)
        if self.flag:
            x1 = self.dropout1(x1)

        x2 = self.layer2(x1)
        if self.flag:
            x2 = self.dropout2(x2)

        x3 = self.layer3(x2)
        if self.flag:
            x3 = self.dropout3(x3)

        x3 = x3.view(x3.size(0), -1)

        x4 = self.fc1(x3)
        if self.flag:
            x4 = self.dropout4(x4)

        x5 = self.fc2(x4)
        if self.flag:
            x5 = self.dropout5(x5)

        logits = self.fc3(x5)

        return logits



class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, input):
        out = self.layer(input)
        return out


class linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(linear, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, out_channels),
                                 nn.ReLU())

    def forward(self, input):
        out = self.fc(input)
        return out




if __name__ == '__main__':
    input = torch.rand(1, 1, 32, 32).cuda()
    # input = torch.ones(1, 1, 32, 32).cuda()
    speedlimit = VggNet(1, 18).cuda()
    speedlimit.eval()
    output = speedlimit(input)
    print(output)
