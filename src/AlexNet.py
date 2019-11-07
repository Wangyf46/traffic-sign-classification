import torch
import torch.nn as nn
import pdb


## TODO-11.6: dropout
class AlexNet(nn.Module):
    def __init__(self, in_channels, out_channels, softmax=False, dropout=False):
        super(AlexNet, self).__init__()
        self.flag = dropout

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2))
        if self.flag:
            self.dropout1 = nn.Dropout(0.25)

        self.layer2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(192),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=3, stride=2))
        if self.flag:
            self.dropout2 = nn.Dropout(0.25)

        self.layer3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(384),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2))
        if self.flag:
            self.dropout3 = nn.Dropout(0.25)

        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, out_channels))

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
        pdb.set_trace()
        x1 = self.layer1(x)
        if self.flag:
            x1 = self.dropout1(x1)
        x2 = self.layer2(x1)
        if self.flag:
            x2 = self.dropout2(x2)
        x3 = self.layer3(x2)
        if self.flag:
            x3 = self.dropout3(x3)

        x3 = x3.view(x3.size(0), 256 * 6 * 6)
        x4 = self.classifier(x3)
        return x4


if __name__ == '__main__':
    input = torch.rand(2, 1, 227, 227).cuda()
    # input = torch.ones(1, 1, 32, 32).cuda()
    speedlimit = AlexNet(1, 18).cuda()
    speedlimit.eval()
    output = speedlimit(input)
    print(output)