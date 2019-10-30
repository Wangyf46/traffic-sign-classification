import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
### testtttttt
'''
    padding: 'SAME'(conv, pool)
    dropout: conv, fc1
    init: 
    softmax
    BN:
    LOSS
    l2 Re
    
'''

class LeNet5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet5, self).__init__()
        self.block1 = conv(in_channels, 32, 2, 0.9)

        self.block2 = conv(32, 64, 2, 0.8)
        self.block3 = conv(64, 128, 2, 0.7)

        # 1st stage output
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding='SAME')

        # 2st stage output
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding='SAME')

        self.fc1 = nn.Sequential(nn.Linear(3584, 1024),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5, training=self.training))

        self.fc2 = nn.Sequential(nn.Linear(1024, 18),
                                nn.BatchNorm2d(18),   ## TODO
                                nn.Softmax())

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



    def forward(self, *input):
        block1_out = self.block1(input)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)

        # 1st stage output
        stage1_out = self.pool1(block1_out)

        # 2st stage output
        stage2_out = self.pool2(block2_out)

        flattened = torch.cat([stage1_out, stage2_out, block3_out], dim=1)
        flattened = flattened.view(flattened.size(0), -1)

        fc1 = self.fc1(flattened)
        logits = self.fc2(fc1)

        return logits


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, p):
        super(conv, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=ksize, stride=ksize, padding='SAME'),
                                   nn.Dropout(p, training=self.training))

    def forward(self, *input):
        out = self.layer(input)

        return out
