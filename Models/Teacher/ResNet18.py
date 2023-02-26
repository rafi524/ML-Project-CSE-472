from torch import nn
from torch.nn import functional as F
import torch
from Models.Teacher.Block1 import Block1
from Models.Teacher.Block2 import Block2

class ResNet18(nn.Module):
    def __init__(self, num_classes, block=Block1, num_blocks=[2,2,2,2]):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.avg_pool = nn.AvgPool2d((4,4), stride=(4,4))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # shape: [batch_size, 64, 32, 32]
        feature4 = self.avg_pool(out).view(out.size(0), -1)  # self.avg_pool(out): [batch_size, 64, 8, 8]
        # feature4: [batch_size, 4096]
        out = self.layer2(out) # shape: [batch_size, 128, 16, 16]
        feature3 = self.avg_pool(out).view(out.size(0), -1)
        # feature3: [batch_size, 2048]
        out = self.layer3(out) # shape: [batch_size, 256, 8, 8]
        feature2 = F.avg_pool2d(out, out.size(-1))  # [batch_size, channel_num]
        feature2 = torch.squeeze(feature2)
        # feature2: [batch_size, 1024]
        out = self.layer4(out) # shape: [batch_size, 512, 4, 4]
        #print(out.shape)
        feature1 = F.avg_pool2d(out, out.size(-1))  # average pooling to [batch_size, channel_num]
        feature1 = torch.squeeze(feature1)
        #print(feature1.shape)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        #feature1 = out    # [batch_size, 512]
        out = self.linear(out)
        # out = F.log_softmax(out, dim=1)

        return out #, feature1, feature2 #, feature3, feature4