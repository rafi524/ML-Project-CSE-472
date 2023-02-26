from torch import nn

class VGG_FeatureExtractor(nn.Module):
    """
        Expecting input of shape (batch, 1, 32, 128)
    """

    def __init__(self, in_channels=1, out_channels=512):
        super(VGG_FeatureExtractor, self).__init__()

        # [64, 128, 256, 512]
        out_channels = [int(out_channels / 8), int(out_channels / 4),
                               int(out_channels / 2), out_channels]  

        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], 3, 1, 1),                 # 64x32x128
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                               # 64x16x64
            nn.Conv2d(out_channels[0], out_channels[1], 3, 1, 1),             # 128x16x64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                               # 128x8x32
            nn.Conv2d(out_channels[1], out_channels[2], 3, 1, 1),             # 256x8x32
            nn.ReLU(True),                                          
            nn.Conv2d(out_channels[2], out_channels[2], 3, 1, 1),             # 256x8x32
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                     # 256x4x32
            nn.Conv2d(out_channels[2], out_channels[3], 3, 1, 1, bias=False), # 512x4x32
            nn.BatchNorm2d(out_channels[3]), 
            nn.ReLU(True),                                                      
            nn.Conv2d(out_channels[3], out_channels[3], 3, 1, 1, bias=False), # 512x4x32
            nn.BatchNorm2d(out_channels[3]), 
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                                     # 512x2x32
            nn.Conv2d(out_channels[3], out_channels[3], 2, 1, 0),             # 512x1x31
            nn.ReLU(True))  

    def forward(self, input):
        return self.ConvNet(input)