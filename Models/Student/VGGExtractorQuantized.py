from torch import nn

from torch import quantization
from torch.nn.intrinsic import qat

class VGGQuantized_FeatureExtractor(nn.Module):
    """
        Expecting input of shape (batch, 1, 32, 128)
    """

    def __init__(self, input_channel=1, out_channels=512, qatconfig='fbgemm'):
        super(VGGQuantized_FeatureExtractor, self).__init__()

        # [64, 128, 256, 512]
        out_channels = [int(out_channels / 8), int(out_channels / 4),
                               int(out_channels / 2), out_channels]  

        qconfig = quantization.get_default_qat_qconfig(qatconfig)
        
        self.ConvNet = nn.Sequential(
            qat.ConvReLU2d(input_channel, out_channels[0], 3, 1, 1, qconfig=qconfig),                 # 64x32x128
            nn.MaxPool2d(2, 2),                                                                       # 64x16x64
            qat.ConvReLU2d(out_channels[0], out_channels[1], 3, 1, 1, qconfig=qconfig),               # 128x16x64
            nn.MaxPool2d(2, 2),                                                                       # 128x8x32
            qat.ConvReLU2d(out_channels[1], out_channels[2], 3, 1, 1, qconfig=qconfig),               # 256x8x32
            qat.ConvReLU2d(out_channels[2], out_channels[2], 3, 1, 1, qconfig=qconfig),               # 256x8x32
            nn.MaxPool2d((2, 1), (2, 1)),                                                             # 256x4x32
            qat.ConvBnReLU2d(out_channels[2], out_channels[3], 3, 1, 1, bias=False, qconfig=qconfig), # 512x4x32
            qat.ConvBnReLU2d(out_channels[3], out_channels[3], 3, 1, 1, bias=False, qconfig=qconfig), # 512x4x32
            nn.MaxPool2d((2, 1), (2, 1)),                                                             # 512x2x32
            qat.ConvReLU2d(out_channels[3], out_channels[3], 2, 1, 0, qconfig=qconfig))               # 512x1x31
        
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.ConvNet(x)
        x = self.dequant(x)
        return x 