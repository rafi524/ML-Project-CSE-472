from torch import nn
from torch.nn.intrinsic import qat
from torch import quantization

from .VGGExtractor import VGG_FeatureExtractor
from .VGGExtractorQuantized import VGGQuantized_FeatureExtractor
from .BiLSTM import BidirectionalLSTM

class CRNN(nn.Module):
    def __init__(self, extractor_type, use_attention, nclass, hidden_size=256, qatconfig=None, bias=True):
        super(CRNN, self).__init__()
        
        if extractor_type == 'VGG':
            if qatconfig is None:
                self.backbone = VGG_FeatureExtractor()
            else:
                self.backbone = VGGQuantized_FeatureExtractor(qatconfig=qatconfig)
                self.backbone = quantization.prepare_qat(self.backbone)
        elif extractor_type == 'ResNet18':
            pass
        else:
            raise ValueError('Extractor type not supported')
        
        self.rnn = BidirectionalLSTM(512, hidden_size)
        self.embedding = nn.Linear(hidden_size * 2, nclass, bias=bias)
        
        if qatconfig is None:
            self._initialize_weights(bias)
        else:
            self._initialize_weights_qat()

        self.use_attention = use_attention
        if self.use_attention:
            self.multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        x = x.float()
        x = self.backbone(x)  
        
        if self.use_attention:
            x_original = x.clone()
            
            x = x.squeeze(2)
            x = x.permute(0, 2, 1)
            x = self.layer_norm(x)
            x, _ = self.multihead_attention(x, x, x)
            x = x.permute(0, 2, 1).unsqueeze(2)

            x = x_original + x

        x = self.rnn(x)
        
        T, b, h = x.size()
        x = x.view(T * b, h)
        x = self.embedding(x)  
        x = x.view(T, b, -1)
        return x
    
    def _initialize_weights(self, bias=True):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if name!="embedding" or (name=="embedding" and bias):
                    nn.init.zeros_(m.bias)
                
    def _initialize_weights_qat(self):
        for m in self.modules():
            if isinstance(m, (qat.ConvReLU2d, qat.ConvBnReLU2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)