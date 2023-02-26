from torch import nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1, "the height of conv must be 1"

        x = x.squeeze(axis=2)
        x = x.permute(2, 0, 1)  # (width, batch, channels)
        x, _ = self.lstm(x)

        return x