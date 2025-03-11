from torch import nn
import speechbrain as sb
import torch

class Lea(nn.Module):
    def __init__(self):
        super(Lea, self).__init__()
        self.lstm = sb.nnet.RNN.LSTM(input_size = 40, hidden_size = 64, bidirectional = False, num_layers = 1)
        self.clf = nn.Linear(in_features=301*64, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out= torch.flatten(out,1)
        out = self.clf(out)
        return out
