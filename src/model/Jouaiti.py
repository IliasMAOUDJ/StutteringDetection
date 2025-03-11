import torch
from torch import nn
import speechbrain as sb
from model.models_utils import TimeDistributed
import torchvision
import torchvision.models

class JouaitiEtAl(nn.Module):
    def __init__(self, in_dim, hidden_dim,num_output_channels):
        super(JouaitiEtAl, self).__init__()
        self.lstm = sb.nnet.RNN.LSTM(input_size = in_dim, hidden_size = hidden_dim, bidirectional = True, num_layers = 2)
        self.cyan = nn.Sequential(TimeDistributed(nn.Linear(num_output_channels, num_output_channels//2)),
                                  nn.ReLU())
        num_output_channels = num_output_channels//2*47
        self.red = nn.Sequential(nn.BatchNorm1d(num_output_channels),
                                nn.Dropout(0.5))
        self.purple = nn.Sequential(nn.Linear(num_output_channels, num_output_channels//2),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_output_channels//2),
                                    nn.Dropout(0.5))
        self.bin_purple = nn.Sequential(nn.Linear(num_output_channels//2, num_output_channels//4),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_output_channels//4),
                                    nn.Dropout(0.5))
        self.bin_blue = nn.Linear(num_output_channels//4, 1, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.cyan(out)
        out = torch.flatten(out, 1)
        out = self.red(out)
        out = self.purple(out)
        bin_out = self.bin_purple(out)
        bin_out = self.bin_blue(bin_out)
        return bin_out


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc= nn.Sequential(
                            nn.Dropout(0.8, inplace=True),
                            nn.Linear(2048, 1)
                        )
        self.resize = torchvision.transforms.Resize((224,224))

    def forward(self,x):
        x = self.resize(x)
        x = torch.stack((x,x,x), dim=1)
        out = self.resnet(x)
        return out
