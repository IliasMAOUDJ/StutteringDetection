import torch
import torch.nn as nn    


class ResNet18Arch(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000, dropout=0.1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(resblock(64, 64, downsample=False),resblock(64, 64, downsample=False))
        self.layer2 = nn.Sequential(resblock(64, 128, downsample=True),resblock(128, 128, downsample=False))
        self.layer3 = nn.Sequential(resblock(128, 256, downsample=True), resblock(256, 256, downsample=False))
        self.layer4 = nn.Sequential(resblock(256, 512, downsample=True), resblock(512, 512, downsample=False))
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        #self.dropout = torch.nn.Dropout(dropout)
        #self.bilstm = torch.nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        #input = self.dropout(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        #input, _ = self.bilstm(input)
        input = self.fc(input)
        return input
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, downsample_stride=(2, 2)):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=downsample_stride, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=downsample_stride),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
##########################################################################################""""

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, embed_dim=768):
        super().__init__()

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
#####################################################################################################
import torchvision.transforms
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class TimeDistributed(nn.Module):
    """
    TimeDistributed for Pytorch which allows to apply a layer to every temporal slice of an input
    Args:
        Module: a Module instance
    PS : Input must be in the shape of (Seq_length, BS, )
    """

    def __init__(self, module, batch_first=False):
        if not isinstance(module, nn.Module):
            raise ValueError(
                "Please initialize `TimeDistributed` with a "
                f"`torch.nn.Module` instance. Received: {module.type}"
            )
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        orig_x = x
        if isinstance(x, PackedSequence):
            x, lens_x = pad_packed_sequence(x, batch_first=self.batch_first)

        if self.batch_first:
            # BS, Seq_length, * -> Seq_length, BS, *
            x = x.transpose(0, 1)
        output = torch.stack([self.module(xt) for xt in x], dim=0)
        if self.batch_first:
            # Seq_length, BS, * -> BS, Seq_length, *
            output = output.transpose(0, 1)

        if isinstance(orig_x, PackedSequence):
            output = pack_padded_sequence(output, lens_x, batch_first=self.batch_first)
        return output
