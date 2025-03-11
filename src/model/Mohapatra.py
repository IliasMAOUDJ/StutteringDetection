import speechbrain as sb
import torch
from torch import nn
import torchaudio

from speechbrain.lobes.models import huggingface_transformers
# Based on https://github.com/payalmohapatra/Speech-Disfluency-Detection-with-Contextual-Representation-and-Data-Distillation 
# and https://dl.acm.org/doi/abs/10.1145/3539490.3539601
class Mohapatra(nn.Module):
    def __init__(self):
        super(Mohapatra, self).__init__()
        # in_channels is batch size
        #self.wav2vec2 = huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
        #                                                                         freeze=True,
        #                                                                         freeze_feature_extractor=True,
        #                                                                         save_path="/hugging_face")
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model().to("cuda:0")
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False
        
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer1_bn = nn.BatchNorm2d(8)
        # input size = (batch_size, 8, 74, 384)
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        size =500
        size2=250
        self.layer2_bn = nn.BatchNorm2d(16)
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(16* 37* 192,size, bias=True)
        self.fc1_bn = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size,size2, bias=True)
        self.fc2_bn = nn.BatchNorm1d(size2)
        self.fc3 = nn.Linear(size2,100, bias=True)
        self.fc3_bn = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100,10, bias=True)
        self.fc4_bn = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10,1, bias=True)

        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x,_ = self.wav2vec2(x)
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out  = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)
        #out = self.sm(out)
        return out
