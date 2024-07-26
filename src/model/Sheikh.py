import torch
from torch import nn
import speechbrain as sb

class Sheikh2022(nn.Module):
    def __init__(self, batch_size):
        super(Sheikh2022, self).__init__()
        self.wav2vec2 = sb.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
                                                                                 freeze=False,
                                                                                 freeze_feature_extractor=True,
                                                                                 save_path="./hugging_face",
                                                                                 output_all_hiddens=True,
                                                                                 output_norm=True)
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=True)
        self.bin_classifier = ClassificationLayer()

    def forward(self, x):
        x = self.wav2vec2(x)
        x = x.permute(1,2,3,0)
        x = self.pooling(x)
        x = x.permute(0,3,2,1)
        x_mean, x_std = x.split(768,dim=2)
        x = torch.stack((x_mean, x_std), 2)
        x = x.squeeze(-1)
        out = None
        out = torch.index_select(x,1,torch.tensor([0,6,10]).to("cuda:0"))
        bin_out = self.bin_classifier(out)
        return bin_out


class ClassificationLayer(nn.Module):
    def __init__(self):
        super(ClassificationLayer, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = sb.nnet.normalization.BatchNorm1d(input_size=1024)
        self.bn2 = sb.nnet.normalization.BatchNorm1d(input_size=256)
        self.fc0 = nn.Linear(in_features= 3*2*768, out_features=1024, bias=True)
        self.fc1 = nn.Linear(in_features= 1024, out_features=256, bias=True)
        self.fc2 = nn.Linear(256,1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x= torch.flatten(x,1)
        out = self.dropout(x)
        out = self.fc0(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.fc2(out)
        return out