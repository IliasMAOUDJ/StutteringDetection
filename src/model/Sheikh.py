import torch
from torch import nn
import speechbrain as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from transformers import AutoModelForAudioClassification,AutoFeatureExtractor

from speechbrain.lobes.models import huggingface_transformers
class Sheikh2022(nn.Module):
    def __init__(self):
        super(Sheikh2022, self).__init__()
        #self.wav2vec2 = huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
        #                                                                         freeze=False,
        #                                                                         freeze_feature_extractor=True,
        #                                                                         save_path="/hugging_face",
        #                                                                         output_all_hiddens=True,
        #                                                                         output_norm=True)
        self.wav2vec2 = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", local_files_only=True, output_hidden_states = True)
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base",local_files_only=True)
        
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=True)
        self.lda = LDA(n_components=1)
        self.bin_classifier = ClassificationLayer(in_dim=4608,h0=768,h1=256)

    def forward(self, x, labels):
        #x = self.wav2vec2(x)
        #x = x.permute(1,0,2,3)

         #x = x.permute(0,3,1,2)
        
        numpy = x.cpu().detach().numpy()
        features = self.feature_extractor(numpy,return_tensors='pt', sampling_rate=16000).to("cuda:0")
        x = self.wav2vec2(**features).hidden_states
        out = None
        for i in [0,6,10]:
            layer = self.pooling(x[i])
            x_mean, x_std = layer.split(768,dim=2)
            layer = torch.stack((x_mean, x_std), 3)
            #layer = layer.squeeze(-1)
            if out is None:
                out = layer
            else: 
                out = torch.cat([out,layer], dim=2)
        bin_out = self.bin_classifier(out)
        return bin_out

class ModifiedSheikh(nn.Module):
    def __init__(self):
        super(ModifiedSheikh, self).__init__()
        self.wav2vec2 = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", local_files_only=True, output_hidden_states = True)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base",local_files_only=True)
        #self.wav2vec2 = huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
        #                                                                            freeze=False,
        #                                                                            freeze_feature_extractor=True,
        #                                                                            save_path="/hugging_face",
        #                                                                            output_all_hiddens=True,
        #                                                                            output_norm=True)
        self.pooling = sb.nnet.pooling.StatisticsPooling()
        self.bin_classifier = ClassificationLayer(in_dim=512*149,h0=512,h1=64)
        self.rnn = sb.nnet.RNN.LSTM(512, (None,149,4*768), num_layers=1, dropout=0.4, bidirectional=False)
        

    def forward(self, x):
        numpy = x.cpu().detach().numpy()
        features = self.feature_extractor(numpy,return_tensors='pt', sampling_rate=16000).to("cuda:0")
        x = self.wav2vec2(**features).hidden_states
        out = None
        for i in [0,5,10,12]:
            layer = x[i]
            #layer = layer.permute(0,2,1)
            #layer = self.pooling(layer)
            #print(layer.shape)
            #x_mean, x_std = layer.split(149,dim=2)
            #layer = torch.cat((x_mean, x_std), 2)
            #print(layer.shape)
            if out is None:
                out = layer
            else: 
                out = torch.cat([out,layer], dim=2)
        out = self.rnn(out)[0]
        #out = torch.index_select(x,1,torch.tensor([0,5,10,12]).to("cuda:0"))
        bin_out = self.bin_classifier(out)
        return bin_out


class ClassificationLayer(nn.Module):
    def __init__(self, in_dim, h0,h1):
        super(ClassificationLayer, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.bn1 = sb.nnet.normalization.BatchNorm1d(input_size=h0)
        self.bn2 = sb.nnet.normalization.BatchNorm1d(input_size=h1)
        self.fc0 = nn.Linear(in_features= in_dim, out_features=h0, bias=True)
        self.fc1 = nn.Linear(in_features= h0, out_features=h1, bias=True)
        self.fc2 = nn.Linear(h1,1, bias=True)
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