
import torch
from torch import nn
import timm
from timm.models.layers.std_conv import StdConv2dSame
import numpy as np
import speechbrain as sb

import logging
logger = logging.getLogger(__name__)

class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork,self).__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward_one(self,x): 
        o, e = self.model(x)
        return o, e
    
    def forward(self,x1,x2):
        o1, e1= self.forward_one(x1)
        _, e2= self.forward_one(x2)
        return e1, e2, o1
#--------------------------------------------------------------------------------------------------------------------------
class MyECAPA(nn.Module):
    def __init__(self, ecapa):
        super().__init__()
        self.ecapa = ecapa

    def forward(self, x: torch.Tensor):
        out = self.ecapa(x)
        return out

    def __repr__(self):
        return "MyECAPA"


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
#----------------------------------------------------------------------------------------------------------
import torchaudio.pipelines
class MyWav2vec2(nn.Module):
    def __init__(self,wav2vec2,layers,with_pooling, pool_time, output_all_hiddens, mean, std, dropout, num_class):
        super().__init__()
        self.with_pooling = with_pooling
        self.layers = list(map(int, layers.split(" ")))
        self.output_all_hiddens = output_all_hiddens
        if(self.with_pooling):

            self.wav2vec2 = wav2vec2
            self.wav2vec2.output_all_hiddens=True
            self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=mean, return_std=std)
            if(mean and std):
                multiplier =2
            else:
                multiplier=1
            self.pool_time = pool_time
            if(pool_time):
                embedding_dim = 768
            else:
                embedding_dim = 149
            self.bin_classifier = ClassificationLayer(embedding_dim*len(self.layers)*multiplier,num_class=1, dropout=dropout)
            self.multi_classifier = ClassificationLayer(embedding_dim*len(self.layers)*multiplier,num_class=num_class, dropout=dropout)
        else:
            if(not self.output_all_hiddens):
                bundle = torchaudio.pipelines.WAV2VEC2_BASE
                self.wav2vec2 = bundle.get_model()
                for name, param in self.wav2vec2.named_parameters():
                    param.requires_grad = False
                self.classifier = ConvLayer(16* 37* 192, num_class)
                #self.classifier = timm.create_model("resnet18", pretrained=True, in_chans=1, num_classes=2)
            else:
                self.wav2vec2 = wav2vec2
                self.classifier= nn.ModuleList()
                for i in range(len(self.layers)):
                    self.classifier.append(ConvLayer(16* 37* 192, num_class))
                self.agg = nn.Linear(len(self.layers)*2,2, bias=True)
        #print("****************************** My Architecture **********************")
        #print(self)
        #print("*********************************************************************")
    def forward(self, x: torch.Tensor):
        x = self.wav2vec2(x)
        if(self.with_pooling):
            return self.forward_with_pooling(x)
        else:
            if(self.output_all_hiddens):
                return self.forward_all_hiddens(x)
            else:
                return self.forward_last(x)

    def forward_with_pooling(self, x: torch.Tensor):
        if(self.pool_time):
            x = x.permute(1,0,2,3)
        else:
            x = x.permute(1,0,3,2)
        out = None
        for i in range(x.shape[1]):
            if(i in self.layers):
                o = x[:,i]
                layer = self.pooling(o)
                if(out is None):
                    out = layer
                    feats = o
                else:
                    out = torch.cat([out, layer], dim=-1)

                    feats = torch.cat([feats, o], dim=-1)
 
        final_bin_out = self.bin_classifier(out)
        final_multi_out = self.multi_classifier(out)
        return final_bin_out.squeeze(1), final_multi_out.squeeze(1), out, feats

    def forward_all_hiddens(self, x: torch.Tensor):
        out = None
        l=0
        for i in range(x.shape[0]):
            #layer = self.pooling(x[i])
            if(i in self.layers):
                layer = self.classifier[l](x[i])
                l+=1
                if(out is None):
                    out = layer
                else:
                    out = torch.cat([out, layer], dim=-1)
        final_out = self.agg(out)
        return final_out, out, x

    def forward_last(self, x: torch.Tensor):
        out_0, _ = x
        out= self.classifier(out_0)
        return out, out_0, _

class ConvLayer(nn.Module):
    def __init__(self, in_features, num_class):
        super(ConvLayer, self).__init__()
        #if(out_features==2):
        #    out_features=1

        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5)
        )
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=2),
            torch.nn.Dropout(p=0.5)
        )

        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(in_features= in_features, out_features=500, bias=False)
        self.fc2 = nn.Linear(500,250, bias=True)
        self.fc3 = nn.Linear(250,100, bias=True)
        self.fc4 = nn.Linear(100,10, bias=True)
        self.fc5 = nn.Linear(10,num_class, bias=True)

        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x=x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
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
        return out

import torchvision.transforms
class PretrainedSEResNet(nn.Module):
    def __init__(self, seresnet = 'seresnet50', dropout=0.0, pretrained=True, patch_size=16, img_size=(128,301), num_class=1, layers_to_freeze=0):
        super().__init__()
        multiplier = 1
        classes = num_class-1

        if "50" in seresnet or "101" in seresnet:
            multiplier = 4
        if "vit_base_resnet50" in seresnet:
            self.model = timm.create_model(seresnet, pretrained=pretrained, in_chans=1,patch_size=patch_size, img_size=img_size, num_classes=classes)
            self.model.patch_embed.backbone.stem.conv = StdConv2dSame(1, 64, kernel_size=7, stride=2, padding=3,bias=False)        
            self.model.head = Identity()
            self.fc = nn.Linear(in_features=768, out_features=classes, bias=True)
        elif "vit_base_patch" in seresnet:
            self.model = timm.create_model(seresnet, pretrained=pretrained, in_chans=1, img_size=img_size, num_classes=classes)
            self.model.head = Identity()
            self.fc = nn.Linear(in_features=768, out_features=classes, bias=True)
        else:
            self.model = timm.create_model(seresnet, pretrained=pretrained, in_chans=1, num_classes=classes)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.model.fc = Identity()
        
        if (pretrained):
            ct = 0
            for name, param in self.model.named_parameters():
                ct += 1
                if ct < layers_to_freeze:
                    param.requires_grad = False
                    #print(name, param.requires_grad)
                    #for name, param in child.named_parameters():
                        
            self.fc = nn.Linear(in_features=512*multiplier, out_features=classes, bias=True)   
        self.dropout = nn.Dropout(dropout)
        #print("****************************** My Architecture **********************")
        #print(self.model)
    def forward(self, x: torch.Tensor):
        
        x=x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
        out = self.model(x)
        embedding = out
        out = self.dropout(out)
        #out = self.fc(out)
        return out, embedding
    
    def __repr__(self):
        return "PretrainedSEResnet"

class ResNetBiLSTM(nn.Module):
    def __init__(self, resnet, bilstm, num_classes, pretrain, freeze, fc_dim, layers_to_freeze=0):
        super(ResNetBiLSTM, self).__init__()
        self.bilstm = bilstm
        if("ResArch" in resnet):
            self.model = ResNet18Arch(in_channels=1,resblock=ResBlock,outputs=fc_dim)
        else:
            self.model = timm.create_model(resnet, pretrained=pretrain, in_chans=1, num_classes=num_classes)
        
        if (freeze):
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < layers_to_freeze:
                    for name, param in child.named_parameters():
                        param.requires_grad = False
                        print(name, param.requires_grad)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.model.fc = Identity()
        self.fc = nn.Linear(in_features=fc_dim, out_features=num_classes, bias=True)   
        #print("****************************** ResNet + BiLSTM **********************")
        #print(self.model)

    def forward(self,x):
        x=x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
        out = self.model(x)
        out = self.bilstm(out)
        #out = self.fc(out[0])
        return out[0]


class ClassificationLayer(nn.Module):
    def __init__(self, in_dim, num_class, dropout):
        super(ClassificationLayer, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_features= in_dim, out_features=500, bias=False)
        self.fc2 = nn.Linear(500,10, bias=True)
        #self.fc4 = nn.Linear(100,10, bias=True)
        self.fc5 = nn.Linear(10, num_class, bias=True)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc5(out)
        
        return out



class JouaitiEtAl(nn.Module):
    def __init__(self, in_dim, hidden_dim,num_output_channels, num_class):
        super(JouaitiEtAl, self).__init__()
        self.lstm = sb.nnet.RNN.LSTM(input_size = in_dim, hidden_size = hidden_dim, bidirectional = True, num_layers = 2)
        self.cyan = nn.ReLU()
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
        self.multi_purple = nn.Sequential(nn.Linear(num_output_channels//2, num_output_channels//4),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_output_channels//4),
                                    nn.Dropout(0.5))
        self.bin_blue = nn.Linear(num_output_channels//4, 1, bias=True)
        self.multi_blue = nn.Linear(num_output_channels//4, num_class, bias=True)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        out, _ = self.lstm(x)
        out = self.cyan(out)
        out = self.red(out)
        out = self.purple(out)
        embeddings = out

        multi_out = self.multi_purple(out)
        multi_out = self.multi_blue(multi_out)

        bin_out = self.bin_purple(out)
        bin_out = self.bin_blue(bin_out)
        return bin_out, multi_out, embeddings




class ResNet18Arch(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, x):
        input = self.layer0(x)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
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