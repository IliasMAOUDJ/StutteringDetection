
import torch
from torch import nn
import timm
import numpy as np
import speechbrain as sb
from noisereduce.torchgate import TorchGate as TG
import logging
logger = logging.getLogger(__name__)

from huggingface_hub.hf_api import HfFolder
#import speechbrain.lobes.models.huggingface_transformers.whisper
HfFolder.save_token("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")

class NoiseReduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.tg = TG(sr=16000, nonstationary=False).to("cuda:0")
        
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.tg(x)
        return out
#----------------------------------------------------------------------------------------------------------
import torchaudio.pipelines
class MyWav2vec2(nn.Module):
    def __init__(self,wav2vec2, source, layers, dropout, dropoutlstm, batch_size, hidden_size=256):
        super().__init__()
        self.layers = list(map(int, layers.split(" ")))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        if("large" in source):
            embedding_dim = 1024
        elif("base" in source):
            embedding_dim = 768
        
        self.wav2vec2 = wav2vec2
        self.wav2vec2.output_all_hiddens=True
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=True)
        self.rnn = sb.nnet.RNN.LSTM(hidden_size, (batch_size,149,2*embedding_dim), num_layers=2, dropout=dropoutlstm, bidirectional=True)
        self.bin_classifier = ClassificationLayer(hidden_size*149*2, dropout=dropout)
    def forward(self, x: torch.Tensor):
        x = self.wav2vec2(x)
        return self.forward_with_pooling(x)

    def forward_with_pooling(self, x: torch.Tensor):
        x = x.permute(1,0,2,3)
        o = self.pooling(x).squeeze(1)
        o_mean, o_std = o.split(149,dim=1)
        o = torch.stack((o_mean, o_std), 1)
        o = o.permute(0,2,1,3)
        o = o.reshape(o.shape[0], o.shape[1], o.shape[2]*o.shape[3])
        o = self.rnn(o)[0]
        out= o.reshape(o.shape[0], o.shape[1]* o.shape[2])
        final_bin_out = self.bin_classifier(out)
        return final_bin_out
    

class ClassificationLayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(ClassificationLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = sb.nnet.normalization.BatchNorm1d(input_size=in_dim)
        self.bn2 = sb.nnet.normalization.BatchNorm1d(input_size=1024)
        self.fc1 = nn.Linear(in_features= in_dim, out_features=1024, bias=False)
        self.fc2 = nn.Linear(1024,1, bias=False)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x= torch.flatten(x,1)
        out = self.bn1(x)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


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


class Lea(nn.Module):
    def __init__(self):
        super(Lea, self).__init__()
        self.lstm = sb.nnet.RNN.LSTM(input_size = 40, hidden_size = 64, bidirectional = False, num_layers = 1)
        self.clf = ClassificationLayer(301*64,0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.clf(out)
        return out


# Based on https://github.com/payalmohapatra/Speech-Disfluency-Detection-with-Contextual-Representation-and-Data-Distillation 
# and https://dl.acm.org/doi/abs/10.1145/3539490.3539601
class Mohapatra(nn.Module):
    def __init__(self, size, size2):
        super(Mohapatra, self).__init__()
        # in_channels is batch size
        self.wav2vec2 = sb.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
                                                                                 freeze=True,
                                                                                 freeze_feature_extractor=True,
                                                                                 save_path="./hugging_face")
        

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
        x = self.wav2vec2(x)
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

# Based on https://github.com/filipovvsky/stuttering_detection 
# and https://www.mdpi.com/2076-3417/13/10/6192

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from model.models_utils import *

class FilipowiczWav(nn.Module):
    def __init__(self) -> None:
        super(FilipowiczWav, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base",
                                                                     num_labels=1)
    def forward(self, x):
        return self.model(x).logits

class FilipowiczRes(nn.Module):
    def __init__(self, input_size, clf_input_size, hidden_size):
        super(FilipowiczRes, self).__init__()
        self.resnet = ResNet18Arch(input_size, ResBlock, 256)
        self.binary_clf_hid = torch.nn.Linear(in_features=clf_input_size, out_features=hidden_size)
        self.binary_clf = torch.nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.resnet(x)
        out = self.binary_clf_hid(x)
        out = self.binary_clf(out)
        return out


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
        self.bin_classifier = ClassificationLayer(3*2*768, dropout=0.4)

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
    
class Bayerl(nn.Module):
    def __init__(self, stop_layer=12, freeze = False, freeze_ft_ex=True):
        super(Bayerl, self).__init__()
        self.wav2vec2 = sb.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
                                                                                 freeze=freeze,
                                                                                 freeze_feature_extractor=freeze_ft_ex,
                                                                                 save_path="./hugging_face",
                                                                                 output_all_hiddens=True,
                                                                                 output_norm=True)
        
        self.fc1 = nn.Sequential(nn.Linear(in_features= 768,out_features=256, bias=True),
                                 nn.LeakyReLU())
                                 
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=False)
        self.cls_head = nn.Linear(in_features=256, out_features=1, bias=False)
        self.bin_classifier = ClassificationLayer(256, dropout=0.4)
        self.stop_layer = stop_layer
    
    def forward(self, x):
        x = self.wav2vec2(x)
        x = x.permute(1,0,2,3)
        out = x[:,self.stop_layer]
        out = self.fc1(out)
        out = self.pooling(out) #.permute(0,2,1)
        bin_out = self.bin_classifier(out)
        #bin_out = self.cls_head(out.squeeze())
        return bin_out

class Whisper(nn.Module):
    def __init__(self):
        super(Whisper, self).__init__()
        self.whisper = sb.lobes.models.huggingface_transformers.whisper.Whisper("openai/whisper-base.en",
                                                                                 encoder_only=True,
                                                                                 freeze=False,
                                                                                 freeze_encoder=True,
                                                                                 save_path="./hugging_face")
        self.fc1 = nn.Sequential(nn.Linear(in_features= 512,out_features=256, bias=True),
                                 nn.LeakyReLU())
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=False)
        self.bin_classifier = ClassificationLayer(256, dropout=0.4)

    def forward(self, x):
        x = self.whisper(x)
        out = self.fc1(x)
        out = self.pooling(out) #.permute(0,2,1)
        bin_out = self.bin_classifier(out)
        return bin_out

class Ameer(nn.Module):
    def __init__(self):
        super(Ameer, self).__init__()
        self.whisper = sb.lobes.models.huggingface_transformers.whisper.Whisper("openai/whisper-base.en",
                                                                                 encoder_only=True,
                                                                                 freeze=False,
                                                                                 freeze_encoder=False,
                                                                                 save_path="./hugging_face")
        self.fc1 = nn.Linear(in_features= 1500*512, out_features=1, bias=False)

    def forward(self, x):
        x = self.whisper(x)
        x = torch.flatten(x,1)
        out = self.fc1(x)
        return out


#doi.org/10.1007/978-3-031-48309-7_11
class Simha(nn.Module):
    def __init__(self,classifier="lstm"):
        super(Simha,self).__init__()  

        #Classifier is SVM, LSTM or Bi-LSTM
        if(classifier=="lstm"):
            self.classifier = sb.nnet.RNN.LSTM(512, (None,94,20), num_layers = 1, dropout=0.2)
            self.fc1 = nn.Linear(in_features= 512*94, out_features=1, bias=False)
        elif(classifier=="bilstm"):
            self.classifier = sb.nnet.RNN.LSTM(512, (None,94,20), num_layers = 2, bidirectional=True, dropout=0.2)
            self.fc1 = nn.Linear(in_features= 512*94*2, out_features=1, bias=False)
        else:
            return

    def forward(self,x):
        out = self.classifier(x)[0]
        out = torch.flatten(out,1)
        out = self.fc1(out)
        return out











import os
import wget
import timm
from timm.models.layers import to_2tuple,trunc_normal_
# override the timm package to relax the input shape constraint.
class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.deit.PatchEmbed = PatchEmbed
        timm.models.layers.patch_embed.PatchEmbed = PatchEmbed
        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.v.patch_embed = PatchEmbed((384,384), (16, 16), 1, 768)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim)
                new_pos_embed = new_pos_embed.transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('configs/pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='/configs/pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('/configs/pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))


    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1) #
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x