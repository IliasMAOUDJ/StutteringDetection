from torch import nn
import speechbrain as sb

"""
This implementation is based on Bayerl, Sebastian Peter, et al. “What Can 
Speech and Language Tell Us About the Working Alliance in Psychotherapy.” 
Interspeech 2022, ISCA, 2022. Crossref, https://doi.org/10.21437/interspeech.2022-347.
"""
class Bayerl(nn.Module):
    def __init__(self, stop_layer=12, freeze = False, freeze_ft_ex=True):
        super(Bayerl, self).__init__()
        self.wav2vec2 = sb.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
                                                                                 freeze=False,
                                                                                 freeze_feature_extractor=True,
                                                                                 save_path="./hugging_face",
                                                                                 output_all_hiddens=True,
                                                                                 output_norm=True)
        
        self.fc1 = nn.Sequential(nn.Linear(in_features= 768,out_features=256, bias=True),
                                 nn.LeakyReLU())
                                 
        self.pooling = sb.nnet.pooling.StatisticsPooling(return_mean=True, return_std=False)
        self.cls_head = nn.Linear(in_features=256, out_features=1, bias=False)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.stop_layer = stop_layer
    
    def forward(self, x):
        x = self.wav2vec2(x)
        x = x.permute(1,0,2,3)
        out = x[:,self.stop_layer]
        out = self.fc1(out)
        bin_out = self.fc2(out)
        return bin_out