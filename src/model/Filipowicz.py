
# Based on https://github.com/filipovvsky/stuttering_detection 
# and https://www.mdpi.com/2076-3417/13/10/6192

from transformers import AutoModelForAudioClassification

from speechbrain.lobes.models import huggingface_transformers
from model.models_utils import *
from torch import nn

class FilipowiczWav(nn.Module):
    def __init__(self) -> None:
        super(FilipowiczWav, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", 
                                                                    local_files_only=True, 
                                                                    output_hidden_states=True,
                                                                     num_labels=1)
        #self.wav2vec2 = huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
        #                                                            freeze=False,
        #                                                            freeze_feature_extractor=False,
        #                                                            save_path="/hugging_face")
        self.projector = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, 1)
    def forward(self, x):
        return self.model(x).logits
        #out = self.wav2vec2(x)
        #hidden_states = self.projector(out)
        #pooled_output = hidden_states.mean(dim=1)
        #logits = self.classifier(pooled_output)
        #return logits

class FilipowiczRes(nn.Module):
    def __init__(self):
        super(FilipowiczRes, self).__init__()
        self.resnet = ResNet18Arch(1, ResBlock, 256)
        self.binary_clf_hid = torch.nn.Linear(in_features=256, out_features=64)
        self.binary_clf = torch.nn.Linear(in_features=64,out_features=1)

    def forward(self, x):
        x = x[:, None, :, :]
        
        x = self.resnet(x)
        out = self.binary_clf_hid(x)
        out = self.binary_clf(out)
        return out