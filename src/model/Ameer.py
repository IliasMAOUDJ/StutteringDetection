import torch
from torch import nn
import speechbrain as sb

from transformers import AutoFeatureExtractor, WhisperModel

from transformers import AutoModelForAudioClassification
from speechbrain.lobes.models import huggingface_transformers
"""

"""
class Ameer(nn.Module):
    def __init__(self):
        super(Ameer, self).__init__()
        self.whisper = AutoModelForAudioClassification.from_pretrained("openai/whisper-base", local_files_only=True, num_labels=1)
        #self.whisper = huggingface_transformers.whisper.Whisper("openai/whisper-base.en",
        #                                                                         encoder_only=True,
        #                                                                         freeze=False,
        #                                                                         freeze_encoder=False,
        #                                                                         save_path="/hugging_face")

        #self.projector = nn.Linear(512, 256)
        #self.classifier = nn.Linear(256, 1)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base",local_files_only=True)

    def forward(self, x): 
        #out = self.whisper(x)
        #hidden_states = self.projector(out)
        #pooled_output = hidden_states.mean(dim=1)
        #logits = self.classifier(pooled_output)
        #return logits
        numpy = x.cpu().detach().numpy()
        features = self.feature_extractor(numpy,return_tensors='pt', sampling_rate=16000).to("cuda:0")
        return self.whisper(features.input_features).logits