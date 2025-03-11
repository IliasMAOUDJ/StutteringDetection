from torch import nn
import torch
from speechbrain.lobes.models import huggingface_transformers
from transformers import AutoModelForAudioClassification,AutoFeatureExtractor
"""
This implementation is based on Bayerl, Sebastian Peter, et al. “What Can 
Speech and Language Tell Us About the Working Alliance in Psychotherapy.” 
Interspeech 2022, ISCA, 2022. Crossref, https://doi.org/10.21437/interspeech.2022-347.
"""
class Bayerl(nn.Module):
    def __init__(self, stop_layer=12):
        super(Bayerl, self).__init__()
        #self.wav2vec2 = huggingface_transformers.wav2vec2.Wav2Vec2("facebook/wav2vec2-base-960h",
        #                                                                         freeze=False,
        #                                                                         freeze_feature_extractor=True,
        #                                                                         save_path="/hugging_face",
        #                                                                         output_all_hiddens=True)
        
        self.wav2vec2 = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", local_files_only=True, output_hidden_states = True)
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base",local_files_only=True)
        """The classiﬁcation head on top of the W2V2 model is equivalent to the implementation from the Transformers library
        https://github.com/huggingface/transformers/blob/04976a32dc555667afa994e8f918cbee88d84a4f/src/transformers/models/wav2vec2/modeling_wav2vec2.py 
         """
        self.projector = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, 1)
        self.stop_layer = stop_layer
    
    def forward(self, x):
        numpy = x.cpu().detach().numpy()
        features = self.feature_extractor(numpy,return_tensors='pt', sampling_rate=16000).to("cuda:0")
        x = self.wav2vec2(**features).hidden_states
        
        out = x[self.stop_layer]
        hidden_states = self.projector(out)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits
