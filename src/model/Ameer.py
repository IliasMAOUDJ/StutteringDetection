import torch
from torch import nn
import speechbrain as sb

"""

"""
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