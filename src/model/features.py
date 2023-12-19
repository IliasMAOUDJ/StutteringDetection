
import wave
import torch
from torchaudio.transforms import Resample
import speechbrain.lobes.features
import torchaudio.compliance.kaldi
import librosa.feature

class FBank(torch.nn.Module):
    def __init__(
        self,
        input_freq,
        resample_freq,
        n_fft=2048,
        n_mels=256,
        n_mfcc=20,
        hop_length=10,
        win_length=25,
        num_channels=1,
        left_frames=0,
        right_frames=0,
        feats: str="mfcc"
    ):
        super().__init__()
        self.feats=feats
        self.num_channels = num_channels
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.fbank = speechbrain.lobes.features.Fbank(sample_rate=resample_freq,
                                                    n_mels=n_mels,
                                                    hop_length=hop_length,
                                                    win_length=win_length,
                                                    n_fft=n_fft, 
                                                    left_frames=left_frames, 
                                                    right_frames=right_frames)

        self.mfcc = speechbrain.lobes.features.MFCC(deltas=False,
                                                    sample_rate=resample_freq, 
                                                    n_mels=n_mels,
                                                    n_mfcc = n_mfcc,
                                                    context=False,
                                                    hop_length=hop_length,
                                                    win_length=win_length,
                                                    n_fft=n_fft, 
                                                    left_frames=left_frames, 
                                                    right_frames=right_frames)
 
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = None
        # Resample the input
        #waveforms = self.resample(waveforms)
        if(self.feats=="fbank"):
            features = self.fbank(waveforms)
        elif(self.feats=="mfcc"):
            #features = librosa.feature.mfcc(waveforms, 8000)
            features= self.mfcc(waveforms)
        #normalize
        height = features.shape[2]
        width = features.shape[1]
        features = features.view(features.size(0), -1)
        features -= features.min(1, keepdim=True)[0]
        features /= features.max(1, keepdim=True)[0]
        features = features.view(features.shape[0], width, height)
        
        return features
