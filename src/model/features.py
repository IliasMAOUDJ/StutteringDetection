
import torch
import speechbrain.lobes.features
import torchaudio.compliance.kaldi
import librosa.feature
import torchaudio.transforms as T
from noisereduce.torchgate import TorchGate as TG

class FBank(torch.nn.Module):
    def __init__(
        self,
        n_fft=2048,
        n_mels=256,
        hop_length=10,
        win_length=25,
        left_frames=0,
        right_frames=0,
        resample_freq=16000,
        feats: str="mfcc"
    ):
        super().__init__()
        self.feats=feats
        self.fbank = speechbrain.lobes.features.Fbank(sample_rate=resample_freq,
                                                    n_mels=n_mels,
                                                    hop_length=hop_length,
                                                    win_length=win_length,
                                                    n_fft=n_fft, 
                                                    left_frames=left_frames, 
                                                    right_frames=right_frames)
        
        
        if resample_freq!=16000:
            self.resample_flag = True
            self.resample = speechbrain.augment.time_domain.Resample(orig_freq=16000, 
                                                                            new_freq=resample_freq,
                                                                            lowpass_filter_width=6)
        else:
            self.resample_flag = False
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = []
        # Resample the input
        if(self.resample_flag):
            waveforms = self.resample(waveforms).to("cuda:0")
        if(self.feats=="fbank"):
            features = self.fbank(waveforms)
        elif(self.feats=="mfcc"):
            for waveform in waveforms:
                cpu = waveform.unsqueeze(0).cpu().detach()
                feature = librosa.feature.mfcc(
                                y=cpu.numpy(),
                                n_mfcc=20,
                                dct_type=2,
                                norm="ortho",
                            )
                feature = torch.tensor(feature).to("cuda:0")
                features.append(feature)
            #features= self.mfcc(waveforms)
            features = torch.stack(features).to("cuda:0")
            features = features.squeeze(1)
            features = features.permute(0,2,1)
        #elif(self.feats=="wav2vec2"):
        #    features, _ = self.model_wav2vec(waveforms)
        return features

class NoiseReduction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tg = TG(sr=16000, nonstationary=False).to("cuda:0")
        
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.tg(x)
        return out

"""n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)

plot_spectrogram(mfcc[0], title="MFCC")

"""

"""melspec = librosa.feature.melspectrogram(
    y=SPEECH_WAVEFORM.numpy()[0],
    sr=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
    htk=True,
    norm=None,
)

mfcc_librosa = librosa.feature.mfcc(
    S=librosa.core.spectrum.power_to_db(melspec),
    n_mfcc=n_mfcc,
    dct_type=2,
    norm="ortho",
)

plot_spectrogram(mfcc_librosa, title="MFCC (librosa)")

mse = torch.square(mfcc - mfcc_librosa).mean().item()
print("Mean Square Difference: ", mse)



import scipy
def ZTW(signal):
    #w1-w2
    M = signal[-1]
    w1 = torch.zeros_like(signal)
    w1[0] = 0
    for i in range(len(w1-1)):
        w1[i+1] = 1 / (4*torch.sin(torch.pi*i/(2*len(w1))))
    
    convolve = torchaudio.transforms.FFTConvolve()
    x = convolve(signal,w1)
    #spectral estimation NGD
    scipy.signal.group_delay(x)

    #double differencied NGD

    #hilbert envelope
    return scipy.signal.hilbert(x)


def ZTWCC(signal):
    X = ZTW(signal)
    #log
    x= torch.log(X)
    #IFFT
    ztwcc = torch.fft.ifft(x)
    return ztwcc
"""