import random
import torchaudio
import speechbrain as sb
import torch 
from speechbrain.dataio.batch import PaddedBatch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

SAMPLE_RATE = 16000
def segmenting_collate_fn(examples, segment_length, num_class):
    # Iterate segments
    segments = []
    for example in examples:
        # Iterate segments
        length = len(example["waveform"]) / SAMPLE_RATE
        max_index = int(max(length // segment_length, 0)) +1 
        my_dict= defaultdict(list)
        for i in range(len(example["stutter_list"])):
            if(example["stutter_list"][i]!="."):
                my_dict[example["stutter_list"][i]].append([example["splits"][i], example["splits"][i+1]])
        
        for seg_index in range(max_index):
            segment = compute_segment(example, length, my_dict, seg_index, segment_length, num_class)
            if(segment is not None):
                segments.append(segment)
    return PaddedBatch(segments)


def compute_segment(example, length, my_dict, seg_index, segment_length, num_class):
    seg_start = seg_index * segment_length
    seg_end = min(seg_start + segment_length, length)
    #stutter_dict = {
    #    ".":0,
    #    "sound_rep":1, 
    #    "word_rep":2, 
    #    "phrase_rep":3, 
    #    "prolongation":4
    #}
    stutter_dict = {
        ".":0, "repetition":1, "blocage":2, "circumlocution":3}
    stutters =[]
    partial_stutters= []
    for el in my_dict:
        for timestamp in my_dict[el]:
            t0 = timestamp[0]
            t1 = timestamp[1]
            if(t0 >= seg_start and t1 <= seg_end):
                stutters.append((stutter_dict[el], t0, t1))
            elif(t0 <= seg_start and t1 > seg_start and t1 <= seg_end) \
            or (t0 >= seg_start and t0 < seg_end and t1 >= seg_end):
                partial_stutters.append((stutter_dict[el], t0, t1))
            elif(t0 <= seg_start and t1 >= seg_end):
                return None
    #if we have only one partial stutters and nothing else, rearrange start and end accordingly
    if(len(partial_stutters)==1 and len(stutters)==0): 
        _, t0, t1 = partial_stutters[0]
        if(t0<seg_start):
            new_seg_start = t0
        else:
            new_seg_start = t1-segment_length     
        waveform = example["waveform"][
                    int(new_seg_start * SAMPLE_RATE):int((new_seg_start+segment_length) * SAMPLE_RATE)]
    #if we have no partial stutters then take whole segment
    elif(len(partial_stutters)==0):
        waveform = example["waveform"][
                    int(seg_start * SAMPLE_RATE):int((seg_start+segment_length) * SAMPLE_RATE)]
    #if we have more than 1 partial --> should be divided in multiple segments but here we keep only first
    elif(len(partial_stutters)>1 and len(stutters)==0):
        #TODO
        _, t0, t1 = partial_stutters[0]
        _, t2, t3 = partial_stutters[1]
        if(t0<t2):
            partial_stutters.pop(-1)
            waveform = example["waveform"][
                        int(t0 * SAMPLE_RATE):int((t0+segment_length) * SAMPLE_RATE)]
        else:
            partial_stutters.pop(0)
            waveform = example["waveform"][
                        int(t2 * SAMPLE_RATE):int((t2+segment_length) * SAMPLE_RATE)]   
    #if we have both partial and stutter
    else:
        for i in stutters:
            _, t0, t1 = i
            for o in partial_stutters:
                _, t2, t3 = o
                if(t0<t2):
                    s = t0
                    e = t3
                else:
                    s = t2
                    e = t1
                if(e-s<segment_length):
                    waveform = example["waveform"][
                                int(s * SAMPLE_RATE):int((s+segment_length) * SAMPLE_RATE)]
                else:
                    waveform = example["waveform"][
                                int(s * SAMPLE_RATE):int((s+segment_length) * SAMPLE_RATE)]
                
    stutter_list = []
    for el in stutters:
        s, _, _ = el
        stutter_list.append(s)
    for el in partial_stutters:
        ps, _, _ = el
        stutter_list.append(ps)
    stutters_tensor = torch.LongTensor(stutter_list)
    if(waveform.shape[0]<segment_length*SAMPLE_RATE):
        waveform = torch.from_numpy(np.pad(waveform, (0,segment_length*SAMPLE_RATE-len(waveform)), 'constant'))

    label = torch.zeros(num_class)
    if(num_class>1):
        if(num_class == 3): #Put repetition as same label
            if(stutters_tensor.any()):
                for v in stutters_tensor[stutters_tensor.nonzero()]:
                    if(v==1 or v==2 or v==3):
                        label[1]=1
                    elif(v==4):
                        label[2] = 1
                    else:
                        label[0] = 1
        elif(num_class==4): #does not take fluent into account
            if(stutters_tensor.any()):
                for v in stutters_tensor[stutters_tensor.nonzero()]:
                    label[v-1] = 1              
        else: #fluent as a label
            if(stutters_tensor.any()): 
                for v in stutters_tensor[stutters_tensor.nonzero()]:
                    label[v] = 1
            else:
                label[0] = 1
    else:
        if(stutters_tensor.any()):
            label[0] = 1
    
    return {
            "id": example["id"] + f"_{seg_index}",
            "label": label.flatten(),
            #"timestamps": timestamps_tensor.flatten(),
            "waveform": waveform
        }

from torchaudio import transforms
def audio_pipeline(wav):
    waveform, sr = torchaudio.load(wav, normalize=True)
    transform = transforms.Resample(sr, 16000)
    waveform = transform(waveform)
    #waveform = waveform.transpose(0, 1)
    return waveform.squeeze(0)

def get_label(contain):
    return contain
def get_speaker_id(spk_id):
        return spk_id

def split_pipeline(breaks):
    splits = [float(f) for f in breaks.strip().split()]
    return torch.FloatTensor(splits)
def stutter_pipeline(stutter_type):
    stutter_list = stutter_type.strip().split()
    return stutter_list


def dataio_prep(hparams):
    "Prepare datasets and data pipelines"
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("waveform")
    def audio_pipeline(wav):
        waveform, sr = torchaudio.load(wav, normalize=True)
        transform = transforms.Resample(sr, 16000)
        waveform = transform(waveform)
        return waveform.squeeze(0)

    @sb.utils.data_pipeline.takes("speaker")
    @sb.utils.data_pipeline.provides("spk_id", "pairs")
    def get_speaker_id(spk_id):
        pairs = []
        if(hparams["siamese"]):
            if dataset=="valid":
                CURRENT_DATA=VALID_PAIR_DATA
            elif dataset=="test":
                CURRENT_DATA=TEST_PAIR_DATA
            else:
                CURRENT_DATA=TRAIN_PAIR_DATA
            if(not hparams["population"]):
                for data in CURRENT_DATA:
                    if(CURRENT_DATA[data]["speaker"]==spk_id):
                        pairs.append(CURRENT_DATA[data])
            else:
                #choices = np.random.choice(CURRENT_DATA,5)
                for i in range(10):
                    choice, _ = random.choice(list(CURRENT_DATA.items()))
                    pairs.append(CURRENT_DATA[choice])

        yield spk_id
        yield pairs

    @sb.utils.data_pipeline.takes("contain_stutter")
    @sb.utils.data_pipeline.provides("label", "disfluency")
    def get_label(contain):
        return contain, contain

    @sb.utils.data_pipeline.takes("breaks")
    @sb.utils.data_pipeline.provides("splits")
    def split_pipeline(breaks):
        splits = [float(f) for f in breaks.strip().split()]
        return torch.FloatTensor(splits)
    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("stutter_list", "origin")
    def stutter_pipeline(stutter_type):
        stutter_list = stutter_type.strip().split()
        return stutter_list, "synthetic"

    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=f"/data/csv/{dataset}_syn.json",
            dynamic_items=[
                get_label, audio_pipeline, split_pipeline, stutter_pipeline
            ],
            output_keys=[
                "id", "waveform", "label", "disfluency", "splits", "stutter_list", "origin"
            ],
        )
    hparams["dataloader_opts"]["collate_fn"] = lambda batch: segmenting_collate_fn(batch, segment_length=3, 
                                                                                    num_class=hparams["num_class"])
    return datasets

def plot_fbank(writer, spec,i, label, epoch, title=None, ylabel="freq_bin", aspect="auto", xmax=None, filename = None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((spec.cpu()), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    #if(filename is not None):
    #plt.savefig(f"./datavis/{i}_{label}.jpg")
    writer.add_figure(f"Image/{i}_{label}", fig, epoch)
