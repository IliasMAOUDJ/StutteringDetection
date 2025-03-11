from functools import partial
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import speechbrain as sb
from speechbrain.utils.epoch_loop import EpochCounterWithStopper
from hyperpyyaml import load_hyperpyyaml
import model.utils
import model.features
from model.Sheikh import Sheikh2022
import torchaudio
import numpy as np
from data_prep_utils import *
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token("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")
import random

class LatimBrain(sb.Brain):
    """Use attentional model to predict words in segments"""
    def compute_feats(self, wavs, lens, stage):
        if(self.hparams.nr):
            with torch.no_grad():
                wavs, lens = self.hparams.noisereducer(wavs)
        if stage == sb.Stage.TRAIN:
            if(self.hparams.augment): 
                wavs, lens = self.hparams.augmenter(wavs, lens)
        if(wavs.shape[1]>48000):
            wavs = wavs[:,:48000]
        elif(wavs.shape[1]<48000):
            pad = torch.zeros([wavs.shape[0], 48000-wavs.shape[1]]).to("cuda:0")
            wavs= torch.cat([wavs,pad], dim=1) 
        if not (self.hparams.raw_input) :
            with torch.no_grad():
                wavs = self.hparams.compute_feats(wavs)
        
        return wavs

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        labels = batch.label.data
        waveforms, lens = batch.waveform
        waveforms = self.compute_feats(waveforms, lens, stage)
        if(isinstance(self.modules.model, Sheikh2022)):
            bin_out = self.modules.model(waveforms, labels)
        else:
            bin_out = self.modules.model(waveforms)
        return {"bin_pred" : bin_out}
               

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label.data
        #if(stage==sb.Stage.VALID):
        #    self.store_embeddings(labels, predictions)
        loss = sb.nnet.losses.bce_loss(predictions["bin_pred"].squeeze(1).float(), labels.squeeze(1).float(), pos_weight=torch.Tensor([self.hparams.positive]).to("cuda:0"))
        binary_preds = torch.round(torch.sigmoid(predictions["bin_pred"])) #torch.argmax(, axis=1)
        self.y_true_binary = torch.cat((self.y_true_binary,labels))
        self.y_preds_binary = torch.cat((self.y_preds_binary,binary_preds))
        return loss

    def store_embeddings(self, labels, predictions):
        if (self.embeddings is None):
               self.embeddings = predictions["embeddings"]
               self.labels = labels.unsqueeze(1)
        else:
               self.embeddings = torch.vstack((self.embeddings, predictions["embeddings"]))
               self.labels = torch.vstack((self.labels, labels.unsqueeze(1)))

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.y_preds_binary = torch.tensor(()).to("cuda:0")
        self.y_true_binary = torch.tensor(()).to("cuda:0")
        self.y_preds_multi = torch.tensor(()).to("cuda:0")
        self.y_true_multi = torch.tensor(()).to("cuda:0")
        #self.embeddings = None
        self.labels = None
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage.""" 
        self.compute_metrics(epoch, stage, stage_loss)
        if stage != sb.Stage.TEST:
            writer.add_scalar(f"Loss/{stage.name.split('.')[-1].lower()}", stage_loss, epoch)
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["macro"] = self.fscore * 100
            
            if stage == sb.Stage.VALID:
                self.stage_loss = stage_loss
                if(isinstance(self.hparams.counter,EpochCounterWithStopper)):
                    self.hparams.counter.update_metric(stage_loss)
                if self.hparams.ckpt_enable:
                    self.checkpointer.save_and_keep_only(
                        meta=stage_stats, min_keys=["loss"], keep_recent=False, name=f"ckpt_{epoch}"
                    )
                if(stage_loss < self.best_loss):
                    self.best_loss = stage_loss
                if(self.fscore > self.best_fscore):
                    self.best_fscore = self.fscore
                    self.best_epoch = epoch
                stage_stats["best_macro"] = 1-self.best_fscore
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stage_stats,
                )
                
                """
                if(epoch%10 ==1 or epoch==self.hparams.number_of_epochs):
                    model.utils.compute_TSNE(self.embeddings, self.labels, stage.name.split('.')[-1].lower(), epoch=epoch, num_class=self.hparams.num_class, writer=writer)
                    #if("wav2vec2" in self.hparams.models and self.hparams.with_pooling):
                    #    model.utils.plot_embeddings(self.feats, self.labels, epoch=epoch, layers=self.hparams.layers, writer=writer)
                if(self.best_epoch==epoch):
                    model.utils.compute_TSNE(self.embeddings, self.labels, "best_epoch", epoch, num_class=self.hparams.num_class, writer=writer)
                """
            elif stage == sb.Stage.TEST:
                self.results = stage_stats
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.counter.current},
                    test_stats=stage_stats,
                )
                self.test_loss = stage_loss
                self.test_macro_fscore = stage_stats["macro"]
                self.test_fscore = self.fscore

    def compute_metrics(self, epoch, stage, stage_loss):
        curr_stage = stage.name.split('.')[-1].lower()
        print(f"******{curr_stage}******")
        self.accuracy, self.fscore, self.missrate, self.cf_matrix, _, _= model.utils.my_confusion_matrix(self.y_true_binary, self.y_preds_binary, bin=0)
        #_, self.fscores, _, self.cf_multi_matrix, _, _, = model.utils.my_confusion_matrix(self.y_true_multi, self.y_preds_multi, bin=2)
        
        print(self.cf_matrix)
        self.hparams.train_logger.log_stats(stats_meta={"\nbin fscore": np.round(self.fscore,4),
                                                        "\nAcc.": np.round(self.accuracy,4)})
        print("---------------")
        if(self.hparams.num_class>1):
            print(self.cf_multi_matrix)
            self.hparams.train_logger.log_stats(stats_meta={ "\nmulti-fscores": np.round(self.fscores,4)})

def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("Show","EpId", "ClipId")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def audio_pipeline(Show, EpId, ClipId):
        EpId = int(EpId)
        if(Show=="FluencyBank"):
            file = f"/data/fluencybank/FluencyBank/{EpId:03}/FluencyBank_{EpId:03}_{ClipId}.wav"
        else:
            file = f"/data/sep28k/sep28k_clips/{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        audio = waveform
        return (EpId, int(ClipId)), audio.squeeze()
    @sb.utils.data_pipeline.takes("Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "Unsure", "DifficultToUnderstand", "PoorAudioQuality","NoStutteredWords")
    @sb.utils.data_pipeline.provides("label", "disfluency")
    def get_label(p, b, sr, wr, inter,unsure, difficultToUnderstand, poor, f):
        label, disfluency = get_labels(p,b,sr,wr,inter, unsure, difficultToUnderstand, poor,f)
        return label, disfluency

    @sb.utils.data_pipeline.takes("Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "Unsure", "DifficultToUnderstand", "PoorAudioQuality","NoStutteredWords")
    @sb.utils.data_pipeline.provides("label", "disfluency")
    def get_label_test(p, b, sr, wr, inter,unsure, difficultToUnderstand, poor, f):
        label, disfluency = get_labels(p,b,sr,wr,inter, unsure, difficultToUnderstand, poor,f, "test")
        return label, disfluency

    datasets={}
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")
        csv_path = None
        if dataset=="train" or dataset =="valid":
            if hparams["train_set"] == "sep28k-E":
                csv_path=f'/data/csv/sep28k-E/SEP28k-E_{dataset}.csv'
            elif hparams["train_set"] == "random":
                csv_path=f'/data/csv/sep28k/sep28k_annotation_{dataset}_{hparams["fold"]}.csv'
            elif hparams["train_set"] == "fluencybank":
                csv_path=f'/data/csv/fluencybank/fluencybank_{dataset}.csv'
            datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_path,
                dynamic_items=[audio_pipeline, get_label],
                output_keys=["id", "waveform", "label", "disfluency"])
        else:
            if hparams["test_set"] == "sep28k-E":
                csv_path=f'/data/csv/sep28k-E/SEP28k-E_{dataset}.csv'
            elif hparams["test_set"] == "random":
                csv_path=f'/data/csv/sep28k/sep28k_annotation_{dataset}_{hparams["fold"]}.csv'
            elif hparams["test_set"] == "fluencybank":
                csv_path=f'/data/csv/fluencybank/fluencybank_{dataset}.csv'
            datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_path,
                dynamic_items=[audio_pipeline, get_label_test],
                output_keys=["id", "waveform", "label", "disfluency"])
        
        
        

        if(hparams["remove_unsure"] or dataset =="test"):
            d = datasets[dataset].filtered_sorted(key_min_value={"disfluency":0})
            datasets[dataset] = d

        if(not dataset=="test" and hparams["balance"]):
            hparams["positive"] = (1/hparams["balance_ratio"])-1
            counter_1 =0
            #count number of positive samples
            for i in range(len(datasets[dataset])):
                if(datasets[dataset][i]["disfluency"]==1):
                    counter_1+=1
            d = datasets[dataset].filtered_sorted(sort_key="disfluency", reverse=True, select_n=None)

            count = 0
            new_data = {}
            #keep all postive samples
            for key in d:
                if count < counter_1:
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                    count+=1
            total_samples = min((1/hparams["balance_ratio"])*counter_1, len(d))
            #while current_nb_samples < ratio * positive_samples
            while count < total_samples :
                x = random.choice(datasets[dataset])
                if x["id"] not in new_data.keys() and x["disfluency"]!=1:
                    id = x["id"]
                    x.pop("id")
                    new_key = x
                    new_data[id] = new_key
                    count+=1
            @sb.utils.data_pipeline.takes("label")
            @sb.utils.data_pipeline.provides("label")
            def label2label(label):
                return label
            @sb.utils.data_pipeline.takes("waveform")
            @sb.utils.data_pipeline.provides("waveform")
            def wav2wav(wav):
                return wav
            new_set = sb.dataio.dataset.DynamicItemDataset(new_data,  dynamic_items=[wav2wav, label2label],
                                                                    output_keys=["id", "waveform", "label"],)
            datasets[dataset] = new_set

        if(dataset == "train"):
            d_fluent = datasets[dataset].filtered_sorted(key_min_value={"label":0},key_max_value={"label":0})
            d_disfluent = datasets[dataset].filtered_sorted(key_min_value={"label":1},key_max_value={"label":1})
            hparams["positive"] = len(d_fluent)/len(d_disfluent)
            print(f"There are {len(d_disfluent)} {hparams['stutter']} samples, therefore positive_weight is set to {hparams['positive']}.")
        print(f"The set includes {len(datasets[dataset])} samples.")
    return datasets



def get_labels(p,b,sr,wr,inter, unsure, difficultToUnderstand, poor, f, dataset="train"):
    active_label = torch.zeros(6)
    out = torch.zeros(1)

    labels = torch.tensor([int(f),int(inter),int(sr),int(wr),int(p),int(b)])
    if(hparams[f"annot_value_{dataset}"]==3):
        #Which labels are considered ?
        if(hparams["stutter"]=="Interjection"):
            active_label[1]=1
        if(hparams["stutter"]=="SoundRep"):
            active_label[2] =1
        if(hparams["stutter"]=="WordRep"):
            active_label[3] =1
        if(hparams["stutter"]=="Prolongation"):
            active_label[4] =1
        if(hparams["stutter"]=="Block"):
            active_label[5] =1
        #Fluency class is always considered
        active_label[0]=1
        # Which labels are considered combined with labels to output only the labels we are interested in
        final = active_label * labels
        final = torch.where(final==3, 3, 0)
        if(torch.count_nonzero(final)==1):   
            if(final[0]==3):
                disfluency = 0 #fluent speech
            else:
                out[0] = 1
                disfluency = 1 #disfluent speech
        else:
            disfluency= -1 #unsure
    else:
        if(int(sr)>=hparams[f"annot_value_{dataset}"] and hparams["stutter"]=="SoundRep"):
            active_label[2] =1
        if(int(wr)>=hparams[f"annot_value_{dataset}"] and hparams["stutter"]=="WordRep"):
            active_label[3] =1
        if(int(p)>=hparams[f"annot_value_{dataset}"] and hparams["stutter"]=="Prolongation"):
            active_label[4] =1
        if(int(b)>=hparams[f"annot_value_{dataset}"] and hparams["stutter"]=="Block"):
            active_label[5] =1
        if(int(inter)>=hparams[f"annot_value_{dataset}"] and hparams["stutter"]=="Interjection"):
            active_label[1] = 1


        #if(torch.any(active_label[1:]>=1) and int(f)>=hparams[f"annot_value_{dataset}"]):
            #sample is both "fluent" and "disfluent"
            #out[0] = 1
            #disfluency = -2
        if(torch.all(labels<hparams[f"annot_value_{dataset}"])):
            # is neither
            disfluency = -2
        elif(torch.count_nonzero(active_label)==0):
            disfluency = 0
        elif(torch.count_nonzero(active_label)>0): #and int(f)<hparams[f"annot_value_{dataset}"]
            out[0]=1
            disfluency = 1

        else:
            #Should never be reached but just in case
            print(active_label)
            print(labels)
            print('UNKNOWN PROCESSING')

    if(disfluency >=0 and (int(unsure)>0 or int(difficultToUnderstand)>0 or int(poor)>0)):
        disfluency=-1
    return out, disfluency
    
import os
import data_prep_utils
from termcolor import colored, cprint
from speechbrain.utils import hpopt as hp
import csv
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    with hp.hyperparameter_optimization(objective_key="loss") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
            try:
                hparams["output_folder"] = hparams["output_folder"]+hp.get_trial_id()
                print("USING ORION TO OPTIMIZE")
            except:
                print("NOT USING ORION")
    
            writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")
        print("*********************************************************************")

        print(colored(overrides, 'green'))
        CLASSES = []
        CLASSES.append(hparams["stutter"])
        print(CLASSES)
        cprint(f"Model architecture used is ", end='')
        print(colored(hparams['models'], 'red'))
        print("*********************************************************************")
        
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        
        if(hparams["train_set"]=="Libri"):
            datasets = data_prep_utils.dataio_prep(hparams)
        else:    
            datasets = dataio_prep(hparams)
        # Initialize trainer
        opt_class = partial(hparams["opt_class"].func, lr=float(hparams["opt_class"].keywords["lr"]))
        detect_brain = LatimBrain(
            modules= hparams["modules"],
            opt_class=opt_class,
            run_opts=run_opts,
            hparams=hparams,
            checkpointer=hparams["checkpointer"],
        )

        if(hparams["train_set"]!="syn"):
            hparams["dataloader_opts"]["collate_fn"] = PaddedBatch
        else:
            hparams["dataloader_opts"]["collate_fn"] = lambda batch: segmenting_collate_fn(batch, segment_length=3, 
                                                                                    num_class=hparams["num_class"])

        detect_brain.best_fscore = -1
        detect_brain.best_loss = 10000000
        detect_brain.best_epoch= -1
        # Fit dataset
        detect_brain.fit(
            epoch_counter=hparams["counter"],
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )
        
        print("*"*20, "Evaluation", "*"*20)
        datatest = hparams["test_set"]
        print(overrides)
        if(hparams["test_set"]!="syn"):
            hparams["dataloader_opts"]["collate_fn"] = PaddedBatch
        else:
            hparams["dataloader_opts"]["collate_fn"] = lambda batch: segmenting_collate_fn(batch, segment_length=3, 
                                                                                    num_class=hparams["num_class"])
        
        detect_brain.evaluate(
            datasets[f"test"],
            test_loader_kwargs=hparams["dataloader_opts"],
        )

        writer.add_hparams(overrides,
                           {
                            'score/F1-macro': detect_brain.test_fscore,
                            'best_score': detect_brain.best_fscore
                            })
        
        print(detect_brain.results)
        hp.report_result(detect_brain.results)
        with open(f'{hparams["output_folder"]}/orion.csv', 'w') as csv_file:  
            wr = csv.writer(csv_file)
            for key, value in detect_brain.results.items():
                wr.writerow([key, value])
        writer.flush()
        writer.close()

        detect_brain.checkpointer.delete_checkpoints(num_to_keep=0)
