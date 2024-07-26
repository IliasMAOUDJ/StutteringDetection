from functools import partial
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import model.utils
import model.features
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
                wavs = self.hparams.noisereducer(wavs)
        if stage == sb.Stage.TRAIN:
            if(self.hparams.augment): 
                wavs = self.hparams.envcorrupt(wavs, lens)
                wavs = self.hparams.timedomainaugment(wavs, lens)
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
        waveforms, lens = batch.waveform
        waveforms = self.compute_feats(waveforms, lens, stage)
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
                self.results = stage_stats
                self.stage_loss = stage_loss
                
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
    @sb.utils.data_pipeline.takes("Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoStutteredWords")
    @sb.utils.data_pipeline.provides("label", "disfluency")
    def get_label(p, b, sr, wr, inter, f):
        label, disfluency = get_labels(p,b,sr,wr,inter,f)
        return label, disfluency

    datasets={}
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")
        csv_path = None
        if dataset=="train" or dataset =="valid":
            if hparams["train_set"] == "sep28k":
                csv_path=f'/data/csv/SEP28k-E_{dataset}.csv'
            elif hparams["train_set"] == "fluencybank":
                csv_path=f'/data/csv/fluencybank_{dataset}.csv'

            if hparams["train_set"] == "syn":
                tmp = data_prep_utils.dataio_prep(hparams)
                datasets[f"{dataset}"] = tmp[f"{dataset}"]
        else:
            if hparams["test_set"] == "sep28k":
                csv_path=f'/data/csv/SEP28k-E_{dataset}.csv'
            elif hparams["test_set"] == "fluencybank":
                csv_path=f'/data/csv/fluencybank_{dataset}.csv'
        if(csv_path is not None):
            datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_path,
                dynamic_items=[audio_pipeline, get_label],
                output_keys=["id", "waveform", "label", "disfluency"],
            )

            
        if(hparams["remove_unsure"]):
            counter_u =0
            for i in range(len(datasets[dataset])):
                    if(datasets[dataset][i]["disfluency"]==-1):
                        counter_u +=1
            d = datasets[dataset].filtered_sorted(sort_key="disfluency", reverse=True, select_n=len(datasets[dataset])-counter_u)
            datasets[dataset] = d
        
        if(dataset=="train" or hparams["balance_test"]):
            if(hparams["balance"] or hparams["balance_test"]):
                counter_1 =0
                #count number of positive samples
                for i in range(len(datasets[dataset])):
                    if(datasets[dataset][i]["disfluency"]==1):
                        counter_1 +=1
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
                    if x["id"] not in new_data.keys() and x["disfluency"]==0:
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
    return datasets
def get_labels(p,b,sr,wr,inter, f):
    label = torch.zeros(6)
    out = torch.zeros(2)
    if(hparams["annot_value"]==3):
        labels = torch.tensor([int(f),int(inter),int(sr),int(wr),int(p),int(b)])
        if(hparams["SoundRep"]):
            #int(sr)>=hparams["annot_value"] and 
            #label[CLASSES.index("SoundRep")] = 1
            label[2] =1
        if(hparams["WordRep"]):
            #int(wr)>=hparams["annot_value"] and 
            #label[CLASSES.index("WordRep")] = 1
            label[3] =1
        if(hparams["Prolongation"]):
            #int(p)>=hparams["annot_value"] and 
            #label[CLASSES.index("Prolongation")] = 1
            label[4] =1
        if(hparams["Block"]):
            #int(b)>=hparams["annot_value"] and 
            #label[CLASSES.index("Block")] = 1
            label[5] =1
        if(hparams["Interjection"]):
            #int(inter)>=hparams["annot_value"]
            #label[CLASSES.index("Interjection")] = 1
            label[1]=1
        if(f==3):
            label[0]=1
        final = label * labels
        if(torch.count_nonzero(final)>=1):   
            if(torch.nonzero(final)[0]==0):
                out[0] = 1
                disfluency = 0
            else:
                out[1] = 1
                disfluency = 1
        else:
            disfluency= -1
    elif(hparams["annot_value"]==2):
        if(int(sr)>=hparams["annot_value"] and hparams["stutter"]=="SoundRep"):
            label[2] =1
        if(int(wr)>=hparams["annot_value"] and hparams["stutter"]=="WordRep"):
            label[3] =1
        if(int(p)>=hparams["annot_value"] and hparams["stutter"]=="Prolongation"):
            label[4] =1
        if(int(b)>=hparams["annot_value"] and hparams["stutter"]=="Block"):
            label[5] =1
        if(int(inter)>=hparams["annot_value"] and hparams["stutter"]=="Interjection"):
            label[1] = 1
        if(torch.count_nonzero(label)==0):
            out[0]=1
            disfluency = 0
        else:
            out[1]=1
            disfluency = 1
    if(hparams["num_class"]==1):
        out = out[1:]
    return out, disfluency
    
import os
import data_prep_utils
from termcolor import colored, cprint
from speechbrain.utils import hpopt as hp
import csv
from speechbrain.utils import checkpoints as ckpt
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    with hp.hyperparameter_optimization(objective_key="loss") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
            try:
                hparams["output_folder"] = hparams["output_folder"]+hp.get_trial_id()
                writer = SummaryWriter("/tensorboard")
            except:
                writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")
        print("*********************************************************************")
        cprint(f"Model architecture used is ", end='')
        cprint(colored(hparams['models'],'red', attrs=["bold"]))

        print(colored(overrides, 'green'))
        CLASSES = ["Fluent"]
        CLASSES.append(hparams["stutter"])
        print(CLASSES)
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
        
        hp.report_result(detect_brain.results)
        with open(f'{hparams["output_folder"]}/orion.csv', 'w') as csv_file:  
            wr = csv.writer(csv_file)
            for key, value in detect_brain.results.items():
                wr.writerow([key, value])
        writer.flush()
        writer.close()

        detect_brain.checkpointer.delete_checkpoints(num_to_keep=0)
