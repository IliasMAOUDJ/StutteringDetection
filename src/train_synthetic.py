from functools import partial
import os
from unicodedata import category
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import model.utils
import torchaudio
import pandas as pd
import numpy as np
from data_prep_utils import *
from huggingface_hub.hf_api import HfFolder
from torchmetrics.classification import BinaryPrecisionRecallCurve

HfFolder.save_token("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")

class LatimBrain(sb.Brain):
    def compute_feats(self, wavs, lens, stage):
        if(self.hparams.nr):
            with torch.no_grad():
                wavs = self.hparams.noisereducer(wavs)
        if stage == sb.Stage.TRAIN:
            if(self.hparams.augment): 
                wavs = self.hparams.envcorrupt(wavs, lens)
                wavs = self.hparams.timedomainaugment(wavs, lens)
                
            if(self.hparams.normalize):
                #feats = (feats - self.hparams.dataset_mean) / (self.hparams.dataset_std)
                wavs = self.hparams.normalizer(wavs, lens, epoch=self.hparams.counter.current)

        if(wavs.shape[1]>48000):
            wavs = wavs[:,:48000]
        elif(wavs.shape[1]<48000):
            pad = torch.zeros([wavs.shape[0], 48000-wavs.shape[1]]).to("cuda:0")
            wavs= torch.cat([wavs,pad], dim=1)
        return wavs

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        waveforms, lens = batch.waveform
        waveforms = self.compute_feats(waveforms, lens, stage)
        bin_out = self.modules.model(waveforms)
        return {"bin_pred" : bin_out}

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label.data
        #ids = torch.tensor([int(x) for x in batch.id]).to("cuda:0")
        binary_labels = torch.tensor(()).to("cuda:0")
        for label in labels:
            if(torch.count_nonzero(label)==0):
                binary_label= torch.tensor([[0]])
            else:
                binary_label = torch.tensor([[1]])
            binary_labels = torch.cat((binary_labels,binary_label.to("cuda:0")))
        binary_loss = sb.nnet.losses.bce_loss(predictions["bin_pred"].float(), binary_labels.float(), pos_weight=torch.Tensor([self.hparams.positive]).to("cuda:0"))
        
        binary_preds = torch.sigmoid(predictions["bin_pred"])
        self.y_true_binary = torch.cat((self.y_true_binary,binary_labels))
        self.y_preds_binary = torch.cat((self.y_preds_binary,binary_preds))
        return binary_loss


    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.y_preds_binary = torch.tensor(()).to("cuda:0")
        self.y_true_binary = torch.tensor(()).to("cuda:0")
        self.all_ids = torch.tensor(()).to("cuda:0")
        self.auroc = BinaryPrecisionRecallCurve().to("cuda:0")
        self.embeddings = None
        self.labels = None
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        self.compute_metrics(epoch, stage, stage_loss)
        if stage != sb.Stage.TEST:
            writer.add_scalar(f"Loss/{stage.name.split('.')[-1].lower()}", stage_loss, epoch)
            writer.add_scalar(f"Fscore/{stage.name.split('.')[-1].lower()}", self.fscore * 100, epoch)
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["macro"] = self.fscore * 100
            stage_stats["rev_macro"] = 100 - self.fscore *100
            
            if stage == sb.Stage.VALID:
                self.stage_loss = stage_loss
                if(stage_loss < self.best_loss or self.fscore > self.best_fscore):
                    self.min_loss = stage_loss
                    self.best_matrix = self.cf_matrix
                    self.best_precision = self.prec
                    self.best_recall = self.recall
                    self.best_fscore = self.fscore
                    self.best_epoch = epoch
                    self.best_loss = stage_loss
                
                if self.hparams.ckpt_enable:
                    self.checkpointer.save_and_keep_only(
                        meta=stage_stats, max_keys=["macro"], keep_recent=False, num_to_keep=1, name=f"ckpt_{epoch}"
                    )
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stage_stats,
                )
            elif stage == sb.Stage.TEST:
                self.test_loss = stage_loss
                if(CURRENT=="loss"):
                    self.test_fscore_loss = self.fscore
                if(CURRENT=="macro"):
                    self.test_fscore_macro = self.fscore
                self.results = stage_stats
                with open(f'{hparams["save_folder"]}/orion.csv', 'a') as csv_file:  
                    wr = csv.writer(csv_file)
                    for key, value in overrides.items():
                        wr.writerow([key, value])
                    for key, value in detect_brain.results.items():
                        wr.writerow([key, value])
                    wr.writerow("\n")

    def compute_metrics(self, epoch, stage, stage_loss):
        curr_stage = stage.name.split('.')[-1].lower()
        print(f"******{curr_stage}******")
        self.accuracy, self.fscore, self.missrate, self.cf_matrix, self.prec, self.recall= model.utils.my_confusion_matrix(self.y_true_binary, self.y_preds_binary, bin=0)
        print(self.cf_matrix)
        self.hparams.train_logger.log_stats(stats_meta={"F1-score": np.round(self.fscore,4)})


def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("record","clipID")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def french_audio_pipeline(record,clipID):
        file = f"/data/databrase/Clips/{record}_{clipID}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        audio = waveform
        return int(clipID), audio.squeeze()

    @sb.utils.data_pipeline.takes("trial", "index")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def iemas_audio_pipeline(trial, index):
        file = f"/data/iemas_nrk/{trial}_{index}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        return (int(index)), waveform.squeeze()
    
    @sb.utils.data_pipeline.takes("name","index")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def synthetic_audio_pipeline(filename,index):
        file = f"/data/synthetic/{filename}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        return int(index), waveform.squeeze()

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label")
    def our_get_label(stutter):
        if "n" in stutter:
            return torch.Tensor([0])
        else:
            return torch.Tensor([1])

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label")
    def iemas_get_label(stutter):
        if not "n" in stutter:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
        
    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label")
    def synthetic_get_label(stutter):
        if stutter=="True":
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])

    datasets={}
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")
        if(hparams[f"{dataset}_set"]=="databrase"):
            audio_func, label_func =  french_audio_pipeline, our_get_label
        elif(hparams[f"{dataset}_set"]=="syn"):
            audio_func, label_func =  synthetic_audio_pipeline, synthetic_get_label
        else:
            audio_func, label_func =  iemas_audio_pipeline, iemas_get_label
        datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                        csv_path=hparams[f'{dataset}_set'],
                                        dynamic_items=[audio_func, label_func],
                                        output_keys=["id", "waveform", "label"],
                                        )
    return datasets
    
import os
import data_prep_utils
from termcolor import colored, cprint
from speechbrain.utils import hpopt as hp
import csv
from collections import Counter
from speechbrain.utils import checkpoints as ckpt
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    with hp.hyperparameter_optimization(objective_key="rev_macro") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
            try:
                hparams["output_folder"] = hparams["output_folder"]+hp.get_trial_id()
                writer = SummaryWriter("/tensorboard")
            except:
                writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")

        print("*********************************************************************")
        
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
            save_env_desc=False
        )
   
        datasets = dataio_prep(hparams)
        hparams["num_class"] = 1
        opt_class = partial(hparams["opt_class"].func, lr=float(hparams["opt_class"].keywords["lr"]))
        # Initialize trainer
        detect_brain = LatimBrain(
            modules= hparams["modules"],
            opt_class=opt_class,
            run_opts=run_opts,
            hparams=hparams,
            checkpointer=hparams["checkpointer"],
        )

        detect_brain.best_fscore = -1
        detect_brain.best_loss = 10000000
        detect_brain.best_epoch= -1

        # Fit dataset
        detect_brain.test_only = False       
        detect_brain.fit(
            epoch_counter=hparams["counter"],
            train_set=datasets[f"train"],
            valid_set=datasets[f"valid"],
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )
        
        print("*"*20, "Evaluation", "*"*20)
        print(overrides)
        print("*"*5, f"Best epoch: {detect_brain.best_epoch} with best f1-score: {detect_brain.best_fscore}", "*"*5)
        print("*"*5, f'Testing on {hparams["test_set"]}', "*"*5)
        CURRENT = "macro"
        detect_brain.evaluate(
            datasets[f"test"],
            max_key="macro",
            test_loader_kwargs=hparams["dataloader_opts"],
        )

        writer.add_hparams(overrides,
                           {
                            'score/F1-macro': detect_brain.test_fscore_macro,
                            'best_score': detect_brain.best_fscore
                            })

        hp.report_result(detect_brain.results)
        
        writer.flush()
        writer.close()
        detect_brain.checkpointer.delete_checkpoints(num_to_keep=0)
