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
from torchmetrics.classification import BinaryAveragePrecision, BinaryPrecisionRecallCurve

HfFolder.save_token("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")
#HfFolder.save_token("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")

class LatimBrain(sb.Brain):
    def compute_feats(self, wavs, lens, stage):
        if(self.hparams.nr):
            with torch.no_grad():
                wavs = self.hparams.noisereducer(wavs)
        #if(self.hparams.preaugment):
        #        #wavs = self.hparams.preaugment(wavs, lens)
        #    mywavs = np.empty_like(wavs.cpu())
        #    for i in range(wavs.shape[0]):
        #        wav = nr.reduce_noise(y=wavs[i].cpu(), sr=16000, time_constant_s=1.5, use_torch=True, device="cuda:0")
        #        mywavs[i]= wav
        #    wavs = torch.tensor(mywavs).to("cuda:0")
        #feats = self.hparams.compute_feats(wavs)
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
        
        #bin_pred = self.modules.bin_classifier(embeddings)
        #return {"bin_pred" : bin_pred,
        #        "embeddings": embeddings} # = predictions

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
        #if(stage==sb.Stage.TEST):
        #    if (self.embeddings is None):
        #        self.embeddings = predictions["embeddings"]
        #        self.labels = labels.unsqueeze(1)
        #    else:
        #        self.embeddings = self.embeddings.squeeze(1)
        #        self.embeddings = torch.vstack((self.embeddings, predictions["embeddings"]))
        #        self.labels = torch.vstack((self.labels, labels.unsqueeze(1)))
        binary_loss = sb.nnet.losses.bce_loss(predictions["bin_pred"].float(), binary_labels.float(), pos_weight=torch.Tensor([self.hparams.positive]).to("cuda:0"))
        
        binary_preds = torch.sigmoid(predictions["bin_pred"]) #torch.argmax(, axis=1)
        #self.all_ids = torch.cat((self.all_ids,ids))
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
                
                """
                if(epoch%5 ==1 or epoch==self.hparams.number_of_epochs):
                    model.utils.compute_TSNE(self.embeddings, self.labels, stage.name.split('.')[-1].lower(), epoch=epoch, num_class=self.hparams.num_class, writer=writer)
                    #if("wav2vec2" in self.hparams.models and self.hparams.with_pooling):
                    #    model.utils.plot_embeddings(self.feats, self.labels, epoch=epoch, layers=self.hparams.layers, writer=writer)
                if(self.best_epoch==epoch):
                    model.utils.compute_TSNE(self.embeddings, self.labels, "best_epoch", epoch, num_class=self.hparams.num_class, writer=writer)
                """
            elif stage == sb.Stage.TEST:
                self.classes = []
                for pred, gt in zip(torch.round(self.y_preds_binary), self.y_true_binary):
                    if(int(gt==pred) and pred.item() ==1):
                        self.classes.append("TP")
                    elif(int(gt==pred) and pred.item() ==0):
                        self.classes.append("TN")
                    elif(int(gt!=pred) and pred.item() ==1):
                        self.classes.append("FP")
                    else:
                        self.classes.append("FN")
                if self.hparams.tsne:
                    model.utils.compute_TSNE(self.embeddings, self.classes, stage, writer)
                    prec, rec, thresh = self.auroc(self.y_preds_binary, self.y_true_binary.to(int))
                    best_thresh = -1
                    best_sc = 0
                    for p,r,t in zip(prec,rec,thresh):
                        if(2*p*r/(p+r)>best_sc):
                            best_sc = (2*p*r/(p+r)).item()
                            best_thresh = t.item()
                            best_p = p.item()
                            best_r = r.item()
                    print("best_score: ", best_sc, " with threshold: ", best_thresh, 
                          "\nprecision: ", best_p, " recall:", best_r)
                    fig_, ax_ = self.auroc.plot(score=True)
                    ax_.scatter(best_p, best_r)
                    writer.add_figure(f"AUPRC/{stage}", plt.gcf())
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
        if stage == sb.Stage.TEST:
            """
            file_exists = os.path.isfile(f'{hparams["output_folder"]}/predictions_{CURRENT}.csv')
            tracks, indices = annotationToTrial(self.all_ids)
            with open(f'{hparams["output_folder"]}/predictions_{CURRENT}.csv', 'a') as csv_file:  
                wr = csv.writer(csv_file)
                if not(file_exists):
                    wr.writerow(["ID", "index", "PRED", "GT", "GoodPred?"])
                trials, _ = np.unique(tracks, return_counts=True)
                tps, fps, fns, tns = {},{},{},{}
                for trial in trials:
                    tps[trial] = 0
                    fps[trial] = 0
                    fns[trial] = 0
                    tns[trial] = 0
                for id, ind, pred, gt in zip(tracks, indices, torch.round(self.y_preds_binary), torch.round(self.y_true_binary)):
                    wr.writerow([id, ind, pred.item(), gt.item(), int(gt==pred)])
                    if(int(gt==pred) and pred.item() ==1):
                        tps[id] +=1
                    elif(int(gt==pred) and pred.item() ==0):
                        tns[id] +=1
                    elif(int(gt!=pred) and pred.item() ==1):
                        fps[id] +=1
                    else:
                        fns[id] +=1
                wr.writerow(["ID", "TP", "TN", "FP", "FN"])
                for trial in trials:
                    wr.writerow([trial, tps[trial], tns[trial], fps[trial], fns[trial]])
                """
        self.accuracy, self.fscore, self.missrate, self.cf_matrix, self.prec, self.recall= model.utils.my_confusion_matrix(self.y_true_binary, self.y_preds_binary, bin=0)
        print(self.cf_matrix)
        self.hparams.train_logger.log_stats(stats_meta={"F1-score": np.round(self.fscore,4)})

def annotationToTrial(ids):
    if("iemas" in {hparams["test_set"]} or "scot" in {hparams["test_set"]} or "hscot" in {hparams["test_set"]}):
        dataset = "iemas"
    else:
        dataset = "databrase"
    annotations = f"/data/{dataset}/annotation.csv"
    df = pd.read_csv(annotations)
    tracks = []
    indices = []
    for id in ids:
        id = id.item()
        line = df[df["ID"]==id]
        if dataset =="iemas":
            trial = line.iloc[0]["trial"]
            index = line.iloc[0]["index"]
        else:
            trial = line.iloc[0]["record"]
            index = line.iloc[0]["clipID"]
        tracks.append(trial)
        indices.append(index)
    return tracks, indices

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
        file = f"/data/iemas/{trial}_{index}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        return (int(index)), waveform.squeeze()
    
    @sb.utils.data_pipeline.takes("name","index")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def synthetic_audio_pipeline(filename,index):
        file = f"/data/synthetic/{filename}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        return int(index), waveform.squeeze()

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label", "disfluency", "origin")
    def our_get_label(stutter):
        if "n" in stutter:
            return torch.Tensor([0]), 0, "databrase"
        else:
            return torch.Tensor([1]), 1, "databrase"

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label", "disfluency", "origin")
    def iemas_get_label(stutter):
        if not "n" in stutter:
            return torch.Tensor([1]), 1, "iemas"
        else:
            return torch.Tensor([0]), 0, "iemas"
        
    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label", "disfluency", "origin")
    def synthetic_get_label(stutter):
        if stutter=="True":
            return torch.Tensor([1]), 1, "syn"
        else:
            return torch.Tensor([0]), 0, "syn"

    datasets={}
    fold = hparams["fold"]
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")



        @sb.utils.data_pipeline.takes("waveform", "label", "disfluency", "origin")
        @sb.utils.data_pipeline.provides("waveform", "label", "disfluency", "origin")
        def copy(waveform, label, disfluency, origin):
            return waveform, label, disfluency, origin
        if dataset == "train" or dataset == "valid":
            csv_path = None
            if hparams["train_set"] == "iemas" and hparams["test_set"] == "databrase":
                if(dataset =="train"):
                    csv_path=f'/data/csv/annotation_train.csv'
                else:
                    csv_path=f'/data/csv/annotation_test.csv'
            elif(hparams["train_set"]=="scot" or hparams["train_set"]=="hscot"):
                csv_path=f'/data/csv/{hparams["train_set"]}_annotation_{dataset}_all.csv'
            elif(hparams["train_set"]=="iemas"):
                if(hparams["subset"] =="all"):
                    csv_path=f'/data/csv/annotation_{dataset}_{fold}.csv'
                elif(hparams["subset"]=="random"):
                    csv_path=f'/data/csv/random_annotation_{dataset}_{fold}.csv'
                elif(hparams["subset"]=="scot" or hparams["subset"]=="hscot"):
                    csv_path=f'/data/csv/{hparams["subset"]}_annotation_{dataset}_{fold}.csv'
            if(csv_path is not None):
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                        csv_path=csv_path,
                                        dynamic_items=[iemas_audio_pipeline, iemas_get_label],
                                        output_keys=["id", "waveform", "label", "disfluency", "origin"],
                                        )
            if hparams["train_set"] == "syn":
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                        csv_path=f'/data/csv/{dataset}_syn.csv',
                                        dynamic_items=[synthetic_audio_pipeline, synthetic_get_label],
                                        output_keys=["id", "waveform", "label", "disfluency", "origin"],
                                        )
                #tmp = data_prep_utils.dataio_prep(hparams)
                #datasets[f"{dataset}"] = tmp[f"{dataset}"]
            if("iemas" in hparams["train_set"] and "databrase" in hparams["train_set"]):
                csv_path=f'/data/csv/annotation_{dataset}_{fold}.csv'
                iemas = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                            csv_path=csv_path,
                                            dynamic_items=[iemas_audio_pipeline, iemas_get_label],
                                            output_keys=["id", "waveform", "label", "disfluency", "origin"])
                
                csv_path=f'/data/databrase/annotation.csv'
                databrase = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                            csv_path=csv_path,
                                            dynamic_items=[french_audio_pipeline, our_get_label],
                                            output_keys=["id", "waveform", "label", "disfluency", "origin"])

                new_data = {}
                for key in iemas:
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                for key in databrase:
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset(new_data,  dynamic_items=[copy],
                                                                        output_keys=["id", "waveform", "label", "disfluency", "origin"])
            
        else:
            if hparams["test_set"] == "syn":
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                        csv_path=f'/data/csv/{dataset}_syn.csv',
                                        dynamic_items=[synthetic_audio_pipeline, synthetic_get_label],
                                        output_keys=["id", "waveform", "label", "disfluency", "origin"],
                                        )
                #tmp = data_prep_utils.dataio_prep(hparams)
                #datasets[f"{dataset}"] = tmp[f"{dataset}"]
            elif("iemas" in hparams["test_set"] and "databrase" in hparams["test_set"]):
                csv_path=f'/data/csv/annotation_{dataset}_{fold}.csv'
                iemas = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                            csv_path=csv_path,
                                            dynamic_items=[iemas_audio_pipeline, iemas_get_label],
                                            output_keys=["id", "waveform", "label", "disfluency", "origin"])
                
                csv_path=f'/data/databrase/annotation_{dataset}.csv'
                databrase = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                            csv_path=csv_path,
                                            dynamic_items=[french_audio_pipeline, our_get_label],
                                            output_keys=["id", "waveform", "label", "disfluency", "origin"])

                new_data = {}
                for key in iemas:
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                for key in databrase:
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset(new_data,  
                                                                dynamic_items=[copy],
                                                                output_keys=["id", "waveform", "label", "disfluency", "origin"])
            else:
                if(hparams["test_set"]=="scot"):
                        csv_path=f'/data/csv/scot_annotation_all.csv'
                elif(hparams["test_set"]=="hscot"):
                        csv_path=f'/data/csv/hscot_annotation_all.csv'
                elif hparams["train_set"]!= hparams["test_set"] and hparams["test_set"]=="databrase":
                    if hparams["test_set"] == "databrase":
                        csv_path = '/data/databrase/annotation.csv'
                else:
                    if(hparams["subset"]=="scot"):
                        csv_path=f'/data/csv/scot_annotation_test_{fold}.csv'
                    elif(hparams["subset"]=="hscot"):
                        csv_path=f'/data/csv/hscot_annotation_test_{fold}.csv'
                    elif(hparams["subset"]=="random"):
                        csv_path=f'/data/csv/random_annotation_{dataset}_{fold}.csv'
                    else:
                        csv_path = f'/data/{hparams["test_set"]}/annotation_test_{fold}.csv'
                    
                if(hparams["test_set"] == "databrase"):
                    audio_func, label_func =  french_audio_pipeline, our_get_label
                else:
                    audio_func, label_func =  iemas_audio_pipeline, iemas_get_label

                datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                                            csv_path=csv_path,
                                            dynamic_items=[audio_func, label_func],
                                            output_keys=["id", "waveform", "label", "disfluency", "origin"],
                                            )
        
        if((not dataset =="test" and hparams["balance"]) or (dataset == "test" and hparams["balance_test"])):
            counter_1 =0
            #count number of positive samples
            for i in range(len(datasets[dataset])):
                if(datasets[dataset][i]["disfluency"]==1):
                    #if(dataset=="test"):
                    #    print(datasets[dataset][i])
                    counter_1 +=1
            d = datasets[dataset].filtered_sorted(sort_key="disfluency", reverse=True, select_n=None)
            count = 0
            new_data = {}
            #keep all postive samples
            for key in d:
                if count < counter_1:
                    #print(key)
                    id = key["id"]
                    key.pop("id")
                    new_key = key
                    new_data[id] = new_key
                    count+=1

            total_samples = min((1/hparams["balance_ratio"])*counter_1, len(d))
            #while current_nb_samples < ratio * positive_samples
            while count < total_samples :
                x = random.choice(datasets[dataset])
                if x["id"] not in new_data.keys():
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
            @sb.utils.data_pipeline.takes("origin")
            @sb.utils.data_pipeline.provides("origin")
            def origin2origin(origin):
                return origin
            new_set = sb.dataio.dataset.DynamicItemDataset(new_data,  dynamic_items=[wav2wav, label2label, origin2origin],
                                                                    output_keys=["id", "waveform", "label", "origin"],)
            datasets[dataset] = new_set
        #Count occurences in each dataset
        databr_cnt, iemas_cnt = 0,0
        databr_cnt_pos, iemas_cnt_pos = 0,0
        syn_cnt, syn_cnt_pos = 0,0
        for i in range(len(datasets[dataset])):
            if(datasets[dataset][i]["origin"]=="databrase"):
                databr_cnt +=1
                if(datasets[dataset][i]["label"]==1):
                    databr_cnt_pos+=1
            if(datasets[dataset][i]["origin"]=="iemas"):
                iemas_cnt += 1
                if(datasets[dataset][i]["label"]==1):
                    iemas_cnt_pos+=1
            if(datasets[dataset][i]["origin"]=="syn"):
                syn_cnt += 1
                if(datasets[dataset][i]["label"]==1):
                    syn_cnt_pos+=1


        print(f"Number of samples from Databrase: {databr_cnt} with {databr_cnt_pos} positives.")
        print(f"Number of samples from IEMAS: {iemas_cnt} with {iemas_cnt_pos} positives")
        print(f"Number of samples from SYNTHETIC: {syn_cnt} with {syn_cnt_pos} positives")
        print(len(datasets[dataset]))
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
        print(f"Training on",colored(hparams["train_set"],'red', attrs=["bold"]))
        print(f"Testing on",colored(hparams["test_set"],'red', attrs=["bold"]))
        cprint(f"Model architecture used is ", end='')
        cprint(colored(hparams['models'],'red', attrs=["bold"]))

        print(colored(overrides, 'green'))
        
        hparams["train_logger"].log_stats(stats_meta={"model": hparams["model"]})
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
        hparams["dataloader_opts"]["collate_fn"] = PaddedBatch

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
        #if(hparams["test_set"]!="syn"):
        hparams["dataloader_opts"]["collate_fn"] = PaddedBatch
        #else:
        #    hparams["dataloader_opts"]["collate_fn"] = lambda batch: segmenting_collate_fn(batch, segment_length=3, 
                                                                                   # num_class=hparams["num_class"])
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
