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
from speechbrain.dataio.sampler import BalancingDataSampler
HfFolder.save_token("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")



class LatimBrain(sb.Brain):
    """Use attentional model to predict words in segments"""
        

    def compute_feats(self, wavs, lens, stage):
        if(self.hparams.features["preaugment"]):
            if stage == sb.Stage.TRAIN:
                wavs = self.hparams.preaugment(wavs, lens)
        feats = self.hparams.compute_feats(wavs)
        if(self.hparams.features["augment"]): 
                if stage == sb.Stage.TRAIN:
                    feats = self.hparams.spec_augment(feats)
            #feats = feats.transpose(1,2)
        if(self.hparams.features["normalize"]):
            feats = (feats - self.hparams.dataset_mean) / (self.hparams.dataset_std)
                #feats = self.hparams.normalize(
                #    feats, lens, epoch=self.hparams.counter.current)
        
        return wavs, feats

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        waveforms, lens = batch.waveform
        aug_waveform, spectrogram = self.compute_feats(waveforms, lens, stage)
        embeddings = None
        if("wav2vec2" in self.hparams.models):
            bin_out, multi_out, embeddings, feats = self.modules.w2v2_model(aug_waveform)
            self.feats = feats
            return {"bin_pred" : bin_out,
            "multi_pred": multi_out,
            "embeddings": embeddings.squeeze(1)}
        if("other" in self.hparams.models):
            if("wave" in self.hparams.models):
                input = aug_waveform
            else: 
                input = spectrogram
            out, multi_out, embeddings = self.modules.custom(input)
            return {"bin_pred" : out, "multi_pred": multi_out, "embeddings": embeddings.squeeze(1)}
        if("lstm" in self.hparams.models):
            lstm_emb = self.modules.resbilstm(spectrogram)
            embeddings = lstm_emb
        if("ecapa" in self.hparams.models):
            ecapa_emb = self.modules.ecapa_model(spectrogram)
            embeddings = ecapa_emb.squeeze(1)

        if("vit" in self.hparams.models or "resnet" in self.hparams.models):
            pred, emb_resnet = self.modules.my_model(spectrogram)
            if(embeddings is None):
                embeddings = emb_resnet.squeeze(1)
            else:
                embeddings= torch.cat([embeddings, emb_resnet], dim=-1)
        
        bin_pred = self.modules.bin_classifier(embeddings)
        multi_pred = self.modules.multi_classifier(embeddings)
        return {"bin_pred" : bin_pred,
                "multi_pred": multi_pred,
                "embeddings": embeddings} # = predictions

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label.data
        binary_labels = torch.tensor(()).to("cuda:0")
        for label in labels:
            if(torch.count_nonzero(label)==0):
                binary_label= torch.tensor([[0]])
            else:
                binary_label = torch.tensor([[1]])
            binary_labels = torch.cat((binary_labels,binary_label.to("cuda:0")))
        
        #labels = labels.type(torch.LongTensor).to("cuda:0")
        #labels = torch.nn.functional.one_hot(lab, 2)
        if(stage==sb.Stage.VALID):
            if (self.embeddings is None):
                self.embeddings = predictions["embeddings"]
                self.labels = labels.unsqueeze(1)
            else:
                self.embeddings = torch.vstack((self.embeddings, predictions["embeddings"]))
                self.labels = torch.vstack((self.labels, labels.unsqueeze(1)))
        
        #if(stage==sb.Stage.TEST):
            #lastlab = torch.argmax(lab, axis=1).cpu().detach().numpy()
            #pred = predictions["pred"]
            #highpred = torch.argmax(torch.softmax(pred, axis=1), axis=1).cpu().detach().numpy()
            #sm = torch.softmax(predictions["pred"], axis=1)
            
            #x = np.choose(highpred.astype(int), sm.swapaxes(0,1))
            #lastlab = lab[:,0]
            #x = sm[:,0]
            #pvalue = torch.abs(1-torch.abs((lastlab-x )))
            #pr = torch.vstack([pvalue, lastlab])
            
            #print(pr[:5])
            #fluent_loss = self.hparams.BCE_loss(predictions["pred"][:,4], torch.where(labels!=4, 1.0,0.0))
            #print(fluent_loss)
        #binary_loss = self.modules.jouaitiloss(predictions["bin_pred"].squeeze(1).float(), binary_labels.squeeze(1).float()).to("cuda:0")
        binary_loss = sb.nnet.losses.bce_loss(predictions["bin_pred"].squeeze(1).float(), binary_labels.squeeze(1).float(), pos_weight=torch.Tensor([self.hparams.positive]).to("cuda:0"))
        binary_preds = torch.round(torch.sigmoid(predictions["bin_pred"])) #torch.argmax(, axis=1)
        self.y_true_binary = torch.cat((self.y_true_binary,binary_labels))
        self.y_preds_binary = torch.cat((self.y_preds_binary,binary_preds))

        if(self.hparams.TestOurs == False):
            multi_loss = self.modules.multi_loss(predictions["multi_pred"].float(), labels.float())
            multi_preds = torch.round(torch.sigmoid(predictions["multi_pred"]))

            self.y_true_multi = torch.cat((self.y_true_multi,labels))
            self.y_preds_multi = torch.cat((self.y_preds_multi,multi_preds))
            return (binary_loss+ multi_loss)
        else:
            return binary_loss


    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.y_preds_binary = torch.tensor(()).to("cuda:0")
        self.y_true_binary = torch.tensor(()).to("cuda:0")
        self.y_preds_multi = torch.tensor(()).to("cuda:0")
        self.y_true_multi = torch.tensor(()).to("cuda:0")
        self.embeddings = None
        self.labels = None
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        
        self.compute_metrics(epoch, stage, stage_loss)
        if stage != sb.Stage.TEST:
            writer.add_scalar(f"Loss/{stage.name.split('.')[-1].lower()}", stage_loss, epoch)
            writer.add_scalar(f"my_fscore/{stage.name.split('.')[-1].lower()}", self.fscore, epoch)
            if(self.hparams.num_class >1):
                for i in range(self.fscores.shape[0]):
                    #if(i==0):
                    #    label = "fluent"
                    label = CLASSES[i]
                    writer.add_scalar(f'fscore/{label}', self.fscores[i], epoch)
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["macro"] = self.fscore * 100
            #stage_stats["micro"] = micro * 100
            
            if stage == sb.Stage.VALID:
                self.results = stage_stats
                self.stage_loss = stage_loss
                if(self.hparams.annealing):
                    old_lr, new_lr = self.update_learning_rate(epoch, self.hparams.lr_annealing)
                    if(old_lr !=new_lr):
                        print("Now updating learning rate")
                        print(f"Old lr: {old_lr}, new lr: {new_lr}")
                
                if self.hparams.ckpt_enable:
                    self.checkpointer.save_and_keep_only(
                        meta=stage_stats, min_keys=["loss"]
                    )
                if(stage_loss < self.best_loss):
                    self.best_loss = stage_loss
                if(self.fscore > self.best_fscore):
                    self.best_matrix = self.cf_matrix
                    self.best_multi_matrix = self.cf_multi_matrix
                    self.best_fscore = self.fscore
                    self.best_multi_fscore = self.fscores
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
                self.hparams.train_logger.log_stats(stats_meta={"\n": self.best_matrix, 
                                                                "\nbin fscore": np.round(self.best_fscore,4),
                                                                "\n": self.best_multi_matrix,
                                                                "\nmulti fscore": self.best_multi_fscore})
                                                        #
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.counter.current},
                    test_stats=stage_stats,
                )
                self.test_loss = stage_loss
                self.test_macro_fscore = stage_stats["macro"]
                self.test_fscore = self.fscore
                self.test_fscores = self.fscores 
                self.test_accuracy = self.accuracy
                self.test_missrate = self.missrate
        
    def update_learning_rate(self, epoch, scheduler):
        if(isinstance(scheduler, sb.nnet.schedulers.NewBobScheduler)):
            old_lr, new_lr = self.hparams.lr_annealing(self.stage_loss)
        elif (isinstance(scheduler, sb.nnet.schedulers.StepScheduler)):
            old_lr, new_lr = self.hparams.lr_annealing(self.epoch)
        elif (isinstance(scheduler, sb.nnet.schedulers.LinearScheduler)):
            old_lr, new_lr = self.hparams.lr_annealing(self.epoch)
        elif (isinstance(scheduler, sb.nnet.schedulers.LinearWarmupScheduler)):
            new_lr = self.hparams.lr_annealing.get_next_value()
        elif (isinstance(scheduler, sb.nnet.schedulers.CyclicCosineScheduler)):
            old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
        elif (isinstance(scheduler, sb.nnet.schedulers.ReduceLROnPlateau)):
            old_lr, new_lr = self.hparams.lr_annealing(self.optimizer, epoch, self.stage_loss)
        else:
            return
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
        return old_lr, new_lr

    def compute_metrics(self, epoch, stage, stage_loss):
        curr_stage = stage.name.split('.')[-1].lower()
        print(f"******{curr_stage}******")
        _, self.bin_macro, self.missrate, self.accuracy, self.fscore, self.missrate, self.cf_matrix= model.utils.my_confusion_matrix(self.y_true_binary, self.y_preds_binary, bin=0)
        self.multi_micro, self.multi_macro, _, _, self.fscores, _, self.cf_multi_matrix= model.utils.my_confusion_matrix(self.y_true_multi, self.y_preds_multi, bin=self.hparams.num_class)
        
        print(self.cf_matrix)
        print("---------------")
        print(self.cf_multi_matrix)
        self.hparams.train_logger.log_stats(stats_meta={"\naccuracy": np.round(self.accuracy,4), 
                                                        "\tmissrate": np.round(self.missrate,4), 
                                                        "\nbin fscore": np.round(self.fscore,4),
                                                        "\nmulti-fscores": np.round(self.fscores,4),
                                                        "\tmicro fscore": np.round(self.multi_micro,4)})

def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("Show","EpId", "ClipId")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def audio_pipeline(Show, EpId, ClipId):
        EpId = int(EpId)
        if(Show=="FluencyBank"):
            file = f"/LibriStutter_data/fluencybank/FluencyBank/{EpId:03}/FluencyBank_{EpId:03}_{ClipId}.wav"
        elif(hparams["dataset"]=="sep28k" or hparams["dataset"]=="s+f"):
            file = f"/LibriStutter_data/sep28k/sep28k_clips/{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav"
        else:
            print("WRONG PATH")
        waveform, _ = torchaudio.load(file, normalize=True)
        audio = waveform
        return (EpId, int(ClipId)), audio.squeeze()
    @sb.utils.data_pipeline.takes("Prolongation", "Block", "SoundRep", "WordRep", "Interjection")
    @sb.utils.data_pipeline.provides("label", "disfluency", "category")
    def get_label(p, b, sr, wr, inter):
        label, disfluency, category = get_labels(p,b,sr,wr,inter, hparams["num_class"])
        return label, disfluency, category
        
    @sb.utils.data_pipeline.takes("audioID","clipID")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def our_audio_pipeline(audioID, clipID):
        file = f"/LibriStutter_data/databrase/Clips/{audioID}_{clipID}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        audio = waveform
        return (int(audioID),int(clipID)), audio.squeeze()

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("label")
    def our_get_label(stutter_type):
        if "ok" in stutter_type:
            return torch.Tensor([1]), "disfluent"
        else:
            return torch.Tensor([0]), "fluent"
    datasets={}
    if(hparams["dataset"] == "ours"):
        csv_path = '/LibriStutter_data/databrase/annotation.csv'
        df = pd.read_csv(csv_path)
        msk = np.random.rand(len(df)) <= 0.8
        train = df[msk]
        test = df[~msk]
        train.to_csv('/LibriStutter_data/databrase/annotation_train.csv')
        test.to_csv('/LibriStutter_data/databrase/annotation_test.csv')
        test.to_csv('/LibriStutter_data/databrase/annotation_valid.csv')
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")
        if hparams["dataset"] == "ours":
            df = pd.read_csv(csv_path)
            datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                    csv_path=f'/LibriStutter_data/databrase/annotation_{dataset}.csv',
                    dynamic_items=[our_audio_pipeline, our_get_label],
                    output_keys=["id", "waveform", "label", "disfluency"],
                    )
        else:
            if dataset == "test" and hparams["TestOurs"]:
                csv_path = '/LibriStutter_data/databrase/annotation.csv'
                datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                    csv_path=csv_path,
                    dynamic_items=[our_audio_pipeline, our_get_label],
                    output_keys=["id", "waveform", "label", "disfluency"],
                    )
            else:
                if(hparams["dataset"]=="fluencybank"):
                    csv_path=f'/LibriStutter_data/fluencybank/fluencybank_{dataset}.csv'
                elif(hparams["dataset"]=="balanced"):
                    csv_path=f'/LibriStutter_data/fluencybank/fluencybank_balanced_labels_{dataset}.csv'
                elif(hparams["dataset"]=="sep28k"):
                    csv_path=f'/LibriStutter_data/sep28k/sep28k_labels_{dataset}.csv'
                elif(hparams["dataset"]=="s+f"):
                    csv_path=f'/LibriStutter_data/sep28k/sep28k_fluencybank_{dataset}.csv'
                datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                    csv_path=csv_path,
                    dynamic_items=[audio_pipeline, get_label],
                    output_keys=["id", "waveform", "label", "disfluency", "category"],
                )
                """
                sampler = BalancingDataSampler(dataset=datasets[f"{dataset}"],
                                                key="category")
                print(sampler.num_samples)
                it = iter(sampler)
                print([next(it) for _ in range(10)])"""
        if(dataset=="train" or hparams["balance_test"]):
            if(hparams["balance"]):
                counter_1 =0
                for i in range(len(datasets[dataset])):
                    if(datasets[dataset][i]["disfluency"]==1):
                        counter_1 +=1
                d = datasets[dataset].filtered_sorted(sort_key="disfluency", reverse=True, select_n=None)
                count = 0
                new_data = {}
                for key in d:
                    if count < counter_1:
                        id = key["id"]
                        key.pop("id")
                        new_key = key
                        new_data[id] = new_key
                    count+=1
                while count < (1/hparams["balance_ratio"])*counter_1 :
                    x = random.randint(0, len(datasets[dataset])-1)
                    if datasets[dataset][x]["id"] not in new_data.keys():
                        key = datasets[dataset][x]
                        id = key["id"]
                        key.pop("id")
                        new_key = key
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
        print(len(datasets[dataset]))
    return datasets
def get_labels(p,b,sr,wr,inter, num_class):
    label = torch.zeros(num_class)
    category = ""
    if(int(sr)>=hparams["annot_value"] and hparams["SoundRep"]):
        label[CLASSES.index("SoundRep")] = 1
        category += "SoundRep"
    if(int(wr)>=hparams["annot_value"] and hparams["WordRep"]):
        label[CLASSES.index("WordRep")] = 1
        category += "WordRep"
    #elif(int(wr)>=hparams["annot_value"] and hparams["WordRep"] and num_class==5):
    #    label[CLASSES.index("SoundRep")+1] = 1
    if(int(p)>=hparams["annot_value"] and hparams["Prolongation"]):
        label[CLASSES.index("Prolongation")] = 1
        category += "Prolongation"
    if(int(b)>=hparams["annot_value"] and hparams["Block"]):
        label[CLASSES.index("Block")] = 1
        category += "Block"
    if(int(inter)>=hparams["annot_value"] and hparams["Interjection"]):
        label[CLASSES.index("Interjection")] = 1
        category += "Interjection"
    if(torch.count_nonzero(label)==0):
        disfluency = 0
        category = "Fluent"
    else:
        disfluency = 1 #fluent
    return label, disfluency, category
    
    
import os
import data_prep_utils
from termcolor import colored, cprint
from speechbrain.utils import hpopt as hp
import csv
from speechbrain.utils import checkpoints as ckpt
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    with hp.hyperparameter_optimization(objective_key="best_macro") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx
    #hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
            try:
                hparams["output_folder"] = hparams["output_folder"]+hp.get_trial_id()
                writer = SummaryWriter("/tensorboard")
            except:
                writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")
        print("*********************************************************************")
        print(f"Training on",colored(hparams['dataset'],'red', attrs=["bold"]))
        cprint(f"Model architecture used is ", end='')
        cprint(colored(hparams['models'],'red', attrs=["bold"]))

        print(colored(overrides, 'green'))
        CLASSES = []
        for c in ["Interjection", "SoundRep", "WordRep", "Prolongation", "Block"]:
            if hparams[c]:
                CLASSES.append(c)
        print(CLASSES)
        print("*********************************************************************")
        
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        
        if(hparams["dataset"]=="Libri"):
            datasets = data_prep_utils.dataio_prep(hparams)
        else:    
            datasets = dataio_prep(hparams)
        # Initialize trainer
        opt_class = partial(hparams["opt_class"].func, lr=float(hparams["opt_class"].keywords["lr"]))
        checkpointer = ckpt.Checkpointer(hparams["output_folder"], recoverables={
                'w2v2_model': hparams["model"],
                'counter': hparams["counter"]

        })
        detect_brain = LatimBrain(
            modules= hparams["modules"],
            opt_class=opt_class,
            run_opts=run_opts,
            hparams=hparams,
            checkpointer=checkpointer,
        )
        #print(detect_brain.modules.chosen_model)
        #for name, param in detect_brain.modules.chosen_model.named_parameters():
        #    print(name)
        detect_brain.best_fscore = -1
        detect_brain.best_loss = 10000000
        detect_brain.best_epoch= -1
        # Fit dataset
        #valid_kwargs = {key: value for key, value in hparams["dataloader_opts"].items() if key != 'ckpt_prefix'}

       
        detect_brain.fit(
            epoch_counter=hparams["counter"],
            train_set=datasets[f"train"],
            valid_set=datasets[f"valid"],
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )
        
        print("*"*20, "Evaluation", "*"*20)
        if(hparams["TestOurs"]):
            datatest = "our data"
        else:
            datatest = hparams["dataset"]

        print(overrides)
        print("*"*5, f"Best epoch: {detect_brain.best_epoch} with best f1-score: {detect_brain.best_fscore}", "*"*5)
        print("*"*5, f"Testing on {datatest}", "*"*5)
        detect_brain.evaluate(
            datasets[f"test"],
            test_loader_kwargs=hparams["dataloader_opts"],
        )
        if(hparams["num_class"])==1:
            writer.add_hparams({
                                'model': hparams["models"],
                                'classes': ' '.join(CLASSES),
                                'num_class': hparams["num_class"],
                                'train_val': hparams["dataset"],
                                'test': datatest,
                                'lr': hparams["learning_rate"],
                                'layers': hparams["layers"],
                                'pool_time': hparams["pool_time"],
                                'positive': hparams["positive"],
                                'annot_threshold': hparams["annot_value"]
                                },
                                {
                                'score/F1': detect_brain.test_fscore,
                                'score/Accuracy/Fluent': detect_brain.test_accuracy,
                                'best_score': detect_brain.best_fscore
                                #'score/MissRate': detect_brain.test_missrate
                                })
        """
        else:
            writer.add_hparams({
                                'num_class': hparams["num_class"],
                                'dataset': hparams["dataset"],
                                'backbone': str(hparams["backbone"]),
                                'seed': hparams["seed"],
                                'lr': hparams["learning_rate"],
                                'dropout': hparams["dropout"]
                                },
                                {
                                'score/F1/Macro': detect_brain.test_macro_fscore,
                                'score/F1/Fluent': detect_brain.test_fscores[0],
                                #'score/F1/Prolongations': detect_brain.test_fscore[CLASSES.index("Prolongations")],
                                #'score/F1/Block': detect_brain.test_fscore[CLASSES.index("Block")],
                                #'score/F1/WordRep': detect_brain.test_fscore[CLASSES.index("WordRep")],
                                #'score/F1/SoundRep': detect_brain.test_fscore[CLASSES.index("SoundRep")],
                                #'score/F1/Interjection': detect_brain.test_fscore[CLASSES.index("Interjection")],
                                })
        """
        hp.report_result(detect_brain.results)
        print(detect_brain.results)
        with open(f'{hparams["output_folder"]}/orion.csv', 'w') as csv_file:  
            wr = csv.writer(csv_file)
            for key, value in detect_brain.results.items():
                wr.writerow([key, value])
        writer.flush()
        writer.close()
