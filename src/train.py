"""
Recipe for training a disfluency detection system on the LibriStutter dataset.

Authors
 * Peter Plantinga 2021
"""
import wave
from torch.utils.tensorboard import SummaryWriter 

import sys
import torch
from libristutter_prepare_population import prepare_libristutter_population
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from libristutter_prepare_patient_specific import prepare_libristutter_patient
from data_prep_utils import *
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

class LatimBrain(sb.Brain):
    """Use attentional model to predict words in segments"""

    def compute_feats(self, wavs, lens, stage):
        if(self.hparams.features["preaugment"]):
            if stage == sb.Stage.TRAIN:
                wavs = self.hparams.preaugment(wavs, lens)
        feats = self.hparams.compute_feats(wavs)
        #if(self.hparams.features["normalize"]):
        #    feats = self.hparams.normalize(
        #        feats, lens, epoch=self.hparams.counter.current
        #    )

        # Augment
        if(self.hparams.features["augment"]):
            if stage == sb.Stage.TRAIN:
                feats = self.hparams.spec_augment(feats)
        feats = feats.transpose(1,2)
        if(self.hparams.features["normalize"]):
            feats = self.hparams.normalize(
                feats, lens, epoch=self.hparams.counter.current
            )
        
        return feats

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        waveforms, lens = batch.waveform
        spectrogram = self.compute_feats(waveforms, lens, stage)
        if(hparams["siamese"]):
            pair_waveforms, pair_lens = batch.pair_waveform
            pair_spectrogram = self.compute_feats(pair_waveforms, pair_lens, stage)
            e1, e2, y_pred = self.modules.network(spectrogram, pair_spectrogram)
            predictions = {
                "e1": e1,
                "e2": e2,
                "y_pred": y_pred
            }
        else:
            y_pred, _ = self.modules.model(spectrogram)
            predictions = {
                "y_pred": y_pred
            }
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label.data
        preds = torch.round(torch.sigmoid(predictions["y_pred"]))
        if(not self.hparams.siamese):
            if(self.hparams.num_class>1):
                if(self.hparams.num_class==5):
                    offset = 1
                    fluent_loss = self.hparams.fluent_loss(predictions["y_pred"][:,0], labels[:,0])
                    self.fluent_metrics.append(batch.id, preds[:,0], labels[:,0])
                else:
                    offset = 0
                    fluent_loss = 0 
                sound_rep_loss = self.modules.sound_rep_loss(predictions["y_pred"][:,0+offset], labels[:,0+offset]).to("cuda:0")
                word_rep_loss = self.modules.word_rep_loss(predictions["y_pred"][:,1+offset], labels[:,1+offset]).to("cuda:0")
                phrase_rep_loss = self.modules.phrase_rep_loss(predictions["y_pred"][:,2+offset], labels[:,2+offset]).to("cuda:0")
                prolongation_loss = self.modules.prolongation_loss(predictions["y_pred"][:,3+offset], labels[:,3+offset]).to("cuda:0")
                loss = sound_rep_loss + word_rep_loss + phrase_rep_loss + prolongation_loss + fluent_loss
                self.sound_rep_metrics.append(batch.id, preds[:,0+offset], labels[:,0+offset])
                self.word_rep_metrics.append(batch.id, preds[:,1+offset], labels[:,1+offset])
                self.phrase_rep_metrics.append(batch.id, preds[:,2+offset], labels[:,2+offset])
                self.prolongation_metrics.append(batch.id, preds[:,3+offset], labels[:,3+offset])
            else:
                #self.stutter_metrics.append(batch.id, preds, labels)
                self.y_true = torch.cat((self.y_true,labels))
                self.y_preds = torch.cat((self.y_preds,preds))
                loss = self.modules.stutter_loss(predictions["y_pred"], labels).to("cuda:0")
        else:
            pair_labels = batch.pair_label
            lab = torch.eq(labels.data[:,0],pair_labels.data[:,0]).float()
            loss, dist = self.hparams.contrastive(predictions["e1"], predictions["e2"], lab)
            self.dist_metrics.append(batch.id, torch.where(dist>self.hparams.margin/2, 0, 1).reshape(-1,1), 
                                                lab.reshape(-1,1))

        return loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        if(hparams["siamese"]):
                self.dist_metrics =sb.utils.metric_stats.BinaryMetricStats()
        else:
            if(self.hparams.num_class>1):
                if(self.hparams.num_class==5):
                    self.fluent_metrics = sb.utils.metric_stats.BinaryMetricStats()
                    self.sound_rep_metrics =sb.utils.metric_stats.BinaryMetricStats()
                    self.word_rep_metrics = sb.utils.metric_stats.BinaryMetricStats()
                    self.phrase_rep_metrics = sb.utils.metric_stats.BinaryMetricStats()
                    self.prolongation_metrics =sb.utils.metric_stats.BinaryMetricStats()

                
            else:
                #self.stutter_metrics =sb.utils.metric_stats.BinaryMetricStats()
                self.y_preds = torch.tensor(()).to("cuda:0")
                self.y_true = torch.tensor(()).to("cuda:0")
    
    def confusion_matrix(self, epoch, curr_stage):
        # constant for classes
        print(f"******{curr_stage}******")
        micro_fscore_cnt={"TP": 0, "FP":0, "FN":0}
        macro_fscores=[]
        classes = ["sound_rep", "word_rep", "phrase_rep", "prolongation"]
        if(not hparams["siamese"]):
            if self.hparams.num_class==4:
                all_metrics = [self.sound_rep_metrics, self.word_rep_metrics, self.phrase_rep_metrics, self.prolongation_metrics]
                cf_matrices = []
                
                for m in range(len(all_metrics)):
                    cf_matrix = np.array(((all_metrics[m].summarize(field="TN"), all_metrics[m].summarize(field="FP")),
                                    (all_metrics[m].summarize(field="FN"), all_metrics[m].summarize(field="TP"))))
                    cf_matrices.append(cf_matrix)
                cf_matrices = np.array(cf_matrices)
            elif self.hparams.num_class==5:
                classes = ["fluent", "sound_rep", "word_rep", "phrase_rep", "prolongation"]
                all_metrics = [self.fluent_metrics, self.sound_rep_metrics, self.word_rep_metrics, self.phrase_rep_metrics, self.prolongation_metrics]
                cf_matrices = []
                for m in range(len(all_metrics)):
                    cf_matrix = np.array(((all_metrics[m].summarize(field="TN"), all_metrics[m].summarize(field="FP")),
                                    (all_metrics[m].summarize(field="FN"), all_metrics[m].summarize(field="TP"))))
                    cf_matrices.append(cf_matrix)
                cf_matrices = np.array(cf_matrices)
            else:
                classes = ["stutter"]
                cf_matrices = confusion_matrix(self.y_true.cpu().detach().numpy(), 
                                                self.y_preds.cpu().detach().numpy())
                cf_matrices = cf_matrices.reshape((1, cf_matrices.shape[0], cf_matrices.shape[1]))
            print(cf_matrices)
            for f in range(cf_matrices.shape[0]):
                curr_class = classes[f] 
                TP = cf_matrices[f,1,1]
                FP = cf_matrices[f,0,1]
                FN = cf_matrices[f,1,0]
                TN = cf_matrices[f,0,0]
                if(TP !=0 ):
                    precision = TP / (TP+FP)
                    recall = TP / (TP+FN)
                else:
                    precision = 0
                    recall = 0
                missrate = 1 - recall
                fscore = 2*TP/(2*TP+FP+FN)
                micro_fscore_cnt["TP"] += TP
                micro_fscore_cnt["FP"] += FP
                micro_fscore_cnt["FN"] += FN
                macro_fscores.append(fscore)
                #writer.add_scalar(f"Precision/{curr_stage}", precision, epoch)
                #writer.add_scalar(f"Miss_Rate/{curr_stage}", missrate, epoch)
                print(f"{curr_class}: fscore= {fscore*100:.2f} \t precision= {precision*100:.2f}\t missrate= {missrate*100:.2f}")
            micro_fscore =                          2*micro_fscore_cnt["TP"]/(
                            2*micro_fscore_cnt["TP"]+micro_fscore_cnt["FP"]+micro_fscore_cnt["FN"])
            #if(curr_stage!="train"):
            #    writer.add_scalar(f"{curr_stage}/Micro_F-Score", micro_fscore, epoch)
            return micro_fscore, macro_fscores, missrate, precision, TN, FP, FN, TP
        else:
            cf_matrix = np.array(((self.dist_metrics.summarize(field="TN"), self.dist_metrics.summarize(field="FP")),
                                 (self.dist_metrics.summarize(field="FN"), self.dist_metrics.summarize(field="TP"))))
            TP = cf_matrix[1,1]
            FP = cf_matrix[0,1]
            FN = cf_matrix[1,0]
            TN = cf_matrix[0,0]
            if(TP !=0 ):
                precision = TP / (TP+FP)
                recall = TP / (TP+FN)
            else:
                precision = 0
                recall = 0
            missrate = 1 - recall
            fscore = 2*TP/(2*TP+FP+FN)
            print(cf_matrix)
            print(f"siamese: fscore= {fscore*100:.2f} \t precision= {precision*100:.2f}\t missrate= {missrate*100:.2f}")
            return fscore, fscore, missrate, precision, TN, FP, FN, TP

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def compute_metrics(self, epoch, stage, stage_loss):
        curr_stage = stage.name.split('.')[-1].lower()
        micro, macro, missrate, precision, TN, FP, FN, TP = self.confusion_matrix(epoch, curr_stage)            
        summary = np.array(macro).mean()
        return summary, micro, missrate, precision, TN, FP, FN, TP

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        summary, micro, missrate, precision, TN, FP, FN, TP = self.compute_metrics(epoch, stage, stage_loss)
        self.epoch = epoch
        self.stage_loss = stage_loss
        writer.add_scalar(f"Loss/{stage.name.split('.')[-1].lower()}", stage_loss, epoch)
        writer.add_scalar(f"F-Score/{stage.name.split('.')[-1].lower()}", summary * 100, epoch)
        
        if stage == sb.Stage.TRAIN:
            writer.add_scalar(f"LR/{stage.name.split('.')[-1].lower()}",self.get_lr() , epoch)
            self.train_loss = stage_loss
            self.positive_train = FN + TP
            self.negative_train = TN + FP
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["macro"] = summary * 100
            if(self.hparams.num_class >1):
                stage_stats["micro"] = micro * 100
            if stage == sb.Stage.VALID:
                if(self.hparams.annealing):
                    self.update_learning_rate(self.hparams.lr_annealing)

                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stage_stats,
                )
                #self.checkpointer.save_and_keep_only(
                #    meta=stage_stats, min_keys=["loss"]
                #)

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.counter.current},
                    test_stats=stage_stats,
                )
                self.positive_test = FN + TP
                self.negative_test = TN + FP
                self.test_loss = stage_loss
                self.test_fscore = stage_stats["macro"]
                self.test_precision = precision
                self.test_missrate = missrate

    def update_learning_rate(self, scheduler):
        if(isinstance(scheduler, sb.nnet.schedulers.NewBobScheduler)):
            _, new_lr = self.hparams.lr_annealing(self.stage_loss)
        elif (isinstance(scheduler, sb.nnet.schedulers.StepScheduler)):
            _, new_lr = self.hparams.lr_annealing(self.epoch)
        elif (isinstance(scheduler, sb.nnet.schedulers.LinearScheduler)):
            _, new_lr = self.hparams.lr_annealing(self.epoch)
        elif (isinstance(scheduler, sb.nnet.schedulers.LinearWarmupScheduler)):
            new_lr = self.hparams.lr_annealing.get_next_value()
        elif (isinstance(scheduler, sb.nnet.schedulers.CyclicCosineScheduler)):
            _, new_lr = self.hparams.lr_annealing(self.optimizer)
        else:
            return
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

import os
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(hparams["data_folder"]):  
        os.makedirs(hparams["data_folder"])
    writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    datasets = dataio_prep(hparams)
    # Initialize trainer
    detect_brain = LatimBrain(
        modules= hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    # Fit dataset
    valid_kwargs = {key: value for key, value in hparams["dataloader_opts"].items() if key != 'ckpt_prefix'}

    epochCounter = sb.utils.epoch_loop.EpochCounter(hparams["number_of_epochs"])
    detect_brain.fit(
        epoch_counter=epochCounter,
        train_set=datasets[f"train"],
        valid_set=datasets[f"valid"],
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=valid_kwargs,
    )

    
    print("*"*20, "Evaluation", "*"*20)
    # Evaluate best checkpoint, using lowest or highest value on validation
    detect_brain.evaluate(
        datasets[f"test"],
        test_loader_kwargs=hparams["dataloader_opts"],
    )
    writer.add_hparams({
                        'num_class': hparams["num_class"],
                        'population': hparams["population"],
                        'siamese': hparams["siamese"],
                        'backbone': str(hparams["backbone"]),
                        'id': hparams["speaker_id"],
                        'samples/nb_train': int(detect_brain.negative_train + detect_brain.positive_train),
                        'seed': hparams["seed"],
                        'feature': hparams["feats"]
                        },
                        {
                        'score/F1': detect_brain.test_fscore,
                        'score/Precision': detect_brain.test_precision,
                        'score/MissRate': detect_brain.test_missrate
                        })
    writer.flush()
    writer.close()
