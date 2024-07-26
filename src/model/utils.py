
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def my_confusion_matrix(y_trues, y_preds, bin):
    # constant for classes  
    y_preds = torch.round(y_preds)
    if(bin<=1):
        cf_matrix= confusion_matrix(y_trues.cpu().detach().numpy(), 
                                    y_preds.cpu().detach().numpy())
        TP = cf_matrix[1,1]
        FP = cf_matrix[0,1]
        FN = cf_matrix[1,0]
        TN = cf_matrix[0,0]
        if(TN!=0):
            fscores = 2*TP/(2*TP+FP+FN)
        else:
            fscores = 0
        accuracy = (TP+TN)/(TP+TN+FN+FP)
        missrate = FN / (FN+TP)
        if(TP+FP > 0):
            precision = TP / (TP+FP)
        else:
            precision = 0
        recall = TP / (TP+FN)
            
    else:
        cf_matrix = confusion_matrix(y_trues.cpu().detach().numpy(), y_preds.cpu().detach().numpy())
        fscores = np.zeros(bin)
        accuracy = np.zeros(bin)
        missrate = np.zeros(bin)
        precision = np.zeros(bin)
        recall = np.zeros(bin)
        for i in range(bin):
            TP = cf_matrix[i,1,1]
            FP = cf_matrix[i,0,1]
            FN = cf_matrix[i,1,0]
            TN = cf_matrix[i,0,0]
            if(TN!=0):
                fscores[i] = 2*TP/(2*TP+FP+FN)
            else:
                fscores[i] = 0
            accuracy[i] = (TP+TN)/(TP+TN+FN+FP)
            recall[i] = TP / (TP+FP)
            precision[i] = TP / (TP+FP)
            missrate[i] = FN / (FN+TP)
    
    return accuracy, fscores, missrate, cf_matrix, precision, recall


def plot_embeddings(embeddings, labels, layers, epoch, writer):
    f_count = 0
    df_count = 0
    layers = list(map(int, layers.split(" ")))
    for j, (features, label) in enumerate(zip(embeddings,labels)):
        fig, ax = plt.subplots(int(features.shape[-1]/768), 1, figsize=(16, 4.3 * features.shape[-1]/768))
        if(label[0,0]==1):
            lab = "fluent"
            f_count +=1
            if(f_count >5):
                continue
        else:
            lab = "disfluent"
            df_count += 1
            if(df_count >5):
                continue
        
        features_reshape = features.view(-1, 149, 768)
        for i, feats in enumerate(features_reshape):
            ax[i].imshow(feats.cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {layers[i]}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        fig.tight_layout()
        writer.add_figure(f'w2v2-layers/{j}/{lab}_{epoch}', fig, global_step=epoch)


        #plt.savefig(f"/LibriStutter_data/features/{epoch}_{label.item()[-1]}_{n}.jpg")
def compute_TSNE(embeddings, labels, stage, writer):
    if (embeddings.dim()==3):
        x_train = embeddings.reshape([embeddings.shape[0],embeddings.shape[1]*embeddings.shape[2]])
    else:
        x_train = embeddings.reshape([embeddings.shape[0],embeddings.shape[1]])
    y_train = labels
    tsne = TSNE(n_components=2, perplexity=5)
    oncpu = x_train.detach().cpu().numpy()
    z = tsne.fit_transform(oncpu)
    df = pd.DataFrame()
    df["label"] = y_train
    df["x"] = z[:,0]
    df["y"] = z[:,1]
    group_codes = {k:idx for idx, k in enumerate(df.label.unique())}
    df['colors'] = df['label'].apply(lambda x: group_codes[x])
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["x"], df["y"], c=df['colors'])
    handles = scatter.legend_elements(num=[0,1,2,3])[0]
    legend = ax.legend(handles = handles, labels=group_codes.keys())
    ax.add_artist(legend)
    writer.add_figure(f"TSNE/{stage}", plt.gcf())