
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.manifold import TSNE

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def my_confusion_matrix(y_trues, y_preds, bin):
    # constant for classes  
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
            
    else:
        cf_matrix = multilabel_confusion_matrix(y_trues.cpu().detach().numpy(), 
                                                y_preds.cpu().detach().numpy())
        #cf_matrix2= confusion_matrix(y_trues.argmax(axis=1).cpu().detach().numpy(), 
        #                            y_preds.argmax(axis=1).cpu().detach().numpy())
        #print(cf_matrix2)
        fscores = np.zeros(bin)
        accuracy = np.zeros(bin)
        missrate = np.zeros(bin)

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
            missrate[i] = FN / (FN+TP)
    macro_fscore = f1_score(y_trues.cpu().detach().numpy(), 
                            y_preds.cpu().detach().numpy(), average='macro', zero_division=0)
    micro_fscore = f1_score(y_trues.cpu().detach().numpy(), 
                            y_preds.cpu().detach().numpy(), average='micro', zero_division=0)
    _, recall, _, _ = score(y_trues.cpu().detach().numpy(), 
                            y_preds.cpu().detach().numpy(),zero_division=0)
    
    return micro_fscore, macro_fscore, recall, accuracy, fscores, missrate, cf_matrix


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

def compute_TSNE(embeddings, labels, stage, epoch, num_class, writer):
    if (embeddings.dim()==3):
        x_train = embeddings.reshape([embeddings.shape[0],embeddings.shape[1]*embeddings.shape[2]])
    else:
        x_train = embeddings.reshape([embeddings.shape[0],embeddings.shape[1]])
    #x_mnist = reshape(x_train, [x_train.shape[0], x_train.shape[1]* x_train.shape[2]])

    y_labels = labels.detach().cpu().numpy()
    if(num_class == 1):
        y_train = ["fluent" if x == 1 else "disfluent" for x in y_labels[:,:,0] ]
    else:
        y_train = [str(x) for x in y_labels]
    tsne = TSNE(n_components=2, perplexity=10, verbose=0, n_iter=50000, random_state=2)
    z = tsne.fit_transform(x_train.detach().cpu().numpy())

    df = pd.DataFrame()
    df["label"] = y_train
    df["x"] = z[:,0]
    df["y"] = z[:,1]
    group_codes = {k:idx for idx, k in enumerate(df.label.unique())}
    df['colors'] = df['label'].apply(lambda x: group_codes[x])
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["x"], df["y"], c=df['colors'])
    handles = scatter.legend_elements(num=[0,1,2,3])[0]
    legend = ax.legend(#*scatter.legend_elements(),
                    handles = handles,
                    loc="lower left", title="Classes", labels=group_codes.keys())
    ax.add_artist(legend)
    writer.add_figure(f"TSNE/{stage}", plt.gcf(), epoch)
    plt.clf()
