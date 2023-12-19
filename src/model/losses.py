import torch
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin
        self.pd = torch.nn.PairwiseDistance()

    def forward(self, output1, output2, label):
        euc_distance = self.pd(output1, output2)
        #label = 0 if outputs are different classes, label =1 if outputs are from same class
        loss_contrastive = ((label) * torch.pow(euc_distance, 2 ) +
                            (1-label) * torch.pow(torch.clamp(self.margin - euc_distance, 0.0), 2))
        mean = torch.mean(loss_contrastive)
        
        return mean, euc_distance #np.where(euc_distance.cpu()>self.margin, 1, 0)


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, f1_score, accuracy_score
class JouaitiLoss(nn.Module):
    def __init__(self):
        super(JouaitiLoss, self).__init__()
        self.requires_grad = True

    def forward(self, preds, labels):
        l = labels.cpu().detach().numpy()
        p = torch.round(torch.sigmoid(preds))
        p = p.cpu().detach().numpy()
        cf_matrix = confusion_matrix(l,p)
        tp = cf_matrix[1,1]
        fp = cf_matrix[0,1]
        fn = cf_matrix[1,0]
        tn = cf_matrix[0,0]
        if(tn+fp!=0):
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        if(tp+fn!=0):
            recall = tp / (tp+fn)
        else:
            recall = 0
        loss = 1 - (0.85*specificity + 0.15*recall)
        return torch.tensor([loss],requires_grad=True)