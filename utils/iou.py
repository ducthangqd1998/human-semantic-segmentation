import numpy as np
import torch 

def iou_score(target, predict, num_classes):
    preds = torch.sigmoid(predict) > 0.5 
    N = preds.size()[0]
    inter = target * preds
    inter = inter.view(N, num_classes,-1).sum(2)

    union= target + preds - (target * preds)
    	## Sum over all pixels N x C x H x W => N x C
    union = union.view(N, num_classes, -1).sum(2)

    loss = inter / union

    ## Return average loss over classes and batch
    return loss.mean()


def iou_score(target, predict):
    predict = torch.sigmoid(predict).view(-1).data.cpu().numpy() > 0.5 
    target = target.view(-1).data.cpu().numpy() > 0.5
    intersection = (predict * target).sum()

    union = (target | predict).sum()

    iou = intersection / union

    return np.mean(iou)
