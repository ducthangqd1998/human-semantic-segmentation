import torch
from tqdm import tqdm
from torchvision import transforms, datasets
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 
from collections import OrderedDict
from utils.loss import calc_loss
from collections import defaultdict
import copy
from utils.iou import iou_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))



def train(model, train_loader, valid_loader, optimizer, scheduler, epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10


    for epoch in range(epochs):
        
        for phase in ["train", "valid"]:
            if phase == "train":
                data_loader = train_loader
                model.train()
            else:
                data_loader = valid_loader
                model.eval()

            total = len(data_loader)

            print("Epoch {} - phase {}".format(epoch, phase))
            total = len(data_loader)

            with tqdm(total= len(data_loader)) as img_pbar:
                metrics = defaultdict(float)
                iou_mean = 0
                epoch_samples = 0

                for i, (input, target) in (enumerate(data_loader)):
                    input = input.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(input)
                        iou = iou_score(target, outputs)
                        iou_mean += iou
                        desc = f'IoU {iou}'
                        img_pbar.set_description(desc)
                        img_pbar.update(1)

                        loss = calc_loss(outputs, target, metrics)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                # statistics
                epoch_samples += list(input.shape)[0]

            if phase == 'train':
                scheduler.step()

            print_metrics(metrics, epoch_samples, phase)

            epoch_loss = metrics['loss'] / total
            iou_mean = iou_mean / total
            print("Epoch Loss: {} - Epoch IoU: {}".format(epoch_loss, iou_mean))

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scheduler': scheduler.state_dict(),
                    }, 'model/clothes-segmentation.pth' )

        print()