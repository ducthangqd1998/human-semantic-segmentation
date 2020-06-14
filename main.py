import torch
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 
from glob import glob
from dataset import Dataset
from train import train
import os

from net.deeplab import Deeplab_v3

from config import *


def main():
    train_img_paths = glob('data/people/train/images/*')
    train_mask_paths = glob('data/people/train/masks/*')
    val_img_paths = glob('data/people/valid/images/*')
    val_mask_paths = glob('data/people/valid/masks/*')

    model = Deeplab_v3(class_number=num_classes)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    val_dataset = Dataset(val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    best_iou = 0

    if os.path.exists('model/clothes-segmentation.pth'):
        checkpoint = torch.load('model/clothes-segmentation.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])

    train(model, train_loader, val_loader, optimizer, exp_lr_scheduler, epochs)


if __name__ == '__main__':
    main()
    