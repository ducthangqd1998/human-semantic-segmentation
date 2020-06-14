import numpy as np
import cv2
import random
from PIL import Image
import torch
import torch.utils.data
from torchvision import datasets, models, transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.image_transform = transforms.Compose([
            transforms.RandomSizedCrop(256)
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, 224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.mask_transform = transforms.Compose([
            transforms.F
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        return img, mask


from glob import glob

if __name__ == '__main__':
    val_img_paths = glob('dataset/validation/images/*')
    val_mask_paths = glob('dataset/validation/masks/*')
    data = Dataset(val_img_paths, val_mask_paths)
    for i in range(len(val_mask_paths)):
        img = data.mapping_mask_image(i, 3)

        for j in img:
            cv2.imshow('image', j.astype('uint8'))
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    