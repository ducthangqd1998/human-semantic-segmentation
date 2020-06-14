from time import time
import numpy as np
from torch.nn import functional as F

from models.deeplabv3 import DeepLabV3Plus
from dataloaders import transforms
from models.utils import utils
import cv2
import torch 
import os
from glob import glob

model = DeepLabV3Plus()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

trained_dict = torch.load('model/DeepLabV3Plus_ResNet50.pth', map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.eval()

def smooth_image(image):
    ret,thresh1 = cv2.threshold(image,125,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh1, (9,9), 11)
    et,thresh2 = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)
    return thresh2
#     image = cv2.imread('img_test/' + i)
    # name = i.split('.')[0]
for i in glob('img_test/*'):
    image = cv2.imread(i)
    h, w = image.shape[:2]
    X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=320, pad_value=0)

    with torch.no_grad():
        mask = model(X)
        print(torch.min(mask))
        # mask = mask.data.cpu().numpy()
        mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
        # mask = mask.data.cpu().numpy()
        mask1 = mask
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
        mask = F.softmax(mask, dim=1)
        mask = mask[0,1,...].numpy()
        mask1 = mask1[0,1,...].numpy()

    # mask[np.where(mask.data.cpu().numpy() > 0.5)] = 255


    mask[np.where(mask > 0.5)] = 255
    mask = mask.astype('uint8')
    mask1[np.where(mask1>0.6)] = 255
    haha = cv2.resize(mask1, (w, h))
    smooth = smooth_image(haha)
    cv2.imshow('mask', cv2.resize(mask, (433, 577)))
    obj = cv2.bitwise_and(image, image, mask = mask)
    cv2.imshow('smooth', cv2.resize(obj, (433, 577)))
    cv2.waitKey(0)
cv2.destroyAllWindows()
    # # image = cv2.resize(image, (224, 224))
    # obj = cv2.bitwise_and(image, image, mask = mask)

    # im = np.zeros([h, w, 4])
    # im[:, :, :3] = obj
    # im[:, :, 3] = mask
    # im = im.astype('uint8')

    # path = 'image_save/' + name + '.png'
    # cv2.imwrite('mask/' + name + '.jpg', mask)
    # cv2.imwrite(path, im) 
# cv2.imshow('mask', mask.astype('uint8'))
# cv2.imshow('image', image)
# cv2.imshow('obj', obj)
# cv2.waitKey(0)
# cv2.destroyAllWindows()