from config import *
import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np 
from net.deeplab import Deeplab_v3
import cv2
from PIL import Image
import os 
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Deeplab_v3(class_number=1, fine_tune=True)
model.eval()
model.load_state_dict(torch.load('model/clothes-segmentation.pth', map_location=torch.device('cpu'))['model_state_dict'])
model.to(device=device)

def predict(image_path):
    img = cv2.imread(image_path)

    img = Image.fromarray(img)

    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img = transform(img)
    img = img.to(device)

    img = img.unsqueeze(0)

    output = model(img)

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy() > 0.4

    output = np.squeeze(output)

    return output

def generate_image(image, title, img_name):
    fig = plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.title(title[0])
    plt.imshow(image[0])

    plt.subplot(2, 2, 2)
    plt.title(title[1])
    plt.imshow(image[1])

    plt.subplot(2, 2, 3)
    plt.title(title[2])
    plt.imshow(image[2])

    plt.subplot(2, 2, 4)
    plt.title(title[3])
    plt.imshow(image[3])

    plt.axis('off')
    # plt.show()
    plt.savefig('results/' + img_name)

    # return fig



if __name__ == '__main__':
    paths = 'img/'
    paths = 'img_test/'
    # paths = 'images_chuan/'

    for file in os.listdir(paths):
        path = paths + file
    # path = 'data/train/images/000004.jpg'
        print(path)
        name = path.split('/')[-1]
        mask = predict(path)

        im = cv2.imread(path)
        w, h = im.shape[:2]
        images = []
        clone_img = im.copy()
        # images.append(clone_img)
        # im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (224, 224))
        images.append(im)

        # im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        img = Image.fromarray(im)

        transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        img = transform(img)

        

        image = img.data.cpu().numpy()

        cv2.imshow('img', im)
        
        title = ['image', 'ao', 'quan', 'vay']
        cv2.imshow('mask', mask.astype('uint8') * 255 )
        ma = mask.astype('uint8') * 255
        cloth = cv2.bitwise_and(im, im, mask = ma)

        
        images.append(cloth.astype('uint8'))
        cv2.imshow('cloth', cloth.astype('uint8'))
        cv2.waitKey(0)

        # generate_image(images, title, name)
       
    cv2.destroyAllWindows()