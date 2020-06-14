from flask import Flask, jsonify, request
from flask import abort
from time import time
import numpy as np
from torch.nn import functional as F

from models.deeplabv3 import DeepLabV3Plus
from dataloaders import transforms
from models.utils import utils
import cv2
import torch 


# Kiểm tra xem máy tính có gpu hay cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Loading model
model = DeepLabV3Plus()
trained_dict = torch.load('model/DeepLabV3Plus_ResNet50.pth', map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
# Thiết lập mô hình ở chế độ eval, chế độ dùng để test hoặc validation, không tranning
# Chế độ train sẽ tính toán các trọng số cập nhập lại cho graph của mô hình, đối vs test hay validation thì ko cần
model.eval()

def people_segment(image):
    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    h, w = image.shape[:2]
    X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=320, pad_value=0)
    with torch.no_grad():
        mask = model(X)
        mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
        mask = F.softmax(mask, dim=1)
        mask = mask[0,1,...].numpy()

    mask[np.where(mask > 0.4)] = 255
    mask = mask.astype('uint8')

    mask = cv2.resize(mask, (w, h))
    # image = cv2.resize(image, (224, 224))
    obj = cv2.bitwise_and(image, image, mask = mask)
    # cv2.imshow('mask', mask.astype('uint8'))
    # cv2.imshow('image', image)
    # cv2.imshow('obj', obj)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # obj = cv2.resize(obj, (), w))
    im = np.zeros([h, w, 4])
    im[:, :, :3] = obj
    im[:, :, 3] = mask
    im = im.astype('uint8')

    path = 'image_save/images.png'
    cv2.imwrite(path, im) 
    # mask
    return path

@app.route('/people-segmentation', methods=['POST'])
def predict_clothes():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        path = people_segment(img_bytes)
        # path = 'image_save/images.png'
        # cv2.imwrite(path, img) 
        return jsonify(path)
    else:
        return abort(404)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)



