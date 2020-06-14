import cv2
from glob import glob


# paths = glob('people_2/train/image/*')
# paths = glob('people_2/train/mask/*')
paths = glob('people_2/valid/image/*')
paths = glob('people_2/valid/mask/*')

z = 0
for link in paths:
    img = cv2.imread(link, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224, 224))
    z += 1
    print(z / len(paths) * 100)
    cv2.imwrite('data/valid/mask/' + link.split('/')[-1], img)