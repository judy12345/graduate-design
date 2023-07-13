import cv2
import os

picpath='/MOTDataset/COCOdataset/tesoripic'
svpath='/MOTDataset/COCOdataset/coco_resize'


for each in os.listdir(picpath):
    img=cv2.imread(os.path.join(picpath,each))
    img=cv2.resize(img,(500,500))
    cv2.imwrite(os.path.join(svpath,each),img)
    print(each)
