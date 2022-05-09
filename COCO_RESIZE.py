import cv2
import os

picpath='/root/autodl-tmp/first-project/ppt/'
svpath='/root/autodl-tmp/first-project/'

#

for each in os.listdir(picpath):
    img=cv2.imread(os.path.join(picpath,each))
    img=cv2.resize(img,(500,500))
    cv2.imwrite(os.path.join(svpath,each),img)
    print(each)