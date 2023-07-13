#encoding=utf8
import os.path

import cv2
from pycocotools.coco import COCO


def resize(x, y, w, h, width, height):
    # return int(x), int(y), w, h
    return int(x/width*500),int(y/height*500),int(w/width*500),int(h/height*500)


def findidsinCoco(filename):
    for ids, val in cocoGt.imgs.items():
        if val['file_name'] == filename:
            print(ids)


if __name__ == "__main__":
    picpath = '/MOTDataset/COCOdataset/coco_resize'
    # 000000560178.jpg
    # 562059
    # 560178
    cocoGt = COCO('/MOTDataset/COCOdataset/instances_val2017.json')
    info = cocoGt.imgs[560266]
    filename = info['file_name']
    width = info['width']
    height = info['height']
    img = cv2.imread(os.path.join(picpath, filename))

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=560266))

    anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=560266)) # 为了显示这两句的作用，上面单独跑了
    # plt.imshow(img)
    # plt.axis('off') # 该语句可以关闭坐标轴
    cocoGt.showAnns(anns)


    for each in labels:
        [x1, y1, w, h] = each['bbox']
        x, y, w, h = resize(x1, y1, w, h, width, height)
        # cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        text = cocoGt.cats[each['category_id']]['name']
        cv2.putText(img, text, (int(x1), int(y1)), font, 1, (255, 0, 0), 2)
    cv2.imwrite('/t1.png', img)

    print(1)