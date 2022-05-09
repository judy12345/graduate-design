import json
import os
#from random import random
import argparse # python的命令行解析的模块，内置于python，不需要安装
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import cv2
from pycocotools.coco import COCO

from y5gradcam.models.yolo_v5_object_detector_voc import YOLOV5TorchObjectDetector

annFile = '/root/autodl-tmp/first-project/voc2012_trainval.json'

coco = COCO(annFile)


def findidsinCoco(filename):
    for ids, val in coco.imgs.items():
        if val['file_name'] == filename:
            return ids, val['width'], val['height']


def deResize(x, y, w, h, width, height):
    xc = x / 512 * width
    yc = y / 512 * height
    ori_w = w / 512 * width
    ori_h = h / 512 * height
    return xc, yc, ori_w, ori_h


def out2coco(out, file):
    ids, width, height = findidsinCoco(file)

    for i in range(len(out[0][0][0])):
        [y1, x1, y2, x2] = out[0][0][0][i]
        score = out[0][3][0][i]
        # if score<0.5:continue
        x1, y1, w, h = deResize(x1, y1, x2 - x1, y2 - y1, width, height)

        catid = out[0][1][0][i]

        t = {}
        t['image_id'] = ids
        t['category_id'] = transcocoids(catid)
        t['bbox'] = [int(x1), int(y1), int(w), int(h)]
        t['score'] = float(score)
        res.append(t)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def saverespic(bbox,confidence, classname, file):
    img = cv2.imread(os.path.join(picpath, file))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(bbox)):
        # [x1, y1, x2, y2] = [each[0],each[1],each[2],each[3]]
        # [x1, y1, x2, y2] = [bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]]
        [y1, x1, y2, x2] = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
        cls_name = classname[i]
        conf=float(confidence[i])
        # x, y, w, h = resize(x1, y1, w, h, width, height)
        # cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),2)
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        #text = cocoGt.cats[each['category_id']]['name']
        #label = float('%s %.2f' % (cls_name, confidence))
        #text=f"{confidence*100,2}%"
        #f'{cls_name,confidence:.2f}'
        cv2.putText(img,'{}, {:.3f}'.format(cls_name,conf),(int(x1), int(y1)), font, 1, (0, 0, 0), 2)
        #label = '%s %.2f' % (names[int(cls)], conf)
        #label = f'{cls_name} {confidence:.2f}'
        #plot_one_box((int(x1), int(y1)),img,color=(0, 0, 0),label=zip(cls_name,confidence),line_thickness=3)
        #cv2.putText(img, cls_name, (int(x-w/2),int(y-h/2)), font, 1, (255, 0, 0), 2)

        # bbox=[each[1],each[0],each[3],each[2]]
        #
        # img = Box.put_box(img, bbox,in_format='XYWH')
        # img = Box.put_text(img, cls_name, (each[1],each[0]))
    cv2.imwrite(os.path.join(svpicpth, file), img)
    # cv2.imwrite('/t.png',img)
    print("sv")


def transcocoids(class_num):
    if class_num >= 0 and class_num <= 10:
        class_num = class_num + 1
    elif class_num >= 11 and class_num <= 23:
        class_num = class_num + 2
    elif class_num >= 24 and class_num <= 25:
        class_num = class_num + 3
    elif class_num >= 26 and class_num <= 39:
        class_num = class_num + 5
    elif class_num >= 40 and class_num <= 59:
        class_num = class_num + 6
    elif class_num == 60:
        class_num = class_num + 7
    elif class_num == 61:
        class_num = class_num + 9
    elif class_num >= 62 and class_num <= 72:
        class_num = class_num + 10
    elif class_num >= 74 and class_num <= 80:
        class_num = class_num + 11
    return class_num


def savejson(savepath):
    j = json.dumps(res)
    f = open(savepath, 'w')
    f.write(j)
    f.flush()
    f.close()


if __name__ == '__main__':
    picpath = '/root/autodl-tmp/first-project/COCOdataset/voc2.1'
    svpicpth = '/root/autodl-tmp/first-project/COCOdataset/voc_resize_By_y5'
    jsonsvpath = '/root/autodl-tmp/first-project/COCOdataset/resjson/voc2_resize_By_y5.json'

    device = 'cuda:0'
    input_size = (512, 512)
    # img = cv2.imread(picpath)
    modelpath = 'best.pt'
    model = YOLOV5TorchObjectDetector(modelpath, device, img_size=input_size, names=None,confidence=0.2)
    # torch_img = model.preprocessing(img[..., ::-1])
    # res=model(torch_img)
    # print('[INFO] Loading the model')
    res=[]
    files = os.listdir(picpath)
    for file in files:
        print(file)
        img = cv2.imread(os.path.join(picpath, file))
        torch_img = model.preprocessing(img[..., ::-1])
        out = model(torch_img)
        #saverespic(out[0][0][0], out[0][2][0], file)
        saverespic(out[0][0][0],out[0][3][0],out[0][2][0], file)
        out2coco(out, file)

    savejson(savepath=jsonsvpath)
