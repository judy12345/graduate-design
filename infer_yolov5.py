import json
import os

import cv2
from pycocotools.coco import COCO

from y5gradcam.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

annFile = '/MOTDataset/COCOdataset/instances_val2017.json'

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


def saverespic(bbox, classname, file):
    img = cv2.imread(os.path.join(picpath, file))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(bbox)):
        # [x1, y1, x2, y2] = [each[0],each[1],each[2],each[3]]
        # [x1, y1, x2, y2] = [bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]]
        [y1, x1, y2, x2] = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
        cls_name = classname[i]
        # x, y, w, h = resize(x1, y1, w, h, width, height)
        # cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
        # text = cocoGt.cats[each['category_id']]['name']
        cv2.putText(img, cls_name, (int(x1), int(y1)), font, 1, (255, 0, 0), 2)
        # cv2.putText(img, cls_name, (int(x-w/2),int(y-h/2)), font, 1, (255, 0, 0), 2)

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
    picpath = '/MOTDataset/COCOdataset/pic_By_oriMethod'
    svpicpth = '/MOTDataset/COCOdataset/pic_By_oriMethod_By_y5'
    jsonsvpath = '/MOTDataset/COCOdataset/resjson/pic_By_oriMethod_By_y5.json'

    device = 'cuda:0'
    input_size = (512, 512)
    # img = cv2.imread(picpath)
    modelpath = 'yolov5s.pt'
    model = YOLOV5TorchObjectDetector(modelpath, device, img_size=input_size, names=None,confidence=0.1)
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

        saverespic(out[0][0][0], out[0][2][0], file)
        out2coco(out, file)

    savejson(savepath=jsonsvpath)
