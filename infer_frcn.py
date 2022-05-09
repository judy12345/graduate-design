# from PIL import Image
import argparse

import torchvision.utils as vutils
from mmcv.parallel import collate, scatter

from mmdetection.mmdet.apis import show_result_pyplot
from models_t import *
from utils.utils_coco import *

sys.path.append('./mmdetection/')
from mmdet.datasets.pipelines import Co/root/autodl-tmp/first-project/COCOdataset/voc2.1mpose
from mmdet.apis.inference import init_detector, LoadImage, inference_detector
from deep_utils import Box
from pycocotools.coco import COCO
annFile ='/root/autodl-tmp/first-project/voc2012_trainval.json'


coco = COCO(annFile)

res=[]



picpath = ''
svpicpth='/root/autodl-tmp/first-project/COCOdataset/voc_resize_By_frcn'
jsonsvpath='/root/autodl-tmp/first-project/COCOdataset/resjson/vocpic_By_myMethod_By_frcn2.json'
parser = argparse.ArgumentParser()
parser.add_argument('--save', default='/root/autodl-tmp/first-project/COCOdataset/frremask3+',
                    help='folder to output images and model checkpoints')
args = parser.parse_args()

os.makedirs(args.save, exist_ok=True)

config = './mmdetection/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth'

model2 = init_detector(config, checkpoint, device='cuda:0')

cfg = model2.cfg

device = next(model2.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)

files = os.listdir(picpath)
files.sort()
import json


def findidsinCoco(filename):
    for ids, val in coco.imgs.items():
        if val['file_name'] == filename:
            return ids,val['width'],val['height']

def deResize(x,y,w,h,width,height):
    xc=x/500*width
    yc=y/500*height
    ori_w=w/500*width
    ori_h=h/500*height
    return xc,yc,ori_w,ori_h


def out2coco(out,file):
    ids,width,height=findidsinCoco(file)

    for i in range(len(out)):
        if out[i].shape[0]==0:
            continue
        for each in out[i]:
            [x1,y1,x2,y2,score]=each
            # if score<0.5:continue
            x1,y1,w,h=deResize(x1,y1,x2-x1,y2-y1,width,height)

            t={}
            t['image_id']=ids
            t['category_id']=transcocoids(i)
            t['bbox']=[int(x1),int(y1),int(w),int(h)]
            t['score']=float(score)
            res.append(t)


def saverespic(out,file,classesName,threshold=0.7):
    img=cv2.imread(os.path.join(picpath,file))
    for i in range(len(out)):
        if out[i].shape[0]==0:
            continue
        for each in out[i]:
            # if each[4]<threshold:
            #     continue


            font = cv2.FONT_HERSHEY_SIMPLEX
            [x1, y1, x2, y2] = [each[0],each[1],each[2],each[3]]
            cls_name = classesName[i]
            # x, y, w, h = resize(x1, y1, w, h, width, height)
            # cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0),2)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
            # text = cocoGt.cats[each['category_id']]['name']
            cv2.putText(img, cls_name, (int(x1), int(y1)), font, 1, (255, 0, 0), 2)

            # bbox=[each[1],each[0],each[3],each[2]]
            #
            # img = Box.put_box(img, bbox,in_format='XYWH')
            # img = Box.put_text(img, cls_name, (each[1],each[0]))
    cv2.imwrite(os.path.join(svpicpth,file),img)
    print("sv")

def transcocoids(class_num):

    if class_num >= 0 and class_num <=10:
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
    f=open(savepath,'w')
    f.write(j)
    f.flush()
    f.close()

for file in files:
    print(file)
    # prepare data
    # data = dict(img=os.path.join(picpath,file))
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # data = scatter(data, [device])[0]

    t=inference_detector(model2,os.path.join(picpath,file))
    model2.show_result(os.path.join(picpath, file),t,out_file=os.path.join(svpicpth,file),score_thr=0.4)
    #saverespic(t,file,model2.CLASSES)
    out2coco(t,file)


savejson(savepath=jsonsvpath)