import torchattacks
import argparse
import os

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from mmcv.parallel import collate, scatter
from torchvision.transforms import Resize

from models_t import *
from utils.utils_coco import *
from y5gradcam.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

sys.path.append('./mmdetection/')
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import init_detector,LoadImage
from y5gradcam.yolov5gradcam import calSaliencyMaps
picpath='/root/autodl-tmp/first-project/COCOdataset/ppt/'
import matplotlib.pyplot as plt
modelpath = 'yolov5s.pt'
print('[INFO] Loading the model')
model = YOLOV5TorchObjectDetector(modelpath, device_modelyolov5, img_size=input_size,
                                  names=None)

attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
adv_images = attack(images, labels)