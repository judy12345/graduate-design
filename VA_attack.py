import argparse
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from mmcv.parallel import collate, scatter
from torchvision.transforms import Resize

from models_t import *
from utils.utils_coco import *
from y5gradcam.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from PIL import Image
import numpy as np
sys.path.append('./mmdetection/')
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import init_detector,LoadImage
from y5gradcam.yolov5gradcam import calSaliencyMaps
picpath='/root/autodl-tmp/first-project/COCOdataset/coco_resize3/'
import matplotlib.pyplot as plt


# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=1, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--max_iter', type=int, default=600, help='max number of iterations to find adversarial example')
parser.add_argument('--conf_thresh', type=float, default=0.5, help='conf thresh')
parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS thresh')
parser.add_argument('--im_size', type=int, default=500, help='the height / width of the input image to network')
parser.add_argument('--max_p', type=int, default=4700, help='max number of pixels that can change')
parser.add_argument('--minN', type=int, default=0, help='min idx of images')
parser.add_argument('--maxN', type=int, default=999, help='max idx of images')
parser.add_argument('--save', default='/root/autodl-tmp/first-project/COCOdataset/123', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.save, exist_ok=True)

pre = transforms.Compose([transforms.ToTensor()])
nor = transforms.Normalize([123.675 / 255., 116.28 / 255., 103.53 / 255.], [58.395 / 255., 57.12 / 255., 57.375 / 255.])

# model1 = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
# pretrained_dict = torch.load('checkpoints/yolov4.pth', map_location=torch.device('cuda'))
# model1.load_state_dict(pretrained_dict)
# model1.eval().cuda()

device_modelyolov5 = 'cuda:0'
input_size = (640, 640)

modelpath = 'yolov5s.pt'
print('[INFO] Loading the model')
model_yolov5 = YOLOV5TorchObjectDetector(modelpath, device_modelyolov5, img_size=input_size,
                                  names=None)



config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
meta = [{'filename': '../images/6.png', 'ori_filename': '../images/6.png', 'ori_shape': (500, 500, 3),
         'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3),
         'scale_factor': np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), 'flip': False, 'flip_direction': None,
         'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                          'std': np.array([58.395, 57.12, 57.375], dtype=np.float32), 'to_rgb': True}}]
model2 = init_detector(config, checkpoint, device='cuda:0')

cfg = model2.cfg
device = next(model2.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
eps = 8 / 255.
eps_iter = 2 / 255.
n_iter = 10
def tog_vanishing(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):#maximum perturbation ǫ. step size α.
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_vanishing_gradient(x_adv)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv

def letterbox_image_padded(image, size=(416, 416)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta
def visualize_detections(detections_dict):
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.figure(figsize=(3 * len(detections_dict), 3))
    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes = detections_dict[title]
        if len(input_img.shape) == 4:
            input_img = input_img[0]
        plt.subplot(1, len(detections_dict), pid + 1)
        plt.title(title)
        plt.imshow(input_img)
        current_axis = plt.gca()
        for box in detections:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0]), 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1]), input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0]), input_img.shape[0])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='small', color='black', bbox={'facecolor': color, 'alpha': 1.0})
        plt.axis('off')
    plt.show()
files = os.listdir(picpath)
files.sort()
# files = files[100:120]

count = 0
count2 = 0
shape1 = 0
shape2 = 0
num1 = 0
num2 = 0
num3 = 0
for file in files:
    # if file in tfiles:
    #     continue
    try:
        print("start:{}".format(file))
        flag = 0
        if count < args.minN:
            count += 1
            continue
        if count > args.maxN:
            break
        print(file)
        # prepare data
        input_img = Image.open(picpath + file)
        x_query, x_meta = letterbox_image_padded(input_img, size=(800,800))
        #detections_query = model2.detect(x_query, conf_threshold=model2.confidence_thresh_default)
        #visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes)})
        x_adv_vanishing = tog_vanishing(victim=model2, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        #detections_adv_vanishing = model2.detect(x_adv_vanishing, conf_threshold=model2.confidence_thresh_default)
        imgp_save = x_adv_vanishing.clone().detach()
        vutils.save_image(imgp_save, args.save + '/' + file)
    except Exception as e:
       print(e)
       continue
