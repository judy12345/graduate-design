# from PIL import Image
import argparse
import os

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from mmcv.parallel import collate, scatter
from torchvision.transforms import Resize

from models_t import *
from utils.utils_coco import *
from y5gradcam.models.yolo_v5_object_detector_mask import YOLOV5TorchObjectDetector

sys.path.append('./mmdetection/')
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import init_detector,LoadImage
from y5gradcam.yolov5gradcam import calSaliencyMaps
picpath='/root/autodl-tmp/first-project/COCOdataset/ppt/'
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
parser.add_argument('--save', default='/root/autodl-tmp/first-project/', help='folder to output images and model checkpoints')
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

modelpath = '/root/autodl-tmp/first-project/YOLOv5-mask/yolov5-master/weights/best.pt'
print('[INFO] Loading the model')
model_yolov5 = YOLOV5TorchObjectDetector(modelpath, device_modelyolov5, img_size=input_size,
                                  names=None)



# config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
meta = [{'filename': '../images/6.png', 'ori_filename': '../images/6.png', 'ori_shape': (500, 500, 3),
          'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3),
          'scale_factor': np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), 'flip': False, 'flip_direction': None,
          'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                           'std': np.array([58.395, 57.12, 57.375], dtype=np.float32), 'to_rgb': True}}]
# model2 = init_detector(config, checkpoint, device='cuda:0')
#
# cfg = model2.cfg
# device = next(model2.parameters()).device  # model device
# # build the data pipeline
# test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
# test_pipeline = Compose(test_pipeline)


# print(test_pipeline)

# def get_mask(image, meta, pixels):
#     mask1 = torch.zeros((1, 3, 500, 500)).cuda()
#     # bbox, label = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
#     bbox = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
#     bbox = np.concatenate(bbox[0])
#
#     bbox = bbox[bbox[:, 4] > 0.3]
#     num = bbox.shape[0]
#     if num > 10: num = 10
#     if num == 0: return mask1.float().cuda()
#     lp = int((pixels)/ (2 * num))
#     for i in range(num):
#         xc = int((bbox[i, 0] + bbox[i, 2]) / 2)
#         yc = int((bbox[i, 1] + bbox[i, 3]) / 2)
#         w = int(bbox[i, 2] - bbox[i, 0])
#         h = int(bbox[i, 3] - bbox[i, 1])
#         lw = int(w / (w + h) * lp)
#         lh = int(h / (w + h) * lp)
#         x1 = max(0, xc - lh // 2)
#         x2 = min(xc + lh // 2, 500)
#         y1 = max(0, yc - lw // 2)
#         y2 = min(yc + lw // 2, 500)
#         mask1[:, :, yc - 1:yc + 2, x1:x2] = 1
#         mask1[:, :, y1:y2, xc - 1:xc + 2] = 1
#     maska = mask1.float().cuda()
#
#     return maska
#
#
# def get_mask2(image, meta, pixels):
#     mask = torch.zeros((1, 3, 500, 500)).cuda()
#     # bbox, label = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
#     bbox = model2(return_loss=False, rescale=True, img=image, img_metas=meta)
#     bbox = np.concatenate(bbox[0])
#     bbox = bbox[bbox[:, 4] > 0.3]
#     num = bbox.shape[0]
#     if num > 10: num = 10
#     if num == 0: return mask.float().cuda()
#     lp = int(pixels / (3 * num))
#     for i in range(num):
#         xc = int((bbox[i, 0] + bbox[i, 2]) / 2)
#         yc = int((bbox[i, 1] + bbox[i, 3]) / 2)
#         w = int(bbox[i, 2] - bbox[i, 0])
#         h = int(bbox[i, 3] - bbox[i, 1])
#         lw = int(w / (w + h) * lp)
#         lh = int(h / (w + h) * lp)
#         x1 = max(0, xc - lw // 2)
#         x2 = min(xc + lw // 2, 500)
#         y1 = max(0, yc - lh // 2)
#         y2 = min(yc + lh // 2, 500)
#         mask[:, :, yc - 1:yc + 2, x1:x2] = 1
#         mask[:, :, y1:y2, xc - 1:xc + 2] = 1
#     mask = mask.float().cuda()
#
#     return mask


def getmask_my(image, meta, pixels):
    mask = torch.zeros((1, 3, 500, 500)).cuda()
    img=cv2.imread(meta[0][0]['filename'])
    # torch_resize = Resize([500, 500])
    #
    # img_resize = torch_resize(image[0]).squeeze(0).permute((1, 2, 0))
    # img_resize = img_resize.detach().cpu().numpy()
    masks = calSaliencyMaps(img=img, modelpath='yolov5s.pt')

    num = len(masks)

    if num == 0: return mask.float().cuda()

    length = int(500 * 500 * 0.03 / num /3 / 2 / 2)

    for each in masks:
        xc, yc = each[0], each[1]

        x1 = max(0, xc - length)
        x2 = min(xc + length, 500)
        y1 = max(0, yc - length)
        y2 = min(yc + length, 500)

        mask[:, :, yc - 1:yc + 2, x1:x2] = 1
        mask[:, :, y1:y2, xc - 1:xc + 2] = 1

    mask = mask.float().cuda()

    return mask


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
pixels = 3000
#pixels = 1800
# tfiles=os.listdir(args.save)

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
        data = dict(img=picpath + file)
        #data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        #data = scatter(data, [device])[0]
        # data['img'] = data['img'][0]
        # data['img_metas'] = data['img_metas'][0]
        # mask = get_mask(data['img'], data['img_metas'], pixels)
        # mask2 = get_mask2(data['img'], data['img_metas'], pixels)
        # mask_my = getmask_my(data['img'], data['img_metas'], pixels)
        mask = getmask_my(data['img'], data['img_metas'], pixels)
        mask = mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        img_pil = cv2.imread(picpath + file)
        img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
        img_pil = np.transpose(img_pil, (2, 0, 1))
        img = torch.from_numpy(img_pil / 255.).float()
        img = img.unsqueeze(0).cuda()

        patch = img.clone().detach()
        #patch = torch.randint(0, 256, img.shape) / 255.
        patch = patch.float().cuda()
        patch.requires_grad = True
        optimizer = optim.SGD([patch], lr=32 / 255.)
        # optimizer = optim.SGD([patch], lr=256/255.)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 60, 100, 190], gamma=0.5, last_epoch=-1)
        for i in range(args.max_iter):
            #scheduler.step()
            imgp = torch.mul(img, 1 - mask) + torch.mul(patch, mask)
            # imgp = torch.mul(img, 1 -  _my) + torch.mul(patch, mask_my)
            Img1 = F.interpolate(imgp, size=(608, 608), mode='bilinear', align_corners=True)
            # out1 = model1(Img1)
            out_yolov5 = model_yolov5(Img1)
            if len(out_yolov5[0][0][0])==0:
                continue

            # loss1 = torch.max(out1[1])
            loss1 = torch.max(out_yolov5[1][0])
            #print(out_yolov5[1][0])
            #loss = -sum(obj - confidence * obj - mask)

            #




            #
            # t = np.concatenate(out2[0])
            # t = torch.from_numpy(t)
            # # loss2 = torch.max(out2[0][:, 4])
            #
            # loss2 = torch.max(t[:, 4])

            loss = loss1
            print('Num:{:4d}, Iter:{:4d}, Loss:{:.4f}, LossYOLO:{:.4f}}'.format(count, i, loss.item(),loss1.item()))
            if loss1.item() < 0.45 :
                flag = 1
                num1 += 1
                shape1 += 1
                break

            optimizer.zero_grad()
            loss.backward()
            patch.grad = torch.sign(patch.grad)
            optimizer.step()
            scheduler.step()
            patch.data.clamp_(0, 1)

        if flag == 0:
            count2 += 1
            patch = img.clone().detach()
            #patch = torch.randint(0, 256, img.shape) / 255.
            patch.requires_grad = True
            optimizer = optim.SGD([patch], lr=32 / 255.)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 50, 80, 115], gamma=0.5,
                                                       last_epoch=-1)

            for i in range(150):

                imgp = torch.mul(img, 1 - mask) + torch.mul(patch, mask)
                Img1 = F.interpolate(imgp, size=(608, 608), mode='bilinear', align_corners=True)
                # out1 = model1(Img1)

                out_yolov5 = model_yolov5(Img1)
                if len(out_yolov5[0][0][0]) == 0:
                    continue

                # loss1 = torch.max(out1[1])
                loss1 = torch.max(out_yolov5[1][0])



                # Img2 = F.interpolate(imgp, size=(800, 800), mode='bilinear', align_corners=True)
                # ImgNor = nor(Img2.squeeze(0))
                # data['img'] = [ImgNor.unsqueeze(0)]
                # out2 = model2(return_loss=False, rescale=True, img=data['img'], img_metas=data['img_metas'])
                #
                # # loss1 = torch.max(out1[1])
                #
                # t = np.concatenate(out2[0])
                # t = torch.from_numpy(t)
                #
                # # loss4 = torch.mean(out2[0][:, 4])
                # loss4 = torch.mean(t[:, 4])
                #
                # loss3 = torch.sum(t[:, 4] > 0.3)
                #
                # loss2 = torch.max(t[:, 4])

                loss = loss1

                print('Num:{:4d}, Iter:{:4d}, Loss:{:.4f}, LossYOLO:{:.4f}'.format(count, i,loss.item(),loss1.item()))

                if loss1.item() < 0.45 :
                    flag = 1
                    break

                optimizer.zero_grad()
                loss.backward()
                patch.grad = torch.sign(patch.grad)
                optimizer.step()
                scheduler.step()
                patch.data.clamp_(0, 1)

        if loss1.item() < 0.45: num2 += 1


        count += 1
        print('-' * 25, num1)

        imgp_save = imgp.clone().detach()
        vutils.save_image(imgp_save, args.save + '/' + file)
    except Exception as e:
        print(e)
        continue

print(num1, num2, num3)
print(count2)
print(shape1, shape2)
torch.cuda.empty_cache()
