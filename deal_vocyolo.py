nc=20  # number of classes
names=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from utils.general import download, Path
def convert_label(path, lb_path, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    in_file = open(path / f'voc_apa/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
access = '/root/autodl-tmp/first-project/YOLOv5-mask/datasets/VOC/images/VOCdevkit'
#  for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
#      imgs_path = dir / 'images' / f'{image_set}{year}'
#      lbs_path = dir / 'labels' / f'{image_set}{year}'
#      imgs_path.mkdir(exist_ok=True, parents=True)
#      lbs_path.mkdir(exist_ok=True, parents=True)
lbs_path = '/root/autodl-tmp/first-project/YOLOv5-mask/datasets/VOC/labels'
imgs_path = '/root/autodl-tmp/first-project/YOLOv5-mask/datasets/VOC/vocImg1'
with open('/root/autodl-tmp/first-project/YOLOv5-mask/datasets/VOC/images/VOCdevkit/test2.txt') as f:
  image_ids = f.read().strip().split()
for id in tqdm(image_ids, desc='root/autodl-tmp/first-project/YOLOv5-mask/datasets/VOC/images/voc2.1'):
      f = access / f'VOC2012/JPEGImages/{id}.jpg'  # old img path
      lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
      f.rename(imgs_path / f.name)  # move image
      convert_label(access, lb_path, id)  # convert labels to YOLO format