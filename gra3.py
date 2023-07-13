import re
import json

#coco x1,y1,w,h

labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

file = open('/home/wuchenxi/Desktop/pythonProject/venv/val_result20.txt')
f = open('filename22.json', 'w')
f.write(str("["))
i = 0
file1 = open('/home/wuchenxi/Desktop/pythonProject/venv/val_result20.txt')

a = int(len(file1.readlines()))#读取txt文件的行数
print(a)
for line in file:
    #********name***********
    picture_index = line.find('.jpg')
    picture_name = line[picture_index-6:picture_index]

    #********w**************
    w_index = line.find(' ')
    line1 = line[w_index+1:]
    w = line[w_index + 1:w_index + 4]

    #********H**************

    H_index = line1.find(" ")
    line11 = line1[H_index + 1:]
    line11_index = line11.find(" ")
    H = line11[:line11_index]

    #********class**********
    class_index = line1.find(" ")
    line2 = line1[class_index+1:]
    clss_index1 = line2.find(" ")
    line3 = line2[clss_index1+1:]
    class_index_end = line3.find(" ")
    class_num = int(line3[:class_index_end])

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

    #******* x1,y1,x2,y2******
    x1_index = line3[class_index_end+1:].find("  ")
    line4 = line3[class_index_end+1:]
    x1 = line4[:x1_index]
    line5 = line4[x1_index+2:]
    y1_index = line5.find(" ")
    y1 = line5[:y1_index]
    line6 = line5[y1_index+1:]
    x2_index = line6.find(" ")
    x2 = line6[:x2_index]
    line7 = line6[x2_index+1:]
    y2 = line7
    ww = abs(float(x2)-float(x1))
    hh = abs(float(y2)-float(y1))
    #print(class_num.strip())

    final_json = dict()
    final_json['image_id'] = int(picture_name)
    print(line)
    print("error::::::::", class_num)
    final_json['category_id'] = int(class_num)#labels[int(class_num)]
    final_json['bbox'] = [float(x1), float(y1), float(ww), float(hh)]
    final_json['score'] = 0.5

    jsondata = json.dumps(final_json)
    #jsondata = json.dumps(final_json, indent=4, separators=(',', ': '))
    i = i+1
    if i < a:
        print("strat+++")
        f.write(jsondata + "," + "\n")
    else:
        print("over#####")
        f.write(jsondata)

f.write(str(']'))



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json

anno_json = '/home/wuchenxi/Desktop/pythonProject/venv/instances_val2014.json'

anno = COCO(anno_json)  # init annotations api
print(anno)
pred = anno.loadRes('/home/wuchenxi/Desktop/pythonProject/venv/filename.json')  # init predictions api
print('anno::', anno)
print('pred::', pred)
eval = COCOeval(anno, pred, 'bbox') # , 'bbox'
            # if is_coco:
            #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
