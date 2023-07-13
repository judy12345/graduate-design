#encoding=utf8
# 读取json文件内容,返回字典格式
import json
import os

picpath='/MOTDataset/COCOdataset/coco_resize'
flist= os.listdir(picpath)
fp=open('/MOTDataset/COCOdataset/instances_val2017.json','r',encoding='utf8')

json_data = json.load(fp)

for i in range(len(json_data['images'])-1, -1, -1):
    if json_data['images'][i]['file_name'] not in flist:
        json_data['images'].pop(i)


for i in range(len(json_data['annotations'])-1, -1, -1):
    idx=json_data['annotations'][i]['image_id']
    idx=str(idx)
    idx=idx.rjust(12,'0')
    imgname='{}.jpg'.format(idx)

    if imgname not in flist:
        json_data['annotations'].pop(i)

savepath='/MOTDataset/COCOdataset/gt.json'
j = json.dumps(json_data)
f=open(savepath,'w')
f.write(j)
f.flush()
f.close()

print(1)