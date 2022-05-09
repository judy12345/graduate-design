from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json
# /MOTDataset/COCOdataset/resjson/pic_By_myMethod_By_y5.json
# /MOTDataset/COCOdataset/resjson/pic_By_oriMethod_By_y5.json

if __name__ == "__main__":
    cocoGt = COCO('/root/autodl-tmp/first-project/COCOdataset/gt.json')
    cocoDt = cocoGt.loadRes('/root/autodl-tmp/first-project/COCOdataset/resjson/re3pic_By_myMethod_By_y5.json')
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


