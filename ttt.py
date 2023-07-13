from y5gradcam import yolov5gradcam
import cv2
if __name__ == '__main__':


    masks=yolov5gradcam.calSaliencyMaps(img=cv2.imread('./y5gradcam/images/img.png'),modelpath='./y5gradcam/yolov5s.pt')
    print(1)