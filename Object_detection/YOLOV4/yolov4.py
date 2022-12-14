import cv2
import time
import firebase_admin
from firebase_admin import db
import numpy as np
import threading

Conf_threshold = 0.5
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
class_name = []
with open('/home/cuong/Documents/GitHub/Human_pose/Object_detection/YOLOV4/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)



class Detection():
    def __init__(self):
        self.net = cv2.dnn.readNet('../Object_detection/YOLOV4/yolov4-tiny.weights', '../Object_detection/YOLOV4/yolov4-tiny.cfg')
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        self.dict_obj=dict()
    def detect(self,frame):
        t1=time.time()
        classes, scores, boxes = self.model.detect(frame, Conf_threshold, NMS_threshold)
        if len(boxes)>0:
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "{} : {:.3f}".format(class_name[int(classid)], score)
                cv2.rectangle(frame, box, color, 1)
                box=np.array(box,dtype=int)
                dict_id={class_name[int(classid)]:{"box":box.tolist(),"score":float(score)}}
                print(dict_id)
                self.dict_obj.update(dict_id)
                
                cv2.putText(frame, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            return self.dict_obj
            # Thread_update(dict_id)
            # print("FPS : {}".format(1/(time.time()-t1)))
# if __name__ == '__main__':
#     cam=cv2.VideoCapture("../video.mp4")
#     model=Detection()
#     while cam.isOpened():
#         _,frame=cam.read()
#         frame=cv2.resize(frame,(1280,720))
#         model.detect(frame)
#         cv2.imshow("ddd",frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
       
        
        
    