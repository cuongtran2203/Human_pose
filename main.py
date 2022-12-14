import cv2 
import numpy as np
from connect.connect_cloud import *
from Human_action_recognition.infer import * 
from Object_detection.YOLOV4.yolov4 import *
from OCR.text_recognition import *
from Camera.camera_multi import *
import sys
class Pipeline () :
    def __init__ (self):
        self.human_action=Classification()
        self.detect=Detection()
        self.text_recog=Text_Recognizer()
        self.cam=cv2.VideoCapture(0)
    def run(self):
        # Thread_read_data(ref_request)
        A='A'
        # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #     while self.cam.isOpened():
        #         _,frame=self.cam.read()
        image=self.human_action.run(self.cam)
        # Thread_update(ref_human,dict_data)
        if A=="A" :
            self.detect.detect(self.cam)
            # Thread_update(ref_obj,obj_dict)
        if A=="B" :
            res=self.text_recog.text_recognizer(image)
            # Thread_update(ref_text,res)
        # cv2.imshow("asss",image)
        # if cv2.waitKey(1) & 0xff == ord("q"):
        #     break
if __name__ == "__main__": 
    pipeline = Pipeline()
    pipeline.run()


