import cv2 
import numpy as np
from connect.connect_cloud import *
from Human_action_recognition.infer import * 
from Object_detection.YOLOV4.yolov4 import *
from OCR.text_recognition import *
from Camera.camera_multi import *
class Pipeline () :
    def __init__ (self):
        self.human_action=Classification()
        self.detect=Detection()
        self.text_recog=Text_Recognizer()
        self.cam=Camera()
    def run(self):
        Thread_read_data(ref_request)
        frame=self.cam.get_frame()
        A=que.get()
        if frame is not None :
            dict_data=self.human_action.run(frame)
            Thread_update(ref_human,dict_data)
            if A=="A" :
                obj_dict=self.detect.detect(frame)
                Thread_update(ref_obj,obj_dict)
            if A=="B" :
                res=self.text_recog.text_recognizer(frame)
                Thread_update(ref_text,res)
if __name__ == "__main__": 
    pipeline = Pipeline()


