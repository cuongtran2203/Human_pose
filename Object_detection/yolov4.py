import cv2
import time
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
class Detection():
    def __init__(self):
        self.net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    def detect(self,frame):
        classes, scores, boxes = self.model.detect(frame, Conf_threshold, NMS_threshold)
        if len(boxes)>0:
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "{} : {:.3f}".format(class_name[int(classid)], score)
                cv2.rectangle(frame, box, color, 1)
                cv2.putText(frame, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            
if __name__ == '__main__':
    cam=cv2.VideoCapture(0)
    model=Detection()
    while cam.isOpened():
        _,frame=cam.read()
        model.detect(frame)
        cv2.imshow("ddd",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
       
        
        
    