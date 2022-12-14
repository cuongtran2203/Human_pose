import cv2
from base_camera import BaseCamera
import time
import sys
class Camera(BaseCamera):
    def __init__(self):
        super().__init__()
        # self.play_sound=Sound()
    # over-wride of BaseCamera class frames method
    # @staticmethod
    def frames(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        count=0
        while True:
            # read current frame
            _, frame = camera.read()
            count+=1
            frame=cv2.resize(frame,(1280,720))
            yield frame

if __name__ == "__main__":
    camera = Camera()
    frame=camera.get_frame()
    cv2.imshow("ssss", frame)
    cv2.waitKey(1)


        
