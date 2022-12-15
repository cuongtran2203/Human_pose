import argparse
import os
import cv2
import numpy as np
import onnxruntime
# import sys
# sys.path.insert("/ultils")
from .ultils.preprocess import preproc as preprocess
from .ultils.ultils import demo_postprocess,multiclass_nms,vis
from .ultils.coco_classes import *
import time
class Detection_ONNX():
    def __init__(self,model_path):
        self.input_shape=(416,416)
        providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
        self.session = onnxruntime.InferenceSession(model_path,providers=providers)
    def detect(self,org_img):
        img, ratio = preprocess(org_img, self.input_shape)
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_shape, p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            org_img,text = vis(org_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.6, class_names=COCO_CLASSES)
        else :
            return None
        return text


if __name__ == '__main__':
    
    img=cv2.imread("")
    t1=time.time()
    model=Detection_ONNX("./yolox_nano.onnx")
    img_s= model.detect(img)
    print("Time process :",(time.time()-t1))
    cv2.imshow("ss",img_s)
    cv2.waitKey(0)

