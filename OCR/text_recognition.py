from paddleocr import PaddleOCR
from PIL import Image
import cv2
class Text_Recognizer() :
    def __init__(self ) :
        self.ocr = PaddleOCR(lang='en') 
    # need to run only once to download and load model into memory
    def text_recognizer(self,img) :
        # img=Image.fromarray(img)
        # img_path = 'i1Abv.png'
        txt=""
        result = self.ocr.ocr(img,det=True,rec=True)
        # print(result)
        for line in result:
            # print(line)
            for l in line :
                # print(l)
                text=l[1][0]
                # print(text)
                txt+=text
        dict_text={"Text":txt}
        print(dict_text)
        return dict_text

        # # draw result

        # for idx in range(len(result)):
        #     res = result[idx]
        #     for line in res:
        #         print(line)
if  __name__ == "__main__" :
    model=Text_Recognizer()
    img=cv2.imread("i1Abv.png")
    model.text_recognizer(img)