import cv2
from deskew import deskew
from image_preP import pre_P
from deploy import main_detect
import easyocr




def main():
    main_detect('ANPR 2.PNG')
    img = cv2.imread('output0.png')
    corrected_img = deskew(img)
    corrected_img=cv2.cvtColor(corrected_img,cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('final image.png',corrected_img)
    # croped gray image
    pre_P(corrected_img)
    number1=reader.readtext('number.png',detail=False)
    number2=reader.readtext('number2.png',detail=False)
    
    print(number1+number2)
    return(number1+number2)


if __name__ == '__main__':
    reader = easyocr.Reader(['en'])
    main()
