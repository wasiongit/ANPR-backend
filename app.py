import cv2
from deskew import deskew
from image_preP import pre_P
from deploy import main_detect
import easyocr
from flask import Flask, render_template, request,redirect, url_for


app = Flask(__name__)


@app.route('/')
def upload_file():
   return render_template('anprpage.html')


@app.route('/anprapi',methods = ['GET', 'POST'])
def attendance():
    # img=None
    if request.method == 'POST':
        f = request.files['imagefile']
        f.save(f'image.png')

        main_detect('image.png')
        # to read the output
        img = cv2.imread('output0.png')
        corrected_img = deskew(img)
        corrected_img=cv2.cvtColor(corrected_img,cv2.COLOR_RGB2GRAY)
        # cv2.imwrite('final image.png',corrected_img)
        # croped gray image
        pre_P(corrected_img)

        number1=reader.readtext('number.png',detail=False)
        number2=reader.readtext('number2.png',detail=False)
        return(number1+number2)

if __name__ == '__main__':
    reader = easyocr.Reader(['en'])
    # raw image
    # img = cv2.imread('output0.png')
    app.run(debug = True)
