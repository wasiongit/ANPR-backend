import cv2
from PIL import Image


def image_merge():
    # images = [Image.open(x) for x in ['anpr/1.png', 'anpr/2.png', 'anpr/3.png','anpr/4.png','anpr/5.png','anpr/6.png','anpr/7.png']]
    images = [Image.open(x) for x in ['anpr/1.png', 'anpr/2.png', 'anpr/3.png','anpr/4.png']]
    # images = [Image.open(x) for x in ['anpr/5.png','anpr/6.png','anpr/7.png']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('number.png')


    # images = [Image.open(x) for x in ['anpr/1.png', 'anpr/2.png', 'anpr/3.png','anpr/4.png']]
    images = [Image.open(x) for x in ['anpr/5.png','anpr/6.png','anpr/7.png']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('number2.png')




def pre_P(img):

    gray=img
    try:
        h,w=gray.shape
    except:
        h,w,_=gray.shape
    fy=fx=363/w
    # fx=171/w
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = fx, fy = fy, interpolation = cv2.INTER_CUBIC)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    n=1
    # for i, ctr in enumerate(sorted_ctrs):
    for i, ctr in enumerate(contours):
        # print(i)
        x, y, w, h = cv2.boundingRect(ctr)

        roi = img[y:y + h, x:x + w]

        area = w*h

        # if length is less than 2Xwidth , skip
        if h<2*w:
            continue
        if h>5*w:
            continue

        # if 800 < area < 3000:
        if 800 < area < 3000:
            # print(area)
            rect = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # rect = cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('rect', rect)
            # cv2.imshow('thresh', thresh)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            # roi = img[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            # roi = cv2.medianBlur(roi, 5)
            # text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11 --oem 3')
            # text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3')
            # plate_num += text
            cv2.imwrite(f'anpr/{n}.png',roi)
            n+=1
    image_merge()

# img = cv2.imread('final image.png')
# pre_P(img)
# cv2.waitKey(0)


