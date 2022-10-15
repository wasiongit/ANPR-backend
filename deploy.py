'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
# import easyocr


##### DEFINING GLOBAL VARIABLE
# EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
OCR_TH = 0.2




### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    # frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    print(f"[INFO] Total {n} detections. . . ")
    # exit()
    ### looping through the detections
    for i in range(n):
        row = cordinates[i]
        print(row[4])
        if row[4] >= 0.5: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1,y1,x2,y2]
            xmin, ymin, xmax, ymax = coords
            nplate = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            cv2.imwrite(f'output{i}.png',nplate)



### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate



### ---------------------------------------------- Main function -----------------------------------------------------

def main_detect(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='ALPR_best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format




    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    



### -------------------  calling the main function-------------------------------
# main(vid_path="./test_images/video.mp4",vid_out="vid_1.mp4") ### for custom video
# main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam

# main(img_path="./test_images/Cars74.jpg") ## for image
# main_detect(img_path="ANPR 2.PNG") ## for image